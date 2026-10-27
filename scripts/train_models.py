from sklearn.pipeline import Pipeline
from mlflow import MlflowClient
import logging
import mlflow
import mlflow.sklearn
import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from scripts.anomaly_model import CloudTrailDetector
from scripts.evaluate import evaluate_model, time_split
from scripts.features_engineer import CloudTrailFeatures
from scripts.tuning import tune
from scripts import weak_labels

load_dotenv()

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "isolation_forest_model"
CHAMPION_ALIAS = "champion"
# Shipped with the model so the custom feature transformer can be unpickled anywhere
SCRIPTS_DIR = Path(__file__).resolve().parent
# MLflow serializes with skops, which only loads types explicitly trusted; both are produced by this pipeline
SKOPS_TRUSTED_TYPES = ["scripts.features_engineer.CloudTrailFeatures", "scripts.anomaly_model.CloudTrailDetector",
                       "sklearn.tree._tree.Tree"]

PARAMS = {
    "n_estimators": 150,
    "max_samples": 256,
    "max_features": 1.0,
    # Share of training events each detector flags; together they alert on at most ~1%
    "contamination": float(os.getenv('CONTAMINATION', 0.005)),
    "behaviour_contamination": float(os.getenv('BEHAVIOUR_CONTAMINATION', 0.005)),
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": 0
}
# Need enough history for a meaningful time-based holdout
MIN_EVENTS_FOR_EVALUATION = 1000
# Weak labels for every training run are also kept here for the supervised (phase 3) model
WEAK_LABELS_DIR = SCRIPTS_DIR.parent / "weak_labels"
VERDICTS_FILE = os.getenv('VERDICTS_FILE', str(SCRIPTS_DIR.parent / "verdicts.csv"))


def code_version():
    """Git commit of the code that built the model (with '+dirty' if uncommitted changes), or 'unknown'."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=SCRIPTS_DIR, capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        commit = git('rev-parse', '--short', 'HEAD')
        return f"{commit}+dirty" if commit and git('status', '--porcelain') else (commit or 'unknown')
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def build_pipeline(params=None):
    return Pipeline([
        ("features", CloudTrailFeatures()),
        ("detector", CloudTrailDetector(**(params or PARAMS))),
    ])


def isolation_forest_model(data, promote=True, holdout_fraction=0.2, min_scenario_detection=0.8,
                           tune_trials=0, alert_budget=0.02, label=True):
    """
    Trains a feature-engineering + anomaly detector (Isolation Forest and per-identity behavioural
    baseline, see anomaly_model.py) pipeline on raw CloudTrail records,
    logs parameters and metrics to MLflow, and registers the pipeline in the MLflow Model Registry.

    The feature step is part of the logged model, so inference reuses exactly the
    category lists and frequency tables learned here.

    Before registering, a copy is trained on the earlier (1 - holdout_fraction) of the data and
    evaluated on the rest with injected attack scenarios (see evaluate.py). The final model is
    then trained on all the data.

    Parameters
    data:
        A DataFrame of raw CloudTrail records (see features_engineer.RAW_COLUMNS).
    promote:
        If True and the evaluation passes, point the "champion" alias at the new version so
        inference picks it up.
    min_scenario_detection:
        Share of testable attack scenarios that must be detected for the model to be promoted.
    tune_trials:
        If > 0, search this many hyperparameter candidates first (see tuning.py) and train with
        the best; 0 uses PARAMS as they are.
    alert_budget:
        Tuning: the largest share of normal events allowed to raise an alert.
    label:
        If True, judge the weak-label sources on the holdout and save weak labels for all the data
        (see weak_labels.py), using analyst verdicts from VERDICTS_FILE as gold where they exist.

    Returns
    model:
        The trained pipeline.
    Scores:
        Anomaly scores for each sample in the dataset.
    preds:
        Anomaly predictions for each sample in the dataset (-1 for anomalies, 1 for normal)
    report:
        The evaluation report, or None if there was too little data to evaluate.
    """

    experiment = os.getenv('MLFLOW_EXPERIMENT_NAME', 'default_experiment')
    # Scheduled runs keep model artifacts in S3 so any machine can load the champion
    if os.getenv('MLFLOW_ARTIFACT_LOCATION') and mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=os.getenv('MLFLOW_ARTIFACT_LOCATION'))
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="IsolationForest_threat_Hunt"):

        params = dict(PARAMS)
        report = None
        evaluable = len(data) >= MIN_EVENTS_FOR_EVALUATION
        if evaluable:
            train, holdout = time_split(data, holdout_fraction)

        # --- OPTIONAL HYPERPARAMETER SEARCH ---
        if tune_trials and evaluable:
            params, trials = tune(train, holdout, params, build_pipeline, n_trials=tune_trials,
                                  alert_budget=alert_budget)
            mlflow.log_dict({'alert_budget': alert_budget, 'chosen': params, 'trials': trials}, "tuning.json")
        elif tune_trials:
            logger.warning("Too little data to tune; using default parameters")

        # Feature code and model must match: a model trained before a feature change must be retrained
        mlflow.set_tag("code_version", code_version())
        mlflow.log_params({**params, "holdout_fraction": holdout_fraction, "tuned": bool(tune_trials and evaluable),
                           "alert_budget": alert_budget, "min_scenario_detection": min_scenario_detection,
                           "training_events": len(data)})

        # --- EVALUATE ON LATER ACTIVITY WITH INJECTED ATTACKS ---
        verdicts = weak_labels.read_verdicts(VERDICTS_FILE) if label else None
        if evaluable:
            eval_model = build_pipeline(params).fit(train)
            report = evaluate_model(eval_model, holdout, history=train)
            if label:
                report['weak_labels'] = weak_labels.evaluate_weak_labels(eval_model, holdout, train, verdicts)
                wl = report['weak_labels']
                mlflow.log_metrics({'wl_label_model_f1': wl['label_model']['f1'], 'wl_best_lf_f1': wl['best_single_lf_f1'],
                                    'wl_majority_vote_f1': wl['majority_vote']['f1']})
                logger.info("Weak labels: label model F1 %.2f, majority vote %.2f, best single source %.2f -> using %s",
                            wl['label_model']['f1'], wl['majority_vote']['f1'], wl['best_single_lf_f1'],
                            wl['chosen_combiner'])
            mlflow.log_dict(report, "evaluation.json")
            mlflow.log_metrics({f"eval_{k}": v for k, v in report.items() if isinstance(v, (int, float))})
            mlflow.log_metrics({f"eval_detected_{name}": int(s['detected']) for name, s in report['scenarios'].items()})
            testable = [s for s in report['scenarios'].values() if s['testable']]
            logger.info("Evaluation: %d/%d testable attack scenarios detected (victim %s%s), %.2f%% of normal "
                        "events alerted", sum(s['detected'] for s in testable), len(testable), report['victim_identity'],
                        "" if report['victim_is_realistic'] else ", not a realistic baseline",
                        100 * report['normal_alert_rate'])
        else:
            logger.warning("Only %d events; skipping evaluation (need %d)", len(data), MIN_EVENTS_FOR_EVALUATION)

        # --- FINAL MODEL ON ALL DATA ---
        model = build_pipeline(params)
        model.fit(data)
        scores = model.decision_function(data)
        preds = model.predict(data)

        # log metrics
        n_anomalies = int((preds == -1).sum())

        mlflow.log_metrics({
            "anomalies_detected": n_anomalies,
            "anomaly_rate_percent": 100 * n_anomalies / len(preds),
            "total_samples": len(preds),
            "n_features": len(model.named_steps["features"].feature_names_out_),
            "mean_anomaly_score": scores.mean(),
            "std_anomaly_score": scores.std()})

        # --- WEAK LABELS FOR ALL DATA (input to the supervised model) ---
        if label:
            combiner = report['weak_labels']['chosen_combiner'] if report and 'weak_labels' in report else 'label_model'
            labels, _, summary = weak_labels.weak_label(data, model, weak_labels.gold_from(data, verdicts), combiner)
            run_dir = weak_labels.save(labels, summary, WEAK_LABELS_DIR / mlflow.active_run().info.run_id)
            mlflow.log_artifacts(str(run_dir), artifact_path="weak_labels")
            mlflow.log_metrics({'wl_labelled_attack_share': summary['labelled_attack_share'],
                                'wl_gold_labels': int(labels['gold'].notna().sum())})
            logger.info("Saved %d weak labels (%s, %.2f%% attack) to %s", len(labels), combiner,
                        100 * summary['labelled_attack_share'], run_dir)

        # Log model
        model_info = mlflow.sklearn.log_model(sk_model=model,
                                              name="threat_hunt_model",
                                              code_paths=[str(SCRIPTS_DIR)],
                                              skops_trusted_types=SKOPS_TRUSTED_TYPES,
                                              registered_model_name=REGISTERED_MODEL_NAME)
        version = model_info.registered_model_version

        passed = report is not None and report['scenario_detection_rate'] >= min_scenario_detection
        mlflow.set_tag("evaluation_passed", passed)
        if promote and passed:
            MlflowClient().set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)
            logger.info("Model version %s is now models:/%s@%s", version, REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
        elif promote:
            logger.warning("Model version %s registered but NOT promoted: evaluation %s. Promote manually with "
                           "`python main.py promote --version %s` if you accept it.", version,
                           "was skipped" if report is None else
                           f"detected {report['scenario_detection_rate']:.0%} of scenarios "
                           f"(need {min_scenario_detection:.0%})", version)

        return model, scores, preds, report


def promote_version(version):
    MlflowClient().set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, str(version))
    logger.info("Model version %s is now models:/%s@%s", version, REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
