"""
Hyperparameter search for the anomaly detector.

CloudTrail has no labels, so "accuracy" cannot be measured directly. Instead each candidate is
trained on the earlier part of the data and scored with evaluate.py (later activity + injected
attack scenarios). Candidates are ranked by, in order:

  1. within the alert budget   share of normal events alerted <= budget (what the SOC can handle),
                               measured on identities seen in training: brand-new identities alert
                               whatever the parameters, so they would only blur the comparison
  2. scenario detection rate   testable attack scenarios caught (model or rules)
  3. model event recall        attack events the model flags on its own; the part these
                               parameters actually control (rules do not depend on them)
  4. fewer normal alerts (same measure)

The current defaults are always trial 0, so tuning never picks something that ranks below them.
Each trial is averaged over several injection seeds to avoid tuning to one random draw.
"""
import itertools
import logging
import mlflow
import numpy as np
from scripts.evaluate import evaluate_model

logger = logging.getLogger(__name__)

SEARCH_SPACE = {
    # more trees: steadier scores, slower
    "n_estimators": [100, 150, 300],
    # rows per tree: small finds isolated points, large sees more structure
    "max_samples": [128, 256, 512, 1024],
    # features per tree: below 1.0 stops a few dominant features hiding the rest
    "max_features": [0.5, 0.75, 1.0],
    # share of training events each detector flags, i.e. how alert volume is split between them
    "contamination": [0.0025, 0.005, 0.01],
    "behaviour_contamination": [0.0025, 0.005, 0.01],
}
TUNED_KEYS = list(SEARCH_SPACE)


def _rank_key(result, alert_budget):
    return (result['established_alert_rate'] <= alert_budget, result['scenario_detection_rate'],
            result['model_event_recall'], -result['established_alert_rate'])


def tune(train, holdout, base_params, build_pipeline, n_trials=12, alert_budget=0.02, seeds=(1, 2), random_state=0):
    """
    Returns (best_params, trials). `best_params` is base_params with the tuned keys replaced;
    `trials` lists each candidate with its averaged evaluation metrics.
    """
    rng = np.random.default_rng(random_state)
    grid = list(itertools.product(*SEARCH_SPACE.values()))
    defaults = tuple(base_params[k] for k in TUNED_KEYS)
    others = [c for c in grid if c != defaults]
    picks = rng.choice(len(others), size=min(n_trials - 1, len(others)), replace=False)
    candidates = [defaults] + [others[i] for i in picks]

    trials = []
    for i, values in enumerate(candidates):
        params = {**base_params, **dict(zip(TUNED_KEYS, values))}
        model = build_pipeline(params).fit(train)
        reports = [evaluate_model(model, holdout, history=train, seed=s) for s in seeds]
        result = {
            'trial': i,
            'params': dict(zip(TUNED_KEYS, values)),
            'scenario_detection_rate': float(np.mean([r['scenario_detection_rate'] for r in reports])),
            'model_event_recall': float(np.mean([r['model_event_recall'] for r in reports])),
            'normal_alert_rate': float(np.mean([r['normal_alert_rate'] for r in reports])),
            'established_alert_rate': float(np.mean([r['established_alert_rate'] for r in reports])),
            'incidents_per_day': float(np.mean([r['incidents_per_day'] for r in reports])),
        }
        result['within_budget'] = result['established_alert_rate'] <= alert_budget
        trials.append(result)
        with mlflow.start_run(run_name=f"tuning_trial_{i}", nested=True):
            mlflow.log_params(result['params'])
            mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, (int, float)) and k != 'trial'})
        logger.info("trial %d %s -> detection %.0f%%, model recall %.0f%%, alerts on known identities %.2f%%",
                    i, result['params'], 100 * result['scenario_detection_rate'], 100 * result['model_event_recall'],
                    100 * result['established_alert_rate'])

    best = max(trials, key=lambda r: _rank_key(r, alert_budget))
    if not best['within_budget']:
        logger.warning("No candidate met the alert budget of %.2f%% normal events; using the least noisy of the "
                       "best detectors. Consider a larger budget or more training history.", 100 * alert_budget)
    logger.info("Best: trial %d %s (defaults were trial 0)", best['trial'], best['params'])
    return {**base_params, **best['params']}, trials
