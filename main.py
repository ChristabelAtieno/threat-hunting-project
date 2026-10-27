"""
CloudTrail threat hunting CLI.

  python main.py train      [--source local|s3|lookup] ...   train, evaluate, register (and promote) a model
  python main.py score      [--hours 1] [--regions all] ...   score recent events, write incidents
  python main.py verdict    --incident-id ID --verdict malicious|benign   record an analyst verdict (gold label)
  python main.py promote    --version N                       point models:/...@champion at a version
  python main.py simulate   --i-own-this-account              live test: safe attacker-like activity in AWS
  python main.py check-simulation [--run-id ID]               live test: which simulated events were caught

Running `python main.py` with no command trains on the local flaws logs, as before.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# AWS credentials / MLflow settings from .env, before any AWS client is created
load_dotenv()
from scripts.train_models import isolation_forest_model

from scripts.load_data import drop_tool_activity, fetch_days, load_data, load_s3, processed_is_current, read_processed

project_root = Path(__file__).resolve().parent
DOCUMENTS_DIR = project_root / "extracted" / "flaws_cloudtrail_logs"
PROCESSED_DIR = project_root / "processed_parquet"

logger = logging.getLogger("threat_hunt")


def train(args):

    print("--- Loading and Normalizing Raw Logs ---")
    if args.source == 'parquet':
        raw_data = read_processed(Path(args.path))
    elif args.source == 's3':
        raw_data = load_s3(args.s3_uri, args.max_files)
    elif args.source == 'lookup':
        raw_data = fetch_days(args.days, args.regions)
    else:
        # Each log folder gets its own parquet cache so datasets never mix
        logs_dir = Path(args.path).resolve()
        processed_dir = PROCESSED_DIR if logs_dir == DOCUMENTS_DIR else project_root / f"processed_parquet_{logs_dir.name}"
        if processed_is_current(processed_dir):
            print(f"Processed data already exists in {processed_dir}. Skip the loading step.")
            raw_data = read_processed(processed_dir)
        else:
            raw_data = load_data(logs_dir, processed_dir)

    raw_data = drop_tool_activity(raw_data)

    if args.source in ('s3', 'lookup'):
        # Keep a snapshot of exactly what this model was trained on (reproducibility, analysis)
        snapshot_dir = project_root / f"processed_parquet_{args.source}"
        snapshot_dir.mkdir(exist_ok=True)
        raw_data.to_parquet(snapshot_dir / "events.parquet", engine='pyarrow')
        print(f"Saved training snapshot to {snapshot_dir / 'events.parquet'}")

    print(f"Final dataset shape: {raw_data.shape}")

    # Feature engineering is the first step of the model pipeline, so it is fitted
    # together with the Isolation Forest and saved with it for inference.
    print("--- Training Feature + Anomaly Detector Pipeline ---")
    _, _, preds, report = isolation_forest_model(raw_data, promote=not args.no_promote,
                                              min_scenario_detection=args.min_detection,
                                              tune_trials=args.tune, alert_budget=args.alert_budget,
                                              label=not args.no_weak_labels)

    print(f"\nAnomalies detected: {(preds == -1).sum()} out of {len(preds)}")
    print(f"Anomaly percentage: {100 * (preds == -1).sum() / len(preds):.2f}%")
    if report and report.get('weak_labels'):
        wl = report['weak_labels']
        print("\n--- Weak labels (judged on injected attacks; known identities' real events assumed benign) ---")
        for name, r in [('LABEL MODEL', wl['label_model']), ('majority vote', wl['majority_vote'])] + list(wl['lfs'].items()):
            print(f"  {name:20s}  precision {r['precision']:.2f}  recall {r['recall']:.2f}  F1 {r['f1']:.2f}")
        print(f"  using: {wl['chosen_combiner']}   beats every single source: {wl['combined_beats_best_lf']}")
    if report:
        print("\n--- Evaluation (time-based holdout + injected attacks) ---")
        print(f"  victim identity: {report['victim_identity']}"
              + ("" if report['victim_is_realistic'] else "  (no realistic identity: baseline scenarios not testable)"))
        for name, s in report['scenarios'].items():
            status = ('CAUGHT' if s['detected'] else 'MISSED') if s['testable'] else 'N/A'
            print(f"  {status:6s} {name:26s} system recall {s['system_event_recall']:.0%}"
                  f"  model-only recall {s['model_event_recall']:.0%}  top severity {s['top_severity']}")
        print(f"  Normal events alerted: {100 * report['normal_alert_rate']:.2f}% overall   "
              f"incidents/day on normal activity: {report['incidents_per_day']}")
        print(f"    known identities: {100 * report['established_alert_rate']:.2f}%   "
              f"identities new since training: {100 * report['new_identity_alert_rate']:.2f}% "
              f"of {report['new_identity_events']} events")


def score(args):
    from scripts.inference import format_incident, run_scoring

    incidents, summary = run_scoring(args.hours, args.regions, args.model_uri, args.output, args.csv, args.state_file)
    print("--- HUNT RESULTS ---")
    for incident in incidents:
        print(format_incident(incident))
    print(json.dumps(summary, indent=2))
    # Non-zero exit when high/critical incidents exist lets a scheduler or CI job page on it
    if args.fail_on_high and any(i['severity'] in ('high', 'critical') for i in incidents):
        sys.exit(2)


def verdict(args):
    from scripts.weak_labels import record_verdict

    lines = Path(args.incidents).read_text(encoding='utf-8').splitlines()
    matches = [i for i in map(json.loads, filter(str.strip, lines)) if i['incident_id'] == args.incident_id]
    if not matches:
        print(f"No incident {args.incident_id} in {args.incidents}")
        sys.exit(1)
    n = record_verdict(args.verdicts, matches[-1], args.verdict, args.analyst, args.note)
    print(f"Recorded {args.verdict} for {n} events of {args.incident_id} in {args.verdicts}")


def promote(args):
    from scripts.train_models import promote_version
    promote_version(args.version)


def simulate(args):
    from scripts.simulate_attacks import describe_plan, run_simulation

    if not args.i_own_this_account:
        print("Dry run. These calls would be made (read-only, or against non-existent resources):\n")
        print('\n'.join(describe_plan(args.region, args.unusual_region)))
        print("\nRe-run with --i-own-this-account to execute them against the account in your AWS credentials.")
        return
    run_id, identity, results = run_simulation(args.region, args.unusual_region)
    print(f"Simulation {run_id} ran as {identity}")
    for r in results:
        print(f"  {r['scenario']:24s} {r['region']:12s} {r['call']:32s} -> {r['outcome']}")
    print(f"\nCloudTrail takes ~5-15 min to surface events. Then run:\n"
          f"  python main.py check-simulation --run-id {run_id}")


def check_simulation(args):
    from scripts.inference import format_incident
    from scripts.simulate_attacks import check_simulation as check

    report = check(args.run_id, args.hours, args.regions, args.model_uri)
    if report is None:
        print("No simulated events found yet. CloudTrail can take up to 15 minutes; try again shortly.")
        sys.exit(1)
    print(report['by_call'].to_string(index=False))
    print(f"\nSimulated events: {report['simulated_events_found']}, alerted: {report['simulated_events_alerted']} "
          f"(model flagged {report['simulated_events_flagged_by_model']})")
    print(f"Background events in window: {report['background_events']}, alerted: {report['background_events_alerted']}")
    print("\n--- Incidents the analyst would see ---")
    for incident in report['incidents']:
        print(format_incident(incident))


def build_parser():
    parser = argparse.ArgumentParser(description="CloudTrail threat hunting")
    parser.add_argument('-v', '--verbose', action='store_true')
    sub = parser.add_subparsers(dest='command')

    p = sub.add_parser('train', help="train, evaluate and register a model")
    p.add_argument('--source', choices=['local', 's3', 'lookup', 'parquet'], default='local')
    p.add_argument('--path', default=str(DOCUMENTS_DIR),
                   help="local: folder of *.json.gz logs; parquet: a saved snapshot, e.g. processed_parquet_lookup/events.parquet")
    p.add_argument('--s3-uri', help="s3://bucket/prefix of a trail's log files")
    p.add_argument('--max-files', type=int, help="cap on S3 log files read")
    p.add_argument('--days', type=int, default=30, help="lookup: days of history (max 90)")
    p.add_argument('--regions', default='all', help="lookup: 'all' or comma-separated regions")
    p.add_argument('--min-detection', type=float, default=0.8, help="share of attack scenarios needed to promote")
    p.add_argument('--no-promote', action='store_true', help="register only; do not move the champion alias")
    p.add_argument('--tune', type=int, default=0, metavar='TRIALS',
                   help="search this many hyperparameter candidates before training (e.g. 12); default: off")
    p.add_argument('--no-weak-labels', action='store_true', help="skip weak-label evaluation and labelling")
    p.add_argument('--alert-budget', type=float, default=0.02,
                   help="tuning: max share of normal events that may alert (default 0.02 = 2%%)")
    p.set_defaults(func=train)

    p = sub.add_parser('score', help="score recent CloudTrail events")
    p.add_argument('--hours', type=float, default=1, help="look-back window in hours (default: 1)")
    p.add_argument('--regions', default='all', help="'all' (default) or comma-separated regions")
    p.add_argument('--model-uri', default=None, help="MLflow model URI (default: MODEL_URI or models:/isolation_forest_model@champion)")
    p.add_argument('--output', help="append incidents as JSON lines to this file")
    p.add_argument('--csv', help="write every scored event to this CSV")
    p.add_argument('--state-file', help="remember the last scored event so scheduled runs never re-alert")
    p.add_argument('--fail-on-high', action='store_true', help="exit 2 if any high/critical incident")
    p.set_defaults(func=score)

    p = sub.add_parser('verdict', help="record an analyst verdict on an incident (gold label for training)")
    p.add_argument('--incident-id', required=True)
    p.add_argument('--verdict', required=True, choices=['malicious', 'benign'])
    p.add_argument('--incidents', default='incidents.jsonl', help="incidents file written by `score --output`")
    p.add_argument('--verdicts', default=os.getenv('VERDICTS_FILE', str(project_root / 'verdicts.csv')))
    p.add_argument('--analyst', default='')
    p.add_argument('--note', default='')
    p.set_defaults(func=verdict)

    p = sub.add_parser('promote', help="make a registered model version the champion")
    p.add_argument('--version', required=True)
    p.set_defaults(func=promote)

    p = sub.add_parser('simulate', help="live test: safe attacker-like activity in your account")
    p.add_argument('--i-own-this-account', action='store_true', help="actually make the calls (default: dry run)")
    p.add_argument('--region', default='us-east-1', help="home region (IAM/STS events land in us-east-1)")
    p.add_argument('--unusual-region', default='sa-east-1', help="a region your account does not normally use")
    p.set_defaults(func=simulate)

    p = sub.add_parser('check-simulation', help="live test: report which simulated events were alerted")
    p.add_argument('--run-id', help="simulation run id (default: any simulation run)")
    p.add_argument('--hours', type=float, default=2)
    p.add_argument('--regions', default='all')
    p.add_argument('--model-uri', default=None)
    p.set_defaults(func=check_simulation)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command is None:
        args = parser.parse_args(['train'])
    if getattr(args, 'model_uri', 'unset') is None:
        from scripts.inference import DEFAULT_MODEL_URI
        args.model_uri = DEFAULT_MODEL_URI
    args.func(args)
