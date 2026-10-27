"""
Scores CloudTrail events with the trained model and the detection rules, and groups alerts into
incidents (one per identity per hour) for analysts.
"""
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
import pandas as pd
from scripts.detections import (RULE_INFO, SEVERITIES, active_rule_severities, alert_severity, evaluate_rules,
                                explain_event, highest_severity, is_routine)
from scripts.features_engineer import prepare_events
from scripts.load_data import drop_own_activity, fetch_events

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL_URI = os.getenv('MODEL_URI', 'models:/isolation_forest_model@champion')

# Raw fields kept next to the scores so an analyst can identify each event
CONTEXT_COLUMNS = ['eventID', 'eventTime', 'actor', 'userIdentity.type', 'eventName', 'eventSource',
                   'awsRegion', 'sourceIPAddress', 'userAgent', 'errorCode']


def load_model(model_uri=DEFAULT_MODEL_URI):
    return mlflow.sklearn.load_model(model_uri)


def score_events(events, model, sort=True):
    """
    Scores raw CloudTrail events with the trained pipeline and the detection rules.

    Returns the identifying context columns plus:
      anomaly_score       Isolation Forest score, lower = more unusual for this account
      behaviour_surprisal how unexpected the event is for this identity, in nats
      forest_anomaly / behaviour_anomaly  which detector flagged it
      anomaly_prediction  -1 if either detector flagged it, else 1
      rules               detection rules that fired
      severity            low/medium/high/critical, or None if the event is not alerted
      reasons             why the event stands out (alerted events only)
      mitre               ATT&CK techniques of the rules that fired
    With sort=True alerted events come first, most severe and most anomalous first;
    otherwise rows keep the input order.
    """
    prepared = prepare_events(events)
    features = model.named_steps['features'].transform(events)
    detector = model.named_steps['detector']

    results = prepared.reindex(columns=CONTEXT_COLUMNS).copy()
    results['anomaly_score'] = detector.decision_function(features)
    results['behaviour_surprisal'] = features['behaviour_surprisal'].to_numpy()
    results['forest_anomaly'] = detector.forest_predict(features) == -1
    results['behaviour_anomaly'] = detector.behaviour_predict(features) == -1
    results['anomaly_prediction'] = detector.predict(features)

    hits = evaluate_rules(prepared, features)
    results['rules'] = [list(hits.columns[row]) for row in hits.to_numpy()]

    n = len(results)
    severities, reasons, mitre = [None] * n, [[] for _ in range(n)], [[] for _ in range(n)]
    candidates = (results['anomaly_prediction'].to_numpy() == -1) | hits.to_numpy().any(axis=1)
    for i in candidates.nonzero()[0]:
        rules = results['rules'].iat[i]
        event, feature_row = prepared.iloc[i], features.iloc[i]
        model_anomaly = results['anomaly_prediction'].iat[i] == -1
        severities[i] = alert_severity(model_anomaly, active_rule_severities(rules, feature_row, model_anomaly))
        if severities[i] is None:
            continue
        reasons[i] = ([RULE_INFO[r]['description'] + (' (routine for this identity)' if is_routine(r, feature_row) else '')
                       for r in rules] + explain_event(event, feature_row))
        mitre[i] = sorted({RULE_INFO[r]['mitre'] for r in rules})
    results['severity'], results['reasons'], results['mitre'] = severities, reasons, mitre

    if not sort:
        return results.reset_index(drop=True)
    rank = results['severity'].map({s: i for i, s in enumerate(SEVERITIES)}).fillna(-1)
    return (results.assign(_rank=rank)
                   .sort_values(['_rank', 'anomaly_score'], ascending=[False, True])
                   .drop(columns='_rank').reset_index(drop=True))


def group_incidents(scored):
    """
    Groups alerted events into incidents: one per identity per clock hour, so a recon sweep of
    40 calls is one item for the analyst rather than 40.
    """
    alerts = scored[scored['severity'].notna()].copy()
    if alerts.empty:
        return []
    alerts['hour'] = pd.to_datetime(alerts['eventTime'], utc=True).dt.floor('h')

    incidents = []
    for (actor, hour), group in alerts.groupby(['actor', 'hour'], dropna=False):
        severity = highest_severity(group['severity'])
        incidents.append({
            'incident_id': f"{actor}|{hour.isoformat() if pd.notna(hour) else 'unknown'}",
            'severity': severity,
            'actor': actor,
            'identity_type': group['userIdentity.type'].iat[0],
            'first_seen': str(group['eventTime'].min()),
            'last_seen': str(group['eventTime'].max()),
            'event_count': len(group),
            'min_anomaly_score': round(float(group['anomaly_score'].min()), 4),
            'detected_by': sorted(({'model:isolation_forest'} if group['forest_anomaly'].any() else set())
                                  | ({'model:behaviour_baseline'} if group['behaviour_anomaly'].any() else set())
                                  | {f"rule:{r}" for rules in group['rules'] for r in rules}),
            'max_behaviour_surprisal': round(float(group['behaviour_surprisal'].max()), 2),
            'mitre': sorted({t for ts in group['mitre'] for t in ts}),
            'reasons': list(dict.fromkeys(r for rs in group['reasons'] for r in rs)),
            'api_calls': group['eventName'].value_counts().to_dict(),
            'source_ips': sorted(group['sourceIPAddress'].dropna().unique().tolist()),
            'regions': sorted(group['awsRegion'].dropna().unique().tolist()),
            'user_agents': sorted(group['userAgent'].dropna().unique().tolist()),
            'event_ids': group['eventID'].tolist(),
        })
    return sorted(incidents, key=lambda i: (-SEVERITIES.index(i['severity']), i['min_anomaly_score']))


def _read_state(state_file):
    if state_file and Path(state_file).exists():
        return json.loads(Path(state_file).read_text())
    return {}


def run_scoring(hours=1, regions=None, model_uri=DEFAULT_MODEL_URI, output=None, csv=None, state_file=None,
                session=None):
    """
    Fetches recent events, scores them and writes incidents as JSON lines.

    With a state file, only events newer than the previous run are alerted on, so a scheduled
    job never re-alerts. Fetching starts at the top of the hour so hourly burst features see the
    identity's whole hour of activity.
    """
    end_time = datetime.now(timezone.utc)
    state = _read_state(state_file)
    alert_from = end_time - timedelta(hours=hours)
    if state.get('last_event_time'):
        alert_from = max(alert_from, pd.Timestamp(state['last_event_time']).to_pydatetime())
    fetch_from = pd.Timestamp(alert_from).floor('h').to_pydatetime()

    events = drop_own_activity(fetch_events(fetch_from, end_time, regions, session=session), session)
    logger.info("Fetched %d events since %s", len(events), fetch_from.isoformat())
    if events.empty:
        return [], {'events_scored': 0}

    model = load_model(model_uri)
    scored = score_events(events, model)

    event_time = pd.to_datetime(scored['eventTime'], utc=True)
    new = (event_time >= pd.Timestamp(alert_from)) & ~scored['eventID'].isin(state.get('last_event_ids', []))
    incidents = group_incidents(scored[new])

    features = model.named_steps['features'].transform(events)
    summary = {
        'run_time': end_time.isoformat(),
        'events_scored': int(new.sum()),
        'events_alerted': int(scored.loc[new, 'severity'].notna().sum()),
        'incidents': len(incidents),
        'model_anomaly_rate': round(float((scored.loc[new, 'anomaly_prediction'] == -1).mean()), 4),
        # Drift signals: a model trained on a stale baseline sees lots of "never seen" values
        'unseen_identity_rate': round(float((features['actor_frequency'] == 0).mean()), 4),
        'unseen_ip_rate': round(float((features['sourceIPAddress_frequency'] == 0).mean()), 4),
        # Regions that could not be read this run: activity there was not checked
        'failed_regions': events.attrs.get('failed_regions', []),
        'model_uri': model_uri,
    }
    detector = model.named_steps['detector']
    contamination = detector.contamination + detector.behaviour_contamination
    if summary['model_anomaly_rate'] > 5 * contamination:
        logger.warning("Model anomaly rate %.1f%% is far above the trained %.1f%%; the baseline may be stale, "
                       "consider retraining", 100 * summary['model_anomaly_rate'], 100 * contamination)

    if output:
        with open(output, 'a', encoding='utf-8') as f:
            for incident in incidents:
                f.write(json.dumps(incident, default=str) + '\n')
    if csv:
        scored[new].to_csv(csv, index=False)
    if state_file and new.any():
        latest = event_time[new].max()
        Path(state_file).write_text(json.dumps({
            'last_event_time': latest.isoformat(),
            'last_event_ids': scored.loc[new & (event_time == latest), 'eventID'].tolist()}))

    return incidents, summary


def format_incident(incident):
    lines = [f"[{incident['severity'].upper()}] {incident['actor']}  "
             f"{incident['first_seen']} -> {incident['last_seen']}  ({incident['event_count']} events)",
             f"    detected by: {', '.join(incident['detected_by'])}"
             + (f"   MITRE: {', '.join(incident['mitre'])}" if incident['mitre'] else ''),
             f"    IPs: {', '.join(incident['source_ips'])}   regions: {', '.join(incident['regions'])}",
             f"    APIs: {', '.join(f'{k} x{v}' for k, v in list(incident['api_calls'].items())[:8])}"]
    lines += [f"    - {reason}" for reason in incident['reasons'][:8]]
    return '\n'.join(lines)
