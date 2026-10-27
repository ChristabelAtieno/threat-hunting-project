"""
Measures the detector before it is promoted: train on the earlier part of the logs, then score the
later part with labelled attack scenarios injected into it.

The scenarios are built from a real, active identity in the holdout so they look like that
identity's credentials being misused. CloudTrail has no ground-truth labels, so this is a
regression check of known attack shapes, not a full measure of real-world recall.
"""
import uuid
import numpy as np
import pandas as pd
from scripts.detections import highest_severity
from scripts.features_engineer import prepare_events
from scripts.inference import group_incidents, score_events

# RFC 5737 documentation ranges: never real traffic
ATTACKER_IP = '203.0.113.66'
ATTACKER_IP_2 = '198.51.100.23'
ATTACKER_AGENT = 'aws-cli/1.18.69 Python/3.8.5 Linux/5.7.0-kali1-amd64 botocore/1.17.0'

RECON_APIS = {
    'GetCallerIdentity': 'sts.amazonaws.com', 'ListUsers': 'iam.amazonaws.com', 'ListRoles': 'iam.amazonaws.com',
    'ListPolicies': 'iam.amazonaws.com', 'ListGroups': 'iam.amazonaws.com', 'ListAccessKeys': 'iam.amazonaws.com',
    'GetAccountAuthorizationDetails': 'iam.amazonaws.com', 'GetAccountSummary': 'iam.amazonaws.com',
    'ListBuckets': 's3.amazonaws.com', 'DescribeInstances': 'ec2.amazonaws.com',
    'DescribeSecurityGroups': 'ec2.amazonaws.com', 'DescribeVpcs': 'ec2.amazonaws.com',
    'DescribeSnapshots': 'ec2.amazonaws.com', 'DescribeRegions': 'ec2.amazonaws.com',
    'ListFunctions20150331': 'lambda.amazonaws.com', 'DescribeDBInstances': 'rds.amazonaws.com',
    'ListKeys': 'kms.amazonaws.com', 'ListSecrets': 'secretsmanager.amazonaws.com',
    'DescribeTrails': 'cloudtrail.amazonaws.com', 'ListDetectors': 'guardduty.amazonaws.com'}


def time_split(data, holdout_fraction=0.2):
    """Splits on eventTime so the model is evaluated on activity that happened after its training data."""
    times = pd.to_datetime(data['eventTime'], utc=True, errors='coerce')
    cutoff = times.quantile(1 - holdout_fraction)
    return data[times < cutoff], data[times >= cutoff]


# A victim needs an established history in training, or "misuse of this identity" is meaningless
MIN_VICTIM_HISTORY = 50
# Scenarios that only make sense against an identity with a realistic, human/workload baseline.
# AWS service-linked roles call hundreds of APIs by design, so "new API / region" is normal for them.
BASELINE_SCENARIOS = {'new_ip_normal_activity', 'unusual_region_compute', 'recon_from_usual_ip',
                      'lateral_movement_role_chain'}
# Needs this many established identities (victim included); otherwise it is not injected
SHARED_IP_IDENTITIES = 3
ATTACKER_IP_3 = '203.0.113.150'


def _is_realistic_victim(actor):
    return not actor.startswith('type:') and ':assumed-role/AWSServiceRoleFor' not in actor


def pick_victim(history):
    """
    The identity whose credentials the scenarios pretend were stolen, plus whether it is realistic.

    Prefers the most active IAM user, then assumed role, then federated user, among identities with
    MIN_VICTIM_HISTORY+ events in `history` (the training period), excluding AWS service-linked
    roles. If none qualifies, falls back to the most active identity and returns realistic=False,
    so baseline-dependent scenarios are reported as not testable rather than missed.
    """
    prepared = prepare_events(history)
    counts = prepared['actor'].value_counts()
    established = prepared[prepared['actor'].map(counts) >= MIN_VICTIM_HISTORY]
    realistic = established[established['actor'].map(_is_realistic_victim).astype(bool)]
    for identity_type in ('IAMUser', 'AssumedRole', 'FederatedUser'):
        candidates = realistic[realistic['userIdentity.type'].eq(identity_type)]
        if not candidates.empty:
            actor = candidates['actor'].value_counts().index[0]
            return history[prepared['actor'] == actor], prepared[prepared['actor'] == actor], True
    actor = counts.index[0]
    return history[prepared['actor'] == actor], prepared[prepared['actor'] == actor], False


def inject_attacks(holdout, history=None, seed=42):
    """
    Returns the holdout with attack events appended; `scenario` is NaN for real events.
    The victim's habits (IP, client, API, hour) come from `history` (default: the holdout).
    """
    rng = np.random.default_rng(seed)
    victim_raw, victim, realistic = pick_victim(holdout if history is None else history)
    template = victim_raw.iloc[0].to_dict()
    if not realistic:
        # No realistic identity to impersonate (e.g. only AWS service activity). Attackers cannot make
        # their calls look AWS-service-originated, so attack as a dormant IAM user with a stolen key.
        template.update(_dormant_user(holdout if history is None else history))
    usual_ip = victim['sourceIPAddress'].mode().iat[0]
    usual_agent = victim_raw['userAgent'].mode().iat[0] if victim_raw['userAgent'].notna().any() else None
    usual_api = victim['eventName'].mode().iat[0]
    usual_source = victim.loc[victim['eventName'] == usual_api, 'eventSource'].iat[0]
    usual_hour = int(victim['eventTime'].dt.hour.mode().iat[0])
    account = _account_id(holdout if history is None else history)
    start_day = pd.to_datetime(holdout['eventTime'], utc=True).min().floor('D') + pd.Timedelta(days=1)
    others = _other_identities(holdout if history is None else history, victim['actor'].iat[0],
                               SHARED_IP_IDENTITIES - 1)
    hop = _role_hop(holdout if history is None else history)

    rows = []

    def add(scenario, day, hour, event_name, event_source, base=None, **fields):
        when = start_day + pd.Timedelta(days=day, hours=hour, minutes=int(rng.integers(0, 59)))
        row = dict(template if base is None else base)
        row.update({'eventID': f'injected-{uuid.UUID(int=int(rng.integers(0, 2**63)))}',
                    'eventTime': when.strftime('%Y-%m-%dT%H:%M:%SZ'), 'eventName': event_name,
                    'eventSource': event_source, 'errorCode': None, 'errorMessage': None,
                    'sourceIPAddress': ATTACKER_IP, 'userAgent': ATTACKER_AGENT, 'scenario': scenario})
        row.update(fields)
        rows.append(row)

    # 1. Stolen access key used for an enumeration sweep from a new IP at night
    for name, source in RECON_APIS.items():
        add('stolen_credentials_recon', 0, 2, name, source, readOnly='true')
    # 2. Persistence / privilege escalation with the stolen key
    for name in ['CreateAccessKey', 'AttachUserPolicy', 'PutUserPolicy', 'CreateLoginProfile']:
        add('privilege_escalation', 1, 3, name, 'iam.amazonaws.com', readOnly='false')
    # 3. Covering tracks
    for name, source in [('StopLogging', 'cloudtrail.amazonaws.com'), ('DeleteTrail', 'cloudtrail.amazonaws.com'),
                         ('DeleteDetector', 'guardduty.amazonaws.com')]:
        add('defense_evasion', 2, 4, name, source, readOnly='false')
    # 4. Crypto-mining in a region the account never uses, from the usual IP and client (subtle)
    for _ in range(5):
        add('unusual_region_compute', 3, usual_hour, 'RunInstances', 'ec2.amazonaws.com', readOnly='false',
            awsRegion='ap-southeast-3', sourceIPAddress=usual_ip, userAgent=usual_agent)
    # 5. Permission probing: many denied calls
    for name, source in list(RECON_APIS.items())[:12]:
        add('access_denied_spray', 4, 1, name, source, readOnly='true', errorCode='AccessDenied',
            errorMessage='User is not authorized to perform this operation', sourceIPAddress=ATTACKER_IP_2)
    # 6. Root console login without MFA
    add('root_console_login', 5, 23, 'ConsoleLogin', 'signin.amazonaws.com',
        **{'userIdentity.type': 'Root', 'userIdentity.arn': f'arn:aws:iam::{account}:root',
           'userIdentity.userName': None, 'userIdentity.accessKeyId': None, 'userIdentity.invokedBy': None,
           'userAgent': 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0',
           'additionalEventData.MFAUsed': 'No', 'responseElements.ConsoleLogin': 'Success',
           'eventType': 'AwsConsoleSignIn'})
    # 7. The victim's normal work, but from a new IP: the hardest case
    for _ in range(3):
        add('new_ip_normal_activity', 6, usual_hour, usual_api, usual_source, userAgent=usual_agent)
    # 8. Compromised workstation or insider: enumeration and denied calls from the victim's own IP and
    #    client at a normal hour. Only behaviour gives it away (mirrors the live simulation), so it
    #    keeps tuning from trading model sensitivity for quietness.
    for i, (name, source) in enumerate(list(RECON_APIS.items())[:16]):
        add('recon_from_usual_ip', 7, usual_hour, name, source, readOnly='true', sourceIPAddress=usual_ip,
            userAgent=usual_agent, **({'errorCode': 'AccessDenied', 'errorMessage': 'not authorized'} if i % 4 else {}))
    # 9. Lateral movement: from its usual IP and client, the victim assumes a role it has never used,
    #    and that role chains into another account's admin role and reads secrets
    add('lateral_movement_role_chain', 8, usual_hour, 'AssumeRole', 'sts.amazonaws.com', readOnly='true',
        sourceIPAddress=usual_ip, userAgent=usual_agent, **{'requestParameters.roleArn': f'arn:aws:iam::{account}:role/prod-admin'})
    chained = {'userIdentity.type': 'AssumedRole', 'userIdentity.userName': None, 'userIdentity.accessKeyId': 'ASIAINJECTED',
               'userIdentity.arn': f'arn:aws:sts::{account}:assumed-role/prod-admin/{victim["actor"].iat[0].split("/")[-1]}'}
    add('lateral_movement_role_chain', 8, usual_hour, 'AssumeRole', 'sts.amazonaws.com', readOnly='true',
        sourceIPAddress=usual_ip, userAgent=usual_agent,
        **chained, **{'requestParameters.roleArn': 'arn:aws:iam::999988887777:role/OrganizationAccountAccessRole'})
    for name, source in [('GetSecretValue', 'secretsmanager.amazonaws.com'), ('ListUsers', 'iam.amazonaws.com')]:
        add('lateral_movement_role_chain', 8, usual_hour, name, source, readOnly='true', sourceIPAddress=usual_ip,
            userAgent=usual_agent, **chained)
    # 10. One attacker machine using several stolen keys: different identities, same new IP, same hour,
    #     each doing its own normal work
    if len(others) == SHARED_IP_IDENTITIES - 1:
        for base in [template] + others:
            for _ in range(2):
                add('shared_attacker_ip', 9, usual_hour, base['eventName'], base['eventSource'], base=base,
                    sourceIPAddress=ATTACKER_IP_3, userAgent=base.get('userAgent'))

    # 11. Privilege hop: an identity that routinely assumes a role assumes a different, existing role it
    #     has never used, from its usual IP, client and hour. Only the identity->role edge is new.
    if hop is not None:
        assumer, assumer_ip, assumer_agent, assumer_hour, role = hop
        for _ in range(2):
            add('role_hop_known_role', 10, assumer_hour, 'AssumeRole', 'sts.amazonaws.com', base=assumer,
                readOnly='true', sourceIPAddress=assumer_ip, userAgent=assumer_agent,
                **{'requestParameters.roleArn': role})

    attacks = pd.DataFrame(rows)
    attacks = attacks.reindex(columns=list(dict.fromkeys(list(holdout.columns) + list(attacks.columns))))
    return pd.concat([holdout.assign(scenario=np.nan), attacks], ignore_index=True)


DORMANT_USER = 'dormant-admin'


def _account_id(history):
    accounts = history['userIdentity.arn'].dropna().astype(str).str.extract(r':(\d{12}):', expand=False).dropna()
    return accounts.mode().iat[0] if not accounts.empty else '123456789012'


def _dormant_user(history):
    """Identity fields for an IAM user that exists but never shows up in the logs."""
    return {'userIdentity.type': 'IAMUser', 'userIdentity.arn': f'arn:aws:iam::{_account_id(history)}:user/{DORMANT_USER}',
            'userIdentity.userName': DORMANT_USER, 'userIdentity.accessKeyId': 'AKIAINJECTEDDORMANT',
            'userIdentity.invokedBy': None, 'userIdentity.sessionContext.attributes.mfaAuthenticated': None}


def _role_hop(history):
    """
    (template event, usual IP, usual client, usual hour, target role) for the established identity that
    assumes roles most often (AWS service-linked roles excluded), and an existing role it has never
    assumed; None if the account has none.
    """
    prepared = prepare_events(history)
    assumes = prepared[prepared['eventName'].eq('AssumeRole') & prepared['requestParameters.roleArn'].notna()
                       & prepared['actor'].map(_is_realistic_victim).astype(bool)]
    if assumes.empty:
        return None
    assumer = assumes['actor'].value_counts().index[0]
    own = prepared[prepared['actor'] == assumer]
    already = set(assumes.loc[assumes['actor'] == assumer, 'requestParameters.roleArn'])
    # existing roles: assumed by someone, or seen acting as an assumed-role identity
    acting = prepared['actor'].str.extract(r'^arn:aws[\w-]*:sts::(\d{12}):assumed-role/([^/]+)$').dropna()
    known = set(assumes['requestParameters.roleArn']) | {f"arn:aws:iam::{a}:role/{r}" for a, r in acting.itertuples(index=False)}
    candidates = sorted(known - already)
    if not candidates or len(own) < MIN_VICTIM_HISTORY:
        return None
    raw = history[(prepared['actor'] == assumer).to_numpy()]
    template = raw[(own['eventName'] == 'AssumeRole').to_numpy()].iloc[0].to_dict()
    return (template, own['sourceIPAddress'].mode().iat[0], raw['userAgent'].mode().iat[0] if raw['userAgent'].notna().any()
            else None, int(own['eventTime'].dt.hour.mode().iat[0]), candidates[0])


def _other_identities(history, victim_actor, n):
    """Template events for the n most active established identities other than the victim."""
    prepared = prepare_events(history)
    counts = prepared['actor'].value_counts()
    candidates = [a for a in counts.index if a != victim_actor and counts[a] >= MIN_VICTIM_HISTORY
                  and not a.startswith('type:')][:n]
    return [history[(prepared['actor'] == a).to_numpy()].iloc[0].to_dict() for a in candidates]


def evaluate_model(model, holdout, history=None, seed=42):
    """
    Scores the holdout with injected attacks and returns a report:
      scenario_detection_rate  share of *testable* scenarios with at least one alerted event (model or rules)
      untestable_scenarios     scenarios skipped: no identity with a realistic baseline, too few
                               established identities to share an IP, or nobody who assumes roles
      model_event_recall       share of injected events the Isolation Forest alone flags
      normal_alert_rate        share of real holdout events that raise an alert (all analyst noise)
      established_alert_rate   the same, for identities seen in training: the noise the model's
                               parameters control (used for tuning and the alert budget)
      new_identity_alert_rate  the same, for identities first seen after training; these alert
                               whatever the parameters (peer-group baselines would address them)
      incidents_per_day        incidents raised on real holdout activity per day (analyst load)
    """
    _, victim, realistic = pick_victim(holdout if history is None else history)
    data = inject_attacks(holdout, history, seed)
    labels = data.pop('scenario')
    scored = score_events(data, model, sort=False)
    scored['scenario'] = labels.to_numpy()

    attack = scored['scenario'].notna()
    alerted = scored['severity'].notna()
    model_flag = scored['anomaly_prediction'] == -1

    scenarios = {}
    for name, group in scored[attack].groupby('scenario'):
        scenarios[name] = {
            'events': len(group),
            'detected': bool(group['severity'].notna().any()),
            'system_event_recall': round(float(group['severity'].notna().mean()), 3),
            'model_event_recall': round(float((group['anomaly_prediction'] == -1).mean()), 3),
            'top_severity': highest_severity(group['severity']),
            'testable': realistic or name not in BASELINE_SCENARIOS,
        }
    testable = [s['detected'] for s in scenarios.values() if s['testable']]
    not_injected = [n for n in ('shared_attacker_ip', 'role_hop_known_role') if n not in scenarios]

    real = scored[~attack]
    known_actors = set(prepare_events(history)['actor']) if history is not None else set()
    established = real['actor'].isin(known_actors) if known_actors else pd.Series(True, index=real.index)
    days = max((pd.to_datetime(real['eventTime'], utc=True).max()
                - pd.to_datetime(real['eventTime'], utc=True).min()).total_seconds() / 86400, 1)
    return {
        'scenario_detection_rate': round(float(np.mean(testable)), 3),
        'untestable_scenarios': sorted([n for n, s in scenarios.items() if not s['testable']] + not_injected),
        'system_event_recall': round(float(alerted[attack].mean()), 3),
        'model_event_recall': round(float(model_flag[attack].mean()), 3),
        'normal_alert_rate': round(float(alerted[~attack].mean()), 4),
        'established_alert_rate': round(float(real.loc[established, 'severity'].notna().mean()), 4)
        if established.any() else 0.0,
        'new_identity_alert_rate': round(float(real.loc[~established, 'severity'].notna().mean()), 4)
        if (~established).any() else 0.0,
        'new_identity_events': int((~established).sum()),
        'normal_model_anomaly_rate': round(float(model_flag[~attack].mean()), 4),
        'incidents_per_day': round(len(group_incidents(real)) / days, 2),
        'holdout_events': int((~attack).sum()),
        'victim_identity': victim['actor'].iat[0] if realistic else f"{DORMANT_USER} (dormant IAM user; no realistic identity)",
        'victim_is_realistic': realistic,
        'scenarios': scenarios,
    }

