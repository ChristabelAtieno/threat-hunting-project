"""
Known-bad detections and plain-English reasons for alerts.

The Isolation Forest finds activity that is unusual for this account; these rules catch activity
that is dangerous even when it is not unusual. Both feed the same alerts.
"""
import numpy as np
import pandas as pd

SEVERITIES = ['low', 'medium', 'high', 'critical']

DEFENSE_EVASION_APIS = {
    'StopLogging', 'DeleteTrail', 'UpdateTrail', 'PutEventSelectors', 'DeleteFlowLogs',
    'DeleteDetector', 'UpdateDetector', 'DisassociateFromMasterAccount', 'DisableSecurityHub',
    'StopConfigurationRecorder', 'DeleteConfigurationRecorder', 'DeleteDeliveryChannel'}

PERSISTENCE_APIS = {
    'CreateUser', 'CreateAccessKey', 'CreateLoginProfile', 'UpdateLoginProfile',
    'AttachUserPolicy', 'AttachRolePolicy', 'AttachGroupPolicy', 'PutUserPolicy', 'PutRolePolicy',
    'PutGroupPolicy', 'CreatePolicyVersion', 'SetDefaultPolicyVersion', 'AddUserToGroup',
    'UpdateAssumeRolePolicy'}

EXPOSURE_APIS = {
    'PutBucketPolicy', 'PutBucketAcl', 'DeleteBucketPublicAccessBlock', 'DeleteAccountPublicAccessBlock',
    'ModifySnapshotAttribute', 'ModifyImageAttribute', 'ModifyDBSnapshotAttribute',
    'AuthorizeSecurityGroupIngress'}

CREDENTIAL_ACCESS_APIS = {'GetPasswordData', 'GetSecretValue'}

ASSUME_ROLE_APIS = {'AssumeRole', 'AssumeRoleWithSAML', 'AssumeRoleWithWebIdentity'}

SENSITIVE_APIS = DEFENSE_EVASION_APIS | PERSISTENCE_APIS | EXPOSURE_APIS | CREDENTIAL_ACCESS_APIS | {'ConsoleLogin'}

# Calls that enumerate resources; only these are marked as part of a recon sweep
DISCOVERY_PREFIXES = ('List', 'Describe', 'Get')

# Burst thresholds per identity per clock hour
RECON_DISTINCT_APIS = 15
ERROR_BURST = 10
# An identity that made the same sensitive call this often in training is doing routine work
# (e.g. CI deploying IAM roles): its rule hits only alert if the model also finds the event unusual
ROUTINE_CALLS = 20

RULES = [
    # name, severity, MITRE ATT&CK technique, description, condition(events, features)
    ('cloud_logging_tampering', 'critical', 'T1562.008',
     'attempt to disable or tamper with security logging/monitoring',
     lambda e, f: e['eventName'].isin(DEFENSE_EVASION_APIS)),
    ('root_account_activity', 'high', 'T1078.004',
     'root user activity',
     lambda e, f: e['userIdentity.type'].eq('Root') & e['userIdentity.invokedBy'].eq('Unknown')),
    ('console_login_without_mfa', 'high', 'T1078.004',
     'console login without MFA',
     lambda e, f: e['eventName'].eq('ConsoleLogin') & e['additionalEventData.MFAUsed'].astype(str).eq('No')
                  & ~e['responseElements.ConsoleLogin'].astype(str).eq('Failure')),
    ('failed_console_login', 'medium', 'T1110',
     'failed console login',
     lambda e, f: e['eventName'].eq('ConsoleLogin') & e['responseElements.ConsoleLogin'].astype(str).eq('Failure')),
    ('iam_persistence_or_privilege_change', 'high', 'T1098',
     'IAM change that can create persistence or escalate privileges',
     lambda e, f: e['eventName'].isin(PERSISTENCE_APIS)),
    ('resource_exposure', 'medium', 'T1530',
     'change that can expose data or open network access',
     lambda e, f: e['eventName'].isin(EXPOSURE_APIS)),
    ('credential_access', 'medium', 'T1555.006',
     'secret or password retrieval',
     lambda e, f: e['eventName'].isin(CREDENTIAL_ACCESS_APIS)),
    ('new_role_assumption_path', 'medium', 'T1078.004',
     'first-time role assumption by role chaining or into another account (lateral movement)',
     lambda e, f: (f['assume_role_new_edge'] == 1) & ((f['role_chaining'] == 1) | (f['cross_account_assume'] == 1))),
    ('recon_burst', 'medium', 'T1580',
     f'{RECON_DISTINCT_APIS}+ distinct APIs by one identity within an hour (enumeration)',
     lambda e, f: (_count(f['actor_distinct_apis_in_hour']) >= RECON_DISTINCT_APIS)
                  & e['eventName'].str.startswith(DISCOVERY_PREFIXES)),
    ('error_burst', 'medium', 'T1087.004',
     f'{ERROR_BURST}+ failed calls by one identity within an hour',
     lambda e, f: (_count(f['actor_errors_in_hour']) >= ERROR_BURST) & (f['has_error'] == 1)),
]
RULE_INFO = {name: {'severity': sev, 'mitre': mitre, 'description': desc} for name, sev, mitre, desc, _ in RULES}
NEVER_DOWNGRADE = {'cloud_logging_tampering', 'root_account_activity'}


def _count(log_feature):
    """Undo the log1p applied to count features."""
    return np.expm1(log_feature).round().astype(int)


def evaluate_rules(events, features):
    """
    Returns one boolean column per rule for prepared events (features_engineer.prepare_events)
    and their feature matrix.
    """
    return pd.DataFrame({name: condition(events, features).fillna(False).astype(bool)
                         for name, _, _, _, condition in RULES}, index=events.index)


def is_routine(rule, feature_row):
    """True if this identity made this call often in training (e.g. CI deploying IAM roles every day)."""
    return rule not in NEVER_DOWNGRADE and _count(feature_row['actor_eventName_frequency']) >= ROUTINE_CALLS


def active_rule_severities(rules, feature_row, model_anomaly):
    """
    Severity of each rule hit that should alert. Routine hits only alert when the model also finds
    the event unusual (new IP, odd hour, burst...), and then one level lower.
    """
    severities = []
    for rule in rules:
        severity = RULE_INFO[rule]['severity']
        if is_routine(rule, feature_row):
            if not model_anomaly:
                continue
            severity = SEVERITIES[max(SEVERITIES.index(severity) - 1, 0)]
        severities.append(severity)
    return severities


def alert_severity(model_anomaly, rule_severities):
    """
    Combines both detectors: model-only anomalies are medium; rule hits keep their severity and
    are raised one level when the model also finds the event unusual.
    """
    if not rule_severities:
        return 'medium' if model_anomaly else None
    level = max(SEVERITIES.index(s) for s in rule_severities)
    if model_anomaly:
        level = min(level + 1, len(SEVERITIES) - 1)
    return SEVERITIES[level]


def highest_severity(severities):
    """Highest of an iterable of severities (None values ignored), or None."""
    present = [s for s in severities if isinstance(s, str)]
    return max(present, key=SEVERITIES.index) if present else None


def explain_event(event, f):
    """Plain-English reasons an event stands out, from its feature values."""
    reasons = []
    actor_known = f['actor_frequency'] > 0

    if not actor_known:
        reasons.append(f"identity {event['actor']} never seen in training")

    for col, label in [('sourceIPAddress', 'source IP'), ('eventName', 'API'),
                       ('awsRegion', 'region'), ('userAgentFamily', 'client')]:
        if f[f'{col}_frequency'] == 0:
            reasons.append(f"{label} {event[col]} never seen in this account")
        elif actor_known and f[f'actor_{col}_frequency'] == 0:
            reasons.append(f"first time this identity used {label} {event[col]}")

    if actor_known and f['actor_hourOfDay_frequency'] == 0:
        reasons.append(f"identity is never active at {int(f['hour_of_day']):02d}:00 UTC")
    elif f['is_night']:
        reasons.append(f"off-hours activity ({int(f['hour_of_day']):02d}:00 UTC)")
    if f['access_denied']:
        reasons.append(f"access denied ({event['errorCode']})")
    elif f['has_error']:
        reasons.append(f"failed call ({event['errorCode']})")

    distinct_apis, errors = _count(f['actor_distinct_apis_in_hour']), _count(f['actor_errors_in_hour'])
    if distinct_apis >= 10:
        reasons.append(f"{distinct_apis} distinct APIs by this identity within the hour")
    if errors >= 5:
        reasons.append(f"{errors} failed calls by this identity within the hour")
    if f['is_sensitive_api']:
        reasons.append(f"sensitive API {event['eventName']}")

    # graph
    if f['assume_role_new_edge']:
        how = ' (role chaining)' if f['role_chaining'] else ''
        how += ' into another account' if f['cross_account_assume'] else ''
        reasons.append(f"first time this identity assumed role {event['requestParameters.roleArn']}{how}")
    shared = _count(f['ip_identities_in_hour'])
    if shared >= 3:
        reasons.append(f"{shared} different identities used IP {event['sourceIPAddress']} within the hour")
    if actor_known and f['peer_service_share'] == 0 and f['eventSource_frequency'] > 0:
        reasons.append(f"none of this identity's peers use {event['eventSource']}")
    new_edges = _count(f['actor_new_edges_in_hour'])
    if new_edges >= 5:
        reasons.append(f"{new_edges} new connections (IP/API/region/client/role) for this identity within the hour")
    return reasons
