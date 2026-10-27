"""
Live test: generates attacker-like CloudTrail activity in your own AWS account, then checks
which of those events the detector alerts on.

Safe by design. Every call is either read-only, or a dangerous API aimed at a resource with a
random name that does not exist, so it fails (TrailNotFound / NoSuchEntity) and nothing is
created, changed or deleted. Failed calls are still recorded by CloudTrail, which is the point.
Calls the identity lacks permission for fail with AccessDenied, which is also recorded.

Every simulated call carries `threat-hunt-sim/<run_id>` in its user agent so it can be found later.
"""
import logging
import uuid
from datetime import datetime, timezone
import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from scripts.detections import highest_severity
from scripts.inference import group_incidents, load_model, score_events
from scripts.load_data import SIMULATION_AGENT, drop_own_activity, fetch_events

logger = logging.getLogger(__name__)

SIM_MARKER = SIMULATION_AGENT
DEFAULT_UNUSUAL_REGION = 'sa-east-1'


def _plan(run_id, home_region, unusual_region, account):
    """(region, service, operation, kwargs, scenario). Fake names make the risky calls fail safely."""
    fake = f"{SIM_MARKER}-{run_id}-does-not-exist"
    readonly_policy = 'arn:aws:iam::aws:policy/ReadOnlyAccess'
    recon = [
        ('sts', 'get_caller_identity', {}), ('iam', 'list_users', {'MaxItems': 5}),
        ('iam', 'list_roles', {'MaxItems': 5}), ('iam', 'list_groups', {'MaxItems': 5}),
        ('iam', 'list_policies', {'Scope': 'Local', 'MaxItems': 5}), ('iam', 'get_account_summary', {}),
        ('iam', 'list_access_keys', {}), ('s3', 'list_buckets', {}),
        ('ec2', 'describe_instances', {'MaxResults': 5}), ('ec2', 'describe_security_groups', {'MaxResults': 5}),
        ('ec2', 'describe_vpcs', {}), ('ec2', 'describe_snapshots', {'OwnerIds': ['self'], 'MaxResults': 5}),
        ('ec2', 'describe_regions', {}), ('lambda', 'list_functions', {'MaxItems': 5}),
        ('rds', 'describe_db_instances', {'MaxRecords': 20}), ('kms', 'list_keys', {'Limit': 5}),
        ('secretsmanager', 'list_secrets', {'MaxResults': 5}), ('cloudtrail', 'describe_trails', {}),
        ('guardduty', 'list_detectors', {}),
    ]
    plan = [(home_region, svc, op, kw, 'recon_sweep') for svc, op, kw in recon]
    plan += [(unusual_region, 'ec2', op, kw, 'unusual_region_recon') for op, kw in
             [('describe_instances', {'MaxResults': 5}), ('describe_key_pairs', {}), ('describe_images', {'Owners': ['self']})]]
    plan += [
        (home_region, 'cloudtrail', 'stop_logging', {'Name': fake}, 'defense_evasion_attempt'),
        (home_region, 'cloudtrail', 'delete_trail', {'Name': fake}, 'defense_evasion_attempt'),
        (home_region, 'iam', 'create_access_key', {'UserName': fake}, 'persistence_attempt'),
        (home_region, 'iam', 'attach_user_policy', {'UserName': fake, 'PolicyArn': readonly_policy}, 'persistence_attempt'),
        (home_region, 'iam', 'create_login_profile', {'UserName': fake, 'Password': f'Sim-{uuid.uuid4()}!'}, 'persistence_attempt'),
    ]
    plan += [(home_region, 'iam', 'get_user', {'UserName': f'{fake}-{i}'}, 'error_burst') for i in range(10)]
    # lateral movement attempt: assume a role (in this account) that does not exist; denied, but recorded
    plan += [(home_region, 'sts', 'assume_role', {'RoleArn': f'arn:aws:iam::{account}:role/{fake}',
                                                   'RoleSessionName': SIM_MARKER}, 'role_hop_attempt')]
    return plan


def describe_plan(home_region='us-east-1', unusual_region=DEFAULT_UNUSUAL_REGION):
    return [f"{scenario:24s} {region:12s} {service}.{op}" for region, service, op, _, scenario
            in _plan('preview', home_region, unusual_region, '<your-account>')]


def run_simulation(home_region='us-east-1', unusual_region=DEFAULT_UNUSUAL_REGION, session=None):
    """Executes the plan and returns (run_id, account_arn, per-call results)."""
    session = session or boto3.Session()
    run_id = uuid.uuid4().hex[:8]
    config = Config(user_agent_extra=f"{SIM_MARKER}/{run_id}", retries={'mode': 'standard', 'max_attempts': 3})
    clients = {}

    def client(service, region):
        if (service, region) not in clients:
            clients[(service, region)] = session.client(service, region_name=region, config=config)
        return clients[(service, region)]

    identity = client('sts', home_region).get_caller_identity()['Arn']
    results = []
    for region, service, op, kwargs, scenario in _plan(run_id, home_region, unusual_region, identity.split(':')[4]):
        try:
            getattr(client(service, region), op)(**kwargs)
            outcome = 'ok'
        except ClientError as e:
            outcome = e.response['Error']['Code']
        except Exception as e:  # e.g. a service not available in the region
            outcome = type(e).__name__
        results.append({'scenario': scenario, 'region': region, 'call': f"{service}.{op}", 'outcome': outcome})
    return run_id, identity, results


def check_simulation(run_id=None, hours=2, regions='all', model_uri=None, session=None):
    """
    Fetches recent events, keeps those made by the simulator and reports which were alerted.
    CloudTrail delivers events to LookupEvents ~5-15 minutes after the call; re-run if events are missing.
    """
    end_time = datetime.now(timezone.utc)
    start_time = (pd.Timestamp(end_time) - pd.Timedelta(hours=hours)).floor('h').to_pydatetime()
    events = drop_own_activity(fetch_events(start_time, end_time, regions, session=session), session)
    marker = f"{SIM_MARKER}/{run_id}" if run_id else SIM_MARKER
    is_sim = events['userAgent'].fillna('').str.contains(marker, regex=False)
    if not is_sim.any():
        return None

    model = load_model(model_uri) if model_uri else load_model()
    # Score the whole window so per-hour burst features see all of the identity's activity
    scored = score_events(events, model, sort=False)
    sim = scored[is_sim.to_numpy()]
    background = scored[~is_sim.to_numpy()]

    calls = (sim.assign(alerted=sim['severity'].notna(), by_model=sim['anomaly_prediction'] == -1)
                .groupby('eventName')
                .agg(events=('eventID', 'size'), alerted=('alerted', 'sum'), by_model=('by_model', 'sum'),
                     top_severity=('severity', highest_severity),
                     rules=('rules', lambda r: sorted({x for rs in r for x in rs})))
                .reset_index())
    return {
        'run_id': run_id,
        'simulated_events_found': len(sim),
        'simulated_events_alerted': int(sim['severity'].notna().sum()),
        'simulated_events_flagged_by_model': int((sim['anomaly_prediction'] == -1).sum()),
        'background_events': len(background),
        'background_events_alerted': int(background['severity'].notna().sum()),
        'by_call': calls,
        'incidents': group_incidents(sim),
    }
