import json
import pandas as pd
import pytest
from scripts.detections import alert_severity, highest_severity
from scripts.evaluate import evaluate_model, inject_attacks, time_split
from scripts.inference import group_incidents, run_scoring, score_events
from scripts.train_models import build_pipeline
from tests.synthetic import ACCOUNT, make_account_activity

ALICE = f'arn:aws:iam::{ACCOUNT}:user/alice'
CI = f'arn:aws:sts::{ACCOUNT}:assumed-role/ci-deployer/github'


@pytest.fixture(scope='module')
def activity():
    return make_account_activity(days=14)


@pytest.fixture(scope='module')
def model(activity):
    return build_pipeline().fit(activity)


def event_of(activity, arn, **overrides):
    event = activity[activity['userIdentity.arn'] == arn].iloc[[0]].copy()
    event['eventID'] = 'test-event'
    for k, v in overrides.items():
        event[k] = v
    return event


def test_severity_combines_model_and_rules():
    assert alert_severity(False, []) is None
    assert alert_severity(True, []) == 'medium'
    assert alert_severity(False, ['high']) == 'high'
    assert alert_severity(True, ['high', 'medium']) == 'critical'
    assert highest_severity([None, 'low', 'high', 'medium']) == 'high'


def test_logging_tampering_is_critical_with_reasons(activity, model):
    scored = score_events(event_of(activity, ALICE, eventName='StopLogging', eventSource='cloudtrail.amazonaws.com'), model)
    row = scored.iloc[0]

    assert row['severity'] == 'critical'
    assert 'cloud_logging_tampering' in row['rules']
    assert 'T1562.008' in row['mitre']
    assert any('API StopLogging never seen in this account' in r for r in row['reasons'])


def test_routine_iam_changes_by_ci_need_model_corroboration(activity, model):
    # ci-deployer attaches role policies every day: quiet from its usual IP and client...
    routine = score_events(event_of(activity, CI, eventName='AttachRolePolicy'), model).iloc[0]
    # ...but alerted from an attacker's IP and tooling at the same time of day
    stolen = score_events(event_of(activity, CI, eventName='AttachRolePolicy', sourceIPAddress='203.0.113.66',
                                   userAgent='aws-cli/1.18.69 Python/3.8.5 Linux/5.7.0-kali1-amd64'), model).iloc[0]

    assert routine['rules'] == ['iam_persistence_or_privilege_change']
    assert routine['severity'] is None
    assert stolen['severity'] in ('high', 'critical')
    assert any('routine for this identity' in r for r in stolen['reasons'])


def test_root_console_login_without_mfa(activity, model):
    login = event_of(activity, ALICE, **{
        'eventName': 'ConsoleLogin', 'eventSource': 'signin.amazonaws.com', 'userIdentity.type': 'Root',
        'userIdentity.arn': f'arn:aws:iam::{ACCOUNT}:root', 'additionalEventData.MFAUsed': 'No',
        'responseElements.ConsoleLogin': 'Success'})

    row = score_events(login, model).iloc[0]

    assert {'root_account_activity', 'console_login_without_mfa'} <= set(row['rules'])
    assert row['severity'] in ('high', 'critical')


def test_normal_activity_is_mostly_quiet(activity, model):
    later = make_account_activity(days=3, start='2026-01-19', seed=1)

    scored = score_events(later, model)

    assert scored['severity'].notna().mean() < 0.05


def test_recon_sweep_becomes_one_incident(activity, model):
    sweep = pd.concat([event_of(activity, ALICE, eventName=f'List{i}', eventTime=f'2026-01-20T02:{i:02d}:00Z',
                                sourceIPAddress='203.0.113.66') for i in range(20)], ignore_index=True)
    sweep['eventID'] = [f'sweep-{i}' for i in range(20)]

    incidents = group_incidents(score_events(sweep, model))

    assert len(incidents) == 1
    assert incidents[0]['event_count'] == 20
    assert 'rule:recon_burst' in incidents[0]['detected_by']
    assert any('source IP 203.0.113.66 never seen' in r for r in incidents[0]['reasons'])


def test_evaluation_catches_injected_scenarios(activity):
    train, holdout = time_split(activity)
    report = evaluate_model(build_pipeline().fit(train), holdout, history=train)

    assert set(report['scenarios']) == {'stolen_credentials_recon', 'privilege_escalation', 'defense_evasion',
                                        'unusual_region_compute', 'access_denied_spray', 'root_console_login',
                                        'new_ip_normal_activity', 'recon_from_usual_ip',
                                        'lateral_movement_role_chain', 'shared_attacker_ip', 'role_hop_known_role'}
    assert report['scenario_detection_rate'] >= 0.8
    assert report['victim_identity'].endswith(('user/alice', 'user/bob'))  # an established human identity
    assert report['scenarios']['recon_from_usual_ip']['model_event_recall'] >= 0.5
    assert report['normal_alert_rate'] < 0.05


def test_victim_must_have_history_in_training(activity):
    from scripts.evaluate import pick_victim
    newcomer = activity.iloc[:5].copy()
    newcomer['userIdentity.arn'] = f'arn:aws:iam::{ACCOUNT}:user/brand-new'
    newcomer['userIdentity.type'] = 'IAMUser'

    _, victim, realistic = pick_victim(pd.concat([activity, newcomer], ignore_index=True))

    assert realistic

    assert victim['actor'].iat[0] != f'arn:aws:iam::{ACCOUNT}:user/brand-new'


def test_service_linked_role_victim_marks_baseline_scenarios_untestable(activity):
    service_role = activity.copy()
    service_role['userIdentity.arn'] = f'arn:aws:sts::{ACCOUNT}:assumed-role/AWSServiceRoleForResourceExplorer/rex'
    service_role['userIdentity.type'] = 'AssumedRole'
    train, holdout = time_split(service_role)

    report = evaluate_model(build_pipeline().fit(train), holdout, history=train)

    assert not report['victim_is_realistic']
    assert report['victim_identity'].startswith('dormant-admin')  # never impersonates an AWS service role
    assert report['untestable_scenarios'] == ['lateral_movement_role_chain', 'new_ip_normal_activity',
                                              'recon_from_usual_ip', 'role_hop_known_role', 'shared_attacker_ip',
                                              'unusual_region_compute']
    assert report['scenario_detection_rate'] == 1.0  # the blatant scenarios are still measured


def test_injection_does_not_modify_holdout(activity):
    _, holdout = time_split(activity)
    before = holdout.copy()

    injected = inject_attacks(holdout)

    pd.testing.assert_frame_equal(holdout, before)
    assert injected['scenario'].notna().sum() > 0


def test_run_scoring_with_state_file_never_realerts(tmp_path, monkeypatch, activity, model):
    window = make_account_activity(days=1, start='2026-01-20', seed=2)
    window.loc[0, ['eventName', 'eventSource']] = ['StopLogging', 'cloudtrail.amazonaws.com']
    monkeypatch.setattr('scripts.inference.fetch_events', lambda *a, **k: window)
    monkeypatch.setattr('scripts.inference.load_model', lambda uri: model)
    state, out = tmp_path / 'state.json', tmp_path / 'incidents.jsonl'

    first, _ = run_scoring(hours=24 * 365, output=str(out), state_file=str(state))
    second, summary = run_scoring(hours=24 * 365, output=str(out), state_file=str(state))

    assert any(i['severity'] == 'critical' for i in first)
    assert second == [] and summary['events_scored'] == 0
    assert len(out.read_text().splitlines()) == len(first)
    assert json.loads(out.read_text().splitlines()[0])['incident_id']


def test_detector_threshold_matches_behaviour_contamination(activity, model):
    from scripts.features_engineer import CloudTrailFeatures
    training_features = CloudTrailFeatures().fit_transform(activity)
    detector = model.named_steps['detector']

    flagged = (detector.behaviour_predict(training_features) == -1).mean()

    assert flagged == pytest.approx(detector.behaviour_contamination, abs=0.003)


def test_burst_rules_mark_only_the_events_in_the_burst(activity, model):
    sweep = pd.concat([event_of(activity, ALICE, eventName=f'List{i}', eventTime=f'2026-01-20T02:{i:02d}:00Z',
                                errorCode='AccessDenied') for i in range(16)], ignore_index=True)
    normal = event_of(activity, ALICE, eventName='PutMetricData', eventTime='2026-01-20T02:30:00Z')
    batch = pd.concat([sweep, normal], ignore_index=True)
    batch['eventID'] = [f'b-{i}' for i in range(len(batch))]

    scored = score_events(batch, model, sort=False)

    assert all({'recon_burst', 'error_burst'} <= set(r) for r in scored['rules'].iloc[:16])
    assert scored['rules'].iat[16] == []  # same identity and hour, but not part of the sweep


def test_tuning_keeps_defaults_as_baseline_and_respects_budget(activity):
    from scripts.train_models import PARAMS
    from scripts.tuning import tune
    train, holdout = time_split(activity)

    best, trials = tune(train, holdout, PARAMS, build_pipeline, n_trials=3, alert_budget=0.05, seeds=(1,))

    assert trials[0]['params'] == {k: PARAMS[k] for k in trials[0]['params']}  # defaults are trial 0
    chosen = next(t for t in trials if all(best[k] == v for k, v in t['params'].items()))
    assert chosen['within_budget']
    assert chosen['scenario_detection_rate'] >= trials[0]['scenario_detection_rate'] or not trials[0]['within_budget']


def test_new_identity_noise_is_reported_separately(activity):
    train, holdout = time_split(activity)
    newcomer = holdout.iloc[:30].copy()
    newcomer['userIdentity.arn'] = f'arn:aws:iam::{ACCOUNT}:user/new-starter'
    newcomer['sourceIPAddress'] = '198.18.7.7'
    holdout = pd.concat([holdout, newcomer], ignore_index=True)

    report = evaluate_model(build_pipeline().fit(train), holdout, history=train)

    assert report['new_identity_events'] == 30
    assert report['new_identity_alert_rate'] > report['established_alert_rate']
    assert report['established_alert_rate'] < 0.05


def test_graph_scenarios_are_caught_with_reasons(activity):
    train, holdout = time_split(activity)
    model = build_pipeline().fit(train)
    data = inject_attacks(holdout, history=train)
    labels = data.pop('scenario')
    scored = score_events(data, model, sort=False)

    chain = scored[(labels == 'lateral_movement_role_chain').to_numpy()]
    shared = scored[(labels == 'shared_attacker_ip').to_numpy()]

    assert chain['severity'].notna().all()
    assert any('new_role_assumption_path' in r for r in chain['rules'])
    assert any('into another account' in x for rs in chain['reasons'] for x in rs)
    assert shared['severity'].notna().all()
    assert any('3 different identities used IP 203.0.113.150' in x for rs in shared['reasons'] for x in rs)
