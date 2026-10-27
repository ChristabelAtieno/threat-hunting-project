import numpy as np
import pandas as pd
import pytest
from scripts.features_engineer import CloudTrailFeatures, prepare_events
from tests.synthetic import ACCOUNT, make_account_activity

ALICE = f'arn:aws:iam::{ACCOUNT}:user/alice'


@pytest.fixture(scope='module')
def activity():
    return make_account_activity(days=7)


@pytest.fixture(scope='module')
def features(activity):
    return CloudTrailFeatures().fit(activity)


def alice_event(activity, **overrides):
    event = activity[activity['userIdentity.arn'] == ALICE].iloc[[0]].copy()
    for k, v in overrides.items():
        event[k] = v
    return event


def test_transform_tolerates_missing_columns(features):
    sparse_batch = pd.DataFrame({'eventName': ['ListBuckets'], 'eventTime': ['2026-01-05T03:00:00Z']})

    out = features.transform(sparse_batch)

    assert list(out.columns) == list(features.feature_names_out_)
    assert out.notna().all().all()


def test_mfa_false_is_not_counted_as_mfa(features, activity):
    batch = activity.iloc[:3].copy()
    batch['userIdentity.sessionContext.attributes.mfaAuthenticated'] = ['true', 'false', None]

    assert features.transform(batch)['has_mfa'].tolist() == [1, 0, 0]


def test_frequencies_come_from_training_not_the_batch(features, activity):
    batch = pd.concat([alice_event(activity), alice_event(activity, sourceIPAddress='203.0.113.9')])

    freq = features.transform(batch)['sourceIPAddress_frequency']

    assert freq.iloc[0] == pytest.approx(np.log1p((activity['sourceIPAddress'] == '198.18.0.10').sum()))
    assert freq.iloc[1] == 0  # never seen in training -> maximally rare


def test_ip_known_to_account_but_new_for_identity(features, activity):
    # bob's IP is common in the account, but alice has never used it
    out = features.transform(alice_event(activity, sourceIPAddress='198.18.0.11')).iloc[0]

    assert out['sourceIPAddress_frequency'] > 0
    assert out['actor_sourceIPAddress_frequency'] == 0


def test_user_agent_family_ignores_versions():
    events = pd.DataFrame({'userAgent': ['aws-cli/2.15.30 Python/3.11', 'aws-cli/2.22.1 Python/3.12',
                                         '[S3Console/0.4, aws-internal/3]', 'Boto3/1.34.0 md/Botocore', None]})

    assert prepare_events(events)['userAgentFamily'].tolist() == ['aws-cli', 'aws-cli', 's3console', 'boto3', 'Unknown']


def test_burst_features_count_per_identity_per_hour(features, activity):
    burst = pd.concat([alice_event(activity, eventName=f'List{i}', eventTime='2026-01-20T02:05:00Z') for i in range(12)])
    burst = pd.concat([burst, alice_event(activity, eventTime='2026-01-20T03:30:00Z')])

    out = features.transform(burst)

    assert np.expm1(out['actor_distinct_apis_in_hour'].iloc[0]).round() == 12
    assert np.expm1(out['actor_events_in_hour'].iloc[-1]).round() == 1  # next hour is a separate bucket


def test_identity_without_arn_falls_back_to_type():
    events = pd.DataFrame({'userIdentity.type': ['AWSService'], 'userIdentity.arn': [None]})

    assert prepare_events(events)['actor'].iat[0] == 'type:AWSService'


def test_new_ip_is_surprising_for_a_single_ip_identity_only(features, activity):
    lambda_role = f'arn:aws:sts::{ACCOUNT}:assumed-role/app-lambda/worker'
    usual = features.transform(alice_event(activity))['behaviour_surprisal'].iat[0]
    alice_new_ip = features.transform(alice_event(activity, sourceIPAddress='203.0.113.9'))
    lambda_event = activity[activity['userIdentity.arn'] == lambda_role].iloc[[0]].copy()
    lambda_new_ip = features.transform(lambda_event.assign(sourceIPAddress='203.0.113.9'))

    assert alice_new_ip['behaviour_surprisal'].iat[0] > usual + 5
    assert alice_new_ip['surprisal_sourceIPAddress'].iat[0] > 5
    assert lambda_new_ip['surprisal_sourceIPAddress'].iat[0] > 5  # one IP too; contrast below

    # an identity that rotates through many IPs is barely surprised by another one
    rotating = pd.concat([alice_event(activity, sourceIPAddress=f'10.1.{i // 250}.{i % 250}') for i in range(400)],
                         ignore_index=True)
    rotating_features = CloudTrailFeatures().fit(pd.concat([activity, rotating], ignore_index=True))
    assert rotating_features.transform(alice_event(activity, sourceIPAddress='203.0.113.9'))[
        'surprisal_sourceIPAddress'].iat[0] < 2


def test_training_encoding_is_leave_one_out(activity):
    one_off = alice_event(activity, sourceIPAddress='203.0.113.200')
    data = pd.concat([activity, one_off], ignore_index=True)

    training_features = CloudTrailFeatures().fit_transform(data)

    # in training the single occurrence looks exactly like a never-seen IP does at inference
    assert training_features['sourceIPAddress_frequency'].iat[-1] == 0
    assert training_features['actor_sourceIPAddress_frequency'].iat[-1] == 0


def test_assumed_role_sessions_share_one_identity():
    acct = '111122223333'
    events = pd.DataFrame({
        'userIdentity.type': ['AssumedRole'] * 5,
        'userIdentity.arn': [f'arn:aws:sts::{acct}:assumed-role/ci-deployer/botocore-session-1',
                             f'arn:aws:sts::{acct}:assumed-role/ci-deployer/botocore-session-2',
                             f'arn:aws:sts::{acct}:assumed-role/AWSReservedSSO_Admin_ab12/alice@example.com',
                             f'arn:aws:sts::{acct}:assumed-role/AWSReservedSSO_Admin_ab12/bob@example.com',
                             f'arn:aws:sts::{acct}:assumed-role/deploy/i-0abc'],
        'userIdentity.sessionContext.sourceIdentity': [None, None, None, None, 'carol']})

    actors = prepare_events(events)['actor'].tolist()

    assert actors[0] == actors[1] == f'arn:aws:sts::{acct}:assumed-role/ci-deployer'  # random sessions merge
    assert actors[2] != actors[3]  # SSO: each person keeps their own baseline
    assert actors[4] == f'arn:aws:sts::{acct}:assumed-role/deploy/carol'  # the human behind the session


def test_graph_features(features, activity):
    acct = ACCOUNT
    alice = alice_event(activity)
    first_assume = alice_event(activity, eventName='AssumeRole', eventSource='sts.amazonaws.com',
                               **{'requestParameters.roleArn': f'arn:aws:iam::{acct}:role/prod-admin'})
    chained = alice_event(activity, eventName='AssumeRole', eventSource='sts.amazonaws.com', **{
        'userIdentity.type': 'AssumedRole', 'userIdentity.arn': f'arn:aws:sts::{acct}:assumed-role/prod-admin/x',
        'requestParameters.roleArn': 'arn:aws:iam::999988887777:role/OrganizationAccountAccessRole'})

    out = features.transform(pd.concat([alice, first_assume, chained], ignore_index=True))

    assert out['assume_role_new_edge'].tolist() == [0, 1, 1]
    assert out['role_chaining'].tolist() == [0, 0, 1]
    assert out['cross_account_assume'].tolist() == [0, 0, 1]
    # alice's IP is only ever used by alice
    assert np.expm1(out['ip_known_identities'].iat[0]).round() == 1


def test_peer_communities_group_similar_identities(features):
    # alice and bob both work with EC2 and S3; the CI role and the Lambda use other services
    alice, bob = f'arn:aws:iam::{ACCOUNT}:user/alice', f'arn:aws:iam::{ACCOUNT}:user/bob'
    lambda_role = f'arn:aws:sts::{ACCOUNT}:assumed-role/app-lambda'

    assert features.communities_[alice] == features.communities_[bob]
    assert features.communities_[alice] != features.communities_[lambda_role]


def test_peer_service_share_reflects_peers_not_self(features, activity):
    # bob uses CloudWatch Logs; alice (his peer) does not. For alice, logs is used by her peers.
    logs_for_alice = alice_event(activity, eventName='DescribeLogGroups', eventSource='logs.amazonaws.com')
    dynamo_for_alice = alice_event(activity, eventName='PutItem', eventSource='dynamodb.amazonaws.com')

    out = features.transform(pd.concat([logs_for_alice, dynamo_for_alice], ignore_index=True))

    assert out['peer_service_share'].iat[0] > 0   # a peer (bob) uses it
    assert out['peer_service_share'].iat[1] == 0  # only the Lambda, outside her community, uses it
