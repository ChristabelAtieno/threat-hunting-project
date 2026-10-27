import json
import pytest
from datetime import datetime, timezone
from botocore.exceptions import ClientError, SSLError
from scripts.load_data import fetch_events


class FakeSession:
    """boto3.Session stand-in: one working region, one disabled, one unreachable, one returning 503."""
    region_name = 'us-east-1'

    def client(self, service, region_name=None, config=None):
        return FakeCloudTrail(region_name)


class FakeCloudTrail:
    def __init__(self, region):
        self.region = region

    def get_paginator(self, name):
        return self

    def paginate(self, **kwargs):
        if self.region == 'me-central-1':
            raise SSLError(endpoint_url='https://cloudtrail.me-central-1.amazonaws.com/', error='cert')
        if self.region == 'il-central-1':
            raise ClientError({'Error': {'Code': '503', 'Message': 'Service Unavailable'},
                               'ResponseMetadata': {'HTTPStatusCode': 503}}, 'LookupEvents')
        if self.region == 'eu-west-1':
            raise ClientError({'Error': {'Code': 'AccessDeniedException'},
                               'ResponseMetadata': {'HTTPStatusCode': 400}}, 'LookupEvents')
        if self.region == 'af-south-1':
            raise ClientError({'Error': {'Code': 'UnrecognizedClientException'}}, 'LookupEvents')
        event = {'eventID': 'e1', 'eventTime': '2026-01-05T10:00:00Z', 'eventName': 'ListBuckets', 'awsRegion': self.region}
        return [{'Events': [{'CloudTrailEvent': json.dumps(event)}]}]


def test_unavailable_and_disabled_regions_do_not_stop_the_run():
    now = datetime.now(timezone.utc)

    events = fetch_events(now, now, regions='us-east-1,af-south-1,il-central-1,me-central-1', session=FakeSession())

    assert events['eventID'].tolist() == ['e1']
    assert events.attrs['failed_regions'] == ['il-central-1', 'me-central-1']


def test_permission_errors_still_fail_loudly():
    now = datetime.now(timezone.utc)

    with pytest.raises(ClientError):
        fetch_events(now, now, regions='us-east-1,eu-west-1', session=FakeSession())


def test_tool_activity_is_excluded():
    import pandas as pd
    from scripts.load_data import drop_own_activity, drop_tool_activity

    class Sts:
        def client(self, *a, **k):
            return self

        def get_caller_identity(self):
            return {'Arn': 'arn:aws:iam::1:user/scorer'}

    # same nullable string dtype as real CloudTrail frames (records_to_frame), including a missing ARN
    events = pd.DataFrame({
        'userIdentity.arn': ['arn:aws:iam::1:user/scorer', 'arn:aws:iam::1:user/attacker',
                             'arn:aws:iam::1:user/scorer', 'arn:aws:iam::1:user/alice', None],
        'userAgent': ['Boto3/1.43 threat-hunt-scorer/1', 'Boto3/1.43 threat-hunt-scorer/1',
                      'Boto3/1.43 threat-hunt-sim/abc', 'console.amazonaws.com', 'Boto3/1.43 threat-hunt-scorer/1']},
        dtype='string')

    scored = drop_own_activity(events, Sts())
    trained = drop_tool_activity(events)

    # a different identity copying the user agent is still scored
    assert scored['userIdentity.arn'].tolist()[0] == 'arn:aws:iam::1:user/attacker'
    assert len(scored) == 4  # the tagged event with no ARN is not provably ours, so it is kept
    assert trained['userIdentity.arn'].tolist() == ['arn:aws:iam::1:user/alice']
