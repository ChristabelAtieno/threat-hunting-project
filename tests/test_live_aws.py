"""
Live tests against real AWS. Skipped by default; run with:  pytest -m live
Needs AWS credentials (cloudtrail:LookupEvents, ec2:DescribeRegions) and, for scoring,
a champion model reachable via MLFLOW_TRACKING_URI (or MODEL_URI).
"""
import os
from datetime import datetime, timedelta, timezone
import boto3
import pytest
from scripts.features_engineer import RAW_COLUMNS
from scripts.load_data import fetch_events, resolve_regions

pytestmark = pytest.mark.live

if boto3.Session().get_credentials() is None:
    pytest.skip("no AWS credentials", allow_module_level=True)

REGION = os.getenv('AWS_REGION', 'us-east-1')


@pytest.fixture(scope='module')
def recent_events():
    end = datetime.now(timezone.utc)
    return fetch_events(end - timedelta(hours=24), end, regions=REGION, max_events=500)


def test_all_regions_resolve():
    # works with or without ec2:DescribeRegions (falls back to botocore's list)
    regions = resolve_regions('all')
    assert 'us-east-1' in regions and len(regions) > 10


def test_fetch_returns_raw_schema(recent_events):
    assert list(recent_events.columns) == RAW_COLUMNS
    if recent_events.empty:
        pytest.skip(f"no CloudTrail events in {REGION} in the last 24h")
    assert recent_events['eventTime'].notna().all()


def test_champion_model_scores_live_events(recent_events):
    if recent_events.empty:
        pytest.skip("no events to score")
    from scripts.inference import load_model, score_events
    try:
        model = load_model()
    except Exception as e:
        pytest.skip(f"no champion model available: {e}")

    scored = score_events(recent_events, model)

    assert len(scored) == len(recent_events)
    assert scored['anomaly_score'].notna().all()
    # A trained baseline should flag a small minority of real activity
    assert (scored['anomaly_prediction'] == -1).mean() < 0.2
