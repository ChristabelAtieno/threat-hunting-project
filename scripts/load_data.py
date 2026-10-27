"""
CloudTrail data sources: exported log files on disk, a trail's S3 bucket, or the LookupEvents API.
All of them return a DataFrame aligned to features_engineer.RAW_COLUMNS.
"""
import gzip
import json
import logging
import os
from datetime import timedelta
from urllib.parse import urlparse
import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError, SSLError
from scripts.features_engineer import RAW_COLUMNS

logger = logging.getLogger(__name__)

# LookupEvents allows 2 requests/second per region; adaptive retries back off on throttling.
# Short timeouts so an unreachable regional endpoint costs seconds, not minutes.
# Every call this tool makes carries TOOL_AGENT in its user agent, so its own CloudTrail reads can be
# recognised and kept out of scoring and training
TOOL_AGENT = 'threat-hunt-scorer'
SIMULATION_AGENT = 'threat-hunt-sim'
AWS_CONFIG = Config(retries={'mode': 'adaptive', 'max_attempts': 5}, connect_timeout=5, read_timeout=30,
                    user_agent_extra=f'{TOOL_AGENT}/1')
# Network failures reaching one regional endpoint (firewall, proxy, TLS interception)
UNREACHABLE_ERRORS = (ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError, SSLError)
# What an opt-in region the account has not enabled returns
DISABLED_REGION_ERRORS = {'UnrecognizedClientException', 'InvalidClientTokenId', 'AuthFailure'}


def records_to_frame(records):
    """Flattens CloudTrail records and keeps RAW_COLUMNS as strings (consistent schema across sources)."""
    if not records:
        return pd.DataFrame(columns=RAW_COLUMNS, dtype='string')
    return pd.json_normalize(records).reindex(columns=RAW_COLUMNS).astype('string')


def load_data(data_path, output_path):

    """
    Loads the json files from the directory
    Keeping only the relevant columns and saves the data into parquet files for faster processing in the next steps

    Parameters
    data_path:
        This is the path where the logs are stored
    output_path:
        This is the path where the parquet files would be stored after being loaded

    Returns
        The loaded data and stores the data into parquet files
    """

    output_path.mkdir(exist_ok=True)

    files = sorted(data_path.glob('*.json.gz'))
    if not files:
        raise FileNotFoundError(f"No *.json.gz CloudTrail logs found in {data_path}")

    for file in files:
        with gzip.open(file, 'rb') as f:
            df = records_to_frame(json.load(f)['Records'])
            output_filename = output_path / file.stem.replace(".json", ".parquet")
            df.to_parquet(output_filename, engine='pyarrow')

    logger.info("Created %d parquet files in %s", len(files), output_path)
    return read_processed(output_path)


def read_processed(processed_path):
    """Reads the parquet files written by load_data into a single DataFrame."""
    return pd.read_parquet(processed_path, engine='pyarrow')


def processed_is_current(processed_path):
    """True if cached parquet files exist and were written with the current RAW_COLUMNS."""
    files = sorted(processed_path.glob('*.parquet'))
    if not files:
        return False
    return list(pd.read_parquet(files[0], engine='pyarrow').columns) == RAW_COLUMNS


def load_s3(s3_uri, max_files=None, session=None):
    """
    Loads CloudTrail log files (*.json.gz) under an S3 prefix, e.g. the bucket a trail delivers to:
    s3://my-trail-bucket/AWSLogs/123456789012/CloudTrail/us-east-1/2026/10/
    """
    parsed = urlparse(s3_uri)
    s3 = (session or boto3.Session()).client('s3', config=AWS_CONFIG)
    paginator = s3.get_paginator('list_objects_v2')

    frames = []
    for page in paginator.paginate(Bucket=parsed.netloc, Prefix=parsed.path.lstrip('/')):
        for obj in page.get('Contents', []):
            if not obj['Key'].endswith('.json.gz'):
                continue
            body = s3.get_object(Bucket=parsed.netloc, Key=obj['Key'])['Body'].read()
            frames.append(records_to_frame(json.loads(gzip.decompress(body))['Records']))
            if max_files and len(frames) >= max_files:
                break
        if max_files and len(frames) >= max_files:
            break

    if not frames:
        raise FileNotFoundError(f"No *.json.gz CloudTrail logs found under {s3_uri}")
    logger.info("Loaded %d log files from %s", len(frames), s3_uri)
    return pd.concat(frames, ignore_index=True)


def resolve_regions(regions, session=None):
    """
    'all' -> every region enabled for the account (attackers like regions nobody watches);
    otherwise a comma-separated list. Defaults to AWS_REGION.
    """
    session = session or boto3.Session()
    if regions in (None, ''):
        return [session.region_name or os.getenv('AWS_REGION', 'us-east-1')]
    if regions == 'all':
        ec2 = session.client('ec2', region_name=session.region_name or 'us-east-1', config=AWS_CONFIG)
        try:
            return sorted(r['RegionName'] for r in ec2.describe_regions()['Regions'])
        except ClientError as e:
            # Without ec2:DescribeRegions, use botocore's region list; regions not enabled for
            # the account are skipped by fetch_events
            logger.warning("Cannot list enabled regions (%s); trying all known CloudTrail regions. "
                           "Grant ec2:DescribeRegions to avoid this.", e.response['Error']['Code'])
            return sorted(session.get_available_regions('cloudtrail'))
    return [r.strip() for r in regions.split(',') if r.strip()]


def fetch_events(start_time, end_time, regions=None, max_events=None, session=None):
    """
    Fetches CloudTrail management events between start_time and end_time via LookupEvents.

    LookupEvents is per region, so each region is queried. IAM, STS and console sign-in events
    are global and recorded in us-east-1. Events appear roughly 5-15 minutes after the API call.
    Credentials come from boto3's default chain (env vars, ~/.aws, or an IAM role).
    """
    session = session or boto3.Session()
    raw_records, failed = [], []
    for region in resolve_regions(regions, session):
        paginator = session.client('cloudtrail', region_name=region, config=AWS_CONFIG).get_paginator('lookup_events')
        before = len(raw_records)
        try:
            pages = paginator.paginate(StartTime=start_time, EndTime=end_time,
                                       PaginationConfig={'MaxItems': max_events, 'PageSize': 50})
            raw_records.extend(json.loads(event['CloudTrailEvent']) for page in pages for event in page['Events'])
        except ClientError as e:
            if e.response['Error']['Code'] in DISABLED_REGION_ERRORS:
                logger.info("%s: skipped (region not enabled for this account)", region)
                continue
            if not _is_transient(e):
                raise  # credentials / permissions: a configuration problem to fix, not to skip
            reason = e.response['Error'].get('Message') or e.response['Error']['Code']
        except UNREACHABLE_ERRORS as e:
            reason = f"unreachable from this network ({type(e).__name__})"
        else:
            logger.info("%s: %d events", region, len(raw_records) - before)
            continue
        del raw_records[before:]
        failed.append(region)
        logger.warning("%s: FAILED after retries (%s); activity there is NOT covered by this run", region, reason)

    events = records_to_frame(raw_records)
    events.attrs['failed_regions'] = failed
    return events


def _is_transient(error):
    """AWS-side or rate-limit errors that persisted through retries (e.g. 503 Service Unavailable)."""
    status = error.response.get('ResponseMetadata', {}).get('HTTPStatusCode') or 0
    return status >= 500 or error.response['Error']['Code'] in {'ThrottlingException', 'RequestLimitExceeded'}


def _with_attrs(subset, events):
    subset = subset.copy()
    subset.attrs = dict(events.attrs)
    return subset


def drop_own_activity(events, session=None):
    """
    Drops this tool's own CloudTrail reads (tagged TOOL_AGENT) made by the identity running it,
    so the scorer never alerts on itself. The identity check means a stolen key that merely
    copies the user agent is still scored.
    """
    tagged = events['userAgent'].fillna('').str.contains(f'{TOOL_AGENT}/', regex=False).astype(bool)
    if not tagged.any():
        return events
    try:
        me = (session or boto3.Session()).client('sts', config=AWS_CONFIG).get_caller_identity()['Arn']
    except Exception as e:
        logger.warning("Could not resolve own identity (%s); scoring this tool's own activity too", e)
        return events
    # CloudTrail columns are nullable strings: a missing ARN compares as NA, which is not ours
    own = tagged & events['userIdentity.arn'].eq(me).fillna(False).astype(bool)
    logger.info("Ignoring %d events made by this tool (%s)", int(own.sum()), me)
    return _with_attrs(events[~own.to_numpy()], events)


def drop_tool_activity(events):
    """Removes scorer and attack-simulation traffic from training data, so neither becomes 'normal'."""
    user_agent = events['userAgent'].fillna('')
    tool = (user_agent.str.contains(f'{TOOL_AGENT}/', regex=False)
            | user_agent.str.contains(f'{SIMULATION_AGENT}/', regex=False)).astype(bool)
    if tool.any():
        logger.info("Excluding %d scorer/simulation events from training", int(tool.sum()))
    return _with_attrs(events[~tool.to_numpy()], events)


def fetch_days(days, regions=None, session=None):
    """Fetches the last `days` days of management events (LookupEvents keeps 90 days) for training."""
    end_time = pd.Timestamp.now(tz='UTC').to_pydatetime()
    return fetch_events(end_time - timedelta(days=days), end_time, regions, session=session)
