"""Synthetic CloudTrail activity for tests: a small account with distinct, habitual identities."""
import numpy as np
import pandas as pd

ACCOUNT = '111122223333'

IDENTITIES = [
    # actor arn, type, userName, IPs, user agent, (api, source) list, regions, working hours, events/day
    (f'arn:aws:iam::{ACCOUNT}:user/alice', 'IAMUser', 'alice', ['198.18.0.10'],
     'console.amazonaws.com', [('DescribeInstances', 'ec2.amazonaws.com'), ('ListBuckets', 's3.amazonaws.com'),
                               ('GetBucketPolicy', 's3.amazonaws.com'), ('DescribeAlarms', 'monitoring.amazonaws.com')],
     ['us-east-1'], range(8, 18), 60),
    (f'arn:aws:iam::{ACCOUNT}:user/bob', 'IAMUser', 'bob', ['198.18.0.11', '198.18.0.12'],
     'aws-cli/2.15.30 Python/3.11.8 Windows/10 exe/AMD64', [('DescribeInstances', 'ec2.amazonaws.com'),
     ('GetObject', 's3.amazonaws.com'), ('DescribeLogGroups', 'logs.amazonaws.com'), ('GetCallerIdentity', 'sts.amazonaws.com'),
     ('AssumeRole', 'sts.amazonaws.com')],
     ['us-east-1', 'eu-west-1'], range(9, 19), 40),
    (f'arn:aws:sts::{ACCOUNT}:assumed-role/ci-deployer/github', 'AssumedRole', None, ['198.18.5.1', '198.18.5.2'],
     'Boto3/1.34.0 md/Botocore#1.34.0 Python/3.12', [('CreateRole', 'iam.amazonaws.com'), ('AttachRolePolicy', 'iam.amazonaws.com'),
     ('UpdateFunctionCode20150331v2', 'lambda.amazonaws.com'), ('DescribeStacks', 'cloudformation.amazonaws.com')],
     ['us-east-1'], range(0, 24), 30),
    (f'arn:aws:sts::{ACCOUNT}:assumed-role/app-lambda/worker', 'AssumedRole', None, ['198.18.9.9'],
     'Boto3/1.34.0 md/Botocore#1.34.0 Python/3.12 exec-env/AWS_Lambda_python3.12',
     [('GetSecretValue', 'secretsmanager.amazonaws.com'), ('PutItem', 'dynamodb.amazonaws.com'),
      ('Decrypt', 'kms.amazonaws.com')], ['us-east-1'], range(0, 24), 120),
    (None, 'AWSService', None, ['cloudtrail.amazonaws.com'], 'cloudtrail.amazonaws.com',
     [('AssumeRole', 'sts.amazonaws.com')], ['us-east-1'], range(0, 24), 50),
]


def make_account_activity(days=14, start='2026-01-05', seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for day in pd.date_range(start, periods=days, freq='D', tz='UTC'):
        for arn, id_type, user, ips, agent, apis, regions, hours, per_day in IDENTITIES:
            if id_type == 'IAMUser' and day.dayofweek >= 5:
                continue
            for _ in range(int(rng.poisson(per_day))):
                api, source = apis[rng.integers(len(apis))]
                when = day + pd.Timedelta(hours=int(rng.choice(list(hours))), minutes=int(rng.integers(60)))
                rows.append({
                    'eventID': f'evt-{len(rows)}', 'eventTime': when.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'eventName': api, 'eventSource': source, 'eventType': 'AwsApiCall',
                    'awsRegion': regions[rng.integers(len(regions))], 'sourceIPAddress': ips[rng.integers(len(ips))],
                    'userAgent': agent, 'readOnly': 'true' if api.startswith(('Describe', 'Get', 'List')) else 'false',
                    'errorCode': 'ThrottlingException' if rng.random() < 0.01 else None, 'errorMessage': None,
                    'userIdentity.arn': arn, 'userIdentity.type': id_type, 'userIdentity.userName': user,
                    'userIdentity.accessKeyId': 'ASIAEXAMPLE' if arn else None,
                    'userIdentity.invokedBy': 'cloudtrail.amazonaws.com' if id_type == 'AWSService' else None,
                    'userIdentity.sessionContext.attributes.mfaAuthenticated': 'true' if user == 'alice' else 'false',
                    'requestParameters.roleArn': f'arn:aws:iam::{ACCOUNT}:role/readonly' if api == 'AssumeRole' else None,
                })
    df = pd.DataFrame(rows)
    df['errorMessage'] = df['errorCode'].where(df['errorCode'].isna(), 'Rate exceeded')
    return df.sort_values('eventTime').reset_index(drop=True)
