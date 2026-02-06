# Real-Time Anomaly Detection with AWS CloudTrail

## Overview
This guide explains how to set up end-to-end anomaly detection using your trained Isolation Forest model to stream and classify AWS CloudTrail logs in real-time using `boto3`.

## Architecture Flow

```
CloudTrail Events → boto3 lookup_events() → Preprocess → Scale → MLflow Model → Classification
                                           (Real-time)           (Decision: -1=Anomaly, 1=Normal)
```

## Setup Steps

### 1. Train & Log Model (Existing Pipeline)
```bash
python main.py
```
This trains the IsolationForest and registers it as `IsolationForestModel` in MLflow.

### 2. Configure AWS Credentials
```bash
# Set up AWS credentials (choose one method)
aws configure  # Interactive setup
# OR
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt  # Now includes boto3
```

### 4. Run Real-Time Stream (NEW)
```python
from scripts.inference import AnomalyDetectionStream

# Initialize with your trained model
detector = AnomalyDetectionStream(
    model_uri='models:/IsolationForestModel/1'  # MLflow registered model
)

# Start streaming CloudTrail logs
anomalies = detector.stream_and_classify(
    trail_name='your-cloudtrail-name',  # Your CloudTrail name
    interval=30,  # Check every 30 seconds
    duration=300  # Run for 5 minutes (None = infinite)
)
```

## Key Features

### `AnomalyDetectionStream` Class

#### Methods:

**`get_cloudtrail_logs(trail_name, max_results)`**
- Queries AWS CloudTrail using boto3
- Returns latest N events
- Handles JSON parsing automatically

**`preprocess_event(event)`**
- Mirrors your training pipeline feature engineering
- Extracts: time features (hour, day_of_week, weekend, night)
- Extracts: error flags, MFA presence
- Returns pandas DataFrame ready for model input

**`classify_event(event)`**
- Single-event inference using MLflow model
- Returns: prediction (-1/1), anomaly flag, score, metadata
- Example output:
```python
{
    'timestamp': '2026-02-06T15:30:45.123456',
    'is_anomaly': True,
    'prediction': -1,
    'anomaly_score': -0.95,
    'event_name': 'DeleteBucket',
    'username': 'suspicious_user',
    'source_ip': '192.168.1.1',
    'aws_region': 'us-east-1',
    'event_id': 'abc123xyz'
}
```

**`stream_and_classify(trail_name, interval, duration)`**
- Continuous streaming with configurable polling
- Prints real-time alerts for anomalies
- Saves results to CSV

## Example Workflow

```python
import pandas as pd
from scripts.inference import AnomalyDetectionStream

# 1. Initialize detector
detector = AnomalyDetectionStream(
    model_uri='models:/IsolationForestModel/1'
)

# 2. Stream for 1 hour, checking every 60 seconds
anomalies = detector.stream_and_classify(
    trail_name='my-cloudtrail',
    interval=60,
    duration=3600
)

# 3. Analyze detected anomalies
df = pd.DataFrame(anomalies)
print(f"Total anomalies: {len(df)}")
print(df[['event_name', 'username', 'source_ip', 'anomaly_score']])

# 4. Save for investigation
df.to_csv('anomalies_2026-02-06.csv', index=False)
```

## AWS IAM Permissions Required

Your AWS user/role needs these CloudTrail permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "cloudtrail:LookupEvents",
                "cloudtrail:DescribeTrails"
            ],
            "Resource": "*"
        }
    ]
}
```

## Feature Parity with Training

The `preprocess_event()` method replicates your training pipeline exactly:
1. ✓ Time feature extraction (hour, day_of_week, weekend, night flags)
2. ✓ Error code handling (hasError flag)
3. ✓ MFA presence feature
4. ✓ Column cleanup
5. ✓ StandardScaler normalization

**Important**: Ensure features match training data schema for accurate predictions.

## Deployment Options

### Option 1: Scheduled Lambda (AWS)
```python
# lambda_handler.py
def lambda_handler(event, context):
    from scripts.inference import AnomalyDetectionStream
    
    detector = AnomalyDetectionStream('models:/IsolationForestModel/1')
    anomalies = detector.stream_and_classify('my-trail', duration=300)
    
    # Send to SNS/SQS for alerts
    return {'statusCode': 200, 'anomalies': len(anomalies)}
```
Schedule with EventBridge: Run every 5 minutes

### Option 2: EC2 Daemon
```bash
# Run continuously on EC2 instance
nohup python -c "
from scripts.inference import AnomalyDetectionStream
detector = AnomalyDetectionStream('models:/IsolationForestModel/1')
detector.stream_and_classify('my-trail', interval=60)
" > anomaly_detection.log 2>&1 &
```

### Option 3: SageMaker Real-Time Endpoint
- Export MLflow model to SageMaker format
- Deploy as real-time endpoint
- Call with streaming CloudTrail events

## Troubleshooting

**Error: "No credentials found"**
- Ensure AWS credentials are configured: `aws configure`
- Or set env vars: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

**Error: "Trail not found"**
- Verify trail name: `aws cloudtrail describe-trails --region us-east-1`
- Ensure CloudTrail is enabled: `aws cloudtrail start-logging`

**Low anomaly detection rate**
- Check model contamination threshold (default 0.01 = 1%)
- Verify feature engineering matches training exactly
- Consider tuning n_estimators or contamination in training

**Model takes too long to load**
- MLflow model loading is I/O bound; cache model in memory
- Use model caching if running multiple inferences:
```python
detector = AnomalyDetectionStream(...)  # Load once
for event in event_stream:
    result = detector.classify_event(event)  # Reuse detector
```

## Next Steps

1. **Connect to alerting**: Route `is_anomaly=True` results to:
   - SNS topics for email/SMS
   - CloudWatch Alarms
   - SecurityHub for SIEM integration

2. **Add alerting/logging**: 
   - Log anomalies to CloudWatch Logs
   - Send to S3 for audit trail
   - Post to Slack/Teams via webhooks

3. **Monitor model drift**: 
   - Track anomaly_score distribution over time
   - Periodically retrain on new logs
   - Use MLflow to version and compare models

4. **Optimize performance**:
   - Batch process events (10-50 per call)
   - Add caching for repeated preprocessing
   - Consider async/threaded polling
