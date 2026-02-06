import boto3
import json
import pandas as pd
import numpy as np
import mlflow.pyfunc
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time

# ================================
# REAL-TIME ANOMALY DETECTION
# ================================

class AnomalyDetectionStream:
    """
    Real-time anomaly detection using MLflow model and AWS CloudTrail logs.
    Streams logs from CloudTrail and classifies them using the trained Isolation Forest model.
    """
    
    def __init__(self, model_uri: str, contamination_threshold: float = 0.01):
        """
        Initialize the anomaly detector with MLflow model.
        
        Parameters
        ----------
        model_uri : str
            MLflow model URI (e.g., 'models:/IsolationForestModel/1')
        contamination_threshold : float
            Anomaly threshold (default 0.01 = 1%)
        """
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.contamination_threshold = contamination_threshold
        self.cloudtrail = boto3.client('cloudtrail')
        self.logs = boto3.client('logs')
        self.scaler = StandardScaler()
        
    def get_cloudtrail_logs(self, trail_name: str, max_results: int = 50):
        """
        Retrieve latest CloudTrail events from AWS.
        
        Parameters
        ----------
        trail_name : str
            Name of CloudTrail to query
        max_results : int
            Maximum events to retrieve
            
        Returns
        -------
        list
            List of CloudTrail event dictionaries
        """
        try:
            response = self.cloudtrail.lookup_events(
                MaxResults=max_results,
                MaxItemsPerPage=50
            )
            events = []
            for event in response.get('Events', []):
                event_data = json.loads(event['CloudTrailEvent'])
                events.append(event_data)
            return events
        except Exception as e:
            print(f"Error retrieving CloudTrail logs: {e}")
            return []
    
    def preprocess_event(self, event: dict) -> pd.DataFrame:
        """
        Preprocess single CloudTrail event for model inference.
        Mirrors the feature engineering pipeline.
        
        Parameters
        ----------
        event : dict
            Single CloudTrail event
            
        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with engineered features
        """
        df = pd.DataFrame([event])
        
        # ---- TIME FEATURES ----
        if 'eventTime' in df.columns:
            df['eventTime'] = pd.to_datetime(df['eventTime'], errors='coerce')
            df['hour'] = df['eventTime'].dt.hour
            df['day_of_week'] = df['eventTime'].dt.dayofweek
            df['isWeekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['isNight'] = df['hour'].between(0, 6).astype(int)
        
        # ---- ERROR FEATURES ----
        if 'errorCode' in df.columns:
            df['errorCode'] = df['errorCode'].fillna('None')
            df['hasError'] = (df['errorCode'] != 'None').astype(int)
        else:
            df['hasError'] = 0
        
        # ---- PRESENCE FEATURES ----
        mfa_col = 'userIdentity.sessionContext.attributes.mfaAuthenticated'
        if mfa_col in df.columns:
            df['has_mfaAuthenticated'] = df[mfa_col].notna().astype(int)
        else:
            df['has_mfaAuthenticated'] = 0
        
        # ---- CLEANUP ----
        cols_to_drop = [col for col in ['eventTime', 'errorMessage', 'errorCode', mfa_col]
                        if col in df.columns]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        return df
    
    def classify_event(self, event: dict) -> dict:
        """
        Classify single CloudTrail event as normal or anomalous.
        
        Parameters
        ----------
        event : dict
            CloudTrail event
            
        Returns
        -------
        dict
            Classification result with prediction, anomaly score, and event details
        """
        try:
            df = self.preprocess_event(event)
            
            # Scale features
            scaled = self.scaler.fit_transform(df)
            
            # Get model prediction
            prediction = self.model.predict(scaled)[0]
            
            # Get anomaly score (decision function)
            try:
                # For pyfunc models, decision function may not be available
                # Use prediction: -1 = anomaly, 1 = normal
                is_anomaly = prediction == -1
                anomaly_score = -1.0 if is_anomaly else 1.0
            except:
                anomaly_score = float(prediction)
                is_anomaly = anomaly_score < 0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'is_anomaly': bool(is_anomaly),
                'prediction': int(prediction),
                'anomaly_score': float(anomaly_score),
                'event_name': event.get('eventName', 'Unknown'),
                'username': event.get('userIdentity', {}).get('userName', 'Unknown'),
                'source_ip': event.get('sourceIPAddress', 'Unknown'),
                'aws_region': event.get('awsRegion', 'Unknown'),
                'event_id': event.get('eventID', 'Unknown')
            }
        except Exception as e:
            print(f"Error classifying event: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'is_anomaly': None
            }
    
    def stream_and_classify(self, trail_name: str, interval: int = 60, duration: int = None):
        """
        Continuously stream and classify CloudTrail logs.
        
        Parameters
        ----------
        trail_name : str
            CloudTrail name to query
        interval : int
            Polling interval in seconds (default 60s)
        duration : int
            Total duration in seconds (None = infinite)
        """
        start_time = time.time()
        anomalies = []
        
        print(f"Starting CloudTrail stream monitoring (interval={interval}s)...")
        print("=" * 80)
        
        while True:
            if duration and (time.time() - start_time) > duration:
                print("\nStream duration exceeded. Stopping...")
                break
            
            events = self.get_cloudtrail_logs(trail_name)
            
            if events:
                print(f"\n[{datetime.now().isoformat()}] Processing {len(events)} events...")
                
                for event in events:
                    result = self.classify_event(event)
                    
                    if result.get('is_anomaly'):
                        anomalies.append(result)
                        print(f"  ⚠️  ANOMALY DETECTED:")
                        print(f"      Event: {result['event_name']}")
                        print(f"      User: {result['username']}")
                        print(f"      IP: {result['source_ip']}")
                        print(f"      Region: {result['aws_region']}")
                        print(f"      Score: {result['anomaly_score']:.4f}")
                    else:
                        print(f"  ✓ Normal: {result['event_name']} by {result['username']}")
            else:
                print(f"[{datetime.now().isoformat()}] No new events")
            
            print(f"Total anomalies detected: {len(anomalies)}")
            time.sleep(interval)
        
        return anomalies


# ================================
# USAGE EXAMPLE
# ================================
if __name__ == "__main__":
    # Initialize detector with trained model
    detector = AnomalyDetectionStream(
        model_uri='models:/IsolationForestModel/1',
        contamination_threshold=0.01
    )
    
    # Stream CloudTrail logs for 5 minutes (300 seconds)
    # Replace 'your-trail-name' with actual CloudTrail name
    anomalies = detector.stream_and_classify(
        trail_name='your-trail-name',
        interval=30,  # Check every 30 seconds
        duration=300  # Run for 5 minutes
    )
    
    # Save detected anomalies to file
    if anomalies:
        df_anomalies = pd.DataFrame(anomalies)
        df_anomalies.to_csv('detected_anomalies.csv', index=False)
        print(f"\nDetected {len(anomalies)} anomalies. Saved to detected_anomalies.csv")
