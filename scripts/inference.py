import os
from dotenv import load_dotenv
import boto3
import mlflow
import mlflow.sklearn
import json
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from scripts.features_engineer import data_preprocessing

load_dotenv()

project_root = Path(__file__).resolve().parent.parent

DIR = project_root / "data" 

session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION'))

client = session.client('cloudtrail')

response = client.lookup_events(MaxResults=1000)

raw_records = [json.loads(event['CloudTrailEvent']) for event in response['Events']]

df = pd.json_normalize(raw_records, sep='.')


# mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))

cols_to_fix = ['readOnly', 'managementEvent']
for col in cols_to_fix:
    if col in df.columns:
        df[col] = df[col].astype(str)

ddf = dd.from_pandas(df, npartitions=1)

processed_ddf = data_preprocessing(ddf, DIR)
ready_data = processed_ddf.compute()
print(ready_data.head())

#load model
model_uri = f"runs:/5f65632c59a849a586b97b665a1e08d6/threat_hunt_model"
loaded_model = mlflow.sklearn.load_model(model_uri)

expected_features = loaded_model.feature_names_in_

final_features = ready_data.reindex(columns=expected_features, fill_value=0)

predictions = loaded_model.predict(final_features)
scores = loaded_model.decision_function(final_features)

ready_data['anomaly_score'] = scores
ready_data['anomaly_prediction'] = predictions

print("--- HUNT RESULTS ---")
print(ready_data.head())

