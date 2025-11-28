import pandas as pd
from scripts.data_loading import load_logs
from scripts.engineer_features import prepare_dataset

# load the logs
df = load_logs("extracted/flaws_cloudtrail_logs")
df = df.explode("Records", ignore_index=True)
df = pd.json_normalize(df["Records"])

print(df.columns.tolist())

"""
#prepare the logs for modelling
df_engineered, df_scaled = prepare_dataset(df)

print("Engineered DataFrame:")
print(df_engineered.head())

print("\nScaled Array Shape:")
print(df_scaled.shape)
"""

