import dask.dataframe as dd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "processed_parquet"
df = dd.read_parquet(PROCESSED_DIR / "*.parquet")
#print(df.head())

#print(df['userIdentity.arn'].value_counts().compute())
#--- TIME FEATURES ---
#df['eventTime'] = dd.to_datetime(df['eventTime'])
#df['hour_of_day'] = df['eventTime'].dt.hour

print(df.isna().sum().compute())

