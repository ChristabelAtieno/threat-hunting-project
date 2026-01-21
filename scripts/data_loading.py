import os, gzip, uuid, json
import pandas as pd



#-------------------------------------------
# Loads AWS CloudTrail logs into a DataFrame
#-------------------------------------------


"""


    events = []

    # If path is a directory
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json.gz")]
    
    # path is a single file
    elif os.path.isfile(path) and path.endswith(".json.gz"):
        files = [path]
    else:
        raise ValueError("Path must be a directory or a .json.gz file.")
    
    #Load files
    for file in files:
        print(f"Loading: {file}")
        try:
            with gzip.open(file, "rt") as f:
                for line in f:
                    obj = json.loads(line)

                    if isinstance(obj, dict) and "Records" in obj:
                        for ev in obj["Records"]:
                            events.append(ev)
                    else:
                         events.append(obj)

        except Exception as e:
            print(f"Error reading {file}: {e}")
    df = pd.DataFrame(events)
    print(f"Loaded {len(df)} rows from {len(files)} file(s).")
    
    return df
"""

def load_logs(output_dir):

    log_dir = "extracted/flaws_cloudtrail_logs"
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 50_000
    records_batch = []

    DROP_PREFIXES = (
                    "requestParameters.",
                    "responseElements.",
                    "additionalEventData.")
    
    def sanitize_for_parquet(df):
        for col in df.columns:
            #if df[col].dtype == "object":
            non_null = df[col].dropna()
            if not non_null.empty and isinstance(non_null.iloc[0], (dict, list)):
                df[col] = df[col].astype(str)
        return df

    for file in sorted(os.listdir(log_dir)):
        if file.endswith(".json.gz"):
            file_path = os.path.join(log_dir, file)

            
            with gzip.open(file_path, "rt", encoding="utf-8",errors="replace") as f:
                data = json.load(f)

                for record in data.get("Records", []):
                    records_batch.append(record)

                    if len(records_batch) >= batch_size:
                        df = pd.json_normalize(records_batch)
                        
                        
                        df = df.loc[:, [c for c in df.columns if not c.startswith(DROP_PREFIXES)]]
                        df = sanitize_for_parquet(df)

                        out_file = os.path.join(
                            output_dir,
                            f"part-{uuid.uuid4().hex}.parquet"
                        )

                        df.to_parquet(out_file, engine="pyarrow", index=False)
                        records_batch = []

    # write remaining
    if records_batch:
        df = pd.json_normalize(records_batch)
        df = df.loc[:, [c for c in df.columns if not c.startswith(DROP_PREFIXES)]]
        df = sanitize_for_parquet(df)

        out_file = os.path.join(
            output_dir,
            f"part-{uuid.uuid4().hex}.parquet"
        )
        df.to_parquet(out_file, engine="pyarrow", index=False)

    return output_dir