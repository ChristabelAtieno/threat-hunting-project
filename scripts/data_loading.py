import os
import gzip
import json
import pandas as pd

#-------------------------------------------
# Loads AWS CloudTrail logs into a DataFrame
#-------------------------------------------
def load_logs(path):

    logs = []

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
                    logs.append(json.loads(line))
        except Exception as e:
            print(f"Error reading {file}: {e}")
    df = pd.DataFrame(logs)
    print(f"Loaded {len(df)} rows from {len(files)} file(s).")
    
    return df

