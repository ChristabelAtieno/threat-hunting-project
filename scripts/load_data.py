from pathlib import Path
import gzip
import json
import pandas as pd

    
def load_data(data_path, output_path):
    
    """
    Loads the json files from the directory
    Keeping only the relevant columns for my objectives

    Parameters
    data_path:
        This is the path where the logs are stored
    output_path:
        This is the path where the parquet files would be stored after being loaded

    Returns
        The loaded data and stores the data into parquet files
    """

    output_path.mkdir(exist_ok=True)

    TOP_FEATURES = ['userIdentity.arn','userIdentity.type','userIdentity.sessionContext.attributes.mfaAuthenticated',
                'eventName','eventSource','sourceIPAddress','awsRegion', 'errorCode',	
                'errorMessage','eventTime',	'requestParameters.bucketName','requestParameters.userName',
                'recipientAccountId','userIdentity.userName', 'userIdentity.accessKeyId']
    for file in data_path.glob('*.json.gz'):
        with gzip.open(file, 'rb') as f:
            data = json.load(f)
            df = pd.json_normalize(data['Records'])
            df = df.reindex(columns=TOP_FEATURES)
            output_filename = output_path / file.stem.replace(".json", ".parquet")
            df.to_parquet(output_filename, engine='pyarrow')

    print(f"Successfully created the parquet files in {output_path}")



  