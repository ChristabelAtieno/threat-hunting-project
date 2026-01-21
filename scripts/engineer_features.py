import os
import pyarrow.parquet as pq
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
# --------------------------------
# PREPROCESS DATA
# --------------------------------
def clean_parquet_files(parquet_dir: str, output_dir: str, columns_to_drop: list = None):
    """
    Cleans all Parquet files in a directory by dropping specified columns
    and saving the cleaned files to an output directory.

    Parameters
    ----------
    parquet_dir : str
        Directory containing original Parquet files.
    output_dir : str
        Directory to save cleaned Parquet files.
    columns_to_drop : list, optional
        List of column names to drop from each Parquet file.
        If None, uses default list of columns to drop.

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    if columns_to_drop is None:
        columns_to_drop = ['userIdentity.arn','serviceEventDetails.snapshotId','managementEvent','readOnly','vpcEndpointId',
                    'userIdentity.sessionContext.sessionIssuer.userName','userIdentity.sessionContext.sessionIssuer.accountId',
                    'userIdentity.sessionContext.sessionIssuer.principalId','apiVersion','userIdentity.sessionContext.attributes.creationDate','userIdentity.sessionContext.sessionIssuer.arn',
                    'userIdentity.sessionContext.sessionIssuer.type','sharedEventID','userIdentity.userName','resources','userIdentity.accessKeyId','userIdentity.accountId',
                    'userIdentity.principalId','eventVersion','responseElements','eventCategory',
                    'requestID','eventID','recipientAccountId','requestParameters']
    
    for file in sorted(os.listdir(parquet_dir)):
        if file.endswith(".parquet"):
            input_path = os.path.join(parquet_dir, file)
            output_path = os.path.join(output_dir, file)

            # Read Parquet file in chunks
            table = pq.read_table(input_path)
            df = table.to_pandas()

            # Drop unwanted columns
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

            # Write back to Parquet
            df.to_parquet(output_path, engine="pyarrow", index=False)
            print(f"Cleaned {file} saved to {output_dir}")

#----------------------------------
# FEATURE ENGINEERING
#----------------------------------
def engineer_features(input_dir: str, output_dir: str):

    os.makedirs(output_dir, exist_ok=True)

    presence_cols = ['userIdentity.sessionContext.attributes.mfaAuthenticated']

    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".parquet"):
            df = pq.read_table(os.path.join(input_dir, file)).to_pandas()

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
            for col in presence_cols:
                flag_name = f'has_{col.split(".")[-1]}'
                if col in df.columns:
                    df[flag_name] = df[col].notna().astype(int)
                else:
                    df[flag_name] = 0

            # ---- CLEANUP ----
            cols_to_drop = [col for col in ['eventTime', 'errorMessage', 'errorCode',
                                             'userIdentity.sessionContext.attributes.mfaAuthenticated']
                            if col in df.columns]
            df = df.drop(columns=cols_to_drop)

            # ---- SAVE ----
            df.to_parquet(os.path.join(output_dir, file), index=False)
            print(f"Engineered features {file} saved to {output_dir}")

# ---------------------------------
# ENCODING CATEGORICAL FEATURES
# ---------------------------------
def encoding_features(input_dir: str, output_dir: str):

    os.makedirs(output_dir, exist_ok=True)

    categorical_cols = ['eventType','awsRegion','userIdentity.type','userIdentity.invokedBy']
    freq_cols = ['userAgent','sourceIPAddress','eventName','eventSource']

    # ----- compute global frequencies ----
    global_counts = {col: Counter() for col in freq_cols}
    total_rows = 0
            
    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(input_dir, file))
            total_rows += len(df)
            
            for col in freq_cols:
                if col in df.columns:
                    global_counts[col].update(df[col].dropna().astype(str))

    # Convert counts → frequencies (handle zero division)
    if total_rows > 0:
        freq_maps = {
            col: {k: v / total_rows for k, v in counter.items()}
            for col, counter in global_counts.items()
        }
    else:
        print("Warning: No rows found in input directory. Returning empty frequency maps.")
        freq_maps = {col: {} for col in freq_cols}

    # -----encoding and saving-----
    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(input_dir, file))

            # ---- ONE-HOT ENCODING----
            existing_cat_cols = [col for col in categorical_cols if col in df.columns]
            if existing_cat_cols:
                df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=False)
            else:
                print(f"Warning: No categorical columns found in {file}")

            # ---- FREQUENCY ENCODING ----
            for col in freq_cols:
                if col in df.columns:
                    df[col + "_freq"] = (
                        df[col]
                        .astype(str)
                        .map(freq_maps[col])
                        .fillna(0)
                    )
                    df.drop(columns=[col], inplace=True)


            # ---- SAVE ----
            df.to_parquet(os.path.join(output_dir, file), index=False)
            print(f"Encoded and saved {file}")

# ---------------------------------
# CONCATENATE FEATURES
# ---------------------------------
def prepare_dataset(input_dir: str):
    """
    Concatenates all Parquet files in a directory into a single DataFrame,
    scales the features, and returns both the DataFrame and scaled NumPy array.

    Parameters
    ----------
    input_dir : str
        Directory containing Parquet files with engineered features.

    Returns
    -------
    df_engineered : pd.DataFrame
        Concatenated DataFrame of all engineered features.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    
    data_frames = []

    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(input_dir, file))
            data_frames.append(df)

    if not data_frames:
        raise ValueError(f"No Parquet files found in {input_dir}")

    # Concatenate all DataFrames
    df_engineered = pd.concat(data_frames, ignore_index=True)

    # Scale the features
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_engineered)

    return df_engineered, scaled_array

