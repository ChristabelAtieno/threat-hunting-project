import dask.dataframe as dd
import numpy as np

def data_preprocessing(data, output_path):
    """
    Preprocesses the input Dask DataFrame by creating new features, encoding categorical variables,
    and saving the engineered features to a Parquet file.

    Parameters
    data : dd.DataFrame
        Input Dask DataFrame containing the raw features.
    output_path : Path
        Path to save the engineered features Parquet file.
    Returns
    None
    """

    output_path.mkdir(exist_ok=True)

    #--- TIME FEATURES ---
    data['eventTime'] = dd.to_datetime(data['eventTime'])
    data['hour_of_day'] = data['eventTime'].dt.hour
    data['day_of_week'] = data['eventTime'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    data['is_night'] = data['hour_of_day'].between(0, 6).astype(int)

    # --- SECURITY FEATURES ---
    data['has_error'] = data['errorMessage'].notnull().astype(int)
    data['has_access_key'] = data['userIdentity.accessKeyId'].notnull().astype(int)
    data['has_mfa'] = data['userIdentity.sessionContext.attributes.mfaAuthenticated'].notnull().astype(int)

    cols_to_drop = [col for col in ['eventTime', 'awsRegion','errorMessage', 'errorCode',
                                 'userIdentity.sessionContext.attributes.mfaAuthenticated',
                                 'userIdentity.accessKeyId'] if col in data.columns]
    
    data = data.drop(columns=cols_to_drop)


    # --- ENCODING CATEGORICAL FEATURES ---
    cat_cols = ['userIdentity.type','eventType','userIdentity.userName']
    data[cat_cols] = data[cat_cols].fillna('Unknown')
    data = data.categorize(columns=cat_cols)
    data = dd.get_dummies(data, columns=cat_cols, dummy_na=True)
    

    # --- COLUMNS WHERE RARITY MATTER MOST ---
    cad_cols = ['userIdentity.arn','eventName','eventSource','sourceIPAddress','userIdentity.invokedBy']
    for col in cad_cols:
        data[col] = data[col].fillna('Unknown')
        counts = data[col].value_counts().compute()
        data[f"{col}_frequency"] = data[col].map(counts).map(np.log1p)

    data = data.drop(columns=cad_cols)

    data.to_parquet(output_path, write_index=False, overwrite=True)
    print(f"Success! Features saved to {output_path}")
    return dd.read_parquet(output_path)
    