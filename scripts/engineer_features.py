import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

#------------------------
#preprocess data
#------------------------
def preprocess(df):
    """Clean, normalize, handle missing, convert types"""

    # parse timestamps
    df['eventTime'] = pd.to_datetime(df['eventTime'], errors='coerce')

    # drop the unnecessary columns
    cols_to_drop = ['eventID','requestID', 'eventVersion', 'recipientAccountId','responseElements','apiVersion','serviceEventDetails','eventCategory','managementEvent',
                'vpcEndpointId','sharedEventID','additionalEventData','resources']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # fill in missing values
    df['userAgent'] = df['userAgent'].fillna('Unknown')
    df['requestParameters'] = df['requestParameters'].fillna("{}")
    df[['errorMessage','errorCode']] = df[['errorMessage','errorCode']].fillna('None')

    return df

#-----------------------------------
#Extract users
#----------------------------------
# extract the users from the user identity column
def extract_user_identity(df):
    def extract_user(x):
        try:
            if isinstance(x, dict):
                return x.get("arn") or x.get("userName") or x.get("principalId")
            else:
                return str(x)
        except:
            return "unknownUser"
    
    df['userIdentitySimple'] = df['userIdentity'].apply(extract_user)
    return df

#-----------------------------------
# add time features
#-----------------------------------
def add_time_features(df):
    df['hour'] = df['eventTime'].dt.hour
    df['day_of_week'] = df['eventTime'].dt.dayofweek
    df['isWeekend'] = df['day_of_week'].isin([5,6]).astype(int)

    df = df.sort_values(by=['userIdentitySimple', 'eventTime'])
    df['timeSinceLastEvent'] = df.groupby('userIdentitySimple')['eventTime'].diff().dt.total_seconds()
    df['timeSinceLastEvent'] = df['timeSinceLastEvent'].fillna(df['timeSinceLastEvent'].median())

    df['isNight'] = df['hour'].between(0, 6).astype(int)
    return df

#------------------------------------
# Features from the request parameters
#-------------------------------------
def extract_request_features(df):
    df['hasRequestParams'] = (df['requestParameters'] != "{}").astype(int)

    def count_keys(x):
        try:
            return len(json.loads(x))
        except:
            return 0
    df['paramNumKeys'] = df['requestParameters'].apply(count_keys)
    df['paramLength'] = df['requestParameters'].astype(str).apply(len)

    # semantic threat features
    keyword_features = {
        'hasRole':'role',
        'hasPolicy':'policy',
        'hasAccessKey':'accessKey',
        'hasUser':'userName',
        'hasBucket':'bucket',
        'hasSecurityGroup':'securityGroup',
        'hasInstance':'instanceId'
    }

    for col, keyword in keyword_features.items():
        df[col] = df['requestParameters'].str.contains(keyword, case=False, na=False).astype(int)
    
    return df

def other_features(df):
    # create isWriteOperation from readOnly
    df['isWriteOperation'] = df['readOnly'].apply(lambda x: 1 if x == False else 0)
    # create binary error flag
    df['hasError'] = (df['errorCode'] != 'None').astype(int)
    # Create semantic flags from errorCode
    df['isAccessDenied'] = (df['errorCode'] == 'AccessDenied').astype(int)
    df['isNoSuchBucket'] = (df['errorCode'] == 'NoSuchBucket').astype(int)
    df['isUnauthorized'] = df['errorCode'].str.contains('Unauthorized', na=False).astype(int)
    return df

def drop_features(df):
    cols = ['readOnly','userIdentity','requestParameters', 'eventTime','errorMessage','errorCode']
    df = df.drop(columns=[c for c in cols if c in df.columns],errors='ignore')
    return df

def encode_features(df):
    df = pd.get_dummies(df, columns=['eventType'], drop_first=True)

    def frequency_encoding(df_inner):
        for col in ['userIdentitySimple', 'eventName', 'eventSource', 'userAgent', 'awsRegion','sourceIPAddress']:
            freq = df_inner[col].value_counts(normalize=True)
            df_inner[f'{col}_freq'] = df_inner[col].map(freq)
        return df_inner
    df = frequency_encoding(df)
    return df

def preprocess_again(df):
    # convert the boolean columns to numeric
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    #fill thenulls after the encoding
    freq_cols = [col for col in df.columns if col.endswith('_freq')]
    df[freq_cols] = df[freq_cols].fillna(0)
    return df

def scale_features(df):
    # scale features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_scaled

#-----------------------------------
# The engineered features
#----------------------------------
def engineer_features(df):
    """Add all derived features"""
    df = extract_user_identity(df)
    df = add_time_features(df)
    df = extract_request_features(df)
    df = other_features(df)
    df = drop_features(df)
    df = encode_features(df)
    return df


def prepare_dataset(df):
    """ The prepared dataset"""
    df = preprocess(df)
    df = engineer_features(df)
    df = preprocess_again(df)
    df_scaled = scale_features(df)
    return df, df_scaled


