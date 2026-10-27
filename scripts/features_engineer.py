import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scripts.detections import ASSUME_ROLE_APIS, SENSITIVE_APIS
from scripts.graph_features import account_of, identity_communities, peer_service_counts

# Raw CloudTrail fields the pipeline reads. Training data and live events are both
# reindexed to this list, so a field missing from a batch becomes NaN instead of a KeyError.
RAW_COLUMNS = ['eventID', 'eventTime', 'eventName', 'eventSource', 'eventType', 'awsRegion',
               'sourceIPAddress', 'userAgent', 'readOnly', 'errorCode', 'errorMessage',
               'userIdentity.arn', 'userIdentity.type', 'userIdentity.userName',
               'userIdentity.accessKeyId', 'userIdentity.invokedBy',
               'userIdentity.sessionContext.attributes.mfaAuthenticated',
               'userIdentity.sessionContext.sourceIdentity',
               'additionalEventData.MFAUsed', 'responseElements.ConsoleLogin',
               'requestParameters.roleArn']

# arn:aws:sts::<account>:assumed-role/<role>/<session> -> arn:aws:sts::<account>:assumed-role/<role>
ASSUMED_ROLE_SESSION = r'^(arn:aws[\w-]*:sts::\d+:assumed-role/[^/]+)/.*$'

# Low-cardinality columns that are one-hot encoded
CATEGORICAL_COLS = ['userIdentity.type', 'eventType']

# Columns where rarity matters most, encoded as log frequency seen during training
FREQUENCY_COLS = ['actor', 'eventName', 'eventSource', 'sourceIPAddress', 'awsRegion',
                  'userAgentFamily', 'hourOfDay', 'userIdentity.invokedBy']

# Behaviour per identity: how often this actor did this in training. 0 means "first time for this actor"
ACTOR_PAIR_COLS = ['sourceIPAddress', 'eventName', 'awsRegion', 'userAgentFamily', 'hourOfDay']


def _text(series, default='Unknown'):
    return series.astype(object).where(series.notna(), default).astype(str)


def prepare_events(X):
    """
    Aligns raw CloudTrail records to RAW_COLUMNS and adds the derived columns the features use:
    parsed eventTime, `actor` (who the baseline is about, see below)
    and `userAgentFamily` (the client tool without its version, e.g. "aws-cli", "boto3").

    actor: the identity ARN, except for assumed roles, where the session name is often random or
    per-instance (botocore-session-123, i-0abc...) and would make every session look new:
      - normal roles      -> the role (arn:...:assumed-role/<role>)
      - SSO roles         -> kept per session, because the session name is the person
      - sourceIdentity    -> the role plus that person
    Events with no ARN use their identity type (e.g. type:AWSService).
    """
    X = X.reindex(columns=RAW_COLUMNS).copy()
    X['eventTime'] = pd.to_datetime(X['eventTime'], utc=True, errors='coerce')

    arn = _text(X['userIdentity.arn'], '')
    role = arn.str.replace(ASSUMED_ROLE_SESSION, r'\1', regex=True)
    actor = role.where(~arn.str.contains(':assumed-role/AWSReservedSSO_', regex=False), arn)
    source_identity = _text(X['userIdentity.sessionContext.sourceIdentity'], '')
    actor = actor.where(source_identity == '', role + '/' + source_identity)
    X['actor'] = actor.where(arn != '', 'type:' + _text(X['userIdentity.type']))

    user_agent = _text(X['userAgent'], '').str.lower()
    X['userAgentFamily'] = user_agent.str.extract(r'^\[?([^/\s\](,]+)', expand=False)
    X['hourOfDay'] = X['eventTime'].dt.hour.astype('Int64').astype(str)

    for col in CATEGORICAL_COLS + FREQUENCY_COLS:
        X[col] = _text(X[col])
    return X


class CloudTrailFeatures(BaseEstimator, TransformerMixin):
    """
    Turns raw CloudTrail records into numeric features for the anomaly model.

    The category lists and frequency tables (global and per identity) are learned in fit() and
    reused in transform(), so live events are encoded against the training baseline rather than
    their own batch. Values never seen in training get a frequency of 0 (maximally rare).

    Hourly burst features (events / errors / distinct APIs per identity per clock hour) are
    computed within the data being transformed, so score whole hours of activity together.

    fit_transform() (used when the pipeline is trained) encodes frequencies leave-one-out: each
    training event counts the *other* events with the same value, so one-off values in training
    get 0, just as never-seen values do at inference. Otherwise no training row could ever have
    frequency 0 and the model would not learn that "never seen" is unusual. As with sklearn's
    TargetEncoder, fit_transform(X) therefore differs from fit(X).transform(X).

    Behavioural surprisal: for each identity and dimension (IP, API, region, client, hour), the
    probability of this value under the identity's history, with the chance of a *new* value
    estimated from how many distinct values the identity has used (Witten-Bell smoothing).
    An identity that always uses one IP is very surprised by a new one; one that rotates IPs is not.
    Unknown identities fall back to account-wide probabilities. `behaviour_surprisal` is the sum,
    in nats; the detector thresholds it.

    Graph features (see graph_features.py): first-time role assumptions, role chaining and
    cross-account hops; how many identities share an IP normally and within the hour; whether the
    identity's peer community uses this service; and how many new connections the identity forms
    within the hour.
    """

    def fit(self, X, y=None):
        X = prepare_events(X)

        self.categories_ = {col: sorted(X[col].unique().tolist()) for col in CATEGORICAL_COLS}
        self.frequencies_ = {col: self._counts(X[col]) for col in FREQUENCY_COLS}
        self.actor_frequencies_ = {col: self._counts(X['actor'] + '|' + X[col]) for col in ACTOR_PAIR_COLS}
        self.actor_distinct_ = {col: {str(k): int(v) for k, v in X.groupby('actor')[col].nunique().items()}
                                for col in ACTOR_PAIR_COLS}
        self.global_distinct_ = {col: int(X[col].nunique()) for col in ['actor'] + ACTOR_PAIR_COLS}
        self.n_events_ = len(X)

        # --- GRAPH TABLES ---
        assume = X['eventName'].isin(ASSUME_ROLE_APIS)
        roles = _text(X['requestParameters.roleArn'], '')
        self.role_edges_ = self._counts((X['actor'] + '|' + roles)[assume])
        self.actor_assumes_ = self._counts(X.loc[assume, 'actor'])
        self.actor_roles_distinct_ = {str(k): int(v) for k, v in roles[assume].groupby(X.loc[assume, 'actor']).nunique().items()}
        self.role_counts_ = self._counts(roles[assume])
        self.n_assumes_ = int(assume.sum())
        self.ip_identities_ = {str(k): int(v) for k, v in X.groupby('sourceIPAddress')['actor'].nunique().items()}
        self.actor_service_counts_ = self._counts(X['actor'] + '|' + X['eventSource'])
        self.communities_ = {str(k): int(v) for k, v in identity_communities(X['actor'], X['eventSource']).items()}
        (self.community_service_counts_, self.community_sizes_,
         self.service_identities_) = peer_service_counts(X['actor'], X['eventSource'], self.communities_)

        self.feature_names_out_ = np.array(self._encode(X).columns)
        return self

    def transform(self, X):
        features = self._encode(prepare_events(X))
        return features.reindex(columns=self.feature_names_out_, fill_value=0)

    def fit_transform(self, X, y=None):
        self.fit(X)
        features = self._encode(prepare_events(X), leave_one_out=True)
        return features.reindex(columns=self.feature_names_out_, fill_value=0)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

    @staticmethod
    def _surprisal(count, distinct, total, leave_one_out):
        """
        -log P(value) with Witten-Bell smoothing: a value seen c times out of n, among k distinct
        values, has P = c / (n + k); a new value has P = k / (n + k).
        """
        count = np.clip(count, 0, None)
        # leave-one-out: if this row was the value's only occurrence, it no longer counts as distinct
        distinct = np.clip(distinct - (leave_one_out & (count == 0)), 1, None)
        denominator = np.clip(total, 0, None) + distinct
        p = np.where(count > 0, count, distinct) / denominator
        return -np.log(np.clip(p, 1e-12, 1.0))

    @staticmethod
    def _counts(series):
        return {str(k): int(v) for k, v in series.value_counts().items()}

    def _encode(self, X, leave_one_out=False):
        exclude_self = 1 if leave_one_out else 0
        features = pd.DataFrame(index=X.index)

        # --- TIME FEATURES ---
        features['hour_of_day'] = X['eventTime'].dt.hour.fillna(-1)
        features['day_of_week'] = X['eventTime'].dt.dayofweek.fillna(-1)
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        features['is_night'] = features['hour_of_day'].between(0, 6).astype(int)

        # --- SECURITY FEATURES ---
        error_code = _text(X['errorCode'], '')
        features['has_error'] = (X['errorCode'].notna() | X['errorMessage'].notna()).astype(int)
        features['access_denied'] = error_code.str.contains('AccessDenied|Unauthorized|Forbidden').astype(int)
        features['has_access_key'] = X['userIdentity.accessKeyId'].notna().astype(int)
        # mfaAuthenticated is the string "true"/"false"; presence alone does not mean MFA was used
        mfa = X['userIdentity.sessionContext.attributes.mfaAuthenticated']
        features['has_mfa'] = mfa.astype(str).str.lower().eq('true').astype(int)
        features['is_root'] = X['userIdentity.type'].eq('Root').astype(int)
        features['is_write'] = _text(X['readOnly'], '').str.lower().eq('false').astype(int)
        features['is_sensitive_api'] = X['eventName'].isin(SENSITIVE_APIS).astype(int)

        # --- ENCODING CATEGORICAL FEATURES ---
        for col in CATEGORICAL_COLS:
            for category in self.categories_[col]:
                features[f"{col}_{category}"] = X[col].eq(category).astype(int)

        # --- COLUMNS WHERE RARITY MATTER MOST ---
        for col in FREQUENCY_COLS:
            counts = X[col].map(self.frequencies_[col]).fillna(0).astype(float) - exclude_self
            features[f"{col}_frequency"] = np.log1p(counts.clip(lower=0))

        # --- PER-IDENTITY BEHAVIOUR (first time this actor used this IP / API / region / client) ---
        for col in ACTOR_PAIR_COLS:
            counts = (X['actor'] + '|' + X[col]).map(self.actor_frequencies_[col]).fillna(0).astype(float) - exclude_self
            features[f"actor_{col}_frequency"] = np.log1p(counts.clip(lower=0))

        # --- BEHAVIOURAL SURPRISAL (how unexpected is this value for this identity) ---
        actor_events = X['actor'].map(self.frequencies_['actor']).fillna(0).to_numpy(float) - exclude_self
        surprisal = self._surprisal(actor_events, self.global_distinct_['actor'], self.n_events_ - exclude_self,
                                    leave_one_out)
        features['surprisal_actor'] = surprisal
        for col in ACTOR_PAIR_COLS:
            pair = (X['actor'] + '|' + X[col]).map(self.actor_frequencies_[col]).fillna(0).to_numpy(float) - exclude_self
            distinct = X['actor'].map(self.actor_distinct_[col]).fillna(0).to_numpy(float)
            per_actor = self._surprisal(pair, distinct, actor_events, leave_one_out)
            overall = X[col].map(self.frequencies_[col]).fillna(0).to_numpy(float) - exclude_self
            account_wide = self._surprisal(overall, self.global_distinct_[col], self.n_events_ - exclude_self,
                                           leave_one_out)
            features[f'surprisal_{col}'] = np.where(actor_events > 0, per_actor, account_wide)
            surprisal = surprisal + features[f'surprisal_{col}'].to_numpy()
        features['behaviour_surprisal'] = surprisal

        # --- GRAPH: ROLE ASSUMPTION (lateral movement, privilege escalation) ---
        hour = X['eventTime'].dt.floor('h')
        assume = X['eventName'].isin(ASSUME_ROLE_APIS)
        target = _text(X['requestParameters.roleArn'], '')
        edge = (X['actor'] + '|' + target).map(self.role_edges_).fillna(0) - exclude_self
        features['is_assume_role'] = assume.astype(int)
        features['assume_role_new_edge'] = (assume & (target != '') & (edge <= 0)).astype(int)
        features['role_chaining'] = (assume & X['userIdentity.type'].eq('AssumedRole')).astype(int)
        actor_account = account_of(_text(X['userIdentity.arn'], ''))
        target_account = account_of(target)
        features['cross_account_assume'] = (assume & actor_account.notna() & target_account.notna()
                                            & (actor_account != target_account)).astype(int)
        # Which role, among this identity's own role assumptions: ~0 for the role it always assumes,
        # high for one it never has. Only AssumeRole events get a value, so other events are unaffected.
        is_assume = assume.to_numpy()
        loo = exclude_self * is_assume
        pair = (X['actor'] + '|' + target).map(self.role_edges_).fillna(0).to_numpy(float) - loo
        n_assumes = X['actor'].map(self.actor_assumes_).fillna(0).to_numpy(float) - loo
        per_actor = self._surprisal(pair, X['actor'].map(self.actor_roles_distinct_).fillna(0).to_numpy(float),
                                    n_assumes, leave_one_out)
        account_wide = self._surprisal(target.map(self.role_counts_).fillna(0).to_numpy(float) - loo,
                                       len(self.role_counts_), self.n_assumes_ - loo, leave_one_out)
        features['surprisal_assumedRole'] = np.where(is_assume, np.where(n_assumes > 0, per_actor, account_wide), 0.0)
        features['behaviour_surprisal'] = features['behaviour_surprisal'] + features['surprisal_assumedRole']

        # --- GRAPH: IPS SHARED BY IDENTITIES (one machine, several stolen keys) ---
        pair_ip = (X['actor'] + '|' + X['sourceIPAddress']).map(self.actor_frequencies_['sourceIPAddress']).fillna(0)
        known_identities = X['sourceIPAddress'].map(self.ip_identities_).fillna(0) - (leave_one_out & (pair_ip == 1))
        features['ip_known_identities'] = np.log1p(known_identities.clip(lower=0))
        features['ip_identities_in_hour'] = np.log1p(
            pd.DataFrame({'ip': X['sourceIPAddress'], 'hour': hour, 'actor': X['actor']})
            .groupby(['ip', 'hour'], dropna=False)['actor'].transform('nunique'))

        # --- GRAPH: PEER COMMUNITY (does anyone like this identity use this service?) ---
        # Peers are the other members of the identity's community; lone or unknown identities are
        # compared with every other identity. The identity itself is never counted.
        member = X['actor'].map(self.communities_)
        community = member.astype('Int64').astype(str)
        counted_self = (X['actor'] + '|' + X['eventSource']).map(self.actor_service_counts_).fillna(0).gt(0).astype(int)
        peers = community.map(self.community_sizes_).fillna(0) - member.notna().astype(int)
        peer_users = (community + '|' + X['eventSource']).map(self.community_service_counts_).fillna(0) - counted_self
        everyone = self.global_distinct_['actor'] - member.notna().astype(int)
        all_users = X['eventSource'].map(self.service_identities_).fillna(0) - counted_self
        share = np.where(peers > 0, peer_users / peers.clip(lower=1), all_users / everyone.clip(lower=1))
        features['peer_service_share'] = np.clip(share, 0, 1)

        # --- BURSTS PER IDENTITY PER CLOCK HOUR (recon sweeps, error storms, many new connections) ---
        new_edges = sum((features[f'actor_{col}_frequency'] == 0).astype(int)
                        for col in ['sourceIPAddress', 'eventName', 'awsRegion', 'userAgentFamily'])
        activity = pd.DataFrame({'actor': X['actor'], 'hour': hour, 'api': X['eventName'],
                                 'error': features['has_error'],
                                 'new_edges': new_edges + features['assume_role_new_edge']})
        per_hour = activity.groupby(['actor', 'hour'], dropna=False)
        features['actor_events_in_hour'] = np.log1p(per_hour['api'].transform('size'))
        features['actor_errors_in_hour'] = np.log1p(per_hour['error'].transform('sum'))
        features['actor_distinct_apis_in_hour'] = np.log1p(per_hour['api'].transform('nunique'))
        features['actor_new_edges_in_hour'] = np.log1p(per_hour['new_edges'].transform('sum'))

        return features.astype(float)
