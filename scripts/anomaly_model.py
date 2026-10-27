import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.ensemble import IsolationForest


class CloudTrailDetector(OutlierMixin, BaseEstimator):
    """
    Two unsupervised detectors over CloudTrailFeatures output; an event is anomalous if either fires.

    - Isolation Forest: multivariate oddities (unusual combinations of time, errors, bursts,
      identity type, rarity). Isolation Forests cannot extrapolate past the values seen in
      training, so on their own they under-react to never-seen values.
    - Behavioural baseline: thresholds `behaviour_surprisal` (how unexpected this event's IP, API,
      region, client and hour are for this identity) at its (1 - behaviour_contamination) quantile
      on leave-one-out training features. This is what catches novel use of an identity.

    decision_function() is the Isolation Forest score (negative = anomalous);
    behaviour_score() is threshold minus surprisal in nats (negative = anomalous).
    """

    def __init__(self, contamination=0.005, behaviour_contamination=0.005, n_estimators=150, max_samples=256,
                 max_features=1.0, bootstrap=False, random_state=42, n_jobs=-1, verbose=0):
        self.contamination = contamination
        self.behaviour_contamination = behaviour_contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        self.forest_ = IsolationForest(contamination=self.contamination, n_estimators=self.n_estimators,
                                       max_samples=self.max_samples, max_features=self.max_features,
                                       bootstrap=self.bootstrap, random_state=self.random_state,
                                       n_jobs=self.n_jobs, verbose=self.verbose).fit(X)
        self.surprisal_threshold_ = float(np.quantile(X['behaviour_surprisal'], 1 - self.behaviour_contamination))
        return self

    def decision_function(self, X):
        return self.forest_.decision_function(X)

    def behaviour_score(self, X):
        return self.surprisal_threshold_ - X['behaviour_surprisal'].to_numpy()

    def forest_predict(self, X):
        return self.forest_.predict(X)

    def behaviour_predict(self, X):
        return np.where(self.behaviour_score(X) < 0, -1, 1)

    def predict(self, X):
        return np.where((self.forest_predict(X) == -1) | (self.behaviour_predict(X) == -1), -1, 1)
