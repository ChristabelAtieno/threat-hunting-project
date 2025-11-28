from sklearn.ensemble import IsolationForest

def isolation_forest_model(df_scaled):
    # isolation forest
    iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=42, n_jobs=-1)

    iso.fit(df_scaled)
    iso_scores = iso.decision_function(df_scaled)
    iso_pred = iso.predict(df_scaled)
    return iso_scores, iso_pred
