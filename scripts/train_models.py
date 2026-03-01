from sklearn.ensemble import IsolationForest
#from dask_ml.ensemble import IsolationForest as DaskIsolationForest
import dask.array as da
import mlflow
import mlflow.sklearn


def isolation_forest_model(data):
    """
    Trains an Isolation Forest model on the provided data, logs parameters and metrics to MLflow,
    and registers the model in the MLflow Model Registry.

    Parameters
    data: 
        A DataFrame containing the features to train the model on.

    Returns
    model: 
        The trained Isolation Forest model.
    """
    model = IsolationForest(n_estimators=500,
                            max_samples='auto', 
                            contamination=0.01, 
                            random_state=42, 
                            n_jobs=-1)

    with mlflow.start_run("isolationforeest_model"):

        model.fit(data)
        scores = model.decision_function(data)
        preds = model.predict(data)

        # log parameters
        mlflow.log_params({
            "n_estimators": 500,
            "max_samples": 'auto',
            "contamination": 0.01,
            "random_state": 42,
            "n_jobs": -1})
        
        # log metrics
        n_anomalies = sum(preds == -1)
        mlflow.log_metrics({
            "anomalies_detected": n_anomalies,
            "anomaly_rate_percent": 100 * n_anomalies / len(preds),
            "total_samples": len(preds)})
        
        # Log model
        mlflow.sklearn.log_model(sk_model=model, 
                                 registered_model_name="isolation_forest_model")
        

        return model, scores, preds
