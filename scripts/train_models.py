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
    model = IsolationForest(n_estimators=300,
                            max_samples='auto', 
                            contamination=0.01, 
                            random_state=42, 
                            n_jobs=-1)

    with mlflow.start_run(run_name="isolation_forest_model"):

        model.fit(data)
        scores = model.decision_function(data)
        preds = model.predict(data)

        # log parameters
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_samples", 'auto')
        mlflow.log_param("contamination", 0.01)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_jobs", -1)
        
        # log metrics
        n_anomalies = sum(preds == -1)
        mlflow.log_metric("anomalies_detected", n_anomalies)
        mlflow.log_metric("anomaly_rate_percent", 100 * n_anomalies / len(preds))
        mlflow.log_metric("total_samples", len(preds))

        # Log model
        mlflow.sklearn.log_model(sk_model=model, 
                                 artifact_path="isolation_forest_model")
        
        # Get current run info for model registration
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/isolation_forest_model"
        
        # Register the model
        mlflow.register_model(model_uri, "IsolationForestModel")

        return model, scores, preds
