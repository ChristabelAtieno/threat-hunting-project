from sklearn.ensemble import IsolationForest
#from dask_ml.ensemble import IsolationForest as DaskIsolationForest
import dask.array as da
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv

load_dotenv()

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
    Scores:
        Anomaly scores for each sample in the dataset.
    preds:
        Anomaly predictions for each sample in the dataset (-1 for anomalies, 1 for normal
    """

    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME', 'default_experiment'))

    with mlflow.start_run(run_name="IsolationForest_threat_Hunt"):

        params = {
            "n_estimators": 150,
            "max_samples": 256,
            "max_features": 1.0,
            "contamination": 0.01,
            "bootstrap": False,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": 0
        }

        mlflow.log_params(params)

        model = IsolationForest(**params)

        model.fit(data)
        scores = model.decision_function(data)
        preds = model.predict(data)
   
        # log metrics
        n_anomalies = sum(preds == -1)
    
        mlflow.log_metrics({
            "anomalies_detected": n_anomalies,
            "anomaly_rate_percent": 100 * n_anomalies / len(preds),
            "total_samples": len(preds),
            "mean_anomaly_score": scores.mean(),
            "std_anomaly_score": scores.std()})
        
        # Log model
        mlflow.sklearn.log_model(sk_model=model,
                                name="threat_hunt_model",
                                registered_model_name="isolation_forest_model")
        
        return model, scores, preds
