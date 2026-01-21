import mlflow
import pandas as pd
from scripts.data_loading import load_logs
from scripts.engineer_features import (
    clean_parquet_files, 
    engineer_features, 
    encoding_features,
    prepare_dataset
)
from scripts.train_models import isolation_forest_model

# ===== PIPELINE EXECUTION =====
if __name__ == "__main__":

    mlflow.set_experiment("cloudtrail_anomaly_detection")

    #print("Step 1: Loading raw CloudTrail logs...")
    #raw_dir = load_logs("notebook/cloudtrail_parquet")

    #print("\nStep 2: Cleaning parquet files...")
    #clean_parquet_files(raw_dir, "notebook/clean_cloudtrail_parquet")

    #print("\nStep 3: Engineering features...")
    #engineer_features("notebook/clean_cloudtrail_parquet", "notebook/features_parquet")

    #print("\nStep 4: Encoding categorical features...")
    #encoding_features("notebook/features_parquet", "notebook/clean_features_parquet")

    print("\nStep 5: Preparing dataset and scaling...")
    df_engineered, df_scaled = prepare_dataset("notebook/clean_features_parquet")

    print("\nStep 6: Training Isolation Forest model...")
    model, scores, predictions = isolation_forest_model(df_scaled)

    print("\n===== RESULTS =====")
    print(f"Engineered DataFrame shape: {df_engineered.shape}")
    print(f"Scaled Array shape: {df_scaled.shape}")
    print(f"Model anomaly scores shape: {scores.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"\nAnomalies detected: {sum(predictions == -1)} out of {len(predictions)}")
    print(f"Anomaly percentage: {100 * sum(predictions == -1) / len(predictions):.2f}%")

