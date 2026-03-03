from pathlib import Path
from scripts.load_data import load_data
from scripts.features_engineer import data_preprocessing
from scripts.train_models import isolation_forest_model
import dask.dataframe as dd
import mlflow

project_root = Path(__file__).resolve().parent
DOCUMENTS_DIR = project_root / "extracted" / "flaws_cloudtrail_logs"
PROCESSED_DIR = project_root / "processed_parquet"
ENGINEERED_DIR = project_root / "engineered_parquet"

if __name__ == "__main__":

    print("--- Loading and Normalizing Raw Logs ---")
    if not PROCESSED_DIR.exists():
        raw_data = load_data(DOCUMENTS_DIR, PROCESSED_DIR)
    else:
        print(f"Processed data already exists in {PROCESSED_DIR}. Skip the loading step.")

    print("--- Engineering Features ---")
    if not ENGINEERED_DIR.exists():
        data_preprocessing(raw_data, ENGINEERED_DIR)
    else:
        print(f"Engineered data already exists in {ENGINEERED_DIR}. Skip the feature engineering step.")

    print("--- Loading Engineered Features ---")
    ddf = dd.read_parquet(ENGINEERED_DIR / "*.parquet")

    print(f"Final dataset shape: {len(ddf.columns)} columns")
    print(ddf.head())

    print("--- Training Isolation Forest Model ---")
    model, scores, preds = isolation_forest_model(ddf)

    print("Model training complete. Anomaly scores and predictions are ready for analysis.")
    print(f"Sample anomaly scores: {scores[:5]}")
    print(f"Sample predictions: {preds[:5]}")
    print(f"Total anomalies detected: {(preds == -1).sum()}")
    print(f"Predictions shape: {preds.shape}")
    print(f"\nAnomalies detected: {sum(preds == -1)} out of {len(preds)}")
    print(f"Anomaly percentage: {100 * sum(preds == -1) / len(preds):.2f}%")

