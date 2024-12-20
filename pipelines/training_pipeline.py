import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import logging
from mlflow.models.signature import infer_signature
from datetime import datetime

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/train_pipeline.log"),
            logging.StreamHandler()
        ]
    )


def run_lightgbm_training_pipeline(data_source, target_column, n_splits=3, mlflow_experiment_name="LightGBM_Disruption_Model"):
    """
    Train LightGBM models with TimeSeriesSplit, log metrics using MLflow.

    Parameters:
    data_source (str): Path to the input dataset file.
    target_column (str): The target column for the model.
    n_splits (int): Number of splits for TimeSeriesSplit.
    mlflow_experiment_name (str): Name of the MLflow experiment to log results.

    Returns:
    dict: Metrics for each split (accuracy, F1 score, ROC AUC score).
    """
    try:
        setup_logging()
        logging.info(f"Loading dataset from {data_source}.")
        
        # Load the dataset
        data = pd.read_parquet(data_source)
        logging.info(f"Dataset loaded successfully. Shape: {data.shape}")
        
        # Features and target
        X = data.drop(columns=[target_column, "Disruption_Type"])
        y = data[target_column]

        # Set MLflow experiment
        mlflow.set_experiment(mlflow_experiment_name)

        # Initialize TimeSeriesSplit
        ts_splits = TimeSeriesSplit(n_splits=n_splits)
        
        metrics = {
            "accuracy": [],
            "f1_score": [],
            "roc_auc": []
        }

        # Check for an active run
        if mlflow.active_run() is None:
            parent_run = mlflow.start_run(run_name=f"LightGBM_Training_Pipeline_{datetime.now()}")
        else:
            parent_run = mlflow.active_run()

        with parent_run:
            # Log dataset as an artifact
            mlflow.log_artifact(data_source, artifact_path="datasets")

            # Iterate through splits
            for fold, (train_index, test_index) in enumerate(ts_splits.split(X)):
                train_size = int(len(train_index) * 0.8)
                valid_index = train_index[train_size:]
                train_index = train_index[:train_size]

                X_train, X_valid, X_test = X.iloc[train_index], X.iloc[valid_index], X.iloc[test_index]
                y_train, y_valid, y_test = y.iloc[train_index], y.iloc[valid_index], y.iloc[test_index]

                # Create LightGBM datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_valid, label=y_valid)

                # Set LightGBM parameters
                params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "learning_rate": 0.01,
                    "feature_fraction": 0.9
                }

                # MLflow Logging
                with mlflow.start_run(run_name=f"Fold_{fold+1}", nested=True):
                    mlflow.log_params(params)

                    # Train LightGBM model
                    lgb_model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=100,
                        valid_sets=[valid_data],
                        callbacks=[lgb.early_stopping(stopping_rounds=10)],
                    )

                    # Log model with input_example and signature
                    input_example = X_train.head(1)
                    signature = infer_signature(X_train, lgb_model.predict(X_train.head(1)))
                    mlflow.lightgbm.log_model(
                        lgb_model,
                        artifact_path="model",
                        input_example=input_example,
                        signature=signature
                    )

                    # Predict on train, validation, and test sets
                    y_train_pred = (lgb_model.predict(X_train) > 0.5).astype(int)
                    y_valid_pred = (lgb_model.predict(X_valid) > 0.5).astype(int)
                    y_test_pred = (lgb_model.predict(X_test) > 0.5).astype(int)

                    # Calculate metrics
                    train_metrics = {
                        "accuracy": accuracy_score(y_train, y_train_pred),
                        "f1_score": f1_score(y_train, y_train_pred),
                        "roc_auc": roc_auc_score(y_train, lgb_model.predict(X_train)),
                    }

                    valid_metrics = {
                        "accuracy": accuracy_score(y_valid, y_valid_pred),
                        "f1_score": f1_score(y_valid, y_valid_pred),
                        "roc_auc": roc_auc_score(y_valid, lgb_model.predict(X_valid)),
                    }

                    test_metrics = {
                        "accuracy": accuracy_score(y_test, y_test_pred),
                        "f1_score": f1_score(y_test, y_test_pred),
                        "roc_auc": roc_auc_score(y_test, lgb_model.predict(X_test)),
                    }

                    # Log metrics
                    for metric_name, value in train_metrics.items():
                        mlflow.log_metric(f"train_{metric_name}", value)

                    for metric_name, value in valid_metrics.items():
                        mlflow.log_metric(f"valid_{metric_name}", value)

                    for metric_name, value in test_metrics.items():
                        mlflow.log_metric(f"test_{metric_name}", value)

                    # Append metrics for reporting
                    metrics["accuracy"].append([train_metrics["accuracy"], valid_metrics["accuracy"], test_metrics["accuracy"]])
                    metrics["f1_score"].append([train_metrics["f1_score"], valid_metrics["f1_score"], test_metrics["f1_score"]])
                    metrics["roc_auc"].append([train_metrics["roc_auc"], valid_metrics["roc_auc"], test_metrics["roc_auc"]])

                    logging.info(f"Fold {fold+1} metrics logged.")

        return metrics

    except Exception as e:
        logging.error(f"An error occurred during the training pipeline: {e}", exc_info=True)
        raise


# Main Script
if __name__ == "__main__":
    setup_logging()
    logging.info("Starting training pipeline with MLflow.")
    try:
        metrics = run_lightgbm_training_pipeline(
            data_source="data/silver_layer/SupplyChain_Dataset.parquet",
            target_column="Disruption"
        )
        logging.info("Training pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Pipeline execution failed: {e}")
