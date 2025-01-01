import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import logging
from mlflow.models.signature import infer_signature
from datetime import datetime
from pipelines.utils import apply_pca

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


def run_inventory_training_pipeline(
    data_source,
    target_column,
    n_splits=3,
    mlflow_experiment_name="LightGBM_Inventory_Regression_Model",
    model_params=None,
    num_boost_round=100,
    early_stopping_rounds=10,
    pca_components=0.95
):
    """
    Train LightGBM regression models with TimeSeriesSplit and log metrics using MLflow.

    Parameters:
    - data_source (str): Path to the input dataset file.
    - target_column (str): The target column for the regression model.
    - n_splits (int): Number of splits for TimeSeriesSplit.
    - mlflow_experiment_name (str): Name of the MLflow experiment to log results.
    - model_params (dict): LightGBM model parameters.
    - num_boost_round (int): Number of boosting rounds for LightGBM.
    - early_stopping_rounds (int): Early stopping rounds for LightGBM.
    - pca_components (float): Number of PCA components to retain or explained variance.

    Returns:
    dict: Metrics for each split (RMSE, MAE, R²).
    """
    try:
        setup_logging()
        logging.info(f"Loading dataset from {data_source}.")

        # Load dataset
        data = pd.read_parquet(data_source)
        logging.info(f"Dataset loaded successfully. Shape: {data.shape}")

        # Ensure the target column exists
        if target_column not in data.columns:
            raise KeyError(f"Target column '{target_column}' not found in the dataset.")

        # Extract features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Apply PCA to numeric columns
        numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
        X = apply_pca(X, numeric_columns, pca_components)

        # Identify PCA columns
        pca_columns = [col for col in X.columns if col.startswith("PCA_Component_")]

        # Use PCA-transformed features
        X = X[pca_columns].values

        # Set MLflow experiment
        mlflow.set_experiment(mlflow_experiment_name)

        # Initialize TimeSeriesSplit
        ts_splits = TimeSeriesSplit(n_splits=n_splits)

        metrics = {
            "rmse": [],
            "mae": [],
            "r2": []
        }

        # Start parent MLflow run
        with mlflow.start_run(run_name=f"LightGBM_Training_Pipeline_{datetime.now()}") as parent_run:
            mlflow.log_artifact(data_source, artifact_path="datasets")

            # Iterate through splits
            for fold, (train_index, test_index) in enumerate(ts_splits.split(X)):
                train_size = int(len(train_index) * 0.8)
                valid_index = train_index[train_size:]
                train_index = train_index[:train_size]

                X_train, X_valid, X_test = X[train_index], X[valid_index], X[test_index]
                y_train, y_valid, y_test = y.iloc[train_index], y.iloc[valid_index], y.iloc[test_index]

                # Create LightGBM datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_valid, label=y_valid)

                # Default LightGBM parameters if none are provided
                if model_params is None:
                    model_params = {
                        "objective": "regression",
                        "metric": "rmse",
                        "boosting_type": "gbdt",
                        "num_leaves": 31,
                        "learning_rate": 0.01,
                        "feature_fraction": 0.9,
                        "random_state": 42
                    }

                with mlflow.start_run(run_name=f"Fold_{fold+1}", nested=True):
                    mlflow.log_params(model_params)

                    # Train the LightGBM model
                    lgb_model = lgb.train(
                        model_params,
                        train_data,
                        num_boost_round=num_boost_round,
                        valid_sets=[valid_data],
                        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)],
                    )

                    # Log model
                    input_example = pd.DataFrame(X_train[:1], columns=[f"PCA_Component_{i+1}" for i in range(X_train.shape[1])])
                    signature = infer_signature(input_example, lgb_model.predict(X_train[:1]))
                    mlflow.lightgbm.log_model(
                        lgb_model,
                        artifact_path="model",
                        input_example=input_example,
                        signature=signature
                    )

                    # Predict on train, validation, and test sets
                    y_train_pred = lgb_model.predict(X_train)
                    y_valid_pred = lgb_model.predict(X_valid)
                    y_test_pred = lgb_model.predict(X_test)

                    # Calculate metrics
                    train_metrics = {
                        "rmse": mean_squared_error(y_train, y_train_pred),
                        "mae": mean_absolute_error(y_train, y_train_pred),
                        "r2": r2_score(y_train, y_train_pred)
                    }

                    valid_metrics = {
                        "rmse": mean_squared_error(y_valid, y_valid_pred),
                        "mae": mean_absolute_error(y_valid, y_valid_pred),
                        "r2": r2_score(y_valid, y_valid_pred)
                    }

                    test_metrics = {
                        "rmse": mean_squared_error(y_test, y_test_pred),
                        "mae": mean_absolute_error(y_test, y_test_pred),
                        "r2": r2_score(y_test, y_test_pred)
                    }

                    # Log metrics
                    for metric_name, value in train_metrics.items():
                        mlflow.log_metric(f"train_{metric_name}", value)

                    for metric_name, value in valid_metrics.items():
                        mlflow.log_metric(f"valid_{metric_name}", value)

                    for metric_name, value in test_metrics.items():
                        mlflow.log_metric(f"test_{metric_name}", value)

                    # Append metrics for final reporting
                    metrics["rmse"].append([train_metrics["rmse"], valid_metrics["rmse"], test_metrics["rmse"]])
                    metrics["mae"].append([train_metrics["mae"], valid_metrics["mae"], test_metrics["mae"]])
                    metrics["r2"].append([train_metrics["r2"], valid_metrics["r2"], test_metrics["r2"]])

                    logging.info(f"Fold {fold+1} metrics: {test_metrics}")

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
