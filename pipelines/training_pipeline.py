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


# def run_inventory_training_pipeline(
#     data_source,
#     target_column,
#     n_splits=3,
#     mlflow_experiment_name="LightGBM_Inventory_Model1",
#     model_params=None,
#     num_boost_round=100,
#     early_stopping_rounds=10,
#     pca_components=0.95
# ):
#     try:
#         # Default LightGBM parameters
#         if model_params is None:
#             model_params = {
#                 "objective": "regression",
#                 "metric": "rmse",
#                 "boosting_type": "gbdt",
#                 "num_leaves": 31,
#                 "learning_rate": 0.01,
#                 "feature_fraction": 0.9,
#                 "random_state": 42
#             }

#         logging.info(f"Loading dataset from {data_source}.")

#         # Load the dataset
#         data = pd.read_parquet(data_source)
#         logging.info(f"Dataset loaded successfully. Columns: {data.columns.tolist()}\nShape: {data.shape}")

#         # Ensure the target column exists
#         if target_column not in data.columns:
#             raise KeyError(f"Target column '{target_column}' not found in the dataset.")

#         # Extract features and target
#         X = data.drop(columns=[target_column])
#         y = data[target_column]

#         # Exclude non-numeric columns
#         numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

#         # Apply PCA and extract transformed features
#         data_with_pca = apply_pca(X, numeric_columns, pca_components)

#         # Identify PCA columns
#         pca_columns = [col for col in data_with_pca.columns if col.startswith("PCA_Component_")]

#         # Use only PCA-transformed features for modeling
#         X_pca = data_with_pca[pca_columns].values

#         # Initialize MLflow experiment
#         mlflow.set_experiment(mlflow_experiment_name)

#         # Initialize TimeSeriesSplit
#         ts_splits = TimeSeriesSplit(n_splits=n_splits)

#         metrics = {"rmse": [], "mae": [], "r2": []}

#         # Start parent MLflow run
#         with mlflow.start_run(run_name=f"Training_Pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as parent_run:
#             mlflow.log_artifact(data_source, artifact_path="datasets")

#             # Iterate through splits
#             for fold, (train_idx, test_idx) in enumerate(ts_splits.split(X_pca)):
#                 train_size = int(len(train_idx) * 0.8)
#                 valid_idx = train_idx[train_size:]
#                 train_idx = train_idx[:train_size]

#                 # Split data
#                 X_train, X_valid, X_test = (
#                     X_pca[train_idx], X_pca[valid_idx], X_pca[test_idx]
#                 )
#                 y_train, y_valid, y_test = (
#                     y.iloc[train_idx], y.iloc[valid_idx], y.iloc[test_idx]
#                 )

#                 # Create LightGBM datasets
#                 train_data = lgb.Dataset(X_train, label=y_train)
#                 valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

#                 # Train the LightGBM model
#                 with mlflow.start_run(run_name=f"Fold_{fold + 1}", nested=True):
#                     mlflow.log_params(model_params)

#                     lgb_model = lgb.train(
#                         model_params,
#                         train_data,
#                         num_boost_round=num_boost_round,
#                         valid_sets=[valid_data],
#                         callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)]
#                     )

#                     # Log the trained model
#                     input_example = pd.DataFrame(X_train[:1], columns=[f"PCA_Component_{i+1}" for i in range(X_train.shape[1])])
#                     signature = infer_signature(input_example, lgb_model.predict(X_train[:1]))
#                     mlflow.lightgbm.log_model(
#                         lgb_model,
#                         artifact_path="model",
#                         input_example=input_example,
#                         signature=signature
#                     )

#                     # Evaluate the model
#                     y_train_pred = lgb_model.predict(X_train)
#                     y_valid_pred = lgb_model.predict(X_valid)
#                     y_test_pred = lgb_model.predict(X_test)

#                     fold_metrics = {
#                         "train": {
#                             "rmse": mean_squared_error(y_train, y_train_pred),
#                             "mae": mean_absolute_error(y_train, y_train_pred),
#                             "r2": r2_score(y_train, y_train_pred)
#                         },
#                         "valid": {
#                             "rmse": mean_squared_error(y_valid, y_valid_pred),
#                             "mae": mean_absolute_error(y_valid, y_valid_pred),
#                             "r2": r2_score(y_valid, y_valid_pred)
#                         },
#                         "test": {
#                             "rmse": mean_squared_error(y_test, y_test_pred),
#                             "mae": mean_absolute_error(y_test, y_test_pred),
#                             "r2": r2_score(y_test, y_test_pred)
#                         }
#                     }

#                     # Log metrics to MLflow
#                     for stage, stage_metrics in fold_metrics.items():
#                         for metric_name, metric_value in stage_metrics.items():
#                             mlflow.log_metric(f"{stage}_{metric_name}", metric_value)

#                     # Append test metrics for final reporting
#                     metrics["rmse"].append(fold_metrics["test"]["rmse"])
#                     metrics["mae"].append(fold_metrics["test"]["mae"])
#                     metrics["r2"].append(fold_metrics["test"]["r2"])

#                     logging.info(f"Fold {fold + 1} metrics: {fold_metrics}")

#         return metrics
#     except Exception as e:
#         logging.error(f"An error occurred during the training pipeline: {e}", exc_info=True)
#         raise


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
        metrics_1 = run_lightgbm_training_pipeline(
            data_source="data/silver_layer/SupplyChain_Dataset.parquet",
            target_column="Disruption"
        )
        metrics_2=run_inventory_training_pipeline(
            data_source="data/golden_layer/Feature_Engineered_Inventory_Management_Dataset.csv",
            target_column="Historical_Demand"
        )
        logging.info("Training pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Pipeline execution failed: {e}")
