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
from pipelines.preprocessing_pipeline import preprocess_data
from sklearn.preprocessing import LabelEncoder

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


# def run_stock_risk_training_pipeline(
#     data_source,
#     target_column,
#     n_splits=3,
#     mlflow_experiment_name="Stock_Risk_Classification",
#     model_params=None,
#     num_boost_round=100,
#     early_stopping_rounds=10
# ):
#     """
#     Train a classification model for Stockout/Overstock Risk.

#     Parameters:
#     - data_source (str): Path to the input dataset file.
#     - target_column (str): The target column for classification.
#     - n_splits (int): Number of splits for TimeSeriesSplit.
#     - mlflow_experiment_name (str): MLflow experiment name.
#     - model_params (dict): LightGBM model parameters.
#     - num_boost_round (int): Number of boosting rounds.
#     - early_stopping_rounds (int): Early stopping rounds.

#     Returns:
#     - dict: Classification metrics for each fold.
#     """
#     try:
#         logger.info("Starting Stock Risk Training Pipeline.")

#         # Load dataset
#         data = pd.read_parquet(data_source)
#         logger.info(f"Loaded dataset with shape: {data.shape}")

#         # Split features and target
#         X = data.drop(columns=[target_column])
#         y = data[target_column]

#         # Initialize TimeSeriesSplit
#         tscv = TimeSeriesSplit(n_splits=n_splits)

#         metrics = []

#         # Set MLflow experiment
#         mlflow.set_experiment(mlflow_experiment_name)

#         with mlflow.start_run():
#             for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
#                 X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#                 y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#                 # LightGBM datasets
#                 train_data = lgb.Dataset(X_train, label=y_train)
#                 test_data = lgb.Dataset(X_test, label=y_test)

#                 # Default model parameters
#                 if model_params is None:
#                     model_params = {
#                         "objective": "binary",
#                         "metric": "auc",
#                         "boosting_type": "gbdt",
#                         "num_leaves": 31,
#                         "learning_rate": 0.01,
#                         "feature_fraction": 0.9,
#                         "random_state": 42
#                     }

#                 # Train LightGBM model
#                 model = lgb.train(
#                     model_params,
#                     train_data,
#                     num_boost_round=num_boost_round,
#                     valid_sets=[test_data],
#                     early_stopping_rounds=early_stopping_rounds
#                 )

#                 # Predict and evaluate
#                 y_pred = model.predict(X_test)
#                 y_pred_binary = (y_pred > 0.5).astype(int)

#                 fold_metrics = {
#                     "accuracy": accuracy_score(y_test, y_pred_binary),
#                     "f1_score": f1_score(y_test, y_pred_binary),
#                     "roc_auc": roc_auc_score(y_test, y_pred)
#                 }
#                 metrics.append(fold_metrics)

#                 logger.info(f"Fold {fold+1} Metrics: {fold_metrics}")

#             # Log overall metrics to MLflow
#             avg_metrics = {
#                 "avg_accuracy": sum(m["accuracy"] for m in metrics) / n_splits,
#                 "avg_f1_score": sum(m["f1_score"] for m in metrics) / n_splits,
#                 "avg_roc_auc": sum(m["roc_auc"] for m in metrics) / n_splits
#             }
#             mlflow.log_metrics(avg_metrics)

#         logger.info("Stock Risk Training Pipeline completed.")
#         return metrics

#     except Exception as e:
#         logger.error(f"Error during training pipeline: {e}")
#         raise

def run_stock_risk_training_pipeline(
    data_source,
    target_column="Overstock_Risk",
    n_splits=3,
    mlflow_experiment_name="Stock_Risk_Classification",
    model_params=None,
    num_boost_round=100,
    early_stopping_rounds=10
):
    try:
        logging.info("Starting Stock Risk Training Pipeline.")

        # Load dataset
        data = pd.read_parquet(data_source)
        logging.info(f"Loaded dataset with shape: {data.shape}")

        # Handle missing target column
        if target_column not in data.columns:
            logging.info(f"Target column '{target_column}' not found, attempting to derive it.")
            if 'Current_Stock' in data.columns and 'Reorder_Level' in data.columns:
                data[target_column] = (data['Current_Stock'] > data['Reorder_Level']).astype(int)
                logging.info(f"Derived '{target_column}' column successfully.")
            else:
                # Add placeholders for missing columns or stop execution
                if 'Current_Stock' not in data.columns:
                    data['Current_Stock'] = 0  # Placeholder value
                    logging.warning("'Current_Stock' not found. Placeholder column added with default value 0.")
                if 'Reorder_Level' not in data.columns:
                    data['Reorder_Level'] = 0  # Placeholder value
                    logging.warning("'Reorder_Level' not found. Placeholder column added with default value 0.")
                data[target_column] = (data['Current_Stock'] > data['Reorder_Level']).astype(int)
                logging.info(f"Derived '{target_column}' column with placeholder values.")

        # Encode categorical columns
        categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()
        if categorical_columns:
            logging.info(f"Encoding categorical columns: {categorical_columns}")
            for col in categorical_columns:
                data[col] = LabelEncoder().fit_transform(data[col])
            logging.info("Categorical columns encoded successfully.")

        # Feature engineering for datetime columns
        datetime_columns = data.select_dtypes(include=["datetime64"]).columns.tolist()
        if datetime_columns:
            logging.info(f"Processing datetime columns: {datetime_columns}")
            for col in datetime_columns:
                data[f"{col}_year"] = data[col].dt.year
                data[f"{col}_month"] = data[col].dt.month
                data[f"{col}_day"] = data[col].dt.day
                data[f"{col}_dayofweek"] = data[col].dt.dayofweek
                if col == "Scheduled_Delivery" and "Actual_Delivery" in data.columns:
                    data["Delivery_Delay_Days"] = (data["Actual_Delivery"] - data["Scheduled_Delivery"]).dt.days
            # Drop original datetime columns
            data.drop(columns=datetime_columns, inplace=True)
            logging.info("Datetime columns processed and removed.")

        # Split features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Initialize TimeSeriesSplit with 3 folds
        tscv = TimeSeriesSplit(n_splits=n_splits)

        metrics = []

        # Set MLflow experiment
        mlflow.set_experiment(mlflow_experiment_name)

        with mlflow.start_run():
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # LightGBM datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                test_data = lgb.Dataset(X_test, label=y_test)

                # Default model parameters
                if model_params is None:
                    model_params = {
                        "objective": "binary",
                        "metric": "auc",
                        "boosting_type": "gbdt",
                        "num_leaves": 31,
                        "learning_rate": 0.01,
                        "feature_fraction": 0.9,
                        "random_state": 42,
                        "verbosity": -1  # Suppresses verbose output
                    }

                # Train LightGBM model with early stopping
                model = lgb.train(
                    model_params,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[test_data],
                    callbacks=[lgb.early_stopping(early_stopping_rounds)]
                )

                # Predict and evaluate
                y_pred = model.predict(X_test)
                y_pred_binary = (y_pred > 0.5).astype(int)

                fold_metrics = {
                    "accuracy": accuracy_score(y_test, y_pred_binary),
                    "f1_score": f1_score(y_test, y_pred_binary),
                    "roc_auc": roc_auc_score(y_test, y_pred)
                }
                metrics.append(fold_metrics)

                logging.info(f"Fold {fold+1} Metrics: {fold_metrics}")

            # Log overall metrics to MLflow
            avg_metrics = {
                "avg_accuracy": sum(m["accuracy"] for m in metrics) / n_splits,
                "avg_f1_score": sum(m["f1_score"] for m in metrics) / n_splits,
                "avg_roc_auc": sum(m["roc_auc"] for m in metrics) / n_splits
            }
            mlflow.log_metrics(avg_metrics)

        logging.info("Stock Risk Training Pipeline completed.")
        return metrics

    except Exception as e:
        logging.error(f"Error during training pipeline: {e}")
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
