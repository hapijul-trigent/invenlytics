import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import logging
from mlflow.models.signature import infer_signature
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sktime.classification.deep_learning import InceptionTimeClassifier
from sktime.datatypes import check_raise
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


def run_lightgbm_training_pipeline(data_source, target_column, n_splits=3, mlflow_experiment_name="Disruption_Model"):
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
                    "num_leaves": 50,
                    "learning_rate": 0.001,
                    "feature_fraction": 0.9
                }

                # MLflow Logging
                with mlflow.start_run(run_name=f"Fold_{fold+1}", nested=True):
                    mlflow.log_params(params)

                    # Train LightGBM model
                    lgb_model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=200,
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

            # Log Average Metrices
            
            # mlflow.log_metrics(
            #     {
            #         'test_accuracy': 
            #     }
            # )
        return metrics

    except Exception as e:
        logging.error(f"An error occurred during the training pipeline: {e}", exc_info=True)
        raise





import mlflow
import pandas as pd
from sktime.classification.deep_learning import InceptionTimeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sktime.datatypes import check_raise
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import logging


def to_sktime_nested_format(X):
    """
    Convert a 2D numpy array to sktime nested format.
    Each row in the input corresponds to one time series (univariate).
    """
    return pd.DataFrame({i: [pd.Series(row)] for i, row in enumerate(X)}).T


def run_inceptiontime_training_pipeline(data_source, target_column, n_splits=3, mlflow_experiment_name="InceptionTime_Disruption_Model"):
    """
    Train InceptionTimeClassifier models with TimeSeriesSplit, log metrics using MLflow.

    Parameters:
    data_source (str): Path to the input dataset file.
    target_column (str): The target column for the model.
    n_splits (int): Number of splits for TimeSeriesSplit.
    mlflow_experiment_name (str): Name of the MLflow experiment to log results.

    Returns:
    dict: Metrics for each split (accuracy, F1 score, ROC AUC score).
    """
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logging.info(f"Loading dataset from {data_source}.")
        
        # Load the dataset
        data = pd.read_parquet(data_source)
        logging.info(f"Dataset loaded successfully. Shape: {data.shape}")
        
        # Features and target
        X = data.drop(columns=[target_column, "Disruption_Type"])
        y = data[target_column]

        # Initialize OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Initialize TimeSeriesSplit
        ts_splits = TimeSeriesSplit(n_splits=n_splits)

        # Initialize metrics container
        metrics = {
            "accuracy": [],
            "f1_score": [],
            "roc_auc": []
        }

        # Set MLflow experiment
        mlflow.set_experiment(mlflow_experiment_name)

        # Check for an active run
        if mlflow.active_run() is None:
            parent_run = mlflow.start_run(run_name=f"InceptionTime_Training_Pipeline_{datetime.now()}")
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

                # One-hot encode categorical features
                X_train_encoded = encoder.fit_transform(X_train[['Region', 'Delivery_Mode']])
                X_valid_encoded = encoder.transform(X_valid[['Region', 'Delivery_Mode']])
                X_test_encoded = encoder.transform(X_test[['Region', 'Delivery_Mode']])

                # Get feature names after one-hot encoding
                encoded_feature_names = encoder.get_feature_names_out(['Region', 'Delivery_Mode'])

                # Create DataFrames for encoded features
                X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_feature_names, index=X_train.index)
                X_valid_encoded_df = pd.DataFrame(X_valid_encoded, columns=encoded_feature_names, index=X_valid.index)
                X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_feature_names, index=X_test.index)

                # Concatenate encoded features with original numerical features
                X_train = pd.concat([X_train.drop(columns=['Region', 'Delivery_Mode']), X_train_encoded_df], axis=1)
                X_valid = pd.concat([X_valid.drop(columns=['Region', 'Delivery_Mode']), X_valid_encoded_df], axis=1)
                X_test = pd.concat([X_test.drop(columns=['Region', 'Delivery_Mode']), X_test_encoded_df], axis=1)

                # Convert data to sktime-compatible format
                X_train = to_sktime_nested_format(X_train.values)
                X_valid = to_sktime_nested_format(X_valid.values)
                X_test = to_sktime_nested_format(X_test.values)

                # Ensure compatibility
                check_raise(X_train, mtype="nested_univ")

                # Initialize InceptionTime model
                model = InceptionTimeClassifier(
                    n_epochs=5, batch_size=64, kernel_size=40, n_filters=32,
                    use_residual=True, use_bottleneck=True, bottleneck_size=32, depth=6,
                    random_state=None, verbose=True, loss='categorical_crossentropy'
                )

                # Train model for the current fold
                with mlflow.start_run(run_name=f"Fold_{fold+1}", nested=True):
                    model.fit(X_train, y_train)

                    # Predict on train, validation, and test sets
                    y_train_pred = model.predict(X_train)
                    y_valid_pred = model.predict(X_valid)
                    y_test_pred = model.predict(X_test)

                    # Compute metrics
                    train_accuracy = accuracy_score(y_train, y_train_pred)
                    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
                    test_accuracy = accuracy_score(y_test, y_test_pred)

                    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                    valid_f1 = f1_score(y_valid, y_valid_pred, average='weighted')
                    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

                    train_roc_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1], multi_class='ovr')
                    valid_roc_auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1], multi_class='ovr')
                    test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], multi_class='ovr')

                    # Log metrics
                    mlflow.log_metric(f"train_accuracy", train_accuracy)
                    mlflow.log_metric(f"valid_accuracy", valid_accuracy)
                    mlflow.log_metric(f"test_accuracy", test_accuracy)

                    mlflow.log_metric(f"train_f1_score", train_f1)
                    mlflow.log_metric(f"valid_f1_score", valid_f1)
                    mlflow.log_metric(f"test_f1_score", test_f1)

                    mlflow.log_metric(f"train_roc_auc", train_roc_auc)
                    mlflow.log_metric(f"valid_roc_auc", valid_roc_auc)
                    mlflow.log_metric(f"test_roc_auc", test_roc_auc)

                    # Append metrics for reporting
                    metrics["accuracy"].append([train_accuracy, valid_accuracy, test_accuracy])
                    metrics["f1_score"].append([train_f1, valid_f1, test_f1])
                    metrics["roc_auc"].append([train_roc_auc, valid_roc_auc, test_roc_auc])

                    logging.info(f"Fold {fold+1} metrics logged.")

        return metrics

    except Exception as e:
        logging.error(f"An error occurred during the training pipeline: {e}", exc_info=True)
        raise


def run_xgboost_training_pipeline(data_source, target_column, n_splits=3, mlflow_experiment_name="Disruption_Model"):
    """
    Train XGBoost models with TimeSeriesSplit, log metrics using MLflow.
    
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

        if mlflow.active_run() is None:
            parent_run = mlflow.start_run(run_name=f"XGBoost_Training_Pipeline_{datetime.now()}")
        else:
            parent_run = mlflow.active_run()

        with parent_run:
            mlflow.log_artifact(data_source, artifact_path="datasets")

            for fold, (train_index, test_index) in enumerate(ts_splits.split(X)):
                train_size = int(len(train_index) * 0.8)
                valid_index = train_index[train_size:]
                train_index = train_index[:train_size]

                X_train, X_valid, X_test = X.iloc[train_index], X.iloc[valid_index], X.iloc[test_index]
                y_train, y_valid, y_test = y.iloc[train_index], y.iloc[valid_index], y.iloc[test_index]

                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "learning_rate": 0.001,
                    "max_depth": 9,
                    "subsample": 0.8
                }

                with mlflow.start_run(run_name=f"Fold_{fold+1}", nested=True):
                    mlflow.log_params(params)

                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dvalid = xgb.DMatrix(X_valid, label=y_valid)

                    xgb_model = xgb.train(
                        params,
                        dtrain,
                        num_boost_round=200,
                        evals=[(dvalid, 'validation')],
                        early_stopping_rounds=10
                    )

                    input_example = X_train.head(1)
                    signature = infer_signature(X_train, xgb_model.predict(xgb.DMatrix(X_train.head(1))))
                    mlflow.xgboost.log_model(
                        xgb_model,
                        artifact_path="model",
                        input_example=input_example,
                        signature=signature
                    )

                    y_train_pred = (xgb_model.predict(xgb.DMatrix(X_train)) > 0.5).astype(int)
                    y_valid_pred = (xgb_model.predict(xgb.DMatrix(X_valid)) > 0.5).astype(int)
                    y_test_pred = (xgb_model.predict(xgb.DMatrix(X_test)) > 0.5).astype(int)

                    train_metrics = {
                        "accuracy": accuracy_score(y_train, y_train_pred),
                        "f1_score": f1_score(y_train, y_train_pred),
                        "roc_auc": roc_auc_score(y_train, xgb_model.predict(xgb.DMatrix(X_train))),
                    }

                    valid_metrics = {
                        "accuracy": accuracy_score(y_valid, y_valid_pred),
                        "f1_score": f1_score(y_valid, y_valid_pred),
                        "roc_auc": roc_auc_score(y_valid, xgb_model.predict(xgb.DMatrix(X_valid))),
                    }

                    test_metrics = {
                        "accuracy": accuracy_score(y_test, y_test_pred),
                        "f1_score": f1_score(y_test, y_test_pred),
                        "roc_auc": roc_auc_score(y_test, xgb_model.predict(xgb.DMatrix(X_test))),
                    }

                    for metric_name, value in train_metrics.items():
                        mlflow.log_metric(f"train_{metric_name}", value)

                    for metric_name, value in valid_metrics.items():
                        mlflow.log_metric(f"valid_{metric_name}", value)

                    for metric_name, value in test_metrics.items():
                        mlflow.log_metric(f"test_{metric_name}", value)

                    metrics["accuracy"].append([train_metrics["accuracy"], valid_metrics["accuracy"], test_metrics["accuracy"]])
                    metrics["f1_score"].append([train_metrics["f1_score"], valid_metrics["f1_score"], test_metrics["f1_score"]])
                    metrics["roc_auc"].append([train_metrics["roc_auc"], valid_metrics["roc_auc"], test_metrics["roc_auc"]])

                    logging.info(f"Fold {fold+1} metrics logged.")

        return metrics

    except Exception as e:
        logging.error(f"An error occurred during the training pipeline: {e}", exc_info=True)
        raise


def run_randomforest_training_pipeline(data_source, target_column, n_splits=3, mlflow_experiment_name="Disruption_Model"):
    """
    Train RandomForest models with TimeSeriesSplit, log metrics using MLflow.
    
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

        if mlflow.active_run() is None:
            parent_run = mlflow.start_run(run_name=f"RandomForest_Training_Pipeline_{datetime.now()}")
        else:
            parent_run = mlflow.active_run()

        with parent_run:
            mlflow.log_artifact(data_source, artifact_path="datasets")

            for fold, (train_index, test_index) in enumerate(ts_splits.split(X)):
                train_size = int(len(train_index) * 0.8)
                valid_index = train_index[train_size:]
                train_index = train_index[:train_size]

                X_train, X_valid, X_test = X.iloc[train_index], X.iloc[valid_index], X.iloc[test_index]
                y_train, y_valid, y_test = y.iloc[train_index], y.iloc[valid_index], y.iloc[test_index]

                with mlflow.start_run(run_name=f"Fold_{fold+1}", nested=True):
                    rf_model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
                    rf_model.fit(X_train, y_train)

                    input_example = X_train.head(1)
                    signature = infer_signature(X_train, rf_model.predict(X_train.head(1)))
                    mlflow.sklearn.log_model(
                        rf_model,
                        artifact_path="model",
                        input_example=input_example,
                        signature=signature
                    )

                    y_train_pred = rf_model.predict(X_train)
                    y_valid_pred = rf_model.predict(X_valid)
                    y_test_pred = rf_model.predict(X_test)

                    train_metrics = {
                        "accuracy": accuracy_score(y_train, y_train_pred),
                        "f1_score": f1_score(y_train, y_train_pred),
                        "roc_auc": roc_auc_score(y_train, rf_model.predict_proba(X_train)[:, 1]),
                    }

                    valid_metrics = {
                        "accuracy": accuracy_score(y_valid, y_valid_pred),
                        "f1_score": f1_score(y_valid, y_valid_pred),
                        "roc_auc": roc_auc_score(y_valid, rf_model.predict_proba(X_valid)[:, 1]),
                    }

                    test_metrics = {
                        "accuracy": accuracy_score(y_test, y_test_pred),
                        "f1_score": f1_score(y_test, y_test_pred),
                        "roc_auc": roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]),
                    }

                    for metric_name, value in train_metrics.items():
                        mlflow.log_metric(f"train_{metric_name}", value)

                    for metric_name, value in valid_metrics.items():
                        mlflow.log_metric(f"valid_{metric_name}", value)

                    for metric_name, value in test_metrics.items():
                        mlflow.log_metric(f"test_{metric_name}", value)

                    metrics["accuracy"].append([train_metrics["accuracy"], valid_metrics["accuracy"], test_metrics["accuracy"]])
                    metrics["f1_score"].append([train_metrics["f1_score"], valid_metrics["f1_score"], test_metrics["f1_score"]])
                    metrics["roc_auc"].append([train_metrics["roc_auc"], valid_metrics["roc_auc"], test_metrics["roc_auc"]])

                    logging.info(f"Fold {fold+1} metrics logged.")

        return metrics

    except Exception as e:
        logging.error(f"An error occurred during the training pipeline: {e}", exc_info=True)
        raise



def run_inventory_training_pipeline(data_source, target_column, n_splits=3, mlflow_experiment_name="LightGBM_Inventory_Model"):
    """
    Train LightGBM models with TimeSeriesSplit and log metrics using MLflow.

    Parameters:
    data_source (str): Path to the input dataset file.
    target_column (str): The target column for the model.
    n_splits (int): Number of splits for TimeSeriesSplit.
    mlflow_experiment_name (str): Name of the MLflow experiment to log results.

    Returns:
    dict: Metrics for each split (RMSE, MAE, R2 score).
    """
    try:
        setup_logging()
        logging.info(f"Loading dataset from {data_source}.")

        # Load the dataset
        data = pd.read_parquet(data_source)
        data.set_index('Scheduled_Delivery', inplace=True)
        logging.info(f"Dataset loaded successfully. Shape: {data.shape}")

        # Features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Set MLflow experiment
        mlflow.set_experiment(mlflow_experiment_name)

        # Initialize TimeSeriesSplit
        ts_splits = TimeSeriesSplit(n_splits=n_splits)

        metrics = {
            "rmse": [],
            "mae": [],
            "r2": []
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
                    "objective": "regression",
                    "metric": "rmse",
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
                    y_train_pred = lgb_model.predict(X_train)
                    y_valid_pred = lgb_model.predict(X_valid)
                    y_test_pred = lgb_model.predict(X_test)

                    # Calculate metrics
                    train_metrics = {
                        "rmse": mean_squared_error(y_train, y_train_pred, squared=False),
                        "mae": mean_absolute_error(y_train, y_train_pred),
                        "r2": r2_score(y_train, y_train_pred),
                    }

                    valid_metrics = {
                        "rmse": mean_squared_error(y_valid, y_valid_pred, squared=False),
                        "mae": mean_absolute_error(y_valid, y_valid_pred),
                        "r2": r2_score(y_valid, y_valid_pred),
                    }

                    test_metrics = {
                        "rmse": mean_squared_error(y_test, y_test_pred, squared=False),
                        "mae": mean_absolute_error(y_test, y_test_pred),
                        "r2": r2_score(y_test, y_test_pred),
                    }

                    # Log metrics
                    for metric_name, value in train_metrics.items():
                        mlflow.log_metric(f"train_{metric_name}", value)

                    for metric_name, value in valid_metrics.items():
                        mlflow.log_metric(f"valid_{metric_name}", value)

                    for metric_name, value in test_metrics.items():
                        mlflow.log_metric(f"test_{metric_name}", value)

                    # Append metrics for reporting
                    metrics["rmse"].append([train_metrics["rmse"], valid_metrics["rmse"], test_metrics["rmse"]])
                    metrics["mae"].append([train_metrics["mae"], valid_metrics["mae"], test_metrics["mae"]])
                    metrics["r2"].append([train_metrics["r2"], valid_metrics["r2"], test_metrics["r2"]])

                    logging.info(f"Fold {fold+1} metrics logged.")

        return metrics

    except Exception as e:
        logging.error(f"An error occurred during the training pipeline: {e}", exc_info=True)
        raise

# Main Script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting training pipeline with MLflow.")
    try:
        metrics = run_inceptiontime_training_pipeline(
            data_source="data/gold_layer/SupplyChainI_Disruption_Dataset.parquet",
            target_column="Disruption"
        )
        logging.info("Training pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Pipeline execution failed: {e}")
