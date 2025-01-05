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



import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, output_path):
    """Plot and save a precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_auc_curve(y_true, y_scores, output_path):
    """Plot and save a ROC-AUC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc_score(y_true, y_scores)))
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_lightgbm_training_pipeline(data_source, target_column, n_splits=3, mlflow_experiment_name="Disruption_Model"):
    """
    Train LightGBM models with TimeSeriesSplit and log metrics and plots using MLflow.
    """
    try:
        setup_logging()
        logging.info(f"Loading dataset from {data_source}.")
        data = pd.read_parquet(data_source)
        data = data[data.select_dtypes(exclude=['datetime64', 'object']).columns]

        # Features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        mlflow.set_experiment(mlflow_experiment_name)
        ts_splits = TimeSeriesSplit(n_splits=n_splits)
        metrics = {"accuracy": [], "f1_score": [], "roc_auc": []}

        parent_run = mlflow.start_run(run_name=f"LightGBM_Pipeline_{datetime.now()}")
        with parent_run:
            mlflow.log_artifact(data_source, artifact_path="datasets")
            for fold, (train_idx, test_idx) in enumerate(ts_splits.split(X)):
                # Split the data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Create LightGBM datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                test_data = lgb.Dataset(X_test, label=y_test)

                # LightGBM parameters
                params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "boosting_type": "gbdt",
                    "num_leaves": 50,
                    "learning_rate": 0.01,
                    "feature_fraction": 0.9,
                }

                # Train model
                with mlflow.start_run(run_name=f"Fold_{fold+1}", nested=True):
                    mlflow.log_params(params)

                    lgb_model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=200,
                        valid_sets=[test_data],
                        callbacks=[lgb.early_stopping(stopping_rounds=10)],
                    )

                    # Log model
                    input_example = X_train.head(1)
                    signature = infer_signature(X_train, lgb_model.predict(X_train.head(1)))
                    mlflow.lightgbm.log_model(
                        lgb_model,
                        artifact_path="model",
                        input_example=input_example,
                        signature=signature,
                    )

                    # Predictions
                    y_train_pred = (lgb_model.predict(X_train) > 0.5).astype(int)
                    y_test_pred = (lgb_model.predict(X_test) > 0.5).astype(int)
                    y_test_scores = lgb_model.predict(X_test)

                    # Compute metrics
                    fold_metrics = {
                        "accuracy": [
                            accuracy_score(y_train, y_train_pred),
                            accuracy_score(y_test, y_test_pred),
                        ],
                        "f1_score": [
                            f1_score(y_train, y_train_pred),
                            f1_score(y_test, y_test_pred),
                        ],
                        "roc_auc": [
                            roc_auc_score(y_train, lgb_model.predict(X_train)),
                            roc_auc_score(y_test, y_test_scores),
                        ],
                    }

                    for metric_name, values in fold_metrics.items():
                        mlflow.log_metric(f"train_{metric_name}", values[0])
                        mlflow.log_metric(f"test_{metric_name}", values[1])

                    for metric_name, values in fold_metrics.items():
                        metrics[metric_name].append(values)

                    # Log plots as artifacts
                    artifacts_dir = f"artifacts_fold_{fold+1}"
                    os.makedirs(artifacts_dir, exist_ok=True)

                    cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
                    plot_confusion_matrix(y_test, y_test_pred, cm_path)
                    mlflow.log_artifact(cm_path, artifact_path="plots")

                    prc_path = os.path.join(artifacts_dir, "precision_recall_curve.png")
                    plot_precision_recall_curve(y_test, y_test_scores, prc_path)
                    mlflow.log_artifact(prc_path, artifact_path="plots")

                    roc_path = os.path.join(artifacts_dir, "roc_auc_curve.png")
                    plot_roc_auc_curve(y_test, y_test_scores, roc_path)
                    mlflow.log_artifact(roc_path, artifact_path="plots")

                    logging.info(f"Fold {fold+1} metrics: {fold_metrics}")

            # Log overall metrics
            for metric_name, values in metrics.items():
                avg_train = sum(v[0] for v in values) / n_splits
                avg_test = sum(v[1] for v in values) / n_splits

                mlflow.log_metric(f"avg_train_{metric_name}", avg_train)
                mlflow.log_metric(f"avg_test_{metric_name}", avg_test)

        return metrics

    except Exception as e:
        logging.error(f"An error occurred in the training pipeline: {e}", exc_info=True)
        raise



def run_xgboost_training_pipeline(data_source, target_column, n_splits=3, mlflow_experiment_name="Disruption_Model"):
    """
    Train XGBoost models with TimeSeriesSplit and log metrics and plots using MLflow.
    """
    try:
        setup_logging()
        logging.info(f"Loading dataset from {data_source}.")
        data = pd.read_parquet(data_source)
        data = data[data.select_dtypes(exclude=['datetime64', 'object']).columns]

        # Features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        mlflow.set_experiment(mlflow_experiment_name)
        ts_splits = TimeSeriesSplit(n_splits=n_splits)
        metrics = {"accuracy": [], "f1_score": [], "roc_auc": []}

        parent_run = mlflow.start_run(run_name=f"XGBoost_Pipeline_{datetime.now()}")
        with parent_run:
            mlflow.log_artifact(data_source, artifact_path="datasets")
            for fold, (train_idx, test_idx) in enumerate(ts_splits.split(X)):
                # Split the data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Create DMatrix for XGBoost
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)

                # XGBoost parameters
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "booster": "gbtree",
                    "eta": 0.01,
                    "max_depth": 6,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "seed": 42,
                }

                # Train model
                with mlflow.start_run(run_name=f"Fold_{fold+1}", nested=True):
                    mlflow.log_params(params)

                    xgb_model = xgb.train(
                        params,
                        dtrain,
                        num_boost_round=200,
                        evals=[(dtest, "eval")],
                        early_stopping_rounds=10,
                    )

                    # Log model
                    input_example = X_train.head(1)
                    signature = infer_signature(X_train, xgb_model.predict(dtrain))
                    mlflow.xgboost.log_model(
                        xgb_model,
                        artifact_path="model",
                        input_example=input_example,
                        signature=signature,
                    )

                    # Predictions
                    y_train_pred = (xgb_model.predict(dtrain) > 0.5).astype(int)
                    y_test_pred = (xgb_model.predict(dtest) > 0.5).astype(int)
                    y_test_scores = xgb_model.predict(dtest)

                    # Compute metrics
                    fold_metrics = {
                        "accuracy": [
                            accuracy_score(y_train, y_train_pred),
                            accuracy_score(y_test, y_test_pred),
                        ],
                        "f1_score": [
                            f1_score(y_train, y_train_pred),
                            f1_score(y_test, y_test_pred),
                        ],
                        "roc_auc": [
                            roc_auc_score(y_train, xgb_model.predict(dtrain)),
                            roc_auc_score(y_test, y_test_scores),
                        ],
                    }

                    for metric_name, values in fold_metrics.items():
                        mlflow.log_metric(f"train_{metric_name}", values[0])
                        mlflow.log_metric(f"test_{metric_name}", values[1])

                    for metric_name, values in fold_metrics.items():
                        metrics[metric_name].append(values)

                    # Log plots as artifacts
                    artifacts_dir = f"artifacts_fold_{fold+1}"
                    os.makedirs(artifacts_dir, exist_ok=True)

                    cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
                    plot_confusion_matrix(y_test, y_test_pred, cm_path)
                    mlflow.log_artifact(cm_path, artifact_path="plots")

                    prc_path = os.path.join(artifacts_dir, "precision_recall_curve.png")
                    plot_precision_recall_curve(y_test, y_test_scores, prc_path)
                    mlflow.log_artifact(prc_path, artifact_path="plots")

                    roc_path = os.path.join(artifacts_dir, "roc_auc_curve.png")
                    plot_roc_auc_curve(y_test, y_test_scores, roc_path)
                    mlflow.log_artifact(roc_path, artifact_path="plots")

                    logging.info(f"Fold {fold+1} metrics: {fold_metrics}")

            # Log overall metrics
            for metric_name, values in metrics.items():
                avg_train = sum(v[0] for v in values) / n_splits
                avg_test = sum(v[1] for v in values) / n_splits

                mlflow.log_metric(f"avg_train_{metric_name}", avg_train)
                mlflow.log_metric(f"avg_test_{metric_name}", avg_test)

        return metrics

    except Exception as e:
        logging.error(f"An error occurred in the training pipeline: {e}", exc_info=True)
        raise



# Main Script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting training pipeline with MLflow.")
    try:
        metrics = run_lightgbm_training_pipeline(
            data_source="/workspaces/invenlytics/data/gold_layer/SupplyChain_DisruptionFeatures.parquet",
            target_column="Disruption"
        )

        metrics = run_xgboost_training_pipeline(
            data_source="/workspaces/invenlytics/data/gold_layer/SupplyChain_DisruptionFeatures.parquet",
            target_column="Disruption"
        )
        logging.info("Training pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Pipeline execution failed: {e}")
