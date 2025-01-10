import pandas as pd
import numpy as np
import logging
import random
from datetime import timedelta, datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
import streamlit as st

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/main_pipeline.log"),
        logging.StreamHandler()
    ]
)

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

# ===========================
# Ingestion Pipeline
# ===========================
def generate_historical_demand(existing_df):
    """
    Generate a Historical_Demand column based on seasonal patterns and random noise.

    Parameters:
        existing_df (pd.DataFrame): The existing dataset.

    Returns:
        pd.DataFrame: The dataset with the Historical_Demand column added.
    """
    logging.info("Generating Historical_Demand column...")

    # Generate historical demand using seasonality and random noise
    num_rows = len(existing_df)
    seasonal_factor = (np.sin(np.linspace(0, 2 * np.pi, num_rows)) + 1) * 50  # Simulating seasonal patterns
    random_noise = np.random.normal(0, 5, num_rows)  # Adding some noise
    trend = np.linspace(100, 200, num_rows)  # Simulating a demand trend over time

    # Combine seasonality, noise, and trend to create historical demand
    historical_demand = seasonal_factor + random_noise + trend
    existing_df['Historical_Demand'] = np.maximum(historical_demand, 0).astype(int)  # Ensure non-negative demand

    logging.info("Historical_Demand column generated successfully.")
    return existing_df

def generate_additional_data(existing_df, total_rows_required=100000):
    """
    Add new columns and generate additional rows to reach the required total rows.

    Parameters:
        existing_df (pd.DataFrame): The existing dataset.
        total_rows_required (int): The total number of rows required.

    Returns:
        pd.DataFrame: A dataframe with new columns and rows.
    """
    logging.info("Adding new columns and generating additional rows...")

    # Number of additional rows needed
    additional_rows_needed = total_rows_required - len(existing_df)

    # New columns to be added
    suppliers = ['Alibaba', 'H&M', 'IKEA', 'Wrogn']
    delivery_modes = ['Air', 'Sea', 'Road', 'Rail']
    disruption_types = ['Weather', 'Supplier Issue', 'Logistics', 'Geopolitical', 'None']
    regions = ['North America', 'Europe', 'Asia', 'South America']
    weather_conditions = ['Sunny', 'Rainy', 'Snowy', 'Cloudy']

    # Add new columns for existing rows if they don't exist
    if 'Supplier' not in existing_df.columns:
        existing_df['Supplier'] = np.random.choice(suppliers, len(existing_df))
    if 'Region' not in existing_df.columns:
        existing_df['Region'] = np.random.choice(regions, len(existing_df))
    if 'Delivery_Mode' not in existing_df.columns:
        existing_df['Delivery_Mode'] = np.random.choice(delivery_modes, len(existing_df))
    if 'Disruption_Type' not in existing_df.columns:
        existing_df['Disruption_Type'] = np.random.choice(disruption_types, len(existing_df))
    if 'Weather_Conditions' not in existing_df.columns:
        existing_df['Weather_Conditions'] = np.random.choice(weather_conditions, len(existing_df))
    if 'Scheduled_Delivery' not in existing_df.columns:
        existing_df['Scheduled_Delivery'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(1, 10, len(existing_df)), unit='D')
    if 'Actual_Delivery' not in existing_df.columns:
        existing_df['Actual_Delivery'] = existing_df['Scheduled_Delivery'] + pd.to_timedelta(np.random.randint(-5, 15, len(existing_df)), unit='D')
    if 'Freight_Cost' not in existing_df.columns:
        existing_df['Freight_Cost'] = np.round(np.random.uniform(100, 1000, len(existing_df)), 2)

    # Generate additional rows if needed
    synthetic_data = []
    if additional_rows_needed > 0:
        logging.info(f"Generating {additional_rows_needed} additional rows...")
        for i in range(additional_rows_needed):
            synthetic_data.append({
                "Supplier": np.random.choice(suppliers),
                "Region": np.random.choice(regions),
                "Delivery_Mode": np.random.choice(delivery_modes),
                "Disruption_Type": np.random.choice(disruption_types),
                "Weather_Conditions": np.random.choice(weather_conditions),
                "Scheduled_Delivery": pd.to_datetime('2024-01-01') + timedelta(days=random.randint(1, 10)),
                "Actual_Delivery": pd.to_datetime('2024-01-01') + timedelta(days=random.randint(5, 15)),
                "Freight_Cost": round(random.uniform(100, 1000), 2)
            })

    synthetic_df = pd.DataFrame(synthetic_data)
    existing_df = pd.concat([existing_df, synthetic_df], ignore_index=True)

    # Generate Historical Demand
    existing_df = generate_historical_demand(existing_df)

    logging.info("Additional rows and columns added successfully.")
    return existing_df

def run(input_file, output_file, total_rows=100000):
    """
    Main function to load existing data, augment it with new rows and columns, and save it as Parquet.

    Parameters:
        input_file (str): Path to the existing dataset (CSV format).
        output_file (str): Path to save the updated dataset (Parquet format).
        total_rows (int): Total number of rows required in the dataset.

    Returns:
        None
    """
    try:
        logging.info("Starting the data augmentation process.")

        # Load existing dataset
        existing_df = pd.read_csv(input_file)
        logging.info(f"Loaded existing dataset with {len(existing_df)} rows.")

        # Generate augmented dataset
        updated_df = generate_additional_data(existing_df, total_rows_required=total_rows)

        # Validate row count
        if len(updated_df) != total_rows:
            raise ValueError(f"Dataset row count mismatch: Expected {total_rows}, but got {len(updated_df)}")

        # Save to Parquet
        updated_df.to_parquet(output_file, index=False)
        updated_df.dropna(inplace=True)
        logging.info(f"Updated dataset saved to {output_file}")
        return updated_df

    except Exception as e:
        logging.error("An error occurred during the pipeline execution:", exc_info=True)

# ===========================
# Preprocessing Pipeline
# ===========================

def run(source, dest):
    """
    Validate, clean, and save the dataset.

    Parameters:
    source (str): Path to the source dataset file.
    dest (str): Path to save the cleaned dataset.

    Returns:
    pd.DataFrame: The cleaned dataset.
    """
    try:
        setup_logging()
        logging.info(f"Loading dataset from {source}.")

        # Load the dataset
        data = pd.read_parquet(source)
        logging.info("Dataset loaded successfully.")
        logging.info(f"Initial columns: {data.columns.tolist()}")

        # Handle missing values
        for col in data.columns:
            if data[col].isnull().any():
                if data[col].dtype == 'object':
                    data[col].fillna("Unknown", inplace=True)
                else:
                    data[col].fillna(data[col].median(), inplace=True)
        logging.info("Missing values handled.")

        # Convert 'timestamp' to datetime if it exists
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            logging.info("'timestamp' column converted to datetime.")
        else:
            logging.warning("'timestamp' column not found in the dataset.")

        # Convert necessary columns to datetime
        if "Scheduled_Delivery" in data.columns and "Actual_Delivery" in data.columns:
            data["Scheduled_Delivery"] = pd.to_datetime(data["Scheduled_Delivery"], errors='coerce')
            data["Actual_Delivery"] = pd.to_datetime(data["Actual_Delivery"], errors='coerce')
        else:
            logging.error("Missing Scheduled_Delivery or Actual_Delivery column.")
            return data
        logging.info("Converted required columns to datetime.")

        # Remove duplicates
        initial_rows = data.shape[0]
        data.drop_duplicates(inplace=True)
        logging.info(f"Duplicate rows removed. {initial_rows - data.shape[0]} rows dropped.")

        # Validate and clean specific columns
        if "Scheduled_Delivery" in data.columns:
            valid_rows = data["Scheduled_Delivery"].notnull().sum()
            data = data[data["Scheduled_Delivery"].notnull()]
            logging.info(f"Validated 'Scheduled_Delivery' column. {valid_rows} rows retained.")

        # Encode categorical variables
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col].astype(str))
        logging.info("Categorical variables encoded.")

        # Reset index if timestamp was used as index
        if "timestamp" in data.index.names:
            data.reset_index(inplace=True)
            logging.info("Reset the index to include the 'timestamp' column.")

        # Save the cleaned dataset
        data.to_parquet(dest, index=False)
        logging.info(f"Cleaned dataset saved to {dest}.")
        logging.info(data.info())

        return data

    except Exception as e:
        logging.error(f"An error occurred during validation and cleaning: {e}", exc_info=True)
        raise

def preprocess_data(data, categorical_columns):
    """
    Preprocess the dataset by encoding categorical columns.

    Parameters:
    - data (pd.DataFrame): Input dataset.
    - categorical_columns (list): List of categorical column names.

    Returns:
    - pd.DataFrame: Preprocessed dataset.
    """
    try:
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
        logging.info(f"Categorical columns {categorical_columns} encoded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise
        


# ===========================
# Feature Engineering Pipeline
# ===========================

def feature_engineering(source, dest):
    """
    Engineer features for inventory optimization.
    """
    try:
        logging.info(f"Loading dataset from {source}...")
        data = pd.read_parquet(source)
        logging.info(f"Dataset loaded. Shape: {data.shape}")

        data.fillna(method="ffill", inplace=True)
        datetime_columns = data.select_dtypes(include=["datetime64"]).columns
        for col in datetime_columns:
            data[f"{col}_year"] = data[col].dt.year
            data[f"{col}_month"] = data[col].dt.month
            data[f"{col}_day"] = data[col].dt.day
            data[f"{col}_dayofweek"] = data[col].dt.dayofweek
        data.drop(columns=datetime_columns, inplace=True)

        data.dropna(inplace=True)
        data.to_parquet(dest, index=False)
        logging.info(f"Feature-engineered data saved to {dest}.")
        return data
    except Exception as e:
        logging.error(f"Error in feature engineering: {e}")
        raise


# ===========================
# Training Pipeline (Unchanged)
# ===========================

def run_inventory_training_pipeline(
    data_source,
    target_column,
    feature_columns=None,  # Added parameter
    n_splits=5,
    mlflow_experiment_name="LightGBM_Inventory_Regression_Model",
    model_params=None,
    num_boost_round=200,
    early_stopping_rounds=20
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

        # Identify numeric columns
        numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

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
            
                X_train, X_valid, X_test = X.iloc[train_index], X.iloc[valid_index], X.iloc[test_index]
                y_train, y_valid, y_test = y.iloc[train_index], y.iloc[valid_index], y.iloc[test_index]

                # Create LightGBM datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_valid, label=y_valid)

                # Default LightGBM parameters if none are provided
                if model_params is None:
                    # model_params = {
                    #     "objective": "regression",
                    #     "metric": "rmse",
                    #     "boosting_type": "gbdt",
                    #     "num_leaves": 31,
                    #     "learning_rate": 0.01,
                    #     "feature_fraction": 0.9,
                    #     "random_state": 42
                    # }
                    # model_params = {
                    #     "objective": "regression",
                    #     "metric": "rmse",
                    #     "boosting_type": "gbdt",
                    #     "num_leaves": 50,
                    #     "learning_rate": 0.03,
                    #     "feature_fraction": 0.8,
                    #     "bagging_fraction": 0.8,
                    #     "bagging_freq": 5,
                    #     "max_depth": 10,
                    #     "lambda_l1": 0.1,
                    #     "lambda_l2": 0.1,
                    #     "random_state": 42,
                    # }
                    model_params = {
                        "objective": "regression",
                        "metric": "rmse",
                        "boosting_type": "gbdt",
                        "num_leaves": 50,
                        "max_depth": 10,
                        "learning_rate": 0.03,
                        "feature_fraction": 0.8,
                        "bagging_fraction": 0.8,
                        "bagging_freq": 5,
                        "min_data_in_leaf": 20,
                        "lambda_l1": 0.1,
                        "lambda_l2": 0.1,
                        "random_state": 42
                    }

                st.session_state['training_pipeline_metrics'] = metrics

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
                    input_example = pd.DataFrame(X_train[:1], columns=X_train.columns)
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



# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    try:
        # Define file paths
        input_csv = "/workspaces/invenlytics/dynamic_supply_chain_logistics_dataset (1) (1).csv"
        bronze_parquet = "/workspaces/invenlytics/data/bronze_layer/supply_chain_datageneration.parquet"
        silver_parquet = "/workspaces/invenlytics/data/silver_layer/preprocessed_supply_chain_preprocessed_file.parquet"
        gold_parquet = "/workspaces/invenlytics/data/gold_layer/SupplyChain_Invetory_Dataset.parquet"

        # Pipeline execution
        data = generate_data(input_csv, bronze_parquet)
        preprocessed_data = preprocess_data(bronze_parquet, silver_parquet)
        feature_data = feature_engineering(silver_parquet, gold_parquet)
        run_inventory_training_pipeline(
            data_source=gold_parquet,
            target_column="Historical_Demand"
        )

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
