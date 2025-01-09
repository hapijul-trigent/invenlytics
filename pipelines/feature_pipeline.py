import logging
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from pipelines.utils import apply_pca

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/feature_pipeline.log"),
            logging.StreamHandler()
        ]
    )

# def run_inventory_optimization_feature_pipeline(source, selected_columns, dest, pca_components=0.95):
#     """
#     Perform feature engineering on the dataset.
#     """
#     try:
#         setup_logging()
#         logging.info(f"Loading dataset from {source}.")

#         # Load dataset
#         data = pd.read_parquet(source)
#         logging.info(f"Dataset loaded successfully. Shape: {data.shape}")

#         # Ensure selected columns exist
#         available_columns = [col for col in selected_columns if col in data.columns]
#         missing_columns = [col for col in selected_columns if col not in data.columns]

#         if missing_columns:
#             logging.warning(f"The following columns are missing and will be ignored: {missing_columns}")

#         data = data[available_columns]

#         # Fill missing values
#         data.fillna(method="ffill", inplace=True)

#         # Handle datetime columns
#         datetime_columns = data.select_dtypes(include=["datetime64"]).columns
#         for col in datetime_columns:
#             data[f"{col}_year"] = data[col].dt.year
#             data[f"{col}_month"] = data[col].dt.month
#             data[f"{col}_day"] = data[col].dt.day
#             data[f"{col}_dayofweek"] = data[col].dt.dayofweek
#         data.drop(columns=datetime_columns, inplace=True)

#         # Add `Stockout_Risk` calculation
#         if all(col in data.columns for col in ["Current_Stock", "Forecasted_Demand"]):
#             data["Stockout_Risk"] = (data["Current_Stock"] < data["Forecasted_Demand"]).astype(int)

#         # Lagged and Rolling Features
#         lag_steps = [3, 7, 14]
#         for lag in lag_steps:
#             data[f"Demand_Lag_{lag}"] = data["Historical_Demand"].shift(lag)
#         rolling_windows = [3, 7]
#         for window in rolling_windows:
#             data[f"Demand_Rolling_Mean_{window}"] = data["Historical_Demand"].rolling(window=window).mean()
#             data[f"Demand_Rolling_Std_{window}"] = data["Historical_Demand"].rolling(window=window).std()

#         # Additional Features
#         data["Demand_EWA"] = data["Historical_Demand"].ewm(span=5).mean()

#         # Drop rows with NaN values introduced by lagging or rolling
#         data.dropna(inplace=True)

#         # Normalize numeric columns
#         # Normalize numeric columns
#         numeric_columns = data.select_dtypes(include=[float, int]).columns
#         if not numeric_columns.any():
#             raise ValueError("No numeric columns found for scaling. Check your dataset.")
#         scaler = MinMaxScaler()
#         data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

#         # Apply PCA with unique naming
#         data = apply_pca(data, numeric_columns, pca_components)

#         # Remove duplicate columns
#         data = data.loc[:, ~data.columns.duplicated()]
#         logging.info("Duplicate columns removed.")

#         # Save processed data
#         data.to_parquet(dest, index=False)
#         logging.info(f"Processed data saved to {dest}.")
#         return data

#     except Exception as e:
#         logging.error(f"An error occurred during inventory feature engineering: {e}", exc_info=True)
#         raise

def run_inventory_optimization_feature_pipeline(source, selected_columns, dest):
    """
    Perform feature engineering on the dataset.
    """
    try:
        setup_logging()
        logging.info(f"Loading dataset from {source}.")

        # Load dataset
        data = pd.read_parquet(source)
        logging.info(f"Dataset loaded successfully. Shape: {data.shape}")

        # Ensure selected columns exist
        available_columns = [col for col in selected_columns if col in data.columns]
        missing_columns = [col for col in selected_columns if col not in data.columns]

        if missing_columns:
            logging.warning(f"The following columns are missing and will be ignored: {missing_columns}")

        data = data[available_columns]

        # Fill missing values
        data.fillna(method="ffill", inplace=True)

        # Handle datetime columns
        datetime_columns = data.select_dtypes(include=["datetime64"]).columns
        for col in datetime_columns:
            data[f"{col}_year"] = data[col].dt.year
            data[f"{col}_month"] = data[col].dt.month
            data[f"{col}_day"] = data[col].dt.day
            data[f"{col}_dayofweek"] = data[col].dt.dayofweek
        data.drop(columns=datetime_columns, inplace=True)

        # Add `Stockout_Risk` calculation
        if all(col in data.columns for col in ["Current_Stock", "Forecasted_Demand"]):
            data["Stockout_Risk"] = (data["Current_Stock"] < data["Forecasted_Demand"]).astype(int)

        # Lagged and Rolling Features
        lag_steps = [3, 7, 14]
        for lag in lag_steps:
            data[f"Demand_Lag_{lag}"] = data["Historical_Demand"].shift(lag)
        rolling_windows = [3, 7]
        for window in rolling_windows:
            data[f"Demand_Rolling_Mean_{window}"] = data["Historical_Demand"].rolling(window=window).mean()
            data[f"Demand_Rolling_Std_{window}"] = data["Historical_Demand"].rolling(window=window).std()

        # Additional Features
        data["Demand_EWA"] = data["Historical_Demand"].ewm(span=5).mean()

        # Drop rows with NaN values introduced by lagging or rolling
        data.dropna(inplace=True)
        logging.info(f"Data shape after dropping NaNs: {data.shape}")

        # Normalize numeric columns
        numeric_columns = data.select_dtypes(include=[float, int]).columns
        if not numeric_columns.any():
            raise ValueError("No numeric columns found for scaling. Check your dataset.")
        scaler = MinMaxScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        # Remove duplicate columns
        data = data.loc[:, ~data.columns.duplicated()]
        logging.info("Duplicate columns removed.")

        # Save processed data
        data.to_parquet(dest, index=False)
        logging.info(f"Processed data saved to {dest}.")
        return data

    except Exception as e:
        logging.error(f"An error occurred during inventory feature engineering: {e}", exc_info=True)
        raise
