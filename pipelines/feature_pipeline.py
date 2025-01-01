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
#     try:
#         logging.info("Starting inventory optimization feature pipeline.")

#         # Load the dataset
#         data = pd.read_parquet(source)
#         logging.info(f"Columns in the dataset: {data.columns.tolist()}")

#         # Ensure selected columns exist in the dataset
#         available_columns = [col for col in selected_columns if col in data.columns]
#         missing_columns = [col for col in selected_columns if col not in data.columns]

#         if missing_columns:
#             logging.warning(f"The following columns are missing from the dataset and will be ignored: {missing_columns}")

#         data = data[available_columns]

#         # Fill missing values
#         data.fillna(method="ffill", inplace=True)

#         # Lagged Features
#         lag_steps = [3, 7, 14]
#         for lag in lag_steps:
#             data[f"Demand_Lag_{lag}"] = data["Historical_Demand"].shift(lag)

#         # Rolling Features
#         rolling_windows = [3, 7]
#         for window in rolling_windows:
#             data[f"Demand_Rolling_Mean_{window}"] = data["Historical_Demand"].rolling(window=window).mean()
#             data[f"Demand_Rolling_Std_{window}"] = data["Historical_Demand"].rolling(window=window).std()

#         # Additional Features
#         data["Demand_EWA"] = data["Historical_Demand"].ewm(span=5).mean()

#         # Drop rows with NaN values introduced by lagging or rolling
#         data.dropna(inplace=True)

#         # Normalize numeric columns
#         scaler = MinMaxScaler()
#         numeric_columns = data.select_dtypes(include=[float, int]).columns
#         data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

#         # Apply PCA
#         data = apply_pca(data, numeric_columns, pca_components)

#         # Save feature-engineered dataset
#         data.to_parquet(dest, index=False)
#         logging.info(f"Inventory optimization dataset with PCA saved to {dest}.")
#         return data

#     except Exception as e:
#         logging.error(f"An error occurred during inventory feature engineering: {e}", exc_info=True)
#         raise


def run_inventory_optimization_feature_pipeline(source, selected_columns, dest, pca_components=0.95):
    try:
        logging.info("Starting inventory optimization feature pipeline.")

        # Load dataset
        data = pd.read_parquet(source)
        logging.info(f"Columns in the dataset: {data.columns.tolist()}")

        # Select only the required columns
        data = data[selected_columns]

        # Remove existing PCA components to prevent duplication
        pca_columns = [col for col in data.columns if col.startswith("PCA_Component_")]
        if pca_columns:
            logging.info(f"Removing existing PCA components: {pca_columns}")
            data.drop(columns=pca_columns, inplace=True)

        # Fill missing values
        data.fillna(method="ffill", inplace=True)

        # Generate lagged features
        for lag in range(1, 8):
            for col in ["Supplier_Reliability", "Weather_Risk", "Port_Congestion"]:
                if col in data.columns:
                    data[f"{col}_lag_{lag}"] = data[col].shift(lag)

        # Generate rolling mean and std features
        for window in [3, 7]:
            for col in ["Supplier_Reliability", "Weather_Risk", "Port_Congestion"]:
                if col in data.columns:
                    data[f"{col}_rolling_mean_{window}"] = data[col].rolling(window).mean()
                    data[f"{col}_rolling_std_{window}"] = data[col].rolling(window).std()

        # Generate expanding mean features
        for col in ["Supplier_Reliability", "Weather_Risk", "Port_Congestion"]:
            if col in data.columns:
                data[f"{col}_expanding_mean"] = data[col].expanding().mean()

        # Add derived features
        data["Is_Holiday"] = data["Scheduled_Delivery"].dt.weekday.isin([5, 6]).astype(int)
        data["Is_Weekend"] = data["Scheduled_Delivery"].dt.weekday.isin([5, 6]).astype(int)
        data["Is_Business_Day"] = ~data["Scheduled_Delivery"].dt.weekday.isin([5, 6]).astype(int)
        data["Week_Of_Year"] = data["Scheduled_Delivery"].dt.isocalendar().week
        data["Quarter"] = data["Scheduled_Delivery"].dt.quarter

        # Apply PCA
        logging.info("Applying PCA...")
        numeric_columns = data.select_dtypes(include=[float, int]).columns
        data = apply_pca(data, numeric_columns, pca_components)

        # Ensure no duplicate columns
        data = data.loc[:, ~data.columns.duplicated()]
        logging.info("Duplicate columns removed successfully.")

        # Save processed data if a destination is provided
        if dest:
            data.to_parquet(dest, index=False)
            logging.info(f"Processed data saved to {dest}")

        return data

    except Exception as e:
        logging.error(f"An error occurred during feature engineering: {e}")
        raise


# Main script
if __name__ == "__main__":
    setup_logging()
    try:
        selected_columns = ["Scheduled_Delivery", "Disruption_Type", "Region", "Delivery_Mode", "Weather_Risk", "Supplier_Reliability", "Port_Congestion", "Delay_Duration"]
        processed_data = run_supplychain_disruption_feature_pipeline(source="data/bronze_layer/SupplyChain_Dataset.parquet", selected_columns=selected_columns, dest="data/silver_layer/SupplyChain_Dataset.parquet")
        logging.info("Feature pipeline executed successfully.")
    except Exception as e:
        logging.critical(f"Pipeline execution failed: {e}")
