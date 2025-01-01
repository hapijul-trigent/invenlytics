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


def run_inventory_optimization_feature_pipeline(source, selected_columns, dest, pca_components=0.95):
    try:
        logging.info("Starting inventory optimization feature pipeline.")

        # Load the dataset
        data = pd.read_parquet(source)
        logging.info(f"Columns in the dataset: {data.columns.tolist()}")

        # Ensure selected columns exist in the dataset
        available_columns = [col for col in selected_columns if col in data.columns]
        missing_columns = [col for col in selected_columns if col not in data.columns]

        if missing_columns:
            logging.warning(f"The following columns are missing from the dataset and will be ignored: {missing_columns}")

        data = data[available_columns]

        # Fill missing values
        data.fillna(method="ffill", inplace=True)

        # Lagged Features
        lag_steps = [3, 7, 14]
        for lag in lag_steps:
            data[f"Demand_Lag_{lag}"] = data["Historical_Demand"].shift(lag)

        # Rolling Features
        rolling_windows = [3, 7]
        for window in rolling_windows:
            data[f"Demand_Rolling_Mean_{window}"] = data["Historical_Demand"].rolling(window=window).mean()
            data[f"Demand_Rolling_Std_{window}"] = data["Historical_Demand"].rolling(window=window).std()

        # Additional Features
        data["Demand_EWA"] = data["Historical_Demand"].ewm(span=5).mean()

        # Drop rows with NaN values introduced by lagging or rolling
        data.dropna(inplace=True)

        # Normalize numeric columns
        scaler = MinMaxScaler()
        numeric_columns = data.select_dtypes(include=[float, int]).columns
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        # Apply PCA
        data = apply_pca(data, numeric_columns, pca_components)

        # Save feature-engineered dataset
        data.to_parquet(dest, index=False)
        logging.info(f"Inventory optimization dataset with PCA saved to {dest}.")
        return data

    except Exception as e:
        logging.error(f"An error occurred during inventory feature engineering: {e}", exc_info=True)
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
