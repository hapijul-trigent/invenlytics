import logging
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas as pd

# Setup logging
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

# Supply chain disruption feature pipeline
def run_supplychain_disruption_feature_pipeline(source=None, selected_columns=None, dest=None, data=pd.DataFrame()):
    """
    Preprocess the dataset by performing target encoding, categorical encoding, 
    and feature engineering (rolling, lag, expanding window features).

    Parameters:
    source (str): The path to the dataset file.
    selected_columns (list): List of columns to select from the dataset.

    Returns:
    pd.DataFrame: The processed dataset with new features.
    """
    try:
        logging.info("Starting preprocessing pipeline.")

        # Load the dataset
        if data.empty:
            data = pd.read_parquet(source)
            logging.info(f"{data.info()}")
        data = data[selected_columns]
        logging.info(f"Dataset loaded successfully. {data.info()}")

        # Target encoding for 'Disruption_Type'
        if "disruption_likelihood_score" in data.columns:
            data["Disruption"] = data["disruption_likelihood_score"].apply(lambda x: 0 if x < 0.5 else 1)
            data.drop(columns=['disruption_likelihood_score'], inplace=True)
        logging.info("Target encoding completed.")

        # Encode categorical features
        encoder = LabelEncoder()
        for col in ["Region", "Delivery_Mode", "Supplier"]:
            if col in data.columns:
                data[col] = encoder.fit_transform(data[col])
        
        # One-hot encoding for Weather_Conditions
        if "Weather_Conditions" in data.columns:
            data = pd.get_dummies(data, columns=["Weather_Conditions"], drop_first=True)
        logging.info("Categorical encoding completed.")

        # Generate holiday-related features
        holiday_calendar = USFederalHolidayCalendar()
        holidays = holiday_calendar.holidays(
            start=data["Scheduled_Delivery"].min(), 
            end=data["Scheduled_Delivery"].max()
        )

        data["Is_Business_Day"] = data["Scheduled_Delivery"].dt.dayofweek < 5
        data["Is_Holiday"] = data["Scheduled_Delivery"].dt.date.isin(holidays.date)
        data["Is_Working_Day"] = data["Is_Business_Day"] & ~data["Is_Holiday"]
        data["Week_Of_Year"] = data["Scheduled_Delivery"].dt.isocalendar().week
        data["Quarter"] = data["Scheduled_Delivery"].dt.quarter
        data["Is_Weekend"] = ~data["Is_Business_Day"]
        logging.info("Holiday-related features created.")

        # # Generate time-based features
        # if "Actual_Delivery" in data.columns:
        #     data["Delivery_Duration"] = (data["Actual_Delivery"] - data["Scheduled_Delivery"]).dt.total_seconds() / 3600

        # # Normalize selected numerical features
        # normalize_cols = [
        #     "traffic_congestion_level", "port_pongestion_level", "weather_condition_severity", 
        #     "fuel_consumption_rate", "driver_behavior_score", "fatigue_monitoring_score"
        # ]
        # scaler = MinMaxScaler()
        # for col in normalize_cols:
        #     if col in data.columns:
        #         data[col] = scaler.fit_transform(data[[col]])
        # logging.info("Normalization completed.")

        # # Define rolling, lag, and expanding window features
        # rolling_cols = ["traffic_congestion_level", "port_congestion_level", "supplier_reliability_score"]
        # lag_cols = rolling_cols
        # window_sizes = [3, 7, 14, 24, 48]
        # lag_steps = [1, 2, 3, 4, 5, 6, 7, 14, 24, 48]

        # # Generate rolling and expanding features
        # for window in window_sizes:
        #     for col in rolling_cols:
        #         if col in data.columns:
        #             data[f"{col}_rolling_mean_{window}"] = data[col].rolling(window=window).mean()
        #             data[f"{col}_rolling_std_{window}"] = data[col].rolling(window=window).std()

        # for col in rolling_cols:
        #     if col in data.columns:
        #         data[f"{col}_expanding_mean"] = data[col].expanding(min_periods=7).mean()
        # logging.info("Rolling and expanding features generated.")

        # # Generate lag features
        # for col in lag_cols:
        #     if col in data.columns:
        #         for lag in lag_steps:
        #             data[f"{col}_lag_{lag}"] = data[col].shift(lag)
        # logging.info("Lag features generated.")

        # Feature interactions
        if "traffic_congestion_level" in data.columns and "port_congestion_level" in data.columns:
            data["Traffic_Port_Interaction"] = data["traffic_congestion_level"] * data["port_congestion_level"]

        # Drop rows with NaN values introduced by feature generation
        data.dropna(inplace=True)
        logging.info("Dropped NaN values. Preprocessing complete.")

        # Save the processed data
        data.to_parquet(dest, index=False)
        logging.info(f"Processed dataset saved to {dest}. {data.info()}")
        return data

    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}", exc_info=True)
        raise

# Main script
if __name__ == "__main__":
    setup_logging()
    try:
        selected_columns = [
            "Scheduled_Delivery", "Actual_Delivery", "Disruption_Type", "Region", "Delivery_Mode", "Supplier",
            "Weather_Conditions",  "traffic_congestion_level", "port_congestion_level", "weather_condition_severity", 
            "fuel_consumption_rate", "driver_behavior_score", "fatigue_monitoring_score", 'supplier_reliability_score'
        ]
        processed_data = run_supplychain_disruption_feature_pipeline(
            source="/workspaces/invenlytics/data/silver_layer/preprocessed_dynamic_supply_chain_logistics_dataset.parquet", 
            selected_columns=selected_columns, 
            dest="/workspaces/invenlytics/data/gold_layer/SupplyChain_DisruptionFeatures.parquet"
        )
        logging.info("Feature pipeline executed successfully.")
    except Exception as e:
        logging.critical(f"Pipeline execution failed: {e}")
