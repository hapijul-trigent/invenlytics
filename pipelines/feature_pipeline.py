import logging
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas as pd

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

def run_supplychain_disruption_feature_pipeline(source, selected_columns, dest):
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
        data = pd.read_parquet(source)
        data = data[selected_columns]
        logging.info("Dataset loaded successfully.")


        # Target encoding for 'Disruption_Type'
        if "Disruption_Type" in data.columns:
            data["Disruption"] = data["Disruption_Type"].apply(lambda x: 0 if x == "None" else 1)
        logging.info("Target encoding completed.")

        # Encode categorical features
        encoder = LabelEncoder()
        for col in ["Region", "Delivery_Mode"]:
            if col in data.columns:
                data[col] = encoder.fit_transform(data[col])
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

        # Sort and set index
        data.sort_values(by="Scheduled_Delivery", inplace=True)
        data.set_index("Scheduled_Delivery", inplace=True)
        logging.info("Data sorted and indexed by 'Scheduled_Delivery'.")

        # Define rolling window and lag features
        window_size = [3, 7]
        rolling_cols = ["Weather_Risk", "Supplier_Reliability", "Port_Congestion", 'Delay_Duration']
        lag_cols = ["Weather_Risk", "Supplier_Reliability", "Port_Congestion"]

        # Generate rolling window features
        for window in window_size:
            for col in rolling_cols:
                if col in data.columns:
                    data[f"{col}_rolling_mean_{window}"] = data[col].rolling(window=window).mean()
                    data[f"{col}_rolling_std_{window}"] = data[col].rolling(window=window).std()
        logging.info("Rolling window features generated.")

        # Generate expanding window features
        for col in rolling_cols:
            if col in data.columns:
                data[f"{col}_expanding_mean"] = data[col].expanding(min_periods=7).mean()
        logging.info("Expanding window features generated.")

        # Generate lag features
        lag_steps = [1, 2, 3, 4, 5, 6, 7]
        for col in lag_cols:
            if col in data.columns:
                for lag in lag_steps:
                    data[f"{col}_lag_{lag}"] = data[col].shift(lag)
        logging.info("Lag features generated.")

        # Drop rows with NaN values introduced by rolling, expanding, and lag features
        data.dropna(inplace=True)
        logging.info("Dropped NaN values. Preprocessing complete.")
        
        data.to_parquet(dest, index=False)
        logging.info(f"Features Saved {dest}")
        logging.info(data.info())
        return data

    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}", exc_info=True)
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
