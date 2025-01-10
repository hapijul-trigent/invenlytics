import pandas as pd
import numpy as np
from datetime import timedelta
import logging
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pandas.tseries.holiday import USFederalHolidayCalendar
from pipelines import inference_pipeline
import streamlit as st

def augment_dataset(input_file, total_rows=100000):
    """
    Load a dataset, augment it with new columns, and return the updated DataFrame.

    Parameters:
        input_file (str): Path to the existing dataset (CSV format).
        total_rows (int): Total number of rows required in the dataset (not used).

    Returns:
        pd.DataFrame: The augmented dataset.
    """
    try:
        # Load existing dataset
        existing_df = pd.read_csv(input_file)

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

        updated_df = existing_df.dropna()

        return updated_df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None




def preprocess_features(data, selected_columns=[
            "Scheduled_Delivery", "Actual_Delivery", "Disruption_Type", "Region", "Delivery_Mode", "Supplier",
            "Weather_Conditions",  "traffic_congestion_level", "port_congestion_level", "weather_condition_severity", 
            "fuel_consumption_rate", "driver_behavior_score", "fatigue_monitoring_score", 'supplier_reliability_score'
        ]):
    """
    Preprocess the dataset by performing target encoding, categorical encoding, 
    and feature engineering (rolling, lag, expanding window features).

    Parameters:
    data (pd.DataFrame): The input DataFrame.
    selected_columns (list): List of columns to select from the dataset.

    Returns:
    pd.DataFrame: The processed dataset with new features.
    """
    try:
        logging.info("Starting preprocessing pipeline.")

        data = data[selected_columns]
        logging.info(f"Dataset loaded successfully. {data.info()}")

        # Target encoding for 'Disruption_Type'
        if "Disruption_Type" in data.columns:
            data["Disruption"] = data["Disruption_Type"].apply(lambda x: 0 if x == "None" else 1)
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

        # Generate time-based features
        if "Actual_Delivery" in data.columns:
            data["Delivery_Duration"] = (data["Actual_Delivery"] - data["Scheduled_Delivery"]).dt.total_seconds() / 3600

        # Normalize selected numerical features
        normalize_cols = [
            "traffic_congestion_level", "port_congestion_level", "weather_condition_severity", 
            "fuel_consumption_rate", "driver_behavior_score", "fatigue_monitoring_score"
        ]
        scaler = MinMaxScaler()
        for col in normalize_cols:
            if col in data.columns:
                data[col] = scaler.fit_transform(data[[col]])
        logging.info("Normalization completed.")

        # Define rolling, lag, and expanding window features
        rolling_cols = ["traffic_congestion_level", "port_congestion_level", "supplier_reliability_score"]
        lag_cols = rolling_cols
        window_sizes = [3, 7, 14, 24, 48]
        lag_steps = [1, 2, 3, 4, 5, 6, 7, 14, 24, 48]

        # Generate rolling and expanding features
        for window in window_sizes:
            for col in rolling_cols:
                if col in data.columns:
                    data[f"{col}_rolling_mean_{window}"] = data[col].rolling(window=window).mean()
                    data[f"{col}_rolling_std_{window}"] = data[col].rolling(window=window).std()

        for col in rolling_cols:
            if col in data.columns:
                data[f"{col}_expanding_mean"] = data[col].expanding(min_periods=7).mean()
        logging.info("Rolling and expanding features generated.")

        # Generate lag features
        for col in lag_cols:
            if col in data.columns:
                for lag in lag_steps:
                    data[f"{col}_lag_{lag}"] = data[col].shift(lag)
        logging.info("Lag features generated.")

        # Feature interactions
        if "traffic_congestion_level" in data.columns and "port_congestion_level" in data.columns:
            data["Traffic_Port_Interaction"] = data["traffic_congestion_level"] * data["port_congestion_level"]

        # Drop rows with NaN values introduced by feature generation
        data.dropna(inplace=True)
       
        logging.info("Dropped NaN values. Preprocessing complete.")

        return data

    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}", exc_info=True)
        raise




def forecast(input_file, logged_model_path, selected_columns=None):
    """Forecast Pipeline"""
    try:
        
        features = preprocess_features(
            data=augment_dataset(input_file=input_file)
        )
        forecast_df = inference_pipeline.run_inference(
            df=features,
            logged_model_path='runs:/77eb38c47111407e98cb2e02fcfad5be/model'
        )
        return forecast_df
    except Exception as e:
        st.error('Some issue occured Try Again')