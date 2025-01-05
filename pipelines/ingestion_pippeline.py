# import os
# import pandas as pd
# import logging
# from datetime import datetime

# def setup_logging(log_file="logs/data_ingestion.log"):
#     """Set up logging for the data ingestion pipeline."""
#     os.makedirs(os.path.dirname(log_file), exist_ok=True)
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )

# def load_existing_data(file_path):
#     """Load existing data from a CSV file."""
#     if os.path.exists(file_path):
#         logging.info(f"Loading existing data from {file_path}...")
#         return pd.read_csv(file_path)
#     else:
#         logging.warning(f"File {file_path} not found. Starting with an empty DataFrame.")
#         return pd.DataFrame()

# def save_data(df, output_file):
#     """Save the processed data to a file."""
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#     logging.info(f"Saving data to {output_file}...")
#     df.to_csv(output_file, index=False)
#     logging.info("Data saved successfully.")

# def generate_additional_data(existing_df, total_rows_required=100000):
#     """
#     Generate additional synthetic data to reach the desired number of rows.

#     Parameters:
#         existing_df (pd.DataFrame): The existing dataset.
#         total_rows_required (int): The total number of rows required.

#     Returns:
#         pd.DataFrame: A dataframe with new columns and rows.
#     """
#     from datetime import timedelta
#     import numpy as np
#     import random

#     logging.info("Generating additional synthetic data...")

#     # Number of rows needed
#     additional_rows_needed = total_rows_required - len(existing_df)
#     if additional_rows_needed <= 0:
#         logging.info("No additional rows needed.")
#         return existing_df

#     suppliers = ['Alibaba', 'H&M', 'IKEA', 'Wrogn']
#     delivery_modes = ['Air', 'Sea', 'Road', 'Rail']
#     disruption_types = ['Weather', 'Supplier Issue', 'Logistics', 'Geopolitical', 'None']
#     regions = ['North America', 'Europe', 'Asia', 'South America']
#     weather_conditions = ['Sunny', 'Rainy', 'Snowy', 'Cloudy']

#     synthetic_data = []
#     for _ in range(additional_rows_needed):
#         synthetic_data.append({
#             "timestamp": datetime.now(),
#             "Supplier": random.choice(suppliers),
#             "Region": random.choice(regions),
#             "Delivery_Mode": random.choice(delivery_modes),
#             "Disruption_Type": random.choice(disruption_types),
#             "Weather_Conditions": random.choice(weather_conditions),
#             "Scheduled_Delivery": pd.to_datetime('2024-01-01') + timedelta(days=random.randint(1, 10)),
#             "Actual_Delivery": pd.to_datetime('2024-01-01') + timedelta(days=random.randint(5, 15)),
#             "Freight_Cost": round(random.uniform(100, 1000), 2),
#             "supplier_reliability_score": round(random.uniform(0.7, 1.0), 2),
#             "lead_time_days": random.randint(1, 15),
#             "historical_demand": round(random.uniform(100, 5000), 2),
#             "iot_temperature": round(random.uniform(-10, 40), 2),
#             "cargo_condition_status": round(random.uniform(0, 1), 2),
#             "route_risk_level": round(random.uniform(0, 10), 2),
#             "customs_clearance_time": round(random.uniform(0, 5), 2),
#             "driver_behavior_score": round(random.uniform(0, 1), 2),
#             "fatigue_monitoring_score": round(random.uniform(0, 1), 2),
#             "disruption_likelihood_score": round(random.uniform(0, 1), 2),
#             "delay_probability": round(random.uniform(0, 1), 2),
#             "risk_classification": random.choice(['Low Risk', 'Moderate Risk', 'High Risk']),
#             "delivery_time_deviation": round(random.uniform(0, 15), 2)
#         })

#     synthetic_df = pd.DataFrame(synthetic_data)
#     logging.info(f"Generated {len(synthetic_df)} synthetic rows.")

#     return pd.concat([existing_df, synthetic_df], ignore_index=True)

# def run(output_file, input_file=None, total_rows_required=100000):
#     """Run the data ingestion pipeline."""
#     setup_logging()

#     # Load existing data if provided
#     if input_file:
#         existing_data = load_existing_data(input_file)
#     else:
#         existing_data = pd.DataFrame()

#     # Generate synthetic data to ensure the dataset has the required rows
#     full_data = generate_additional_data(existing_data, total_rows_required=total_rows_required)

#     # Save the final dataset
#     save_data(full_data, output_file)

# if __name__ == '__main__':
#     run(
#         output_file="data/bronze_layer/SupplyChain_Dataset_100k.parquet",
#         input_file="dynamic_supply_chain_logistics_dataset (1).csv",
#         total_rows_required=100000
#     )


import pandas as pd
import numpy as np
import random
import logging
from datetime import timedelta

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/ingestion_pipeline.log"),
            logging.StreamHandler()
        ]
    )

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
        logging.info(f"Updated dataset saved to {output_file}")

    except Exception as e:
        logging.error("An error occurred during the pipeline execution:", exc_info=True)

if __name__ == "__main__":
    setup_logging()

    # Define file paths
    input_csv_path = "/workspaces/invenlytics/dynamic_supply_chain_logistics_dataset (1).csv"
    output_parquet_path = "data/bronze_layer/Updated_SupplyChain_Dataset.parquet"

    # Run the pipeline
    run(input_file=input_csv_path, output_file=output_parquet_path, total_rows=100000)
