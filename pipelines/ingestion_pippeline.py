# import pandas as pd
# import numpy as np
# import random
# import logging
# from datetime import timedelta

# def setup_logging():
#     """Setup logging configuration."""
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler("logs/ingestion_pipeline.log"),
#             logging.StreamHandler()
#         ]
#     )

# def generate_additional_data(existing_df, total_rows_required=100000):
#     """
#     Add new columns and generate additional rows to reach the required total rows.

#     Parameters:
#         existing_df (pd.DataFrame): The existing dataset.
#         total_rows_required (int): The total number of rows required.

#     Returns:
#         pd.DataFrame: A dataframe with new columns and rows.
#     """
#     logging.info("Adding new columns and generating additional rows...")

#     # Number of additional rows needed
#     additional_rows_needed = total_rows_required - len(existing_df)

#     # New columns to be added
#     suppliers = ['Alibaba', 'H&M', 'IKEA', 'Wrogn']
#     delivery_modes = ['Air', 'Sea', 'Road', 'Rail']
#     disruption_types = ['Weather', 'Supplier Issue', 'Logistics', 'Geopolitical', 'None']
#     regions = ['North America', 'Europe', 'Asia', 'South America']
#     weather_conditions = ['Sunny', 'Rainy', 'Snowy', 'Cloudy']

#     # Add new columns for existing rows if they don't exist
#     if 'Supplier' not in existing_df.columns:
#         existing_df['Supplier'] = np.random.choice(suppliers, len(existing_df))
#     if 'Region' not in existing_df.columns:
#         existing_df['Region'] = np.random.choice(regions, len(existing_df))
#     if 'Delivery_Mode' not in existing_df.columns:
#         existing_df['Delivery_Mode'] = np.random.choice(delivery_modes, len(existing_df))
#     if 'Disruption_Type' not in existing_df.columns:
#         existing_df['Disruption_Type'] = np.random.choice(disruption_types, len(existing_df))
#     if 'Weather_Conditions' not in existing_df.columns:
#         existing_df['Weather_Conditions'] = np.random.choice(weather_conditions, len(existing_df))
#     if 'Scheduled_Delivery' not in existing_df.columns:
#         existing_df['Scheduled_Delivery'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(1, 10, len(existing_df)), unit='D')
#     if 'Actual_Delivery' not in existing_df.columns:
#         existing_df['Actual_Delivery'] = existing_df['Scheduled_Delivery'] + pd.to_timedelta(np.random.randint(-5, 15, len(existing_df)), unit='D')
#     if 'Freight_Cost' not in existing_df.columns:
#         existing_df['Freight_Cost'] = np.round(np.random.uniform(100, 1000, len(existing_df)), 2)

#     # Generate additional rows if needed
#     synthetic_data = []
#     if additional_rows_needed > 0:
#         logging.info(f"Generating {additional_rows_needed} additional rows...")
#         for i in range(additional_rows_needed):
#             synthetic_data.append({
#                 "Supplier": np.random.choice(suppliers),
#                 "Region": np.random.choice(regions),
#                 "Delivery_Mode": np.random.choice(delivery_modes),
#                 "Disruption_Type": np.random.choice(disruption_types),
#                 "Weather_Conditions": np.random.choice(weather_conditions),
#                 "Scheduled_Delivery": pd.to_datetime('2024-01-01') + timedelta(days=random.randint(1, 10)),
#                 "Actual_Delivery": pd.to_datetime('2024-01-01') + timedelta(days=random.randint(5, 15)),
#                 "Freight_Cost": round(random.uniform(100, 1000), 2)
#             })

#     synthetic_df = pd.DataFrame(synthetic_data)
#     existing_df = pd.concat([existing_df, synthetic_df], ignore_index=True)

#     logging.info("Additional rows and columns added successfully.")
#     return existing_df

# def run(input_file, output_file, total_rows=100000):
#     """
#     Main function to load existing data, augment it with new rows and columns, and save it as Parquet.

#     Parameters:
#         input_file (str): Path to the existing dataset (CSV format).
#         output_file (str): Path to save the updated dataset (Parquet format).
#         total_rows (int): Total number of rows required in the dataset.

#     Returns:
#         None
#     """
#     try:
#         logging.info("Starting the data augmentation process.")

#         # Load existing dataset
#         existing_df = pd.read_csv(input_file)
#         logging.info(f"Loaded existing dataset with {len(existing_df)} rows.")

#         # Generate augmented dataset
#         updated_df = generate_additional_data(existing_df, total_rows_required=total_rows)

#         # Validate row count
#         if len(updated_df) != total_rows:
#             raise ValueError(f"Dataset row count mismatch: Expected {total_rows}, but got {len(updated_df)}")

#         # Save to Parquet
#         updated_df.to_parquet(output_file, index=False)
#         logging.info(f"Updated dataset saved to {output_file}")
#         return updated_df

#     except Exception as e:
#         logging.error("An error occurred during the pipeline execution:", exc_info=True)


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
        logging.info(f"Updated dataset saved to {output_file}")
        return updated_df

    except Exception as e:
        logging.error("An error occurred during the pipeline execution:", exc_info=True)

if __name__ == "__main__":
    setup_logging()

    # Define file paths
    input_csv_path = "/workspaces/invenlytics/dynamic_supply_chain_logistics_dataset (1) (1).csv"
    output_parquet_path = "/workspaces/invenlytics/data/bronze_layer/inventory_data.parquet"
    

    # Run the pipeline
    run(input_file=input_csv_path, output_file=output_parquet_path, total_rows=100000)
