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
            logging.FileHandler("logs/preprocessing_pipeline.log"),
            logging.StreamHandler()
        ]
    )

# def run(source, dest):
#     """
#     Validate, clean, and save the dataset.

#     Parameters:
#     source (str): Path to the source dataset file.
#     dest (str): Path to save the cleaned dataset.

#     Returns:
#     pd.DataFrame: The cleaned dataset.
#     """
#     try:
#         setup_logging()
#         logging.info(f"Loading dataset from {source}.")

#         # Load the dataset
#         data = pd.read_parquet(source)
#         logging.info("Dataset loaded successfully.")

#         # Handle missing values
#         for col in data.columns:
#             if data[col].isnull().any():
#                 if data[col].dtype == 'object':
#                     data[col].fillna("Unknown", inplace=True)
#                 else:
#                     data[col].fillna(data[col].median(), inplace=True)
#         logging.info("Missing values handled.")

#         # Convert necessary columns to datetime
#         if "Scheduled_Delivery" in data.columns and "Actual_Delivery" in data.columns:
#             data["Scheduled_Delivery"] = pd.to_datetime(data["Scheduled_Delivery"], errors='coerce')
#             data["Actual_Delivery"] = pd.to_datetime(data["Actual_Delivery"], errors='coerce')
#         else:
#             logging.error("Missing Scheduled_Delivery or Actual_Delivery column.")
#             return data
#         logging.info("Converted required columns to datetime.")

#         # Remove duplicates
#         initial_rows = data.shape[0]
#         data.drop_duplicates(inplace=True)
#         logging.info(f"Duplicate rows removed. {initial_rows - data.shape[0]} rows dropped.")

#         # Validate and clean specific columns
#         if "Scheduled_Delivery" in data.columns:
#             valid_rows = data["Scheduled_Delivery"].notnull().sum()
#             data = data[data["Scheduled_Delivery"].notnull()]
#             logging.info(f"Validated 'Scheduled_Delivery' column. {valid_rows} rows retained.")

#         # Encode categorical variables
#         categorical_columns = data.select_dtypes(include=['object']).columns
#         for col in categorical_columns:
#             encoder = LabelEncoder()
#             data[col] = encoder.fit_transform(data[col].astype(str))
#         logging.info("Categorical variables encoded.")

#         # Save the cleaned dataset
#         data.to_parquet(dest, index=False)
#         logging.info(f"Cleaned dataset saved to {dest}.")
#         logging.info(data.info())
#         if "timestamp" not in data.columns:
#             logging.error("Timestamp column is missing!")
#             raise ValueError("The timestamp column is missing from the dataset.")


#         return data

#     except Exception as e:
#         logging.error(f"An error occurred during validation and cleaning: {e}", exc_info=True)
#         raise

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
        
if __name__ == '__main__':
    run(
        source="/workspaces/invenlytics/data/bronze_layer/supply_chain_datageneration.parquet", 
        dest="/workspaces/invenlytics/data/silver_layer/preprocessed_supply_chain_preprocessed_file.parquet"
    )
