# import logging
# from sklearn.preprocessing import LabelEncoder
# from pandas.tseries.holiday import USFederalHolidayCalendar
# import pandas as pd

# def setup_logging():
#     """Setup logging configuration."""
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler("logs/preprocessing_pipeline.log"),
#             logging.StreamHandler()
#         ]
#     )


# def run(source, dest):
#     """
#     Validate, clean, and save the dataset.

#     Parameters:
#     source (str): Path to the source dataset file.
#     selected_columns (list): List of columns to validate and clean.
#     dest (str): Path to save the cleaned dataset.

#     Returns:
#     pd.DataFrame: The cleaned dataset.
#     """
#     try:
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
#             logging.error("Misiing Scheduled_Delivery / Actual_Delivery")
#             return data
#         logging.info("Converted required columns to datetime.")


#         # Remove duplicates
#         data.drop_duplicates(inplace=True)
#         logging.info("Duplicate rows removed.")

#         # Validate and clean specific columns
#         if "Scheduled_Delivery" in data.columns:
#             data = data[pd.notnull(data["Scheduled_Delivery"])]
#             data["Scheduled_Delivery"] = pd.to_datetime(data["Scheduled_Delivery"], errors='coerce')
#             data = data[data["Scheduled_Delivery"].notnull()]
#             logging.info("Validated 'Scheduled_Delivery' column.")

#         # # Outlier handling for numeric columns
#         # numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
#         # for col in numeric_columns:
#         #     q1, q3 = data[col].quantile([0.25, 0.75])
#         #     iqr = q3 - q1
#         #     lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
#         #     data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
#         # logging.info("Outliers handled.")

#         # Save the cleaned dataset
#         data.to_parquet(dest, index=False)
#         logging.info(f"Cleaned dataset saved to {dest}.")
#         logging.info(data.info())
#         return data

#     except Exception as e:
#         logging.error(f"An error occurred during validation and cleaning: {e}", exc_info=True)
#         raise


# if __name__ == '__main__':
#     run(
#         source="data/bronze_layer/SupplyChain_Dataset.parquet", 
#         dest="data/silver_layer/preprocessed_SupplyChain_Dataset.parquet"
#     )


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

        # Handle missing values
        for col in data.columns:
            if data[col].isnull().any():
                if data[col].dtype == 'object':
                    data[col].fillna("Unknown", inplace=True)
                else:
                    data[col].fillna(data[col].median(), inplace=True)
        logging.info("Missing values handled.")

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
        source="data/bronze_layer/SupplyChain_Dataset.parquet", 
        dest="data/silver_layer/preprocessed_SupplyChain_Dataset.parquet"
    )
