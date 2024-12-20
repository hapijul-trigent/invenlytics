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
    selected_columns (list): List of columns to validate and clean.
    dest (str): Path to save the cleaned dataset.

    Returns:
    pd.DataFrame: The cleaned dataset.
    """
    try:
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
            logging.error("Misiing Scheduled_Delivery / Actual_Delivery")
            return data
        logging.info("Converted required columns to datetime.")


        # Remove duplicates
        data.drop_duplicates(inplace=True)
        logging.info("Duplicate rows removed.")

        # Validate and clean specific columns
        if "Scheduled_Delivery" in data.columns:
            data = data[pd.notnull(data["Scheduled_Delivery"])]
            data["Scheduled_Delivery"] = pd.to_datetime(data["Scheduled_Delivery"], errors='coerce')
            data = data[data["Scheduled_Delivery"].notnull()]
            logging.info("Validated 'Scheduled_Delivery' column.")

        # # Outlier handling for numeric columns
        # numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        # for col in numeric_columns:
        #     q1, q3 = data[col].quantile([0.25, 0.75])
        #     iqr = q3 - q1
        #     lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        #     data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        # logging.info("Outliers handled.")

        # Save the cleaned dataset
        data.to_parquet(dest, index=False)
        logging.info(f"Cleaned dataset saved to {dest}.")
        logging.info(data.info())
        return data

    except Exception as e:
        logging.error(f"An error occurred during validation and cleaning: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    run(
        source="data/bronze_layer/SupplyChain_Dataset.parquet", 
        dest="data/silver_layer/preprocessed_SupplyChain_Dataset.parquet"
    )