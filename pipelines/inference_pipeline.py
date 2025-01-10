# import mlflow
# import pandas as pd
# import logging
# from pipelines import feature_pipeline
# import tempfile
# import os
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def load_model(logged_model_path):
#     """
#     Load the MLflow model from the given path.
#     """
#     logger.info("Loading model from MLflow...")
#     try:
#         model = mlflow.pyfunc.load_model(logged_model_path)
#         logger.info("Model loaded successfully.")
#         return model
#     except Exception as e:
#         logger.error(f"Failed to load the model. Error: {e}")
#         raise

# def validate_data(df, expected_columns):
#     """
#     Validate the input DataFrame against the model's expected schema.
#     """
#     logger.info("Validating input data...")
#     missing_columns = set(expected_columns) - set(df.columns)
#     if missing_columns:
#         logger.warning(f"Missing columns: {missing_columns}. Adding default values...")
#         for col in missing_columns:
#             df[col] = 0  # Assign default values for missing columns
#     logger.info("Validation successful.")
#     return df



# def preprocess_data(df, pipeline):
#     """
#     Apply the inventory optimization feature pipeline for preprocessing.
#     """
#     logger.info("Running inventory feature pipeline...")
#     try:
#         with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
#             temp_file_path = temp_file.name
#             df.to_parquet(temp_file_path)
#             logger.info(f"Temporary file created at {temp_file_path}.")

#         processed_df = pipeline.run_inventory_optimization_feature_pipeline(
#             source=temp_file_path,
#             selected_columns=df.columns.tolist(),
#             dest=None
#         )

#         # Remove duplicate columns
#         processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]
#         logger.info("Duplicate columns removed from the processed DataFrame.")

#         os.remove(temp_file_path)
#         logger.info("Temporary file removed.")

#         return processed_df
#     except Exception as e:
#         logger.error(f"Feature pipeline preprocessing failed. Error: {e}")
#         raise


# def run_inference(df, logged_model_path, pipeline):
#     """
#     Perform inference using the inventory management model and preprocessed data.
#     """
#     try:
#         # Load the model
#         model = load_model(logged_model_path)

#         # Preprocess the data
#         processed_df = preprocess_data(df, pipeline)

#         # Remove duplicate columns
#         processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]
        
#         # Validate processed data against model's schema
#         expected_columns = model.metadata.get_input_schema().input_names()

#         # Add missing columns with default values
#         missing_columns = set(expected_columns) - set(processed_df.columns)
#         if missing_columns:
#             logger.warning(f"Missing columns in processed data: {missing_columns}. Adding default values...")
#             for col in missing_columns:
#                 processed_df[col] = 0  # Add missing columns with default value

#         # Retain only expected columns
#         processed_df = processed_df[[col for col in expected_columns if col in processed_df.columns]]

#         # Perform inference
#         logger.info("Performing inference...")
#         predictions = model.predict(processed_df)

#         # Add predictions to DataFrame
#         processed_df["Demand_Forecast"] = predictions
#         logger.info("Inference completed successfully.")

#         return processed_df
#     except Exception as e:
#         logger.error(f"An error occurred during inference. Error: {e}")
#         raise


# if __name__ == "__main__":
#     try:
#         # Load the dataset
#         input_data_path = "data/gold_layer/Inventory_Management_Dataset.parquet"
#         logger.info(f"Loading data from {input_data_path}...")
#         df = pd.read_parquet(input_data_path)

#         # Define the model path
#         model_path = "runs:/your_mlflow_model_id/model"  # Replace with your actual model path

#         # Run inference
#         predictions = run_inference(df, model_path, feature_pipeline)

#         # Save predictions
#         output_data_path = "data/inference_layer/Inventory_Demand_Forecast.parquet"
#         predictions.to_parquet(output_data_path, index=False)
#         logger.info(f"Predictions saved to {output_data_path}.")
#     except Exception as e:
#         logger.error(f"An error occurred in the inference pipeline. Error: {e}")


import mlflow
import pandas as pd
import logging
import numpy as np
from pipelines import feature_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(logged_model_path):
    """
    Load the MLflow model from the given path.
    """
    logger.info("Loading model from MLflow...")
    try:
        model = mlflow.pyfunc.load_model(logged_model_path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load the model. Error: {e}")
        raise

def validate_data(df, expected_columns):
    """
    Validate the input DataFrame against the expected schema.
    """
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Input DataFrame does not match the model's expected schema. Missing columns: {missing_columns}")

def preprocess_data(df):
    """
    Apply feature engineering and preprocess the data.
    """
    logger.info("Preprocessing data...")
    try:
        # Ensure numeric columns are filled
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        # Forward-fill for other missing values
        df.fillna(method='ffill', inplace=True)

        # Any additional preprocessing logic specific to inventory optimization
        logger.info("Data preprocessing completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise
        
def run_inference(df, logged_model_path):
    """
    Perform inference for inventory optimization.
    """
    try:
        # Load the MLflow model
        model = load_model(logged_model_path)

        # Get the model's expected schema
        schema = model.metadata.get_input_schema()
        expected_columns = schema.input_names()

        # Preprocess the input data
        df = preprocess_data(df)

        # Validate the processed data against the model's expected schema
        validate_data(df, expected_columns)

        # Perform inference
        logger.info("Running inference...")
        df["Demand_Forecast"] = model.predict(df)
        logger.info("Inference completed successfully.")

        return df
    except Exception as e:
        logger.error(f"An error occurred during inference: {e}")
        raise

if __name__ == "__main__":
    try:
        # Load the dataset
        input_data_path = "data/gold_layer/SupplyChain_Invetory_Dataset.parquet"
        logger.info(f"Loading data from {input_data_path}...")
        df = pd.read_parquet(input_data_path)

        # Define model path
        logged_model_path = "runs:/56ad624fcdc34cb5b7c6064c46a752d4/model"  # Replace with your actual model path

        # Run inference
        predictions = inference_pipeline.run_inference(df, logged_model_path)

        # Display predictions
        print(predictions)

        # Save predictions
        output_path = "data/inference_layer/Inventory_Demand_Forecast.parquet"
        predictions.to_parquet(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}.")
    except Exception as e:
        logger.error(f"Error in main process: {e}")