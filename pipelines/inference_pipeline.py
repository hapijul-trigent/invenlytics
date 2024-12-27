import mlflow
import pandas as pd
import logging
from pipelines import feature_pipeline
import numpy as np

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

def preprocess_data(df, pipeline):
    """
    Apply the feature pipeline to preprocess the data.
    """
    logger.info("Running feature pipeline...")
    try:
        if pipeline:
            df = pipeline.run_supplychain_disruption_feature_pipeline(data=df)
        logger.info("Feature pipeline completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed during feature engineering. Error: {e}")
        raise

def run_inference(df, logged_model_path, pipeline=None):
    """
    Perform inference using the given model and data.
    """
    try:
        model = load_model(logged_model_path)
        processed_df = preprocess_data(df, pipeline)
        expected_columns = model.metadata.get_input_schema().input_names()
        validate_data(processed_df, expected_columns)
        processed_df.drop(columns=['Disruption_Type', 'Disruption', 'Disruption_Type'], inplace=True)
        logger.info("Inferencing...")
        processed_df['Disruption_Forecast'] = model.predict(processed_df) < 0.5
        logger.info("Inference completed.")
        return processed_df
    except Exception as e:
        logger.error(f"An error occurred during inference. Error: {e}")
        raise
