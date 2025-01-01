import os
import numpy as np
import pandas as pd
from pipelines import feature_pipeline, inference_pipeline
import logging

logger = logging.getLogger()

if __name__ == "__main__":
    try:
        # Load the dataset
        input_data_path = "data/gold_layer/SupplyChain_Invetory_Dataset.parquet"
        logger.info(f"Loading data from {input_data_path}...")
        df = pd.read_parquet(input_data_path)

        # Define the model path
        model_path = "runs:/e5d74812117e4ce285a523937725991f/model"  # Replace with your actual model path

        # Run inference
        predictions = inference_pipeline.run_inference(df, model_path, feature_pipeline)

        # Ensure output directory exists
        output_directory = "data/inference_layer"
        os.makedirs(output_directory, exist_ok=True)

        # Save predictions
        output_data_path = os.path.join(output_directory, "Inventory_Demand_Forecast.parquet")
        predictions.to_parquet(output_data_path, index=False)
        logger.info(f"Predictions saved to {output_data_path}.")
    except Exception as e:
        logger.error(f"An error occurred in the inference pipeline. Error: {e}")



