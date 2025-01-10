# import os
# import numpy as np
# import pandas as pd
# from pipelines import inference_pipeline
# import logging

# logger = logging.getLogger()

# if __name__ == "__main__":
#     try:
#         # Load the dataset
#         input_data_path = "data/gold_layer/SupplyChain_Invetory_Dataset.parquet"
#         logger.info(f"Loading data from {input_data_path}...")
#         df = pd.read_parquet(input_data_path)

#         # Define the model path
#         model_path = "runs:/b8cb68cd92a44bd78f6992bef60e03ac/model"  # Replace with your actual model path

#         # Run inference (corrected function call)
#         predictions = inference_pipeline.run_inference(df, model_path)

#         # Ensure output directory exists
#         output_directory = "data/inference_layer"
#         os.makedirs(output_directory, exist_ok=True)

#         # Save predictions
#         output_data_path = os.path.join(output_directory, "Inventory_Demand_Forecast.parquet")
#         predictions.to_parquet(output_data_path, index=False)
#         logger.info(f"Predictions saved to {output_data_path}.")
#     except Exception as e:
#         logger.error(f"An error occurred in the inference pipeline. Error: {e}")


import numpy as np
import pandas as pd
from pipelines import inference_pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

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
