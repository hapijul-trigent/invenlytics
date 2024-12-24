import numpy as np
import pandas as pd
from pipelines import feature_pipeline, inference_pipeline
import logging
logger = logging.getLogger()

if __name__ == "__main__":
    # Load the data
    try:
        df = pd.read_parquet('data/gold_layer/SupplyChainI_Disruption_Dataset.parquet')
        
        # Define model path
        logged_model = 'runs:/cea2d26a4e7e4373b91f849e1e821321/model'
        
        # Run inference
        predictions = inference_pipeline.run_inference(df, logged_model, pipeline=None)
        print(predictions)
    except Exception as e:
        logger.error(f"Error in main process: {e}")
