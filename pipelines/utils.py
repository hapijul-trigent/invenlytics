import pandas as pd
from sklearn.decomposition import PCA
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_pca(data, numeric_columns, n_components=0.95):
    """
    Apply PCA to the numeric columns of the dataset.
    
    Parameters:
    - data (pd.DataFrame): Input DataFrame.
    - numeric_columns (list): List of numeric columns to apply PCA on.
    - n_components (int, float): Number of components to keep, or the variance ratio to preserve.
    
    Returns:
    - pd.DataFrame: DataFrame with PCA components appended.
    """
    try:
        logger.info("Applying PCA...")
        
        # Ensure numeric columns exist in the DataFrame
        if not all(col in data.columns for col in numeric_columns):
            missing_cols = [col for col in numeric_columns if col not in data.columns]
            raise ValueError(f"Missing columns for PCA: {missing_cols}")

        # Extract numeric data
        numeric_data = data[numeric_columns]

        # Initialize PCA
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(numeric_data)

        # Create a DataFrame for PCA components
        pca_columns = [f"PCA_Component_{i+1}" for i in range(pca_transformed.shape[1])]
        pca_df = pd.DataFrame(pca_transformed, columns=pca_columns, index=data.index)

        # Log explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        logger.info(f"Explained variance ratio: {explained_variance}")
        logger.info(f"Total variance explained by PCA: {explained_variance.sum()}")

        # Append PCA components to the original DataFrame
        data_with_pca = pd.concat([data, pca_df], axis=1)

        return data_with_pca

    except Exception as e:
        logger.error(f"Error occurred during PCA: {e}", exc_info=True)
        raise
