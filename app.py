# import streamlit as st
# import pandas as pd
# import numpy as np
# from pipelines.ingestion_pippeline import run as data_ingestion_pipeline
# from pipelines.feature_pipeline import run_inventory_optimization_feature_pipeline as feature_pipeline
# from pipelines.training_pipeline import run_inventory_training_pipeline
# from pipelines.inference_pipeline import run_inference
# import os
# import logging

# # Streamlit app configuration
# st.set_page_config(page_title="Pipeline App", layout="centered")

# st.title("Pipeline App for Supply Chain and Inventory Optimization")
# st.write("Train your model, forecast demand, and optimize your pipeline workflows.")

# def validate_and_prepare_dataset(data, required_columns):
#     """
#     Ensure the dataset contains all required columns and has valid types.
#     """
#     # Check for missing columns
#     missing_columns = [col for col in required_columns if col not in data.columns]
#     for col in missing_columns:
#         logging.warning(f"Column '{col}' is missing. Filling with default values.")
#         data[col] = 0  # Default value for missing columns

#     # Ensure numeric columns are of the correct type
#     numeric_columns = data.select_dtypes(include=[float, int]).columns
#     for col in required_columns:
#         if col in numeric_columns:
#             continue
#         if col in data.columns:
#             logging.warning(f"Column '{col}' has non-numeric data. Attempting to convert.")
#             data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

#     return data

# # Step 1: Upload Training Dataset
# st.header("Step 1: Upload Training Dataset")
# uploaded_training_file = st.file_uploader("Upload your training dataset (CSV):", type="csv")

# if uploaded_training_file:
#     training_data = pd.read_csv(uploaded_training_file)
#     st.write("Preview of training dataset:")
#     st.dataframe(training_data)
#     st.session_state['training_data'] = training_data

#     # Run data ingestion pipeline
#     if st.button("Run Data Ingestion Pipeline"):
#         with st.spinner("Running data ingestion pipeline..."):
#             output_file = "data/bronze_layer/ingested_training_data.parquet"
#             os.makedirs(os.path.dirname(output_file), exist_ok=True)
#             data_ingestion_pipeline(
#                 input_file=uploaded_training_file.name,
#                 output_file=output_file,
#                 total_rows=100000
#             )
#             st.session_state['ingested_training_path'] = output_file
#         st.success("Data ingestion complete!")

# # Step 2: Preprocess and Feature Engineering on Training Data
# st.header("Step 2: Preprocess and Apply Feature Engineering")

# ingested_path = st.session_state.get('ingested_training_path', None)
# if ingested_path:
#     if st.button("Run Preprocessing and Feature Engineering"):
#         with st.spinner("Processing training data..."):
#             # Save and preprocess data
#             preprocessed_training_path = "data/silver_layer/preprocessed_training_data.parquet"
#             os.makedirs(os.path.dirname(preprocessed_training_path), exist_ok=True)
#             pd.read_parquet(ingested_path).to_parquet(preprocessed_training_path, index=False)

#             # Apply feature pipeline
#             feature_engineered_path = "data/gold_layer/engineered_training_data.parquet"
#             os.makedirs(os.path.dirname(feature_engineered_path), exist_ok=True)
#             feature_pipeline(
#                 source=preprocessed_training_path,
#                 selected_columns=pd.read_parquet(ingested_path).columns.tolist(),
#                 dest=feature_engineered_path
#             )
#             st.session_state['feature_engineered_path'] = feature_engineered_path
#         st.success("Feature engineering complete!")

# # Step 3: Train Model
# st.header("Step 3: Train Model")

# pipeline_choice = st.selectbox("Select the pipeline to run:", ["Supply Chain", "Inventory Optimization"])
# training_path = st.session_state.get('feature_engineered_path', None)

# if training_path:
#     available_columns = pd.read_parquet(training_path).columns.tolist()
#     target_column = st.selectbox("Choose the target column for training:", available_columns)

#     if st.button("Run Training"):
#         with st.spinner("Training model..."):
#             if pipeline_choice == "Supply Chain":
#                 mlflow_model_path = "runs:/supply_chain_model_id/model"
#                 run_supply_chain_training_pipeline(
#                     data_source=training_path,
#                     target_column=target_column,
#                     mlflow_experiment_name="SupplyChain_Model",
#                     model_params={"learning_rate": 0.01, "num_leaves": 31, "max_depth": 10}
#                 )
#             else:
#                 mlflow_model_path = "runs:/inventory_model_id/model"
#                 run_inventory_training_pipeline(
#                     data_source=training_path,
#                     target_column=target_column,
#                     mlflow_experiment_name="Inventory_Model",
#                     model_params={"learning_rate": 0.01, "num_leaves": 31, "max_depth": 10}
#                 )
#             st.session_state['mlflow_model_path'] = mlflow_model_path
#         st.success(f"Training complete! Model saved at: {mlflow_model_path}")

# # Step 4: Generate or Upload Test Dataset
# st.header("Step 4: Provide Test Data")
# test_data_option = st.radio("How would you like to provide test data?", ["Generate Synthetic Data", "Upload Test Data"])

# # Validate and Prepare Test Data
# if test_data_option == "Upload Test Data":
#     uploaded_test_file = st.file_uploader("Upload your test dataset (CSV):", type="csv")
#     if uploaded_test_file:
#         test_data = pd.read_csv(uploaded_test_file)
#         required_columns = ["Historical_Demand", "Current_Stock", "Forecasted_Demand"]
#         test_data = validate_and_prepare_dataset(test_data, required_columns)
#         st.write("Preview of test dataset (validated):")
#         st.dataframe(test_data)
#         st.session_state['test_data'] = test_data
# elif test_data_option == "Generate Synthetic Data":
#     if st.button("Generate Test Data"):
#         with st.spinner("Generating synthetic test data..."):
#             training_columns = st.session_state['training_data'].columns.tolist()
#             synthetic_data = pd.DataFrame({
#                 col: np.random.rand(100) if col != target_column else None for col in training_columns
#             })
#             st.session_state['test_data'] = synthetic_data
#             st.write("Synthetic test data generated:")
#             st.dataframe(synthetic_data)

# # Step 5: Run Inference
# st.header("Step 5: Run Inference")

# if 'test_data' in st.session_state and 'mlflow_model_path' in st.session_state:
#     if st.button("Run Inference"):
#         with st.spinner("Running inference..."):
#             # Define paths
#             feature_engineered_test_path = "data/gold_layer/engineered_test_data.parquet"
#             test_data_path = "data/silver_layer/test_data.parquet"
#             os.makedirs(os.path.dirname(test_data_path), exist_ok=True)

#             # Save uploaded test data
#             test_data = st.session_state['test_data']
#             test_data.to_parquet(test_data_path, index=False)

#             # Run feature pipeline for test data
#             feature_pipeline(
#                 source=test_data_path,
#                 selected_columns=test_data.columns.tolist(),
#                 dest=feature_engineered_test_path
#             )

#             # Run inference
#             try:
#                 predictions = run_inference(
#                     df=pd.read_parquet(feature_engineered_test_path),
#                     logged_model_path=st.session_state['mlflow_model_path'],
#                     pipeline=feature_pipeline
#                 )
#                 st.session_state['predictions'] = predictions
#                 st.success("Inference complete! Predictions are ready.")
#             except Exception as e:
#                 st.error(f"Inference failed: {e}")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
# from pipelines.ingestion_pippeline import run as data_ingestion_pipeline
# from pipelines.preprocessing_pipeline import run as preprocessing_pipeline
# from pipelines.feature_pipeline import run_inventory_optimization_feature_pipeline
# from pipelines.training_pipeline import run_inventory_training_pipeline

# # Streamlit app configuration
# st.set_page_config(page_title="Pipeline App", layout="wide")

# st.title("Pipeline App for Inventory Optimization")
# st.write("A UI to run end-to-end pipelines: Data Ingestion, Preprocessing, Feature Engineering, and Training.")

# # Step 1: Upload CSV File
# st.header("Step 1: Upload Dataset")
# uploaded_file = st.file_uploader("Upload your dataset (CSV format):", type="csv")

# if uploaded_file:
#     # Read the uploaded file
#     uploaded_data = pd.read_csv(uploaded_file)
#     st.write("Preview of the uploaded dataset:")
#     st.dataframe(uploaded_data)
#     st.session_state['uploaded_data'] = uploaded_data

#     # Step 2: Run Data Ingestion
#     if st.button("Run Data Ingestion Pipeline"):
#         with st.spinner("Running data ingestion pipeline..."):
#             ingestion_output_path = "data/bronze_layer/ingested_data.parquet"
#             os.makedirs(os.path.dirname(ingestion_output_path), exist_ok=True)
#             data_ingestion_pipeline(
#                 input_file=uploaded_file.name,
#                 output_file=ingestion_output_path,
#                 total_rows=100000
#             )
#             ingested_data = pd.read_parquet(ingestion_output_path)
#             st.session_state['ingested_data'] = ingested_data
#             st.success("Data ingestion complete!")
#             st.write("Ingested dataset:")
#             st.dataframe(ingested_data)

# # Step 3: Preprocess and Feature Engineering
# if 'ingested_data' in st.session_state:
#     st.header("Step 3: Preprocess and Feature Engineering")
#     if st.button("Run Preprocessing and Feature Engineering"):
#         with st.spinner("Running preprocessing and feature pipeline..."):
#             preprocessed_output_path = "data/silver_layer/preprocessed_data.parquet"
#             feature_output_path = "data/gold_layer/feature_engineered_data.parquet"
#             os.makedirs(os.path.dirname(preprocessed_output_path), exist_ok=True)
#             os.makedirs(os.path.dirname(feature_output_path), exist_ok=True)

#             # Run preprocessing
#             preprocessing_pipeline(
#                 source="data/bronze_layer/ingested_data.parquet",
#                 dest=preprocessed_output_path
#             )

#             # Run feature pipeline (specific to inventory optimization)
#             run_inventory_optimization_feature_pipeline(
#                 source=preprocessed_output_path,
#                 selected_columns=st.session_state['ingested_data'].columns.tolist(),
#                 dest=feature_output_path
#             )

#             feature_data = pd.read_parquet(feature_output_path)
#             st.session_state['feature_data'] = feature_data
#             st.success("Preprocessing and feature engineering complete!")
#             st.write("Feature-engineered dataset:")
#             st.dataframe(feature_data)

# # Step 4: Select Target Column and Train Model
# if 'feature_data' in st.session_state:
#     st.header("Step 4: Select Target Column and Train Model")
#     target_column = st.selectbox("Select Target Column:", st.session_state['feature_data'].columns)
#     st.session_state['target_column'] = target_column

#     # Step 5: Train the Model
#     if st.button("Run Training"):
#         with st.spinner("Training model..."):
#             # Run the training pipeline
#             metrics = run_inventory_training_pipeline(
#                 data_source="data/gold_layer/feature_engineered_data.parquet",
#                 target_column=target_column,
#                 mlflow_experiment_name="Inventory_Optimization_Experiment"
#             )
#             st.success("Model training complete!")

#             # Aggregate metrics for better display
#             aggregated_metrics = {
#                 "train_rmse": np.mean([fold_metrics[0] for fold_metrics in metrics["rmse"]]),
#                 "valid_rmse": np.mean([fold_metrics[1] for fold_metrics in metrics["rmse"]]),
#                 "test_rmse": np.mean([fold_metrics[2] for fold_metrics in metrics["rmse"]]),
#                 "train_mae": np.mean([fold_metrics[0] for fold_metrics in metrics["mae"]]),
#                 "valid_mae": np.mean([fold_metrics[1] for fold_metrics in metrics["mae"]]),
#                 "test_mae": np.mean([fold_metrics[2] for fold_metrics in metrics["mae"]]),
#                 "train_r2": np.mean([fold_metrics[0] for fold_metrics in metrics["r2"]]),
#                 "valid_r2": np.mean([fold_metrics[1] for fold_metrics in metrics["r2"]]),
#                 "test_r2": np.mean([fold_metrics[2] for fold_metrics in metrics["r2"]]),
#             }

#             # Display aggregated metrics
#             st.write("Aggregated Training Metrics:")
#             st.json(aggregated_metrics)

#             # Optionally display raw fold-wise metrics
#             with st.expander("Show Fold-wise Metrics"):
#                 st.write("Fold-wise Metrics:")
#                 st.json(metrics)


import streamlit as st
import pandas as pd
import numpy as np
import os
from pipelines.ingestion_pippeline import run as data_ingestion_pipeline
from pipelines.preprocessing_pipeline import run as preprocessing_pipeline
from pipelines.feature_pipeline import run_inventory_optimization_feature_pipeline
from pipelines.training_pipeline import run_inventory_training_pipeline
from pipelines.inference_pipeline import run_inference

# Streamlit app configuration
st.set_page_config(page_title="Pipeline App", layout="wide")

st.title("Pipeline App for Inventory Optimization")
st.write("A UI to run end-to-end pipelines: Data Ingestion, Preprocessing, Feature Engineering, Training, and Inference.")

# Step 1: Upload CSV File
st.header("Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Upload your dataset (CSV format):", type="csv")

if uploaded_file:
    # Read the uploaded file
    uploaded_data = pd.read_csv(uploaded_file)
    st.write("Preview of the uploaded dataset:")
    st.dataframe(uploaded_data)
    st.session_state['uploaded_data'] = uploaded_data

    # Step 2: Run Data Ingestion
    if st.button("Run Data Ingestion Pipeline"):
        with st.spinner("Running data ingestion pipeline..."):
            ingestion_output_path = "data/bronze_layer/ingested_data.parquet"
            os.makedirs(os.path.dirname(ingestion_output_path), exist_ok=True)
            data_ingestion_pipeline(
                input_file=uploaded_file.name,
                output_file=ingestion_output_path,
                total_rows=100000
            )
            ingested_data = pd.read_parquet(ingestion_output_path)
            st.session_state['ingested_data'] = ingested_data
            st.success("Data ingestion complete!")
            st.write("Ingested dataset:")
            st.dataframe(ingested_data)

# Step 3: Preprocess and Feature Engineering
if 'ingested_data' in st.session_state:
    st.header("Step 3: Preprocess and Feature Engineering")
    if st.button("Run Preprocessing and Feature Engineering"):
        with st.spinner("Running preprocessing and feature pipeline..."):
            preprocessed_output_path = "data/silver_layer/preprocessed_data.parquet"
            feature_output_path = "data/gold_layer/feature_engineered_data.parquet"
            os.makedirs(os.path.dirname(preprocessed_output_path), exist_ok=True)
            os.makedirs(os.path.dirname(feature_output_path), exist_ok=True)

            # Run preprocessing
            preprocessing_pipeline(
                source="data/bronze_layer/ingested_data.parquet",
                dest=preprocessed_output_path
            )

            # Run feature pipeline (specific to inventory optimization)
            run_inventory_optimization_feature_pipeline(
                source=preprocessed_output_path,
                selected_columns=st.session_state['ingested_data'].columns.tolist(),
                dest=feature_output_path
            )

            feature_data = pd.read_parquet(feature_output_path)
            st.session_state['feature_data'] = feature_data
            st.success("Preprocessing and feature engineering complete!")
            st.write("Feature-engineered dataset:")
            st.dataframe(feature_data)

# Step 4: Select Target Column and Train Model
if 'feature_data' in st.session_state:
    st.header("Step 4: Select Target Column and Train Model")
    target_column = st.selectbox("Select Target Column:", st.session_state['feature_data'].columns)
    st.session_state['target_column'] = target_column

    # Step 5: Train the Model
    if st.button("Run Training"):
        with st.spinner("Training model..."):
            metrics = run_inventory_training_pipeline(
                data_source="data/gold_layer/feature_engineered_data.parquet",
                target_column=target_column,
                mlflow_experiment_name="Inventory_Optimization_Experiment"
            )
            st.success("Model training complete!")

            aggregated_metrics = {
                "train_rmse": np.mean([fold_metrics[0] for fold_metrics in metrics["rmse"]]),
                "valid_rmse": np.mean([fold_metrics[1] for fold_metrics in metrics["rmse"]]),
                "test_rmse": np.mean([fold_metrics[2] for fold_metrics in metrics["rmse"]]),
                "train_mae": np.mean([fold_metrics[0] for fold_metrics in metrics["mae"]]),
                "valid_mae": np.mean([fold_metrics[1] for fold_metrics in metrics["mae"]]),
                "test_mae": np.mean([fold_metrics[2] for fold_metrics in metrics["mae"]]),
                "train_r2": np.mean([fold_metrics[0] for fold_metrics in metrics["r2"]]),
                "valid_r2": np.mean([fold_metrics[1] for fold_metrics in metrics["r2"]]),
                "test_r2": np.mean([fold_metrics[2] for fold_metrics in metrics["r2"]]),
            }

            st.write("Aggregated Training Metrics:")
            st.json(aggregated_metrics)

            with st.expander("Show Fold-wise Metrics"):
                st.write("Fold-wise Metrics:")
                st.json(metrics)

            # Debugging: Show Run IDs
            st.write("MLflow Run IDs:")
            for fold in range(len(metrics["rmse"])):
                st.write(f"Fold {fold+1}: {mlflow.active_run().info.run_id}")

# Step 6: Run Inference
if 'feature_data' in st.session_state and 'target_column' in st.session_state:
    st.header("Step 6: Run Inference")
    model_path = st.text_input("Enter MLflow Model Path (e.g., runs:/model_id/model):", "")

    if st.button("Run Inference"):
        if model_path:
            with st.spinner("Running inference..."):
                try:
                    # Run the inference pipeline
                    feature_data = st.session_state['feature_data']
                    predictions = run_inference(feature_data, model_path)

                    # Display predictions
                    st.success("Inference complete!")
                    st.write("Predictions:")
                    st.dataframe(predictions)
                except Exception as e:
                    st.error(f"Inference failed: {e}")
        else:
            st.error("Please provide a valid MLflow model path.")

