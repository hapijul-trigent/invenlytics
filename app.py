# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import base64
# from pipelines.ingestion_pippeline import run as ingestion_pipeline
# from pipelines.preprocessing_pipeline import run as preprocessing_pipeline
# from pipelines.training_pipeline import run_inventory_training_pipeline

# # Streamlit app configuration
# st.set_page_config(page_title="Pipeline Selection App", layout="centered")

# st.title("Pipeline Selection App")
# st.write("This application helps you manage and run supply chain or inventory optimization pipelines.")

# # Step 1: Upload Dataset
# st.header("Step 1: Upload Your Data")
# uploaded_file = st.file_uploader("Choose a CSV file to upload:", type="csv")

# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)
#     st.write("Preview of uploaded data:")
#     st.dataframe(data)
#     st.session_state['data'] = data

# # Step 2: Run Data Ingestion
# st.header("Step 2: Run Data Ingestion")

# if 'data' in st.session_state:
#     if st.button("Run Ingestion"):
#         with st.spinner("Ingesting data..."):
#             ingestion_pipeline(
#                 input_file=uploaded_file,
#                 output_file="data/bronze_layer/SupplyChain_Dataset.parquet",
#                 total_rows=1000  # Adjust as needed
#             )
#         st.success("Data ingestion complete!")

# # Step 3: Run Preprocessing
# st.header("Step 3: Run Preprocessing")

# if 'data' in st.session_state:
#     if st.button("Run Preprocessing"):
#         with st.spinner("Preprocessing data..."):
#             preprocessing_pipeline(
#                 source="data/bronze_layer/SupplyChain_Dataset.parquet",
#                 dest="data/silver_layer/preprocessed_SupplyChain_Dataset.parquet"
#             )
#         st.success("Data preprocessing complete!")

# # Step 4: Choose Pipeline
# st.header("Step 4: Choose Your Pipeline")
# pipeline_choice = st.selectbox("Select the pipeline to run:", ["Supply Chain", "Inventory Optimization"])

# # Step 5: Select Target Column
# if 'data' in st.session_state:
#     st.header("Step 5: Select Target Column")
#     st.write("Available columns in the dataset:", st.session_state['data'].columns.tolist())
#     target_column = st.selectbox("Choose the target column:", st.session_state['data'].columns.tolist())
#     st.session_state['target_column'] = target_column

# # Step 6: Configure Training Parameters
# st.header("Step 6: Configure Training Parameters")

# params = {
#     "learning_rate": st.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01),
#     "num_leaves": st.slider("Number of Leaves", min_value=2, max_value=100, value=31),
#     "max_depth": st.slider("Max Depth", min_value=1, max_value=50, value=10)
# }

# # Step 7: Run Training
# st.header("Step 7: Run Training")

# if 'target_column' in st.session_state:
#     if st.button("Run Training"):
#         with st.spinner("Training model..."):
#             if pipeline_choice == "Supply Chain":
#                 run_supply_chain_training_pipeline(
#                     data_source="data/silver_layer/preprocessed_SupplyChain_Dataset.parquet",
#                     target_column=st.session_state['target_column'],
#                     mlflow_experiment_name="SupplyChain_Model",
#                     model_params=params
#                 )
#             elif pipeline_choice == "Inventory Optimization":
#                 run_inventory_training_pipeline(
#                     data_source="data/silver_layer/preprocessed_SupplyChain_Dataset.parquet",
#                     target_column=st.session_state['target_column'],
#                     mlflow_experiment_name="Inventory_Model",
#                     model_params=params
#                 )
#         st.success(f"{pipeline_choice} training complete!")

# # Step 8: Results Visualization (Optional)
# st.header("Step 8: Visualize Results (Optional)")

# if st.button("View Results"):
#     # Placeholder for predictions
#     predictions = pd.DataFrame({"predictions": [1, 2, 3, 4, 5]})
#     st.write("Prediction Results:")
#     st.dataframe(predictions)
    
#     # Visualization
#     st.subheader("Prediction Distribution")
#     fig = px.histogram(predictions, x="predictions")
#     st.plotly_chart(fig)

#     # Download Results
#     csv = predictions.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
#     st.markdown(href, unsafe_allow_html=True)


# import streamlit as st
# import pandas as pd
# from pipelines.ingestion_pippeline import run as ingestion_pipeline
# from pipelines.preprocessing_pipeline import run as preprocessing_pipeline
# from pipelines.training_pipeline import run_inventory_training_pipeline

# # Streamlit app configuration
# st.set_page_config(page_title="Pipeline Selection App", layout="centered")

# st.title("Pipeline Selection App")
# st.write("This application helps you manage and run supply chain or inventory optimization pipelines.")

# # Step 1: Upload Dataset
# st.header("Step 1: Upload Your Data")
# uploaded_file = st.file_uploader("Choose a CSV file to upload:", type="csv")

# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)
#     st.write("Preview of uploaded data:")
#     st.dataframe(data)
#     st.session_state['data'] = data

# # Step 2: Run Data Ingestion
# st.header("Step 2: Run Data Ingestion")

# if 'data' in st.session_state:
#     if st.button("Run Ingestion"):
#         with st.spinner("Ingesting data..."):
#             ingestion_pipeline(
#                 input_file=uploaded_file,
#                 output_file="data/bronze_layer/SupplyChain_Dataset.parquet",
#                 total_rows=1000  # Adjust as needed
#             )
#         st.success("Data ingestion complete!")

# # Step 3: Run Preprocessing
# st.header("Step 3: Run Preprocessing")

# if 'data' in st.session_state:
#     if st.button("Run Preprocessing"):
#         with st.spinner("Preprocessing data..."):
#             preprocessing_pipeline(
#                 source="data/bronze_layer/SupplyChain_Dataset.parquet",
#                 dest="data/silver_layer/preprocessed_SupplyChain_Dataset.parquet"
#             )
#         st.success("Data preprocessing complete!")

# # Step 4: Choose Pipeline and Training Path
# st.header("Step 4: Choose Pipeline and Training Path")
# pipeline_choice = st.selectbox("Select the pipeline to run:", ["Supply Chain", "Inventory Optimization"])

# # Set the dataset path based on the pipeline choice
# if pipeline_choice == "Supply Chain":
#     training_data_path = "data/gold_layer/SupplyChainI_Disruption_Dataset.parquet"
# else:
#     training_data_path = "data/gold_layer/SupplyChain_Invetory_Dataset.parquet"

# # Step 5: Verify Preprocessed Data and Select Target Column
# st.header("Step 5: Verify Data and Select Target Column")

# try:
#     # Load training data
#     training_data = pd.read_parquet(training_data_path)
#     st.write("Columns in the training dataset:")
#     st.dataframe(training_data.head())
#     available_columns = training_data.columns.tolist()
    
#     # Dynamic target column selection
#     target_column = st.selectbox("Choose the target column for training:", available_columns)
#     st.session_state['target_column'] = target_column
# except Exception as e:
#     st.error(f"Error loading training data from {training_data_path}. Ensure the preprocessing step is completed.")

# # Step 6: Configure Training Parameters
# st.header("Step 6: Configure Training Parameters")

# params = {
#     "learning_rate": st.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01),
#     "num_leaves": st.slider("Number of Leaves", min_value=2, max_value=100, value=31),
#     "max_depth": st.slider("Max Depth", min_value=1, max_value=50, value=10)
# }

# # Step 7: Run Training
# st.header("Step 7: Run Training")

# if 'target_column' in st.session_state:
#     if st.button("Run Training"):
#         target_column = st.session_state['target_column']
#         with st.spinner(f"Training model on {pipeline_choice} pipeline..."):
#             if pipeline_choice == "Supply Chain":
#                 run_supply_chain_training_pipeline(
#                     data_source=training_data_path,
#                     target_column=target_column,
#                     mlflow_experiment_name="SupplyChain_Model",
#                     model_params=params
#                 )
#             elif pipeline_choice == "Inventory Optimization":
#                 run_inventory_training_pipeline(
#                     data_source=training_data_path,
#                     target_column=target_column,
#                     mlflow_experiment_name="Inventory_Model",
#                     model_params=params
#                 )
#         st.success(f"Training complete using pipeline: {pipeline_choice} and target column: {target_column}!")

# # Step 8: Results Visualization (Optional)
# st.header("Step 8: Visualize Results (Optional)")

# if st.button("View Results"):
#     # Placeholder for predictions
#     predictions = pd.DataFrame({"predictions": [1, 2, 3, 4, 5]})
#     st.write("Prediction Results:")
#     st.dataframe(predictions)


# import streamlit as st
# import pandas as pd
# from pipelines.training_pipeline import run_inventory_training_pipeline
# from pipelines.inference_pipeline import run_inference
# from pipelines import feature_pipeline

# # Streamlit app configuration
# st.set_page_config(page_title="Pipeline App with Inference", layout="centered")

# st.title("Pipeline App with Inference")
# st.write("Train your model and forecast demand with test data.")

# # Step 1: Upload Training Dataset
# st.header("Step 1: Upload Training Dataset")
# uploaded_training_file = st.file_uploader("Upload your training dataset (CSV):", type="csv")

# if uploaded_training_file is not None:
#     training_data = pd.read_csv(uploaded_training_file)
#     st.write("Preview of training dataset:")
#     st.dataframe(training_data)
#     st.session_state['training_data'] = training_data

# # Step 2: Preprocess Training Data
# st.header("Step 2: Preprocess Training Data")

# if 'training_data' in st.session_state:
#     if st.button("Run Preprocessing"):
#         with st.spinner("Preprocessing training data..."):
#             # Placeholder preprocessing logic
#             preprocessed_training_path = "data/silver_layer/preprocessed_training_data.parquet"
#             training_data.to_parquet(preprocessed_training_path, index=False)
#             st.session_state['preprocessed_training_path'] = preprocessed_training_path
#         st.success("Training data preprocessing complete!")

# # Step 3: Train Model
# st.header("Step 3: Train Model")

# pipeline_choice = st.selectbox("Select the pipeline to run:", ["Supply Chain", "Inventory Optimization"])
# training_path = st.session_state.get('preprocessed_training_path', None)

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

# # Step 4: Upload Test Dataset
# st.header("Step 4: Upload Test Dataset")
# uploaded_test_file = st.file_uploader("Upload your test dataset (CSV):", type="csv")

# if uploaded_test_file:
#     test_data = pd.read_csv(uploaded_test_file)
#     st.write("Preview of test dataset:")
#     st.dataframe(test_data)
#     st.session_state['test_data'] = test_data

# # Step 5: Run Inference
# st.header("Step 5: Run Inference")

# if 'test_data' in st.session_state and 'mlflow_model_path' in st.session_state:
#     if st.button("Run Inference"):
#         with st.spinner("Running inference..."):
#             predictions = run_inference(
#                 df=st.session_state['test_data'],
#                 logged_model_path=st.session_state['mlflow_model_path'],
#                 pipeline=feature_pipeline
#             )
#             st.session_state['predictions'] = predictions
#         st.success("Inference complete! Predictions are ready.")

# # Step 6: Display and Download Predictions
# if 'predictions' in st.session_state:
#     st.header("Step 6: Forecasted Demand")
#     predictions = st.session_state['predictions']
#     st.write("Forecasted demand:")
#     st.dataframe(predictions)

#     # Download button
#     csv = predictions.to_csv(index=False)
#     st.download_button(
#         label="Download Predictions as CSV",
#         data=csv,
#         file_name="forecasted_demand.csv",
#         mime="text/csv"
#     )


import streamlit as st
import pandas as pd
import numpy as np
from pipelines.feature_pipeline import run_inventory_optimization_feature_pipeline as feature_pipeline
from pipelines.training_pipeline import run_inventory_training_pipeline
from pipelines.inference_pipeline import run_inference

# Streamlit app configuration
st.set_page_config(page_title="Pipeline App with Feature Pipeline", layout="centered")

st.title("Pipeline App with Feature Pipeline")
st.write("Train your model and forecast demand using synthetic test data.")

# Step 1: Upload Training Dataset
st.header("Step 1: Upload Training Dataset")
uploaded_training_file = st.file_uploader("Upload your training dataset (CSV):", type="csv")

if uploaded_training_file:
    training_data = pd.read_csv(uploaded_training_file)
    st.write("Preview of training dataset:")
    st.dataframe(training_data)
    st.session_state['training_data'] = training_data

# Step 2: Preprocess and Feature Engineering on Training Data
st.header("Step 2: Preprocess and Apply Feature Engineering")

if 'training_data' in st.session_state:
    if st.button("Run Preprocessing and Feature Engineering"):
        with st.spinner("Processing training data..."):
            # Save and preprocess data
            preprocessed_training_path = "data/silver_layer/preprocessed_training_data.parquet"
            st.session_state['training_data'].to_parquet(preprocessed_training_path, index=False)
            
            # Apply feature pipeline
            feature_engineered_path = "data/gold_layer/engineered_training_data.parquet"
            feature_pipeline(
                source=preprocessed_training_path,
                selected_columns=st.session_state['training_data'].columns.tolist(),
                dest=feature_engineered_path
            )
            st.session_state['feature_engineered_path'] = feature_engineered_path
        st.success("Feature engineering complete!")

# Step 3: Train Model
st.header("Step 3: Train Model")

pipeline_choice = st.selectbox("Select the pipeline to run:", ["Supply Chain", "Inventory Optimization"])
training_path = st.session_state.get('feature_engineered_path', None)

if training_path:
    available_columns = pd.read_parquet(training_path).columns.tolist()
    target_column = st.selectbox("Choose the target column for training:", available_columns)

    if st.button("Run Training"):
        with st.spinner("Training model..."):
            if pipeline_choice == "Supply Chain":
                mlflow_model_path = "runs:/supply_chain_model_id/model"
                run_supply_chain_training_pipeline(
                    data_source=training_path,
                    target_column=target_column,
                    mlflow_experiment_name="SupplyChain_Model",
                    model_params={"learning_rate": 0.01, "num_leaves": 31, "max_depth": 10}
                )
            else:
                mlflow_model_path = "runs:/inventory_model_id/model"
                run_inventory_training_pipeline(
                    data_source=training_path,
                    target_column=target_column,
                    mlflow_experiment_name="Inventory_Model",
                    model_params={"learning_rate": 0.01, "num_leaves": 31, "max_depth": 10}
                )
            st.session_state['mlflow_model_path'] = mlflow_model_path
        st.success(f"Training complete! Model saved at: {mlflow_model_path}")

# Step 4: Generate or Upload Test Dataset
st.header("Step 4: Provide Test Data")
test_data_option = st.radio("How would you like to provide test data?", ["Generate Synthetic Data", "Upload Test Data"])

if test_data_option == "Upload Test Data":
    uploaded_test_file = st.file_uploader("Upload your test dataset (CSV):", type="csv")
    if uploaded_test_file:
        test_data = pd.read_csv(uploaded_test_file)
        st.write("Preview of test dataset:")
        st.dataframe(test_data)
        st.session_state['test_data'] = test_data
elif test_data_option == "Generate Synthetic Data":
    if st.button("Generate Test Data"):
        with st.spinner("Generating synthetic test data..."):
            training_columns = st.session_state['training_data'].columns.tolist()
            synthetic_data = pd.DataFrame({
                col: np.random.rand(100) if col != target_column else None for col in training_columns
            })
            st.session_state['test_data'] = synthetic_data
            st.write("Synthetic test data generated:")
            st.dataframe(synthetic_data)

# Step 5: Run Inference
st.header("Step 5: Run Inference")

if 'test_data' in st.session_state and 'mlflow_model_path' in st.session_state:
    if st.button("Run Inference"):
        with st.spinner("Running inference..."):
            feature_engineered_test_path = "data/gold_layer/engineered_test_data.parquet"
            test_data = st.session_state['test_data']

            # Apply feature pipeline to test data
            test_data.to_parquet("data/silver_layer/test_data.parquet", index=False)
            feature_pipeline(
                source="data/silver_layer/test_data.parquet",
                selected_columns=test_data.columns.tolist(),
                dest=feature_engineered_test_path
            )

            # Run inference
            predictions = run_inference(
                df=pd.read_parquet(feature_engineered_test_path),
                logged_model_path=st.session_state['mlflow_model_path'],
                pipeline=feature_pipeline
            )
            st.session_state['predictions'] = predictions
        st.success("Inference complete! Predictions are ready.")

# Step 6: Display and Download Predictions
if 'predictions' in st.session_state:
    st.header("Step 6: Forecasted Demand")
    predictions = st.session_state['predictions']
    st.write("Forecasted demand:")
    st.dataframe(predictions)

    # Download button
    csv = predictions.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="forecasted_demand.csv",
        mime="text/csv"
    )
