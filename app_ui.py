import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from pipelines import ingestion_pippeline, preprocessing_pipeline, feature_pipeline, training_pipeline
from PIL import Image
import plotly.express as px
from pipelines.feature_pipeline import run_inventory_optimization_feature_pipeline
from pipelines.training_pipeline import run_inventory_training_pipeline, display_pipeline_metrics

# favicon = Image.open("static/Trigent_Logo.png")
st.set_page_config(
    page_title="Invenlytics | Trigent AXLR8 Labs",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# Add logo and title
logo_path = "https://trigent.com/wp-content/uploads/Trigent_Axlr8_Labs.png"
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="{logo_path}" alt="Trigent Logo" style="max-width:100%;">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("""
<style>

	.stTabs [data-baseweb="tab-list"] {
		gap: 3px;
    }

	.stTabs [data-baseweb="tab"] {
		height: 40px;
        white-space: pre-wrap;
		background-color: #fdba32;
		border-radius: 4px 4px 0px 0px;
		gap: 3px;
		padding: 10px 5px 10px 5px;
        color: white;
    }

	.stTabs [aria-selected="true"] {
  		background-color: #FFFFFF;
        color: #fdba32;
        border: 2px solid #fdba32;
        border-bottom: none;

	}

</style>""", unsafe_allow_html=True)

st.title("Manage Supplychain with ease.")
st.caption("Forecast DIsruption and Inventory demand to be ahead ")
st.divider()


def get_or_create_session_state_variable(key, default_value=None):
    """
    Retrieves the value of a variable from Streamlit's session state.
    If the variable doesn't exist, it creates it with the provided default value.

    Args:
        key (str): The key of the variable in session state.
        default_value (Any): The default value to assign if the variable doesn't exist.

    Returns:
        Any: The value of the session state variable.
    """
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]

# Initialize session state
get_or_create_session_state_variable('data')
get_or_create_session_state_variable('transformed_data')
get_or_create_session_state_variable('model')
get_or_create_session_state_variable('evaluation_results')
get_or_create_session_state_variable('length')
get_or_create_session_state_variable('ingestion_input', 'dynamic_supply_chain_logistics_dataset (1) (1).csv')
get_or_create_session_state_variable('ingestion_output', "/workspaces/invenlytics/pipelines/data/bronze_layer/supplychain_inventory.parquet")
get_or_create_session_state_variable('selected_features')
get_or_create_session_state_variable('features')

# App Layout
st.title("End-to-End Forecasting Pipeline")

# Tabs
tabs = st.tabs([
    "Ingest and Preprocess Data", "Transform Data", "EDA", "Feature Engineering", "Training Pipeline", "Evaluation", "Inference"
])

with tabs[0]:
    st.header("Ingest and Preprocess Data")
    option = st.radio("Choose Data Source:", ('Generate Synthetic Data', 'Upload CSV'))

    if option == 'Generate Synthetic Data':
        rows = st.slider("Number of Rows to Generate:", 100, 10000, 1000)
        
    elif option == 'Upload CSV': 
        uploaded_file = st.file_uploader("Upload CSV File")
        if uploaded_file is not None:
            st.session_state['ingestion_input'] = uploaded_file

    if st.button('Ingest and Preprocess Data', key='ingestion_button', use_container_width=True):
        # Step 1: Ingest Data
        ingestion_pippeline.setup_logging()
        st.session_state['data'] = ingestion_pippeline.run(
            input_file=st.session_state['ingestion_input'],
            output_file="/workspaces/invenlytics/pipelines/data/bronze_layer/Inventory_data_ingest_pipeline.parquet",
            total_rows=100000
        )

        # Step 2: Preprocess Data
        if st.session_state['data'] is not None:
            st.write("Raw Data Ingested:")
            st.dataframe(st.session_state['data'].head())

            # Define preprocessing output path
            preprocessed_path = "/workspaces/invenlytics/pipelines/data/silver_layer/preprocess_inventory_data.parquet"

            # Run preprocessing pipeline
            st.session_state['data'] = preprocessing_pipeline.run(
                source="/workspaces/invenlytics/pipelines/data/bronze_layer/Inventory_data_ingest_pipeline.parquet",
                dest=preprocessed_path
            )

            st.write("Preprocessed Data:")
            st.dataframe(st.session_state['data'].head())

            st.write("Preprocessed data saved to:", preprocessed_path)

# Tab 2: Transform Data
with tabs[1]:
    if st.session_state['data'] is None:
        st.warning("No data available for transformation. Please complete the data ingestion step.")
    else:
        columns = st.multiselect(
            "Select Features", 
            st.session_state['data'].columns.tolist(), 
            default=st.session_state['data'].columns.tolist(), 
            key='slsected_features'
        )

        if st.button("Transform Data", use_container_width=True):
            st.session_state['transformed_data'] = st.session_state['data'][columns]

            if st.session_state['transformed_data'] is not None:
                st.write("Transformed Data Preview:")
                st.dataframe(st.session_state['transformed_data'].head(100), hide_index=True)


# Tab 3: EDA
with tabs[2]:

    if st.session_state['transformed_data'] is not None:
        # # Filters panel
        # with st.expander("Filter Data", expanded=True):
        #     region_filter = st.multiselect("Select Region:", options=st.session_state['transformed_data']['Region'].unique(), default=st.session_state['transformed_data']['Region'].unique())
        #     supplier_filter = st.multiselect("Select Supplier:", options=st.session_state['transformed_data']['Supplier'].unique(), default=st.session_state['transformed_data']['Supplier'].unique())
            
        # # Apply filters
        # st.session_state['transformed_data'] = st.session_state['transformed_data'][(st.session_state['transformed_data']['Region'].isin(region_filter)) &
        #             (st.session_state['transformed_data']['Supplier'].isin(supplier_filter))]
        
        # Display dataset
        st.subheader("Dataset Overview")
        st.dataframe(st.session_state['transformed_data'].head())
        
        # 2x2 grid for visualizations
        col1, col2 = st.columns(2, gap='large')


        with col1:
            # Delivery time deviation analysis
            st.subheader("Delivery Time Deviation Analysis")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.violinplot(data=st.session_state['transformed_data'], x='Region', y='delivery_time_deviation', palette="viridis", ax=ax)
            ax.set_title("Delivery Time Deviation by Region")
            st.pyplot(fig)

        with col2:
            # Correlation heatmap
            st.subheader("Correlation Analysis")
            fig, ax = plt.subplots(figsize=(11, 8))
            corr = st.session_state['transformed_data'].select_dtypes(['int64', 'float64']).corr()[['disruption_likelihood_score', 'customs_clearance_time']]
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            # Disruption likelihood distribution
            st.subheader("Disruption Likelihood Analysis")
            fig, ax = plt.subplots(figsize=(11, 8))
            sns.histplot(st.session_state['transformed_data']['disruption_likelihood_score'], kde=True, color="blue", ax=ax)
            ax.set_title("Disruption Likelihood Distribution")
            st.pyplot(fig)

        with col1:
            st.subheader("Disruption Type Ratio Analysis")
            plt.figure(figsize=(8, 5))
            st.session_state['transformed_data']['Disruption_Type'].value_counts().head(10).plot(kind='bar', color='skyblue')
            plt.title("Top 10 Disruption Types")
            plt.xlabel("Disruption Type")
            plt.ylabel("Count")
            st.pyplot(plt.gcf())
        
        with col1:  
            # Risk classification by region
            st.subheader("Risk Classification by Region")
            plt.figure(figsize=(12, 6))
            sns.countplot(data=st.session_state['transformed_data'], x='Region', hue='risk_classification')
            plt.title("Risk Classification by Region")
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())
        

        with col2:
            st.subheader("Effect of Traffic Congestion on Delay")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=st.session_state['transformed_data'], x='traffic_congestion_level', y='delay_probability', hue='risk_classification')
            plt.title("Traffic Congestion vs Delay Probability")
            plt.show()
            st.pyplot(plt.gcf())

        # Feature selection and preprocessing
        features = ['fuel_consumption_rate', 'eta_variation_hours', 'traffic_congestion_level',
                    'warehouse_inventory_level', 'port_congestion_level', 'shipping_costs',
                    'supplier_reliability_score', 'lead_time_days', 'route_risk_level',
                    'customs_clearance_time', 'driver_behavior_score', 'fatigue_monitoring_score']
        target = 'disruption_likelihood_score'

# with tabs[2]:
#     st.header("Exploratory Data Analysis (EDA)")

#     if st.session_state['data'] is not None:
#         st.subheader("Dataset Overview")
#         st.dataframe(st.session_state['data'].head(), use_container_width=True)

#         col1, col2 = st.columns(2, gap='large')

#         # 1. Distribution of Historical Demand
#         with col1:
#             if 'Historical_Demand' in st.session_state['data'].columns:
#                 st.subheader("Distribution of Historical Demand")
#                 fig, ax = plt.subplots(figsize=(8, 5))
#                 sns.histplot(
#                     data=st.session_state['data'], 
#                     x='Historical_Demand', 
#                     bins=30, 
#                     kde=True, 
#                     color="blue", 
#                     ax=ax
#                 )
#                 ax.set_title("Distribution of Historical Demand")
#                 ax.set_xlabel("Historical Demand")
#                 ax.set_ylabel("Frequency")
#                 st.pyplot(fig)

#         # 2. Delivery Time Deviation Analysis
#         with col2:
#             if 'Region' in st.session_state['data'].columns and 'delivery_time_deviation' in st.session_state['data'].columns:
#                 st.subheader("Delivery Time Deviation Analysis by Region")
#                 fig, ax = plt.subplots(figsize=(8, 5))
#                 sns.violinplot(
#                     data=st.session_state['data'], 
#                     x='Region', 
#                     y='delivery_time_deviation', 
#                     palette="viridis", 
#                     ax=ax
#                 )
#                 ax.set_title("Delivery Time Deviation by Region")
#                 ax.set_xlabel("Region")
#                 ax.set_ylabel("Delivery Time Deviation")
#                 st.pyplot(fig)

#         # 3. Correlation Heatmap
#         st.subheader("Correlation Analysis")
#         fig, ax = plt.subplots(figsize=(12, 8))
#         numeric_data = st.session_state['data'].select_dtypes(include=['number'])
#         corr = numeric_data.corr()
#         sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
#         ax.set_title("Correlation Heatmap")
#         st.pyplot(fig)

#         # 4. Risk Classification by Region
#         st.subheader("Risk Classification by Region")
#         if 'Region' in st.session_state['data'].columns and 'risk_classification' in st.session_state['data'].columns:
#             fig, ax = plt.subplots(figsize=(12, 6))
#             sns.countplot(
#                 data=st.session_state['data'], 
#                 x='Region', 
#                 hue='risk_classification', 
#                 palette="Set3", 
#                 ax=ax
#             )
#             ax.set_title("Risk Classification by Region")
#             ax.set_xlabel("Region")
#             ax.set_ylabel("Count")
#             plt.xticks(rotation=45)
#             st.pyplot(fig)


with tabs[3]:
    st.header("Feature Engineering")

    # Initialize session state for data and features if not present
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'features' not in st.session_state:
        st.session_state['features'] = None

    # Check if data is available
    if st.session_state['data'] is None:
        st.warning("No data available for feature engineering. Please complete the data preprocessing step.")
    else:
        # Default feature list
        default_features = [
            "timestamp", "fuel_consumption_rate", "traffic_congestion_level",
            "warehouse_inventory_level", "supplier_reliability_score",
            "Historical_Demand", "Scheduled_Delivery", "Actual_Delivery", 
            "Region", "Delivery_Mode", "Freight_Cost"
        ]

        # Get available columns from the data
        available_columns = st.session_state['data'].columns.tolist()
        
        # Filter default features to only include those that exist in the data
        default_selected = [col for col in default_features if col in available_columns]

        # Multiselect for feature selection
        selected_features = st.multiselect(
            "Select Features for Engineering",
            options=available_columns,
            default=default_selected,
            key='feature_selection'
        )

        # Feature engineering button
        if st.button("Engineer Features"):
            if not selected_features:
                st.error("Please select at least one feature for engineering.")
            else:
                try:
                    # Define paths
                    source_path = "/workspaces/invenlytics/pipelines/data/silver_layer/preprocess_inventory_data.parquet"
                    destination_path = "/workspaces/invenlytics/pipelines/data/gold_layer/Feature_Engineering_data.parquet"

                    # Run the feature pipeline
                    st.session_state['features'] = run_inventory_optimization_feature_pipeline(
                        source=source_path,
                        selected_columns=selected_features,
                        dest=destination_path
                    )

                    if st.session_state['features'] is not None:
                        # Show success message and preview
                        st.success("Feature Engineering completed successfully!")
                        st.write("Preview of Engineered Features:")
                        st.dataframe(
                            data=st.session_state['features'].head(),
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Display feature information
                        st.write("Feature Engineering Summary:")
                        st.write(f"- Total features created: {len(st.session_state['features'].columns)}")
                        st.write(f"- Number of rows: {len(st.session_state['features'])}")
                    else:
                        st.error("Feature engineering completed but no features were generated.")
                    
                except Exception as e:
                    st.error(f"An error occurred during feature engineering: {str(e)}")
                    st.exception(e)
with tabs[4]:
    st.header("Training Pipeline")

    if st.session_state.get('features') is None:
        st.warning("No engineered features available. Please complete the feature engineering step first.")
        st.stop()  # Stop further execution in this tab
    else:
        # Multiselect to choose features for training
        model_features = st.multiselect(
            "Select Features for Training",
            options=st.session_state['features'].columns.tolist(),
            default=st.session_state['features'].columns.tolist()
        )

        # Dropdown to select the target column
        model_target = st.selectbox(
            "Select Target Variable",
            options=st.session_state['features'].columns.tolist(),
            key='target'
        )

        if st.button("Train Model"):
            try:
                # Ensure the target column is selected
                if not model_target:
                    st.error("Please select a target variable for training.")
                else:
                    # Define features (X) and target (y)
                    X = st.session_state['features'][model_features]
                    y = st.session_state['features'][model_target]

                    # Run the training pipeline
                    st.session_state['model'] = training_pipeline.run_inventory_training_pipeline(
                        data_source="/workspaces/invenlytics/pipelines/data/gold_layer/Feature_Engineering_data.parquet",
                        target_column=model_target,
                    )

                    st.success("Model Training Completed.")
            except Exception as e:
                st.error(f"An error occurred during model training: {e}")


with tabs[5]:
    st.header("Evaluation")

    # Debug: Display the session state
    st.write("Session State Debug:", st.session_state)

    if 'training_pipeline_metrics' in st.session_state:
        display_pipeline_metrics(st.session_state['training_pipeline_metrics'])
    else:
        st.warning("No training pipeline metrics found. Please run the training pipeline first.")

# Tab 7: Inference
# with tabs[6]:
#     st.header("Inference")

#     if st.session_state['model'] is not None:
#         uploaded_file = st.file_uploader("Upload Test Data")
#         if uploaded_file is not None:
#             test_data = pd.read_csv(uploaded_file)
#             predictions = st.session_state['model'].predict(test_data)
#             st.write("Predictions:")
#             st.dataframe(pd.DataFrame(predictions, columns=['Predicted']))