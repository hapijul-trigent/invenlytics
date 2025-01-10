import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from pipelines.v2 import ingestion_pipeline, feature_pipeline, training_pipeline, forecast_pipeline
from PIL import Image
import plotly.express as px

favicon = Image.open("static/Trigent_Logo.png")
st.set_page_config(
    page_title="Invenlytics | Trigent AXLR8 Labs",
    page_icon=favicon,
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
    .tag {
        background-color: black !important;
        color: white !important;
        border: 1px solid white;
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
get_or_create_session_state_variable('ingestion_input', 'data/bronze_layer/dynamic_supply_chain_logistics_dataset.csv')
get_or_create_session_state_variable('ingestion_output', "data/bronze_layer/SupplyChain_Dataset.parquet")
get_or_create_session_state_variable('selected_features')
get_or_create_session_state_variable('features')
get_or_create_session_state_variable('metrics')
get_or_create_session_state_variable('logged_model_uri')


def visualize_lgbm_training_scores(scores):
    """
    Visualizes LightGBM model training scores in a grid format using Streamlit.

    Args:
        scores (dict): A dictionary containing metrics and their corresponding values.
                      Each metric is a 2D list with training, validation, and testing scores.
    """
    st.title("LightGBM Training Scores Visualization")

    averages = {}

    for metric, values in scores.items():
        st.subheader(f"{metric.capitalize()} Scores")

        # Convert the 2D list of values into a Pandas DataFrame for display
        df = pd.DataFrame(values, 
                           index=[f"Fold {i+1}" for i in range(len(values))], 
                           columns=["Training", "Validation", "Testing"])

        # Calculate average scores for each metric
        averages[metric] = df.mean(axis=0).tolist()

        # Create a two-column layout
        col1, col2 = st.columns(2)

        # Display the DataFrame in the first column
        with col1:
            st.write("Grid View:")
            st.dataframe(df.style.format(precision=4).background_gradient(cmap="YlGnBu"), use_container_width=True)

        # Display the line chart in the second column
        with col2:
            st.write("Line Chart View:")
            st.line_chart(df)

    # Visualize average scores
    # Transpose DataFrame for easier plotting
    avg_df = pd.DataFrame(averages, index=["Training", "Validation", "Testing"]).T
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_df.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_title("Average Scores by Metric")
    ax.set_xlabel("Score")
    ax.set_ylabel("Metrics")
    st.pyplot(fig)

def highlight_disruption_forecast(df):
    """
    Apply conditional formatting to highlight rows where the Disruption_Forecast column is True.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.io.formats.style.Styler: Styled DataFrame with highlighted rows.
    """
    def highlight_row(row):
        color = 'background-color: yellow' if row['Disruption_Forecast'] else ''
        return [color] * len(row)
    
    if 'Disruption_Forecast' in df.columns:
        return df.style.apply(highlight_row, axis=1)
    else:
        raise ValueError("The column 'Disruption_Forecast' does not exist in the DataFrame.")




# Tabs
tabs = st.tabs([
    "Ingest and Preprocess Data", "Transform Data", "EDA", "Feature Engineering", "Training Pipeline", "Evaluation", "Forecast"
])

# Tab 1: Ingest Data
with tabs[0]:
    st.header("Ingest and Preprocess Data")
    option = st.radio("Choose Data Source:", ('Generate Synthetic Data', 'Upload CSV'))
    rows = 0
    if option == 'Generate Synthetic Data':
        rows = st.slider("Number of Rows to Generate:", 100, 100000, 1000)
        
    elif option == 'Upload CSV': 
        uploaded_file = st.file_uploader("Upload CSV File")
        if uploaded_file is not None:
            st.session_state['ingestion_input'] = uploaded_file
    if st.button('Ingest', key='ingestion_button', use_container_width=True):
        ingestion_pipeline.setup_logging()
        st.session_state['data'] = ingestion_pipeline.run(
            input_file=st.session_state['ingestion_input'],
            output_file=st.session_state['ingestion_output'],
            total_rows=rows
        )
    if st.session_state['data'] is not None:
        st.write("Preview of Data:")
        st.dataframe(st.session_state['data'], use_container_width=True, hide_index=True)

# Tab 2: Transform Data
with tabs[1]:
    st.header("Transform Data")

    if st.session_state['data'] is not None:
        columns = st.multiselect("Select Features", st.session_state['data'].columns.tolist(), default=st.session_state['data'].columns.tolist(), key='slsected_features')

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

# Tab 4: Feature Engineering
with tabs[3]:
    st.header("Feature Engineering")

    if st.session_state['transformed_data'] is not None:
        default_features = [
                "Scheduled_Delivery", "Actual_Delivery", "Region", "Delivery_Mode", "Supplier",
                "Weather_Conditions",  "traffic_congestion_level", "port_congestion_level", "weather_condition_severity", 
                "fuel_consumption_rate", "driver_behavior_score", "fatigue_monitoring_score", 'supplier_reliability_score'
        ]
        selected_features = st.multiselect("Select Features", st.session_state['transformed_data'].columns.tolist(), default=default_features)
        st.session_state['model_target'] = st.selectbox("Select Target", st.session_state['transformed_data'].columns.tolist(), key='target')
        if st.button("Engineer Features"):
            st.session_state['features'] = feature_pipeline.run_supplychain_disruption_feature_pipeline(
                source="/workspaces/invenlytics/data/silver_layer/preprocessed_dynamic_supply_chain_logistics_dataset.parquet", 
                selected_columns=selected_features+[st.session_state['model_target']], 
                dest="/workspaces/invenlytics/data/gold_layer/SupplyChain_DisruptionFeatures.parquet"
            )

            st.dataframe(data= st.session_state['features'], hide_index=True, use_container_width=True)

def visualize_feature_importance(model, model_type="lightgbm"):
    """
    Visualizes feature importance for a trained LightGBM or XGBoost model.
    
    Parameters:
    - model: Trained model object (LightGBM or XGBoost).
    - model_type: Type of the model ("lightgbm" or "xgboost").
    """
    # Extract feature importance based on the model type
    if model_type.lower() == "lightgbm":
        feature_importance = pd.DataFrame({
            'Feature': model.feature_name(),
            'Importance': model.feature_importance()
        })
    elif model_type.lower() == "xgboost":
        feature_importance = pd.DataFrame({
            'Feature': list(model.get_score().keys()),
            'Importance': list(model.get_score().values())
        })
    else:
        st.error("Invalid model type. Please use 'lightgbm' or 'xgboost'.")
        return

    # Sort by importance in descending order
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # Get top 7 features
    top_features = feature_importance.head(7)

    # Streamlit app UI
    st.title("Feature Importance Visualizer")
    st.write(f"This app visualizes the top 7 feature importances of a trained {model_type.upper()} model.")

    # Options for chart type
    chart_type = st.selectbox("Select chart type:", ["Bar Chart", "Horizontal Bar Chart"])

    # Plot the top 7 feature importance
    fig, ax = plt.subplots(figsize=(8, 6))

    if chart_type == "Bar Chart":
        ax.bar(top_features['Feature'], top_features['Importance'], color='skyblue')
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance")
        ax.set_title(f"Top 7 Feature Importance - {model_type.upper()} - Bar Chart")
        ax.tick_params(axis='x', rotation=45)
    elif chart_type == "Horizontal Bar Chart":
        ax.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")
        ax.set_title(f"Top 7 Feature Importance - {model_type.upper()} - Horizontal Bar Chart")

    st.pyplot(fig)



# Tab 5: Training Pipeline
with tabs[4]:
    st.header("Training Pipeline")
    select_model = st.selectbox('Select Model', options=['LightGBM', 'XGBoost'])
    if st.session_state['features'] is not None:
        # st.session_state['timestamp'] = st.selectbox("Select Time Stamp", st.session_state['features'].columns.tolist(), key='timestampid')
        st.session_state['model_features'] = st.multiselect("Select Features", st.session_state['features'].columns.tolist(), default=st.session_state['features'].columns.tolist())
        st.session_state['model_target'] = st.selectbox("Select Target", st.session_state['features'].columns.tolist())
        
        if st.button("Train Model"):
            with st.spinner('Training....'):
                if select_model == 'LightGBM':
                    st.session_state['model'],  st.session_state['metrics'], st.session_state['logged_model_uri'] = training_pipeline.run_lightgbm_training_pipeline(
                    data_source="/workspaces/invenlytics/data/gold_layer/SupplyChain_DisruptionFeatures.parquet",
                    target_column=st.session_state['model_target']
                    )
                else:
                    st.session_state['model'],  st.session_state['metrics']= training_pipeline.run_xgboost_training_pipeline(
                    data_source="/workspaces/invenlytics/data/gold_layer/SupplyChain_DisruptionFeatures.parquet",
                    target_column=st.session_state['model_target']
                    )
            
            st.success("Model Training Completed.")
    if st.session_state['model']:
            visualize_feature_importance(st.session_state['model'], model_type= 'lightgbm' if select_model == 'LightGBM' else 'xgboost')
            from pprint import pprint
            pprint(st.session_state['metrics'])
            st.write(st.session_state['logged_model_uri'])
            

# Tab 6: Evaluation
with tabs[5]:
    st.header("Evaluation")
    if st.session_state['data'] is not None and st.session_state['transformed_data'] is not None and st.session_state['features'] is not None and st.session_state['metrics'] is not None:
        visualize_lgbm_training_scores(scores=st.session_state['metrics'])

# # Tab 7: Inference
with tabs[6]:
    st.header("Forecast")

    if st.session_state['model'] is not None:
        uploaded_file = st.file_uploader("Upload Test Data")
        if uploaded_file is not None:
            forecasts = forecast_pipeline.forecast(
                input_file=uploaded_file,
                logged_model_path=st.session_state['logged_model_uri']
            )
            # forecasts = highlight_disruption_forecast(df=forecasts)
            st.write("Forecasts:")
            st.dataframe(forecasts)
