from pipelines import ingestion_pippeline, preprocessing_pipeline, feature_pipeline, training_pipeline
import logging
if __name__ == '__main__':
    disruption_forecast_features = ["Scheduled_Delivery", "Disruption_Type", "Region", "Delivery_Mode", "Weather_Risk", "Supplier_Reliability", "Port_Congestion", "Delay_Duration"]
    inventory_features_columns =[
    "Current_Stock",
    "Forecasted_Demand",
    "Historical_Demand",
    "Scheduled_Delivery",
    "Lead_Time_Days",
    # Add other required columns here
    ]


    ingestion_pippeline.setup_logging()
    ingestion_pippeline.run(
        input_file="/workspaces/invenlytics/dynamic_supply_chain_logistics_dataset (1) (1).csv",
        output_file="/workspaces/invenlytics/data/bronze_layer/supply_chain_datageneration.parquet",
        total_rows=100000
    )

    preprocessing_pipeline.setup_logging()
    preprocessing_pipeline.run(
        source="/workspaces/invenlytics/data/bronze_layer/supply_chain_datageneration.parquet", 
        dest="/workspaces/invenlytics/data/silver_layer/preprocessed_supply_chain_preprocessed_file.parquet"
    )
    feature_pipeline.run_inventory_optimization_feature_pipeline(
        source="/workspaces/invenlytics/data/silver_layer/preprocessed_supply_chain_preprocessed_file.parquet", 
        selected_columns=inventory_features_columns, 
        dest="/workspaces/invenlytics/data/gold_layer/SupplyChain_Invetory_Dataset.parquet"
    )
    training_pipeline.run_inventory_training_pipeline(
        data_source="/workspaces/invenlytics/data/gold_layer/SupplyChain_Invetory_Dataset.parquet",
        target_column='Historical_Demand',
    )
    # training_pipeline.run_inventory_rf_training_pipeline(
    #     data_source="/workspaces/invenlytics/data/gold_layer/SupplyChain_Invetory_Dataset.parquet",
    #     target_column='Historical_Demand',
    # )
    # training_pipeline.run_stockout_risk_training_pipeline(
    # data_source="data/gold_layer/SupplyChain_Invetory_Dataset.parquet",
    # target_column="Stockout_Risk"
    # )

    # training_pipeline.run_overstock_risk_training_pipeline(
    # data_source="data/gold_layer/SupplyChain_Invetory_Dataset.parquet",
    # target_column='Overstock_Risk'
    # )

