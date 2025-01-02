from pipelines import ingestion_pippeline, preprocessing_pipeline, feature_pipeline, training_pipeline

import logging
if __name__ == '__main__':
    disruption_forecast_features = ["Scheduled_Delivery", "Disruption_Type", "Region", "Delivery_Mode", "Weather_Risk", "Supplier_Reliability", "Port_Congestion", "Delay_Duration"]
    inventory_features_columns = ["Scheduled_Delivery", "Historical_Demand"]

    ingestion_pippeline.setup_logging()
    ingestion_pippeline.run(
        output_file="data/bronze_layer/SupplyChain_Dataset_100k.parquet",
        input_file="dynamic_supply_chain_logistics_dataset (1).csv",
        total_rows_required=100000
    )
    preprocessing_pipeline.setup_logging()
    preprocessing_pipeline.run(
        source="data/bronze_layer/SupplyChain_Dataset.parquet", 
        dest="data/silver_layer/preprocessed_SupplyChain_Dataset.parquet"
    )
    feature_pipeline.run_inventory_optimization_feature_pipeline(
        source="data/silver_layer/preprocessed_SupplyChain_Dataset.parquet", 
        selected_columns=inventory_features_columns, 
        dest="data/gold_layer/SupplyChain_Invetory_Dataset.parquet"
    )
   




    training_pipeline.run_inventory_training_pipeline(
        data_source="data/gold_layer/SupplyChain_Invetory_Dataset.parquet",
        target_column='Historical_Demand',
    )
    training_pipeline.run_stockout_risk_training_pipeline(
    data_source="data/gold_layer/SupplyChain_Invetory_Dataset.parquet",
    target_column='Stockout_Risk'
    )
    training_pipeline.run_overstock_risk_training_pipeline(
    data_source="data/gold_layer/SupplyChain_Invetory_Dataset.parquet",
    target_column='Overstock_Risk'
    )

