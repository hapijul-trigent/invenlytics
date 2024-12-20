import pandas as pd
import numpy as np
import random
import logging
from datetime import timedelta

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/ingestion_pipeline.log"),
            logging.StreamHandler()
        ]
    )


def run(output_file="data/bronze_layer/SupplyChain_Dataset.parquet", total_rows=100000):
    """
    Generate a synthetic dataset for supply chain and inventory management.

    Parameters:
        output_file (str): The path to save the generated dataset as a CSV file.
        total_rows (int): The number of rows to generate in the dataset.

    Returns:
        None
    """
    try:
        # Initialize logging
        logging.info("Starting the data generation process.")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Parameters for dataset
        start_date = '2014-01-01'
        end_date = '2024-12-31'

        # Common lists for columns
        suppliers = ['Alibaba', 'H&M', 'IKEA', 'Wrogn']
        delivery_modes = ['Air', 'Sea', 'Road', 'Rail']
        disruption_types = ['Weather', 'Supplier Issue', 'Logistics', 'Geopolitical', 'None']
        regions = ['North America', 'Europe', 'Asia', 'South America']
        weather_conditions = ['Sunny', 'Rainy', 'Snowy', 'Cloudy']
        delivery_statuses = ['On Time', 'Delayed', 'Cancelled']
        categories = {
            'Electronics': ['Laptop', 'Smartphone', 'Headphones', 'Camera', 'Tablet'],
            'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Sweater', 'Dress'],
            'Furniture': ['Chair', 'Table', 'Sofa', 'Bed', 'Desk'],
            'Food': ['Coffee', 'Tea', 'Chips', 'Juice', 'Cookies'],
            'Books': ['The Great Adventure', 'Programming Guide', 'Modern Art History', 'Healthy Living']
        }
        product_ids = [f"P{i:05d}" for i in range(1, 2001)]  # 2000 unique products
        supplier_ids = [f"S{i:03d}" for i in range(1, 201)]  # 200 unique suppliers

        # Generate date range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')

        # Generate synthetic data
        synthetic_data = []
        for i in range(total_rows):
            scheduled_date = random.choice(date_range)
            actual_date = scheduled_date + pd.to_timedelta(np.random.randint(-5, 15), unit='D')
            category = random.choice(list(categories.keys()))
            product_name = random.choice(categories[category])
            current_stock = random.randint(0, 1000)
            damaged_stock = int(current_stock * random.uniform(0.01, 0.05))  # 1-5% of stock
            dead_stock = random.randint(0, int(current_stock * 0.3))  # Max 30% dead stock

            synthetic_entry = {
                "Shipment_ID": f"SHIP_{i+1}",
                "Supplier": random.choice(suppliers),
                "Region": random.choice(regions),
                "Delivery_Mode": random.choice(delivery_modes),
                "Scheduled_Delivery": scheduled_date,
                "Actual_Delivery": actual_date,
                "Freight_Cost": round(random.uniform(100, 1000), 2),
                "Disruption_Type": random.choices(disruption_types, weights=[0.2, 0.3, 0.2, 0.1, 0.2])[0],
                "Weather_Risk": round(random.uniform(0, 1), 2),
                "Supplier_Reliability": round(random.uniform(0.7, 1.0), 2),
                "Port_Congestion": round(random.uniform(0, 1), 2),
                "Stockout_Risk": round(random.uniform(0, 1), 2),
                "Recovery_Time_Days": random.randint(1, 10),
                "Delay_Duration": max((actual_date - scheduled_date).days, 0),
                "Product_ID": random.choice(product_ids),
                "Product_Name": product_name,
                "Category": category,
                "Current_Stock": current_stock,
                "Reorder_Level": random.randint(10, 200),
                "Lead_Time_Days": random.randint(1, 30),
                "Supplier_ID": random.choice(supplier_ids),
                "Supplier_Lead_Time": random.randint(1, 15),
                "Historical_Demand": random.randint(20, 300),
                "Forecasted_Demand": random.randint(30, 400),
                "Seasonality_Index": round(random.uniform(0.5, 2.0), 2),
                "Delivery_Status": random.choice(delivery_statuses),
                "Delay_Days": random.randint(0, 15),
                "Weather_Conditions": random.choice(weather_conditions),
                "Economic_Indicators": round(random.uniform(0.5, 5.0), 2),
                "Holiday_Flag": random.choice([0, 1]),
                "Order_Frequency": random.randint(1, 50),
                "Batch_Size": random.randint(10, 500),
                "Inventory_Turnover": round(random.uniform(1.0, 10.0), 2),
                "Safety_Stock_Level": random.randint(10, 300),
                "Dead_Stock": dead_stock,
                "Damaged_Stock": damaged_stock
            }
            synthetic_data.append(synthetic_entry)

        # Convert to DataFrame
        combined_synthetic_data = pd.DataFrame(synthetic_data)

        # Save the dataset to a CSV file
        combined_synthetic_data.to_parquet(output_file, index=False)
        logging.info(f"Combined synthetic dataset with {total_rows} rows has been saved as '{output_file}'.")
        logging.info(combined_synthetic_data.info())
    except Exception as e:
        logging.error("An error occurred during data generation:", exc_info=True)


if __name__ == "__main__":
    run()
