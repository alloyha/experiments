"""
Polars Processing Engine Implementation
=====================================

Implementation of Polars processing engine for high-performance analytics.
"""

from typing import Dict, List, Any
from .abstract_engine import ProcessingEngineBackend
from ..storage.abstract_backend import StorageType


class PolarsProcessingEngine(ProcessingEngineBackend):
    """Polars high-performance DataFrame processing"""
    
    def generate_query(self, table_metadata: Dict[str, Any], 
                      operation: str, **kwargs) -> str:
        """Generate Polars code"""
        
        if operation == 'fast_analytics':
            table_name = table_metadata['name']
            return f"""
# Polars High-Performance Analytics for {table_name}
import polars as pl
from datetime import datetime, timedelta

# Lazy loading for maximum memory efficiency
df = pl.scan_parquet("data/{table_name}.parquet")

# Lightning-fast analytical operations with lazy evaluation
result = (df
    .filter(pl.col("transaction_date") >= datetime.now() - timedelta(days=90))
    .with_columns([
        pl.col("transaction_date").dt.month().alias("month"),
        pl.col("transaction_date").dt.day_of_week().alias("day_of_week"),
        pl.col("amount").round(2).alias("amount_rounded")
    ])
    .group_by(["month", "day_of_week", "customer_tier"])
    .agg([
        pl.col("amount").sum().alias("total_revenue"),
        pl.col("amount").mean().alias("avg_order_value"),
        pl.col("amount").median().alias("median_order_value"),
        pl.col("amount").std().alias("order_value_std"),
        pl.col("customer_id").n_unique().alias("unique_customers"),
        pl.col("transaction_id").count().alias("transaction_count"),
        pl.col("amount").quantile(0.95).alias("p95_order_value")
    ])
    .with_columns([
        (pl.col("total_revenue") / pl.col("transaction_count")).alias("avg_per_transaction"),
        (pl.col("unique_customers") / pl.col("transaction_count") * 100).alias("customer_diversity_pct")
    ])
    .sort(["month", "total_revenue"], descending=[False, True])
    .collect()  # Execute all lazy operations
)

print(f"Analytics complete! Processed {{len(result)}} customer segments in milliseconds")
print("Top performing segments:")
print(result.head(10))

# Export results
result.write_parquet(f"results/{table_name}_analytics_{{datetime.now().strftime('%Y%m%d')}}.parquet")
result.write_csv(f"results/{table_name}_analytics_{{datetime.now().strftime('%Y%m%d')}}.csv")
"""
        
        elif operation == 'data_preprocessing':
            return f"""
# Polars Data Preprocessing for {table_metadata['name']}
import polars as pl
import numpy as np

# Read data with schema inference
df = pl.read_parquet("raw_data/{table_metadata['name']}.parquet")

# High-performance data cleaning and preprocessing
cleaned_df = (df
    # Remove duplicates
    .unique(subset=["transaction_id"])
    
    # Handle missing values
    .with_columns([
        pl.col("amount").fill_null(pl.col("amount").median()),
        pl.col("customer_tier").fill_null("Unknown"),
        pl.col("product_category").fill_null("Other")
    ])
    
    # Data validation and filtering
    .filter(
        (pl.col("amount") > 0) & 
        (pl.col("amount") < 10000) &  # Remove outliers
        (pl.col("transaction_date").is_not_null())
    )
    
    # Feature engineering
    .with_columns([
        # Time-based features
        pl.col("transaction_date").dt.year().alias("year"),
        pl.col("transaction_date").dt.month().alias("month"),
        pl.col("transaction_date").dt.day_of_week().alias("day_of_week"),
        pl.col("transaction_date").dt.hour().alias("hour"),
        
        # Amount-based features
        pl.col("amount").log().alias("log_amount"),
        pl.col("amount").rank().alias("amount_rank"),
        
        # Customer features (using window functions)
        pl.col("amount").sum().over("customer_id").alias("customer_total_spent"),
        pl.col("amount").count().over("customer_id").alias("customer_transaction_count"),
        pl.col("amount").mean().over("customer_id").alias("customer_avg_order_value"),
        
        # Product features
        pl.col("amount").mean().over("product_category").alias("category_avg_price"),
        pl.col("amount").std().over("product_category").alias("category_price_std")
    ])
    
    # Create derived features
    .with_columns([
        (pl.col("amount") / pl.col("customer_avg_order_value")).alias("amount_vs_customer_avg"),
        (pl.col("amount") / pl.col("category_avg_price")).alias("amount_vs_category_avg"),
        pl.when(pl.col("amount") > pl.col("customer_avg_order_value") * 2)
          .then(1)
          .otherwise(0)
          .alias("is_high_value_transaction")
    ])
    
    # Categorical encoding
    .with_columns([
        pl.col("customer_tier").map_dict({{
            "Bronze": 1, "Silver": 2, "Gold": 3, "Platinum": 4
        }}, default=0).alias("customer_tier_encoded"),
        
        pl.col("product_category").map_dict({{
            "Electronics": 1, "Clothing": 2, "Home": 3, "Sports": 4
        }}, default=0).alias("product_category_encoded")
    ])
)

# Split into train/validation sets
train_df = cleaned_df.filter(pl.col("transaction_date") < pl.date(2024, 10, 1))
validation_df = cleaned_df.filter(pl.col("transaction_date") >= pl.date(2024, 10, 1))

# Save preprocessed data
train_df.write_parquet("processed_data/{table_metadata['name']}_train.parquet")
validation_df.write_parquet("processed_data/{table_metadata['name']}_validation.parquet")

print(f"Preprocessing complete!")
print(f"Train set: {{len(train_df):,}} records")
print(f"Validation set: {{len(validation_df):,}} records") 
print(f"Feature count: {{len(cleaned_df.columns)}}")
"""
        
        elif operation == 'feature_engineering':
            return f"""
# Polars Feature Engineering for {table_metadata['name']}
import polars as pl
from datetime import datetime, timedelta

# Load transaction data
df = pl.scan_parquet("data/{table_metadata['name']}.parquet")

# Advanced feature engineering with window functions
features_df = (df
    .sort("customer_id", "transaction_date")
    .with_columns([
        # Recency features
        (pl.col("transaction_date").max().over("customer_id") - pl.col("transaction_date")).dt.total_days().alias("days_since_last_transaction"),
        
        # Frequency features  
        pl.col("transaction_id").count().over("customer_id").alias("total_transactions"),
        (pl.col("transaction_date").max().over("customer_id") - pl.col("transaction_date").min().over("customer_id")).dt.total_days().alias("customer_lifetime_days"),
        
        # Monetary features
        pl.col("amount").sum().over("customer_id").alias("total_spent"),
        pl.col("amount").mean().over("customer_id").alias("avg_order_value"),
        pl.col("amount").std().over("customer_id").alias("order_value_std"),
        pl.col("amount").min().over("customer_id").alias("min_order_value"),
        pl.col("amount").max().over("customer_id").alias("max_order_value"),
        
        # Trend features (using lag)
        pl.col("amount").shift(1).over("customer_id").alias("prev_order_amount"),
        pl.col("transaction_date").shift(1).over("customer_id").alias("prev_transaction_date"),
        
        # Rolling features
        pl.col("amount").rolling_mean(window_size=5).over("customer_id").alias("rolling_5_avg_amount"),
        pl.col("amount").rolling_std(window_size=5).over("customer_id").alias("rolling_5_std_amount"),
        
        # Seasonal features
        pl.col("transaction_date").dt.quarter().alias("quarter"),
        pl.col("transaction_date").dt.is_weekend().alias("is_weekend"),
        
        # Product diversity
        pl.col("product_category").n_unique().over("customer_id").alias("product_categories_purchased"),
    ])
    .with_columns([
        # Derived features
        (pl.col("total_spent") / pl.col("total_transactions")).alias("calculated_avg_order_value"),
        (pl.col("total_transactions") / pl.col("customer_lifetime_days")).alias("transaction_frequency"),
        (pl.col("amount") - pl.col("prev_order_amount")).alias("order_amount_change"),
        ((pl.col("transaction_date") - pl.col("prev_transaction_date")).dt.total_days()).alias("days_between_orders"),
        
        # RFM scores (Recency, Frequency, Monetary)
        pl.col("days_since_last_transaction").rank().alias("recency_rank"),
        pl.col("total_transactions").rank(descending=True).alias("frequency_rank"), 
        pl.col("total_spent").rank(descending=True).alias("monetary_rank"),
    ])
    .with_columns([
        # Customer segmentation
        pl.when(
            (pl.col("recency_rank") <= 1000) & 
            (pl.col("frequency_rank") <= 1000) & 
            (pl.col("monetary_rank") <= 1000)
        ).then("Champion")
        .when(
            (pl.col("recency_rank") <= 2000) & 
            (pl.col("frequency_rank") <= 1000) & 
            (pl.col("monetary_rank") <= 1000)
        ).then("Loyal Customer")
        .when(
            (pl.col("recency_rank") <= 3000) & 
            (pl.col("frequency_rank") <= 3000) & 
            (pl.col("monetary_rank") <= 3000)
        ).then("Potential Loyalist")
        .otherwise("At Risk")
        .alias("customer_segment")
    ])
    .collect()
)

# Feature selection and final dataset
final_features = features_df.select([
    "customer_id",
    "transaction_id", 
    "transaction_date",
    "amount",
    "total_transactions",
    "transaction_frequency",
    "avg_order_value",
    "order_value_std",
    "rolling_5_avg_amount",
    "days_since_last_transaction",
    "customer_segment",
    "quarter",
    "is_weekend",
    "product_categories_purchased"
])

# Save feature set
final_features.write_parquet(f"features/{table_metadata['name']}_features.parquet")

print(f"Feature engineering complete!")
print(f"Final feature set: {{len(final_features):,}} records with {{len(final_features.columns)}} features")
print("Customer segments distribution:")
print(final_features.group_by("customer_segment").agg(pl.count().alias("count")).sort("count", descending=True))
"""
        
        return f"# Polars {operation} for {table_metadata.get('name', 'unknown')}"
    
    def supports_storage_backend(self, storage_type: StorageType) -> bool:
        """Polars works well with file-based storage"""
        supported = {
            StorageType.DUCKDB,  # Direct integration
            StorageType.POSTGRESQL,  # Through connectors
            # Note: Parquet files would be primary, CSV, JSON, etc.
        }
        return storage_type in supported
    
    def estimate_query_performance(self, query: str, 
                                 table_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate Polars performance"""
        has_lazy_operations = 'scan_' in query
        has_aggregations = any(func in query for func in ['.agg(', '.group_by(', '.sum()', '.mean()'])
        has_window_functions = '.over(' in query
        has_joins = '.join(' in query
        
        if has_lazy_operations and has_aggregations:
            return {
                'complexity': 'medium',
                'estimated_execution_time': '100ms - 2 seconds',
                'memory_efficiency': 'excellent',
                'cpu_utilization': 'high',
                'optimal_for': 'analytical_aggregations',
                'lazy_evaluation': True
            }
        elif has_window_functions:
            return {
                'complexity': 'medium',
                'estimated_execution_time': '200ms - 5 seconds', 
                'memory_efficiency': 'excellent',
                'cpu_utilization': 'high',
                'optimal_for': 'time_series_analysis'
            }
        else:
            return {
                'complexity': 'low',
                'estimated_execution_time': '10-100ms',
                'memory_efficiency': 'excellent',
                'cpu_utilization': 'high',
                'optimal_for': 'data_preprocessing'
            }
    
    def get_optimal_query_patterns(self) -> List[str]:
        """Get optimal patterns for Polars"""
        return [
            'fast_aggregations',
            'data_preprocessing',
            'feature_engineering',
            'time_series_analysis',
            'data_quality_checks',
            'statistical_analysis',
            'machine_learning_prep',
            'window_functions',
            'high_performance_analytics'
        ]