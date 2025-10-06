"""
Snowflake Storage Backend Implementation
=======================================

Implementation of Snowflake storage backend for cloud data warehouse.
"""

from typing import Dict, List, Any, Optional
from .abstract_backend import StorageBackend, StorageType


class SnowflakeStorageBackend(StorageBackend):
    """Snowflake cloud data warehouse"""
    
    def __init__(self, warehouse_name: str = "ANALYTICS_WH", 
                 database_name: str = "ANALYTICS_DB", 
                 schema_name: str = "PUBLIC"):
        self.warehouse_name = warehouse_name
        self.database_name = database_name
        self.schema_name = schema_name
        self.storage_type = StorageType.SNOWFLAKE
    
    def generate_ddl(self, table_metadata: Dict[str, Any]) -> str:
        """Generate Snowflake DDL with advanced features"""
        
        table_name = table_metadata['name']
        
        return f"""
-- Snowflake DDL for {table_name}
-- Cloud data warehouse with advanced analytics and governance

-- Set context
USE WAREHOUSE {self.warehouse_name};
USE DATABASE {self.database_name};
USE SCHEMA {self.schema_name};

-- Create main fact table with clustering and time travel
CREATE OR REPLACE TABLE {table_name} (
    transaction_id VARCHAR(50) NOT NULL,
    customer_id VARCHAR(50) NOT NULL,
    transaction_date TIMESTAMP_NTZ NOT NULL,
    amount NUMBER(10,2) NOT NULL,
    product_category VARCHAR(100) NOT NULL,
    customer_tier VARCHAR(20) NOT NULL,
    
    -- Derived columns for performance
    transaction_year NUMBER(4,0) AS (YEAR(transaction_date)),
    transaction_month NUMBER(2,0) AS (MONTH(transaction_date)),
    transaction_day_of_week NUMBER(1,0) AS (DAYOFWEEK(transaction_date)),
    transaction_hour NUMBER(2,0) AS (HOUR(transaction_date)),
    amount_bucket VARCHAR(10) AS (
        CASE 
            WHEN amount >= 1000 THEN 'High'
            WHEN amount >= 100 THEN 'Medium'
            ELSE 'Low'
        END
    ),
    
    -- Metadata columns
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    source_system VARCHAR(50) DEFAULT 'unified_platform',
    
    -- Primary key constraint
    CONSTRAINT pk_{table_name} PRIMARY KEY (transaction_id)
)
CLUSTER BY (customer_tier, product_category, DATE(transaction_date))
DATA_RETENTION_TIME_IN_DAYS = 90  -- Time Travel capability
COMMENT = 'Transaction fact table for customer analytics with time travel';

-- Create customer dimension with SCD Type 2
CREATE OR REPLACE TABLE dim_customer (
    customer_sk NUMBER AUTOINCREMENT,  -- Surrogate key
    customer_id VARCHAR(50) NOT NULL,
    customer_tier VARCHAR(20) NOT NULL,
    first_transaction_date TIMESTAMP_NTZ,
    total_lifetime_value NUMBER(12,2) DEFAULT 0,
    total_transactions NUMBER(10,0) DEFAULT 0,
    preferred_category VARCHAR(100),
    risk_score FLOAT DEFAULT 0.5,
    
    -- SCD Type 2 columns
    effective_date TIMESTAMP_NTZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    end_date TIMESTAMP_NTZ DEFAULT '9999-12-31'::TIMESTAMP_NTZ,
    is_current BOOLEAN DEFAULT TRUE,
    record_version NUMBER(5,0) DEFAULT 1,
    
    -- Change tracking
    change_reason VARCHAR(100),
    changed_by VARCHAR(100) DEFAULT CURRENT_USER(),
    
    -- Metadata
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    CONSTRAINT pk_dim_customer PRIMARY KEY (customer_sk)
)
CLUSTER BY (customer_id, is_current)
DATA_RETENTION_TIME_IN_DAYS = 365  -- Longer retention for dimension
COMMENT = 'Customer dimension with SCD Type 2 support and change tracking';

-- Create product dimension
CREATE OR REPLACE TABLE dim_product (
    product_category VARCHAR(100) NOT NULL,
    category_description VARCHAR(500),
    avg_price_range VARCHAR(50),
    seasonality_pattern VARCHAR(100),
    margin_percentage FLOAT,
    category_manager VARCHAR(100),
    
    -- Metadata
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    CONSTRAINT pk_dim_product PRIMARY KEY (product_category)
)
COMMENT = 'Product category dimension with business metadata';

-- Create secure view for analytics with row-level security
CREATE OR REPLACE SECURE VIEW customer_analytics_secure AS
SELECT 
    dc.customer_id,
    dc.customer_tier,
    COUNT(t.transaction_id) as total_transactions,
    SUM(t.amount) as total_spent,
    AVG(t.amount) as avg_transaction_value,
    STDDEV(t.amount) as transaction_value_stddev,
    MIN(t.transaction_date) as first_transaction,
    MAX(t.transaction_date) as last_transaction,
    DATEDIFF('day', MAX(t.transaction_date), CURRENT_TIMESTAMP()) as days_since_last_transaction,
    DATEDIFF('day', MIN(t.transaction_date), MAX(t.transaction_date)) as customer_lifetime_days,
    
    -- Product diversity
    COUNT(DISTINCT t.product_category) as categories_purchased,
    MODE(t.product_category) as favorite_category,
    
    -- Temporal patterns
    MODE(HOUR(t.transaction_date)) as preferred_hour,
    MODE(DAYOFWEEK(t.transaction_date)) as preferred_day_of_week,
    
    -- Risk scoring with CASE
    CASE 
        WHEN DATEDIFF('day', MAX(t.transaction_date), CURRENT_TIMESTAMP()) <= 30 THEN 'Low Risk'
        WHEN DATEDIFF('day', MAX(t.transaction_date), CURRENT_TIMESTAMP()) <= 90 THEN 'Medium Risk'
        WHEN DATEDIFF('day', MAX(t.transaction_date), CURRENT_TIMESTAMP()) <= 180 THEN 'High Risk'
        ELSE 'Critical Risk'
    END as churn_risk,
    
    -- RFM scores using window functions
    NTILE(5) OVER (ORDER BY DATEDIFF('day', MAX(t.transaction_date), CURRENT_TIMESTAMP()) DESC) as recency_score,
    NTILE(5) OVER (ORDER BY COUNT(t.transaction_id)) as frequency_score,
    NTILE(5) OVER (ORDER BY SUM(t.amount)) as monetary_score
    
FROM {table_name} t
JOIN dim_customer dc ON t.customer_id = dc.customer_id AND dc.is_current = TRUE
WHERE t.transaction_date >= CURRENT_TIMESTAMP() - INTERVAL '2 years'
GROUP BY dc.customer_id, dc.customer_tier
HAVING COUNT(t.transaction_id) >= 2;

-- Create stream for real-time change tracking
CREATE OR REPLACE STREAM {table_name}_stream 
ON TABLE {table_name}
APPEND_ONLY = FALSE
COMMENT = 'Stream to capture all changes to transaction table';

-- Create task for automated data processing
CREATE OR REPLACE TASK update_customer_metrics
WAREHOUSE = {self.warehouse_name}
SCHEDULE = 'USING CRON 0 */4 * * * UTC'  -- Every 4 hours
WHEN SYSTEM$STREAM_HAS_DATA('{table_name}_stream')
AS
MERGE INTO dim_customer dc
USING (
    SELECT 
        customer_id,
        customer_tier,
        MIN(transaction_date) as first_transaction_date,
        SUM(amount) as total_lifetime_value,
        COUNT(*) as total_transactions,
        MODE(product_category) as preferred_category
    FROM {table_name}
    GROUP BY customer_id, customer_tier
) src
ON dc.customer_id = src.customer_id AND dc.is_current = TRUE
WHEN MATCHED AND (
    dc.customer_tier != src.customer_tier OR 
    dc.total_lifetime_value != src.total_lifetime_value
) THEN
    UPDATE SET 
        end_date = CURRENT_TIMESTAMP(),
        is_current = FALSE,
        change_reason = 'Automated update from transaction data',
        updated_at = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN
    INSERT (
        customer_id, customer_tier, first_transaction_date,
        total_lifetime_value, total_transactions, preferred_category
    )
    VALUES (
        src.customer_id, src.customer_tier, src.first_transaction_date,
        src.total_lifetime_value, src.total_transactions, src.preferred_category
    );

-- Create stored procedure for SCD Type 2 updates
CREATE OR REPLACE PROCEDURE update_customer_scd2(
    p_customer_id VARCHAR,
    p_new_tier VARCHAR,
    p_change_reason VARCHAR
)
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    -- End current record
    UPDATE dim_customer 
    SET 
        end_date = CURRENT_TIMESTAMP(),
        is_current = FALSE,
        updated_at = CURRENT_TIMESTAMP()
    WHERE customer_id = p_customer_id 
      AND is_current = TRUE;
    
    -- Insert new record
    INSERT INTO dim_customer (
        customer_id, customer_tier, change_reason,
        effective_date, is_current, record_version
    )
    SELECT 
        p_customer_id,
        p_new_tier,
        p_change_reason,
        CURRENT_TIMESTAMP(),
        TRUE,
        COALESCE(MAX(record_version), 0) + 1
    FROM dim_customer 
    WHERE customer_id = p_customer_id;
    
    RETURN 'Customer ' || p_customer_id || ' updated successfully';
END;
$$;

-- Create masking policy for PII data
CREATE OR REPLACE MASKING POLICY customer_id_mask AS (val VARCHAR) RETURNS VARCHAR ->
CASE 
    WHEN CURRENT_ROLE() IN ('ANALYST_ROLE', 'ADMIN_ROLE') THEN val
    ELSE '***MASKED***'
END;

-- Apply masking policy
ALTER TABLE {table_name} MODIFY COLUMN customer_id 
SET MASKING POLICY customer_id_mask;

-- Create row access policy for customer tier restrictions
CREATE OR REPLACE ROW ACCESS POLICY customer_tier_policy AS (customer_tier VARCHAR) RETURNS BOOLEAN ->
    CASE 
        WHEN CURRENT_ROLE() = 'ADMIN_ROLE' THEN TRUE
        WHEN CURRENT_ROLE() = 'GOLD_ANALYST_ROLE' AND customer_tier IN ('Gold', 'Platinum') THEN TRUE
        WHEN CURRENT_ROLE() = 'GENERAL_ANALYST_ROLE' AND customer_tier IN ('Bronze', 'Silver') THEN TRUE
        ELSE FALSE
    END;

-- Apply row access policy
ALTER TABLE {table_name} ADD ROW ACCESS POLICY customer_tier_policy ON (customer_tier);

-- Enable change tracking
ALTER TABLE {table_name} SET CHANGE_TRACKING = TRUE;
ALTER TABLE dim_customer SET CHANGE_TRACKING = TRUE;

-- Create tags for data governance
CREATE OR REPLACE TAG pii_tag;
CREATE OR REPLACE TAG financial_data_tag;
CREATE OR REPLACE TAG retention_period_tag;

-- Apply tags
ALTER TABLE {table_name} SET TAG (
    financial_data_tag = 'high_sensitivity',
    retention_period_tag = '7_years'
);

ALTER TABLE {table_name} MODIFY COLUMN customer_id SET TAG (pii_tag = 'customer_identifier');
ALTER TABLE {table_name} MODIFY COLUMN amount SET TAG (financial_data_tag = 'transaction_amount');
"""
    
    def generate_dml(self, table_metadata: Dict[str, Any], 
                    operation: str = 'insert') -> str:
        """Generate Snowflake DML operations"""
        
        table_name = table_metadata['name']
        
        if operation == 'bulk_insert':
            return f"""
-- Snowflake Bulk Insert for {table_name}
-- Optimized for cloud data warehouse loading

-- Use multi-table insert for efficiency
INSERT ALL
    INTO {table_name} (transaction_id, customer_id, transaction_date, amount, product_category, customer_tier)
    VALUES ('txn_001', 'cust_001', '2024-01-15 10:30:00'::TIMESTAMP_NTZ, 99.99, 'Electronics', 'Gold')
    INTO {table_name} (transaction_id, customer_id, transaction_date, amount, product_category, customer_tier)
    VALUES ('txn_002', 'cust_002', '2024-01-15 11:15:00'::TIMESTAMP_NTZ, 149.50, 'Clothing', 'Silver')
    INTO {table_name} (transaction_id, customer_id, transaction_date, amount, product_category, customer_tier)
    VALUES ('txn_003', 'cust_003', '2024-01-15 12:00:00'::TIMESTAMP_NTZ, 299.99, 'Home', 'Platinum')
SELECT 1 FROM DUAL;

-- Bulk load from stage using COPY INTO
COPY INTO {table_name}
FROM @my_stage/{table_name}/
FILE_FORMAT = (
    TYPE = 'CSV',
    FIELD_DELIMITER = ',',
    RECORD_DELIMITER = '\\n',
    SKIP_HEADER = 1,
    DATE_FORMAT = 'YYYY-MM-DD',
    TIMESTAMP_FORMAT = 'YYYY-MM-DD HH24:MI:SS',
    FIELD_OPTIONALLY_ENCLOSED_BY = '"',
    NULL_IF = ('NULL', 'null', ''),
    ERROR_ON_COLUMN_COUNT_MISMATCH = TRUE
)
ON_ERROR = 'SKIP_FILE_3'  -- Skip file if more than 3 errors
PURGE = TRUE  -- Remove files after successful load
RETURN_FAILED_ONLY = TRUE;

-- MERGE operation for upserts with change tracking
MERGE INTO {table_name} target
USING (
    SELECT 
        transaction_id,
        customer_id,
        transaction_date,
        amount,
        product_category,
        customer_tier
    FROM VALUES
        ('txn_001', 'cust_001', '2024-01-15 10:30:00'::TIMESTAMP_NTZ, 109.99, 'Electronics', 'Gold'),
        ('txn_004', 'cust_004', '2024-01-15 13:30:00'::TIMESTAMP_NTZ, 79.99, 'Sports', 'Bronze')
    AS source(transaction_id, customer_id, transaction_date, amount, product_category, customer_tier)
) source
ON target.transaction_id = source.transaction_id
WHEN MATCHED AND (
    target.amount != source.amount OR 
    target.product_category != source.product_category OR
    target.customer_tier != source.customer_tier
) THEN
    UPDATE SET 
        amount = source.amount,
        product_category = source.product_category,
        customer_tier = source.customer_tier,
        updated_at = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN
    INSERT (
        transaction_id, customer_id, transaction_date, 
        amount, product_category, customer_tier,
        created_at, updated_at
    )
    VALUES (
        source.transaction_id, source.customer_id, source.transaction_date,
        source.amount, source.product_category, source.customer_tier,
        CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()
    );
"""
        
        elif operation == 'analytical_queries':
            return f"""
-- Snowflake Analytical Queries for {table_name}
-- Advanced analytics with Snowflake-specific functions

-- Customer cohort analysis with QUALIFY
SELECT 
    cohort_month,
    transaction_month,
    COUNT(DISTINCT customer_id) as active_customers,
    SUM(amount) as cohort_revenue,
    AVG(amount) as avg_revenue_per_customer,
    ROUND(COUNT(DISTINCT customer_id) * 100.0 / first_value(COUNT(DISTINCT customer_id)) 
          OVER (PARTITION BY cohort_month ORDER BY transaction_month), 2) as retention_rate,
    DATEDIFF('month', cohort_month, transaction_month) as period_number
FROM (
    SELECT 
        customer_id,
        DATE_TRUNC('month', MIN(transaction_date) OVER (PARTITION BY customer_id)) as cohort_month,
        DATE_TRUNC('month', transaction_date) as transaction_month,
        amount
    FROM {table_name}
    WHERE transaction_date >= CURRENT_DATE() - INTERVAL '12 months'
)
GROUP BY cohort_month, transaction_month
QUALIFY cohort_month >= CURRENT_DATE() - INTERVAL '12 months'
ORDER BY cohort_month, transaction_month;

-- Advanced time series analysis with SNOWFLAKE.ML functions
WITH daily_revenue AS (
    SELECT 
        DATE(transaction_date) as date,
        SUM(amount) as daily_revenue,
        COUNT(*) as daily_transactions,
        COUNT(DISTINCT customer_id) as daily_customers
    FROM {table_name}
    WHERE transaction_date >= CURRENT_DATE() - INTERVAL '2 years'
    GROUP BY DATE(transaction_date)
),
time_series_features AS (
    SELECT 
        date,
        daily_revenue,
        daily_transactions,
        daily_customers,
        
        -- Moving averages with different windows
        AVG(daily_revenue) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as ma_7,
        AVG(daily_revenue) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as ma_30,
        AVG(daily_revenue) OVER (ORDER BY date ROWS BETWEEN 89 PRECEDING AND CURRENT ROW) as ma_90,
        
        -- Exponential moving average approximation
        AVG(daily_revenue) OVER (ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) as ema_14,
        
        -- Seasonal decomposition
        AVG(daily_revenue) OVER (PARTITION BY DAYOFWEEK(date) ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as dow_seasonal,
        AVG(daily_revenue) OVER (PARTITION BY DAY(date) ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as dom_seasonal,
        
        -- Lag features
        LAG(daily_revenue, 1) OVER (ORDER BY date) as lag_1_day,
        LAG(daily_revenue, 7) OVER (ORDER BY date) as lag_1_week,
        LAG(daily_revenue, 30) OVER (ORDER BY date) as lag_1_month,
        
        -- Volatility measures
        STDDEV(daily_revenue) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as volatility_30,
        
        -- Trend calculation using linear regression over window
        REGR_SLOPE(daily_revenue, ROW_NUMBER() OVER (ORDER BY date)) 
            OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as trend_slope_30
        
    FROM daily_revenue
)
SELECT 
    date,
    daily_revenue,
    ma_7,
    ma_30,
    
    -- Trend indicators
    CASE 
        WHEN daily_revenue > ma_30 * 1.1 THEN 'Strong Uptrend'
        WHEN daily_revenue > ma_30 * 1.05 THEN 'Uptrend'
        WHEN daily_revenue < ma_30 * 0.9 THEN 'Strong Downtrend'
        WHEN daily_revenue < ma_30 * 0.95 THEN 'Downtrend'
        ELSE 'Stable'
    END as trend_signal,
    
    -- Seasonality strength
    ABS(daily_revenue - dow_seasonal) / NULLIF(dow_seasonal, 0) as dow_seasonality_strength,
    
    -- Anomaly detection using z-score
    ABS(daily_revenue - ma_30) / NULLIF(volatility_30, 0) as z_score,
    CASE 
        WHEN ABS(daily_revenue - ma_30) / NULLIF(volatility_30, 0) > 3 THEN 'Anomaly'
        WHEN ABS(daily_revenue - ma_30) / NULLIF(volatility_30, 0) > 2 THEN 'Outlier'
        ELSE 'Normal'
    END as anomaly_flag,
    
    -- Growth rates
    ROUND((daily_revenue - lag_1_day) * 100.0 / NULLIF(lag_1_day, 0), 2) as day_over_day_pct,
    ROUND((daily_revenue - lag_1_week) * 100.0 / NULLIF(lag_1_week, 0), 2) as week_over_week_pct,
    ROUND((daily_revenue - lag_1_month) * 100.0 / NULLIF(lag_1_month, 0), 2) as month_over_month_pct

FROM time_series_features
WHERE date >= CURRENT_DATE() - INTERVAL '90 days'
ORDER BY date DESC;

-- Customer segmentation using clustering-like analysis
WITH customer_features AS (
    SELECT 
        customer_id,
        customer_tier,
        COUNT(*) as frequency,
        SUM(amount) as monetary,
        DATEDIFF('day', MAX(transaction_date), CURRENT_DATE()) as recency,
        AVG(amount) as avg_order_value,
        STDDEV(amount) as order_value_std,
        COUNT(DISTINCT product_category) as category_diversity,
        DATEDIFF('day', MIN(transaction_date), MAX(transaction_date)) as customer_lifetime,
        
        -- Normalize features using z-score
        (COUNT(*) - AVG(COUNT(*)) OVER ()) / NULLIF(STDDEV(COUNT(*)) OVER (), 0) as frequency_z,
        (SUM(amount) - AVG(SUM(amount)) OVER ()) / NULLIF(STDDEV(SUM(amount)) OVER (), 0) as monetary_z,
        (DATEDIFF('day', MAX(transaction_date), CURRENT_DATE()) - 
         AVG(DATEDIFF('day', MAX(transaction_date), CURRENT_DATE())) OVER ()) / 
         NULLIF(STDDEV(DATEDIFF('day', MAX(transaction_date), CURRENT_DATE())) OVER (), 0) as recency_z
         
    FROM {table_name}
    WHERE transaction_date >= CURRENT_DATE() - INTERVAL '2 years'
    GROUP BY customer_id, customer_tier
    HAVING COUNT(*) >= 2
)
SELECT 
    customer_id,
    customer_tier,
    frequency,
    monetary,
    recency,
    
    -- Advanced segmentation using multiple criteria
    CASE 
        WHEN frequency_z > 1 AND monetary_z > 1 AND recency_z < -0.5 THEN 'Champions'
        WHEN frequency_z > 0.5 AND monetary_z > 0.5 AND recency_z < 0 THEN 'Loyal Customers'
        WHEN frequency_z < -0.5 AND monetary_z < -0.5 AND recency_z < -1 THEN 'New Customers'
        WHEN frequency_z > 0 AND monetary_z > 0 AND recency_z > 0.5 THEN 'At Risk'
        WHEN frequency_z < -0.5 AND monetary_z > 1 AND recency_z > 1 THEN 'Cannot Lose Them'
        WHEN frequency_z < 0 AND monetary_z < 0 AND recency_z > 1 THEN 'Lost Customers'
        ELSE 'Others'
    END as advanced_segment,
    
    -- Customer value score
    ROUND((frequency_z + monetary_z - recency_z) / 3, 2) as customer_value_score,
    
    -- Recommended actions
    CASE 
        WHEN frequency_z > 1 AND monetary_z > 1 AND recency_z < -0.5 THEN 'Reward and retain'
        WHEN frequency_z > 0.5 AND monetary_z > 0.5 AND recency_z < 0 THEN 'Upsell opportunities'
        WHEN frequency_z < -0.5 AND monetary_z < -0.5 AND recency_z < -1 THEN 'Onboarding program'
        WHEN frequency_z > 0 AND monetary_z > 0 AND recency_z > 0.5 THEN 'Win-back campaign'
        WHEN frequency_z < -0.5 AND monetary_z > 1 AND recency_z > 1 THEN 'Urgent retention'
        ELSE 'Monitor'
    END as recommended_action

FROM customer_features
ORDER BY customer_value_score DESC;
"""
        
        return f"-- Snowflake {operation} for {table_name}"
    
    def supports_feature(self, feature: str) -> bool:
        """Check if Snowflake supports a feature"""
        supported_features = {
            'time_travel', 'fail_safe', 'zero_copy_cloning', 'clustering',
            'automatic_scaling', 'multi_cluster_warehouses', 'secure_views',
            'row_access_policies', 'column_level_security', 'masking_policies',
            'streams', 'tasks', 'stored_procedures', 'user_defined_functions',
            'external_tables', 'materialized_views', 'change_tracking',
            'data_sharing', 'marketplace', 'snowpark', 'ml_functions'
        }
        return feature in supported_features
    
    def estimate_storage_cost(self, table_metadata: Dict[str, Any], 
                             rows_per_day: int = 10000) -> Dict[str, Any]:
        """Estimate Snowflake storage and compute costs"""
        
        # Rough estimates based on Snowflake pricing
        bytes_per_row = 200  # Average row size
        monthly_storage_gb = (rows_per_day * 30 * bytes_per_row) / (1024**3)
        
        # Snowflake pricing (approximate, varies by region)
        storage_cost_per_gb_month = 0.025
        compute_cost_per_credit = 2.00
        estimated_credits_per_month = 100  # Depends on warehouse size and usage
        
        monthly_storage_cost = monthly_storage_gb * storage_cost_per_gb_month
        monthly_compute_cost = estimated_credits_per_month * compute_cost_per_credit
        
        return {
            'storage_type': self.storage_type.value,
            'estimated_monthly_storage_gb': round(monthly_storage_gb, 2),
            'estimated_monthly_storage_cost_usd': round(monthly_storage_cost, 2),
            'estimated_monthly_compute_cost_usd': round(monthly_compute_cost, 2),
            'total_estimated_monthly_cost_usd': round(monthly_storage_cost + monthly_compute_cost, 2),
            'cost_optimization_features': [
                'automatic_suspend_resume', 'multi_cluster_scaling',
                'result_caching', 'clustering_keys', 'materialized_views'
            ],
            'pricing_model': 'separate_storage_and_compute',
            'scalability': 'elastic_with_auto_scaling'
        }
    
    def get_performance_optimizations(self) -> List[str]:
        """Get Snowflake-specific performance optimizations"""
        return [
            'use_clustering_keys_for_large_tables',
            'implement_proper_warehouse_sizing',
            'leverage_result_caching',
            'use_materialized_views_for_aggregations',
            'optimize_data_types_for_storage',
            'implement_time_travel_retention_policies',
            'use_secure_views_for_row_level_security',
            'leverage_zero_copy_cloning_for_dev_test',
            'implement_streams_and_tasks_for_automation',
            'use_external_tables_for_data_lake_integration',
            'optimize_join_order_and_predicates',
            'implement_data_sharing_for_cross_account_access'
        ]