{{
  config(
    materialized='table',
    schema='diamond',
    indexes=[
      {'columns': ['month_date'], 'type': 'btree'},
      {'columns': ['customer_segment'], 'type': 'btree'},
    ],
    meta={
      'owner': 'analytics-team',
      'layer': 'diamond',
      'grain': 'one row per customer-segment-month',
      'refresh_frequency': '60min',
    }
  )
}}

-- 💎 DIAMOND LAYER - AGGREGATED METRICS
-- 
-- Purpose: Pre-aggregated daily sales metrics for extreme query performance
-- Grain: One row per customer_segment + day
-- Use Case: Dashboards, trend analysis, KPI tracking
-- Performance: Sub-second queries on millions of rows

SELECT
  -- Time Dimension (Aggregation Grain)
  DATE_TRUNC('day', o.order_date)::DATE as month_date,
  EXTRACT(YEAR FROM o.order_date) as year_num,
  EXTRACT(QUARTER FROM o.order_date) as quarter_num,
  EXTRACT(MONTH FROM o.order_date) as month_num,
  
  -- Business Dimension
  o.customer_segment,
  
  -- Pre-Calculated Metrics (OLAP Measures)
  COUNT(DISTINCT o.order_id) as total_orders,
  COUNT(DISTINCT o.customer_id) as unique_customers,
  SUM(o.order_amount) as total_sales,
  ROUND(AVG(o.order_amount), 2) as avg_order_value,
  MIN(o.order_amount) as min_order_amount,
  MAX(o.order_amount) as max_order_amount,
  
  -- Additional Metrics for Analysis
  ROUND(SUM(o.order_amount) / COUNT(DISTINCT o.customer_id), 2) as sales_per_customer,
  ROUND(100.0 * SUM(o.order_amount) / SUM(SUM(o.order_amount)) OVER (PARTITION BY DATE_TRUNC('day', o.order_date)::DATE), 2) as segment_pct_of_daily_sales

FROM {{ ref('obt_customer_orders') }} o
GROUP BY 1, 2, 3, 4, 5
