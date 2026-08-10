{{
  config(
    materialized='table',
    schema='diamond',
    indexes=[
      {'columns': ['order_date'], 'type': 'btree'},
      {'columns': ['customer_id'], 'type': 'btree'},
      {'columns': ['customer_segment'], 'type': 'btree'},
    ],
    meta={
      'owner': 'analytics-team',
      'layer': 'diamond',
      'grain': 'one row per order',
      'refresh_frequency': 'real-time',
      'pii_columns': ['customer_name', 'customer_email'],
    }
  )
}}

-- 💎 DIAMOND LAYER - SEMANTIC OBT (One Big Table)
--
-- Purpose: Consolidated semantic layer for OLAP cube analysis
-- Grain: One row per order with all customer and product dimensions
-- Use Case: BI tools, dashboards, ad-hoc analytics without JOINs
--
-- Business Logic:
--   1. Consolidate fct_orders with dim_customers and dim_products
--   2. Flatten all dimensions into a single denormalized table
--   3. Pre-calculate key metrics for faster aggregations
--   4. Add business-friendly naming for BI tools
--
-- dim_customers is SCD2 (one row per customer per changed IDENTITY
-- attribute version, see models/gold/dim_customers.sql) -- the join below is
-- a point-in-time join on the order's own date, so each order gets
-- attributed to the customer's name/email/region AS THEY WERE when the
-- order happened, not whatever they are today. This is the correct way to
-- propagate SCD2 history into a fact table.
--
-- customer_segment/customer_lifetime_value come from customer_metrics_current
-- instead -- those are order-derived aggregates recomputed on every refresh,
-- not slowly-changing attributes, so they're tagged as of the CURRENT
-- metrics rather than historized per order (see that model's header comment).

SELECT
  -- Order Keys & IDs
  fo.order_id,
  fo.customer_id,

  -- Customer Dimension (Flattened from dim_customers.properties, as of order_date)
  dc.properties->>'name' as customer_name,
  dc.properties->>'email' as customer_email,
  cm.customer_segment,
  cm.customer_lifetime_value,
  dc.properties->>'customer_region' as customer_region,

  -- Order Facts & Dates
  fo.order_date,
  DATE_TRUNC('day', fo.order_date)::DATE as order_date_day,
  DATE_TRUNC('week', fo.order_date)::DATE as order_date_week,
  DATE_TRUNC('month', fo.order_date)::DATE as order_date_month,
  DATE_TRUNC('quarter', fo.order_date)::DATE as order_date_quarter,
  DATE_TRUNC('year', fo.order_date)::DATE as order_date_year,
  EXTRACT(YEAR FROM fo.order_date) as order_year,
  EXTRACT(QUARTER FROM fo.order_date) as order_quarter,
  EXTRACT(MONTH FROM fo.order_date) as order_month,
  EXTRACT(WEEK FROM fo.order_date) as order_week,
  EXTRACT(DOW FROM fo.order_date) as order_day_of_week,
  
  -- Order Amounts & Metrics
  fo.total_amount as order_amount,
  fo.total_amount as gross_sales,
  ROUND(fo.total_amount * 0.9, 2) as net_sales,  -- Example: 10% discount
  CASE 
    WHEN fo.total_amount < 100 THEN 'Small'
    WHEN fo.total_amount < 500 THEN 'Medium'
    WHEN fo.total_amount < 1000 THEN 'Large'
    ELSE 'Enterprise'
  END as order_size_category,
  
  -- Metrics
  1 as order_count,  -- For easy SUM aggregations
  
  -- Metadata
  fo.dbt_loaded_at as record_created_at,
  CURRENT_TIMESTAMP as record_updated_at

FROM {{ ref('fct_orders') }} fo
LEFT JOIN {{ ref('dim_customers') }} dc
  ON fo.customer_id = dc.id
  AND fo.order_date >= dc.valid_from
  AND (dc.valid_to IS NULL OR fo.order_date < dc.valid_to)
LEFT JOIN {{ ref('customer_metrics_current') }} cm
  ON fo.customer_id = cm.customer_id

WHERE 1=1
  -- Optionally: Filter to recent data for performance
  -- AND fo.order_date >= CURRENT_DATE - INTERVAL '24 months'
