{#- database='pg' only makes sense when THIS model is being compiled/run
   under the duckdb target (it's the alias DuckDB attaches Postgres under).
   obt_customer_orders (diamond, postgres target) refs() this model too, so
   the conditional is required -- same pattern as dim_customers_history /
   dim_products_history. #}
{{
  config(
    materialized='table',
    database=('pg' if target.type == 'duckdb' else none),
    schema='gold'
  )
}}

-- ============================================================================
-- GOLD: Customer order-derived metrics -- CURRENT STATE ONLY, not SCD2
--
-- Purpose: total_orders/lifetime_value/segment/status change on every new
--          order, which is not what SCD Type 2 is meant to model (it's for
--          slowly-changing descriptive attributes, not continuously
--          recomputed aggregates). This table always reflects "as of the
--          last refresh"; for point-in-time customer IDENTITY attributes
--          (name/email/phone/country) as they were at any given moment, see
--          dim_customers.sql (SCD2) instead.
-- Source: Iceberg Silver (slv_customers, slv_orders), read via dbt-duckdb's
--         native "iceberg" plugin.
-- Materialized in: PostgreSQL (attached via DuckDB)
-- ============================================================================

SELECT
    c.customer_id,
    COUNT(DISTINCT o.order_id) as total_orders,
    COALESCE(SUM(o.total_amount), 0)::NUMERIC(18,2) as customer_lifetime_value,
    COALESCE(AVG(o.total_amount), 0)::NUMERIC(18,2) as avg_order_value,
    COALESCE(MIN(to_timestamp(o.order_date / 1000000.0)), to_timestamp(c.created_at / 1000000.0))::DATE as first_order_date,
    COALESCE(MAX(to_timestamp(o.order_date / 1000000.0)), to_timestamp(c.created_at / 1000000.0))::DATE as last_order_date,
    CASE
        WHEN MAX(o.order_date) IS NULL THEN 'Never Ordered'
        WHEN to_timestamp(MAX(o.order_date) / 1000000.0) < now() - INTERVAL '90 days' THEN 'Inactive'
        WHEN to_timestamp(MAX(o.order_date) / 1000000.0) < now() - INTERVAL '30 days' THEN 'At Risk'
        ELSE 'Active'
    END as customer_status,
    CASE
        WHEN COUNT(DISTINCT o.order_id) = 0 THEN 'New'
        WHEN to_timestamp(MAX(o.order_date) / 1000000.0) < now() - INTERVAL '90 days' THEN 'Churned'
        WHEN COALESCE(SUM(o.total_amount), 0) >= 2000 THEN 'VIP'
        WHEN COALESCE(SUM(o.total_amount), 0) >= 500 THEN 'Premium'
        ELSE 'Standard'
    END as customer_segment,
    DATE_DIFF('day', COALESCE(to_timestamp(MAX(o.order_date) / 1000000.0), to_timestamp(c.created_at / 1000000.0)), now()) as days_since_last_order,
    current_timestamp as dbt_loaded_at,
    now() as dbt_updated_at

FROM {{ source('iceberg_silver', 'slv_customers') }} c
LEFT JOIN {{ source('iceberg_silver', 'slv_orders') }} o
    ON c.customer_id = o.customer_id
    AND o.order_date IS NOT NULL
GROUP BY
    c.customer_id,
    c.created_at
