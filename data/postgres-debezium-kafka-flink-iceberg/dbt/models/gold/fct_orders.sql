{#- see dim_customers.sql for why database is conditional on target.type #}
{{
  config(
    materialized='table',
    database=('pg' if target.type == 'duckdb' else none),
    schema='gold'
  )
}}

-- ============================================================================
-- GOLD: Customer Orders Fact Table
--
-- Purpose: Business-ready fact table for order analytics
-- Source: Iceberg Silver (slv_orders, slv_customers), read via dbt-duckdb's
--         native "iceberg" plugin (pyiceberg SqlCatalog against the same
--         JDBC catalog Flink writes to) -- Iceberg is the reliable source
--         of truth here, not the Postgres OLTP tables.
-- Materialized in: PostgreSQL (attached via DuckDB, optimized for BI tools)
-- Granularity: One row per order with customer dimensions
-- Freshness: Updated on each dbt run
-- ============================================================================

SELECT
    o.order_id,
    c.customer_id,
    c.name as customer_name,
    c.email as customer_email,
    c.country,
    c.phone,
    o.total_amount,
    o.status as order_status,
    to_timestamp(o.order_date / 1000000.0) as order_date,
    EXTRACT(YEAR FROM to_timestamp(o.order_date / 1000000.0))::INTEGER as order_year,
    EXTRACT(MONTH FROM to_timestamp(o.order_date / 1000000.0))::INTEGER as order_month,
    EXTRACT(QUARTER FROM to_timestamp(o.order_date / 1000000.0))::INTEGER as order_quarter,
    strftime(to_timestamp(o.order_date / 1000000.0), '%Y-%m') as order_year_month,
    current_timestamp as dbt_loaded_at,
    now() as dbt_updated_at

FROM {{ source('iceberg_silver', 'slv_orders') }} o
INNER JOIN {{ source('iceberg_silver', 'slv_customers') }} c
    ON o.customer_id = c.customer_id
WHERE o.order_date IS NOT NULL
  AND o.total_amount IS NOT NULL
