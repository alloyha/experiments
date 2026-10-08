{#- database='pg' only makes sense when THIS model is being compiled/run
   under the duckdb target (it's the alias DuckDB attaches Postgres under).
   obt_product_sales (diamond, postgres target) refs() this model too, so
   the conditional is required -- same pattern as customer_metrics_current. #}
{{
  config(
    materialized='table',
    database=('pg' if target.type == 'duckdb' else none),
    schema='gold'
  )
}}

-- ============================================================================
-- GOLD: Product order-derived metrics -- CURRENT STATE ONLY, not SCD2
--
-- Purpose: units_sold/revenue change on every order_item, which isn't what
-- SCD Type 2 is meant to model (it's for slowly-changing descriptive
-- attributes, not continuously recomputed aggregates). This table always
-- reflects "as of the last refresh"; for point-in-time product IDENTITY
-- attributes (name/category/price) as they were at any given moment, see
-- dim_products.sql (SCD2) instead.
-- Source: Iceberg Silver (slv_products, slv_order_items, slv_orders),
--         read via dbt-duckdb's native "iceberg" plugin.
-- Materialized in: PostgreSQL (attached via DuckDB)
-- ============================================================================

SELECT
    p.product_id,
    COUNT(DISTINCT oi.order_id) as orders_containing_product,
    COALESCE(SUM(oi.quantity), 0) as units_sold,
    COALESCE(SUM(oi.quantity * oi.unit_price), 0)::NUMERIC(18,2) as total_revenue,
    COALESCE(AVG(oi.unit_price), 0)::NUMERIC(18,2) as avg_sold_price,
    MIN(to_timestamp(o.order_date / 1000000.0))::DATE as first_sold_at,
    MAX(to_timestamp(o.order_date / 1000000.0))::DATE as last_sold_at,
    current_timestamp as dbt_loaded_at,
    now() as dbt_updated_at

FROM {{ source('iceberg_silver', 'slv_products') }} p
LEFT JOIN {{ source('iceberg_silver', 'slv_order_items') }} oi
    ON p.product_id = oi.product_id
LEFT JOIN {{ source('iceberg_silver', 'slv_orders') }} o
    ON oi.order_id = o.order_id
GROUP BY
    p.product_id
