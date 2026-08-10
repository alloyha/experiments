{#- see customer_metrics_current.sql for why database is conditional on target.type #}
{{
  config(
    materialized='table',
    database=('pg' if target.type == 'duckdb' else none),
    schema='gold'
  )
}}

-- ============================================================================
-- GOLD (staging): every distinct product version that ever existed,
-- reconstructed from Bronze's full append-only CDC history.
--
-- Purpose: feeds dim_products.sql (SCD2). Reading straight from
-- brz_products_cdc -- NOT from slv_products or a "latest snapshot" -- is
-- what makes the SCD2 history complete regardless of how often this model
-- (or dim_products) gets rebuilt: even two price changes that both happened
-- between two refreshes still show up as two distinct rows here, because
-- we're detecting every value transition in the full commit log, not
-- polling "whatever's current right now".
--
-- Source: Iceberg Bronze (brz_products_cdc), read via dbt-duckdb's native
--         "iceberg" plugin (pyiceberg SqlCatalog against the same JDBC
--         catalog Flink writes to).
-- Materialized in: PostgreSQL (attached via DuckDB)
-- ============================================================================

-- see dim_customers_attributes_history.sql for why this dedup step exists:
-- Bronze can contain exact-duplicate CDC events tied on (id, event_timestamp)
-- after a Flink job resubmission replays already-seen Kafka messages, which
-- would otherwise make the LAG-based transition detection below inconsistent.
WITH deduped AS (
    SELECT DISTINCT ON (product_id, event_timestamp)
        product_id,
        name AS product_name,
        category,
        price,
        event_timestamp
    FROM {{ source('iceberg_bronze', 'brz_products_cdc') }}
    WHERE operation <> 'delete'
    ORDER BY product_id, event_timestamp, ingested_at DESC
),

raw AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY event_timestamp) AS rn
    FROM deduped
),

with_prev AS (
    SELECT
        *,
        LAG(product_name) OVER (PARTITION BY product_id ORDER BY event_timestamp) AS prev_name,
        LAG(category) OVER (PARTITION BY product_id ORDER BY event_timestamp) AS prev_category,
        LAG(price) OVER (PARTITION BY product_id ORDER BY event_timestamp) AS prev_price
    FROM raw
)

-- Keep only the FIRST row per product (rn=1) and every row where at least
-- one tracked attribute actually changed vs the immediately previous CDC
-- event -- collapsing consecutive duplicate values (e.g. a benign UPDATE
-- that only touched updated_at) down to real version boundaries.
SELECT
    product_id,
    product_name,
    category,
    price,
    event_timestamp AS changed_at
FROM with_prev
WHERE rn = 1
   OR product_name IS DISTINCT FROM prev_name
   OR category IS DISTINCT FROM prev_category
   OR price IS DISTINCT FROM prev_price
