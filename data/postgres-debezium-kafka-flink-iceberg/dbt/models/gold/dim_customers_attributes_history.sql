{#- see dim_products_history.sql for the identical pattern applied to products;
   database is conditional on target.type for the same reason as elsewhere. #}
{{
  config(
    materialized='table',
    database=('pg' if target.type == 'duckdb' else none),
    schema='gold'
  )
}}

-- ============================================================================
-- GOLD (staging): every distinct version of a customer's IDENTITY attributes
-- (name/email/phone/country) that ever existed, reconstructed from Bronze's
-- full append-only CDC history.
--
-- Purpose: feeds dim_customers.sql (SCD2). Deliberately scoped to the
-- slowly-changing descriptive attributes only -- order-derived metrics
-- (total_orders, lifetime_value, segment, ...) change on every order, which
-- isn't what SCD2 is for; those live in customer_metrics_current instead
-- (current-state only, no version history).
--
-- Reading straight from brz_customers_cdc -- NOT from slv_customers or a
-- "latest snapshot" -- is what makes the SCD2 history complete regardless
-- of how often this model gets rebuilt: two attribute changes that both
-- happened between two refreshes still show up as two distinct rows here.
--
-- Source: Iceberg Bronze (brz_customers_cdc), read via dbt-duckdb's native
--         "iceberg" plugin (pyiceberg SqlCatalog against the same JDBC
--         catalog Flink writes to).
-- Materialized in: PostgreSQL (attached via DuckDB)
-- ============================================================================

-- Bronze can contain exact-duplicate CDC events for the same (id, event_timestamp)
-- pair -- e.g. Flink resubmitting the streaming job without prior checkpoint
-- state replays already-seen Kafka messages verbatim. Collapse those first:
-- without this, two identical rows tied on event_timestamp make the LAG-based
-- transition detection below inconsistent (window functions don't guarantee a
-- shared tie-break order across separately-computed ROW_NUMBER/LAG calls),
-- which can fabricate a spurious zero-width SCD2 version.
WITH deduped AS (
    SELECT DISTINCT ON (customer_id, event_timestamp)
        customer_id,
        name AS customer_name,
        email AS customer_email,
        phone,
        country,
        event_timestamp
    FROM {{ source('iceberg_bronze', 'brz_customers_cdc') }}
    WHERE operation <> 'delete'
    ORDER BY customer_id, event_timestamp, ingested_at DESC
),

raw AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY event_timestamp) AS rn
    FROM deduped
),

with_prev AS (
    SELECT
        *,
        LAG(customer_name) OVER (PARTITION BY customer_id ORDER BY event_timestamp) AS prev_name,
        LAG(customer_email) OVER (PARTITION BY customer_id ORDER BY event_timestamp) AS prev_email,
        LAG(phone) OVER (PARTITION BY customer_id ORDER BY event_timestamp) AS prev_phone,
        LAG(country) OVER (PARTITION BY customer_id ORDER BY event_timestamp) AS prev_country
    FROM raw
)

SELECT
    customer_id,
    customer_name,
    customer_email,
    phone,
    country,
    country AS customer_region,
    event_timestamp AS changed_at
FROM with_prev
WHERE rn = 1
   OR customer_name IS DISTINCT FROM prev_name
   OR customer_email IS DISTINCT FROM prev_email
   OR phone IS DISTINCT FROM prev_phone
   OR country IS DISTINCT FROM prev_country
