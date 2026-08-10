-- ============================================================================
-- GOLD: dim_products -- SCD Type 2, reconstructed from Bronze's FULL history
--
-- Idempotent key: id (= slv_products.product_id). This model is a full,
-- deterministic rebuild every cycle from dim_products_history (which itself
-- reconstructs every distinct version straight from brz_products_cdc's
-- append-only commit log) -- there is no incremental/pre_hook bookkeeping
-- here, and none is needed: since the SOURCE is the complete history (not a
-- "latest snapshot"), recomputing valid_from/valid_to/is_current from
-- scratch via LAG()/LEAD() always produces the exact same, fully correct
-- timeline, however often (or rarely) this model runs.
--
-- properties      : full JSONB snapshot of the product's business
--                    attributes as of this version.
-- properties_diff : JUST the keys whose value changed vs the immediately
--                    previous version (everything, on the very first
--                    version of a given id) -- e.g. a price reajustment
--                    shows up as properties_diff = {"price": 54.90}.
-- valid_from/valid_to/is_current: standard SCD2 validity window. Every
--                    version's valid_to, and every version's valid_from
--                    EXCEPT the first, come directly from the real CDC event
--                    timestamps that introduced them. The FIRST version's
--                    valid_from is forced to -infinity instead -- see
--                    dim_customers.sql for why (facts can predate the
--                    moment Bronze/Debezium first observed the row).
-- ============================================================================

{{ config(materialized='table', schema='gold') }}

WITH source AS (
    SELECT
        product_id AS id,
        {{ dim_products_properties('h') }} AS properties,
        changed_at
    FROM {{ ref('dim_products_history') }} AS h
),

with_bounds AS (
    SELECT
        *,
        LAG(properties) OVER (PARTITION BY id ORDER BY changed_at) AS prev_properties,
        LEAD(changed_at) OVER (PARTITION BY id ORDER BY changed_at) AS next_changed_at
    FROM source
)

SELECT
    md5(id::text || '|' || changed_at::text) AS scd_key,
    id,
    properties,
    COALESCE(
        (
            SELECT jsonb_object_agg(new_kv.key, new_kv.value)
            FROM jsonb_each(properties) AS new_kv
            LEFT JOIN jsonb_each(COALESCE(prev_properties, '{}'::jsonb)) AS old_kv
                ON new_kv.key = old_kv.key
            WHERE old_kv.value IS DISTINCT FROM new_kv.value
        ),
        '{}'::jsonb
    ) AS properties_diff,
    CASE WHEN prev_properties IS NULL THEN '-infinity'::timestamptz ELSE changed_at END AS valid_from,
    next_changed_at AS valid_to,
    next_changed_at IS NULL AS is_current
FROM with_bounds
