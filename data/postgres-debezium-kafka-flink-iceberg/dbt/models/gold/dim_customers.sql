-- ============================================================================
-- GOLD: dim_customers -- SCD Type 2, reconstructed from Bronze's FULL history
--
-- Idempotent key: id (= slv_customers.customer_id). Scoped to the customer's
-- slowly-changing IDENTITY attributes (name/email/phone/country) only --
-- order-derived metrics (total_orders, lifetime_value, segment, ...) live in
-- customer_metrics_current instead (current-state, no version history: they
-- change on every order, which isn't what SCD2 is meant to model).
--
-- This model is a full, deterministic rebuild every cycle from
-- dim_customers_attributes_history (which itself reconstructs every
-- distinct version straight from brz_customers_cdc's append-only commit
-- log) -- there is no incremental/pre_hook bookkeeping here, and none is
-- needed: since the SOURCE is the complete history (not a "latest
-- snapshot"), recomputing valid_from/valid_to/is_current from scratch via
-- LAG()/LEAD() always produces the exact same, fully correct timeline,
-- however often (or rarely) this model runs -- and it can never miss an
-- attribute change that happened between two refreshes, unlike reading from
-- a periodically-polled "current state" table would.
--
-- properties       : full JSONB snapshot of the customer's identity
--                     attributes as of this version.
-- properties_diff   : JUST the keys whose value changed vs the immediately
--                     previous version (everything, on the very first
--                     version of a given id).
-- valid_from/valid_to/is_current: standard SCD2 validity window. valid_to
--                     comes directly from the next version's CDC event
--                     timestamp. valid_from for every version EXCEPT the
--                     first is likewise the real CDC event timestamp that
--                     introduced it -- but the FIRST version's valid_from is
--                     forced to -infinity, not the timestamp Bronze happened
--                     to first observe it: that's when Debezium's snapshot/
--                     CDC capture first saw the row, which can be later than
--                     facts that genuinely predate it (e.g. orders placed
--                     before the initial Debezium snapshot ran). Without this,
--                     a point-in-time join against such facts would find no
--                     matching dimension version at all.
-- ============================================================================

{{ config(materialized='table', schema='gold') }}

WITH source AS (
    SELECT
        customer_id AS id,
        {{ dim_customers_properties('h') }} AS properties,
        changed_at
    FROM {{ ref('dim_customers_attributes_history') }} AS h
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
