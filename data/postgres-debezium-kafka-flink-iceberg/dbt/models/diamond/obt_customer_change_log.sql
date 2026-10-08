{{
  config(
    materialized='table',
    schema='diamond',
    indexes=[
      {'columns': ['customer_id'], 'type': 'btree'},
    ],
    meta={
      'owner': 'analytics-team',
      'layer': 'diamond',
      'grain': 'one row per customer attribute change (SCD2 version)',
      'refresh_frequency': 'real-time',
      'pii_columns': ['customer_name', 'customer_email'],
    }
  )
}}

-- 💎 DIAMOND LAYER - CUSTOMER CHANGE LOG (audit view over dim_customers)
--
-- Purpose: dim_customers already carries every version + properties_diff;
-- this flattens it into an audit-friendly timeline for BI tools that would
-- rather not parse JSONB directly -- a version_number, a human-readable
-- change_summary string, and how long each version was in effect.
--
-- version_duration is NULL for the very first version of a customer (its
-- valid_from is forced to -infinity -- see dim_customers.sql for why --
-- so valid_to - valid_from would be a meaningless "infinite" interval) and
-- for the current version (valid_to IS NULL, still open-ended).
--
-- Grain: one row per (customer_id, SCD2 version) -- same grain as
-- dim_customers itself; this is a presentation layer over it, not a new
-- reconstruction.
-- Source: dim_customers

SELECT
  dc.scd_key,
  dc.id AS customer_id,
  ROW_NUMBER() OVER (PARTITION BY dc.id ORDER BY dc.valid_from) AS version_number,

  dc.properties->>'name' AS customer_name,
  dc.properties->>'email' AS customer_email,
  dc.properties_diff AS changed_fields,
  (
    SELECT string_agg(kv.key || ': ' || kv.value, ', ')
    FROM jsonb_each_text(dc.properties_diff) AS kv
  ) AS change_summary,

  dc.valid_from,
  dc.valid_to,
  dc.is_current,
  CASE
    WHEN dc.valid_from = '-infinity'::timestamptz OR dc.valid_to IS NULL THEN NULL
    ELSE dc.valid_to - dc.valid_from
  END AS version_duration

FROM {{ ref('dim_customers') }} dc
