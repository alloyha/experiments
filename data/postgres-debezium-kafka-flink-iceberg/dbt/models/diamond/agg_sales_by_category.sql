{{
  config(
    materialized='view',
    schema='diamond',
    meta={
      'owner': 'analytics-team',
      'layer': 'diamond',
      'grain': 'one row per product category',
      'refresh_frequency': 'real-time (view, not materialized)',
    }
  )
}}

-- 💎 DIAMOND LAYER - SALES BY CATEGORY (aggregate over obt_product_sales)
--
-- Materialized as a VIEW, not a table -- storage-lean strategy: the source
-- (obt_product_sales) is already a small, indexed Postgres table, so
-- recomputing this on every query is cheap and avoids storing yet another
-- copy of aggregated data that would need to be rebuilt every dbt cycle.
--
-- Grain: one row per category
-- Source: obt_product_sales

SELECT
  category,
  COUNT(*) AS total_products,
  COALESCE(SUM(units_sold), 0) AS total_units_sold,
  COALESCE(SUM(total_revenue), 0)::NUMERIC(18,2) AS total_revenue,
  ROUND(AVG(current_price), 2) AS avg_price,
  ROUND(
    100.0 * SUM(total_revenue)
    / NULLIF(SUM(SUM(total_revenue)) OVER (), 0),
    2
  ) AS pct_of_total_revenue

FROM {{ ref('obt_product_sales') }}
GROUP BY category
