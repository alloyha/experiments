{{
  config(
    materialized='view',
    schema='diamond',
    meta={
      'owner': 'analytics-team',
      'layer': 'diamond',
      'grain': 'one row per customer_region',
      'refresh_frequency': 'real-time (view, not materialized)',
    }
  )
}}

-- 💎 DIAMOND LAYER - CUSTOMER REGION SUMMARY (aggregate over obt_customer_360)
--
-- Materialized as a VIEW, not a table -- same storage-lean reasoning as
-- agg_sales_by_category: obt_customer_360 is already small and indexed, so
-- there's no benefit to storing a second, separately-refreshed copy of this
-- rollup. Mirrors agg_customer_segment_summary's shape, but by geography
-- instead of segment.
--
-- Grain: one row per customer_region
-- Source: obt_customer_360

SELECT
  customer_region,

  COUNT(*) AS total_customers,
  SUM(customer_lifetime_value) AS total_revenue,
  ROUND(AVG(customer_lifetime_value), 2) AS avg_ltv,
  ROUND(AVG(avg_order_value), 2) AS avg_order_value,
  SUM(total_orders) AS total_orders,

  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total_customers,
  ROUND(
    100.0 * SUM(customer_lifetime_value)
    / NULLIF(SUM(SUM(customer_lifetime_value)) OVER (), 0),
    2
  ) AS pct_of_total_revenue

FROM {{ ref('obt_customer_360') }}
GROUP BY customer_region
