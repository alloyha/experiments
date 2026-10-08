{{
  config(
    materialized='table',
    schema='diamond',
    indexes=[
      {'columns': ['customer_segment'], 'type': 'btree'},
    ],
    meta={
      'owner': 'analytics-team',
      'layer': 'diamond',
      'grain': 'one row per customer_segment',
      'refresh_frequency': 'real-time',
    }
  )
}}

-- 💎 DIAMOND LAYER - CUSTOMER SEGMENT SUMMARY (aggregate over obt_customer_360)
--
-- Purpose: pre-aggregated rollup of the customer_360 cube for executive
-- dashboards -- same pattern as agg_daily_sales_by_segment (which rolls up
-- obt_customer_orders), but at customer grain instead of order grain: how
-- many customers are in each segment, how much revenue/LTV they represent,
-- and what share of the total each segment is.
--
-- Grain: one row per customer_segment
-- Source: obt_customer_360

SELECT
  customer_segment,

  COUNT(*) AS total_customers,
  SUM(customer_lifetime_value) AS total_revenue,
  ROUND(AVG(customer_lifetime_value), 2) AS avg_ltv,
  ROUND(AVG(avg_order_value), 2) AS avg_order_value,
  SUM(total_orders) AS total_orders,
  ROUND(AVG(days_since_last_order), 1) AS avg_days_since_last_order,

  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total_customers,
  ROUND(
    100.0 * SUM(customer_lifetime_value)
    / NULLIF(SUM(SUM(customer_lifetime_value)) OVER (), 0),
    2
  ) AS pct_of_total_revenue

FROM {{ ref('obt_customer_360') }}
GROUP BY customer_segment
