{{
  config(
    materialized='incremental',
    unique_key=['month_date', 'customer_region'],
    incremental_strategy='merge',
    schema='diamond',
    indexes=[
      {'columns': ['month_date'], 'type': 'btree'},
      {'columns': ['customer_region'], 'type': 'btree'},
    ],
    meta={
      'owner': 'analytics-team',
      'layer': 'diamond',
      'grain': 'one row per customer_region + month',
      'refresh_frequency': '60min (incremental)',
    }
  )
}}

-- 💎 DIAMOND LAYER - MONTHLY SALES BY REGION (aggregate over obt_customer_orders)
--
-- Storage-lean strategy: INCREMENTAL, not a full-table rebuild every cycle.
-- This grain grows forever (new month = new rows), so a full rebuild would
-- rewrite the ENTIRE history every 60s just to add the current month's
-- numbers -- unnecessary WAL/write amplification in Postgres for months
-- that already closed and will never change again.
--
-- The is_incremental() filter below only reprocesses the CURRENT (still
-- accumulating) month plus anything newer -- fully closed past months are
-- never touched. incremental_strategy='merge' on (month_date,
-- customer_region) means re-processing the current month UPDATES its
-- existing row instead of duplicating it.
--
-- Grain: one row per customer_region + month
-- Source: obt_customer_orders

SELECT
  o.order_date_month AS month_date,
  o.order_year AS year_num,
  o.order_quarter AS quarter_num,
  o.order_month AS month_num,
  o.customer_region,

  COUNT(DISTINCT o.order_id) AS total_orders,
  COUNT(DISTINCT o.customer_id) AS unique_customers,
  SUM(o.order_amount) AS total_sales,
  ROUND(AVG(o.order_amount), 2) AS avg_order_value,

  ROUND(
    100.0 * SUM(o.order_amount)
    / SUM(SUM(o.order_amount)) OVER (PARTITION BY o.order_date_month),
    2
  ) AS region_pct_of_monthly_sales

FROM {{ ref('obt_customer_orders') }} o
WHERE o.customer_region IS NOT NULL
{% if is_incremental() %}
  AND o.order_date_month >= (SELECT COALESCE(MAX(month_date), '1900-01-01'::date) FROM {{ this }})
{% endif %}
GROUP BY 1, 2, 3, 4, 5
