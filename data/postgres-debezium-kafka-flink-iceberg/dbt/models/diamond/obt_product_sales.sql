{{
  config(
    materialized='table',
    schema='diamond',
    indexes=[
      {'columns': ['category'], 'type': 'btree'},
    ],
    meta={
      'owner': 'analytics-team',
      'layer': 'diamond',
      'grain': 'one row per product, current state (with real sales metrics)',
      'refresh_frequency': 'real-time',
    }
  )
}}

-- 💎 DIAMOND LAYER - PRODUCT SALES OBT
--
-- Purpose: the product-grain cube obt_product_catalog couldn't provide --
-- real revenue/units sold, because orders didn't reference products. Now
-- possible via order_items. Mirrors obt_customer_360's pattern exactly:
-- dim_products (current identity) + product_metrics_current (sales
-- aggregates, current-state -- not SCD2, same reasoning as
-- customer_metrics_current).
--
-- Same lag-handling as obt_customer_360: dim_products (Bronze-sourced)
-- refreshes every cycle, but product_metrics_current depends on Silver's
-- slower batch, so a just-created product can briefly have no
-- product_metrics_current row yet. COALESCE to zero rather than surfacing
-- NULL -- a product with no sales data yet and one with confirmed zero
-- sales are the same thing from a BI consumer's perspective.
--
-- Grain: one row per product_id
-- Source: dim_products (SCD2, current version), product_metrics_current

SELECT
  dp.id AS product_id,
  dp.properties->>'name' AS product_name,
  dp.properties->>'category' AS category,
  (dp.properties->>'price')::numeric AS current_price,

  COALESCE(pm.orders_containing_product, 0) AS orders_containing_product,
  COALESCE(pm.units_sold, 0) AS units_sold,
  COALESCE(pm.total_revenue, 0) AS total_revenue,
  COALESCE(pm.avg_sold_price, 0) AS avg_sold_price,
  pm.first_sold_at,
  pm.last_sold_at,

  current_timestamp AS record_updated_at

FROM {{ ref('dim_products') }} dp
LEFT JOIN {{ ref('product_metrics_current') }} pm
  ON dp.id = pm.product_id
WHERE dp.is_current = true
