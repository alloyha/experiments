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
      'grain': 'one row per product, current state',
      'refresh_frequency': 'real-time',
    }
  )
}}

-- 💎 DIAMOND LAYER - PRODUCT CATALOG OBT
--
-- Purpose: single denormalized "as of right now" view per product, mirroring
-- obt_customer_360's pattern for the product grain. This is a catalog/price
-- -history cube, NOT a sales cube -- it has no units_sold/revenue, because
-- those require order_items (see obt_product_sales for that).
--
-- price_changes_count/min/max_price_ever come from dim_products_history
-- (Bronze's full reconstructed version log), not from dim_products itself --
-- price_points includes the product's very first snapshot, which isn't
-- itself a "change", hence the -1.
--
-- Grain: one row per product_id
-- Source: dim_products (SCD2, current version), dim_products_history

WITH price_history AS (
    SELECT
        product_id,
        COUNT(*) AS price_points,
        MIN(price) AS min_price_ever,
        MAX(price) AS max_price_ever,
        MIN(changed_at) AS first_seen_at,
        MAX(changed_at) AS last_changed_at
    FROM {{ ref('dim_products_history') }}
    GROUP BY product_id
)

SELECT
  dp.id AS product_id,
  dp.properties->>'name' AS product_name,
  dp.properties->>'category' AS category,
  (dp.properties->>'price')::numeric AS current_price,

  GREATEST(ph.price_points - 1, 0) AS price_changes_count,
  ph.min_price_ever,
  ph.max_price_ever,
  ph.first_seen_at,
  ph.last_changed_at,

  current_timestamp AS record_updated_at

FROM {{ ref('dim_products') }} dp
LEFT JOIN price_history ph
  ON dp.id = ph.product_id
WHERE dp.is_current = true
