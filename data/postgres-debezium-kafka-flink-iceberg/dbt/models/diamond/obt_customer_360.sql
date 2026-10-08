{{
  config(
    materialized='table',
    schema='diamond',
    indexes=[
      {'columns': ['customer_segment'], 'type': 'btree'},
      {'columns': ['customer_region'], 'type': 'btree'},
    ],
    meta={
      'owner': 'analytics-team',
      'layer': 'diamond',
      'grain': 'one row per customer, current state',
      'refresh_frequency': 'real-time',
      'pii_columns': ['customer_name', 'customer_email'],
    }
  )
}}

-- 💎 DIAMOND LAYER - CUSTOMER 360 OBT
--
-- Purpose: single denormalized "as of right now" view per customer, for BI
-- tools that want one row per customer instead of one row per order.
--
-- This is deliberately a CURRENT-STATE cube, not a point-in-time one:
--   - identity attributes come from dim_customers WHERE is_current -- the
--     customer's name/email/phone/country/region as they are TODAY, not as
--     of any particular order (that's what obt_customer_orders is for).
--   - order-derived metrics come from customer_metrics_current, which is
--     itself always "as of the last refresh" (see that model's header).
--
-- dim_customers (Bronze-sourced) refreshes on every cycle; customer_metrics_
-- current depends on Silver's batch, which runs on its own 60s cadence in a
-- different container. Since data-generator never stops, there are always a
-- few customers created within the last cycle whose identity has already
-- reached dim_customers but whose metrics haven't reached Silver yet -- the
-- LEFT JOIN below finds no customer_metrics_current row for them. Rather
-- than surface that as NULL (which would make a real "New" customer
-- indistinguishable from a genuine data quality gap, and break a NOT NULL
-- test on customer_segment), COALESCE these fields to the same values
-- customer_metrics_current itself would assign to a zero-order customer --
-- these two situations are observationally identical from a BI consumer's
-- perspective anyway (no order history yet either way).
--
-- customer_since is NOT dim_customers.valid_from of the current row -- for a
-- customer who has changed an attribute since being created, that would be
-- the timestamp of their MOST RECENT change, not when they first appeared.
-- The true first-seen timestamp comes from MIN(changed_at) over the full
-- history staging model instead.
--
-- Grain: one row per customer_id
-- Source: dim_customers (SCD2, current version), customer_metrics_current,
--         dim_customers_attributes_history (for customer_since)

WITH first_seen AS (
    SELECT
        customer_id,
        MIN(changed_at) AS customer_since
    FROM {{ ref('dim_customers_attributes_history') }}
    GROUP BY customer_id
)

SELECT
  dc.id AS customer_id,
  dc.properties->>'name' AS customer_name,
  dc.properties->>'email' AS customer_email,
  dc.properties->>'phone' AS customer_phone,
  dc.properties->>'country' AS customer_country,
  dc.properties->>'customer_region' AS customer_region,

  COALESCE(cm.customer_segment, 'New') AS customer_segment,
  COALESCE(cm.customer_status, 'Never Ordered') AS customer_status,
  COALESCE(cm.total_orders, 0) AS total_orders,
  COALESCE(cm.customer_lifetime_value, 0) AS customer_lifetime_value,
  COALESCE(cm.avg_order_value, 0) AS avg_order_value,
  cm.first_order_date,
  cm.last_order_date,
  cm.days_since_last_order,

  fs.customer_since,

  current_timestamp AS record_updated_at

FROM {{ ref('dim_customers') }} dc
LEFT JOIN {{ ref('customer_metrics_current') }} cm
  ON dc.id = cm.customer_id
LEFT JOIN first_seen fs
  ON dc.id = fs.customer_id
WHERE dc.is_current = true
