{{
  config(
    materialized='table',
    schema='marts',
    location='s3://iceberg-warehouse/marts/customer_orders',
    indexes=[
        {'columns': ['customer_id'], 'type': 'HASH'}
    ]
  )
}}

SELECT
    c.id as customer_id,
    c.name as customer_name,
    c.email,
    c.country,
    COUNT(DISTINCT o.id) as total_orders,
    SUM(o.total_amount) as total_spent,
    MAX(o.order_date) as last_order_date,
    CURRENT_TIMESTAMP() as dbt_updated_at
FROM {{ ref('stg_customers') }} c
LEFT JOIN {{ source('kafka_cdc', 'orders') }} o
    ON c.id = o.customer_id
    AND o.__deleted = false
GROUP BY
    c.id,
    c.name,
    c.email,
    c.country
