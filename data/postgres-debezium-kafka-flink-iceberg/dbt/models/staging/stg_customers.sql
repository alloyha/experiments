{{
  config(
    materialized='table',
    schema='staging',
    location='s3://iceberg-warehouse/staging/customers'
  )
}}

SELECT
    id,
    name,
    email,
    phone,
    country,
    created_at,
    updated_at,
    CURRENT_TIMESTAMP() as dbt_loaded_at
FROM {{ source('kafka_cdc', 'customers') }}
WHERE __deleted = false
