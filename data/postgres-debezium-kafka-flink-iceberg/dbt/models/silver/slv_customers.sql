-- Silver Layer: Cleaned & Transformed Streaming Data
-- Filters out deleted records, deduplicates, applies data quality rules.

{{ config(
    materialized='table',
    meta={
        'owner': 'data-eng',
        'layer': 'silver',
        'dependencies': ['brz_customers_cdc']
    },
    columns=[
        {'name': 'customer_id', 'type': 'INT'},
        {'name': 'name', 'type': 'STRING'},
        {'name': 'email', 'type': 'STRING'},
        {'name': 'phone', 'type': 'STRING'},
        {'name': 'country', 'type': 'STRING'},
        {'name': 'created_at', 'type': 'BIGINT'},
        {'name': 'updated_at', 'type': 'BIGINT'},
        {'name': 'operation', 'type': 'STRING'},
        {'name': 'event_timestamp', 'type': 'TIMESTAMP(3) WITH LOCAL TIME ZONE'},
        {'name': 'ingested_at', 'type': 'TIMESTAMP(3) WITH LOCAL TIME ZONE NOT NULL'}
    ],
    connector_properties=iceberg_connector_properties('silver', 'slv_customers')
) }}

SELECT
    customer_id,
    name,
    email,
    phone,
    country,
    created_at,
    updated_at,
    operation,
    event_timestamp,
    ingested_at
FROM (
    SELECT
        customer_id,
        name,
        LOWER(email) as email,
        phone,
        country,
        created_at,
        updated_at,
        operation,
        event_timestamp,
        ingested_at,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY event_timestamp DESC
        ) as rn
    FROM {{ ref('brz_customers_cdc') }}
    WHERE operation <> 'delete'
)
WHERE rn = 1  -- Only latest version of each customer
