-- Silver Layer: Cleaned & Transformed Streaming Data
-- Filters out deleted records, deduplicates, applies data quality rules.

{{ config(
    materialized='table',
    meta={
        'owner': 'data-eng',
        'layer': 'silver',
        'dependencies': ['brz_orders_cdc']
    },
    columns=[
        {'name': 'order_id', 'type': 'INT'},
        {'name': 'customer_id', 'type': 'INT'},
        {'name': 'order_date', 'type': 'BIGINT'},
        {'name': 'total_amount', 'type': 'DECIMAL(10,2)'},
        {'name': 'status', 'type': 'STRING'},
        {'name': 'created_at', 'type': 'BIGINT'},
        {'name': 'updated_at', 'type': 'BIGINT'},
        {'name': 'operation', 'type': 'STRING'},
        {'name': 'event_timestamp', 'type': 'TIMESTAMP(3) WITH LOCAL TIME ZONE'},
        {'name': 'ingested_at', 'type': 'TIMESTAMP(3) WITH LOCAL TIME ZONE NOT NULL'}
    ],
    connector_properties=iceberg_connector_properties('silver', 'slv_orders')
) }}

SELECT
    order_id,
    customer_id,
    order_date,
    total_amount,
    status,
    created_at,
    updated_at,
    operation,
    event_timestamp,
    ingested_at
FROM (
    SELECT
        order_id,
        customer_id,
        order_date,
        total_amount,
        status,
        created_at,
        updated_at,
        operation,
        event_timestamp,
        ingested_at,
        ROW_NUMBER() OVER (
            PARTITION BY order_id
            ORDER BY event_timestamp DESC
        ) as rn
    FROM {{ ref('brz_orders_cdc') }}
    WHERE operation <> 'delete'
)
WHERE rn = 1  -- Only latest version of each order
