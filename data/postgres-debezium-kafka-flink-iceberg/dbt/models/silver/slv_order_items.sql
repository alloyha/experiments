-- Silver Layer: Cleaned & Transformed Streaming Data
-- Filters out deleted records, deduplicates, applies data quality rules.

{{ config(
    materialized='table',
    meta={
        'owner': 'data-eng',
        'layer': 'silver',
        'dependencies': ['brz_order_items_cdc']
    },
    columns=[
        {'name': 'order_item_id', 'type': 'INT'},
        {'name': 'order_id', 'type': 'INT'},
        {'name': 'product_id', 'type': 'INT'},
        {'name': 'quantity', 'type': 'INT'},
        {'name': 'unit_price', 'type': 'DECIMAL(10,2)'},
        {'name': 'created_at', 'type': 'BIGINT'},
        {'name': 'operation', 'type': 'STRING'},
        {'name': 'event_timestamp', 'type': 'TIMESTAMP(3) WITH LOCAL TIME ZONE'},
        {'name': 'ingested_at', 'type': 'TIMESTAMP(3) WITH LOCAL TIME ZONE NOT NULL'}
    ],
    connector_properties=iceberg_connector_properties('silver', 'slv_order_items')
) }}

SELECT
    order_item_id,
    order_id,
    product_id,
    quantity,
    unit_price,
    created_at,
    operation,
    event_timestamp,
    ingested_at
FROM (
    SELECT
        order_item_id,
        order_id,
        product_id,
        quantity,
        unit_price,
        created_at,
        operation,
        event_timestamp,
        ingested_at,
        ROW_NUMBER() OVER (
            PARTITION BY order_item_id
            ORDER BY event_timestamp DESC
        ) as rn
    FROM {{ ref('brz_order_items_cdc') }}
    WHERE operation <> 'delete'
)
WHERE rn = 1  -- Only latest version of each order_item (immutable in practice)
