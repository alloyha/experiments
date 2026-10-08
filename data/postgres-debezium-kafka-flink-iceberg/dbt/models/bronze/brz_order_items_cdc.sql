-- Bronze Layer: Raw CDC Data from Kafka
-- This model represents raw data from Kafka CDC without transformations.

{{ config(
    materialized='table',
    type='streaming',
    meta={
        'owner': 'data-eng',
        'layer': 'bronze',
        'source': 'kafka_cdc'
    },
    connector_properties=iceberg_connector_properties('bronze', 'brz_order_items_cdc')
) }}

-- Raw order_items from Kafka CDC topic (debezium json envelope). Line items
-- are immutable once created (no update_order_item generator path), so
-- unlike customers/products this never needs SCD2 -- see product_metrics_current
-- for how it's aggregated.
SELECT
    `after`.`id` as order_item_id,
    `after`.`order_id` as order_id,
    `after`.`product_id` as product_id,
    `after`.`quantity` as quantity,
    CAST(`after`.`unit_price` AS DECIMAL(10,2)) as unit_price,
    `after`.`created_at` as created_at,
    `op` as operation,
    TO_TIMESTAMP_LTZ(`ts_ms`, 3) as event_timestamp,
    CURRENT_TIMESTAMP as ingested_at
FROM {{ source('kafka', 'cdc_source_order_items') }}
WHERE `after` IS NOT NULL
