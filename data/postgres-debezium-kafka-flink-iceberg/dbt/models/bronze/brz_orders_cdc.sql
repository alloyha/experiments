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
    connector_properties=iceberg_connector_properties('bronze', 'brz_orders_cdc')
) }}

-- Raw orders from Kafka CDC topic (debezium json envelope)
SELECT
    `after`.`id` as order_id,
    `after`.`customer_id` as customer_id,
    `after`.`order_date` as order_date,
    CAST(`after`.`total_amount` AS DECIMAL(10,2)) as total_amount,
    `after`.`status` as status,
    `after`.`created_at` as created_at,
    `after`.`updated_at` as updated_at,
    `op` as operation,
    TO_TIMESTAMP_LTZ(`ts_ms`, 3) as event_timestamp,
    CURRENT_TIMESTAMP as ingested_at
FROM {{ source('kafka', 'cdc_source_orders') }}
WHERE `after` IS NOT NULL
