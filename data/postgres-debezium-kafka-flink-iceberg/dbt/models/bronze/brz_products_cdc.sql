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
    connector_properties=iceberg_connector_properties('bronze', 'brz_products_cdc')
) }}

-- Raw products from Kafka CDC topic (debezium json envelope)
SELECT
    `after`.`id` as product_id,
    `after`.`name` as name,
    `after`.`category` as category,
    CAST(`after`.`price` AS DECIMAL(10,2)) as price,
    `after`.`created_at` as created_at,
    `after`.`updated_at` as updated_at,
    `op` as operation,
    TO_TIMESTAMP_LTZ(`ts_ms`, 3) as event_timestamp,
    CURRENT_TIMESTAMP as ingested_at
FROM {{ source('kafka', 'cdc_source_products') }}
WHERE `after` IS NOT NULL
