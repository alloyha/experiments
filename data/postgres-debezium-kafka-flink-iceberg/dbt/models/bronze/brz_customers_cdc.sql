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
    connector_properties=iceberg_connector_properties('bronze', 'brz_customers_cdc')
) }}

-- Raw customers from Kafka CDC topic (debezium json envelope)
SELECT
    `after`.`id` as customer_id,
    `after`.`name` as name,
    `after`.`email` as email,
    `after`.`phone` as phone,
    `after`.`country` as country,
    `after`.`created_at` as created_at,
    `after`.`updated_at` as updated_at,
    `op` as operation,
    TO_TIMESTAMP_LTZ(`ts_ms`, 3) as event_timestamp,
    CURRENT_TIMESTAMP as ingested_at
FROM {{ source('kafka', 'cdc_source_customers') }}
WHERE `after` IS NOT NULL
