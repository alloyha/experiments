-- macros/generate_schema_name.sql
-- dbt-core's default generate_schema_name concatenates `<profile schema>_
-- <model's custom schema>` (e.g. profile schema='gold' + model schema='gold'
-- => "gold_gold"). Gold is built under the duckdb target (writing into the
-- attached Postgres db) while Diamond is built under the plain postgres
-- target and locates Gold's tables via `ref()` -- for that resolution to
-- land on the same physical schema regardless of which target/adapter
-- created it, schema naming must be deterministic and independent of the
-- profile's own default schema. Always use just the model's custom schema.

{% macro generate_schema_name(custom_schema_name, node) -%}
  {%- if custom_schema_name is none -%}
    {{ target.schema }}
  {%- else -%}
    {{ custom_schema_name | trim }}
  {%- endif -%}
{%- endmacro %}


-- macros/scd2_properties.sql
-- The JSONB "properties" payload for the SCD2 dim_customers/dim_products
-- models, factored out to a single macro so every reference (the history
-- reconstruction's LAG/LEAD comparisons and the final SELECT) stays in sync.
--
-- dim_customers_properties is deliberately scoped to identity attributes
-- only (name/email/phone/country) -- order-derived metrics (total_orders,
-- lifetime_value, segment, status) change on every order and live in
-- customer_metrics_current instead (current-state, not SCD2 -- see that
-- model's header comment for why).

{% macro dim_customers_properties(alias) %}
  jsonb_build_object(
    'name', {{ alias }}.customer_name,
    'email', {{ alias }}.customer_email,
    'phone', {{ alias }}.phone,
    'country', {{ alias }}.country,
    'customer_region', {{ alias }}.customer_region
  )
{% endmacro %}

{% macro dim_products_properties(alias) %}
  jsonb_build_object(
    'name', {{ alias }}.product_name,
    'category', {{ alias }}.category,
    'price', {{ alias }}.price
  )
{% endmacro %}

-- macros/iceberg_connector_properties.sql
-- Single source of truth for the Iceberg JDBC-catalog connector properties
-- shared by every bronze/silver model and by the session-pointer registration
-- blocks in setup_flink_bronze_sources() below. The catalog metastore lives
-- in its OWN Postgres database (`iceberg_catalog`) -- separate from cdc_db
-- (OLTP/Debezium) and analytics_db (Gold/Diamond) -- so this is the ONE place
-- that URI needs to be correct.

{% macro iceberg_connector_properties(catalog_database, catalog_table) %}
  {{ return({
    'connector': 'iceberg',
    'catalog-name': 'iceberg_catalog',
    'catalog-impl': 'org.apache.iceberg.jdbc.JdbcCatalog',
    'uri': 'jdbc:postgresql://postgres:5432/iceberg_catalog',
    'jdbc.user': 'postgres',
    'jdbc.password': 'postgres',
    'warehouse': 's3://iceberg-warehouse/',
    'io-impl': 'org.apache.iceberg.aws.s3.S3FileIO',
    'client.region': 'us-east-1',
    's3.endpoint': 'http://minio:9000',
    's3.path-style-access': 'true',
    's3.access-key-id': 'minioadmin',
    's3.secret-access-key': 'minioadmin',
    'catalog-database': catalog_database,
    'catalog-table': catalog_table
  }) }}
{% endmacro %}

-- Renders the same properties as a `'key' = 'value', ...` WITH-clause body,
-- for use inside a raw CREATE TABLE statement built with run_query() (the
-- session-pointer registrations below can't use connector_properties config
-- since they aren't materializations).
{% macro iceberg_connector_with_clause(catalog_database, catalog_table) %}
  {% set props = iceberg_connector_properties(catalog_database, catalog_table) %}
  {% for property_name in props %} '{{ property_name }}' = '{{ props[property_name] }}'{% if not loop.last %},{% endif %}
  {% endfor %}
{% endmacro %}

-- macros/get_date_trunc.sql

{% macro get_date_trunc(column, date_part='day') %}
    DATE_TRUNC('{{ date_part }}', {{ column }})
{% endmacro %}

-- macros/cents_to_dollars.sql

{% macro cents_to_dollars(column) %}
    {{ column }} / 100.0
{% endmacro %}

-- macros/generate_surrogate_key.sql

{% macro generate_surrogate_key(columns) %}
    MD5(CONCAT(
    {%- for column in columns -%}
        CAST({{ column }} AS VARCHAR)
        {%- if not loop.last %}, {% endif -%}
    {%- endfor -%}
    ))
{% endmacro %}

-- macros/generate_alias_hash.sql

{% macro generate_alias_hash(column_name) %}
    SUBSTR(MD5({{ column_name }}), 1, 8)
{% endmacro %}

-- macros/create_sources.sql
-- Overrides dbt-flink-adapter's built-in create_sources() (wired via its own
-- on-run-start hook). The built-in version issues a CREATE TABLE for every
-- source in the whole project -- including our Postgres-only sources, which
-- have no connector_properties/data_type set and break parsing on the Flink
-- target. We only ever need the Kafka source, handled explicitly below.

{% macro create_sources() %}
{% endmacro %}

-- macros/setup_flink_bronze_sources.sql
-- (Re)creates the Kafka-backed source tables the bronze models read from.
-- Runs on every invocation against the flink target since the Flink SQL
-- Gateway session (and its in-memory catalog) is reused across dbt runs.
--
-- Also re-registers brz_customers_cdc as a session-scoped pointer to the
-- existing Iceberg table (schema + connector properties only, no AS SELECT --
-- this must NEVER resubmit the streaming insert job). Necessary because Flink
-- SQL Gateway sessions are ephemeral: bronze's actual CREATE TABLE AS SELECT
-- runs once at container startup in session A, but later dbt invocations
-- (silver's 60s loop) restore/create a different session B whose in-memory
-- catalog has no record of brz_customers_cdc, breaking `FROM brz_customers_cdc`
-- in silver even though the underlying Iceberg data/job is alive and well.
--
-- IMPORTANT: this macro body must render to whitespace only (aside from the
-- log() calls' side effects) -- dbt executes an on-run-start hook's rendered
-- text as a SQL statement, so any literal comment/prose left inside the
-- macro body (as opposed to above it, here) becomes a broken "statement" that
-- Flink's parser rejects with a SqlParserEOFException.

{% macro setup_flink_bronze_sources() %}
  {% if execute and target.type == 'flink' %}
    {% do run_query('DROP TABLE IF EXISTS cdc_source_customers') %}

    {% set create_customers_source %}
      CREATE TABLE cdc_source_customers (
        `after` ROW<`id` INT, `name` STRING, `email` STRING, `phone` STRING, `country` STRING, `created_at` BIGINT, `updated_at` BIGINT>,
        `op` STRING,
        `ts_ms` BIGINT
      ) WITH (
        'connector' = 'kafka',
        'topic' = 'customers',
        'properties.bootstrap.servers' = 'kafka:29092',
        'properties.group.id' = 'flink-bronze-customers',
        'format' = 'json',
        'json.fail-on-missing-field' = 'false',
        'json.ignore-parse-errors' = 'true',
        'scan.startup.mode' = 'earliest-offset'
      )
    {% endset %}
    {% do run_query(create_customers_source) %}
    {{ log("Kafka source table cdc_source_customers (re)created", info=true) }}

    {% set register_bronze_pointer %}
      CREATE TABLE IF NOT EXISTS brz_customers_cdc (
        `customer_id` INT,
        `name` STRING,
        `email` STRING,
        `phone` STRING,
        `country` STRING,
        `created_at` BIGINT,
        `updated_at` BIGINT,
        `operation` STRING,
        `event_timestamp` TIMESTAMP(3) WITH LOCAL TIME ZONE,
        `ingested_at` TIMESTAMP(3) WITH LOCAL TIME ZONE NOT NULL
      ) WITH (
        {{ iceberg_connector_with_clause('bronze', 'brz_customers_cdc') }}
      )
    {% endset %}
    {% do run_query(register_bronze_pointer) %}
    {{ log("brz_customers_cdc pointer registered in current Flink session", info=true) }}

    {% do run_query('DROP TABLE IF EXISTS cdc_source_orders') %}

    {% set create_orders_source %}
      CREATE TABLE cdc_source_orders (
        `after` ROW<`id` INT, `customer_id` INT, `order_date` BIGINT, `total_amount` STRING, `status` STRING, `created_at` BIGINT, `updated_at` BIGINT>,
        `op` STRING,
        `ts_ms` BIGINT
      ) WITH (
        'connector' = 'kafka',
        'topic' = 'orders',
        'properties.bootstrap.servers' = 'kafka:29092',
        'properties.group.id' = 'flink-bronze-orders',
        'format' = 'json',
        'json.fail-on-missing-field' = 'false',
        'json.ignore-parse-errors' = 'true',
        'scan.startup.mode' = 'earliest-offset'
      )
    {% endset %}
    {% do run_query(create_orders_source) %}
    {{ log("Kafka source table cdc_source_orders (re)created", info=true) }}

    {% set register_bronze_orders_pointer %}
      CREATE TABLE IF NOT EXISTS brz_orders_cdc (
        `order_id` INT,
        `customer_id` INT,
        `order_date` BIGINT,
        `total_amount` DECIMAL(10,2),
        `status` STRING,
        `created_at` BIGINT,
        `updated_at` BIGINT,
        `operation` STRING,
        `event_timestamp` TIMESTAMP(3) WITH LOCAL TIME ZONE,
        `ingested_at` TIMESTAMP(3) WITH LOCAL TIME ZONE NOT NULL
      ) WITH (
        {{ iceberg_connector_with_clause('bronze', 'brz_orders_cdc') }}
      )
    {% endset %}
    {% do run_query(register_bronze_orders_pointer) %}
    {{ log("brz_orders_cdc pointer registered in current Flink session", info=true) }}

    {% do run_query('DROP TABLE IF EXISTS cdc_source_products') %}

    {% set create_products_source %}
      CREATE TABLE cdc_source_products (
        `after` ROW<`id` INT, `name` STRING, `category` STRING, `price` STRING, `created_at` BIGINT, `updated_at` BIGINT>,
        `op` STRING,
        `ts_ms` BIGINT
      ) WITH (
        'connector' = 'kafka',
        'topic' = 'products',
        'properties.bootstrap.servers' = 'kafka:29092',
        'properties.group.id' = 'flink-bronze-products',
        'format' = 'json',
        'json.fail-on-missing-field' = 'false',
        'json.ignore-parse-errors' = 'true',
        'scan.startup.mode' = 'earliest-offset'
      )
    {% endset %}
    {% do run_query(create_products_source) %}
    {{ log("Kafka source table cdc_source_products (re)created", info=true) }}

    {% set register_bronze_products_pointer %}
      CREATE TABLE IF NOT EXISTS brz_products_cdc (
        `product_id` INT,
        `name` STRING,
        `category` STRING,
        `price` DECIMAL(10,2),
        `created_at` BIGINT,
        `updated_at` BIGINT,
        `operation` STRING,
        `event_timestamp` TIMESTAMP(3) WITH LOCAL TIME ZONE,
        `ingested_at` TIMESTAMP(3) WITH LOCAL TIME ZONE NOT NULL
      ) WITH (
        {{ iceberg_connector_with_clause('bronze', 'brz_products_cdc') }}
      )
    {% endset %}
    {% do run_query(register_bronze_products_pointer) %}
    {{ log("brz_products_cdc pointer registered in current Flink session", info=true) }}
  {% endif %}
{% endmacro %}

-- macros/flink__create_table_as override.
-- The adapter-bundled version always emits `DROP TABLE IF EXISTS x; CREATE
-- TABLE x WITH (...) AS (SELECT ...)`. For our connector='iceberg' tables,
-- Flink's `DROP TABLE` only removes the ephemeral session-local declaration --
-- it never calls the Iceberg catalog's own dropTable, so the physical table
-- (and all its previously committed rows) survives every "drop+recreate"
-- cycle. Each subsequent CTAS then appends a fresh batch on top of the old
-- one instead of replacing it, so a table meant to hold "current state"
-- (e.g. silver, deduplicated by primary key) accumulates duplicates forever.
--
-- Fix: for non-streaming ('table') models, replace CTAS with an explicit
-- `CREATE TABLE IF NOT EXISTS` (schema declared via the `columns` model
-- config) followed by `INSERT OVERWRITE`, which Flink executes as a true
-- full-table replace in batch mode. Streaming models (type='streaming', e.g.
-- bronze) are left on the original CTAS path unchanged -- overwrite has no
-- meaning for an unbounded append-only sink.

{% macro flink__create_table_as(temporary, relation, sql) -%}
  {% set type = config.get('type', none) %}
  {% set columns = config.get('columns', none) %}
  {% if type != 'streaming' and columns %}
    {% set connector_properties = config.get('default_connector_properties', {}) %}
    {% set _dummy = connector_properties.update(config.get('connector_properties', {})) %}
    {% set create_sql %}
      create table if not exists {{ this.render() }} (
        {% for col in columns %} `{{ col.name }}` {{ col.type }}{% if not loop.last %},{% endif %}
        {% endfor %}
      ) with (
        {% for property_name in connector_properties %} '{{ property_name }}' = '{{ connector_properties[property_name] }}'{% if not loop.last %},{% endif %}
        {% endfor %}
      )
    {% endset %}
    {% do run_query(create_sql) %}
    {{ return("insert overwrite " ~ this.render() ~ " " ~ sql) }}
  {% else %}
    {{ return(default__flink__create_table_as(temporary, relation, sql)) }}
  {% endif %}
{%- endmacro %}

-- Original adapter behavior, kept verbatim and renamed so the override above
-- can delegate to it for streaming models without duplicating the logic.
{% macro default__flink__create_table_as(temporary, relation, sql) -%}
  {% set type = config.get('type', None) %}
  {%- set sql_header = config.get('sql_header', none) -%}
  {% set connector_properties = config.get('default_connector_properties', {}) %}
  {% set _dummy = connector_properties.update(config.get('connector_properties', {})) %}
  {% set execution_config = config.get('default_execution_config', {}) %}
  {% set _dummy = execution_config.update(config.get('execution_config', {})) %}
  {% set upgrade_mode = config.get('upgrade_mode', 'stateless') %}
  {% set job_state = config.get('job_state', 'running') %}

  {{ sql_header if sql_header is not none }}
  /** upgrade_mode('{{upgrade_mode}}') */ /** job_state('{{job_state}}') */
  {% if execution_config %}/** execution_config('{% for cfg_name in execution_config %}{{cfg_name}}={{execution_config[cfg_name]}}{% if not loop.last %};{% endif %}{% endfor %}') */{% endif %}
  /** drop_statement('drop {% if temporary: -%}temporary {%- endif %}table if exists `{{ this.render() }}`') */
  create {% if temporary: -%}temporary {%- endif %}table
    {{ this.render() }}
    {% if type %}/** mode('{{type}}')*/{% endif %}
  with (
    {% for property_name in connector_properties %} '{{ property_name }}' = '{{ connector_properties[property_name] }}'{% if not loop.last %},{% endif %}
    {% endfor %}
  )
  as (
    {{ sql }}
  );
{%- endmacro %}
