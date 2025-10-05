-- ============================================================
-- DDL Generator Unit Tests
-- ============================================================
-- Tests for all DDL generation functions in ddl_generator.sql

DO $$ BEGIN RAISE NOTICE 'Running DDL Generator Tests...'; END $$;

-- ============================================================
-- Test generate_foreign_keys_sql function
-- ============================================================

-- Test basic foreign key generation
DO $$
DECLARE
  v_fks jsonb := '[
    {"column": "customer_id", "reference": {"table": "customers", "column": "id", "ref_type": "business"}},
    {"column": "product_sk", "reference": {"table": "products", "ref_type": "surrogate"}}
  ]'::jsonb;
  v_result text;
BEGIN
  SELECT generate_foreign_keys_sql('test_schema', 'test_table', v_fks) INTO v_result;
  
  PERFORM assert_contains(v_result, 'customers', 'Should reference customers table');
  PERFORM assert_contains(v_result, 'products_sk', 'Should generate surrogate key reference');
  PERFORM assert_contains(v_result, 'FOREIGN KEY', 'Should contain foreign key syntax');
  
  RAISE NOTICE '[PASS] ddl_generator: generate_foreign_keys_sql_basic - Basic FK generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] ddl_generator: generate_foreign_keys_sql_basic - %', SQLERRM;
END;
$$;

-- Test empty foreign keys
DO $$
DECLARE
  v_result text;
BEGIN
  SELECT generate_foreign_keys_sql('test_schema', 'test_table', '[]'::jsonb) INTO v_result;
  
  PERFORM assert_true(v_result = '', 'Empty FK array should return empty string');
  
  RAISE NOTICE '[PASS] ddl_generator: generate_foreign_keys_sql_empty - Empty FK handling works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] ddl_generator: generate_foreign_keys_sql_empty - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test generate_indexes_sql function
-- ============================================================

-- Test index generation
DO $$
DECLARE
  v_result text;
BEGIN
  SELECT generate_indexes_sql(
    'test_schema', 
    'test_table', 
    'test_base',
    '["email", "status"]'::jsonb,
    true,
    '["tier", "category"]'::jsonb,
    'SCD1'
  ) INTO v_result;
  
  PERFORM assert_contains(v_result, 'CREATE INDEX', 'Should create indexes');
  PERFORM assert_contains(v_result, 'properties->>', 'Should create expression indexes');
  PERFORM assert_contains(v_result, 'gin (properties)', 'Should create GIN index');
  PERFORM assert_contains(v_result, 'updated_at', 'Should create timestamp index for SCD1');
  
  RAISE NOTICE '[PASS] ddl_generator: generate_indexes_sql_basic - Index generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] ddl_generator: generate_indexes_sql_basic - %', SQLERRM;
END;
$$;

-- Test SCD2 index generation
DO $$
DECLARE
  v_result text;
BEGIN
  SELECT generate_indexes_sql(
    'test_schema', 
    'test_table', 
    'test_base',
    '[]'::jsonb,
    false,
    '["tier"]'::jsonb,
    'SCD2'
  ) INTO v_result;
  
  PERFORM assert_contains(v_result, 'created_at', 'Should create created_at index for SCD2');
  PERFORM assert_contains(v_result, 'WHERE is_current', 'Should create partial indexes for SCD2');
  
  RAISE NOTICE '[PASS] ddl_generator: generate_indexes_sql_scd2 - SCD2 index generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] ddl_generator: generate_indexes_sql_scd2 - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test generate_scd1_ddl function
-- ============================================================

-- Test basic SCD1 table generation
DO $$
DECLARE
  v_entry jsonb := '{
    "name": "test_dim",
    "primary_keys": ["customer_id"],
    "index_properties_gin": true,
    "index_properties_keys": ["tier"],
    "physical_columns": [
      {"name": "account_id", "type": "text", "nullable": false},
      {"name": "region_code", "type": "text"}
    ]
  }'::jsonb;
  v_result text;
BEGIN
  SELECT generate_scd1_ddl('test_schema', 'test_dim', v_entry, '["id"]'::jsonb, false) INTO v_result;
  
  PERFORM assert_contains(v_result, 'CREATE TABLE', 'Should create table');
  PERFORM assert_contains(v_result, 'test_dim_sk bigserial PRIMARY KEY', 'Should have surrogate key');
  PERFORM assert_contains(v_result, 'customer_id text NOT NULL', 'Should have business key');
  PERFORM assert_contains(v_result, 'account_id text NOT NULL', 'Should have physical columns');
  PERFORM assert_contains(v_result, 'region_code text', 'Should have nullable physical column');
  PERFORM assert_contains(v_result, 'properties jsonb', 'Should have properties column');
  PERFORM assert_contains(v_result, 'row_hash text NOT NULL', 'Should have row_hash');
  PERFORM assert_contains(v_result, 'UNIQUE', 'Should have unique constraint');
  
  RAISE NOTICE '[PASS] ddl_generator: generate_scd1_ddl_basic - SCD1 DDL generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] ddl_generator: generate_scd1_ddl_basic - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test generate_scd2_ddl function
-- ============================================================

-- Test basic SCD2 table generation
DO $$
DECLARE
  v_entry jsonb := '{
    "name": "test_dim",
    "primary_keys": ["customer_id"],
    "physical_columns": [
      {"name": "account_id", "type": "bigint", "nullable": false}
    ]
  }'::jsonb;
  v_result text;
BEGIN
  SELECT generate_scd2_ddl('test_schema', 'test_dim', v_entry, '["id"]'::jsonb, false) INTO v_result;
  
  PERFORM assert_contains(v_result, 'CREATE TABLE', 'Should create table');
  PERFORM assert_contains(v_result, 'test_dim_sk bigserial PRIMARY KEY', 'Should have surrogate key');
  PERFORM assert_contains(v_result, 'customer_id text NOT NULL', 'Should have business key');
  PERFORM assert_contains(v_result, 'account_id bigint NOT NULL', 'Should have physical columns');
  PERFORM assert_contains(v_result, 'valid_from timestamptz NOT NULL', 'Should have SCD2 columns');
  PERFORM assert_contains(v_result, 'valid_to timestamptz', 'Should have valid_to');
  PERFORM assert_contains(v_result, 'is_current boolean', 'Should have is_current');
  PERFORM assert_contains(v_result, 'properties_diff jsonb', 'Should have properties_diff');
  PERFORM assert_contains(v_result, 'WHERE is_current', 'Should have current record index');
  
  RAISE NOTICE '[PASS] ddl_generator: generate_scd2_ddl_basic - SCD2 DDL generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] ddl_generator: generate_scd2_ddl_basic - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test generate_fact_ddl function
-- ============================================================

-- Test fact table generation
DO $$
DECLARE
  v_entry jsonb := '{
    "name": "fact_orders",
    "dimension_references": [
      {"dimension": "dim_customers", "fk_column": "customer_sk", "required": true},
      {"dimension": "dim_products", "fk_column": "product_sk", "required": false}
    ],
    "measures": [
      {"name": "quantity", "type": "integer", "nullable": false},
      {"name": "revenue", "type": "numeric(10,2)"},
      {"name": "discount_pct", "type": "numeric(5,2)"}
    ],
    "degenerate_dimensions": ["order_number", "line_item_id"],
    "event_timestamp_column": "order_timestamp"
  }'::jsonb;
  v_result text;
BEGIN
  SELECT generate_fact_ddl('test_schema', 'fact_orders', v_entry) INTO v_result;
  
  PERFORM assert_contains(v_result, 'CREATE TABLE', 'Should create table');
  PERFORM assert_contains(v_result, 'fact_orders_sk bigserial PRIMARY KEY', 'Should have surrogate key');
  PERFORM assert_contains(v_result, 'customer_sk bigint NOT NULL', 'Should have required FK');
  PERFORM assert_contains(v_result, 'product_sk bigint,', 'Should have optional FK');
  PERFORM assert_contains(v_result, 'quantity integer NOT NULL', 'Should have typed measures');
  PERFORM assert_contains(v_result, 'revenue numeric(10,2)', 'Should have decimal measures');
  PERFORM assert_contains(v_result, 'order_number text', 'Should have degenerate dimensions');
  PERFORM assert_contains(v_result, 'order_timestamp timestamptz NOT NULL', 'Should have event timestamp');
  PERFORM assert_contains(v_result, 'CREATE INDEX', 'Should create indexes');
  PERFORM assert_contains(v_result, 'FOREIGN KEY', 'Should create foreign keys');
  
  -- Should NOT contain properties JSONB
  IF v_result LIKE '%properties jsonb%' THEN
    RAISE EXCEPTION 'Fact tables should not have properties JSONB column';
  END IF;
  
  RAISE NOTICE '[PASS] ddl_generator: generate_fact_ddl_basic - Fact DDL generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] ddl_generator: generate_fact_ddl_basic - %', SQLERRM;
END;
$$;

-- Test fact table validation - missing dimension_references
SELECT run_test(
  'ddl_generator',
  'generate_fact_ddl_missing_dimensions',
  $$SELECT generate_fact_ddl('test_schema', 'fact_test', '{"name": "fact_test", "measures": [{"name": "amount", "type": "numeric"}]}'::jsonb)$$,
  NULL,
  true -- Should raise exception
);

-- Test fact table validation - missing measures
SELECT run_test(
  'ddl_generator',
  'generate_fact_ddl_missing_measures',
  $$SELECT generate_fact_ddl('test_schema', 'fact_test', '{"name": "fact_test", "dimension_references": [{"dimension": "dim_test", "fk_column": "test_sk"}]}'::jsonb)$$,
  NULL,
  true -- Should raise exception
);

-- ============================================================
-- Test generate_table_ddl function
-- ============================================================

-- Test dimension table dispatch
DO $$
DECLARE
  v_entry jsonb := '{
    "name": "dim_test",
    "entity_type": "dimension",
    "grain": "One row per customer",
    "scd": "SCD1",
    "primary_keys": ["customer_id"]
  }'::jsonb;
  v_result text;
BEGIN
  SELECT generate_table_ddl(v_entry, 'test_schema', false) INTO v_result;
  
  PERFORM assert_contains(v_result, 'CREATE TABLE', 'Should create table');
  PERFORM assert_contains(v_result, 'dim_test', 'Should use correct table name');
  PERFORM assert_contains(v_result, 'COMMENT ON TABLE', 'Should add grain comment');
  PERFORM assert_contains(v_result, 'One row per customer', 'Should include grain in comment');
  
  RAISE NOTICE '[PASS] ddl_generator: generate_table_ddl_dimension - Dimension dispatch works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] ddl_generator: generate_table_ddl_dimension - %', SQLERRM;
END;
$$;

-- Test fact table dispatch
DO $$
DECLARE
  v_entry jsonb := '{
    "name": "fact_orders",
    "entity_type": "transaction_fact",
    "grain": "One row per order line item",
    "entity_rationale": "Records each order transaction",
    "dimension_references": [
      {"dimension": "dim_customers", "fk_column": "customer_sk", "required": true}
    ],
    "measures": [
      {"name": "amount", "type": "numeric(10,2)"}
    ]
  }'::jsonb;
  v_result text;
BEGIN
  SELECT generate_table_ddl(v_entry, 'test_schema', false) INTO v_result;
  
  PERFORM assert_contains(v_result, 'CREATE TABLE', 'Should create table');
  PERFORM assert_contains(v_result, 'fact_orders', 'Should use correct table name');
  PERFORM assert_contains(v_result, 'customer_sk bigint', 'Should have dimension FK');
  PERFORM assert_contains(v_result, 'amount numeric(10,2)', 'Should have typed measures');
  PERFORM assert_contains(v_result, 'COMMENT ON TABLE', 'Should add grain comment');
  PERFORM assert_contains(v_result, 'Records each order transaction', 'Should include rationale');
  
  RAISE NOTICE '[PASS] ddl_generator: generate_table_ddl_fact - Fact dispatch works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] ddl_generator: generate_table_ddl_fact - %', SQLERRM;
END;
$$;

-- Test unknown entity type
SELECT run_test(
  'ddl_generator',
  'generate_table_ddl_unknown_entity_type',
  $$SELECT generate_table_ddl('{"name": "test", "entity_type": "unknown"}'::jsonb, 'test_schema', false)$$,
  NULL,
  true -- Should raise exception
);

-- Test missing name
SELECT run_test(
  'ddl_generator',
  'generate_table_ddl_missing_name',
  $$SELECT generate_table_ddl('{"entity_type": "dimension"}'::jsonb, 'test_schema', false)$$,
  NULL,
  true -- Should raise exception
);

-- ============================================================
-- Test generate_ddl function (pipeline level)
-- ============================================================

-- Test full pipeline DDL generation
DO $$
DECLARE
  v_pipeline jsonb := '{
    "schema": "analytics",
    "defaults": {
      "scd": "SCD1",
      "index_properties_gin": true
    },
    "tables": [
      {
        "name": "dim_customers",
        "entity_type": "dimension",
        "grain": "One row per customer",
        "scd": "SCD2",
        "primary_keys": ["customer_id"]
      },
      {
        "name": "fact_orders",
        "entity_type": "transaction_fact",
        "grain": "One row per order",
        "dimension_references": [
          {"dimension": "dim_customers", "fk_column": "customer_sk", "required": true}
        ],
        "measures": [
          {"name": "amount", "type": "numeric(10,2)"}
        ]
      }
    ]
  }'::jsonb;
  v_result text;
BEGIN
  SELECT generate_ddl(v_pipeline, 'analytics', true) INTO v_result;
  
  PERFORM assert_contains(v_result, 'CREATE SCHEMA', 'Should create schema');
  PERFORM assert_contains(v_result, 'dim_customers', 'Should contain dimension table');
  PERFORM assert_contains(v_result, 'fact_orders', 'Should contain fact table');
  PERFORM assert_contains(v_result, 'is_current boolean', 'Should apply SCD2 to dimension');
  PERFORM assert_contains(v_result, 'customer_sk bigint', 'Should have FK in fact');
  
  RAISE NOTICE '[PASS] ddl_generator: generate_ddl_pipeline - Pipeline DDL generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] ddl_generator: generate_ddl_pipeline - %', SQLERRM;
END;
$$;

-- Test single table DDL generation
DO $$
DECLARE
  v_table jsonb := '{
    "name": "single_table",
    "entity_type": "dimension",
    "grain": "test grain",
    "primary_keys": ["id"]
  }'::jsonb;
  v_result text;
BEGIN
  SELECT generate_ddl(v_table, 'test_schema', false) INTO v_result;
  
  PERFORM assert_contains(v_result, 'single_table', 'Should contain table name');
  PERFORM assert_contains(v_result, 'CREATE TABLE', 'Should create table');
  
  RAISE NOTICE '[PASS] ddl_generator: generate_ddl_single_table - Single table DDL generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] ddl_generator: generate_ddl_single_table - %', SQLERRM;
END;
$$;

DO $$ BEGIN RAISE NOTICE 'DDL Generator Tests Completed'; RAISE NOTICE ''; END $$;