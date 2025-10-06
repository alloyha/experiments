-- ============================================================
-- DML Generator Unit Tests
-- ============================================================
-- Tests for all DML generation functions in dml_generator.sql

DO $$ BEGIN RAISE NOTICE 'Running DML Generator Tests...'; END $$;

-- ============================================================
-- Test generate_scd1_upsert_dml function
-- ============================================================

-- Test basic SCD1 upsert
DO $$
DECLARE
  v_primary_keys jsonb := '["customer_id", "product_id"]'::jsonb;
  v_result text;
BEGIN
  SELECT generate_scd1_upsert_dml('test_schema', 'test_table', v_primary_keys) INTO v_result;
  
  PERFORM assert_contains(v_result, 'jsonb_to_recordset(:rows)', 'Should use parameter binding');
  PERFORM assert_contains(v_result, 'INSERT INTO test_schema.test_table', 'Should insert into correct table');
  PERFORM assert_contains(v_result, 'ON CONFLICT', 'Should handle conflicts');
  PERFORM assert_contains(v_result, 'customer_id text', 'Should define record structure');
  PERFORM assert_contains(v_result, 'product_id text', 'Should include all primary keys');
  PERFORM assert_contains(v_result, 'row_hash', 'Should include row_hash');
  PERFORM assert_contains(v_result, 'IS DISTINCT FROM', 'Should check for changes');
  
  RAISE NOTICE '[PASS] dml_generator: generate_scd1_upsert_dml_basic - SCD1 upsert generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] dml_generator: generate_scd1_upsert_dml_basic - %', SQLERRM;
END;
$$;

-- Test invalid primary keys
SELECT run_test(
  'dml_generator',
  'generate_scd1_upsert_dml_invalid_keys',
  $$SELECT generate_scd1_upsert_dml('test_schema', 'test_table', '{"not": "array"}'::jsonb)$$,
  NULL,
  true -- Should raise exception
);

-- ============================================================
-- Test generate_scd2_process_dml function
-- ============================================================

-- Test basic SCD2 process
DO $$
DECLARE
  v_primary_keys jsonb := '["customer_id"]'::jsonb;
  v_result text;
BEGIN
  SELECT generate_scd2_process_dml('test_schema', 'test_table', v_primary_keys) INTO v_result;
  
  PERFORM assert_contains(v_result, 'jsonb_to_recordset(:rows)', 'Should use parameter binding');
  PERFORM assert_contains(v_result, 'UPDATE test_schema.test_table', 'Should update existing records');
  PERFORM assert_contains(v_result, 'INSERT INTO test_schema.test_table', 'Should insert new versions');
  PERFORM assert_contains(v_result, 'valid_to = now()', 'Should expire old records');
  PERFORM assert_contains(v_result, 'is_current = false', 'Should mark as not current');
  PERFORM assert_contains(v_result, 'jsonb_diff', 'Should calculate differences');
  PERFORM assert_contains(v_result, 'properties_diff', 'Should include properties_diff');
  
  RAISE NOTICE '[PASS] dml_generator: generate_scd2_process_dml_basic - SCD2 process generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] dml_generator: generate_scd2_process_dml_basic - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test generate_snapshot_reconcile_dml function
-- ============================================================

-- Test snapshot reconciliation
DO $$
DECLARE
  v_primary_keys jsonb := '["id"]'::jsonb;
  v_result text;
BEGIN
  SELECT generate_snapshot_reconcile_dml('test_schema', 'test_table', v_primary_keys) INTO v_result;
  
  PERFORM assert_contains(v_result, 'jsonb_to_recordset(:rows)', 'Should use parameter binding');
  PERFORM assert_contains(v_result, ':snapshot_ts', 'Should use snapshot timestamp parameter');
  PERFORM assert_contains(v_result, 'to_insert', 'Should identify inserts');
  PERFORM assert_contains(v_result, 'to_update', 'Should identify updates');
  PERFORM assert_contains(v_result, 'to_expire', 'Should identify deletions');
  PERFORM assert_contains(v_result, 'UNION ALL', 'Should combine inserts and updates');
  
  RAISE NOTICE '[PASS] dml_generator: generate_snapshot_reconcile_dml_basic - Snapshot reconcile generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] dml_generator: generate_snapshot_reconcile_dml_basic - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test generate_fact_insert_dml function
-- ============================================================

-- Test fact insert DML
DO $$
DECLARE
  v_entry jsonb := '{
    "dimension_references": [
      {"dimension": "dim_customers", "fk_column": "customer_sk", "required": true, "business_key": "customer_id"},
      {"dimension": "dim_products", "fk_column": "product_sk", "required": false, "business_key": "product_id"}
    ],
    "measures": [
      {"name": "quantity", "type": "integer"},
      {"name": "revenue", "type": "numeric(10,2)"}
    ],
    "degenerate_dimensions": ["order_number"],
    "event_timestamp_column": "order_timestamp"
  }'::jsonb;
  v_result text;
BEGIN
  SELECT generate_fact_insert_dml('test_schema', 'fact_orders', v_entry) INTO v_result;
  
  PERFORM assert_contains(v_result, 'jsonb_to_recordset(:rows)', 'Should use parameter binding');
  PERFORM assert_contains(v_result, 'INSERT INTO test_schema.fact_orders', 'Should insert into fact table');
  PERFORM assert_contains(v_result, 'LEFT JOIN test_schema.dim_customers', 'Should join dimensions');
  PERFORM assert_contains(v_result, 'dim_customers_sk', 'Should select surrogate keys');
  PERFORM assert_contains(v_result, 'customer_id text', 'Should define business key columns');
  PERFORM assert_contains(v_result, 'quantity integer', 'Should define measure columns');
  PERFORM assert_contains(v_result, 'order_number text', 'Should include degenerate dimensions');
  PERFORM assert_contains(v_result, 'order_timestamp timestamptz', 'Should include event timestamp');
  PERFORM assert_contains(v_result, 'is_current', 'Should filter for current records');
  PERFORM assert_contains(v_result, 'IS NOT NULL', 'Should validate required dimensions');
  
  RAISE NOTICE '[PASS] dml_generator: generate_fact_insert_dml_basic - Fact insert DML generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] dml_generator: generate_fact_insert_dml_basic - %', SQLERRM;
END;
$$;

-- Test fact insert with default business keys
DO $$
DECLARE
  v_entry jsonb := '{
    "dimension_references": [
      {"dimension": "dim_customers", "fk_column": "customer_sk", "required": true}
    ],
    "measures": [
      {"name": "amount", "type": "numeric(10,2)"}
    ]
  }'::jsonb;
  v_result text;
BEGIN
  SELECT generate_fact_insert_dml('test_schema', 'fact_test', v_entry) INTO v_result;
  
  -- Should default to dimension name + '_id' for business key
  PERFORM assert_contains(v_result, 'customers_id text', 'Should default business key name');
  
  RAISE NOTICE '[PASS] dml_generator: generate_fact_insert_dml_default_keys - Default business key naming works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] dml_generator: generate_fact_insert_dml_default_keys - %', SQLERRM;
END;
$$;

-- Test fact insert validation - missing dimension_references
SELECT run_test(
  'dml_generator',
  'generate_fact_insert_dml_missing_dimensions',
  $$SELECT generate_fact_insert_dml('test_schema', 'fact_test', '{"measures": [{"name": "amount", "type": "numeric"}]}'::jsonb)$$,
  NULL,
  true -- Should raise exception
);

-- Test fact insert validation - missing measures
SELECT run_test(
  'dml_generator',
  'generate_fact_insert_dml_missing_measures',
  $$SELECT generate_fact_insert_dml('test_schema', 'fact_test', '{"dimension_references": [{"dimension": "dim_test", "fk_column": "test_sk"}]}'::jsonb)$$,
  NULL,
  true -- Should raise exception
);

-- ============================================================
-- Test generate_dml function (main dispatcher)
-- ============================================================

-- Test dimension DML generation (SCD1)
DO $$
DECLARE
  v_metadata jsonb := '{
    "name": "dim_customers",
    "entity_type": "dimension",
    "scd": "SCD1",
    "primary_keys": ["customer_id"]
  }'::jsonb;
  v_result text;
BEGIN
  SELECT generate_dml(v_metadata, 'test_schema') INTO v_result;
  
  PERFORM assert_contains(v_result, 'DML templates for test_schema.dim_customers', 'Should include header');
  PERFORM assert_contains(v_result, 'ON CONFLICT', 'Should generate SCD1 upsert');
  PERFORM assert_contains(v_result, 'customer_id', 'Should include primary keys');
  
  RAISE NOTICE '[PASS] dml_generator: generate_dml_dimension_scd1 - Dimension SCD1 DML generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] dml_generator: generate_dml_dimension_scd1 - %', SQLERRM;
END;
$$;

-- Test dimension DML generation (SCD2)
DO $$
DECLARE
  v_metadata jsonb := '{
    "name": "dim_customers",
    "entity_type": "dimension",
    "scd": "SCD2",
    "primary_keys": ["customer_id"]
  }'::jsonb;
  v_result text;
BEGIN
  SELECT generate_dml(v_metadata, 'test_schema') INTO v_result;
  
  PERFORM assert_contains(v_result, 'DML templates for test_schema.dim_customers', 'Should include header');
  PERFORM assert_contains(v_result, 'UPDATE', 'Should generate SCD2 process');
  PERFORM assert_contains(v_result, 'INSERT', 'Should generate SCD2 insert');
  PERFORM assert_contains(v_result, 'valid_to = now()', 'Should expire old records');
  
  RAISE NOTICE '[PASS] dml_generator: generate_dml_dimension_scd2 - Dimension SCD2 DML generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] dml_generator: generate_dml_dimension_scd2 - %', SQLERRM;
END;
$$;

-- Test fact DML generation
DO $$
DECLARE
  v_metadata jsonb := '{
    "name": "fact_orders",
    "entity_type": "transaction_fact",
    "dimension_references": [
      {"dimension": "dim_customers", "fk_column": "customer_sk", "required": true}
    ],
    "measures": [
      {"name": "amount", "type": "numeric(10,2)"}
    ]
  }'::jsonb;
  v_result text;
BEGIN
  SELECT generate_dml(v_metadata, 'test_schema') INTO v_result;
  
  PERFORM assert_contains(v_result, 'DML templates for test_schema.fact_orders', 'Should include header');
  PERFORM assert_contains(v_result, 'INSERT INTO test_schema.fact_orders', 'Should generate fact insert');
  PERFORM assert_contains(v_result, 'LEFT JOIN', 'Should join dimensions');
  PERFORM assert_contains(v_result, 'customer_sk', 'Should include dimension FKs');
  
  -- Should NOT contain UPDATE or UPSERT for facts
  IF v_result LIKE '%ON CONFLICT%' OR v_result LIKE '%UPDATE%' THEN
    RAISE EXCEPTION 'Facts should be insert-only, no UPDATE or UPSERT';
  END IF;
  
  RAISE NOTICE '[PASS] dml_generator: generate_dml_fact - Fact DML generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] dml_generator: generate_dml_fact - %', SQLERRM;
END;
$$;

-- Test pipeline DML generation
DO $$
DECLARE
  v_pipeline jsonb := '{
    "schema": "analytics",
    "defaults": {
      "scd": "SCD1"
    },
    "tables": [
      {
        "name": "dim_customers",
        "entity_type": "dimension",
        "primary_keys": ["customer_id"]
      },
      {
        "name": "fact_orders",
        "entity_type": "transaction_fact",
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
  SELECT generate_dml(v_pipeline, 'analytics') INTO v_result;
  
  PERFORM assert_contains(v_result, 'dim_customers', 'Should include dimension DML');
  PERFORM assert_contains(v_result, 'fact_orders', 'Should include fact DML');
  PERFORM assert_contains(v_result, 'ON CONFLICT', 'Should use default SCD1 for dimension');
  PERFORM assert_contains(v_result, 'LEFT JOIN', 'Should generate fact insert with joins');
  
  RAISE NOTICE '[PASS] dml_generator: generate_dml_pipeline - Pipeline DML generation works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] dml_generator: generate_dml_pipeline - %', SQLERRM;
END;
$$;

-- Test unknown entity type
SELECT run_test(
  'dml_generator',
  'generate_dml_unknown_entity_type',
  $$SELECT generate_dml('{"name": "test", "entity_type": "unknown", "primary_keys": ["id"]}'::jsonb, 'test_schema')$$,
  NULL,
  true -- Should raise exception
);

-- Test array metadata format
DO $$
DECLARE
  v_metadata jsonb := '[
    {
      "name": "table1",
      "entity_type": "dimension",
      "primary_keys": ["id"]
    },
    {
      "name": "fact1",
      "entity_type": "transaction_fact",
      "dimension_references": [{"dimension": "table1", "fk_column": "table1_sk"}],
      "measures": [{"name": "amount", "type": "numeric"}]
    }
  ]'::jsonb;
  v_result text;
BEGIN
  SELECT generate_dml(v_metadata, 'test_schema') INTO v_result;
  
  PERFORM assert_contains(v_result, 'table1', 'Should process array format');
  PERFORM assert_contains(v_result, 'fact1', 'Should process multiple tables');
  
  RAISE NOTICE '[PASS] dml_generator: generate_dml_array_format - Array format processing works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] dml_generator: generate_dml_array_format - %', SQLERRM;
END;
$$;

-- Test set semantics for SCD2
DO $$
DECLARE
  v_metadata jsonb := '{
    "name": "dim_test",
    "entity_type": "dimension",
    "scd": "SCD2",
    "set_semantics": true,
    "primary_keys": ["id"]
  }'::jsonb;
  v_result text;
BEGIN
  SELECT generate_dml(v_metadata, 'test_schema') INTO v_result;
  
  PERFORM assert_contains(v_result, 'snapshot_ts', 'Should use snapshot reconcile for set semantics');
  PERFORM assert_contains(v_result, 'to_expire', 'Should handle deletions');
  
  RAISE NOTICE '[PASS] dml_generator: generate_dml_set_semantics - Set semantics processing works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] dml_generator: generate_dml_set_semantics - %', SQLERRM;
END;
$$;

DO $$ BEGIN RAISE NOTICE 'DML Generator Tests Completed'; RAISE NOTICE ''; END $$;