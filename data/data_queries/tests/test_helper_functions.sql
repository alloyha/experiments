-- ============================================================
-- Helper Functions Unit Tests
-- ============================================================
-- Tests for all helper functions in helper_functions.sql

DO $$ BEGIN RAISE NOTICE 'Running Helper Functions Tests...'; END $$;

-- ============================================================
-- Test jsonb_diff function
-- ============================================================

-- Test basic diff
SELECT run_test(
  'helper_functions',
  'jsonb_diff_basic',
  $$SELECT jsonb_diff('{"a": 1, "b": 2}'::jsonb, '{"a": 1, "c": 3}'::jsonb)$$
);

-- Test with null inputs
SELECT run_test(
  'helper_functions',
  'jsonb_diff_null_inputs',
  $$SELECT jsonb_diff(NULL, NULL)$$
);

-- Test identical objects
SELECT run_test(
  'helper_functions',
  'jsonb_diff_identical',
  $$SELECT jsonb_diff('{"a": 1}'::jsonb, '{"a": 1}'::jsonb)$$
);

-- Test size limits
SELECT run_test(
  'helper_functions',
  'jsonb_diff_size_limit',
  $$SELECT jsonb_diff('{"key": "' || repeat('x', 2000000) || '"}'::jsonb, '{}'::jsonb)$$,
  NULL,
  true -- Should raise exception
);

-- ============================================================
-- Test validate_identifier function
-- ============================================================

-- Test valid identifier
SELECT run_test(
  'helper_functions',
  'validate_identifier_valid',
  $$SELECT validate_identifier('valid_name', 'test column')$$
);

-- Test invalid identifier - too long
SELECT run_test(
  'helper_functions',
  'validate_identifier_too_long',
  $$SELECT validate_identifier(repeat('x', 100), 'test column')$$,
  NULL,
  true -- Should raise exception
);

-- Test invalid identifier - starts with number
SELECT run_test(
  'helper_functions',
  'validate_identifier_starts_with_number',
  $$SELECT validate_identifier('1invalid', 'test column')$$,
  NULL,
  true -- Should raise exception
);

-- Test null identifier
SELECT run_test(
  'helper_functions',
  'validate_identifier_null',
  $$SELECT validate_identifier(NULL, 'test column')$$,
  NULL,
  true -- Should raise exception
);

-- ============================================================
-- Test validate_metadata_entry function
-- ============================================================

-- Test valid dimension metadata
DO $$
BEGIN
  PERFORM run_test(
    'helper_functions',
    'validate_metadata_entry_valid_dimension',
    'SELECT validate_metadata_entry(''{"name": "test_table", "entity_type": "dimension", "grain": "test grain", "primary_keys": ["id"]}''::jsonb)'
  );
END;
$$;

-- Test valid fact metadata
DO $$
DECLARE
  v_fact_metadata jsonb := '{
    "name": "fact_test",
    "entity_type": "transaction_fact", 
    "grain": "One row per transaction",
    "dimension_references": [{"dimension": "dim_test", "fk_column": "test_sk"}],
    "measures": [{"name": "amount", "type": "numeric(10,2)"}]
  }'::jsonb;
BEGIN
  BEGIN
    PERFORM validate_metadata_entry(v_fact_metadata);
    RAISE NOTICE '[PASS] helper_functions: validate_metadata_entry_valid_fact - Valid fact metadata accepted';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '[FAIL] helper_functions: validate_metadata_entry_valid_fact - Unexpected exception: %', SQLERRM;
  END;
END;
$$;

-- Test missing name
SELECT run_test(
  'helper_functions',
  'validate_metadata_entry_missing_name',
  $$SELECT validate_metadata_entry('{"entity_type": "dimension"}'::jsonb)$$,
  NULL,
  true -- Should raise exception
);

-- Test invalid entity type
SELECT run_test(
  'helper_functions',
  'validate_metadata_entry_invalid_entity_type',
  $$SELECT validate_metadata_entry('{"name": "test", "entity_type": "invalid_type"}'::jsonb)$$,
  NULL,
  true -- Should raise exception
);

-- Test fact without dimension_references
SELECT run_test(
  'helper_functions',
  'validate_metadata_entry_fact_no_dimensions',
  $$SELECT validate_metadata_entry('{"name": "test_fact", "entity_type": "transaction_fact", "grain": "test"}'::jsonb)$$,
  NULL,
  true -- Should raise exception
);

-- Test fact without measures
SELECT run_test(
  'helper_functions',
  'validate_metadata_entry_fact_no_measures',
  $$SELECT validate_metadata_entry('{
    "name": "test_fact", 
    "entity_type": "transaction_fact", 
    "grain": "test",
    "dimension_references": [{"dimension": "dim_test", "fk_column": "test_sk"}]
  }'::jsonb)$$,
  NULL,
  true -- Should raise exception
);

-- Test invalid measure additivity
SELECT run_test(
  'helper_functions',
  'validate_metadata_entry_invalid_additivity',
  $$SELECT validate_metadata_entry('{
    "name": "test_fact",
    "entity_type": "transaction_fact",
    "grain": "test",
    "dimension_references": [{"dimension": "dim_test", "fk_column": "test_sk"}],
    "measures": [{"name": "amount", "type": "numeric", "additivity": "invalid"}]
  }'::jsonb)$$,
  NULL,
  true -- Should raise exception
);

-- ============================================================
-- Test apply_defaults_to_entry function
-- ============================================================

-- Test applying defaults
DO $$
DECLARE
  v_result jsonb;
  v_entry jsonb := '{"name": "test_table"}'::jsonb;
  v_defaults jsonb := '{"scd": "SCD1", "index_properties_gin": true}'::jsonb;
BEGIN
  SELECT apply_defaults_to_entry(v_entry, v_defaults) INTO v_result;
  
  PERFORM assert_true(
    v_result ? 'scd' AND v_result->>'scd' = 'SCD1',
    'Should apply default SCD type'
  );
  
  PERFORM assert_true(
    v_result ? 'index_properties_gin' AND (v_result->>'index_properties_gin')::boolean,
    'Should apply default index_properties_gin'
  );
  
  PERFORM assert_true(
    v_result->>'name' = 'test_table',
    'Should preserve existing values'
  );
  
  RAISE NOTICE '[PASS] helper_functions: apply_defaults_to_entry_basic - Defaults applied correctly';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] helper_functions: apply_defaults_to_entry_basic - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test retry_with_backoff function (if exists)
-- ============================================================

-- Test successful execution
SELECT run_test(
  'helper_functions',
  'retry_with_backoff_success',
  $$SELECT retry_with_backoff('SELECT 1', 3, 10)$$
);

-- ============================================================
-- Test validate_grain function (if exists)
-- ============================================================

-- Note: This would require actual tables to exist, so we'll test the function exists
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'validate_grain') THEN
    RAISE NOTICE '[PASS] helper_functions: validate_grain_function_exists - Function exists';
  ELSE
    RAISE NOTICE '[SKIP] helper_functions: validate_grain_function_exists - Function not found';
  END IF;
END;
$$;

-- ============================================================
-- Test validate_dimensional_model function (if exists)
-- ============================================================

-- Test valid dimensional model
DO $$
DECLARE
  v_pipeline jsonb := '{
    "schema": "test_schema",
    "entities": [
      {
        "name": "dim_customers",
        "entity_type": "dimension",
        "grain": "One row per customer"
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
  v_result jsonb;
BEGIN
  IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'validate_dimensional_model') THEN
    SELECT validate_dimensional_model(v_pipeline) INTO v_result;
    
    PERFORM assert_true(
      (v_result->>'valid')::boolean,
      'Valid dimensional model should pass validation'
    );
    
    RAISE NOTICE '[PASS] helper_functions: validate_dimensional_model_valid - Valid model passed';
  ELSE
    RAISE NOTICE '[SKIP] helper_functions: validate_dimensional_model_valid - Function not found';
  END IF;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] helper_functions: validate_dimensional_model_valid - %', SQLERRM;
END;
$$;

-- Test model with missing dimension reference
DO $$
DECLARE
  v_pipeline jsonb := '{
    "schema": "test_schema",
    "entities": [
      {
        "name": "fact_orders",
        "entity_type": "transaction_fact",
        "grain": "One row per order",
        "dimension_references": [
          {"dimension": "dim_missing", "fk_column": "missing_sk", "required": true}
        ],
        "measures": [
          {"name": "amount", "type": "numeric(10,2)"}
        ]
      }
    ]
  }'::jsonb;
  v_result jsonb;
BEGIN
  IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'validate_dimensional_model') THEN
    SELECT validate_dimensional_model(v_pipeline) INTO v_result;
    
    PERFORM assert_true(
      NOT (v_result->>'valid')::boolean,
      'Invalid dimensional model should fail validation'
    );
    
    RAISE NOTICE '[PASS] helper_functions: validate_dimensional_model_invalid - Invalid model failed as expected';
  ELSE
    RAISE NOTICE '[SKIP] helper_functions: validate_dimensional_model_invalid - Function not found';
  END IF;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] helper_functions: validate_dimensional_model_invalid - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test validate_batch function (if exists)
-- ============================================================

DO $$
DECLARE
  v_test_rows jsonb := '[
    {"id": "1", "name": "test1"},
    {"id": "", "name": "test2"},
    {"name": "test3"}
  ]'::jsonb;
  v_result jsonb;
BEGIN
  IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'validate_batch') THEN
    SELECT validate_batch('test_schema', 'test_table', v_test_rows) INTO v_result;
    
    PERFORM assert_true(
      (v_result->>'total_rows')::int = 3,
      'Should count total rows correctly'
    );
    
    PERFORM assert_true(
      (v_result->>'invalid_rows')::int > 0,
      'Should detect invalid rows'
    );
    
    RAISE NOTICE '[PASS] helper_functions: validate_batch_basic - Batch validation working';
  ELSE
    RAISE NOTICE '[SKIP] helper_functions: validate_batch_basic - Function not found';
  END IF;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] helper_functions: validate_batch_basic - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test lock_key_from_text function
-- ============================================================

-- Test basic functionality
DO $$
DECLARE
  v_result1 bigint;
  v_result2 bigint;
BEGIN
  SELECT lock_key_from_text('test_key') INTO v_result1;
  SELECT lock_key_from_text('test_key') INTO v_result2;
  
  PERFORM assert_equals(v_result1, v_result2, 'Same input should produce same hash');
  
  RAISE NOTICE '[PASS] helper_functions: lock_key_from_text_deterministic - Function is deterministic';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] helper_functions: lock_key_from_text_deterministic - %', SQLERRM;
END;
$$;

-- Test with null input
DO $$
DECLARE
  v_result bigint;
BEGIN
  SELECT lock_key_from_text(NULL) INTO v_result;
  
  PERFORM assert_true(v_result IS NOT NULL, 'Should handle null input');
  
  RAISE NOTICE '[PASS] helper_functions: lock_key_from_text_null - Handles null input';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] helper_functions: lock_key_from_text_null - %', SQLERRM;
END;
$$;

DO $$ BEGIN RAISE NOTICE 'Helper Functions Tests Completed'; RAISE NOTICE ''; END $$;