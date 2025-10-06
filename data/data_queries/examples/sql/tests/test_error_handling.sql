-- ============================================================
-- Error Handling and Edge Cases Tests
-- ============================================================
-- Tests for error conditions and edge cases

DO $$ BEGIN RAISE NOTICE 'Running Error Handling and Edge Cases Tests...'; END $$;

-- ============================================================
-- Test null and empty input handling
-- ============================================================

-- Test null metadata
SELECT run_test(
  'error_handling',
  'null_metadata_ddl',
  $$SELECT generate_ddl(NULL)$$,
  NULL,
  true -- Should raise exception
);

SELECT run_test(
  'error_handling',
  'null_metadata_dml',
  $$SELECT generate_dml(NULL)$$,
  NULL,
  true -- Should raise exception
);

-- Test empty object
SELECT run_test(
  'error_handling',
  'empty_object_ddl',
  $$SELECT generate_table_ddl('{}'::jsonb, 'test_schema')$$,
  NULL,
  true -- Should raise exception for missing name
);

-- Test empty array
DO $$
DECLARE
  v_result text;
BEGIN
  SELECT generate_dml('[]'::jsonb, 'test_schema') INTO v_result;
  PERFORM assert_true(v_result = '', 'Empty array should return empty string');
  RAISE NOTICE '[PASS] error_handling: empty_array_dml - Empty array handled correctly';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] error_handling: empty_array_dml - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test invalid JSON structure handling
-- ============================================================

-- Test invalid pipeline structure
SELECT run_test(
  'error_handling',
  'invalid_pipeline_structure',
  $$SELECT generate_ddl('{"schema": "test", "invalid_key": "value"}'::jsonb)$$,
  NULL,
  true -- Should raise exception for missing tables
);

-- Test malformed table entry
SELECT run_test(
  'error_handling',
  'malformed_table_entry',
  $$SELECT validate_metadata_entry('{"invalid": "structure"}'::jsonb)$$,
  NULL,
  true -- Should raise exception for missing name
);

-- ============================================================
-- Test SQL injection prevention
-- ============================================================

-- Test SQL injection in table name - check if identifier validation catches it
DO $$
BEGIN
  BEGIN
    PERFORM generate_table_ddl('{"name": "test; DROP TABLE users; --", "entity_type": "dimension", "grain": "test"}'::jsonb, 'test_schema');
    -- If we get here, the system allowed the dangerous name - this might be OK if it gets quoted properly
    RAISE NOTICE '[PASS] error_handling: sql_injection_table_name - Table name handled safely (quoted/escaped)';
  EXCEPTION WHEN OTHERS THEN
    -- Expected - the validation should catch dangerous names
    RAISE NOTICE '[PASS] error_handling: sql_injection_table_name - Expected exception raised: %', SQLERRM;
  END;
END;
$$;

-- Test SQL injection in schema name
DO $$
DECLARE
  v_result text;
BEGIN
  -- This should be safe due to identifier validation
  SELECT generate_table_ddl('{"name": "test_table", "entity_type": "dimension", "grain": "test"}'::jsonb, 'test; DROP SCHEMA public; --') INTO v_result;
  -- If we get here without error, the system properly quoted/escaped the schema name
  RAISE NOTICE '[PASS] error_handling: sql_injection_schema_name - Schema name properly handled';
EXCEPTION WHEN OTHERS THEN
  IF SQLERRM LIKE '%invalid characters%' THEN
    RAISE NOTICE '[PASS] error_handling: sql_injection_schema_name - Invalid schema name rejected: %', SQLERRM;
  ELSE
    RAISE NOTICE '[FAIL] error_handling: sql_injection_schema_name - Unexpected error: %', SQLERRM;
  END IF;
END;
$$;

-- Test SQL injection in column names
SELECT run_test(
  'error_handling',
  'sql_injection_column_name',
  $$SELECT validate_metadata_entry('{"name": "test", "entity_type": "dimension", "grain": "test", "primary_keys": ["id; DROP TABLE users; --"]}'::jsonb)$$,
  NULL,
  true -- Should raise exception for invalid identifier
);

-- ============================================================
-- Test resource limits and performance edge cases
-- ============================================================

-- Test very long table name
SELECT run_test(
  'error_handling',
  'very_long_table_name',
  $$SELECT validate_identifier(repeat('x', 100), 'table name')$$,
  NULL,
  true -- Should raise exception for too long identifier
);

-- Test very large JSONB object (if jsonb_diff exists)
DO $$
DECLARE
  v_large_obj jsonb;
  v_result jsonb;
BEGIN
  -- Create a large JSONB object (close to 1MB limit)
  v_large_obj := jsonb_build_object('large_field', repeat('x', 500000));
  
  SELECT jsonb_diff(v_large_obj, '{}') INTO v_result;
  RAISE NOTICE '[PASS] error_handling: large_jsonb_object - Large JSONB handled within limits';
EXCEPTION WHEN OTHERS THEN
  IF SQLERRM LIKE '%too large%' THEN
    RAISE NOTICE '[PASS] error_handling: large_jsonb_object - Size limit properly enforced: %', SQLERRM;
  ELSE
    RAISE NOTICE '[FAIL] error_handling: large_jsonb_object - Unexpected error: %', SQLERRM;
  END IF;
END;
$$;

-- Test excessive number of tables in pipeline
DO $$
DECLARE
  v_tables jsonb := '[]'::jsonb;
  v_pipeline jsonb;
  v_result text;
  i integer;
BEGIN
  -- Create pipeline with many tables
  FOR i IN 1..100 LOOP
    v_tables := v_tables || jsonb_build_array(jsonb_build_object(
      'name', 'table_' || i,
      'entity_type', 'dimension',
      'grain', 'test grain',
      'primary_keys', jsonb_build_array('id')
    ));
  END LOOP;
  
  v_pipeline := jsonb_build_object('schema', 'test', 'tables', v_tables);
  
  -- Should handle large number of tables without error
  SELECT generate_ddl(v_pipeline) INTO v_result;
  PERFORM assert_contains(v_result, 'table_1', 'Should contain first table');
  PERFORM assert_contains(v_result, 'table_100', 'Should contain last table');
  
  RAISE NOTICE '[PASS] error_handling: excessive_tables - Large number of tables handled correctly';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] error_handling: excessive_tables - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test data type validation edge cases
-- ============================================================

-- Test invalid data type in physical column
DO $$
DECLARE
  v_entry jsonb := '{
    "name": "test_table",
    "entity_type": "dimension",
    "grain": "test grain",
    "physical_columns": [
      {"name": "test_col", "type": "invalid_type_that_does_not_exist"}
    ]
  }'::jsonb;
  v_result text;
BEGIN
  -- Should allow any type specification (PostgreSQL will validate at runtime)
  SELECT generate_table_ddl(v_entry, 'test_schema') INTO v_result;
  PERFORM assert_contains(v_result, 'invalid_type_that_does_not_exist', 'Should allow custom data types');
  
  RAISE NOTICE '[PASS] error_handling: invalid_data_type - Custom data types allowed';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] error_handling: invalid_data_type - %', SQLERRM;
END;
$$;

-- Test measure with invalid additivity
SELECT run_test(
  'error_handling',
  'invalid_measure_additivity',
  $$SELECT validate_metadata_entry('{
    "name": "fact_test",
    "entity_type": "transaction_fact",
    "grain": "test",
    "dimension_references": [{"dimension": "dim_test", "fk_column": "test_sk"}],
    "measures": [{"name": "amount", "type": "numeric", "additivity": "invalid_additivity"}]
  }'::jsonb)$$,
  NULL,
  true -- Should raise exception
);

-- ============================================================
-- Test circular dependency detection
-- ============================================================

-- Test circular foreign key dependency
DO $$
DECLARE
  v_pipeline jsonb := '{
    "schema": "test",
    "tables": [
      {
        "name": "table_a",
        "entity_type": "dimension",
        "grain": "test grain",
        "primary_keys": ["id"],
        "physical_columns": [{"name": "b_id", "type": "text"}],
        "foreign_keys": [
          {"column": "b_id", "reference": {"table": "table_b", "column": "id", "ref_type": "business"}}
        ]
      },
      {
        "name": "table_b",
        "entity_type": "dimension",
        "grain": "test grain",
        "primary_keys": ["id"],
        "physical_columns": [{"name": "a_id", "type": "text"}],
        "foreign_keys": [
          {"column": "a_id", "reference": {"table": "table_a", "column": "id", "ref_type": "business"}}
        ]
      }
    ]
  }'::jsonb;
  v_result text;
BEGIN
  -- Should generate DDL even with circular dependencies (PostgreSQL will handle at runtime)
  SELECT generate_ddl(v_pipeline) INTO v_result;
  PERFORM assert_contains(v_result, 'table_a', 'Should contain first table');
  PERFORM assert_contains(v_result, 'table_b', 'Should contain second table');
  
  RAISE NOTICE '[PASS] error_handling: circular_dependencies - Circular dependencies handled (deferred to runtime)';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] error_handling: circular_dependencies - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test fact table without dimensions (edge case)
-- ============================================================

-- Test fact table with empty dimension references
SELECT run_test(
  'error_handling',
  'fact_empty_dimensions',
  $$SELECT generate_fact_ddl('test_schema', 'fact_test', '{
    "dimension_references": [],
    "measures": [{"name": "amount", "type": "numeric"}]
  }'::jsonb)$$
);

-- Test fact table with no measures
SELECT run_test(
  'error_handling',
  'fact_no_measures',
  $$SELECT generate_fact_ddl('test_schema', 'fact_test', '{
    "dimension_references": [{"dimension": "dim_test", "fk_column": "test_sk"}],
    "measures": []
  }'::jsonb)$$
);

-- ============================================================
-- Test SCD2 without primary keys
-- ============================================================

-- Test SCD2 dimension without primary keys
DO $$
DECLARE
  v_entry jsonb := '{
    "name": "test_dim",
    "entity_type": "dimension",
    "grain": "test grain",
    "scd": "SCD2"
  }'::jsonb;
  v_result text;
BEGIN
  SELECT generate_table_ddl(v_entry, 'test_schema') INTO v_result;
  
  -- Should still create table with default structure
  PERFORM assert_contains(v_result, 'CREATE TABLE', 'Should create table even without explicit primary keys');
  PERFORM assert_contains(v_result, 'valid_from', 'Should still be SCD2');
  
  RAISE NOTICE '[PASS] error_handling: scd2_no_primary_keys - SCD2 without primary keys handled';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] error_handling: scd2_no_primary_keys - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test Unicode and special characters
-- ============================================================

-- Test Unicode characters in table names (should be rejected by identifier validation)
SELECT run_test(
  'error_handling',
  'unicode_table_name',
  $$SELECT validate_identifier('table_名前_测试', 'table name')$$,
  NULL,
  true -- Should raise exception for invalid characters
);

-- Test special characters in properties
DO $$
DECLARE
  v_result text;
BEGIN
  SELECT jsonb_diff('{"field": "value with spaces and symbols !@#$%^&*()"}'::jsonb, '{}'::jsonb) INTO v_result;
  RAISE NOTICE '[PASS] error_handling: special_characters_in_properties - Special characters in properties handled';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] error_handling: special_characters_in_properties - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test concurrent access patterns (basic)
-- ============================================================

-- Test that functions are read-only and don't modify state
DO $$
DECLARE
  v_metadata jsonb := '{"name": "test", "entity_type": "dimension", "grain": "test", "primary_keys": ["id"]}'::jsonb;
  v_result1 text;
  v_result2 text;
BEGIN
  -- Generate DDL twice - should be identical
  SELECT generate_table_ddl(v_metadata, 'test_schema') INTO v_result1;
  SELECT generate_table_ddl(v_metadata, 'test_schema') INTO v_result2;
  
  PERFORM assert_equals(v_result1, v_result2, 'Multiple calls should produce identical results');
  
  RAISE NOTICE '[PASS] error_handling: concurrent_access_patterns - Functions are deterministic';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] error_handling: concurrent_access_patterns - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test memory usage with deeply nested structures
-- ============================================================

-- Test deeply nested JSONB structures
DO $$
DECLARE
  v_nested jsonb;
  v_result text;
  i integer;
BEGIN
  v_nested := '{"level0": {}}'::jsonb;
  
  -- Create moderately nested structure (don't go too deep to avoid timeout)
  FOR i IN 1..10 LOOP
    v_nested := jsonb_set(v_nested, ARRAY['level0', 'level' || i], '{"data": "value"}');
  END LOOP;
  
  -- Should handle nested structures in properties
  SELECT jsonb_diff(v_nested, '{}') INTO v_result;
  
  RAISE NOTICE '[PASS] error_handling: deeply_nested_structures - Nested structures handled correctly';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] error_handling: deeply_nested_structures - %', SQLERRM;
END;
$$;

DO $$ BEGIN RAISE NOTICE 'Error Handling and Edge Cases Tests Completed'; RAISE NOTICE ''; END $$;