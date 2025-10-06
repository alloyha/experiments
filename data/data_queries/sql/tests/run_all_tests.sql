-- ============================================================
-- PostgreSQL Schema Generator - Unit Test Suite
-- ============================================================
-- 
-- This file contains comprehensive unit tests for all functions
-- in the enhanced PostgreSQL schema generator system.
--
-- Usage:
--   psql -f tests/run_all_tests.sql
--
-- Test Categories:
-- 1. Helper Functions Tests
-- 2. DDL Generator Tests  
-- 3. DML Generator Tests
-- 4. Integration Tests
-- 5. Error Handling Tests
-- ============================================================

-- Create test schema and setup
CREATE SCHEMA IF NOT EXISTS test_schema;
SET search_path = test_schema, public;

-- Test results table
CREATE TABLE IF NOT EXISTS test_results (
  test_id serial PRIMARY KEY,
  test_category text NOT NULL,
  test_name text NOT NULL,
  test_status text NOT NULL CHECK (test_status IN ('PASS', 'FAIL', 'SKIP')),
  test_message text,
  execution_time_ms numeric(10,3),
  test_timestamp timestamptz DEFAULT now()
);

-- Test execution framework
CREATE OR REPLACE FUNCTION run_test(
  p_category text,
  p_test_name text,
  p_test_sql text,
  p_expected_result text DEFAULT NULL,
  p_should_raise_exception boolean DEFAULT false
) RETURNS void LANGUAGE plpgsql AS $$
DECLARE
  v_start_time timestamptz;
  v_end_time timestamptz;
  v_duration_ms numeric(10,3);
  v_result text;
  v_status text := 'PASS';
  v_message text := 'Test passed successfully';
BEGIN
  v_start_time := clock_timestamp();
  
  BEGIN
    -- Execute the test SQL
    IF p_should_raise_exception THEN
      -- Test should raise an exception
      EXECUTE p_test_sql;
      v_status := 'FAIL';
      v_message := 'Expected exception was not raised';
    ELSE
      -- Test should complete successfully
      EXECUTE p_test_sql INTO v_result;
      
      -- Check expected result if provided
      IF p_expected_result IS NOT NULL AND v_result IS DISTINCT FROM p_expected_result THEN
        v_status := 'FAIL';
        v_message := format('Expected: %s, Got: %s', p_expected_result, v_result);
      END IF;
    END IF;
    
  EXCEPTION WHEN OTHERS THEN
    IF p_should_raise_exception THEN
      v_status := 'PASS';
      v_message := format('Expected exception raised: %s', SQLERRM);
    ELSE
      v_status := 'FAIL';
      v_message := format('Unexpected exception: %s', SQLERRM);
    END IF;
  END;
  
  v_end_time := clock_timestamp();
  v_duration_ms := extract(epoch from (v_end_time - v_start_time)) * 1000;
  
  -- Record test result
  INSERT INTO test_results (test_category, test_name, test_status, test_message, execution_time_ms)
  VALUES (p_category, p_test_name, v_status, v_message, v_duration_ms);
  
  -- Print result
  RAISE NOTICE '[%] %: % - %', v_status, p_category, p_test_name, v_message;
END;
$$;

-- Assert function for more complex tests
CREATE OR REPLACE FUNCTION assert_true(
  p_condition boolean,
  p_message text DEFAULT 'Assertion failed'
) RETURNS void LANGUAGE plpgsql AS $$
BEGIN
  IF NOT p_condition THEN
    RAISE EXCEPTION '%', p_message;
  END IF;
END;
$$;

CREATE OR REPLACE FUNCTION assert_equals(
  p_actual anyelement,
  p_expected anyelement,
  p_message text DEFAULT 'Values are not equal'
) RETURNS void LANGUAGE plpgsql AS $$
BEGIN
  IF p_actual IS DISTINCT FROM p_expected THEN
    RAISE EXCEPTION '%: Expected %, got %', p_message, p_expected, p_actual;
  END IF;
END;
$$;

CREATE OR REPLACE FUNCTION assert_contains(
  p_text text,
  p_substring text,
  p_message text DEFAULT 'Text does not contain expected substring'
) RETURNS void LANGUAGE plpgsql AS $$
BEGIN
  IF p_text NOT LIKE '%' || p_substring || '%' THEN
    RAISE EXCEPTION '%: Text "%s" does not contain "%s"', p_message, p_text, p_substring;
  END IF;
END;
$$;

-- Clear previous test results
TRUNCATE test_results;

-- Run all test files
DO $$
BEGIN
  RAISE NOTICE '============================================================';
  RAISE NOTICE 'Starting PostgreSQL Schema Generator Test Suite';
  RAISE NOTICE '============================================================';
END;
$$;

-- Load all test files
\i tests/test_helper_functions.sql
\i tests/test_ddl_generator.sql
\i tests/test_dml_generator.sql
\i tests/test_integration.sql
\i tests/test_error_handling.sql

-- Display test summary
DO $$
DECLARE
  v_total_tests int;
  v_passed_tests int;
  v_failed_tests int;
  v_avg_duration numeric(10,3);
  rec record;
BEGIN
  SELECT 
    count(*),
    count(*) FILTER (WHERE test_status = 'PASS'),
    count(*) FILTER (WHERE test_status = 'FAIL'),
    avg(execution_time_ms)
  INTO v_total_tests, v_passed_tests, v_failed_tests, v_avg_duration
  FROM test_results;
  
  RAISE NOTICE '============================================================';
  RAISE NOTICE 'TEST SUMMARY';
  RAISE NOTICE '============================================================';
  RAISE NOTICE 'Total Tests: %', v_total_tests;
  RAISE NOTICE 'Passed: % (%.1f%%)', v_passed_tests, (v_passed_tests::numeric / v_total_tests * 100);
  RAISE NOTICE 'Failed: % (%.1f%%)', v_failed_tests, (v_failed_tests::numeric / v_total_tests * 100);
  RAISE NOTICE 'Average Execution Time: %.3f ms', v_avg_duration;
  RAISE NOTICE '============================================================';
  
  IF v_failed_tests > 0 THEN
    RAISE NOTICE 'FAILED TESTS:';
    FOR rec IN 
      SELECT test_category, test_name, test_message 
      FROM test_results 
      WHERE test_status = 'FAIL'
      ORDER BY test_category, test_name
    LOOP
      RAISE NOTICE '  [%] %: %', rec.test_category, rec.test_name, rec.test_message;
    END LOOP;
  END IF;
END;
$$;