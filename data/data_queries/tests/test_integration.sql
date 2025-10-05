-- ============================================================
-- Integration Tests
-- ============================================================
-- End-to-end tests for complete workflows

DO $$ BEGIN RAISE NOTICE 'Running Integration Tests...'; END $$;

-- ============================================================
-- Test complete dimensional model workflow
-- ============================================================

-- Test complete dimensional model generation
DO $$
DECLARE
  v_pipeline jsonb := '{
    "schema": "analytics",
    "layer": "silver",
    "defaults": {
      "scd": "SCD1",
      "index_properties_gin": true,
      "index_properties_keys": []
    },
    "tables": [
      {
        "name": "dim_customers",
        "entity_type": "dimension",
        "entity_rationale": "Stores customer attributes that change over time",
        "grain": "One row per customer per attribute version",
        "scd": "SCD2",
        "primary_keys": ["customer_id"],
        "physical_columns": [
          {"name": "account_id", "type": "text", "nullable": false},
          {"name": "region_code", "type": "text", "nullable": true}
        ],
        "foreign_keys": [
          {
            "column": "account_id",
            "reference": {"table": "dim_accounts", "column": "account_id", "ref_type": "business"}
          }
        ],
        "index_properties_keys": ["tier", "status"]
      },
      {
        "name": "dim_products",
        "entity_type": "dimension",
        "grain": "One row per product",
        "primary_keys": ["product_id"],
        "physical_columns": [
          {"name": "category_id", "type": "integer", "nullable": false}
        ]
      },
      {
        "name": "fact_orders",
        "entity_type": "transaction_fact",
        "entity_rationale": "Records each order transaction",
        "grain": "One row per order line item",
        "dimension_references": [
          {"dimension": "dim_customers", "fk_column": "customer_sk", "required": true},
          {"dimension": "dim_products", "fk_column": "product_sk", "required": true}
        ],
        "measures": [
          {"name": "quantity", "type": "integer", "additivity": "additive", "nullable": false},
          {"name": "revenue", "type": "numeric(10,2)", "additivity": "additive", "unit": "USD"},
          {"name": "discount_pct", "type": "numeric(5,2)", "additivity": "non_additive"}
        ],
        "degenerate_dimensions": ["order_number", "line_item_id"],
        "event_timestamp_column": "order_timestamp"
      }
    ]
  }'::jsonb;
  v_ddl text;
  v_dml text;
BEGIN
  -- Generate DDL
  SELECT generate_ddl(v_pipeline, 'analytics', true) INTO v_ddl;
  
  -- Generate DML  
  SELECT generate_dml(v_pipeline, 'analytics') INTO v_dml;
  
  -- Validate DDL contains all expected elements
  PERFORM assert_contains(v_ddl, 'CREATE SCHEMA IF NOT EXISTS analytics', 'Should create schema');
  PERFORM assert_contains(v_ddl, 'dim_customers', 'Should contain customer dimension');
  PERFORM assert_contains(v_ddl, 'dim_products', 'Should contain product dimension');
  PERFORM assert_contains(v_ddl, 'fact_orders', 'Should contain fact table');
  
  -- Validate dimension structure (SCD2)
  PERFORM assert_contains(v_ddl, 'dim_customers_sk bigserial PRIMARY KEY', 'Customer dimension should have surrogate key');
  PERFORM assert_contains(v_ddl, 'customer_id text NOT NULL', 'Should have business key');
  PERFORM assert_contains(v_ddl, 'account_id text NOT NULL', 'Should have physical columns');
  PERFORM assert_contains(v_ddl, 'region_code text,', 'Should have nullable physical column');
  PERFORM assert_contains(v_ddl, 'valid_from timestamptz', 'SCD2 should have validity columns');
  PERFORM assert_contains(v_ddl, 'is_current boolean', 'SCD2 should have current flag');
  PERFORM assert_contains(v_ddl, 'properties_diff jsonb', 'SCD2 should track changes');
  
  -- Validate product dimension structure (SCD1)
  PERFORM assert_contains(v_ddl, 'dim_products_sk bigserial PRIMARY KEY', 'Product dimension should have surrogate key');
  PERFORM assert_contains(v_ddl, 'category_id integer NOT NULL', 'Should have typed physical column');
  
  -- Validate fact table structure
  PERFORM assert_contains(v_ddl, 'fact_orders_sk bigserial PRIMARY KEY', 'Fact should have surrogate key');
  PERFORM assert_contains(v_ddl, 'customer_sk bigint NOT NULL', 'Should have required FK');
  PERFORM assert_contains(v_ddl, 'product_sk bigint NOT NULL', 'Should have required FK');
  PERFORM assert_contains(v_ddl, 'quantity integer NOT NULL', 'Should have typed measures');
  PERFORM assert_contains(v_ddl, 'revenue numeric(10,2)', 'Should have decimal measures');
  PERFORM assert_contains(v_ddl, 'order_number text', 'Should have degenerate dimensions');
  PERFORM assert_contains(v_ddl, 'order_timestamp timestamptz NOT NULL', 'Should have event timestamp');
  
  -- Should NOT have properties JSONB in fact table
  IF v_ddl LIKE '%fact_orders%' AND v_ddl LIKE '%properties jsonb%' THEN
    RAISE EXCEPTION 'Fact table should not have properties JSONB';
  END IF;
  
  -- Validate indexes
  PERFORM assert_contains(v_ddl, 'CREATE INDEX', 'Should create indexes');
  PERFORM assert_contains(v_ddl, 'WHERE is_current', 'Should have SCD2 partial indexes');
  
  -- Validate foreign keys
  PERFORM assert_contains(v_ddl, 'FOREIGN KEY', 'Should create foreign keys');
  PERFORM assert_contains(v_ddl, 'dim_customers_sk', 'Should reference dimension surrogate keys');
  
  -- Validate comments
  PERFORM assert_contains(v_ddl, 'COMMENT ON TABLE', 'Should add table comments');
  PERFORM assert_contains(v_ddl, 'One row per customer per attribute version', 'Should include grain');
  PERFORM assert_contains(v_ddl, 'Records each order transaction', 'Should include rationale');
  
  -- Validate DML contains all expected elements
  PERFORM assert_contains(v_dml, 'dim_customers', 'Should contain customer dimension DML');
  PERFORM assert_contains(v_dml, 'dim_products', 'Should contain product dimension DML');
  PERFORM assert_contains(v_dml, 'fact_orders', 'Should contain fact table DML');
  
  -- Validate dimension DML (SCD2 for customers, SCD1 for products)
  PERFORM assert_contains(v_dml, 'UPDATE analytics.dim_customers', 'Customer dimension should use SCD2');
  PERFORM assert_contains(v_dml, 'valid_to = now()', 'Should expire old customer records');
  PERFORM assert_contains(v_dml, 'ON CONFLICT', 'Product dimension should use SCD1');
  
  -- Validate fact DML
  PERFORM assert_contains(v_dml, 'INSERT INTO analytics.fact_orders', 'Should insert into fact table');
  PERFORM assert_contains(v_dml, 'LEFT JOIN analytics.dim_customers', 'Should join to resolve customer SK');
  PERFORM assert_contains(v_dml, 'LEFT JOIN analytics.dim_products', 'Should join to resolve product SK');
  PERFORM assert_contains(v_dml, 'dim_customers.dim_customers_sk', 'Should select customer surrogate key');
  PERFORM assert_contains(v_dml, 'dim_products.dim_products_sk', 'Should select product surrogate key');
  
  -- Facts should be insert-only
  IF v_dml LIKE '%fact_orders%' AND (v_dml LIKE '%UPDATE%' OR v_dml LIKE '%ON CONFLICT%') THEN
    RAISE EXCEPTION 'Fact tables should be insert-only';
  END IF;
  
  RAISE NOTICE '[PASS] integration: complete_dimensional_model - Complete dimensional model workflow works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] integration: complete_dimensional_model - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test metadata validation workflow
-- ============================================================

-- Test complete metadata validation
DO $$
DECLARE
  v_valid_pipeline jsonb := '{
    "schema": "test",
    "tables": [
      {
        "name": "dim_test",
        "entity_type": "dimension",
        "grain": "test grain",
        "primary_keys": ["id"]
      },
      {
        "name": "fact_test",
        "entity_type": "transaction_fact",
        "grain": "test grain",
        "dimension_references": [
          {"dimension": "dim_test", "fk_column": "test_sk", "required": true}
        ],
        "measures": [
          {"name": "amount", "type": "numeric", "additivity": "additive"}
        ]
      }
    ]
  }'::jsonb;
  v_table jsonb;
  v_valid boolean;
BEGIN
  -- Validate each table in the pipeline
  FOR v_table IN SELECT jsonb_array_elements(v_valid_pipeline->'tables') LOOP
    SELECT validate_metadata_entry(v_table) INTO v_valid;
    PERFORM assert_true(v_valid, 'All tables should pass validation');
  END LOOP;
  
  -- Test dimensional model validation if function exists
  IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'validate_dimensional_model') THEN
    DECLARE
      v_model_result jsonb;
    BEGIN
      SELECT validate_dimensional_model(v_valid_pipeline) INTO v_model_result;
      PERFORM assert_true(
        (v_model_result->>'valid')::boolean,
        'Valid dimensional model should pass cross-validation'
      );
    END;
  END IF;
  
  RAISE NOTICE '[PASS] integration: metadata_validation_workflow - Metadata validation workflow works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] integration: metadata_validation_workflow - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test defaults application workflow
-- ============================================================

-- Test that defaults are properly applied throughout the workflow
DO $$
DECLARE
  v_pipeline jsonb := '{
    "schema": "test",
    "defaults": {
      "scd": "SCD2",
      "index_properties_gin": true,
      "entity_type": "dimension"
    },
    "tables": [
      {
        "name": "test_table",
        "grain": "test grain",
        "primary_keys": ["id"]
      }
    ]
  }'::jsonb;
  v_ddl text;
  v_dml text;
BEGIN
  -- Generate DDL and DML
  SELECT generate_ddl(v_pipeline) INTO v_ddl;
  SELECT generate_dml(v_pipeline) INTO v_dml;
  
  -- Should apply SCD2 default
  PERFORM assert_contains(v_ddl, 'valid_from timestamptz', 'Should apply SCD2 default');
  PERFORM assert_contains(v_ddl, 'is_current boolean', 'Should apply SCD2 default');
  
  -- Should apply GIN index default
  PERFORM assert_contains(v_ddl, 'gin (properties)', 'Should apply GIN index default');
  
  -- Should generate SCD2 DML
  PERFORM assert_contains(v_dml, 'UPDATE', 'Should generate SCD2 DML from default');
  PERFORM assert_contains(v_dml, 'valid_to = now()', 'Should expire records in SCD2');
  
  RAISE NOTICE '[PASS] integration: defaults_application_workflow - Defaults application workflow works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] integration: defaults_application_workflow - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test foreign key relationship workflow
-- ============================================================

-- Test foreign key relationships work end-to-end
DO $$
DECLARE
  v_pipeline jsonb := '{
    "schema": "test",
    "tables": [
      {
        "name": "dim_accounts",
        "entity_type": "dimension",
        "grain": "One row per account",
        "primary_keys": ["account_id"],
        "physical_columns": [
          {"name": "account_code", "type": "text", "nullable": false}
        ]
      },
      {
        "name": "dim_customers",
        "entity_type": "dimension",
        "grain": "One row per customer",
        "primary_keys": ["customer_id"],
        "physical_columns": [
          {"name": "account_id", "type": "text", "nullable": false}
        ],
        "foreign_keys": [
          {
            "column": "account_id",
            "reference": {"table": "dim_accounts", "column": "account_id", "ref_type": "business"}
          }
        ]
      },
      {
        "name": "fact_transactions",
        "entity_type": "transaction_fact",
        "grain": "One row per transaction",
        "dimension_references": [
          {"dimension": "dim_customers", "fk_column": "customer_sk", "required": true},
          {"dimension": "dim_accounts", "fk_column": "account_sk", "required": true}
        ],
        "measures": [
          {"name": "amount", "type": "numeric(10,2)"}
        ]
      }
    ]
  }'::jsonb;
  v_ddl text;
  v_dml text;
BEGIN
  -- Generate DDL
  SELECT generate_ddl(v_pipeline) INTO v_ddl;
  
  -- Should create business key FK from customer to account
  PERFORM assert_contains(v_ddl, 'REFERENCES test.dim_accounts(account_id)', 'Should create business key FK');
  
  -- Should create surrogate key FKs from fact to dimensions
  PERFORM assert_contains(v_ddl, 'REFERENCES test.dim_customers(dim_customers_sk)', 'Should create surrogate key FK to customers');
  PERFORM assert_contains(v_ddl, 'REFERENCES test.dim_accounts(dim_accounts_sk)', 'Should create surrogate key FK to accounts');
  
  -- Generate DML
  SELECT generate_dml(v_pipeline) INTO v_dml;
  
  -- Fact DML should join to resolve surrogate keys
  PERFORM assert_contains(v_dml, 'LEFT JOIN test.dim_customers', 'Should join to customer dimension');
  PERFORM assert_contains(v_dml, 'LEFT JOIN test.dim_accounts', 'Should join to account dimension');
  
  RAISE NOTICE '[PASS] integration: foreign_key_relationship_workflow - Foreign key relationship workflow works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] integration: foreign_key_relationship_workflow - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test medallion architecture layers
-- ============================================================

-- Test that layer information is preserved and used correctly
DO $$
DECLARE
  v_pipeline jsonb := '{
    "schema": "medallion",
    "layer": "silver",
    "tables": [
      {
        "name": "dim_bronze_source",
        "entity_type": "dimension",
        "grain": "One row per source record",
        "layer": "bronze",
        "primary_keys": ["source_id"]
      },
      {
        "name": "dim_silver_customer",
        "entity_type": "dimension",
        "grain": "One row per customer",
        "layer": "silver",
        "primary_keys": ["customer_id"]
      },
      {
        "name": "fact_gold_metrics",
        "entity_type": "transaction_fact",
        "grain": "One row per aggregated metric",
        "layer": "gold",
        "dimension_references": [
          {"dimension": "dim_silver_customer", "fk_column": "customer_sk", "required": true}
        ],
        "measures": [
          {"name": "metric_value", "type": "numeric"}
        ]
      }
    ]
  }'::jsonb;
  v_ddl text;
BEGIN
  -- Should successfully generate DDL for all layers
  SELECT generate_ddl(v_pipeline) INTO v_ddl;
  
  PERFORM assert_contains(v_ddl, 'dim_bronze_source', 'Should handle bronze layer');
  PERFORM assert_contains(v_ddl, 'dim_silver_customer', 'Should handle silver layer');
  PERFORM assert_contains(v_ddl, 'fact_gold_metrics', 'Should handle gold layer');
  
  RAISE NOTICE '[PASS] integration: medallion_architecture_layers - Medallion architecture support works';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] integration: medallion_architecture_layers - %', SQLERRM;
END;
$$;

-- ============================================================
-- Test large pipeline performance
-- ============================================================

-- Test that system can handle larger pipelines efficiently
DO $$
DECLARE
  v_large_pipeline jsonb;
  v_tables jsonb := '[]'::jsonb;
  v_ddl text;
  v_dml text;
  v_start_time timestamptz;
  v_end_time timestamptz;
  v_duration_ms numeric;
  i integer;
BEGIN
  v_start_time := clock_timestamp();
  
  -- Build a pipeline with multiple tables
  FOR i IN 1..10 LOOP
    v_tables := v_tables || jsonb_build_array(jsonb_build_object(
      'name', 'dim_test_' || i,
      'entity_type', 'dimension',
      'grain', 'Test dimension ' || i,
      'primary_keys', jsonb_build_array('id_' || i)
    ));
  END LOOP;
  
  -- Add a fact table that references multiple dimensions
  v_tables := v_tables || jsonb_build_array(jsonb_build_object(
    'name', 'fact_test',
    'entity_type', 'transaction_fact',
    'grain', 'Test fact table',
    'dimension_references', jsonb_build_array(
      jsonb_build_object('dimension', 'dim_test_1', 'fk_column', 'test_1_sk', 'required', true),
      jsonb_build_object('dimension', 'dim_test_2', 'fk_column', 'test_2_sk', 'required', true)
    ),
    'measures', jsonb_build_array(
      jsonb_build_object('name', 'amount', 'type', 'numeric')
    )
  ));
  
  v_large_pipeline := jsonb_build_object(
    'schema', 'test_large',
    'tables', v_tables
  );
  
  -- Generate DDL and DML
  SELECT generate_ddl(v_large_pipeline) INTO v_ddl;
  SELECT generate_dml(v_large_pipeline) INTO v_dml;
  
  v_end_time := clock_timestamp();
  v_duration_ms := extract(epoch from (v_end_time - v_start_time)) * 1000;
  
  -- Should complete within reasonable time (< 5 seconds)
  PERFORM assert_true(v_duration_ms < 5000, 'Large pipeline should complete within 5 seconds');
  
  -- Should contain all tables
  PERFORM assert_contains(v_ddl, 'dim_test_1', 'Should contain first dimension');
  PERFORM assert_contains(v_ddl, 'dim_test_10', 'Should contain last dimension');
  PERFORM assert_contains(v_ddl, 'fact_test', 'Should contain fact table');
  
  RAISE NOTICE '[PASS] integration: large_pipeline_performance - Large pipeline completed in %.1f ms', v_duration_ms;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE '[FAIL] integration: large_pipeline_performance - %', SQLERRM;
END;
$$;

DO $$ BEGIN RAISE NOTICE 'Integration Tests Completed'; RAISE NOTICE ''; END $$;