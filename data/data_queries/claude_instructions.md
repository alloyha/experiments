# Code Agent Instructions: PostgreSQL Schema Generator Maintenance

## System Overview

You are maintaining a PostgreSQL-based metadata-driven schema generation system with three core SQL files:

1. **`ddl_generator.sql`** - DDL generation (CREATE TABLE, indexes, constraints)
2. **`dml_generator.sql`** - DML generation (INSERT, UPDATE, UPSERT templates)
3. **`helper_functions.sql`** - Utilities (validation, diff, monitoring)

## Architecture Principles

### Core Pattern
- Metadata (JSON) → Functions → Generated SQL (DDL/DML)
- Entities can be dimensions (with history tracking) or facts (with typed measures)
- Tables have surrogate keys + business keys/grain (physical columns) + flexible attributes (JSONB)

### Entity Types

**Dimensions:** Store descriptive attributes about business entities
```sql
CREATE TABLE {schema}.{dim_table} (
  {table}_sk bigserial PRIMARY KEY,           -- Surrogate key
  {business_key_1} text NOT NULL,             -- Physical business key
  {physical_column_1} text,                   -- Additional typed columns (for FKs, performance)
  properties jsonb DEFAULT '{}'::jsonb,        -- Flexible attributes
  row_hash text NOT NULL,
  valid_from timestamptz,                     -- SCD2 only
  valid_to timestamptz,                       -- SCD2 only
  is_current boolean,                         -- SCD2 only
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);
```

**Facts:** Store measurements/events with foreign keys to dimensions
```sql
CREATE TABLE {schema}.{fact_table} (
  {table}_sk bigserial PRIMARY KEY,
  {dimension_1}_sk bigint NOT NULL,           -- FK to dimension surrogate key
  {dimension_2}_sk bigint NOT NULL,
  {measure_1} numeric,                        -- Typed measure columns (not JSONB)
  {measure_2} integer,
  {degenerate_dim} text,                      -- Transaction IDs, etc.
  event_timestamp timestamptz NOT NULL,
  created_at timestamptz DEFAULT now()
);
```

### Enhanced Metadata Schema
```json
{
  "schema": "analytics",
  "layer": "silver",  // bronze | silver | gold (medallion architecture)
  
  "defaults": {
    "scd": "SCD1",
    "index_properties_gin": true,
    "index_properties_keys": []
  },
  
  "entities": [
    {
      "name": "dim_customers",
      "entity_type": "dimension",
      "entity_rationale": "Stores customer attributes that change over time",
      "grain": "One row per customer per attribute version",
      "scd": "SCD2",
      "layer": "silver",
      
      "primary_keys": ["customer_id"],
      "physical_columns": [
        {"name": "account_id", "type": "text", "nullable": false},
        {"name": "region_code", "type": "text", "nullable": true}
      ],
      "required_properties": ["email", "tier"],
      "index_columns": ["email"],
      "index_properties_keys": ["tier", "status"],
      
      "foreign_keys": [
        {
          "column": "account_id",
          "reference": {"table": "dim_accounts", "column": "id", "ref_type": "business"}
        }
      ],
      
      "contract_version": "1.0.0",
      "sla": {
        "freshness_minutes": 60,
        "completeness_pct": 99.5,
        "row_count_expected_range": [1000, 50000]
      }
    },
    {
      "name": "fact_orders",
      "entity_type": "transaction_fact",
      "entity_rationale": "Records each order transaction",
      "grain": "One row per order line item",
      "layer": "gold",
      
      "dimension_references": [
        {"dimension": "dim_customers", "fk_column": "customer_sk", "required": true},
        {"dimension": "dim_products", "fk_column": "product_sk", "required": true},
        {"dimension": "date_dim", "fk_column": "order_date_sk", "role": "order_date", "required": true}
      ],
      "measures": [
        {"name": "quantity", "type": "integer", "additivity": "additive", "nullable": false},
        {"name": "revenue", "type": "numeric(10,2)", "additivity": "additive", "unit": "USD"},
        {"name": "discount_pct", "type": "numeric(5,2)", "additivity": "non_additive"}
      ],
      "degenerate_dimensions": ["order_number", "line_item_id"],
      "event_timestamp_column": "order_timestamp",
      
      "sla": {
        "latency_minutes": 15,
        "completeness_pct": 99.9
      }
    }
  ]
}
```

## File Responsibilities

### `ddl_generator.sql`

**Purpose:** Generate CREATE TABLE statements, indexes, constraints

**Key Functions:**
- `generate_scd1_ddl()` - SCD1 dimension generator
- `generate_scd2_ddl()` - SCD2 dimension generator  
- `generate_fact_ddl()` - **NEW** Fact table generator with typed measures
- `generate_table_ddl()` - Dispatcher based on entity_type
- `generate_ddl()` - Main entry point, processes pipeline metadata
- `generate_foreign_keys_sql()` - FK constraint generation
- `generate_indexes_sql()` - Index generation (B-tree, GIN, expression)

**Critical Rules:**
- All DDL must be idempotent (IF NOT EXISTS, IF EXISTS checks)
- Surrogate key name: `{table_name}_sk` (not generic `sk`)
- **Entity type determines structure:** dimensions get SCD columns, facts get measure columns
- **Grain must be declared** in metadata and documented in table comments
- Business keys from `primary_keys` array become physical columns
- **Physical columns:** Typed columns outside JSONB for performance and foreign keys
- Foreign keys can reference business keys (`ref_type: "business"`) or surrogate keys (`ref_type: "surrogate"`)
- Expression indexes for JSONB: `((properties->>'key_name'))`
- SCD2 partial indexes: `WHERE is_current` for current records only
- Facts use typed measure columns, not JSONB

### `dml_generator.sql`

**Purpose:** Generate parameterized INSERT/UPDATE/UPSERT templates

**Key Functions:**
- `generate_scd1_upsert_dml()` - Upsert with conflict handling
- `generate_scd2_process_dml()` - Expire old versions + insert new
- `generate_snapshot_reconcile_dml()` - Set-based snapshot reconciliation
- `generate_fact_insert_dml()` - **NEW** Fact table insert with surrogate key resolution
- `generate_dml()` - Main entry point

**Critical Rules:**
- Templates use `:rows` parameter (jsonb_to_recordset)
- SCD1: `ON CONFLICT DO UPDATE` with `row_hash` comparison
- SCD2: Two-phase (UPDATE to expire, then INSERT new versions)
- **Facts are insert-only (immutable events)** - no UPDATE or UPSERT
- Change detection via `row_hash` (md5 or sha256 of sorted attributes)
- `properties_diff` calculated via `jsonb_diff()` helper function
- Fact DML must resolve dimension surrogate keys via JOIN on business keys

### `helper_functions.sql`

**Purpose:** Reusable utilities

**Key Functions:**
- `jsonb_diff(new, old)` - Compute property changes
- `validate_identifier()` - SQL identifier validation
- `validate_metadata_entry()` - **ENHANCED** Metadata schema validation (entity_type, grain, measures)
- `retry_with_backoff()` - Deadlock retry logic
- `execute_with_monitoring()` - Query timing/logging
- `apply_defaults_to_entry()` - Merge pipeline defaults
- `validate_grain()` - **NEW** Ensure grain columns exist and are indexed
- `validate_dimensional_model()` - **NEW** Cross-entity validation (FK integrity, conformed dimensions)

## Critical Modifications Needed

### 1. Add Physical Columns Support

**Problem:** Foreign keys don't work because referenced columns are in JSONB

**Solution:** Extend `generate_scd1_ddl()` and `generate_scd2_ddl()`

```sql
-- After business key columns, before properties:
IF p_entry ? 'physical_columns' AND jsonb_typeof(p_entry->'physical_columns') = 'array' THEN
  FOR v_col IN SELECT jsonb_array_elements(p_entry->'physical_columns') LOOP
    v_col_name := v_col->>'name';
    v_col_type := COALESCE(v_col->>'type', 'text');
    v_col_nullable := COALESCE((v_col->>'nullable')::boolean, true);
    
    v_sql := v_sql || format(' %I %s%s,', 
      v_col_name, 
      v_col_type,
      CASE WHEN NOT v_col_nullable THEN ' NOT NULL' ELSE '' END
    );
  END LOOP;
END IF;
```

**Update `generate_foreign_keys_sql()` to handle surrogate key references:**
```sql
v_ref_type := COALESCE(v_fk->'reference'->>'ref_type', 'business');

IF v_ref_type = 'surrogate' THEN
  -- Reference the surrogate key column
  v_ref_column := v_fk->'reference'->>'table' || '_sk';
ELSE
  -- Reference the business key column
  v_ref_column := v_fk->'reference'->>'column';
END IF;
```

### 2. Add Fact Table Generator

**New function in `ddl_generator.sql`:**
```sql
CREATE OR REPLACE FUNCTION generate_fact_ddl(
  p_schema_name text,
  p_table_name text,
  p_entry jsonb
) RETURNS text LANGUAGE plpgsql AS $$
DECLARE
  v_sql text := '';
  v_dim_ref jsonb;
  v_measure jsonb;
  v_degenerate text;
  v_event_ts_col text;
BEGIN
  -- Validate metadata
  IF NOT (p_entry ? 'dimension_references') THEN
    RAISE EXCEPTION 'Fact table % must have dimension_references array', p_table_name;
  END IF;
  
  IF NOT (p_entry ? 'measures') THEN
    RAISE EXCEPTION 'Fact table % must have measures array', p_table_name;
  END IF;
  
  -- Start table creation
  v_sql := format('CREATE TABLE IF NOT EXISTS %I.%I (', p_schema_name, p_table_name);
  v_sql := v_sql || format('%I bigserial PRIMARY KEY,', p_table_name || '_sk');
  
  -- Add dimension foreign keys
  FOR v_dim_ref IN SELECT jsonb_array_elements(p_entry->'dimension_references') LOOP
    v_sql := v_sql || format(' %I bigint%s,', 
      v_dim_ref->>'fk_column',
      CASE WHEN COALESCE((v_dim_ref->>'required')::boolean, false) 
           THEN ' NOT NULL' 
           ELSE '' END
    );
  END LOOP;
  
  -- Add degenerate dimensions
  IF p_entry ? 'degenerate_dimensions' THEN
    FOR v_degenerate IN SELECT jsonb_array_elements_text(p_entry->'degenerate_dimensions') LOOP
      v_sql := v_sql || format(' %I text,', v_degenerate);
    END LOOP;
  END IF;
  
  -- Add typed measure columns (NOT JSONB)
  FOR v_measure IN SELECT jsonb_array_elements(p_entry->'measures') LOOP
    v_sql := v_sql || format(' %I %s%s,', 
      v_measure->>'name',
      v_measure->>'type',
      CASE WHEN COALESCE((v_measure->>'nullable')::boolean, true) 
           THEN '' 
           ELSE ' NOT NULL' END
    );
  END LOOP;
  
  -- Add event timestamp
  v_event_ts_col := COALESCE(p_entry->>'event_timestamp_column', 'event_timestamp');
  v_sql := v_sql || format(' %I timestamptz NOT NULL,', v_event_ts_col);
  v_sql := v_sql || ' created_at timestamptz DEFAULT now()';
  v_sql := v_sql || ');';
  
  -- Add indexes
  v_sql := v_sql || E'\n' || format(
    'CREATE INDEX IF NOT EXISTS ix_%s_event_ts ON %I.%I (%I);',
    lower(regexp_replace(p_table_name, '\W', '_', 'g')),
    p_schema_name, p_table_name, v_event_ts_col
  );
  
  -- Add dimension FK indexes
  FOR v_dim_ref IN SELECT jsonb_array_elements(p_entry->'dimension_references') LOOP
    v_sql := v_sql || E'\n' || format(
      'CREATE INDEX IF NOT EXISTS ix_%s_%s ON %I.%I (%I);',
      lower(regexp_replace(p_table_name, '\W', '_', 'g')),
      lower(regexp_replace(v_dim_ref->>'fk_column', '\W', '_', 'g')),
      p_schema_name, p_table_name, v_dim_ref->>'fk_column'
    );
  END LOOP;
  
  -- Add foreign key constraints
  FOR v_dim_ref IN SELECT jsonb_array_elements(p_entry->'dimension_references') LOOP
    v_sql := v_sql || E'\n' || format(
      'ALTER TABLE %I.%I ADD CONSTRAINT fk_%s_%s FOREIGN KEY (%I) REFERENCES %I.%I(%s);',
      p_schema_name, p_table_name,
      lower(regexp_replace(p_table_name, '\W', '_', 'g')),
      lower(regexp_replace(v_dim_ref->>'dimension', '\W', '_', 'g')),
      v_dim_ref->>'fk_column',
      p_schema_name, v_dim_ref->>'dimension',
      v_dim_ref->>'dimension' || '_sk'  -- Always reference surrogate key
    );
  END LOOP;
  
  RETURN v_sql;
END;
$$;
```

**Update `generate_table_ddl()` to dispatch by entity_type:**
```sql
CREATE OR REPLACE FUNCTION generate_table_ddl(
  p_entry jsonb,
  p_default_schema text,
  p_include_schema_creation boolean DEFAULT false
) RETURNS text LANGUAGE plpgsql AS $$
DECLARE
  v_name text := p_entry->>'name';
  v_schema text := COALESCE(p_entry->>'schema', p_default_schema);
  v_entity_type text := COALESCE(p_entry->>'entity_type', 'dimension');
  v_scd text := COALESCE(p_entry->>'scd','SCD1');
  v_sql text := '';
  v_default_primary_keys jsonb := jsonb_build_array('id');
BEGIN
  IF v_name IS NULL OR trim(v_name) = '' THEN
    RAISE EXCEPTION 'Entry must have a non-empty "name"';
  END IF;
  
  -- Validate grain is declared
  IF NOT (p_entry ? 'grain') THEN
    RAISE WARNING 'Entity % has no grain declaration. This may cause modeling errors.', v_name;
  END IF;

  IF v_entity_type = 'dimension' THEN
    IF upper(v_scd) = 'SCD1' THEN
      v_sql := generate_scd1_ddl(v_schema, v_name, p_entry, v_default_primary_keys, p_include_schema_creation);
    ELSIF upper(v_scd) = 'SCD2' THEN
      v_sql := generate_scd2_ddl(v_schema, v_name, p_entry, v_default_primary_keys, p_include_schema_creation);
    ELSE
      v_sql := generate_scd1_ddl(v_schema, v_name, p_entry, v_default_primary_keys, p_include_schema_creation);
    END IF;
    
  ELSIF v_entity_type = 'transaction_fact' OR v_entity_type = 'fact' THEN
    v_sql := generate_fact_ddl(v_schema, v_name, p_entry);
    
  ELSE
    RAISE EXCEPTION 'Unknown entity_type: %. Must be "dimension" or "transaction_fact"', v_entity_type;
  END IF;
  
  -- Add grain as table comment
  IF p_entry ? 'grain' THEN
    v_sql := v_sql || E'\n' || format(
      'COMMENT ON TABLE %I.%I IS %L;',
      v_schema, v_name, 
      'Grain: ' || (p_entry->>'grain') || 
      CASE WHEN p_entry ? 'entity_rationale' 
           THEN E'\n' || (p_entry->>'entity_rationale') 
           ELSE '' END
    );
  END IF;

  RETURN v_sql;
END;
$$;
```

### 3. Add Fact Insert DML Generator

**New function in `dml_generator.sql`:**
```sql
CREATE OR REPLACE FUNCTION generate_fact_insert_dml(
  p_schema_name text,
  p_table_name text,
  p_entry jsonb
) RETURNS text LANGUAGE plpgsql AS $$
DECLARE
  v_dim_ref jsonb;
  v_measure jsonb;
  v_degenerate text;
  v_insert_cols text := '';
  v_select_cols text := '';
  v_joins text := '';
  v_sql text;
BEGIN
  -- Build column lists for dimension FKs
  FOR v_dim_ref IN SELECT jsonb_array_elements(p_entry->'dimension_references') LOOP
    v_insert_cols := v_insert_cols || format('%I, ', v_dim_ref->>'fk_column');
    
    -- Add JOIN to resolve surrogate key from business key
    v_select_cols := v_select_cols || format('dim_%s.%s, ',
      v_dim_ref->>'dimension',
      v_dim_ref->>'dimension' || '_sk'
    );
    
    v_joins := v_joins || format(E'\nLEFT JOIN %I.%I dim_%s ON data.%s = dim_%s.%s AND dim_%s.is_current',
      p_schema_name,
      v_dim_ref->>'dimension',
      v_dim_ref->>'dimension',
      -- Assume business key column name matches dimension name + '_id'
      v_dim_ref->>'dimension' || '_id',
      v_dim_ref->>'dimension',
      v_dim_ref->>'dimension' || '_id',
      v_dim_ref->>'dimension'
    );
  END LOOP;
  
  -- Add degenerate dimensions and measures
  IF p_entry ? 'degenerate_dimensions' THEN
    FOR v_degenerate IN SELECT jsonb_array_elements_text(p_entry->'degenerate_dimensions') LOOP
      v_insert_cols := v_insert_cols || format('%I, ', v_degenerate);
      v_select_cols := v_select_cols || format('data.%I, ', v_degenerate);
    END LOOP;
  END IF;
  
  FOR v_measure IN SELECT jsonb_array_elements(p_entry->'measures') LOOP
    v_insert_cols := v_insert_cols || format('%I, ', v_measure->>'name');
    v_select_cols := v_select_cols || format('data.%I, ', v_measure->>'name');
  END LOOP;
  
  -- Add event timestamp
  v_insert_cols := v_insert_cols || 'event_timestamp, created_at';
  v_select_cols := v_select_cols || 'data.event_timestamp, now()';
  
  v_sql := format($f$
WITH data AS (
  SELECT * FROM jsonb_to_recordset(:rows) AS r(
    %s  -- Column definitions will be generated based on metadata
  )
)
INSERT INTO %I.%I (%s)
SELECT %s
FROM data
%s
WHERE 1=1;  -- Add validation: all required dimension FKs must resolve
$f$,
    -- Column definitions placeholder
    'PLACEHOLDER_FOR_COLUMN_DEFS',
    p_schema_name, p_table_name,
    v_insert_cols,
    v_select_cols,
    v_joins
  );
  
  RETURN v_sql;
END;
$$;
```

### 4. Add Data Quality Validation

**New table in migration:**
```sql
CREATE TABLE IF NOT EXISTS batch.data_quality_rules (
  rule_id serial PRIMARY KEY,
  schema_name text NOT NULL,
  table_name text NOT NULL,
  rule_type text NOT NULL, -- 'required', 'type_check', 'regex', 'range', 'unique'
  column_path text,  -- 'id' or 'properties->email'
  rule_config jsonb DEFAULT '{}'::jsonb,
  severity text DEFAULT 'error', -- 'warning' | 'error' | 'critical'
  is_active boolean DEFAULT true,
  created_at timestamptz DEFAULT now()
);

CREATE INDEX ix_dq_rules_table ON batch.data_quality_rules(schema_name, table_name) 
  WHERE is_active = true;
```

**Add to `helper_functions.sql`:**
```sql
CREATE OR REPLACE FUNCTION validate_batch(
  p_schema text,
  p_table text,
  p_rows jsonb
) RETURNS jsonb LANGUAGE plpgsql AS $$
DECLARE
  v_rule RECORD;
  v_errors jsonb := '[]'::jsonb;
  v_row_idx int := 0;
  v_row jsonb;
  v_value text;
  v_valid_count int := 0;
BEGIN
  -- Iterate through rows
  FOR v_row IN SELECT value FROM jsonb_array_elements(p_rows) LOOP
    v_row_idx := v_row_idx + 1;
    
    -- Check all active rules for this table
    FOR v_rule IN 
      SELECT * FROM batch.data_quality_rules 
      WHERE schema_name = p_schema 
        AND table_name = p_table 
        AND is_active = true
    LOOP
      -- Extract value from row
      EXECUTE format('SELECT $1->>%L', v_rule.column_path) INTO v_value USING v_row;
      
      -- Apply rule based on type
      CASE v_rule.rule_type
        WHEN 'required' THEN
          IF v_value IS NULL OR trim(v_value) = '' THEN
            v_errors := v_errors || jsonb_build_object(
              'row_index', v_row_idx,
              'rule_id', v_rule.rule_id,
              'severity', v_rule.severity,
              'message', format('Required field %s is missing', v_rule.column_path)
            );
          END IF;
          
        WHEN 'regex' THEN
          IF v_value IS NOT NULL AND v_value !~ (v_rule.rule_config->>'pattern') THEN
            v_errors := v_errors || jsonb_build_object(
              'row_index', v_row_idx,
              'rule_id', v_rule.rule_id,
              'severity', v_rule.severity,
              'message', format('Field %s does not match pattern %s', 
                v_rule.column_path, v_rule.rule_config->>'pattern')
            );
          END IF;
      END CASE;
    END LOOP;
  END LOOP;
  
  v_valid_count := jsonb_array_length(p_rows) - jsonb_array_length(v_errors);
  
  RETURN jsonb_build_object(
    'total_rows', jsonb_array_length(p_rows),
    'valid_rows', v_valid_count,
    'invalid_rows', jsonb_array_length(v_errors),
    'errors', v_errors,
    'has_critical_errors', EXISTS(
      SELECT 1 FROM jsonb_array_elements(v_errors) e 
      WHERE e->>'severity' IN ('error', 'critical')
    )
  );
END;
$$;
```

### 5. Enhanced Metadata Validation

**Update `validate_metadata_entry()` in `helper_functions.sql`:**
```sql
CREATE OR REPLACE FUNCTION validate_metadata_entry(entry jsonb)
RETURNS boolean LANGUAGE plpgsql AS $$
DECLARE
  pk_item text;
  idx_item text;
  v_entity_type text;
BEGIN
  IF entry IS NULL THEN
    RAISE EXCEPTION 'Metadata entry cannot be null';
  END IF;
  
  -- Validate required name field
  IF NOT (entry ? 'name') OR trim(coalesce(entry->>'name', '')) = '' THEN
    RAISE EXCEPTION 'Metadata entry must have a non-empty "name" field';
  END IF;
  
  PERFORM validate_identifier(entry->>'name', 'table name');
  
  -- Validate entity_type
  v_entity_type := COALESCE(entry->>'entity_type', 'dimension');
  IF v_entity_type NOT IN ('dimension', 'transaction_fact', 'fact', 'bridge') THEN
    RAISE EXCEPTION 'Invalid entity_type: %. Must be dimension, transaction_fact, or bridge', v_entity_type;
  END IF;
  
  -- Validate grain is present (warning only)
  IF NOT (entry ? 'grain') THEN
    RAISE WARNING 'Entity % has no grain declaration', entry->>'name';
  END IF;
  
  -- Entity-specific validation
  IF v_entity_type IN ('dimension') THEN
    -- Validate primary_keys for dimensions
    IF entry ? 'primary_keys' THEN
      IF jsonb_typeof(entry->'primary_keys') != 'array' THEN
        RAISE EXCEPTION 'primary_keys must be an array of column names';
      END IF;
      
      FOR pk_item IN SELECT jsonb_array_elements_text(entry->'primary_keys') LOOP
        PERFORM validate_identifier(pk_item, 'primary key column');
      END LOOP;
    END IF;
    
    -- Validate SCD type
    IF entry ? 'scd' THEN
      IF upper(entry->>'scd') NOT IN ('SCD0', 'SCD1', 'SCD2', 'SCD3') THEN
        RAISE EXCEPTION 'Invalid SCD type: %. Must be SCD0, SCD1, SCD2, or SCD3', entry->>'scd';
      END IF;
    END IF;
    
  ELSIF v_entity_type IN ('transaction_fact', 'fact') THEN
    -- Validate dimension_references for facts
    IF NOT (entry ? 'dimension_references') THEN
      RAISE EXCEPTION 'Fact table % must have dimension_references array', entry->>'name';
    END IF;
    
    -- Validate measures for facts
    IF NOT (entry ? 'measures') THEN
      RAISE EXCEPTION 'Fact table % must have measures array', entry->>'name';
    END IF;
    
    -- Validate measure additivity
    DECLARE
      v_measure jsonb;
    BEGIN
      FOR v_measure IN SELECT jsonb_array_elements(entry->'measures') LOOP
        IF v_measure ? 'additivity' THEN
          IF (v_measure->>'additivity') NOT IN ('additive', 'semi_additive', 'non_additive') THEN
            RAISE EXCEPTION 'Measure % has invalid additivity: %. Must be additive, semi_additive, or non_additive',
              v_measure->>'name', v_measure->>'additivity';
          END IF;
        END IF;
      END LOOP;
    END;
  END IF;
  
  -- Validate physical_columns if present
  IF entry ? 'physical_columns' THEN
    IF jsonb_typeof(entry->'physical_columns') != 'array' THEN
      RAISE EXCEPTION 'physical_columns must be an array';
    END IF;
  END IF;
  
  RETURN true;
END;
$$;
```

## Testing Strategy

### Unit Tests (in SQL)
```sql
-- Test: Generate fact DDL
DO $$
DECLARE
  v_metadata jsonb := '{
    "name": "fact_test",
    "entity_type": "transaction_fact",
    "grain": "One row per transaction",
    "dimension_references": [
      {"dimension": "dim_customer", "fk_column": "customer_sk", "required": true}
    ],
    "measures": [
      {"name": "amount", "type": "numeric(10,2)", "additivity": "additive"}
    ]
  }'::jsonb;
  v_ddl text;
BEGIN
  v_ddl := generate_fact_ddl('test_schema', 'fact_test', v_metadata);
  ASSERT v_ddl LIKE '%customer_sk bigint NOT NULL%', 'Should have FK column';
  ASSERT v_ddl LIKE '%amount numeric(10,2)%', 'Should have typed measure column';
  ASSERT v_ddl NOT LIKE '%properties jsonb%', 'Facts should not have properties JSONB';
END$$;

-- Test: Validate grain requirement
DO $$
DECLARE
  v_metadata jsonb := '{"name": "test_table", "entity_type": "dimension"}'::jsonb;
  v_result boolean;
BEGIN
  v_result := validate_metadata_entry(v_metadata);
  -- Should succeed but log warning about missing grain
END$$;
```

### Integration Tests (Python + pytest)
```python
def test_fact_table_with_surrogate_key_resolution(db):
    """Test fact table resolves dimension surrogate keys correctly"""
    
    # Setup dimension
    db.execute("""
        INSERT INTO dim_customers (customer_sk, customer_id, properties, is_current)
        VALUES (1, 'C001', '{}', true)
    """)
    
    # Insert fact with business key
    fact_data = [{
        "customer_id": "C001",
        "amount": 100.50,
        "event_timestamp": "2024-01-15T10:00:00Z"
    }]
    
    db.execute(
        generate_fact_insert_dml('analytics', 'fact_orders', metadata),
        {"rows": json.dumps(fact_data)}
    )
    
    # Verify surrogate key was resolved
    result = db.execute("""
        SELECT customer_sk, amount 
        FROM fact_orders 
        WHERE customer_sk = 1
    """)
    
    assert result[0]['customer_sk'] == 1
    assert result[0]['amount'] == 100.50
```

## Common Mistakes to Avoid

1. **Don't use JSONB for measures in fact tables** - Use typed numeric columns for aggregation performance.

2. **Don't apply SCD versioning to fact tables** - Facts are immutable events. Insert only.

3. **Don't create foreign keys to JSONB paths** - Use physical columns for all FK relationships.

4. **Don't skip grain declaration** - Every entity must have a clear, documented grain.

5. **Don't reference business keys in fact tables** - Facts should reference dimension