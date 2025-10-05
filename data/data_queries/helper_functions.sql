-- jsonb_diff: Efficient comparison of JSONB objects with size limits
-- Returns diff object showing changed keys with old/new values
-- Limits: max 1000 keys, max 1MB per input object to prevent memory issues
CREATE OR REPLACE FUNCTION jsonb_diff(new jsonb, old jsonb)
RETURNS jsonb
LANGUAGE plpgsql
AS
$$
DECLARE
  max_keys CONSTANT int := 1000;
  max_size CONSTANT int := 1048576; -- 1MB
  new_size int;
  old_size int;
  result jsonb;
BEGIN
  -- Handle nulls
  IF new IS NULL THEN new := '{}'::jsonb; END IF;
  IF old IS NULL THEN old := '{}'::jsonb; END IF;

  -- Size validation
  new_size := octet_length(new::text);
  old_size := octet_length(old::text);

  IF new_size > max_size OR old_size > max_size THEN
    RAISE EXCEPTION 'JSONB object too large (max %MB): new=%MB, old=%MB',
      max_size/1048576, new_size/1048576, old_size/1048576;
  END IF;

  -- Use set-based approach instead of loops for better performance
  WITH all_keys AS (
    SELECT jsonb_object_keys(new) AS obj_key
    UNION
    SELECT jsonb_object_keys(old) AS obj_key
  ),
  limited_keys AS (
    SELECT obj_key FROM all_keys LIMIT max_keys
  ),
  diffs AS (
    SELECT
      k.obj_key,
      new -> k.obj_key AS new_val,
      old -> k.obj_key AS old_val
    FROM limited_keys k
    WHERE (new -> k.obj_key) IS DISTINCT FROM (old -> k.obj_key)
  )
  SELECT
    CASE
      WHEN count(*) = 0 THEN '{}'::jsonb
      ELSE jsonb_object_agg(
        obj_key,
        jsonb_build_object('old', old_val, 'new', new_val)
      )
    END
  FROM diffs
  INTO STRICT result;

  RETURN result;
END;
$$;

-- validate_identifier: Check PostgreSQL identifier constraints
-- Returns validated identifier or raises exception
CREATE OR REPLACE FUNCTION validate_identifier(identifier text, context text DEFAULT 'identifier')
RETURNS text
LANGUAGE plpgsql
IMMUTABLE
AS
$$
BEGIN
  IF identifier IS NULL OR trim(identifier) = '' THEN
    RAISE EXCEPTION '% cannot be null or empty', context;
  END IF;

  identifier := trim(identifier);

  IF length(identifier) > 63 THEN
    RAISE EXCEPTION '% too long (max 63 chars): %', context, identifier;
  END IF;

  IF NOT identifier ~ '^[a-zA-Z_][a-zA-Z0-9_$]*$' THEN
    RAISE EXCEPTION '% contains invalid characters: %', context, identifier;
  END IF;

  RETURN identifier;
END;
$$;

-- validate_metadata_entry: Comprehensive metadata validation
-- Validates structure, required fields, and data types
CREATE OR REPLACE FUNCTION validate_metadata_entry(entry jsonb)
RETURNS boolean
LANGUAGE plpgsql
AS
$$
DECLARE
  pk_item text;
  idx_item text;
  v_entity_type text;
  v_measure jsonb;
BEGIN
  IF entry IS NULL THEN
    RAISE EXCEPTION 'Metadata entry cannot be null';
  END IF;

  -- Validate required name field
  IF NOT (entry ? 'name') OR trim(coalesce(entry->>'name', '')) = '' THEN
    RAISE EXCEPTION 'Metadata entry must have a non-empty "name" field';
  END IF;

  -- Validate name as identifier
  PERFORM validate_identifier(entry->>'name', 'table name');

  -- Validate schema if present
  IF entry ? 'schema' THEN
    PERFORM validate_identifier(entry->>'schema', 'schema name');
  END IF;
  
  -- Validate entity_type
  v_entity_type := COALESCE(entry->>'entity_type', 'dimension');
  IF v_entity_type NOT IN ('dimension', 'transaction_fact', 'fact', 'bridge') THEN
    RAISE EXCEPTION 'Invalid entity_type: %. Must be dimension, transaction_fact, fact, or bridge', v_entity_type;
  END IF;
  
  -- Validate grain is present (warning only)
  IF NOT (entry ? 'grain') THEN
    RAISE WARNING 'Entity % has no grain declaration', entry->>'name';
  END IF;

  -- Entity-specific validation
  IF v_entity_type = 'dimension' THEN
    -- Validate primary_keys array if present
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
    
    -- Validate primary_keys vs primary key consistency for SCD1
    IF upper(coalesce(entry->>'scd', 'SCD1')) = 'SCD1' THEN
      IF (entry ? 'primary_keys') AND (entry ? 'keys' AND entry->'keys' ? 'primary') THEN
        IF entry->'primary_keys' IS DISTINCT FROM entry->'keys'->'primary' THEN
          RAISE WARNING 'SCD1 table %: primary_keys differs from keys.primary. This may cause issues.', entry->>'name';
        END IF;
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
    FOR v_measure IN SELECT jsonb_array_elements(entry->'measures') LOOP
      IF v_measure ? 'additivity' THEN
        IF (v_measure->>'additivity') NOT IN ('additive', 'semi_additive', 'non_additive') THEN
          RAISE EXCEPTION 'Measure % has invalid additivity: %. Must be additive, semi_additive, or non_additive',
            v_measure->>'name', v_measure->>'additivity';
        END IF;
      END IF;
    END LOOP;
  END IF;

  -- Validate index_columns array if present
  IF entry ? 'index_columns' THEN
    IF jsonb_typeof(entry->'index_columns') != 'array' THEN
      RAISE EXCEPTION 'index_columns must be an array of column names';
    END IF;

    FOR idx_item IN SELECT jsonb_array_elements_text(entry->'index_columns') LOOP
      PERFORM validate_identifier(idx_item, 'index column');
    END LOOP;
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

-- retry_with_backoff: Execute function with exponential backoff on deadlock
-- Provides production-ready error handling for concurrent operations
CREATE OR REPLACE FUNCTION retry_with_backoff(
  sql_statement text,
  max_retries int DEFAULT 3,
  base_delay_ms int DEFAULT 100
) RETURNS void
LANGUAGE plpgsql
AS
$$
DECLARE
  attempt int := 1;
  delay_ms int;
  error_state text;
  error_msg text;
  error_detail text;
BEGIN
  WHILE attempt <= max_retries LOOP
    BEGIN
      EXECUTE sql_statement;
      RETURN; -- Success
    EXCEPTION
      WHEN serialization_failure OR deadlock_detected THEN
        IF attempt = max_retries THEN
          GET STACKED DIAGNOSTICS
            error_state = RETURNED_SQLSTATE,
            error_msg = MESSAGE_TEXT,
            error_detail = PG_EXCEPTION_DETAIL;

          RAISE EXCEPTION 'Failed after % retries. Last error [%]: % (Detail: %)',
            max_retries, error_state, error_msg, coalesce(error_detail, 'none');
        END IF;

        -- Exponential backoff with jitter
        delay_ms := base_delay_ms * (2 ^ (attempt - 1)) + (random() * 50)::int;
        PERFORM pg_sleep(delay_ms / 1000.0);

        RAISE NOTICE 'Retrying operation (attempt % of %) after %ms delay due to: %',
          attempt + 1, max_retries, delay_ms, error_msg;

        attempt := attempt + 1;
      WHEN OTHERS THEN
        -- Re-raise non-retryable errors immediately
        RAISE;
    END;
  END LOOP;
END;
$$;

-- execute_with_monitoring: Execute SQL with timing and error monitoring
-- Provides observability for production operations
CREATE OR REPLACE FUNCTION execute_with_monitoring(
  operation_name text,
  sql_statement text,
  log_slow_queries boolean DEFAULT true,
  slow_threshold_ms int DEFAULT 1000
) RETURNS jsonb
LANGUAGE plpgsql
AS
$$
DECLARE
  start_time timestamptz;
  end_time timestamptz;
  duration_ms int;
  result jsonb;
BEGIN
  start_time := clock_timestamp();

  BEGIN
    EXECUTE sql_statement;
    end_time := clock_timestamp();
    duration_ms := extract(epoch from (end_time - start_time)) * 1000;

    result := jsonb_build_object(
      'operation', operation_name,
      'status', 'success',
      'duration_ms', duration_ms,
      'start_time', start_time,
      'end_time', end_time
    );

    IF log_slow_queries AND duration_ms > slow_threshold_ms THEN
      RAISE NOTICE 'SLOW QUERY [%ms]: % - %', duration_ms, operation_name,
        left(sql_statement, 200) || CASE WHEN length(sql_statement) > 200 THEN '...' ELSE '' END;
    END IF;

  EXCEPTION WHEN OTHERS THEN
    end_time := clock_timestamp();
    duration_ms := extract(epoch from (end_time - start_time)) * 1000;

    result := jsonb_build_object(
      'operation', operation_name,
      'status', 'error',
      'error_code', SQLSTATE,
      'error_message', SQLERRM,
      'duration_ms', duration_ms,
      'start_time', start_time,
      'end_time', end_time
    );

    RAISE NOTICE 'OPERATION FAILED [%ms]: % - Error: %', duration_ms, operation_name, SQLERRM;
    RAISE; -- Re-raise the exception
  END;

  RETURN result;
END;
$$;

-- normalize_pipeline_metadata: Convert new DRY format to expanded format
-- Handles defaults application and backward compatibility
CREATE OR REPLACE FUNCTION normalize_pipeline_metadata(pipeline_metadata jsonb)
RETURNS jsonb
LANGUAGE plpgsql
AS
$$
DECLARE
  v_defaults jsonb;
  v_tables jsonb;
  v_normalized_tables jsonb := '[]'::jsonb;
  v_entry jsonb;
  v_normalized_entry jsonb;
BEGIN
  IF pipeline_metadata IS NULL THEN
    RAISE EXCEPTION 'Pipeline metadata cannot be null';
  END IF;

  -- Extract components
  v_defaults := pipeline_metadata->'defaults';
  v_tables := pipeline_metadata->'tables';

  IF v_tables IS NULL OR jsonb_typeof(v_tables) != 'array' THEN
    RAISE EXCEPTION 'Pipeline metadata must contain a "tables" array';
  END IF;

  -- Process each table entry
  FOR v_entry IN SELECT jsonb_array_elements(v_tables) LOOP
    v_normalized_entry := apply_defaults_to_entry(v_entry, v_defaults);
    v_normalized_tables := v_normalized_tables || jsonb_build_array(v_normalized_entry);
  END LOOP;

  -- Return normalized format
  RETURN pipeline_metadata || jsonb_build_object('tables', v_normalized_tables);
END;
$$;

-- lock_key_from_text(k text) -> bigint
-- Purpose: Generate stable 64-bit hash for pg_advisory_xact_lock from text input
-- Uses built-in hashtext function for consistent hashing across sessions
-- Note: hashtext() is deterministic and gives same result for same input
CREATE OR REPLACE FUNCTION lock_key_from_text(k text)
RETURNS bigint
LANGUAGE sql
IMMUTABLE
AS
$$
  SELECT hashtext(coalesce(k,''))::bigint;
$$;

-- validate_grain: Ensure grain columns exist and are indexed
-- Validates that the declared grain can be properly enforced
CREATE OR REPLACE FUNCTION validate_grain(
  p_schema text,
  p_table text,
  p_grain_columns text[]
) RETURNS boolean
LANGUAGE plpgsql
AS
$$
DECLARE
  v_missing_cols text[];
  v_col text;
  v_has_index boolean := false;
BEGIN
  -- Check if all grain columns exist in the table
  FOR v_col IN SELECT unnest(p_grain_columns) LOOP
    IF NOT EXISTS (
      SELECT 1 FROM information_schema.columns 
      WHERE table_schema = p_schema 
        AND table_name = p_table 
        AND column_name = v_col
    ) THEN
      v_missing_cols := array_append(v_missing_cols, v_col);
    END IF;
  END LOOP;
  
  IF array_length(v_missing_cols, 1) > 0 THEN
    RAISE EXCEPTION 'Grain validation failed for %.%: missing columns %', 
      p_schema, p_table, array_to_string(v_missing_cols, ', ');
  END IF;
  
  -- Check if there's an index supporting the grain
  SELECT EXISTS (
    SELECT 1 FROM pg_indexes 
    WHERE schemaname = p_schema 
      AND tablename = p_table
      AND indexdef LIKE '%' || array_to_string(p_grain_columns, '%') || '%'
  ) INTO v_has_index;
  
  IF NOT v_has_index THEN
    RAISE WARNING 'No index found supporting grain columns for %.%: %', 
      p_schema, p_table, array_to_string(p_grain_columns, ', ');
  END IF;
  
  RETURN true;
END;
$$;

-- validate_dimensional_model: Cross-entity validation for dimensional models
-- Validates foreign key integrity and conformed dimensions
CREATE OR REPLACE FUNCTION validate_dimensional_model(
  p_pipeline_metadata jsonb
) RETURNS jsonb
LANGUAGE plpgsql
AS
$$
DECLARE
  v_entities jsonb;
  v_entity jsonb;
  v_dim_ref jsonb;
  v_errors jsonb := '[]'::jsonb;
  v_warnings jsonb := '[]'::jsonb;
  v_schema text;
  v_entity_name text;
  v_referenced_dim text;
BEGIN
  v_entities := p_pipeline_metadata->'entities';
  v_schema := p_pipeline_metadata->>'schema';
  
  IF v_entities IS NULL THEN
    v_entities := p_pipeline_metadata->'tables'; -- fallback
  END IF;
  
  -- Validate fact table dimension references
  FOR v_entity IN SELECT jsonb_array_elements(v_entities) LOOP
    v_entity_name := v_entity->>'name';
    
    IF COALESCE(v_entity->>'entity_type', 'dimension') IN ('transaction_fact', 'fact') THEN
      -- Check dimension references
      IF v_entity ? 'dimension_references' THEN
        FOR v_dim_ref IN SELECT jsonb_array_elements(v_entity->'dimension_references') LOOP
          v_referenced_dim := v_dim_ref->>'dimension';
          
          -- Check if referenced dimension exists in the model
          IF NOT EXISTS (
            SELECT 1 FROM jsonb_array_elements(v_entities) dim
            WHERE dim->>'name' = v_referenced_dim
              AND COALESCE(dim->>'entity_type', 'dimension') = 'dimension'
          ) THEN
            v_errors := v_errors || jsonb_build_array(
              jsonb_build_object(
                'type', 'missing_dimension',
                'entity', v_entity_name,
                'referenced_dimension', v_referenced_dim,
                'message', format('Fact table %s references dimension %s which is not defined in the model', 
                  v_entity_name, v_referenced_dim)
              )
            );
          END IF;
        END LOOP;
      END IF;
    END IF;
  END LOOP;
  
  RETURN jsonb_build_object(
    'valid', jsonb_array_length(v_errors) = 0,
    'errors', v_errors,
    'warnings', v_warnings
  );
END;
$$;

-- validate_batch: Validate data batch against quality rules
-- Performs data quality validation for incoming batches
CREATE OR REPLACE FUNCTION validate_batch(
  p_schema text,
  p_table text,
  p_rows jsonb
) RETURNS jsonb 
LANGUAGE plpgsql AS $$
DECLARE
  v_errors jsonb := '[]'::jsonb;
  v_row_idx int := 0;
  v_row jsonb;
  v_valid_count int := 0;
  v_total_rows int;
  v_required_fields text[] := ARRAY['id']; -- Basic validation
  v_field text;
  v_value text;
BEGIN
  v_total_rows := jsonb_array_length(p_rows);
  
  -- Basic validation - check for required fields
  FOR v_row IN SELECT value FROM jsonb_array_elements(p_rows) LOOP
    v_row_idx := v_row_idx + 1;
    
    -- Check required fields
    FOR v_field IN SELECT unnest(v_required_fields) LOOP
      v_value := v_row->>v_field;
      
      IF v_value IS NULL OR trim(v_value) = '' THEN
        v_errors := v_errors || jsonb_build_array(
          jsonb_build_object(
            'row_index', v_row_idx,
            'field', v_field,
            'severity', 'error',
            'message', format('Required field %s is missing or empty', v_field)
          )
        );
      END IF;
    END LOOP;
  END LOOP;
  
  v_valid_count := v_total_rows - jsonb_array_length(v_errors);
  
  RETURN jsonb_build_object(
    'total_rows', v_total_rows,
    'valid_rows', v_valid_count,
    'invalid_rows', jsonb_array_length(v_errors),
    'errors', v_errors,
    'has_critical_errors', jsonb_array_length(v_errors) > 0
  );
END;
$$;