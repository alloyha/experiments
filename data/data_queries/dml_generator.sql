-- ================================================
-- Migration: Drop DML functions with old parameter names before recreating
-- ================================================
DROP FUNCTION IF EXISTS generate_scd1_upsert_dml(text, text, jsonb);
DROP FUNCTION IF EXISTS generate_scd2_process_dml(text, text, jsonb);
DROP FUNCTION IF EXISTS generate_snapshot_reconcile_dml(text, text, jsonb);

-- ================================================
-- SCD1: Batch upsert using jsonb_to_recordset (fixed dollar quoting)
-- ================================================
CREATE OR REPLACE FUNCTION generate_scd1_upsert_dml(
  p_schema_name text,
  p_table_name text,
  p_primary_keys jsonb  -- jsonb array of primary key column names
) RETURNS text LANGUAGE plpgsql AS
$$
DECLARE
  v_cols text;
  v_pk_cols text;
  v_sql text;
BEGIN
  IF p_primary_keys IS NULL OR jsonb_typeof(p_primary_keys) <> 'array' THEN
    RAISE EXCEPTION 'primary_keys must be a jsonb array of column names';
  END IF;

  v_cols := array_to_string(ARRAY(
    SELECT format('%I', trim(element)) FROM jsonb_array_elements_text(p_primary_keys) AS element
  ) || ARRAY['properties','row_hash','created_at','updated_at'], ', ');

  v_pk_cols := array_to_string(ARRAY(
    SELECT format('%I', trim(element)) FROM jsonb_array_elements_text(p_primary_keys) AS element
  ), ', ');

  v_sql := format($f$
WITH data AS (
  SELECT * FROM jsonb_to_recordset(:rows) AS r(%s)
)
INSERT INTO %I.%I (%s)
SELECT * FROM data
ON CONFLICT (%s) DO UPDATE
SET properties = EXCLUDED.properties,
    row_hash = EXCLUDED.row_hash,
    updated_at = now()
WHERE %I.%I.row_hash IS DISTINCT FROM EXCLUDED.row_hash;
$f$,
    array_to_string(ARRAY(
      SELECT format('%I text', trim(element)) FROM jsonb_array_elements_text(p_primary_keys) AS element
    ) || ARRAY['properties jsonb','row_hash text','created_at timestamptz','updated_at timestamptz'], ', '),
    p_schema_name, p_table_name,
    v_cols,
    v_pk_cols,
    p_schema_name, p_table_name
  );

  RETURN v_sql;
END;
$$;


-- ================================================
-- SCD2: Batch process using jsonb_to_recordset (FIXED FORMAT ARGUMENTS)
-- ================================================
CREATE OR REPLACE FUNCTION generate_scd2_process_dml(
  p_schema_name text,
  p_table_name text,
  p_primary_keys jsonb
) RETURNS text LANGUAGE plpgsql AS
$$
DECLARE
  v_pk_pred text;
  v_pk_pred_update text;
  v_insert_cols text;
  v_select_cols text;
  v_sql text;
  v_cols_def text;
  v_first_pk text;
  v_pk_list text;
BEGIN
  IF p_primary_keys IS NULL OR jsonb_typeof(p_primary_keys) <> 'array' THEN
    RAISE EXCEPTION 'primary_keys must be a jsonb array of column names';
  END IF;

  -- Get first primary key for single column references
  SELECT trim(element) INTO v_first_pk
  FROM jsonb_array_elements_text(p_primary_keys) AS element
  LIMIT 1;

  -- Get comma-separated list of primary keys
  v_pk_list := array_to_string(ARRAY(
    SELECT trim(element) FROM jsonb_array_elements_text(p_primary_keys) AS element
  ), ', ');

  v_pk_pred := array_to_string(ARRAY(
    SELECT format('%I = d.%I', trim(element), trim(element))
    FROM jsonb_array_elements_text(p_primary_keys) AS element
  ), ' AND ');

  v_pk_pred_update := array_to_string(ARRAY(
    SELECT format('t.%I = c.%I', trim(element), trim(element))
    FROM jsonb_array_elements_text(p_primary_keys) AS element
  ), ' AND ');

  v_insert_cols := array_to_string(ARRAY(
    SELECT format('%I', trim(element)) FROM jsonb_array_elements_text(p_primary_keys) AS element
  ) || ARRAY['properties','properties_diff','row_hash','valid_from','valid_to','is_current','created_at','updated_at'], ', ');

  v_select_cols := array_to_string(ARRAY(
    SELECT format('%I', trim(element)) FROM jsonb_array_elements_text(p_primary_keys) AS element
  ) || ARRAY['properties','calculated_properties_diff','row_hash','valid_from','valid_to','is_current','created_at','updated_at'], ', ');

  v_cols_def := array_to_string(ARRAY(
    SELECT format('%I text', trim(element)) FROM jsonb_array_elements_text(p_primary_keys) AS element
  ) || ARRAY['properties jsonb','properties_diff jsonb','row_hash text','valid_from timestamptz','valid_to timestamptz','is_current boolean','created_at timestamptz','updated_at timestamptz'], ', ');

  v_sql := format($f$
-- First, expire old versions for updated records
WITH data AS (
  SELECT * FROM jsonb_to_recordset(:rows) AS d(%s)
)
, current_records AS (
  SELECT t.*
  FROM %I.%I t
  WHERE t.is_current = true
)
, changes AS (
  SELECT d.*,
         c.row_hash as existing_hash,
         CASE
           WHEN c.%I IS NULL THEN 'INSERT'
           WHEN d.row_hash <> c.row_hash THEN 'UPDATE'
           ELSE 'UNCHANGED'
         END as change_type,
         CASE
           WHEN c.%I IS NULL THEN '{}'::jsonb
           WHEN d.row_hash <> c.row_hash THEN jsonb_diff(d.properties, c.properties)
           ELSE '{}'::jsonb
         END as calculated_properties_diff
  FROM data d
  LEFT JOIN current_records c USING (%s)
)
UPDATE %I.%I t
SET valid_to = now(), is_current = false
FROM changes c
WHERE %s
  AND t.is_current = true
  AND c.change_type = 'UPDATE';

-- Then, insert new versions (both new records and updates)
WITH data AS (
  SELECT * FROM jsonb_to_recordset(:rows) AS d(%s)
)
, current_records AS (
  SELECT t.*
  FROM %I.%I t
  WHERE t.is_current = true
)
, changes AS (
  SELECT d.*,
         c.row_hash as existing_hash,
         CASE
           WHEN c.%I IS NULL THEN 'INSERT'
           WHEN d.row_hash <> c.row_hash THEN 'UPDATE'
           ELSE 'UNCHANGED'
         END as change_type,
         CASE
           WHEN c.%I IS NULL THEN '{}'::jsonb
           WHEN d.row_hash <> c.row_hash THEN jsonb_diff(d.properties, c.properties)
           ELSE '{}'::jsonb
         END as calculated_properties_diff
  FROM data d
  LEFT JOIN current_records c USING (%s)
)
INSERT INTO %I.%I (%s)
SELECT %s
FROM changes c
WHERE c.change_type IN ('INSERT', 'UPDATE');
$f$,
    v_cols_def,
    p_schema_name, p_table_name,
    v_first_pk,
    v_first_pk,
    v_pk_list,
    p_schema_name, p_table_name,
    v_pk_pred_update,
    -- Second CTE for INSERT
    v_cols_def,
    p_schema_name, p_table_name,
    v_first_pk,
    v_first_pk,
    v_pk_list,
    p_schema_name, p_table_name,
    v_insert_cols,
    v_select_cols
  );

  RETURN v_sql;
END;
$$;


-- ================================================
-- Snapshot reconcile (set-based) using jsonb_to_recordset (FIXED FORMAT ARGUMENTS)
-- ================================================
CREATE OR REPLACE FUNCTION generate_snapshot_reconcile_dml(
  p_schema_name text,
  p_table_name text,
  p_primary_keys jsonb
) RETURNS text LANGUAGE plpgsql AS
$$
DECLARE
  v_keys_array text[];
  v_keys_list text;
  v_insert_cols text;
  v_insert_vals text;
  v_sql text;
  v_first_pk text;
BEGIN
  IF p_primary_keys IS NULL OR jsonb_typeof(p_primary_keys) <> 'array' THEN
    RAISE EXCEPTION 'primary_keys must be a jsonb array of column names';
  END IF;

  v_keys_array := ARRAY(SELECT trim(element) FROM jsonb_array_elements_text(p_primary_keys) AS element);
  v_keys_list := array_to_string(ARRAY(SELECT format('%I', x) FROM unnest(v_keys_array) AS x), ', ');
  v_first_pk := v_keys_array[1];

  v_insert_cols := v_keys_list || ', properties, properties_diff, row_hash, valid_from, valid_to, is_current, created_at, updated_at';
  v_insert_vals := array_to_string(ARRAY(SELECT format('s.%I', x) FROM unnest(v_keys_array)), ', ')
                   || ', s.properties, jsonb_diff(s.properties, coalesce(c.properties, ''{}''::jsonb)), md5(s.properties::text), :snapshot_ts, null, true, now(), now()';

  v_sql := format($f$
WITH snap AS (
  SELECT * FROM jsonb_to_recordset(:rows) AS s(%s)
)
, cur AS (
  SELECT %s, properties, row_hash
  FROM %I.%I
  WHERE is_current = true
)
, to_insert AS (
  SELECT s.*
  FROM snap s
  LEFT JOIN cur c USING (%s)
  WHERE c.%I IS NULL
)
, to_update AS (
  SELECT s.*, c.properties as old_properties
  FROM snap s
  JOIN cur c USING (%s)
  WHERE md5(s.properties::text) <> c.row_hash
)
, to_expire AS (
  SELECT %s FROM cur c
  LEFT JOIN snap s USING (%s)
  WHERE s.%I IS NULL
)
UPDATE %I.%I t
SET valid_to = :snapshot_ts, is_current = false
WHERE (%s) IN (SELECT %s FROM to_expire)
  AND is_current = true;

INSERT INTO %I.%I (%s)
SELECT %s FROM (
  SELECT * FROM to_insert
  UNION ALL
  SELECT * FROM to_update
) s;
$f$,
    v_insert_cols || ', properties jsonb, properties_diff jsonb, row_hash text, valid_from timestamptz, valid_to timestamptz, is_current boolean, created_at timestamptz, updated_at timestamptz',
    v_keys_list, p_schema_name, p_table_name,
    v_keys_list, v_first_pk,
    v_keys_list,
    v_keys_list,
    v_keys_list, v_first_pk,
    p_schema_name, p_table_name,
    v_keys_list, v_keys_list,
    p_schema_name, p_table_name,
    v_insert_cols,
    v_insert_vals
  );

  RETURN v_sql;
END;
$$;


-- ================================================
-- Fact Insert: Generate insert DML for fact tables with surrogate key resolution
-- ================================================
CREATE OR REPLACE FUNCTION generate_fact_insert_dml(
  p_schema_name text,
  p_table_name text,
  p_entry jsonb
) RETURNS text LANGUAGE plpgsql AS
$$
DECLARE
  v_dim_ref jsonb;
  v_measure jsonb;
  v_degenerate text;
  v_insert_cols text := '';
  v_select_cols text := '';
  v_joins text := '';
  v_col_defs text := '';
  v_sql text;
  v_event_ts_col text;
BEGIN
  -- Validate required metadata
  IF NOT (p_entry ? 'dimension_references') THEN
    RAISE EXCEPTION 'Fact table % must have dimension_references array', p_table_name;
  END IF;
  
  IF NOT (p_entry ? 'measures') THEN
    RAISE EXCEPTION 'Fact table % must have measures array', p_table_name;
  END IF;
  
  -- Build column definitions for jsonb_to_recordset
  -- Add business key columns for dimension resolution
  FOR v_dim_ref IN SELECT jsonb_array_elements(p_entry->'dimension_references') LOOP
    DECLARE
      v_business_key text := COALESCE(
        v_dim_ref->>'business_key', 
        regexp_replace(v_dim_ref->>'dimension', '^dim_', '') || '_id'
      );
    BEGIN
      v_col_defs := v_col_defs || format('%I text, ', v_business_key);
    END;
  END LOOP;
  
  -- Add degenerate dimensions to column definitions
  IF p_entry ? 'degenerate_dimensions' THEN
    FOR v_degenerate IN SELECT jsonb_array_elements_text(p_entry->'degenerate_dimensions') LOOP
      v_col_defs := v_col_defs || format('%I text, ', v_degenerate);
    END LOOP;
  END IF;
  
  -- Add measures to column definitions
  FOR v_measure IN SELECT jsonb_array_elements(p_entry->'measures') LOOP
    v_col_defs := v_col_defs || format('%I %s, ', 
      v_measure->>'name',
      v_measure->>'type'
    );
  END LOOP;
  
  -- Add event timestamp to column definitions
  v_event_ts_col := COALESCE(p_entry->>'event_timestamp_column', 'event_timestamp');
  v_col_defs := v_col_defs || format('%I timestamptz', v_event_ts_col);
  
  -- Build insert and select column lists for dimension FKs
  FOR v_dim_ref IN SELECT jsonb_array_elements(p_entry->'dimension_references') LOOP
    DECLARE
      v_business_key text := COALESCE(
        v_dim_ref->>'business_key', 
        regexp_replace(v_dim_ref->>'dimension', '^dim_', '') || '_id'
      );
      v_fk_col text := v_dim_ref->>'fk_column';
      v_dim_table text := v_dim_ref->>'dimension';
    BEGIN
      v_insert_cols := v_insert_cols || format('%I, ', v_fk_col);
      
      -- Add JOIN to resolve surrogate key from business key
      v_select_cols := v_select_cols || format('dim_%s.%s, ',
        v_dim_table,
        v_dim_table || '_sk'
      );
      
      -- Build JOIN clause - handle both SCD1 and SCD2
      v_joins := v_joins || format(E'\nLEFT JOIN %I.%I dim_%s ON data.%I = dim_%s.%I',
        p_schema_name,
        v_dim_table,
        v_dim_table,
        v_business_key,
        v_dim_table,
        v_business_key
      );
      
      -- Add SCD2 current record filter if needed
      v_joins := v_joins || format(' AND COALESCE(dim_%s.is_current, true) = true', v_dim_table);
    END;
  END LOOP;
  
  -- Add degenerate dimensions and measures to insert/select
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
  
  -- Add event timestamp and created_at
  v_insert_cols := v_insert_cols || format('%I, created_at', v_event_ts_col);
  v_select_cols := v_select_cols || format('data.%I, now()', v_event_ts_col);
  
  v_sql := format($f$
WITH data AS (
  SELECT * FROM jsonb_to_recordset(:rows) AS r(%s)
)
INSERT INTO %I.%I (%s)
SELECT %s
FROM data
%s
WHERE 1=1
  -- Validate all required dimension FKs resolved
%s;
$f$,
    v_col_defs,
    p_schema_name, p_table_name,
    v_insert_cols,
    v_select_cols,
    v_joins,
    -- Add validation for required dimensions
    CASE 
      WHEN EXISTS(
        SELECT 1 FROM jsonb_array_elements(p_entry->'dimension_references') dr 
        WHERE COALESCE((dr->>'required')::boolean, false)
      ) THEN E'\n  -- Check required dimensions resolved\n  AND ' ||
        array_to_string(
          ARRAY(
            SELECT format('dim_%s.%s IS NOT NULL', 
              dr->>'dimension', 
              dr->>'dimension' || '_sk'
            )
            FROM jsonb_array_elements(p_entry->'dimension_references') dr 
            WHERE COALESCE((dr->>'required')::boolean, false)
          ), 
          E'\n  AND '
        )
      ELSE ''
    END
  );
  
  RETURN v_sql;
END;
$$;


-- ================================================
-- Helper function for applying defaults to metadata entries
-- ================================================
CREATE OR REPLACE FUNCTION apply_defaults_to_entry(
  p_entry jsonb,
  p_defaults jsonb
) RETURNS jsonb LANGUAGE plpgsql AS
$$
DECLARE
  v_result jsonb := p_entry;
  v_key text;
  v_value jsonb;
BEGIN
  -- Apply defaults for any missing keys
  IF p_defaults IS NOT NULL THEN
    FOR v_key, v_value IN SELECT * FROM jsonb_each(p_defaults)
    LOOP
      IF NOT (v_result ? v_key) THEN
        v_result := v_result || jsonb_build_object(v_key, v_value);
      END IF;
    END LOOP;
  END IF;

  RETURN v_result;
END;
$$;

-- ================================================
-- Main generate_dml (format calls inside are unchanged)
-- ================================================
CREATE OR REPLACE FUNCTION generate_dml(
  p_metadata jsonb,
  p_default_schema text DEFAULT 'public'
) RETURNS text LANGUAGE plpgsql AS
$$
DECLARE
  v_entry jsonb;
  v_processed_entry jsonb;
  v_sql_all text := '';
  v_is_array boolean := (jsonb_typeof(p_metadata) = 'array');
  v_is_pipeline boolean := (p_metadata ? 'tables');
  v_name text;
  v_schema text;
  v_entity_type text;
  v_scd text;
  v_bk jsonb;
  v_set_semantics boolean;
  v_dml text;
  v_default_bk jsonb := jsonb_build_array('id');
  v_defaults jsonb;
  v_tables jsonb;
BEGIN
  IF p_metadata IS NULL THEN
    RAISE EXCEPTION 'metadata is required';
  END IF;

  -- Handle pipeline format
  IF v_is_pipeline THEN
    v_defaults := p_metadata->'defaults';
    v_tables := p_metadata->'tables';
    v_schema := COALESCE(p_metadata->>'schema', p_default_schema);

    FOR v_entry IN SELECT jsonb_array_elements(v_tables)
    LOOP
      v_processed_entry := apply_defaults_to_entry(v_entry, v_defaults);
      v_name := v_processed_entry->>'name';
      v_entity_type := COALESCE(v_processed_entry->>'entity_type', 'dimension');
      v_scd := COALESCE(v_processed_entry->>'scd','SCD1');
      v_set_semantics := coalesce((v_processed_entry->>'set_semantics')::boolean,false);
      v_bk := COALESCE(v_processed_entry->'primary_keys', v_processed_entry->'keys'->'primary', v_default_bk);

      v_sql_all := v_sql_all || E'\n\n-- DML templates for ' || COALESCE(v_schema||'.'||v_name, v_name) || E'\n';

      IF v_entity_type = 'dimension' THEN
        IF upper(v_scd) = 'SCD1' THEN
          v_dml := generate_scd1_upsert_dml(v_schema, v_name, v_bk);
          v_sql_all := v_sql_all || v_dml;

        ELSIF upper(v_scd) = 'SCD2' THEN
          IF v_set_semantics THEN
            v_dml := generate_snapshot_reconcile_dml(v_schema, v_name, v_bk);
          ELSE
            v_dml := generate_scd2_process_dml(v_schema, v_name, v_bk);
          END IF;
          v_sql_all := v_sql_all || v_dml;

        ELSE
          -- fallback SCD1
          v_dml := generate_scd1_upsert_dml(v_schema, v_name, v_bk);
          v_sql_all := v_sql_all || v_dml;
        END IF;
        
      ELSIF v_entity_type IN ('transaction_fact', 'fact') THEN
        -- Facts are insert-only (immutable events)
        v_dml := generate_fact_insert_dml(v_schema, v_name, v_processed_entry);
        v_sql_all := v_sql_all || v_dml;
        
      ELSE
        RAISE EXCEPTION 'Unknown entity_type: %. Must be "dimension", "transaction_fact", or "fact"', v_entity_type;
      END IF;
    END LOOP;

    RETURN v_sql_all;
  END IF;

  IF v_is_array THEN
    FOR v_entry IN SELECT * FROM jsonb_array_elements(p_metadata)
    LOOP
      v_name := v_entry->>'name';
      v_schema := COALESCE(v_entry->>'schema', p_default_schema);
      v_entity_type := COALESCE(v_entry->>'entity_type', 'dimension');
      v_scd := COALESCE(v_entry->>'scd','SCD1');
      v_set_semantics := coalesce((v_entry->>'set_semantics')::boolean,false);
      v_bk := COALESCE(v_entry->'primary_keys', v_entry->'keys'->'primary', v_default_bk);

      v_sql_all := v_sql_all || E'\n\n-- DML templates for ' || COALESCE(v_schema||'.'||v_name, v_name) || E'\n';

      IF v_entity_type = 'dimension' THEN
        IF upper(v_scd) = 'SCD1' THEN
          v_dml := generate_scd1_upsert_dml(v_schema, v_name, v_bk);
          v_sql_all := v_sql_all || v_dml;

        ELSIF upper(v_scd) = 'SCD2' THEN
          IF v_set_semantics THEN
            v_dml := generate_snapshot_reconcile_dml(v_schema, v_name, v_bk);
          ELSE
            v_dml := generate_scd2_process_dml(v_schema, v_name, v_bk);
          END IF;
          v_sql_all := v_sql_all || v_dml;

        ELSE
          -- fallback SCD1
          v_dml := generate_scd1_upsert_dml(v_schema, v_name, v_bk);
          v_sql_all := v_sql_all || v_dml;
        END IF;
        
      ELSIF v_entity_type IN ('transaction_fact', 'fact') THEN
        -- Facts are insert-only (immutable events)
        v_dml := generate_fact_insert_dml(v_schema, v_name, v_entry);
        v_sql_all := v_sql_all || v_dml;
        
      ELSE
        RAISE EXCEPTION 'Unknown entity_type: %. Must be "dimension", "transaction_fact", or "fact"', v_entity_type;
      END IF;
    END LOOP;

  ELSE
    -- single object metadata
    v_entry := p_metadata;
    v_name := v_entry->>'name';
    v_schema := COALESCE(v_entry->>'schema', p_default_schema);
    v_entity_type := COALESCE(v_entry->>'entity_type', 'dimension');
    v_scd := COALESCE(v_entry->>'scd','SCD1');
    v_set_semantics := coalesce((v_entry->>'set_semantics')::boolean,false);
    v_bk := COALESCE(v_entry->'primary_keys', v_entry->'keys'->'primary', v_default_bk);

    v_sql_all := v_sql_all || E'\n\n-- DML templates for ' || COALESCE(v_schema||'.'||v_name, v_name) || E'\n';

    IF v_entity_type = 'dimension' THEN
      IF upper(v_scd) = 'SCD1' THEN
        v_dml := generate_scd1_upsert_dml(v_schema, v_name, v_bk);
        v_sql_all := v_sql_all || v_dml;

      ELSIF upper(v_scd) = 'SCD2' THEN
        IF v_set_semantics THEN
          v_dml := generate_snapshot_reconcile_dml(v_schema, v_name, v_bk);
        ELSE
          v_dml := generate_scd2_process_dml(v_schema, v_name, v_bk);
        END IF;
        v_sql_all := v_sql_all || v_dml;
      ELSE
        -- fallback SCD1
        v_dml := generate_scd1_upsert_dml(v_schema, v_name, v_bk);
        v_sql_all := v_sql_all || v_dml;
      END IF;
      
    ELSIF v_entity_type IN ('transaction_fact', 'fact') THEN
      -- Facts are insert-only (immutable events)
      v_dml := generate_fact_insert_dml(v_schema, v_name, v_entry);
      v_sql_all := v_sql_all || v_dml;
      
    ELSE
      RAISE EXCEPTION 'Unknown entity_type: %. Must be "dimension", "transaction_fact", or "fact"', v_entity_type;
    END IF;
  END IF;

  RETURN v_sql_all;
END;
$$;