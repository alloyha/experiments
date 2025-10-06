  -- ============================================================
  -- Migration: Drop functions with old parameter names before recreating
  -- ============================================================
  DROP FUNCTION IF EXISTS generate_scd1_ddl(text, text, jsonb, jsonb);
  DROP FUNCTION IF EXISTS generate_scd2_ddl(text, text, jsonb, jsonb);
  DROP FUNCTION IF EXISTS generate_database_creation_sql_from_pipeline(jsonb);
  DROP FUNCTION IF EXISTS generate_database_creation_sql_from_pipeline(jsonb, boolean);
  DROP FUNCTION IF EXISTS generate_database_creation_sql(text);
  DROP FUNCTION IF EXISTS generate_database_creation_sql(text, boolean);
  DROP FUNCTION IF EXISTS generate_table_ddl(jsonb, text);
  DROP FUNCTION IF EXISTS generate_ddl_for_table_by_index(jsonb, integer, boolean);
  DROP FUNCTION IF EXISTS generate_ddl(jsonb, text, boolean);
  DROP FUNCTION IF EXISTS generate_ddl(jsonb, text);
  DROP FUNCTION IF EXISTS generate_ddl(jsonb);

  -- ============================================================
  -- apply_defaults_to_entry: Apply pipeline defaults to table entry
  -- ============================================================
  CREATE OR REPLACE FUNCTION apply_defaults_to_entry(
    p_entry jsonb,
    p_defaults jsonb
  ) RETURNS jsonb LANGUAGE plpgsql AS
  $$
  DECLARE
    v_result jsonb;
    v_key text;
  BEGIN
    v_result := p_entry;

    -- Apply defaults for missing keys
    IF p_defaults IS NOT NULL THEN
      FOR v_key IN SELECT jsonb_object_keys(p_defaults) LOOP
        IF NOT (v_result ? v_key) THEN
          v_result := v_result || jsonb_build_object(v_key, p_defaults->v_key);
        END IF;
      END LOOP;
    END IF;

    -- Convert simplified foreign_keys to old format for backward compatibility
    IF v_result ? 'foreign_keys' THEN
      -- Merge with existing keys structure or create new one
      v_result := v_result || jsonb_build_object('keys',
        COALESCE(v_result->'keys', '{}'::jsonb) || jsonb_build_object(
          'primary', COALESCE(v_result->'primary_keys', jsonb_build_array('id')),
          'foreign', v_result->'foreign_keys'
        )
      );
    ELSE
      -- Ensure keys structure exists if not present
      IF NOT (v_result ? 'keys') THEN
        v_result := v_result || jsonb_build_object('keys', jsonb_build_object(
          'primary', COALESCE(v_result->'primary_keys', jsonb_build_array('id')),
          'foreign', jsonb_build_array()
        ));
      END IF;
    END IF;

    RETURN v_result;
  END;
  $$;

  -- ============================================================
  -- apply_defaults_to_entry: Apply pipeline defaults to table entry
  -- ============================================================
  CREATE OR REPLACE FUNCTION generate_foreign_keys_sql(
    p_schema_name text,
    p_table_name text,
    p_foreign_keys jsonb
  ) RETURNS text LANGUAGE plpgsql AS
  $$
  DECLARE
    v_sql text := '';
    v_fk jsonb;
    v_constraint_name text;
    v_column text;
    v_ref_table text;
    v_ref_column text;
    v_ref_type text;
  BEGIN
    -- Note: Foreign keys are only created for physical columns (primary keys)
    -- Business attributes in properties JSONB cannot have foreign key constraints
    -- This function generates constraints but will only work if the referenced
    -- columns were created as physical columns in the table structure

    IF p_foreign_keys IS NOT NULL AND jsonb_typeof(p_foreign_keys) = 'array' THEN
      FOR v_fk IN SELECT value FROM jsonb_array_elements(p_foreign_keys) AS value LOOP
        v_column := v_fk->>'column';
        v_ref_table := v_fk->'reference'->>'table';
        v_ref_type := COALESCE(v_fk->'reference'->>'ref_type', 'business');
        
        IF v_ref_type = 'surrogate' THEN
          -- Reference the surrogate key column
          v_ref_column := v_ref_table || '_sk';
        ELSE
          -- Reference the business key column
          v_ref_column := v_fk->'reference'->>'column';
        END IF;

        IF v_column IS NOT NULL AND v_ref_table IS NOT NULL AND v_ref_column IS NOT NULL THEN
          v_constraint_name := format('fk_%s_%s_%s', lower(p_table_name), lower(v_column), lower(v_ref_table));
          -- Add a comment indicating this constraint requires physical columns
          v_sql := v_sql || E'\n' || format('-- Note: Foreign key on %I requires it to be a physical column', v_column);
          v_sql := v_sql || E'\n' || format(
            $fk$DO $do$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = '%s'
                  AND conrelid = '%I.%I'::regclass
              ) THEN
                EXECUTE 'ALTER TABLE %I.%I ADD CONSTRAINT %s FOREIGN KEY (%I) REFERENCES %I.%I (%I)';
              END IF;
            END$do$;
            $fk$,
            v_constraint_name, p_schema_name, p_table_name,
            p_schema_name, p_table_name, v_constraint_name, v_column, p_schema_name, v_ref_table, v_ref_column
          );
        END IF;
      END LOOP;
    END IF;

    RETURN v_sql;
  END;
  $$;

  -- Responsibility: emit index DDL given schema.table and metadata entry
  -- ============================================================
  CREATE OR REPLACE FUNCTION generate_indexes_sql(
    p_schema_name text,
    p_table_name text,
    p_base_name text,
    p_index_columns jsonb,           -- jsonb array of strings (index_columns)
    p_index_properties_gin boolean,
    p_index_properties_keys jsonb,   -- jsonb array of strings for expression indexes (hot keys)
    p_scd_type text DEFAULT 'SCD1'   -- SCD type to determine which timestamp column to index
  ) RETURNS text LANGUAGE plpgsql AS
  $$
  DECLARE
    v_sql text := '';
    v_col text;
    v_prop_key text;
  BEGIN
    -- index on listed index_columns (expression indexes on properties JSONB)
    IF p_index_columns IS NOT NULL AND jsonb_typeof(p_index_columns) = 'array' THEN
      FOR v_col IN SELECT jsonb_array_elements_text(p_index_columns) LOOP
        v_col := trim(v_col);
        IF v_col <> '' THEN
          v_sql := v_sql || E'\n' || format('CREATE INDEX IF NOT EXISTS ix_%s_%s ON %I.%I ((properties->>%L));', p_base_name, regexp_replace(lower(v_col),'\W','_','g'), p_schema_name, p_table_name, v_col);
        END IF;
      END LOOP;
    END IF;

    -- timestamp index (created_at for SCD2, updated_at for SCD1)
    IF upper(p_scd_type) = 'SCD1' THEN
      v_sql := v_sql || E'\n' || format('CREATE INDEX IF NOT EXISTS ix_%s_updated_at ON %I.%I (updated_at);', p_base_name, p_schema_name, p_table_name);
    ELSE
      v_sql := v_sql || E'\n' || format('CREATE INDEX IF NOT EXISTS ix_%s_created_at ON %I.%I (created_at);', p_base_name, p_schema_name, p_table_name);
    END IF;

    -- GIN on properties
    IF p_index_properties_gin THEN
      v_sql := v_sql || E'\n' || format('CREATE INDEX IF NOT EXISTS ix_%s_properties_gin ON %I.%I USING gin (properties);', p_base_name, p_schema_name, p_table_name);
    END IF;

    -- expression indexes for hot property keys (partial on is_current for SCD2 only)
    IF p_index_properties_keys IS NOT NULL AND jsonb_typeof(p_index_properties_keys) = 'array' THEN
      FOR v_prop_key IN SELECT jsonb_array_elements_text(p_index_properties_keys) LOOP
        v_prop_key := trim(v_prop_key);
        IF v_prop_key <> '' THEN
          IF upper(p_scd_type) = 'SCD1' THEN
            -- SCD1: simple expression index (no is_current column)
            v_sql := v_sql || E'\n' ||
              format('CREATE INDEX IF NOT EXISTS ix_%s_prop_%s ON %I.%I ((properties->>%L));', p_base_name, lower(regexp_replace(v_prop_key,'\W','_','g')), p_schema_name, p_table_name, v_prop_key);
          ELSE
            -- SCD2: partial index on current records only
            v_sql := v_sql || E'\n' ||
              format('CREATE INDEX IF NOT EXISTS ix_%s_prop_%s_current ON %I.%I ((properties->>%L)) WHERE is_current;', p_base_name, lower(regexp_replace(v_prop_key,'\W','_','g')), p_schema_name, p_table_name, v_prop_key);
          END IF;
        END IF;
      END LOOP;
    END IF;

    RETURN v_sql;
  END;
  $$;


  -- ============================================================
  -- generate_scd1_ddl: single-current upsert-style table generator
  -- ============================================================
  CREATE OR REPLACE FUNCTION generate_scd1_ddl(
    p_schema_name text,
    p_table_name text,
    p_entry jsonb,
    p_default_primary_keys jsonb,  -- jsonb array fallback for primary_keys
    p_include_schema_creation boolean DEFAULT false
  ) RETURNS text LANGUAGE plpgsql AS
  $$
  DECLARE
    v_sql text := '';
    v_base_name text;
    v_primary_keys jsonb;
    v_pk_col text;
    v_index_columns jsonb;
    v_index_properties_gin boolean := coalesce((p_entry->>'index_properties_gin')::boolean, true);
    v_index_properties_keys jsonb := p_entry->'index_properties_keys';
  BEGIN
    -- Optional schema creation (moved to higher level)
    IF p_include_schema_creation THEN
      v_sql := format('CREATE SCHEMA IF NOT EXISTS %I;', p_schema_name) || E'\n';
    END IF;

    -- base name for indexes
    v_base_name := lower(regexp_replace(p_schema_name||'_'||p_table_name,'\W','_','g'));

    -- determine primary keys (array)
    v_primary_keys := COALESCE(p_entry->'primary_keys', p_default_primary_keys);

    -- build table: surrogate key + business key columns + properties
    v_sql := v_sql || format('CREATE TABLE IF NOT EXISTS %I.%I (', p_schema_name, p_table_name);

    -- Always start with surrogate key for consistency with SCD2
    v_sql := v_sql || format('%I_sk bigserial PRIMARY KEY,', p_table_name);

    IF v_primary_keys IS NOT NULL AND jsonb_typeof(v_primary_keys) = 'array' THEN
      -- Add business key columns with unique constraint
      FOR v_pk_col IN SELECT jsonb_array_elements_text(v_primary_keys) LOOP
        v_sql := v_sql || format(' %I text NOT NULL,', trim(v_pk_col));
      END LOOP;
    END IF;

    -- Add physical columns if specified
    IF p_entry ? 'physical_columns' AND jsonb_typeof(p_entry->'physical_columns') = 'array' THEN
      DECLARE
        v_col jsonb;
        v_col_name text;
        v_col_type text;
        v_col_nullable boolean;
      BEGIN
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
      END;
    END IF;

    -- properties + bookkeeping with both timestamps, no event tracking
    v_sql := v_sql || ' properties jsonb DEFAULT ''{}''::jsonb, row_hash text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()';

    v_sql := v_sql || ');';

    -- Add unique constraint on business keys for SCD1 (one current version only), only if it does not exist
    IF v_primary_keys IS NOT NULL AND jsonb_typeof(v_primary_keys) = 'array' THEN
      v_sql := v_sql || E'\n' || format(
        $ux$DO $do$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = 'ux_%s_business_keys'
              AND conrelid = '%I.%I'::regclass
          ) THEN
            EXECUTE 'ALTER TABLE %I.%I ADD CONSTRAINT ux_%s_business_keys UNIQUE (%s)';
          END IF;
        END$do$;
        $ux$,
        v_base_name, p_schema_name, p_table_name,
        p_schema_name, p_table_name, v_base_name,
        array_to_string(ARRAY(SELECT format('%I', x) FROM jsonb_array_elements_text(v_primary_keys) AS x), ', ')
      );
    END IF;

    -- index columns
    v_index_columns := p_entry->'index_columns';

    v_sql := v_sql || generate_indexes_sql(p_schema_name, p_table_name, v_base_name, v_index_columns, v_index_properties_gin, v_index_properties_keys, 'SCD1');

    -- Add foreign key constraints
    v_sql := v_sql || generate_foreign_keys_sql(p_schema_name, p_table_name, p_entry->'keys'->'foreign');

    RETURN v_sql;
  END;
  $$;


  -- ============================================================
  -- generate_scd2_ddl: history SCD2 table generator
  -- ============================================================
  CREATE OR REPLACE FUNCTION generate_scd2_ddl(
    p_schema_name text,
    p_table_name text,
    p_entry jsonb,
    p_default_primary_keys jsonb,
    p_include_schema_creation boolean DEFAULT false
  ) RETURNS text LANGUAGE plpgsql AS
  $$
  DECLARE
    v_sql text := '';
    v_base_name text;
    v_primary_keys jsonb;
    v_pk_list text := '';
    v_pk_col text;
    v_index_columns jsonb;
    v_index_properties_gin boolean := coalesce((p_entry->>'index_properties_gin')::boolean, true);
    v_index_properties_keys jsonb := p_entry->'index_properties_keys';
  BEGIN
    -- Optional schema creation (moved to higher level)
    IF p_include_schema_creation THEN
      v_sql := format('CREATE SCHEMA IF NOT EXISTS %I;', p_schema_name) || E'\n';
    END IF;

    -- base name
    v_base_name := lower(regexp_replace(p_schema_name||'_'||p_table_name,'\W','_','g'));

    -- primary keys (for SCD2, we need business keys as physical columns for change detection)
    v_primary_keys := COALESCE(p_entry->'primary_keys', p_default_primary_keys);

    -- build column defs for primary keys
    IF v_primary_keys IS NOT NULL AND jsonb_typeof(v_primary_keys) = 'array' THEN
      FOR v_pk_col IN SELECT jsonb_array_elements_text(v_primary_keys) LOOP
        v_pk_list := v_pk_list || format('%I text NOT NULL,', trim(v_pk_col));
      END LOOP;
    END IF;

    -- Add physical columns if specified
    IF p_entry ? 'physical_columns' AND jsonb_typeof(p_entry->'physical_columns') = 'array' THEN
      DECLARE
        v_col jsonb;
        v_col_name text;
        v_col_type text;
        v_col_nullable boolean;
      BEGIN
        FOR v_col IN SELECT jsonb_array_elements(p_entry->'physical_columns') LOOP
          v_col_name := v_col->>'name';
          v_col_type := COALESCE(v_col->>'type', 'text');
          v_col_nullable := COALESCE((v_col->>'nullable')::boolean, true);
          
          v_pk_list := v_pk_list || format('%I %s%s,', 
            v_col_name, 
            v_col_type,
            CASE WHEN NOT v_col_nullable THEN ' NOT NULL' ELSE '' END
          );
        END LOOP;
      END;
    END IF;

    -- surrogate key with table-specific name
    v_sql := v_sql || format('CREATE TABLE IF NOT EXISTS %I.%I (', p_schema_name, p_table_name)
          || format('%I_sk bigserial PRIMARY KEY,', p_table_name)
          || ' ' || v_pk_list ||
          ' properties jsonb DEFAULT ''{}''::jsonb, properties_diff jsonb DEFAULT ''{}''::jsonb, row_hash text NOT NULL, valid_from timestamptz NOT NULL, valid_to timestamptz, is_current boolean NOT NULL DEFAULT true, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()'
          || ');';

    -- unique current index on primary key(s)
    IF v_primary_keys IS NOT NULL AND jsonb_typeof(v_primary_keys) = 'array' THEN
      v_sql := v_sql || E'\n' || format('CREATE UNIQUE INDEX IF NOT EXISTS ux_%s_current ON %I.%I (%s) WHERE is_current;', v_base_name, p_schema_name, p_table_name,
        array_to_string(ARRAY(SELECT format('%I', x) FROM jsonb_array_elements_text(v_primary_keys) AS x), ', '));
      -- validity index
      v_sql := v_sql || E'\n' || format('CREATE INDEX IF NOT EXISTS ix_%s_validity ON %I.%I (%s, valid_from, valid_to);', v_base_name, p_schema_name, p_table_name,
        array_to_string(ARRAY(SELECT format('%I', x) FROM jsonb_array_elements_text(v_primary_keys) AS x), ', '));
    END IF;

    -- other indexes (created_at, GIN, property keys)
    v_index_columns := p_entry->'index_columns';
    v_sql := v_sql || generate_indexes_sql(p_schema_name, p_table_name, v_base_name, v_index_columns, v_index_properties_gin, v_index_properties_keys, 'SCD2');

    -- Add foreign key constraints
    v_sql := v_sql || generate_foreign_keys_sql(p_schema_name, p_table_name, p_entry->'keys'->'foreign');

    -- Add retention policy if specified
    IF (p_entry->'retention_days') IS NOT NULL THEN
      v_sql := v_sql || E'\n' || format('-- Retention policy: %s days', p_entry->>'retention_days');
      v_sql := v_sql || E'\n' || format('-- Consider implementing: DELETE FROM %I.%I WHERE created_at < now() - interval ''%s days'';',
        p_schema_name, p_table_name, p_entry->>'retention_days');
    END IF;

    -- Add current materialized view if requested
    IF coalesce((p_entry->>'create_current_matview')::boolean, false) THEN
      v_sql := v_sql || E'\n' || format('CREATE MATERIALIZED VIEW IF NOT EXISTS %I.%I_current AS', p_schema_name, p_table_name);
      v_sql := v_sql || E'\n' || format('SELECT * FROM %I.%I WHERE is_current = true;', p_schema_name, p_table_name);
      v_sql := v_sql || E'\n' || format('CREATE UNIQUE INDEX IF NOT EXISTS ux_%s_current_matview ON %I.%I_current (%s);',
        v_base_name, p_schema_name, p_table_name,
        array_to_string(ARRAY(SELECT format('%I', x) FROM jsonb_array_elements_text(v_primary_keys) AS x), ', '));
    END IF;

    RETURN v_sql;
  END;
  $$;


  -- ============================================================
  -- generate_fact_ddl: Generate fact table DDL with typed measures
  -- ============================================================
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
    v_base_name text;
  BEGIN
    -- Validate metadata
    IF NOT (p_entry ? 'dimension_references') THEN
      RAISE EXCEPTION 'Fact table % must have dimension_references array', p_table_name;
    END IF;
    
    IF NOT (p_entry ? 'measures') THEN
      RAISE EXCEPTION 'Fact table % must have measures array', p_table_name;
    END IF;
    
    v_base_name := lower(regexp_replace(p_schema_name||'_'||p_table_name,'\W','_','g'));
    
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
      v_base_name,
      p_schema_name, p_table_name, v_event_ts_col
    );
    
    -- Add dimension FK indexes
    FOR v_dim_ref IN SELECT jsonb_array_elements(p_entry->'dimension_references') LOOP
      v_sql := v_sql || E'\n' || format(
        'CREATE INDEX IF NOT EXISTS ix_%s_%s ON %I.%I (%I);',
        v_base_name,
        lower(regexp_replace(v_dim_ref->>'fk_column', '\W', '_', 'g')),
        p_schema_name, p_table_name, v_dim_ref->>'fk_column'
      );
    END LOOP;
    
    -- Add foreign key constraints
    FOR v_dim_ref IN SELECT jsonb_array_elements(p_entry->'dimension_references') LOOP
      v_sql := v_sql || E'\n' || format(
        $fk$DO $do$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = 'fk_%s_%s'
              AND conrelid = '%I.%I'::regclass
          ) THEN
            EXECUTE 'ALTER TABLE %I.%I ADD CONSTRAINT fk_%s_%s FOREIGN KEY (%I) REFERENCES %I.%I(%s)';
          END IF;
        END$do$;
        $fk$,
        v_base_name,
        lower(regexp_replace(v_dim_ref->>'dimension', '\W', '_', 'g')),
        p_schema_name, p_table_name,
        p_schema_name, p_table_name,
        v_base_name,
        lower(regexp_replace(v_dim_ref->>'dimension', '\W', '_', 'g')),
        v_dim_ref->>'fk_column',
        p_schema_name, v_dim_ref->>'dimension',
        v_dim_ref->>'dimension' || '_sk'  -- Always reference surrogate key
      );
    END LOOP;
    
    RETURN v_sql;
  END;
  $$;


  -- ============================================================
  -- generate_table_ddl: single-entry dispatcher
  -- ============================================================
    CREATE OR REPLACE FUNCTION generate_table_ddl(
    p_entry jsonb,
    p_default_schema text,
    p_include_schema_creation boolean DEFAULT false
  ) RETURNS text LANGUAGE plpgsql AS
  $$
  DECLARE
    v_name text := p_entry->>'name';
    v_schema text := COALESCE(p_entry->>'schema', p_default_schema);
    v_entity_type text := COALESCE(p_entry->>'entity_type', 'dimension');
    v_scd text := COALESCE(p_entry->>'scd','SCD1');
    v_sql text := '';
    v_default_primary_keys jsonb := jsonb_build_array('id'); -- fallback
  BEGIN
    IF v_name IS NULL OR trim(v_name) = '' THEN
      RAISE EXCEPTION 'Entry must have a non-empty \"name\"';
    END IF;
    
    -- Validate table name for security
    PERFORM validate_identifier(v_name, 'table name');
    
    -- Validate grain is declared
    IF NOT (p_entry ? 'grain') THEN
      RAISE WARNING 'Entity % has no grain declaration. This may cause modeling errors.', v_name;
    END IF;

    IF v_entity_type = 'dimension' THEN
      IF upper(v_scd) = 'SCD1' THEN
        v_sql := generate_scd1_ddl(v_schema, v_name, p_entry, v_default_primary_keys, p_include_schema_creation);
      ELSIF upper(v_scd) = 'SCD2' OR upper(v_scd) = 'SCD2+FACT' OR upper(v_scd) = 'SCD2+fact' THEN
        v_sql := generate_scd2_ddl(v_schema, v_name, p_entry, v_default_primary_keys, p_include_schema_creation);
        -- Note: Event tables removed for simplified implementation
      ELSE
        -- unknown scd -> default to scd1
        v_sql := generate_scd1_ddl(v_schema, v_name, p_entry, v_default_primary_keys, p_include_schema_creation);
      END IF;
      
    ELSIF v_entity_type = 'transaction_fact' OR v_entity_type = 'fact' THEN
      v_sql := generate_fact_ddl(v_schema, v_name, p_entry);
      
    ELSE
      RAISE EXCEPTION 'Unknown entity_type: %. Must be \"dimension\", \"transaction_fact\", or \"fact\"', v_entity_type;
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


  -- ============================================================
  -- generate_schema_creation_sql_from_pipeline: Extract schema name and generate creation SQL
  -- ============================================================
  -- ============================================================
  -- generate_schema_creation_sql_from_pipeline: Extract schema name and generate creation SQL
  -- ============================================================
  CREATE OR REPLACE FUNCTION generate_schema_creation_sql_from_pipeline(
    p_pipeline_metadata jsonb
  ) RETURNS text LANGUAGE plpgsql AS
  $$
  DECLARE
    v_schema text;
  BEGIN
    v_schema := p_pipeline_metadata->>'schema';

    IF v_schema IS NULL OR trim(v_schema) = '' THEN
      RETURN '-- No schema specified in pipeline metadata, using database default schema';
    END IF;

    RETURN format('CREATE SCHEMA IF NOT EXISTS %I;', trim(v_schema));
  END;
  $$;

  -- ============================================================
  -- get_table_count_from_pipeline: Count tables in pipeline metadata
  -- ============================================================
  CREATE OR REPLACE FUNCTION get_table_count_from_pipeline(
    p_pipeline_metadata jsonb
  ) RETURNS integer LANGUAGE plpgsql AS
  $$
  DECLARE
    v_tables jsonb;
  BEGIN
    v_tables := p_pipeline_metadata->'tables';

    IF v_tables IS NULL OR jsonb_typeof(v_tables) != 'array' THEN
      RETURN 0;
    END IF;

    RETURN jsonb_array_length(v_tables);
  END;
  $$;

  -- ============================================================
  -- get_table_entry_from_pipeline: Extract a single table entry by index
  -- ============================================================
  CREATE OR REPLACE FUNCTION get_table_entry_from_pipeline(
    p_pipeline_metadata jsonb,
    p_table_index integer  -- 0-based index
  ) RETURNS jsonb LANGUAGE plpgsql AS
  $$
  DECLARE
    v_tables jsonb;
    v_defaults jsonb;
    v_schema text;
    v_entry jsonb;
    v_processed_entry jsonb;
  BEGIN
    v_tables := p_pipeline_metadata->'tables';
    v_defaults := p_pipeline_metadata->'defaults';
    v_schema := p_pipeline_metadata->>'schema';

    IF v_tables IS NULL OR jsonb_typeof(v_tables) != 'array' THEN
      RAISE EXCEPTION 'Pipeline metadata must contain a "tables" array';
    END IF;

    IF p_table_index < 0 OR p_table_index >= jsonb_array_length(v_tables) THEN
      RAISE EXCEPTION 'Table index % is out of range (0 to %)', p_table_index, jsonb_array_length(v_tables) - 1;
    END IF;

    -- Get the table entry
    v_entry := v_tables->p_table_index;

    -- Apply defaults and add schema info
    v_processed_entry := apply_defaults_to_entry(v_entry, v_defaults);

    -- Add schema to the entry if not already present
    IF NOT (v_processed_entry ? 'schema') AND v_schema IS NOT NULL THEN
      v_processed_entry := v_processed_entry || jsonb_build_object('schema', v_schema);
    END IF;

    RETURN v_processed_entry;
  END;
  $$;

  -- ============================================================
  -- generate_ddl_for_table_by_index: Generate DDL for a specific table by index
  -- ============================================================
  CREATE OR REPLACE FUNCTION generate_ddl_for_table_by_index(
    p_pipeline_metadata jsonb,
    p_table_index integer,  -- 0-based index
    p_include_schema_creation boolean DEFAULT false
  ) RETURNS text LANGUAGE plpgsql AS
  $$
  DECLARE
    v_table_entry jsonb;
    v_schema text;
    v_ddl text;
  BEGIN
    -- Get the processed table entry
    v_table_entry := get_table_entry_from_pipeline(p_pipeline_metadata, p_table_index);
    v_schema := COALESCE(v_table_entry->>'schema', p_pipeline_metadata->>'schema');

    -- Generate DDL for just this table
    v_ddl := generate_table_ddl(v_table_entry, v_schema, p_include_schema_creation);

    RETURN v_ddl;
  END;
  $$;

  -- ============================================================
  -- list_table_names_from_pipeline: Get list of table names in order
  -- ============================================================
  CREATE OR REPLACE FUNCTION list_table_names_from_pipeline(
    p_pipeline_metadata jsonb
  ) RETURNS text[] LANGUAGE plpgsql AS
  $$
  DECLARE
    v_tables jsonb;
    v_entry jsonb;
    v_names text[] := '{}';
  BEGIN
    v_tables := p_pipeline_metadata->'tables';

    IF v_tables IS NULL OR jsonb_typeof(v_tables) != 'array' THEN
      RETURN v_names;
    END IF;

    FOR v_entry IN SELECT value FROM jsonb_array_elements(v_tables) AS value LOOP
      v_names := array_append(v_names, v_entry->>'name');
    END LOOP;

    RETURN v_names;
  END;
  $$;

  -- ============================================================
  -- generate_ddl: Generate DDL for all tables in pipeline metadata
  -- This is the main function called by workflow.js
  -- ============================================================
  CREATE OR REPLACE FUNCTION generate_ddl(
    p_pipeline_metadata jsonb,
    p_default_schema text DEFAULT NULL,
    p_include_schema_creation boolean DEFAULT true
  ) RETURNS text LANGUAGE plpgsql AS
  $$
  DECLARE
    v_sql text := '';
    v_schema text;
    v_tables jsonb;
    v_table_count integer;
    v_table_ddl text;
    i integer;
  BEGIN
    -- Check if this is a single table definition or pipeline metadata
    IF p_pipeline_metadata ? 'name' AND NOT (p_pipeline_metadata ? 'tables') THEN
      -- This is a single table definition, convert to pipeline format
      v_schema := p_default_schema;
      IF p_include_schema_creation AND v_schema IS NOT NULL THEN
        v_sql := v_sql || format('CREATE SCHEMA IF NOT EXISTS %I;', v_schema) || E'\n\n';
      END IF;

      -- Generate DDL for single table
      v_sql := v_sql || format('-- Table %s', p_pipeline_metadata->>'name') || E'\n';
      v_sql := v_sql || generate_table_ddl(p_pipeline_metadata, v_schema, false);

      RETURN v_sql;
    END IF;

    -- Get schema from pipeline or use default
    v_schema := COALESCE(p_pipeline_metadata->>'schema', p_default_schema);
    v_tables := p_pipeline_metadata->'tables';

    IF v_tables IS NULL OR jsonb_typeof(v_tables) != 'array' THEN
      RAISE EXCEPTION 'Pipeline metadata must contain a "tables" array';
    END IF;

    -- Add schema creation if requested
    IF p_include_schema_creation AND v_schema IS NOT NULL THEN
      v_sql := v_sql || generate_schema_creation_sql_from_pipeline(p_pipeline_metadata) || E'\n\n';
    END IF;

    -- Generate DDL for each table
    v_table_count := get_table_count_from_pipeline(p_pipeline_metadata);

    FOR i IN 0..(v_table_count - 1) LOOP
      -- Generate DDL for this table (schema creation handled above)
      v_table_ddl := generate_ddl_for_table_by_index(p_pipeline_metadata, i, false);

      IF i > 0 THEN
        v_sql := v_sql || E'\n\n';
      END IF;

      v_sql := v_sql || format('-- Table %s (%s of %s)',
        (get_table_entry_from_pipeline(p_pipeline_metadata, i)->>'name'),
        i + 1,
        v_table_count
      ) || E'\n';
      v_sql := v_sql || v_table_ddl;
    END LOOP;

    RETURN v_sql;
  END;
  $$;


