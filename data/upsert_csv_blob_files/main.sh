#!/usr/bin/env bash
set -euo pipefail

# ---------- CONFIG ----------
CSV_PATTERN="2025-09/*EMPRE*"
TABLE_NAME="empresa"
SCHEMA_NAME="public"
FULL_TABLE_NAME="${SCHEMA_NAME}.${TABLE_NAME}"
POSTGRES_URL="postgresql://postgres:postgres@localhost:5432/postgres"

# CSV Format
DELIMITER=";"       # Delimiter character
HAS_HEADER=false    # Does CSV have header row?

# Provide PK columns by numeric index (1-based). Empty = no PK
PK_NUMS="1"

# Optional: enable pv for monitoring
USE_PV=true

# Profiling output
PROFILE_LOG="${TABLE_NAME}_profile_$(date +%Y%m%d_%H%M%S).log"

# ---------- Profiling Setup ----------
exec 3>&1 4>&2
exec 1> >(tee -a "$PROFILE_LOG")
exec 2>&1

PROFILE_START=$(date +%s)
declare -A TIMINGS

profile_start() {
    local step_name=$1
    TIMINGS["${step_name}_start"]=$(date +%s.%N)
    echo "[PROFILE] Starting: $step_name at $(date +"%Y-%m-%d %H:%M:%S")"
}

profile_end() {
    local step_name=$1
    TIMINGS["${step_name}_end"]=$(date +%s.%N)
    local duration=$(echo "${TIMINGS[${step_name}_end]} - ${TIMINGS[${step_name}_start]}" | bc)
    TIMINGS["${step_name}_duration"]=$duration
    printf "[PROFILE] Completed: %s in %.2f seconds\n" "$step_name" "$duration"
}

# ---------- Helpers ----------
ts() { date +"%Y-%m-%d %H:%M:%S"; }
join_by() { local IFS="$1"; shift; echo "$*"; }

# ---------- Step 0: Detect columns ----------
profile_start "detect_columns"
echo "$(ts) Detecting columns from CSV..."
echo "$(ts) Using delimiter: '$DELIMITER'"

# Debug: Show actual file content
echo "$(ts) First 3 lines of CSV (raw):"
head -3 $(ls $CSV_PATTERN | head -1) || echo "Could not read file"
echo "---"

# First, let's peek at the first line to debug
echo "$(ts) Sample first row (parsed by DuckDB):"
duckdb -csv -noheader -c "
SELECT * FROM read_csv_auto('$CSV_PATTERN', 
    delim='$DELIMITER',
    header=$HAS_HEADER,
    sample_size=1
) LIMIT 1;"

# Get column count using DESCRIBE
echo "$(ts) Detecting column count..."
NUM_COLS=$(duckdb -csv -noheader -c "
DESCRIBE SELECT * FROM read_csv_auto('$CSV_PATTERN', 
    delim='$DELIMITER',
    header=$HAS_HEADER,
    sample_size=10000
);" | wc -l | tr -d '[:space:]')

if [ -z "$NUM_COLS" ] || [ "$NUM_COLS" -le 0 ]; then
    echo "ERROR: Cannot detect number of columns"
    exit 1
fi

# Synthetic column names: c1,c2,...,cN
COLUMN_LIST=$(seq -s, 1 "$NUM_COLS" | sed 's/\([0-9]\+\)/c\1/g')
IFS=',' read -r -a COL_ARR <<< "$COLUMN_LIST"

# Map numeric PK indices to column names
if [ -n "$PK_NUMS" ]; then
    IFS=',' read -r -a PK_IDX <<< "$PK_NUMS"
    PK_COLUMNS=$(for i in "${PK_IDX[@]}"; do echo -n "${COL_ARR[$((i-1))]},"; done | sed 's/,$//')
else
    PK_COLUMNS=""
fi

echo "$(ts) Detected $NUM_COLS columns: $COLUMN_LIST"
[ -n "$PK_COLUMNS" ] && echo "$(ts) PK columns (indices $PK_NUMS): $PK_COLUMNS"
profile_end "detect_columns"

# ---------- Step 1: Create target table ----------
profile_start "create_table"
echo "$(ts) Creating target table if not exists..."

col_defs=$(printf '"%s" TEXT,' "${COL_ARR[@]}")
col_defs=${col_defs%,}  # remove trailing comma
PK_SQL=""
[ -n "$PK_COLUMNS" ] && PK_SQL=", PRIMARY KEY ($PK_COLUMNS)"

DDL="CREATE SCHEMA IF NOT EXISTS ${SCHEMA_NAME};
CREATE TABLE IF NOT EXISTS ${FULL_TABLE_NAME} ($col_defs$PK_SQL);"

echo "$DDL"
psql "$POSTGRES_URL" -v ON_ERROR_STOP=1 -f <(echo "$DDL")
profile_end "create_table"

# ---------- Step 2: Session tuning ----------
profile_start "session_tuning"
echo "$(ts) Optimizing PostgreSQL session..."
psql "$POSTGRES_URL" -v ON_ERROR_STOP=1 <<'PSQL_TUNE'
SET synchronous_commit = OFF;
SET maintenance_work_mem = '1GB';
SET work_mem = '128MB';
PSQL_TUNE
profile_end "session_tuning"

# ---------- Step 3: Count rows ----------
profile_start "count_rows"
echo "$(ts) Counting rows in CSV..."
TOTAL_ROWS=$(duckdb -csv -noheader -c "
SELECT COUNT(*) FROM read_csv_auto('$CSV_PATTERN', 
    delim='$DELIMITER',
    header=$HAS_HEADER,
    ignore_errors=true,
    parallel=true
);" | tail -1 | tr -d '[:space:]')
echo "$(ts) Total rows (approx): $TOTAL_ROWS"
profile_end "count_rows"

# ---------- Step 4: Create staging table ----------
profile_start "create_staging"
STAGING_TABLE="${FULL_TABLE_NAME}_staging_$(date +%s%N)"
echo "$(ts) Creating staging table: $STAGING_TABLE"

STG_COLS=$(printf '"%s" TEXT,' "${COL_ARR[@]}")
STG_COLS=${STG_COLS%,}
STG_DDL="CREATE UNLOGGED TABLE $STAGING_TABLE ($STG_COLS);"
psql "$POSTGRES_URL" -v ON_ERROR_STOP=1 -c "$STG_DDL"
profile_end "create_staging"

# ---------- Step 5: Load CSV into staging ----------
profile_start "load_staging"
echo "$(ts) Streaming data from CSV into staging..."

LOAD_CMD="COPY $STAGING_TABLE($COLUMN_LIST) FROM STDIN WITH (FORMAT CSV, HEADER FALSE, DELIMITER '$DELIMITER');"

if [ "$USE_PV" = true ] && command -v pv &> /dev/null; then
    duckdb -c "
    COPY (
        SELECT * FROM read_csv_auto('$CSV_PATTERN', 
            delim='$DELIMITER', 
            header=$HAS_HEADER, 
            ignore_errors=true,
            parallel=true
        )
    ) TO STDOUT (FORMAT CSV, HEADER FALSE, DELIMITER '$DELIMITER');" \
      | pv -l -s "$TOTAL_ROWS" -N "CSV->PG" \
      | tr -d '\000' \
      | psql "$POSTGRES_URL" -v ON_ERROR_STOP=1 -c "$LOAD_CMD"
else
    duckdb -c "
    COPY (
        SELECT * FROM read_csv_auto('$CSV_PATTERN', 
            delim='$DELIMITER', 
            header=$HAS_HEADER, 
            ignore_errors=true,
            parallel=true
        )
    ) TO STDOUT (FORMAT CSV, HEADER FALSE, DELIMITER '$DELIMITER');" \
      | tr -d '\000' \
      | psql "$POSTGRES_URL" -v ON_ERROR_STOP=1 -c "$LOAD_CMD"
fi

STAGING_ROWS=$(psql "$POSTGRES_URL" -t -A -c "SELECT COUNT(*) FROM $STAGING_TABLE;" | tr -d '[:space:]')
echo "$(ts) Staging rows: $STAGING_ROWS"
profile_end "load_staging"

# ---------- Step 6: Run upsert ----------
profile_start "upsert"
if [ -n "$PK_COLUMNS" ]; then
    echo "$(ts) Running upsert into $FULL_TABLE_NAME..."
    
    # Check for duplicates in staging
    DUPLICATE_COUNT=$(psql "$POSTGRES_URL" -t -A -c "
        SELECT COUNT(*) - COUNT(DISTINCT $PK_COLUMNS) 
        FROM $STAGING_TABLE
    " | tr -d '[:space:]')
    
    if [ "$DUPLICATE_COUNT" -gt 0 ]; then
    echo "$(ts) WARNING: Found $DUPLICATE_COUNT duplicate PK values in staging data"
    echo "$(ts) Deduplicating staging table (keeping last occurrence)..."
    
    # Create deduplicated temp table
    DEDUP_TABLE="dedup_$(date +%s%N)"  # No schema prefix
    psql "$POSTGRES_URL" -v ON_ERROR_STOP=1 <<DEDUP_SQL
CREATE TEMP TABLE $DEDUP_TABLE AS
SELECT DISTINCT ON ($PK_COLUMNS) *
FROM $STAGING_TABLE
ORDER BY $PK_COLUMNS;
DEDUP_SQL
    
        # IMPORTANT: Set the source for upsert
        UPSERT_SOURCE=$DEDUP_TABLE
        
        # Count rows in the TEMP table (no schema prefix)
        DEDUP_ROWS=$(psql "$POSTGRES_URL" -t -A -c "SELECT COUNT(*) FROM $DEDUP_TABLE;" | tr -d '[:space:]')
        echo "$(ts) After deduplication: $DEDUP_ROWS unique rows (removed $DUPLICATE_COUNT duplicates)"
    else
        UPSERT_SOURCE=$STAGING_TABLE
        DEDUP_ROWS=$STAGING_ROWS
    fi
        
    # Count existing rows that will be updated
    EXISTING_ROWS=$(psql "$POSTGRES_URL" -t -A -c "
        SELECT COUNT(*) FROM $FULL_TABLE_NAME t 
        WHERE EXISTS (SELECT 1 FROM $UPSERT_SOURCE s WHERE s.$PK_COLUMNS = t.$PK_COLUMNS)
    " | tr -d '[:space:]')
    
    NEW_ROWS=$((DEDUP_ROWS - EXISTING_ROWS))
    
    echo "$(ts) Rows to INSERT: $NEW_ROWS"
    echo "$(ts) Rows to UPDATE: $EXISTING_ROWS"
    
    # Build UPDATE SET clause excluding PK columns
    UPDATE_SET=""
    for c in "${COL_ARR[@]}"; do
        # Check if column is NOT in PK_COLUMNS
        if [[ ",$PK_COLUMNS," != *",$c,"* ]]; then
            UPDATE_SET="${UPDATE_SET}\"$c\" = EXCLUDED.\"$c\","
        fi
    done
    UPDATE_SET=${UPDATE_SET%,}  # remove trailing comma
    
    UPSERT_SQL="
    INSERT INTO $FULL_TABLE_NAME($COLUMN_LIST)
    SELECT $COLUMN_LIST FROM $UPSERT_SOURCE
    ON CONFLICT ($PK_COLUMNS) DO UPDATE SET $UPDATE_SET;"
    
    psql "$POSTGRES_URL" -v ON_ERROR_STOP=1 -c "$UPSERT_SQL"
    
    echo "$(ts) Upsert complete: $NEW_ROWS inserted, $EXISTING_ROWS updated"
else
    echo "$(ts) No PK defined; simple insert..."
    psql "$POSTGRES_URL" -v ON_ERROR_STOP=1 -c "INSERT INTO $FULL_TABLE_NAME SELECT * FROM $STAGING_TABLE;"
    echo "$(ts) Insert complete: $STAGING_ROWS rows"
fi
profile_end "upsert"

# ---------- Step 7: Post-load finalize ----------
profile_start "finalize"
echo "$(ts) Running VACUUM ANALYZE..."
psql "$POSTGRES_URL" -v ON_ERROR_STOP=1 <<PSQL_FINAL
ALTER TABLE $FULL_TABLE_NAME SET (autovacuum_enabled = true);
VACUUM ANALYZE $FULL_TABLE_NAME;
PSQL_FINAL
profile_end "finalize"

# ---------- Step 8: Verification ----------
profile_start "verification"
LOADED_ROWS=$(psql "$POSTGRES_URL" -t -A -c "SELECT COUNT(*) FROM $FULL_TABLE_NAME;" | tr -d '[:space:]')
echo "$(ts) Loaded rows in target table: $LOADED_ROWS (expected approx: $TOTAL_ROWS)"

# Check for data loss
if [ "$LOADED_ROWS" -lt "$TOTAL_ROWS" ]; then
    DIFF=$((TOTAL_ROWS - LOADED_ROWS))
    echo "WARNING: Missing $DIFF rows ($(awk "BEGIN {printf \"%.2f\", ($DIFF/$TOTAL_ROWS)*100}")%)"
fi
profile_end "verification"

# ---------- Step 9: Drop staging ----------
profile_start "cleanup"
echo "$(ts) Dropping staging table $STAGING_TABLE ..."
psql "$POSTGRES_URL" -v ON_ERROR_STOP=1 -c "DROP TABLE $STAGING_TABLE;"
profile_end "cleanup"

# ---------- Profile Summary ----------
PROFILE_END=$(date +%s)
TOTAL_DURATION=$((PROFILE_END - PROFILE_START))

echo ""
echo "========================================="
echo "           PROFILE SUMMARY"
echo "========================================="
printf "%-25s %10s %8s\n" "Step" "Duration" "% Total"
echo "-----------------------------------------"

for step in detect_columns create_table session_tuning count_rows create_staging load_staging upsert finalize verification cleanup; do
    if [ -n "${TIMINGS[${step}_duration]:-}" ]; then
        duration=${TIMINGS[${step}_duration]}
        percent=$(awk "BEGIN {printf \"%.1f\", ($duration/$TOTAL_DURATION)*100}")
        printf "%-25s %8.2fs %7s%%\n" "$step" "$duration" "$percent"
    fi
done

echo "-----------------------------------------"
printf "%-25s %8ds\n" "TOTAL" "$TOTAL_DURATION"
echo "========================================="
echo ""
echo "Throughput: $(awk "BEGIN {printf \"%.0f\", $LOADED_ROWS/$TOTAL_DURATION}") rows/sec"
echo "Profile log: $PROFILE_LOG"
echo ""

exit 0
