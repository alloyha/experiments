-- =====================================================
-- Euler Diagram - PASTE-SAFE Single File Version
-- Save this to a file and run: psql -f euler.sql
-- Or use: \i euler.sql from within psql
-- =====================================================

BEGIN;

-- Clean slate
DROP TABLE IF EXISTS input_sets CASCADE;
DROP MATERIALIZED VIEW IF EXISTS euler_diagram_cache CASCADE;
DROP TABLE IF EXISTS euler_changes CASCADE;

-- =====================================================
-- Table Setup
-- =====================================================

CREATE TABLE input_sets (
    set_key VARCHAR(50) NOT NULL,
    element VARCHAR(100) NOT NULL,
    PRIMARY KEY (set_key, element)
);

CREATE INDEX idx_element ON input_sets(element, set_key);
CREATE INDEX idx_set_key ON input_sets(set_key);

-- =====================================================
-- Materialized View (Cache)
-- =====================================================

CREATE MATERIALIZED VIEW euler_diagram_cache AS
WITH element_memberships AS (
    SELECT 
        element,
        ARRAY_AGG(set_key ORDER BY set_key) as member_sets
    FROM input_sets
    GROUP BY element
)
SELECT 
    ARRAY_TO_STRING(member_sets, ',') as euler_key,
    member_sets as sets_involved,
    ARRAY_AGG(element ORDER BY element) as elements,
    COUNT(*) as element_count
FROM element_memberships
GROUP BY member_sets;

CREATE UNIQUE INDEX idx_euler_cache_unique ON euler_diagram_cache(euler_key);
CREATE INDEX idx_euler_cache_sets ON euler_diagram_cache USING GIN (sets_involved);

-- =====================================================
-- Core Functions
-- =====================================================

CREATE OR REPLACE FUNCTION euler_diagram_fast()
RETURNS TABLE (euler_key TEXT, sets_involved VARCHAR[], elements VARCHAR[], element_count BIGINT)
LANGUAGE sql STABLE PARALLEL SAFE AS
$_$ SELECT euler_key, sets_involved, elements, element_count FROM euler_diagram_cache 
    ORDER BY ARRAY_LENGTH(sets_involved, 1) DESC, euler_key $_$;

CREATE OR REPLACE FUNCTION euler_boundaries_fast()
RETURNS TABLE (set_key VARCHAR, boundary_sets VARCHAR[])
LANGUAGE sql STABLE PARALLEL SAFE AS
$_$ WITH RECURSIVE set_list AS (
        SELECT DISTINCT set_key FROM input_sets
    ),
    set_intersections AS (
        SELECT DISTINCT s.set_key, UNNEST(edc.sets_involved) as neighbor_set
        FROM set_list s
        JOIN euler_diagram_cache edc ON s.set_key = ANY(edc.sets_involved)
        WHERE ARRAY_LENGTH(edc.sets_involved, 1) > 1
    )
    SELECT set_key, ARRAY_AGG(DISTINCT neighbor_set ORDER BY neighbor_set) as boundary_sets
    FROM set_intersections
    WHERE neighbor_set != set_key
    GROUP BY set_key
    ORDER BY set_key $_$;

CREATE OR REPLACE FUNCTION euler_match_fast(items VARCHAR[])
RETURNS TABLE (matched_set_key VARCHAR)
LANGUAGE sql STABLE PARALLEL SAFE AS
$_$ WITH set_elements AS (
        SELECT set_key, ARRAY_AGG(element) as elements
        FROM input_sets
        WHERE element = ANY(items)
        GROUP BY set_key
    )
    SELECT set_key as matched_set_key
    FROM set_elements
    WHERE elements @> items
    ORDER BY set_key $_$;

CREATE OR REPLACE FUNCTION refresh_euler_cache()
RETURNS VOID LANGUAGE sql AS
$_$ REFRESH MATERIALIZED VIEW CONCURRENTLY euler_diagram_cache $_$;

-- =====================================================
-- Analysis Functions
-- =====================================================

CREATE OR REPLACE FUNCTION analyze_euler_performance()
RETURNS TABLE (metric VARCHAR, value TEXT)
LANGUAGE sql STABLE AS
$_$ SELECT 'Total Elements'::VARCHAR, COUNT(DISTINCT element)::TEXT FROM input_sets
    UNION ALL SELECT 'Total Sets'::VARCHAR, COUNT(DISTINCT set_key)::TEXT FROM input_sets
    UNION ALL SELECT 'Total Rows'::VARCHAR, COUNT(*)::TEXT FROM input_sets
    UNION ALL SELECT 'Unique Combinations'::VARCHAR, COUNT(*)::TEXT FROM euler_diagram_cache
    UNION ALL SELECT 'Avg Elements per Combo'::VARCHAR, ROUND(AVG(element_count), 2)::TEXT FROM euler_diagram_cache
    UNION ALL SELECT 'Max Intersection Size'::VARCHAR, MAX(ARRAY_LENGTH(sets_involved, 1))::TEXT FROM euler_diagram_cache
    UNION ALL SELECT 'Cache Size'::VARCHAR, pg_size_pretty(pg_total_relation_size('euler_diagram_cache'))
    UNION ALL SELECT 'Table Size'::VARCHAR, pg_size_pretty(pg_total_relation_size('input_sets'))
    UNION ALL SELECT 'Index Size'::VARCHAR, pg_size_pretty(pg_indexes_size('input_sets')) $_$;

-- =====================================================
-- Benchmark Function (Simpler Version)
-- =====================================================

CREATE OR REPLACE FUNCTION benchmark_euler()
RETURNS TABLE (function_name VARCHAR, execution_time_ms NUMERIC)
LANGUAGE plpgsql AS
$_BENCH_$
DECLARE
    start_ts TIMESTAMP;
    end_ts TIMESTAMP;
BEGIN
    start_ts := clock_timestamp();
    PERFORM * FROM (
        WITH element_memberships AS (
            SELECT element, ARRAY_AGG(set_key ORDER BY set_key) as member_sets
            FROM input_sets GROUP BY element
        )
        SELECT member_sets FROM element_memberships GROUP BY member_sets
    ) x;
    end_ts := clock_timestamp();
    
    RETURN QUERY SELECT 'euler_diagram (no cache)'::VARCHAR, 
                        EXTRACT(MILLISECONDS FROM (end_ts - start_ts));
    
    start_ts := clock_timestamp();
    PERFORM * FROM euler_diagram_fast();
    end_ts := clock_timestamp();
    
    RETURN QUERY SELECT 'euler_diagram_fast (cached)'::VARCHAR,
                        EXTRACT(MILLISECONDS FROM (end_ts - start_ts));
END;
$_BENCH_$;

COMMIT;

-- =====================================================
-- Sample Data and Tests
-- =====================================================

BEGIN;

TRUNCATE input_sets CASCADE;

INSERT INTO input_sets VALUES 
    ('A', '1'), ('A', '2'), ('A', '3'), ('A', '4'),
    ('B', '2'), ('B', '3'), ('B', '5'), ('B', '6'),
    ('C', '3'), ('C', '4'), ('C', '6'), ('C', '7');

REFRESH MATERIALIZED VIEW euler_diagram_cache;

COMMIT;

-- =====================================================
-- Run Tests
-- =====================================================

SELECT '=== Euler Diagram ===' as test;
SELECT * FROM euler_diagram_fast();

SELECT '=== Boundaries ===' as test;
SELECT * FROM euler_boundaries_fast();

SELECT '=== Performance Analysis ===' as test;
SELECT * FROM analyze_euler_performance();

SELECT '=== Benchmark (Small Dataset) ===' as test;
SELECT * FROM benchmark_euler();

-- =====================================================
-- Large Dataset Test
-- =====================================================

BEGIN;

INSERT INTO input_sets
SELECT 'SET_' || (i % 10), 'ELEM_' || i
FROM generate_series(1, 100000) i;

REFRESH MATERIALIZED VIEW CONCURRENTLY euler_diagram_cache;

COMMIT;

SELECT '=== Benchmark (100K Elements) ===' as test;
SELECT * FROM benchmark_euler();

SELECT '=== Final Analysis ===' as test;
SELECT * FROM analyze_euler_performance();

-- =====================================================
-- Usage Examples
-- =====================================================

-- Query cache (fast)
-- SELECT * FROM euler_diagram_fast();

-- Find boundaries
-- SELECT * FROM euler_boundaries_fast();

-- Match sets containing elements
-- SELECT * FROM euler_match_fast(ARRAY['ELEM_100', 'ELEM_200']::VARCHAR[]);

-- Refresh after updates
-- SELECT refresh_euler_cache();

-- Check performance
-- SELECT * FROM analyze_euler_performance();
-- SELECT * FROM benchmark_euler();
