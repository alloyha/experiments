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

CREATE OR REPLACE FUNCTION refresh_euler_cache()
RETURNS VOID LANGUAGE sql AS
$_$ REFRESH MATERIALIZED VIEW euler_diagram_cache $_$;

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
$_$
WITH RECURSIVE set_list AS (
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
ORDER BY set_key
$_$;


CREATE OR REPLACE FUNCTION euler_match_fast(items VARCHAR[])
RETURNS TABLE (matched_set_key VARCHAR)
LANGUAGE sql STABLE PARALLEL SAFE AS
$$
SELECT s.set_key
FROM input_sets s
WHERE s.element = ANY(items)
GROUP BY s.set_key
HAVING COUNT(DISTINCT s.element) = array_length(items,1)
ORDER BY s.set_key;
$$;

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

-- =====================================================
-- Data Generation Functions
-- =====================================================

-- Function: Generate realistic test data with overlaps
CREATE OR REPLACE FUNCTION generate_test_data(
    num_elements INT DEFAULT 100000,
    num_sets INT DEFAULT 10,
    overlap_probability NUMERIC DEFAULT 0.30
)
RETURNS VOID
LANGUAGE plpgsql AS
$_GEN_$
BEGIN
    TRUNCATE input_sets CASCADE;
    
    -- Generate with controlled randomness using hash-based distribution
    INSERT INTO input_sets
    SELECT 
        'SET_' || set_num,
        'ELEM_' || elem_id
    FROM 
        generate_series(1, num_elements) elem_id,
        generate_series(0, num_sets - 1) set_num
    WHERE 
        -- Deterministic but pseudo-random assignment
        -- Each element has overlap_probability chance to be in each set
        (hashtext('ELEM_' || elem_id || '_SET_' || set_num)::BIGINT % 100) < (overlap_probability * 100)
    ON CONFLICT DO NOTHING;
    
    REFRESH MATERIALIZED VIEW CONCURRENTLY euler_diagram_cache;
    
    RAISE NOTICE 'Generated % rows for % unique elements across % sets', 
        (SELECT COUNT(*) FROM input_sets),
        (SELECT COUNT(DISTINCT element) FROM input_sets),
        num_sets;
END;
$_GEN_$;

-- Function: Generate data with specific overlap patterns
CREATE OR REPLACE FUNCTION generate_overlap_patterns(
    num_elements INT DEFAULT 10000,
    pattern VARCHAR DEFAULT 'mixed'
)
RETURNS VOID
LANGUAGE plpgsql AS
$_PATTERN_$
BEGIN
    TRUNCATE input_sets CASCADE;
    
    IF pattern = 'high_overlap' THEN
        -- 70% of elements in multiple sets
        INSERT INTO input_sets
        SELECT 'SET_' || set_num, 'ELEM_' || elem_id
        FROM generate_series(1, num_elements) elem_id,
             generate_series(0, 9) set_num
        WHERE (hashtext('ELEM_' || elem_id || '_SET_' || set_num)::BIGINT % 100) < 70
        ON CONFLICT DO NOTHING;
        
    ELSIF pattern = 'low_overlap' THEN
        -- 10% of elements in multiple sets
        INSERT INTO input_sets
        SELECT 'SET_' || set_num, 'ELEM_' || elem_id
        FROM generate_series(1, num_elements) elem_id,
             generate_series(0, 9) set_num
        WHERE (hashtext('ELEM_' || elem_id || '_SET_' || set_num)::BIGINT % 100) < 10
        ON CONFLICT DO NOTHING;
        
    ELSIF pattern = 'chain' THEN
        -- Chain pattern: A∩B, B∩C, C∩D, etc.
        INSERT INTO input_sets
        SELECT 'SET_' || (elem_id % 10), 'ELEM_' || elem_id
        FROM generate_series(1, num_elements) elem_id
        UNION ALL
        SELECT 'SET_' || ((elem_id + 1) % 10), 'ELEM_' || elem_id
        FROM generate_series(1, num_elements) elem_id
        WHERE elem_id % 3 = 0  -- 33% also in next set
        ON CONFLICT DO NOTHING;
        
    ELSE  -- 'mixed' (default)
        -- Realistic mix: some exclusive, some in 2-3 sets, few in many
        INSERT INTO input_sets
        SELECT 'SET_' || set_num, 'ELEM_' || elem_id
        FROM generate_series(1, num_elements) elem_id,
             generate_series(0, 9) set_num
        WHERE (hashtext('ELEM_' || elem_id || '_SET_' || set_num)::BIGINT % 100) < 30
        ON CONFLICT DO NOTHING;
    END IF;
    
    REFRESH MATERIALIZED VIEW CONCURRENTLY euler_diagram_cache;
    
    RAISE NOTICE 'Generated % pattern with % rows for % unique elements', 
        pattern,
        (SELECT COUNT(*) FROM input_sets),
        (SELECT COUNT(DISTINCT element) FROM input_sets);
END;
$_PATTERN_$;

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
-- Large Dataset Test with Realistic Overlaps
-- =====================================================

BEGIN;

-- Strategy: Generate overlapping sets with controlled randomness
-- Each element appears in 1-3 sets on average

-- Method 1: Random assignment with overlaps
INSERT INTO input_sets
SELECT 
    'SET_' || set_num,
    'ELEM_' || elem_id
FROM 
    generate_series(1, 100000) elem_id,
    generate_series(0, 9) set_num
WHERE 
    -- Each element has ~30% chance to be in each set
    -- This creates realistic overlaps
    (hashtext('ELEM_' || elem_id || '_SET_' || set_num)::BIGINT % 100) < 30
ON CONFLICT DO NOTHING;

-- Alternative Method 2: Controlled overlap patterns (commented out)
-- Uncomment to use this instead:

-- INSERT INTO input_sets
-- SELECT 
--     'SET_' || (
--         CASE 
--             WHEN random() < 0.5 THEN (elem_id % 10)::TEXT
--             WHEN random() < 0.75 THEN ((elem_id % 10) || ',' || ((elem_id + 1) % 10))::TEXT
--             ELSE ((elem_id % 10) || ',' || ((elem_id + 2) % 10))::TEXT
--         END
--     ),
--     'ELEM_' || elem_id
-- FROM generate_series(1, 100000) elem_id;

COMMIT;

REFRESH MATERIALIZED VIEW CONCURRENTLY euler_diagram_cache;

-- Verify overlap statistics
SELECT '=== Overlap Statistics ===' as test;
SELECT 
    'Elements in 1 set' as category,
    COUNT(*) as count
FROM (
    SELECT element 
    FROM input_sets 
    GROUP BY element 
    HAVING COUNT(DISTINCT set_key) = 1
) x
UNION ALL
SELECT 
    'Elements in 2 sets' as category,
    COUNT(*) as count
FROM (
    SELECT element 
    FROM input_sets 
    GROUP BY element 
    HAVING COUNT(DISTINCT set_key) = 2
) x
UNION ALL
SELECT 
    'Elements in 3+ sets' as category,
    COUNT(*) as count
FROM (
    SELECT element 
    FROM input_sets 
    GROUP BY element 
    HAVING COUNT(DISTINCT set_key) >= 3
) x
UNION ALL
SELECT 
    'Total unique elements' as category,
    COUNT(DISTINCT element) as count
FROM input_sets
UNION ALL
SELECT 
    'Total rows (with duplicates across sets)' as category,
    COUNT(*) as count
FROM input_sets;

SELECT '=== Benchmark (100K Elements) ===' as test;
SELECT * FROM benchmark_euler();

-- Show overlap statistics
SELECT '=== Overlap Statistics ===' as test;
SELECT 
    sets_in_combo,
    COUNT(*) as num_combinations,
    SUM(element_count) as total_elements
FROM (
    SELECT 
        ARRAY_LENGTH(sets_involved, 1) as sets_in_combo,
        element_count
    FROM euler_diagram_cache
) x
GROUP BY sets_in_combo
ORDER BY sets_in_combo;

SELECT '=== Sample Intersections ===' as test;
SELECT 
    euler_key,
    element_count,
    CASE 
        WHEN element_count > 10 THEN (elements[1:3] || ARRAY['...'])::VARCHAR[]
        ELSE elements
    END as sample_elements
FROM euler_diagram_cache
WHERE ARRAY_LENGTH(sets_involved, 1) > 1
ORDER BY element_count DESC
LIMIT 10;

SELECT '=== Final Analysis ===' as test;
SELECT * FROM analyze_euler_performance();

-- =====================================================
-- Usage Examples
-- =====================================================

-- Example 1: Generate realistic test data (30% overlap)
-- SELECT generate_test_data(100000, 10, 0.30);

-- Example 2: Generate high overlap scenario (70% overlap)
-- SELECT generate_test_data(50000, 8, 0.70);

-- Example 3: Generate specific patterns
-- SELECT generate_overlap_patterns(10000, 'high_overlap');
-- SELECT generate_overlap_patterns(10000, 'low_overlap');
-- SELECT generate_overlap_patterns(10000, 'chain');
-- SELECT generate_overlap_patterns(10000, 'mixed');

-- Example 4: Custom data with specific overlap structure
BEGIN;
TRUNCATE input_sets CASCADE;

-- All elements in set A
INSERT INTO input_sets SELECT 'A', 'ELEM_' || i FROM generate_series(1, 1000) i;

-- 50% overlap with B
INSERT INTO input_sets SELECT 'B', 'ELEM_' || i FROM generate_series(1, 500) i
UNION ALL SELECT 'B', 'ELEM_' || i FROM generate_series(1001, 1500) i;

-- 25% overlap with both A and B
INSERT INTO input_sets SELECT 'C', 'ELEM_' || i FROM generate_series(1, 250) i
UNION ALL SELECT 'C', 'ELEM_' || i FROM generate_series(2001, 2750) i;

REFRESH MATERIALIZED VIEW CONCURRENTLY euler_diagram_cache;
COMMIT;

-- Example 5: Query cache (fast)
SELECT * FROM euler_diagram_fast();

-- Example 6: Find boundaries
SELECT * FROM euler_boundaries_fast();

-- Example 7: Match sets containing elements
SELECT euler_key, array_length(elements,1) 
FROM euler_diagram_cache 
WHERE elements @> ARRAY['ELEM_100', 'ELEM_200']::VARCHAR[];

-- Example 8: Refresh after updates
SELECT refresh_euler_cache();

-- Example 9: Performance analysis
SELECT * FROM analyze_euler_performance();
SELECT * FROM benchmark_euler();

-- Example 10: Test different scales
-- SELECT generate_test_data(1000, 5, 0.20);    -- Small: 1K elements, 5 sets, 20% overlap
-- SELECT generate_test_data(50000, 10, 0.40);  -- Medium: 50K elements, 10 sets, 40% overlap
-- SELECT generate_test_data(500000, 20, 0.15); -- Large: 500K elements, 20 sets, 15% overlap

-- =====================================================
-- Performance Comparison Script
-- =====================================================

-- Comprehensive performance tests:

DO $do$
DECLARE
    scenarios TEXT[] := ARRAY['low_overlap', 'mixed', 'high_overlap'];
    scenario TEXT;
    before_time NUMERIC;
    after_time NUMERIC;
BEGIN
    FOREACH scenario IN ARRAY scenarios LOOP
        RAISE NOTICE '=== Testing % scenario ===', scenario;
        
        PERFORM generate_overlap_patterns(10000, scenario);

        SELECT execution_time_ms INTO before_time
        FROM benchmark_euler()
        WHERE function_name = 'euler_diagram (no cache)'
        LIMIT 1;

        SELECT execution_time_ms INTO after_time
        FROM benchmark_euler()
        WHERE function_name = 'euler_diagram_fast (cached)'
        LIMIT 1;

        RAISE NOTICE 'No cache: % ms, Cached: % ms, Speedup: %',
            before_time, after_time,
            CASE WHEN after_time IS NULL OR after_time = 0 THEN NULL
                 ELSE ROUND(before_time / after_time::NUMERIC, 1) END;
    END LOOP;
END
$do$ LANGUAGE plpgsql;
