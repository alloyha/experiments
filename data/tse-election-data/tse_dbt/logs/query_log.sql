-- created_at: 2026-10-08T16:56:24.830676742+00:00
-- finished_at: 2026-10-08T16:56:24.848078096+00:00
-- elapsed: 17ms
-- outcome: success
-- dialect: duckdb
-- node_id: not available
-- query_id: not available
-- desc: list_relations_in_parallel
SELECT table_catalog, table_schema, table_name, table_type FROM information_schema.tables WHERE table_schema = 'main' AND lower(table_catalog) = lower('tse_analytics');
