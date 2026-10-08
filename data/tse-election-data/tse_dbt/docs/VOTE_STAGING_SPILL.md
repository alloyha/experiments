# Vote staging spill strategy

The first OOM-safe attempt used `arg_max()` directly while scanning the 2018
candidate-vote CSV. At this grain the number of hash groups is close to the
number of rows, so the aggregate hash table itself exceeded the 2 GB DuckDB
memory budget.

The pipeline is now deliberately two-stage:

1. `bronze_candidate_votes_raw`
   - reads the multi-GiB CSV once;
   - projects only keys/measures required downstream;
   - persists all source versions in DuckDB;
   - no global hash aggregation or sort.

2. `silver_candidate_votes_munzona`
   - reads the narrow persisted table;
   - uses `row_number()` by business grain ordered by generation timestamp;
   - DuckDB may spill the sort to `data/warehouse/.duckdb_tmp`;
   - retains only the latest published version.

This trades temporary disk space for bounded memory.

The local profile remains conservative:
- dbt threads = 1
- DuckDB threads = 1
- memory_limit = 2GB
- preserve_insertion_order = false
- max temp spill = 50GB
