# OOM-safe local DuckDB execution

The 2018 build can scan several GiB of CSV. The local profile is intentionally conservative:

- dbt concurrency: 1 node
- DuckDB worker threads: 1
- DuckDB memory limit: 2GB by default
- insertion-order preservation: disabled so DuckDB can spill/reorder large loads

Override memory only after checking WSL RAM:

```bash
free -h
df -h .
```

Example for a WSL VM with enough headroom:

```bash
export TSE_DUCKDB_MEMORY_LIMIT=4GB
```

Keep the dbt/DuckDB thread count at 1 until the heavy 2018 models are stable.

The vote staging model no longer sorts the wide 3.8 GiB CSV with
`row_number() over (... order by generated_at)`. It projects a narrow payload
and resolves historical duplicates with `arg_max(..., generated_at)`.

The electorate municipality intermediate is now persisted so its large raw
aggregation executes once, rather than once for every downstream model.
