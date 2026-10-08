# Candidate raw materialization

`stg_candidate_votes_raw` is intentionally a **view** over the immutable
prepared Parquet object.

Rationale:

- the source CSV is immutable evidence;
- the prepared Parquet is the immutable technical representation;
- materializing the same ~8.7M rows again as a DuckDB staging table adds large
  write amplification without adding durable information;
- semantic aggregation/deduplication belongs downstream.

The first physical analytical materialization is
`stg_candidate_votes_munzona`.

Heavy physical-integrity checks against the prepared raw object are not part of
the normal dbt build and must be enabled explicitly.

## Canonical municipality validation

The full-data municipality canonicality assertion over `stg_candidate_votes_raw`
is intentionally tagged `integrity_heavy` and disabled in normal builds because
the model is a view over ~8.7M Parquet rows and a regex/length assertion forces a
complete scan.

Normal builds validate the deterministic transformation itself with
`assert_normalize_municipality_code_examples`. Full-data validation remains
available explicitly with `run_heavy_integrity_tests=true`.
