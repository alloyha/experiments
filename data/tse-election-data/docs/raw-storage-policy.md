# Raw storage and cache policy

The local lake separates durable truth from regenerable cache.

## Durable

- `_metadata/ingest_manifest.jsonl`: append-only ingestion audit trail.
- `_metadata/resource_state.json`: latest control-plane state per resource.
- immutable `source/*` objects under a SHA-256 directory.
- `prepared/*.parquet` and their preparation metadata when produced.
- `manifest.json` beside an immutable source version.
- warehouse candidate fact partitions.
- the DuckDB warehouse when used as a persisted analytical snapshot.

## Active index

`_metadata/current_objects.jsonl` is the authoritative index of **physically
available active tabular objects**. Every `source_object` and `object` referenced
by this file must exist.

The index is derived control-plane state. It may drop a row when a regenerable
extracted object is pruned.

## Regenerable cache

- `extracted/*`
- `tse_dbt/target/*`
- `tse_dbt/logs/*`
- `tse_dbt/dbt_packages/*`
- orphaned DuckDB temporary storage

`prune_extracted_raw.py --apply` may remove extracted objects even when they were
active, but it atomically removes the corresponding rows from
`current_objects.jsonl`. The immutable source remains.

On a later ingest, if CKAN metadata is unchanged and the immutable source object
still exists, the ingestor rehydrates the missing extracted representation from
that local source without a network download.

## SHA history

Immutable SHA directories are never overwritten. Superseded SHA versions may be
pruned only through an explicit retention operation; the append-only ingestion
manifest remains the audit record.
