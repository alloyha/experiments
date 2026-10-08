# Raw active-index idempotency

`current_objects.jsonl` may contain repeated records for the same active object.

A raw object must be scanned exactly once by dbt regardless of how many duplicate
index records exist.

The raw-reader therefore enforces two protections:

1. `current_raw_files()` returns distinct object paths.
2. `read_raw_csv()` joins against a distinct active index scoped to the selected
   election years and election types.

Without both protections, a duplicate index record can multiply data first in
the `read_csv([...])` path list and again in the metadata join.
