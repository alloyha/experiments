# fact_candidate_votes materialization

For DuckDB, `fact_candidate_votes` is rebuilt with CTAS as a table rather than
using `delete+insert`.

Measured on the 2018 candidate snapshot (~8.68M rows):

- clean CTAS rebuild: about 2 minutes
- delete+insert over the existing partition: more than 10 minutes

DuckDB does not benefit from deleting and reinserting the entire active snapshot
when a full CTAS is substantially cheaper. The fact remains the first physical
analytical materialization after the prepared Parquet-backed staging views.

If a future backend provides efficient native partition overwrite, that backend
can use a backend-specific incremental materialization instead of reusing the
DuckDB strategy.
