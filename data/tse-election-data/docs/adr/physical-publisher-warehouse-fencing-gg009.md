# GG-009 — Standalone physical publication uses warehouse-first locking

Status: proposed; deterministic DuckDB tests required in WSL.

The public `publish_partition` entrypoint acquires the same adjacent
`<DuckDB>.build.lock` as dbt/Gold readers **before** reading the active
partition generation or connecting to DuckDB. It then acquires the existing
partition `.publish.lock`, retaining immutable Parquet generations, staged
verification, atomic symlink publication, and generation CAS.

The CTAS coordinator already owns the warehouse lock for its complete build
and publication transaction. It borrows that lock through an existing open FD,
which the publisher verifies refers to this exact warehouse's lock inode.
Reacquiring the same `flock` through a second FD would self-deadlock. The
borrowed FD is an internal cooperative API, **not** an authentication boundary:
external writers must not invent one. There is no new global lock.

`--warehouse-lock-timeout` governs the warehouse gate and `--lock-timeout`
governs the per-partition gate. Existing caller APIs and tests are preserved.

Scope exclusions: uncoordinated DuckDB clients and destructive maintenance
(`clean-warehouse`) remain out of contract. GG-009 is not yet promoted; GG-012
restart / publication uncertainty is separately pending.
