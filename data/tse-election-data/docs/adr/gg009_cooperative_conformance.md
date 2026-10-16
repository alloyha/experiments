# GG-009 — Cooperative concurrency conformance (bounded certification)

**Scope:** implemented for the cooperating TSE project entrypoints below on
POSIX-local filesystems supporting advisory `flock`, atomic rename and `fsync`.
This **does not** certify arbitrary DuckDB clients, external processes which
ignore advisory locks, NFS/S3 semantics, or a production-ready graph service.
GG-012 crash-uncertain publication and restart equivalence remain independent
critical requirements.

## Existing authoritative protocol

1. The stable adjacent `<warehouse>.build.lock` is acquired before opening or
   changing the DuckDB warehouse. Read consumers hold SH; cooperating writers,
   certifiers and maintenance hold EX.
2. When a physical partition is involved, `<partition>/.publish.lock` follows
   the warehouse lock (never the reverse). Multi-cycle audits acquire all
   partitions in deterministic sorted cycle order. Locks span the entire
   freshness/coverage check and certificate consumption.
3. Coordinated dbt subprocesses inherit the writer-lock FD via `pass_fds`.
   A `SIGKILL` of the coordinator is not a release while the child still lives.
4. Physical publication uses immutable generations, generation compare-and-swap
   and atomic pointer replacement; cleanup durably rotates the warehouse
   lifecycle epoch before destructive action.
5. Certificates are scoped to warehouse path, lifecycle epoch, source root,
   cycle, inputs and derived output fingerprints; stale or absent proof denies
   the guarded reader. These fingerprints detect changes; they are not signed
   authenticity or cryptographic proofs of all Gold row content.

## Conformance matrix (real existing regression tests)

| Surface | Coordination or authority boundary | Verification |
|---|---|---|
| dbt writer gateway | EX warehouse lock; child inherits FD; isolated dbt artifacts | `test_coordinated_dbt_gateway.py`, `test_graph_dbt_writer_handoff.py` |
| Cube certifier | EX warehouse lock through revoke, snapshots, dbt run/test, reissue | `test_candidate_cube_writer_fencing.py` |
| Standalone physical publisher | EX warehouse then EX partition; immutable generation/CAS | `test_physical_publisher_warehouse_fencing.py` |
| Gold graph issuer | EX warehouse then EX selected partition; revoke before certify | `test_graph_guarded_read.py`, `test_graph_freshness.py` |
| Gold graph guarded consumer | SH warehouse then SH partition through pre/post revalidation | `test_graph_guarded_read.py`, `test_graph_dbt_writer_handoff.py` |
| Multi-cycle candidate audit | SH warehouse then SH all published partitions through complete freshness audit | `test_candidate_cube_guarded_audit.py` |
| Warehouse maintenance | EX stable warehouse lock; rotate durable epoch before unlink | `test_warehouse_maintenance_lifecycle.py` |
| Failure of coordinator/readers | POSIX file lock released on death unless inherited by live writer | `test_coordinated_dbt_gateway.py`, `test_graph_guarded_read.py` |
| Independent warehouses | Warehouse locks scoped to each DuckDB, no global lock | `test_physical_publisher_warehouse_fencing.py`, `test_candidate_cube_guarded_audit.py` |
| External, non-cooperating DuckDB clients | **Outside GG-009 guarantee** | No claim of protection |

`graph_freshness.py audit` and the cycle-scoped graph SQL diagnostic compute
read-only evidence but **do not provide a concurrent authorization boundary**.
They may be used for inspection, never to serve an authorized graph result.
Use `graph_guarded_read.py::read_graph` for that contract. The reporter itself
labels the response `cooperatively_guarded_not_production_certified`.

## Falsifiable CI condition

`make unit` runs the real dbt handoff test, two-process lock tests, publisher,
issuer, reader, cube audit and cleanup regression suites. The registry points
to `read_graph` and binds a positive read to the real dbt-mutation falsifier.
`test_gg009_conformance_registry.py` checks that each row's executable
regression is still present: it does **not** replace running the tests.

Previous isolated worktree evidence: `452 tests ... OK`, invariant registry PASS,
Medallion DAG PASS, and PR #61 CI successful. **This conformance registry
amendment requires a new full local run and green CI before integration.**
The `--strict` registry gate remains red because GG-012 is still planned.

## Exclusions and non-promises

- Do not treat the unguarded graph audit output as certified delivery.
- Do not infer filesystem-level isolation from advisory POSIX locks.
- Do not confuse GG-009 implementation status with the separate full domain
  promotion policy, which also requires GG-012 and real-cycle certification.
- Do not delete immutable generations in the presence of potential readers.
- Do not introduce a global lock or a legacy lock compatibility layer.
