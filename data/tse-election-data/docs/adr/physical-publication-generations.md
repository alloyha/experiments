# ADR — Physical candidate-fact publication generations (PR36)

**Status:** Proposed (local gates passed; pending PR review).

## Decision

For each `(election_type, year)` partition, the external Gold interface remains
`election_type=<type>/year=<year>/data.parquet`. On a POSIX filesystem,
`data.parquet` is now a **symlink** to an immutable sibling named
`generation-<random-id>.parquet`. The symlink is switched with a single
same-directory `os.replace`, after validating, flushing, and renaming the
staged Parquet into its immutable generation name.

All cooperating publishers acquire `.publish.lock` with `fcntl.flock`. A
publisher must compare the generation observed before source preparation with
the generation observed under that lock. The Makefile's
`candidate-fact-partition` target captures this token *before* its dbt CTAS and
passes it as `--expected-generation`. A changed token makes the publisher reject
rather than silently overwrite the later generation. Lock acquisition has a
bounded timeout. Process death releases the kernel lock.

An existing PR35 regular `data.parquet` is migrated with a hard link to a
`generation-legacy-*.parquet` sibling before its name is replaced with the new
pointer. A reader may follow `data.parquet` without acquiring any lock. Existing
immutable generations are deliberately retained so readers resolving an old
pointer can still open it. No automatic generation garbage collection is
performed in this PR.

## Failure windows

| Failure moment | Recovery rule |
| --- | --- |
| During `COPY` to hidden temp file | Previous pointer unchanged; next locked writer removes abandoned temps. |
| After generation durability, before pointer swap | Previous pointer unchanged; orphan immutable generation retained. |
| During atomic symlink swap | Reader resolves old or new complete generation; no intermediate partial file is published. |
| After swap, before return | Complete new generation remains referenced. Re-run must capture the current token. |
| While owning partition lock | Kernel releases lock on process exit or SIGKILL. |

`fsync` on the staged Parquet and partition directory is part of the protocol.
There is no second mutable JSON manifest that could disagree with the active
symlink. Durable outcomes still depend on POSIX-compatible storage, hardware,
and the filesystem's documented `fsync` / atomic-rename behavior.

## Known boundaries

- **Local filesystem only.** This design relies on Unix symlinks, hard links,
  POSIX advisory flock and same-directory rename. Do not claim that it supports
  arbitrary object stores, Windows filesystems, NFS, or uncooperative writers.
- The Makefile's generation CAS protects against an intervening *publication*,
  but does **not** serialize simultaneous dbt CTAS builds in the same DuckDB
  warehouse. Concurrent source preparation requires job/warehouse isolation or
  a higher-level lease. A manually invoked publisher without an explicit
  `--expected-generation` only fences writers racing after its own start.
- Read consistency is **per partition/path resolution**. Multi-partition
  queries, multiple independent opens, and full multi-cycle snapshots are not
  globally transactional.
- Generation files are retained; safe garbage collection requires a separate
  reader-lifetime/retention policy. Temporary files can be pruned under lock.
- Fault-injection and subprocess SIGKILL tests are not equivalent to physical
  power-loss testing. A real-data, dbt/Gold integration gate is required.

## Required evidence before merge

1. PR35 regression tests plus PR36 concurrency/crash/fencing tests.
2. Run `candidate-fact-partition` against a temporary DuckDB and filesystem.
3. Confirm the 2026/general Parquet is reachable by the Gold glob, that
   content and scope match CTAS, and that 162/162 dbt nodes remain green.
4. Keep the canonical TSE raw source and existing published Parquet untouched.
