# CCC B2 — fenced transaction for canonical `party_id`

## Observed evidence (2026-10-20)

The user's read-only warehouse audit reported 717 rows in `main.dim_party`, **all**
using the deterministic five-component legacy identity. The five observed cycles
were 2018:general=88, 2020:municipal=395, 2022:general=58,
2024:municipal=134, 2026:general=42. All other checked identity, cycle,
null and uniqueness invariants were zero.

The corrected dbt model is on `main` after PR #86. This one-time repair changes
only `dim_party.party_id` to `election_id || ':' || party_number` and preserves
all other columns and rows. It is **not** a replacement for source freshness,
reingestion, cross-election party SCD2, consumer compatibility, or physical GG
recertification.

## Safety properties

1. Read-only is the **default**; no incidental writes or database creation.
2. Requires the *entire observed* cycle/row inventory and the legacy count.
   Any new/removed cycle, null, duplicate, non-legacy drift or canonical collision
   refuses the operation. No partial migration is accepted.
3. Opt-in `--apply` requires an exclusive stable warehouse flock compatible with
   `run_coordinated_dbt`, a checkpoint and a byte-complete, fsynced backup in an
   existing destination directory (with enough free space). Refuses symlink or
   existing destination; leaves a complete backup for recovery.
4. While the same warehouse lock is held, rotates its *durable lifecycle epoch*
   **before** the transactional update, so existing GG certificates cannot authorize
   a new warehouse state, even following a crash. **Every preexisting GG
   attestation for this warehouse is invalidated** and must be re-certified
   separately if consumers rely on it.
5. Transactionally modifies ONLY exact legacy `party_id` values. Validates all
   other columns and counts before `COMMIT`. Re-audits after committing.
6. Direct uncoordinated DuckDB writes and external consumers remain outside the
   cooperative lock model. Stop other jobs and plan consumer cutover.

## Decision and commands

Operate from a worktree containing `scripts/audit_party_id_materialization.py`.
First, run the auditor's 10 tests, the nine migration-guard tests, and `make check`.
Use the **existing actual** 1.2GB warehouse path in `--db`, not the freshly
created worktree's default `data/warehouse` path. Before repair, rehearse on a
separately copied database if space permits. The generated tool never performs
an implicit write without `--apply`.

Plan (safe read-only):

```sh
python -m scripts.migrate_party_id_canonical \
  --db /path/to/existing/tse_analytics.duckdb \
  --expect-cycle 2018:general=88 \
  --expect-cycle 2020:municipal=395 \
  --expect-cycle 2022:general=58 \
  --expect-cycle 2024:municipal=134 \
  --expect-cycle 2026:general=42 \
  --expect-legacy 717
```

Opt-in write (only after reviewing the plan, workspace state, free disk space,
backups and downstream consumers):

```sh
python -m scripts.migrate_party_id_canonical \
  --db /path/to/existing/tse_analytics.duckdb \
  --expect-cycle 2018:general=88 \
  --expect-cycle 2020:municipal=395 \
  --expect-cycle 2022:general=58 \
  --expect-cycle 2024:municipal=134 \
  --expect-cycle 2026:general=42 \
  --expect-legacy 717 \
  --apply --backup /path/to/backup/tse_analytics.before_party_id.duckdb
```

Backups are deliberate and potentially large (approximately the warehouse's
full size). **Do not write** until there is adequate free space and an approved
maintenance window. If a write fails after the epoch rotates, even when the
DuckDB transaction rolls back, existing certificates remain invalid and require
renewal. Restore only under the same cooperative lock with an approved recovery
procedure; don't simply overwrite an open DuckDB file.

## Acceptance after a successful apply

- Read-only auditor `PASS` with **five explicit** `--expect-cycle` arguments.
- Per-cycle row counts remain 88, 395, 58, 134, 42; nulls/duplicates/wrong
  cycle/wrong event/legacy/noncanonical all remain zero.
- Verify `party_performance` and `cube_party_performance_municipality` are
  actually views in the materialized warehouse, and that joined IDs reflect
  the canonical value; absence or materialized tables must be investigated.
- Source completeness, persisted candidate fact Parquet, external consumers,
  GG recertification and ODCS promotion are separate evidentiary obligations.
