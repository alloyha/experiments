#!/usr/bin/env python3
"""CCC B2: fenced, auditable canonical party-ID repair of one existing DuckDB.

Default mode is strictly read-only. --apply is gated by exact per-cycle counts,
full legacy coverage, a durable backup and a warehouse lock. Only party_id is
changed; this is NOT a dbt rebuild or external-consumer certification.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from scripts.audit_party_id_materialization import (
    CYCLE_PATTERN,
    audit_database,
    inspect_catalog,
    relation,
    summarize_rows,
)


class MigrationRejected(ValueError):
    """Nonconforming source, unsupported scope or unavailable safety precondition."""


def parse_expected(entries: list[str]) -> dict[str, int]:
    if not entries:
        raise MigrationRejected('Specify the full approved inventory with --expect-cycle YEAR:type=ROWS')
    expected: dict[str, int] = {}
    for item in entries:
        cycle, sep, count = item.partition('=')
        if not sep or not CYCLE_PATTERN.fullmatch(cycle) or not count.isdecimal():
            raise MigrationRejected(f'Invalid expected cycle/count: {item!r}')
        if cycle in expected or int(count) <= 0:
            raise MigrationRejected(f'Duplicate or empty expected cycle: {cycle}')
        expected[cycle] = int(count)
    return expected


def evaluate_before(rows: list[dict[str, Any]], expected: dict[str, int],
                    expected_legacy: int) -> dict[str, Any]:
    observed = {row['cycle']: row['rows'] for row in rows}
    if observed != expected:
        raise MigrationRejected(f'Observed cycle inventory differs: expected={expected}, observed={observed}')
    total = sum(row['rows'] for row in rows)
    if expected_legacy != total:
        raise MigrationRejected(f'Expected legacy count {expected_legacy} != observed rows {total}')
    for row in rows:
        for attr in ('null_key_rows', 'invalid_cycle_ids', 'invalid_election_ids',
                     'duplicate_grain_rows', 'duplicate_party_id_rows', 'null_party_ids'):
            if row[attr]:
                raise MigrationRejected(f'{row["cycle"]} has non-ID structural defect: {attr}={row[attr]}')
        if row['legacy_party_ids'] != row['rows'] or row['noncanonical_party_ids'] != row['rows']:
            raise MigrationRejected(f'{row["cycle"]} has mixed or unexpected party IDs')
    return {'status':'READY', 'cycles':expected, 'legacy_rows':total,
            'scope':'Only exact historical legacy IDs; no raw, fact or other Gold data rewritten'}


def verify_after(rows: list[dict[str, Any]], expected: dict[str, int]) -> None:
    observed = {row['cycle']: row['rows'] for row in rows}
    if observed != expected:
        raise MigrationRejected('Row inventory changed during migration')
    for row in rows:
        if row['status'] != 'PASS' or row['legacy_party_ids'] or row['noncanonical_party_ids']:
            raise MigrationRejected(f'Post-migration identity rejected: {row}')


def ensure_distinct_canonical_ids(conn: Any, schema: str, total: int) -> None:
    target = relation(schema)
    distinct = conn.execute(
        f'SELECT count(DISTINCT cast(election_id as varchar) || \':\' || '
        f'cast(party_number as varchar)) FROM {target}'
    ).fetchone()[0]
    if distinct != total:
        raise MigrationRejected(f'Canonical party IDs collide: distinct={distinct}, rows={total}')


def capture_rows(conn: Any, schema: str) -> tuple[list[str], list[tuple]]:
    target = relation(schema)
    cur = conn.execute(
        f'SELECT * FROM {target} '
        'ORDER BY election_year, election_type, election_code, party_number'
    )
    return [description[0] for description in cur.description], cur.fetchall()


def unchanged_except_party_id(before: tuple[list[str], list[tuple]],
                              after: tuple[list[str], list[tuple]]) -> None:
    columns, old_rows = before
    new_columns, new_rows = after
    if columns != new_columns or len(old_rows) != len(new_rows) or 'party_id' not in columns:
        raise MigrationRejected('Column definitions or row count changed')
    idx = columns.index('party_id')
    event_idx = columns.index('election_id')
    number_idx = columns.index('party_number')
    for old, new in zip(old_rows, new_rows):
        if old[:idx] != new[:idx] or old[idx+1:] != new[idx+1:]:
            raise MigrationRejected('Non-party_id field mutated')
        if new[idx] != f'{old[event_idx]}:{old[number_idx]}':
            raise MigrationRejected('Incorrect canonical identity after update')


def create_durable_backup(db: Path, backup: Path) -> None:
    db = db.resolve(strict=True)
    # Backup destination cannot be a symlink, already exist, or be the database.
    backup = backup.expanduser().absolute()
    if backup.is_symlink() or backup.exists() or backup.resolve(strict=False) == db:
        raise MigrationRejected(f'Backup target must be absent, distinct and non-symlink: {backup}')
    if not backup.parent.is_dir():
        raise MigrationRejected(f'Backup parent must exist: {backup.parent}')
    file_size = db.stat().st_size
    free = shutil.disk_usage(backup.parent).free
    if free < file_size + 256 * 1024 * 1024:
        raise MigrationRejected(f'Insufficient free space for safe backup: need {file_size + 256*1024*1024}, available {free}')
    try:
        with db.open('rb') as source, backup.open('xb') as dest:
            shutil.copyfileobj(source, dest, 8 * 1024 * 1024)
            dest.flush()
            os.fsync(dest.fileno())
        if backup.stat().st_size != file_size:
            raise MigrationRejected('Backup size mismatch')
        dir_fd = os.open(backup.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        backup.unlink(missing_ok=True)
        raise


def migration(db: Path, schema: str, expected: dict[str, int], legacy: int,
              *, apply: bool, backup: Path | None = None,
              lock_timeout: float = 30.0) -> dict[str, Any]:
    db = db.expanduser().absolute()
    if db.is_symlink() or not db.is_file():
        raise MigrationRejected('Warehouse must exist as a regular, non-symlink file')
    db = db.resolve(strict=True)
    if not apply:
        audit = audit_database(db, schema, list(expected))
        outcome = evaluate_before(audit['cycles'], expected, legacy)
        import duckdb
        with duckdb.connect(str(db), read_only=True) as conn:
            ensure_distinct_canonical_ids(conn, schema, outcome['legacy_rows'])
        return {'warehouse':str(db), 'mode':'plan', **outcome,
                'warning':'No write executed; external consumers and GG attestation remain pending'}
    if backup is None:
        raise MigrationRejected('--apply requires a distinct --backup path')
    if schema != 'main':
        raise MigrationRejected('Writable migration currently supports only main.dim_party')
    from scripts.warehouse_lifecycle import maintenance_lock, rotate_epoch
    import duckdb
    with maintenance_lock(db, timeout=lock_timeout):
        with duckdb.connect(str(db)) as conn:
            inspect_catalog(conn, schema)
            before_rows = summarize_rows(conn, schema)
            outcome = evaluate_before(before_rows, expected, legacy)
            ensure_distinct_canonical_ids(conn, schema, legacy)
            # Converge the persistent file before the byte-wise durable backup.
            conn.execute('CHECKPOINT')
        wal = db.with_name(db.name + '.wal')
        if wal.exists() and wal.stat().st_size:
            raise MigrationRejected(f'Unexpected nonempty DuckDB WAL after checkpoint: {wal}')
        create_durable_backup(db, backup)
        with duckdb.connect(str(db)) as conn:
            inspect_catalog(conn, schema)
            evaluate_before(summarize_rows(conn, schema), expected, legacy)
            ensure_distinct_canonical_ids(conn, schema, legacy)
            old = capture_rows(conn, schema)
            # Revoke all existing GG authority before the first committed write.
            # If a crash occurs now, certification must be repeated, not assumed.
            rotate_epoch(db)
            conn.execute('BEGIN TRANSACTION')
            try:
                target = relation(schema)
                conn.execute(
                    f'UPDATE {target} SET party_id = cast(election_id as varchar) '
                    f"|| ':' || cast(party_number as varchar) "
                    'WHERE party_id = cast(election_id as varchar) '
                    f"|| ':' || cast(election_code as varchar) || ':' "
                    '|| cast(party_number as varchar)'
                )
                new = capture_rows(conn, schema)
                unchanged_except_party_id(old, new)
                verify_after(summarize_rows(conn, schema), expected)
                ensure_distinct_canonical_ids(conn, schema, legacy)
                conn.execute('COMMIT')
            except BaseException:
                conn.execute('ROLLBACK')
                raise
        final = audit_database(db, schema, list(expected))
        verify_after(final['cycles'], expected)
        return {'warehouse':str(db), 'mode':'apply', 'status':'PASS',
                'updated_rows':legacy, 'cycles':expected,
                'durable_backup':str(backup.expanduser().absolute()),
                'gg_proof':'Lifecycle epoch rotated: ALL existing GG certificates require recertification',
                'external_consumers':'NOT certified'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--schema', default='main')
    parser.add_argument('--expect-cycle', action='append', default=[],
                        help='Approved YEAR:general=COUNT, YEAR:municipal=COUNT; repeat')
    parser.add_argument('--expect-legacy', required=True, type=int)
    parser.add_argument('--apply', action='store_true', help='Opt in to fenced write + epoch rotation')
    parser.add_argument('--backup', type=Path, help='Required absent durable backup path for --apply')
    parser.add_argument('--lock-timeout', type=float, default=30)
    args = parser.parse_args(argv)
    try:
        expected = parse_expected(args.expect_cycle)
        report = migration(args.db, args.schema, expected, args.expect_legacy,
                           apply=args.apply, backup=args.backup,
                           lock_timeout=args.lock_timeout)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (MigrationRejected, ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f'CCC B2 MIGRATION REJECTED: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
