#!/usr/bin/env python3
"""Migrate an OFFLINE COPY of the TSE Gold DuckDB to conformed cycle/event keys.

Never edits the input database, raw cache, or published Parquet.  The output
is a new DuckDB file in the SAME directory as the source (preserves relative
paths embedded in existing views).  All migrations occur in one DuckDB
transaction.  Reconciles source and copy via conserved physical row/measure
signatures and renders the two actual dbt singular tests without executing dbt.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid


PHYSICAL = (
    'dim_election', 'dim_candidate', 'dim_geography', 'dim_party',
    'fact_electorate_municipality', 'fact_tally_munzona',
    'fact_turnout', 'fact_party_votes',
)

SIGNATURES = {
    'dim_election': ('election_year', 'election_type', 'election_code'),
    'dim_candidate': ('candidate_id', 'declared_assets_value'),
    'dim_geography': ('municipality_code',),
    'dim_party': ('party_number',),
    'fact_electorate_municipality': ('electorate',),
    'fact_tally_munzona': ('eligible_voters', 'turnout', 'abstentions', 'nominal_valid_votes'),
    'fact_turnout': ('eligible_voters', 'turnout', 'abstentions'),
    'fact_party_votes': ('nominal_valid_votes', 'party_valid_votes'),
    'fact_candidate_votes': ('nominal_valid_votes',),
    'candidate_ranking': ('nominal_valid_votes',),
    'election_results': ('top_nominal_valid_votes',),
}

IDENTITY = "cast(election_year as varchar) || ':' || election_type"
EVENT = IDENTITY + " || ':' || cast(election_code as varchar)"
SINGULAR = ('assert_conformed_election_identity.sql',
            'assert_conformed_dimension_membership.sql')


class MigrationError(RuntimeError):
    pass


def quoted_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def columns(conn, name: str) -> set[str]:
    return {row[1] for row in conn.execute(f"pragma table_info('{name}')").fetchall()}


def relation_kind(conn, name: str) -> str:
    row = conn.execute("""select table_type from information_schema.tables
                          where table_schema='main' and table_name=?""", [name]).fetchone()
    if not row:
        raise MigrationError(f'missing required relation: {name}')
    return row[0]


def signature(conn, name: str, measures: tuple[str, ...]) -> tuple[str, ...]:
    """Preserve row count and signed sums; don't use SUM on non-numeric columns."""
    available = columns(conn, name)
    selected = [m for m in measures if m in available]
    sums = [f'cast(sum({m}) as varchar)' for m in selected
            if m not in ('election_year', 'election_type', 'election_code',
                         'candidate_id', 'municipality_code', 'party_number')]
    sql = f'select count(*)' + (', ' + ', '.join(sums) if sums else '') + f' from main.{name}'
    return tuple(str(x) for x in conn.execute(sql).fetchone())


def signatures(conn) -> dict[str, tuple[str, ...]]:
    return {table: signature(conn, table, measures) for table, measures in SIGNATURES.items()}


def dbt_sql(root: Path, filename: str) -> str:
    path = root / 'tse_dbt' / 'tests' / filename
    if not path.is_file():
        raise MigrationError(f'missing dbt singular assertion: {path}')
    contents = path.read_text(encoding='utf-8')
    names = re.findall(r"\{\{\s*ref\('([a-z_]+)'\)\s*\}\}", contents)
    if not names or len(names) != contents.count('{{'):
        raise MigrationError(f'unsupported dbt expression in {filename}')
    for name in names:
        if name not in {*PHYSICAL, 'dim_election_cycle', 'fact_candidate_votes'}:
            raise MigrationError(f'unexpected dbt ref {name} in {filename}')
        contents = re.sub(r"\{\{\s*ref\('" + name + r"'\)\s*\}\}", 'main.' + name, contents)
    return contents.strip().rstrip(';')


def ensure_assertions(conn, root: Path) -> dict[str, int]:
    counts = {}
    for name in SINGULAR:
        sql = dbt_sql(root, name)
        count = conn.execute(f'select count(*) from ({sql}) as violations').fetchone()[0]
        counts[name] = int(count)
        if count:
            examples = conn.execute(f'select * from ({sql}) as violations limit 5').fetchall()
            raise MigrationError(f'{name}: {count} violation(s); sample: {examples!r}')
    return counts


def validate_candidate_files(fact_root: Path) -> str:
    candidates = sorted(fact_root.glob('election_type=*/year=*/data.parquet'))
    if not candidates or not all(x.is_file() for x in candidates):
        raise MigrationError('candidate Parquet partitions missing or not readable')
    return str(fact_root / 'election_type=*' / 'year=*' / 'data.parquet')


# Authoritative persisted relations containing observed election codes.
# Never manufacture codes from year/type, office, or a calendar.
EVENT_EVIDENCE = (
    'dim_candidate', 'dim_party', 'fact_tally_munzona',
    'fact_turnout', 'fact_party_votes', 'fact_candidate_votes',
)


def observed_event_evidence(conn) -> dict[tuple[int, str, str], dict]:
    """Find distinct witnessed events, with independent source provenance."""
    observed: dict[tuple[int, str, str], dict] = {}
    for relation in EVENT_EVIDENCE:
        available = columns(conn, relation)
        needed = {'election_year', 'election_type', 'election_code'}
        if not needed.issubset(available):
            raise MigrationError(f'{relation} lacks event evidence: {needed - available}')
        scope_expr = ('cast(election_scope as varchar)' if 'election_scope' in available
                      else 'cast(null as varchar)')
        sql = ("select distinct cast(election_year as integer), "
               "cast(election_type as varchar), cast(election_code as varchar), "
               f"{scope_expr} from main.{relation}")
        for year, typ, code, scope in conn.execute(sql).fetchall():
            if year is None or not typ or code is None or not code.strip():
                raise MigrationError(f'invalid event key observed in {relation}')
            key = (int(year), typ, code)
            event = observed.setdefault(key, {'sources': set(), 'scopes': set()})
            event['sources'].add(relation)
            if scope is not None and scope.strip():
                event['scopes'].add(scope)
    return observed


def reconcile_observed_events(conn, *, authorize: bool) -> list[dict]:
    """Repair missing event dimension rows only with observed, unambiguous keys.

    This writes to the isolated candidate copy, never to the source DuckDB.
    Returns exact provenance records for auditing and signature reconciliation.
    """
    original = conn.execute(
        "select election_year, election_type, cast(election_code as varchar), "
        "election_id from main.dim_election").fetchall()
    known = {(int(y), typ, code) for y, typ, code, _ in original}
    if len(known) != len(original) or len({eid for *_, eid in original}) != len(original):
        raise MigrationError('dim_election contains duplicate event identities')
    evidence = observed_event_evidence(conn)
    missing = sorted(k for k in evidence if k not in known)
    if not missing:
        return []
    sample = [{'election_id': f'{y}:{typ}:{code}',
               'sources': sorted(evidence[(y, typ, code)]['sources']),
               'scopes': sorted(evidence[(y, typ, code)]['scopes'])}
              for y, typ, code in missing[:10]]
    if not authorize:
        raise MigrationError(
            f'{len(missing)} election events missing from dim_election. '
            f'Sample: {sample!r}. Review provenance then explicitly pass '
            '--reconcile-observed-events')
    needed = {'election_id', 'cycle_id', 'election_year', 'election_type',
              'election_code', 'election_scope', 'cycle_label'}
    if not needed.issubset(columns(conn, 'dim_election')):
        raise MigrationError(f'dim_election missing columns for enrichment: '
                             f'{needed - columns(conn, "dim_election")}')
    repairs = []
    for year, typ, code in missing:
        item = evidence[(year, typ, code)]
        scopes = item['scopes']
        if len(scopes) != 1:
            raise MigrationError(f'unknown/contradictory scope for '
                                 f'{year}:{typ}:{code}: {sorted(scopes)}, '
                                 f'sources={sorted(item["sources"])}')
        scope = next(iter(scopes))
        expected_scope = {'general': 'federal_state', 'municipal': 'municipal'}.get(typ)
        if scope != expected_scope:
            raise MigrationError(f'contradictory election_scope for '
                                 f'{year}:{typ}:{code}: {scope!r}, '
                                 f'expected {expected_scope!r}')
        eid = f'{year}:{typ}:{code}'
        conn.execute('''insert into main.dim_election
            (election_id, cycle_id, election_year, election_type,
             election_code, election_scope, cycle_label)
            values (?, ?, ?, ?, ?, ?, ?)''',
            [eid, f'{year}:{typ}', year, typ, code, scope, f'{year} {typ} {code}'])
        repairs.append({'election_id': eid, 'election_scope': scope,
                        'sources': sorted(item['sources'])})
    return repairs


def migrate(conn, fact_root: Path, *, allow_event_enrichment: bool = False) -> list[dict]:
    for table in PHYSICAL:
        if relation_kind(conn, table) != 'BASE TABLE':
            raise MigrationError(f'{table} must be a persisted Gold table')
        for col in ('election_year', 'election_type'):
            if col not in columns(conn, table):
                raise MigrationError(f'{table} lacks {col}')
    if relation_kind(conn, 'fact_candidate_votes') != 'VIEW':
        raise MigrationError('fact_candidate_votes must be Parquet-backed view')
    pattern = validate_candidate_files(fact_root)
    sql_parquet = f'read_parquet({quoted_path(Path(pattern))}, hive_partitioning=true, union_by_name=true)'
    parquet_columns = {field[0] for field in conn.execute(f'select * from {sql_parquet} limit 0').description}
    required = {'election_year', 'election_type', 'election_id', 'election_code', 'nominal_valid_votes'}
    if not required.issubset(parquet_columns):
        raise MigrationError(f'candidate Parquet lacks columns: {required - parquet_columns}')
    conn.execute('begin transaction')
    try:
        for table in PHYSICAL:
            if 'cycle_id' not in columns(conn, table):
                conn.execute(f'alter table main.{table} add column cycle_id varchar')
            conn.execute(f'''update main.{table} set cycle_id = {IDENTITY}
                             where cycle_id is distinct from ({IDENTITY})''')
        conn.execute(f'''update main.dim_candidate set election_id = {EVENT}
                         where election_id is distinct from ({EVENT})''')
        if 'cycle_id' in parquet_columns:
            projection = f'select * replace ({IDENTITY} as cycle_id)'
        else:
            projection = f'select *, {IDENTITY} as cycle_id'
        conn.execute(f'''create or replace view main.fact_candidate_votes as
                         {projection} from {sql_parquet}''')
        added_events = reconcile_observed_events(conn, authorize=allow_event_enrichment)
        conn.execute(f'''create or replace table main.dim_election_cycle as
                        select distinct {IDENTITY} as cycle_id,
                               election_year, election_type
                        from main.dim_election''')
        conn.execute('commit')
        return added_events
    except BaseException:
        conn.execute('rollback')
        raise


def validate_cycle_uniqueness(conn) -> int:
    count, distinct = conn.execute('''select count(*), count(distinct cycle_id)
                                      from main.dim_election_cycle''').fetchone()
    if count < 1 or count != distinct:
        raise MigrationError('dim_election_cycle is empty or has duplicate cycle keys')
    for rel in PHYSICAL:
        if conn.execute(f'select count(*) from main.{rel} where cycle_id is null').fetchone()[0]:
            raise MigrationError(f'{rel} contains null cycle_id')
    return count


@contextlib.contextmanager
def cooperative_source_lock(source: Path):
    # Mirrors the existing warehouse build coordinator's lock path.
    lock = source.with_name(source.name + '.build.lock')
    with lock.open('a+b') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MigrationError('warehouse build lock is held; stop writers') from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def assert_clean_wal(path: Path) -> None:
    wal = Path(str(path) + '.wal')
    if wal.exists():
        raise MigrationError(f'WAL exists; close DuckDB cleanly first: {wal}')


def run(source: Path, dest: Path, fact_root: Path, project_root: Path, *,
        ack_quiescent: bool, reconcile_observed_events: bool = False) -> dict:
    try:
        import duckdb
    except ImportError as exc:
        raise MigrationError('duckdb Python package required (use project .venv)') from exc
    if not ack_quiescent:
        raise MigrationError('--ack-quiescent is required (close DuckDB CLI and stop all DB writers)')
    if source.expanduser().is_symlink():
        raise MigrationError('source DB cannot be a symlink')
    source = source.expanduser().resolve(strict=True)
    dest = dest.expanduser().absolute()
    fact_root = fact_root.expanduser().resolve(strict=True)
    project_root = project_root.expanduser().resolve(strict=True)
    if source.suffix != '.duckdb' or dest.suffix != '.duckdb':
        raise MigrationError('source and candidate must end in .duckdb')
    if source.is_symlink() or not source.is_file():
        raise MigrationError('source must be a regular database file, not a symlink')
    if dest.exists() or dest.is_symlink():
        raise MigrationError(f'output exists; refusing overwrite: {dest}')
    if source == dest.resolve() or source.parent != dest.parent.resolve():
        raise MigrationError('new candidate must be a different file in the SAME directory as source')
    if not (project_root / 'tse_dbt' / 'tests').is_dir():
        raise MigrationError('project root must contain tse_dbt/tests')
    validate_candidate_files(fact_root)
    # An updated DuckDB database rewrites column blocks; require conservative free space.
    required_space = max(2 * 1024**3, source.stat().st_size * 3)
    if shutil.disk_usage(source.parent).free < required_space:
        raise MigrationError(f'insufficient free space, need {required_space} bytes free')
    stage = dest.with_name(dest.name + '.partial-' + uuid.uuid4().hex)
    with cooperative_source_lock(source):
        assert_clean_wal(source)
        original_sha = sha256_file(source)
        print('Source SHA256:', original_sha, flush=True)
        try:
            with source.open('rb') as src, stage.open('xb') as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            if sha256_file(stage) != original_sha or sha256_file(source) != original_sha:
                raise MigrationError('source changed while copying; close all writers')
            with duckdb.connect(str(stage), read_only=False) as conn:
                baseline = signatures(conn)
                print('Baseline signatures captured:', len(baseline), flush=True)
                added_events = migrate(conn, fact_root,
                    allow_event_enrichment=reconcile_observed_events)
                print(f'Reconciled witnessed dim_election events: {len(added_events)}', flush=True)
                cycles = validate_cycle_uniqueness(conn)
                assertions = ensure_assertions(conn, project_root)
                migrated = signatures(conn)
                changes = {k: [baseline[k], migrated[k]] for k in baseline
                           if k != 'dim_election' and baseline[k] != migrated[k]}
                expected_count = int(baseline['dim_election'][0]) + len(added_events)
                if int(migrated['dim_election'][0]) != expected_count:
                    changes['dim_election'] = [baseline['dim_election'],
                                               migrated['dim_election']]
                if changes:
                    raise MigrationError(f'measure signature drift: {changes}')
                # Ensure all logical changes are durable without a surviving WAL.
                conn.execute('checkpoint')
            assert_clean_wal(stage)
            if sha256_file(source) != original_sha:
                raise MigrationError('source mutated during validation')
            assert_clean_wal(source)
            # Atomic publication of COPY ONLY; original is never replaced.
            os.rename(stage, dest)
            parent_fd = os.open(str(source.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            report = {
                'result': 'PASS', 'source_db': str(source), 'source_sha256': original_sha,
                'candidate_db': str(dest), 'candidate_sha256': sha256_file(dest),
                'fact_root': str(fact_root), 'cycles': cycles,
                'reconciled_event_count': len(added_events),
                'reconciled_event_evidence': added_events,
                'signature_models': len(baseline), 'dbt_assertions': assertions,
                'source_untouched': True,
                'raw_rehydration': False,
                'next_step': 'Review candidate and run additional dbt tests before promotion; NEVER overwrite original automatically',
            }
            return report
        finally:
            stage.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-db', required=True, type=Path)
    parser.add_argument('--candidate-db', required=True, type=Path)
    parser.add_argument('--fact-root', required=True, type=Path)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--ack-quiescent', action='store_true')
    parser.add_argument('--reconcile-observed-events', action='store_true',
                        help='Explicitly permit adding missing dim_election rows from witnessed Gold fact/dimension events in the isolated copy')
    parser.add_argument('--report', type=Path, help='Optional JSON report path outside source warehouse')
    args = parser.parse_args(argv)
    try:
        if args.report and args.report.expanduser().resolve().exists():
            raise MigrationError('report path already exists; refusing overwrite')
        report = run(args.source_db, args.candidate_db, args.fact_root,
                     args.project_root, ack_quiescent=args.ack_quiescent,
                     reconcile_observed_events=args.reconcile_observed_events)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.report:
            target = args.report.expanduser().resolve()
            target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        return 0
    except (MigrationError, OSError, ValueError) as exc:
        print('GOLD MIGRATION DENIED:', exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
