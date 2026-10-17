#!/usr/bin/env python3
"""Read-only performance falsifier for graph candidate-vote conservation.

Runs the checked-in eight-invariant diagnostic's real cycle-scoped views, using
exactly the unmodified expected/actual CTEs of the candidate-vote SQL contract.
Contrasts (a) existing two-way EXCEPT ALL with (b) UNION ALL signed multiplicity.
All DuckDB connections are in-memory; live warehouse is ATTACHed READ_ONLY.

Never issue freshness certificates or modify warehouse, Parquet, models or tests.
Run only AFTER the current graph certification has finished, as locks conflict.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import median
import sys
import time
from threading import Event, Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CYCLES = ('2018:general', '2020:municipal', '2022:general',
          '2024:municipal', '2026:general')
COLUMNS = ('source_id', 'target_id', 'cycle_id', 'election_id', 'round_number',
           'office_code', 'is_transit_vote', 'nominal_valid_votes',
           'zone_rows', 'zone_count')
TAIL = """(select 'missing_or_changed' as issue, * from expected
 except all select 'missing_or_changed', * from actual)
union all
(select 'unexpected_or_changed' as issue, * from actual
 except all select 'unexpected_or_changed', * from expected)"""
TEMPLATE_MARKER = "(select 'missing_or_changed' as issue, * from expected"


class BenchmarkError(RuntimeError):
    pass


def canonical_sql(sql: str) -> str:
    return ' '.join(sql.strip().split())


def checked_ctes(rendered: str) -> str:
    """Fail closed if the deployed original conservation assertion changes."""
    prefix, separator, suffix = rendered.partition(TEMPLATE_MARKER)
    if not separator or canonical_sql(separator + suffix) != canonical_sql(TAIL):
        raise BenchmarkError('SQL conservation tail changed; review rewrite before benchmarking')
    if not re.search(r'\bexpected\s+as\s*\(', prefix, flags=re.I) or not re.search(
            r'\bactual\s+as\s*\(', prefix, flags=re.I):
        raise BenchmarkError('expected/actual CTEs are missing')
    # This ends with the comma after the actual CTE; strip comments before
    # appending a new named CTE, preserving the original model SQL unmodified.
    prefix = prefix.rstrip()
    if not prefix.endswith(',') and prefix.endswith(')'):
        prefix += ','
    elif not prefix.endswith(','):
        raise BenchmarkError('unexpected CTE boundary')
    return prefix


def original_query(rendered: str) -> str:
    checked_ctes(rendered)
    # Count ALL rows of two-way multiset difference; no LIMIT 1 shortcut.
    # Wrapping the original query preserves its exact SQL and semantics.
    return f'SELECT count(*)::HUGEINT AS mismatch_rows FROM ({rendered}) AS multiset_delta'


def signed_query(rendered: str) -> str:
    prefix = checked_ctes(rendered)
    cols = ', '.join(COLUMNS)
    return f"""{prefix}
combined AS (
    SELECT {cols}, 1::HUGEINT AS __delta FROM expected
    UNION ALL
    SELECT {cols}, (-1)::HUGEINT AS __delta FROM actual
), signed_delta AS (
    SELECT SUM(__delta) AS multiplicity_delta
    FROM combined
    GROUP BY {cols}
    HAVING SUM(__delta) <> 0
)
SELECT COALESCE(SUM(ABS(multiplicity_delta)), 0)::HUGEINT AS mismatch_rows
FROM signed_delta"""



def rewrite_expected_zone_count(rendered: str) -> str:
    """Replace exactly one source expected-rollup DISTINCT after checking the contract."""
    checked_ctes(rendered)
    # The assertion's expected rollup is physically independent of graph_edges.
    pattern = r'\bcount\s*\(\s*distinct\s+zone\s*\)\s+as\s+zones\b'
    distinct_calls = re.findall(r'\bcount\s*\(\s*distinct\s+zone\s*\)', rendered, flags=re.I)
    rewritten, matches = re.subn(pattern, 'count(*) as zones', rendered, flags=re.I)
    if matches != 1 or len(distinct_calls) != 1:
        raise BenchmarkError('expected rollup must contain exactly one DISTINCT-zone count')
    return rewritten


def rewrite_graph_zone_count(rendered_model: str) -> str:
    """Replace only the CANDIDATE rollup, never the independent party rollup."""
    before, marker, after = rendered_model.partition('party_vote_rollup as (')
    if not marker or 'candidate_vote_rollup as (' not in before:
        raise BenchmarkError('graph model CTE layout changed; review count-star rewrite')
    pattern = r'\bcount\s*\(\s*distinct\s+zone\s*\)\s+as\s+zone_count\b'
    distinct_calls = re.findall(r'\bcount\s*\(\s*distinct\s+zone\s*\)', before, flags=re.I)
    changed, n = re.subn(pattern, 'count(*) as zone_count', before, flags=re.I)
    if n != 1 or len(distinct_calls) != 1:
        raise BenchmarkError('candidate graph rollup must contain exactly one DISTINCT-zone count')
    return changed + marker + after


def zone_uniqueness_sql() -> str:
    """Full, exact (group, zone) multiplicity proof using physical scoped Parquet."""
    return """WITH occurrences AS (
      SELECT cycle_id, election_id, round_number, uf, municipality_code,
             office_code, is_transit_vote, candidate_id, zone,
             COUNT(*)::HUGEINT AS n
      FROM fact_candidate_votes
      GROUP BY cycle_id, election_id, round_number, uf, municipality_code,
               office_code, is_transit_vote, candidate_id, zone
    )
    SELECT COALESCE(SUM(CASE WHEN zone IS NULL THEN n ELSE 0 END),0)::HUGEINT AS null_rows,
           COALESCE(SUM(CASE WHEN zone IS NOT NULL THEN n-1 ELSE 0 END),0)::HUGEINT AS duplicate_nonnull_rows,
           COALESCE(SUM(n),0)::HUGEINT AS total_rows,
           COUNT(*)::HUGEINT AS distinct_group_zone_keys
    FROM occurrences"""


def prove_zone_uniqueness(*, db: Path, root: Path, cycle: str, out: Path,
                          memory_limit: str, threads: int, max_seconds: int,
                          max_temp_gb: int) -> dict:
    """Independent fail-closed prerequisite; never consume a previous untrusted summary."""
    import duckdb
    from scripts import certify_gold_graph_cycle as diagnostic
    y, kind = cycle.split(':')
    parquet, before = diagnostic.selected_publication(root, int(y), kind)
    prefix = cycle.replace(':', '-') + '-countstar-guard'
    profile = out / (prefix + '.profile.json')
    target = out / (prefix + '.result.json')
    spill = out / 'spill' / prefix
    spill.mkdir(parents=True, exist_ok=False)
    sql = zone_uniqueness_sql()
    con = duckdb.connect(':memory:')
    try:
        con.execute('SET memory_limit = ' + diagnostic.quote_sql(memory_limit))
        con.execute(f'SET threads = {threads}')
        con.execute('SET temp_directory = ' + diagnostic.quote_sql(str(spill)))
        con.execute('SET max_temp_directory_size = ' + diagnostic.quote_sql(f'{max_temp_gb}GB'))
        diagnostic.setup_scoped_views(con, db, parquet, int(y), kind)
        configure_json_profiler(con, profile)
        started = time.perf_counter()
        values = fetchone_with_timeout(con, sql, max_seconds=max_seconds, label=prefix)
        elapsed = round(time.perf_counter() - started, 3)
    finally:
        con.close()
    perf = metrics(profile)
    _, after = diagnostic.selected_publication(root, int(y), kind)
    if before != after:
        raise BenchmarkError('publication drift while proving group-zone uniqueness')
    null_rows, duplicate_rows, source_rows, keys = (int(v) for v in values)
    if source_rows <= 0 or keys <= 0 or null_rows < 0 or duplicate_rows < 0:
        raise BenchmarkError('invalid guard inventory')
    record = {
        'scope': 'physical_group_zone_uniqueness_not_freshness',
        'result': 'PASS' if null_rows == duplicate_rows == 0 else 'DENIED',
        'cycle': cycle, 'null_zone_rows': null_rows,
        'duplicate_nonnull_zone_rows': duplicate_rows,
        'source_rows': source_rows, 'distinct_group_zone_keys': keys,
        'source_publication': after,
        'guard_sql_sha256': hashlib.sha256(sql.encode()).hexdigest(),
        'elapsed_seconds': elapsed, 'profiling': perf,
        'profile_json': str(profile),
    }
    durable_json(target, record)
    print(f'[graph-bench] {prefix}: null={null_rows} duplicated={duplicate_rows} '
          f'source_rows={source_rows} seconds={elapsed}', flush=True)
    if record['result'] != 'PASS':
        raise BenchmarkError('COUNT(*) not equivalent to COUNT(DISTINCT zone) for ' + cycle)
    return record

def query_for(rendered: str, variant: str) -> str:
    if variant == 'except_all':
        return original_query(rendered)
    if variant == 'signed':
        return signed_query(rendered)
    if variant == 'signed_countstar':
        return signed_query(rewrite_expected_zone_count(rendered))
    raise BenchmarkError('unrecognized benchmark variant: ' + variant)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def durable_json(path: Path, document: dict) -> None:
    """Create-only result; partial successful trials survive later interruption."""
    with path.open('x', encoding='utf-8') as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def assert_wal_free(db: Path) -> None:
    wal = Path(str(db) + '.wal')
    if wal.exists() or wal.is_symlink():
        raise BenchmarkError('DuckDB WAL detected, refuse benchmark: ' + str(wal))


def metrics(profile_path: Path) -> dict:
    if not profile_path.is_file():
        raise BenchmarkError('DuckDB profiler produced no JSON after result EOF and connection close: ' + str(profile_path))
    p = json.loads(profile_path.read_text(encoding='utf-8'))
    return {key: p.get(key) for key in (
        'latency', 'cpu_time', 'system_peak_buffer_memory',
        'system_peak_temp_dir_size', 'total_bytes_read', 'total_bytes_written',
        'cumulative_rows_scanned', 'cumulative_cardinality')}


def fetchone_with_timeout(con, sql: str, *, max_seconds: int, label: str):
    """Fetch exactly one row AND consume EOF to finalize DuckDB's profiling.

    The Python DuckDB API keeps the query result/transaction active after the
    first fetchone(), even for a scalar aggregate. A subsequent fetchone()
    returning None is required to complete it and publish profiler output.
    Avoid fetchall(), which could exhaust host RAM if a changed query returned
    unexpectedly many rows.

    Cooperatively interrupt an overlong DuckDB query without unsupported SETs.

    DuckDBPyConnection.interrupt() is supported on the deployed Python API,
    unlike `SET max_execution_time` in some DuckDB releases. This is a
    best-effort per-query timeout, NOT an OS-level process kill deadline.
    """
    if max_seconds <= 0:
        raise BenchmarkError('max_seconds must be positive')
    interrupt = getattr(con, 'interrupt', None)
    if not callable(interrupt):
        raise BenchmarkError('DuckDB connection lacks interrupt(); cannot enforce query timeout')
    completed = Event()
    expired = Event()

    def watchdog():
        if not completed.wait(max_seconds):
            expired.set()
            # The SQL thread owns `execute` and `fetchone`; only the documented
            # interrupt method is called from this watchdog thread.
            try:
                interrupt()
            except Exception:
                # Fail closed after execute returns/errors. A failed interrupt
                # cannot establish a hard bound on the running native query.
                pass

    started = time.monotonic()
    watcher = Thread(target=watchdog, name='duckdb-query-watchdog', daemon=True)
    watcher.start()
    try:
        result = con.execute(sql)
        row = result.fetchone()
        if row is None:
            raise BenchmarkError(f'{label}: scalar query returned no rows')
        # Crucial: EOF finalizes DuckDB's result and flushes JSON profiling.
        if result.fetchone() is not None:
            raise BenchmarkError(f'{label}: expected exactly one scalar row')
    except Exception as exc:
        if expired.is_set():
            raise BenchmarkError(f'{label}: query exceeded {max_seconds}s; interrupted') from exc
        raise
    finally:
        completed.set()
        watcher.join(timeout=1)
    if expired.is_set() or time.monotonic() - started > max_seconds:
        raise BenchmarkError(f'{label}: query exceeded {max_seconds}s; disregard result')
    return row


def configure_json_profiler(con, output: Path) -> None:
    """Select JSON format BEFORE a .json destination (DuckDB type validation).

    DuckDB rejects setting `profiling_output = '*.json'` while the profiling
    format is still its default `query_tree`. Configure immediately before
    the one SELECT we intend to profile so later SQL cannot overwrite it.
    """
    if output.suffix != '.json':
        raise BenchmarkError('JSON profiler output requires a .json suffix')
    con.execute("SET enable_profiling = 'json'")
    con.execute("SET profiling_output = '" + str(output).replace("'", "''") + "'")


def run_trial(*, db: Path, root: Path, cycle: str, variant: str,
              repeat: int, out: Path, memory_limit: str, threads: int,
              max_seconds: int, max_temp_gb: int, explain_only: bool,
              uniqueness_guard: dict | None = None) -> dict:
    import duckdb
    from scripts import certify_gold_graph_cycle as diagnostic

    year, kind = cycle.split(':')
    year = int(year)
    parquet, publication_before = diagnostic.selected_publication(root, year, kind)
    if variant == 'signed_countstar' and not explain_only:
        if (uniqueness_guard is None or uniqueness_guard.get('result') != 'PASS' or
            uniqueness_guard.get('cycle') != cycle or
            uniqueness_guard.get('source_publication') != publication_before or
            uniqueness_guard.get('null_zone_rows') != 0 or
            uniqueness_guard.get('duplicate_nonnull_zone_rows') != 0):
            raise BenchmarkError('signed_countstar requires verified exact group-zone uniqueness proof')
    sql_path = diagnostic.TESTS / 'assert_graph_candidate_vote_conservation.sql'
    sql_digest = sha256(sql_path)
    model_path = diagnostic.MODELS / 'graph_edges.sql'
    model_digest = sha256(model_path)
    rendered = diagnostic.render(sql_path, year=year, election_type=kind)
    query = query_for(rendered, variant)
    prefix = f'{cycle.replace(":", "-")}-{repeat:02d}-{variant}'
    plan_path = out / f'{prefix}.plan.txt'
    profile_path = out / f'{prefix}.profile.json'
    trial_path = out / f'{prefix}.result.json'
    spill = out / 'spill' / prefix
    spill.mkdir(parents=True, exist_ok=False)

    print(f'[graph-bench] cycle={cycle} repeat={repeat} variant={variant} '
          f'memory={memory_limit} threads={threads} at={datetime.now(timezone.utc).isoformat()}',
          flush=True)
    con = duckdb.connect(':memory:')
    try:
        con.execute('SET memory_limit = ' + diagnostic.quote_sql(memory_limit))
        con.execute(f'SET threads = {threads}')
        con.execute('SET temp_directory = ' + diagnostic.quote_sql(str(spill)))
        con.execute('SET max_temp_directory_size = ' + diagnostic.quote_sql(f'{max_temp_gb}GB'))
        diagnostic.setup_scoped_views(con, db, parquet, year, kind)
        if variant == 'signed_countstar':
            model_sql = diagnostic.render(model_path, year=year, election_type=kind)
            adjusted = rewrite_graph_zone_count(model_sql)
            # This is an in-memory view only: historical Gold remains ATTACH READ_ONLY.
            con.execute('CREATE OR REPLACE VIEW graph_edges AS ' + adjusted)
        # Verify true physical source is in scope; do not time this plan.
        probe = con.execute("EXPLAIN SELECT count(*) FROM graph_edges "
                            "WHERE edge_type='VOTED_FOR_CANDIDATE'").fetchone()[1]
        if 'READ_PARQUET' not in probe.upper():
            raise BenchmarkError('graph diagnostic is no longer physically bound to Parquet')
        plan = con.execute('EXPLAIN ' + query).fetchone()[1]
        with plan_path.open('x', encoding='utf-8') as f:
            f.write(plan)
        if explain_only:
            result = None
            elapsed = 0.0
            perf = {}
        else:
            # Only the comparison query is profiled; the profiler file is not
            # overwritten by EXPLAIN, setup statements or subsequent audits.
            configure_json_profiler(con, profile_path)
            started = time.perf_counter()
            result = int(fetchone_with_timeout(
                con, query, max_seconds=max_seconds, label=prefix)[0])
            elapsed = round(time.perf_counter() - started, 3)
            # No other SQL may run on this connection after the profiled SELECT.
    finally:
        con.close()
    # Some DuckDB versions only flush profile output on EOF / connection close.
    perf = {} if explain_only else metrics(profile_path)
    _, publication_after = diagnostic.selected_publication(root, year, kind)
    if (publication_before != publication_after or sql_digest != sha256(sql_path)
            or model_digest != sha256(model_path)):
        raise BenchmarkError('Parquet, contract SQL or graph model drifted while running ' + prefix)
    record = {
        'scope': 'readonly_graph_candidate_vote_conservation_benchmark_not_freshness',
        'result': 'PLAN_ONLY' if explain_only else ('PASS' if result == 0 else 'FAIL'),
        'cycle': cycle, 'variant': variant, 'repeat': repeat,
        'diff_rows': result, 'elapsed_seconds': elapsed,
        'duckdb_version': duckdb.__version__, 'memory_limit': memory_limit,
        'threads': threads, 'max_temp_gb': max_temp_gb,
        'sql_sha256': sql_digest, 'parquet': publication_after,
        'graph_model_sha256': model_digest,
        'guarded_countstar': variant == 'signed_countstar',
        'profiling': perf,
        'explain_plan': str(plan_path),
        'profile_json': str(profile_path) if not explain_only else None,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    durable_json(trial_path, record)
    print('[graph-bench] ' + prefix + ': diff_rows=' + str(result) +
          ' seconds=' + str(elapsed) + ' temp_peak=' +
          str(perf.get('system_peak_temp_dir_size')), flush=True)
    if not explain_only and result != 0:
        raise BenchmarkError(f'{prefix} detected {result} conservation multiset mismatches')
    return record


def cardinality_probe(*, db: Path, root: Path, cycle: str, out: Path,
                      memory_limit: str, threads: int, max_seconds: int,
                      max_temp_gb: int) -> dict:
    """Optional exact physical source/grain/group measurement (one extra scan)."""
    import duckdb
    from scripts import certify_gold_graph_cycle as diagnostic
    year, kind = cycle.split(':')
    year = int(year)
    parquet, published_before = diagnostic.selected_publication(root, year, kind)
    prefix = cycle.replace(':', '-') + '-cardinality'
    profile_path = out / (prefix + '.profile.json')
    result_path = out / (prefix + '.result.json')
    spill = out / 'spill' / prefix
    spill.mkdir(parents=True, exist_ok=False)
    print(f'[graph-bench] cycle={cycle} source-group-cardinality probe', flush=True)
    sql = """WITH grouped AS (
       SELECT cycle_id, election_id, round_number, uf, municipality_code,
              office_code, is_transit_vote, candidate_id,
              count(*) AS zone_rows, count(DISTINCT zone) AS zone_count
       FROM fact_candidate_votes
       GROUP BY ALL
    )
    SELECT count(*) AS groups, coalesce(sum(zone_rows), 0) AS source_rows,
           coalesce(sum(zone_count), 0) AS group_distinct_zone_sum,
           coalesce(max(zone_rows), 0) AS largest_group_rows
    FROM grouped"""
    con = duckdb.connect(':memory:')
    try:
        con.execute('SET memory_limit = ' + diagnostic.quote_sql(memory_limit))
        con.execute(f'SET threads = {threads}')
        con.execute('SET temp_directory = ' + diagnostic.quote_sql(str(spill)))
        con.execute('SET max_temp_directory_size = ' + diagnostic.quote_sql(f'{max_temp_gb}GB'))
        diagnostic.setup_scoped_views(con, db, parquet, year, kind)
        configure_json_profiler(con, profile_path)
        started = time.perf_counter()
        groups, rows, unique_zones, max_rows = fetchone_with_timeout(
            con, sql, max_seconds=max_seconds, label=prefix)
        elapsed = round(time.perf_counter() - started, 3)
    finally:
        con.close()
    perf = metrics(profile_path)
    _, published_after = diagnostic.selected_publication(root, year, kind)
    if published_before != published_after:
        raise BenchmarkError('source publication drift during cardinality probe')
    result = {
        'result': 'PASS', 'cycle': cycle, 'groups': int(groups), 'source_rows': int(rows),
        'group_distinct_zone_sum': int(unique_zones), 'largest_group_rows': int(max_rows),
        'rows_per_group': round(int(rows) / groups, 4) if groups else None,
        'elapsed_seconds': elapsed, 'profiling': perf,
        'parquet': published_after, 'profile_json': str(profile_path),
        'scope': 'physical_source_group_cardinality_not_freshness',
    }
    durable_json(result_path, result)
    print(f'[graph-bench] {prefix}: rows={rows} groups={groups} '
          f'max_group={max_rows} seconds={elapsed} temp_peak='
          f'{perf.get("system_peak_temp_dir_size")}', flush=True)
    return result


def speedups(results: list[dict]) -> dict:
    """Report exact wall-clock ratios without treating warm cache as independent."""
    by_cycle: dict[str, dict[int, dict[str, dict]]] = {}
    for item in results:
        if item['result'] != 'PASS':
            continue
        by_cycle.setdefault(item['cycle'], {}).setdefault(item['repeat'], {})[item['variant']] = item
    report = {}
    for cycle, reps in by_cycle.items():
        rows = []
        for rep, data in reps.items():
            if set(data) != {'except_all', 'signed'}:
                continue
            before = data['except_all']['elapsed_seconds']
            after = data['signed']['elapsed_seconds']
            rows.append({'repeat': rep, 'except_all_s': before, 'signed_s': after,
                         'speedup_except_over_signed': round(before / after, 4) if after > 0 else None,
                         'saved_seconds': round(before - after, 3)})
        countstar_rows = []
        for rep, data in reps.items():
            if {'signed', 'signed_countstar'} <= set(data):
                baseline = data['signed']['elapsed_seconds']
                optimized = data['signed_countstar']['elapsed_seconds']
                countstar_rows.append({
                    'repeat': rep, 'signed_s': baseline,
                    'signed_countstar_s': optimized,
                    'speedup_signed_over_countstar': round(baseline / optimized, 4) if optimized > 0 else None,
                    'saved_seconds': round(baseline - optimized, 3),
                })
        report[cycle] = {
            'signed_countstar_pairs': countstar_rows,
            'pairs': rows,
            'median_except_all_s': median(row['except_all_s'] for row in rows) if rows else None,
            'median_signed_s': median(row['signed_s'] for row in rows) if rows else None,
        }
    return report


def ordered_variants(repeat: int, selected: tuple[str, ...]) -> tuple[str, ...]:
    """A/B-B/A for paired trials; one explicitly selected arm otherwise."""
    if repeat < 1 or not selected or len(selected) > 2 or len(set(selected)) != len(selected) or any(
            variant not in ('except_all', 'signed', 'signed_countstar') for variant in selected):
        raise BenchmarkError('variants must be unique supported benchmark arms')
    if len(selected) == 1:
        return selected
    if set(selected) not in ({'except_all', 'signed'}, {'signed', 'signed_countstar'}):
        raise BenchmarkError('unsupported paired benchmark arms')
    return selected if repeat % 2 else tuple(reversed(selected))


def execute(args: argparse.Namespace) -> dict:
    from scripts import consolidate_candidate_cube_transfer as evidence
    from scripts.warehouse_lifecycle import lifecycle_epoch

    db = args.db.expanduser().resolve(strict=True)
    root = args.source_root.expanduser().resolve(strict=True)
    if db.suffix != '.duckdb' or root == db.parent or db.is_symlink():
        raise BenchmarkError('expected separate real DuckDB and Parquet source root')
    if not re.fullmatch('[0-9a-f]{64}', args.expected_main_sha256):
        raise BenchmarkError('--expected-main-sha256 must be an exact SHA-256')
    if not args.cycles or any(c not in CYCLES for c in args.cycles) or len(set(args.cycles)) != len(args.cycles):
        raise BenchmarkError('cycles must be unique and from approved five-cycle set')
    selected = tuple(args.variants)
    ordered_variants(1, selected)  # Refuse duplicate or unsupported arms before creating output.
    if (args.repeats < 1 or args.threads < 1 or args.max_seconds <= 0 or
        args.max_temp_gb < 1 or args.lock_timeout < 0):
        raise BenchmarkError('invalid nonpositive resource limit')
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise BenchmarkError('output directory must be new (create-only)')
    if db.parent in args.output_dir.expanduser().absolute().parents:
        raise BenchmarkError('benchmark output must be OUTSIDE the live warehouse directory')

    assert_wal_free(db)
    locks = [evidence.lock_path(db)] + [evidence.partition_for(root, c) / '.publish.lock'
                                              for c in args.cycles]
    # Fail fast if an ongoing graph certification holds the exclusive inode.
    with evidence.noncreating_shared_locks(locks, timeout=args.lock_timeout):
        assert_wal_free(db)
        main_sha = sha256(db)
        if main_sha != args.expected_main_sha256:
            raise BenchmarkError('main database digest differs from explicitly approved snapshot')
        epoch = lifecycle_epoch(db)
        # No files in the target path before all input checks have passed.
        args.output_dir.mkdir(parents=True, mode=0o700)
        results = []
        cardinalities = []
        guards = []
        try:
            for cycle in args.cycles:
                guard = None
                if 'signed_countstar' in selected and not args.plan_only:
                    guard = prove_zone_uniqueness(
                        db=db, root=root, cycle=cycle, out=args.output_dir,
                        memory_limit=args.memory_limit, threads=args.threads,
                        max_seconds=args.max_seconds, max_temp_gb=args.max_temp_gb)
                    guards.append(guard)
                if args.probe_cardinality and not args.plan_only:
                    cardinalities.append(cardinality_probe(
                        db=db, root=root, cycle=cycle, out=args.output_dir,
                        memory_limit=args.memory_limit, threads=args.threads,
                        max_seconds=args.max_seconds, max_temp_gb=args.max_temp_gb))
                for rep in range(1, args.repeats + 1):
                    order = ordered_variants(rep, selected)
                    for variant in order:
                        r = run_trial(db=db, root=root, cycle=cycle, variant=variant,
                                      repeat=rep, out=args.output_dir,
                                      memory_limit=args.memory_limit, threads=args.threads,
                                      max_seconds=args.max_seconds, max_temp_gb=args.max_temp_gb,
                                      explain_only=args.plan_only, uniqueness_guard=guard)
                        if guard is not None and r['parquet'] != guard['source_publication']:
                            raise BenchmarkError('source publication changed after uniqueness proof')
                        results.append(r)
            if lifecycle_epoch(db) != epoch or sha256(db) != main_sha or Path(str(db) + '.wal').exists():
                raise BenchmarkError('warehouse SHA, WAL or epoch changed during benchmark')
            report = {
                'result': ('PLAN_ONLY' if args.plan_only else
                           ('PASS' if len(selected) == 2 else 'SINGLE_ARM_PASS')),
                'scope': 'readonly_graph_candidate_conservation_comparison_not_freshness',
                'db': str(db), 'main_sha256': main_sha,
                'lifecycle_epoch': epoch, 'source_root': str(root),
                'cycles': list(args.cycles), 'repeats': args.repeats,
                'variants': list(selected),
                'paired_comparison_complete': len(selected) == 2 and not args.plan_only,
                'results': results, 'source_cardinality': cardinalities,
                'independent_countstar_guards': guards,
                'performance_comparison': speedups(results), 'limitations': [
                    'Not a graph freshness certificate; no SQL/model changes made',
                    'Sequential DuckDB processes share operating-system file cache',
                    'Explicit memory limit and threads control DuckDB, not total host RSS',
                    'Spill inference requires the JSON profile, not elapsed time alone',
                    'Single-arm runs cannot establish speedup without paired measurements',
                    'COUNT(*) candidate-zone substitution only after independent full source uniqueness guard',
                    'Only in-memory experimental graph view is changed; canonical Gold and assertions remain unchanged',
                ],
            }
            durable_json(args.output_dir / 'summary.json', report)
            return report
        finally:
            # Deliberately preserve partial reports, plans and profiles on errors.
            pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--expected-main-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cycles', nargs='+', default=['2020:municipal'])
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--variants', nargs='+', choices=('except_all', 'signed', 'signed_countstar'),
                   default=['except_all', 'signed'],
                   help='Select only signed or except_all for a diagnostic, or both for paired benchmarking')
    p.add_argument('--memory-limit', default='4GB')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--max-seconds', type=int, default=2400)
    p.add_argument('--max-temp-gb', type=int, default=6)
    p.add_argument('--lock-timeout', type=float, default=0,
                   help='Fail fast if graph certification or publisher holds lock')
    p.add_argument('--plan-only', action='store_true', help='Produce EXPLAIN only, without data scans')
    p.add_argument('--probe-cardinality', action='store_true', help='Add one expensive exact physical-source group/zone count per cycle')
    args = p.parse_args(argv)
    try:
        report = execute(args)
        print(json.dumps({'result': report['result'], 'summary': str(args.output_dir / 'summary.json'),
                          'trials': len(report['results'])}, indent=2))
        return 0
    except Exception as exc:
        print(f'GRAPH BENCHMARK DENIED/FAILED: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
