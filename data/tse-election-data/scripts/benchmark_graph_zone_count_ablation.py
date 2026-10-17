#!/usr/bin/env python3
"""Read-only ablation of COUNT(DISTINCT zone) on the pinned candidate fact.

Compares the same exact eight-key rollup with/without distinct-zone counting.
Does not modify graph SQL, Gold, Parquet publications or certificates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_graph_candidate_conservation as bench
from scripts import consolidate_candidate_cube_transfer as evidence
from scripts import certify_gold_graph_cycle as diagnostic
from scripts.warehouse_lifecycle import lifecycle_epoch

ARMS = ('no_distinct', 'with_distinct')
KEYS = ('cycle_id', 'election_id', 'round_number', 'uf', 'municipality_code',
        'office_code', 'is_transit_vote', 'candidate_id')
OUTPUT_COLUMNS = ('groups', 'source_rows', 'votes', 'nonnull_zone_rows')
SCOPE = 'readonly_graph_candidate_rollup_distinct_zone_ablation_not_certification'


def build_query(variant: str, uf: str | None = None) -> str:
    """Both arms have identical grouping keys, sums and ordinary counts."""
    if variant not in ARMS:
        raise bench.BenchmarkError('unsupported zone-count ablation arm')
    if uf is not None and not re.fullmatch('[A-Z]{2}', uf):
        raise bench.BenchmarkError('UF must consist of two uppercase ASCII letters')
    where = 'WHERE uf = ' + diagnostic.quote_sql(uf) if uf is not None else ''
    distinct_inner = ', COUNT(DISTINCT zone)::BIGINT AS zone_count' if variant == 'with_distinct' else ''
    distinct_outer = ', COALESCE(SUM(zone_count),0)::HUGEINT AS distinct_zone_total' if variant == 'with_distinct' else ''
    return f"""WITH rolled AS (
        SELECT {', '.join(KEYS)},
               SUM(nominal_valid_votes)::HUGEINT AS votes,
               COUNT(*)::BIGINT AS zone_rows,
               COUNT(zone)::BIGINT AS nonnull_zones
               {distinct_inner}
        FROM fact_candidate_votes
        {where}
        GROUP BY {', '.join(KEYS)}
    )
    SELECT COUNT(*)::HUGEINT AS groups,
           COALESCE(SUM(zone_rows),0)::HUGEINT AS source_rows,
           COALESCE(SUM(votes),0)::HUGEINT AS votes,
           COALESCE(SUM(nonnull_zones),0)::HUGEINT AS nonnull_zone_rows
           {distinct_outer}
    FROM rolled"""


def unpack_row(variant: str, row: tuple) -> dict:
    expected = 5 if variant == 'with_distinct' else 4
    if len(row) != expected:
        raise bench.BenchmarkError('unexpected ablation scalar result width')
    values = {key: int(value) for key, value in zip(OUTPUT_COLUMNS, row[:4])}
    if values['groups'] <= 0 or values['source_rows'] < values['groups']:
        raise bench.BenchmarkError('no groups or invalid source row inventory')
    if not (0 <= values['nonnull_zone_rows'] <= values['source_rows']):
        raise bench.BenchmarkError('invalid non-NULL zone inventory')
    if variant == 'with_distinct':
        values['distinct_zone_total'] = int(row[4])
        if not (0 <= values['distinct_zone_total'] <= values['nonnull_zone_rows']):
            raise bench.BenchmarkError('invalid distinct-zone inventory')
        values['null_zone_rows'] = values['source_rows'] - values['nonnull_zone_rows']
        values['duplicate_nonnull_zone_rows'] = values['nonnull_zone_rows'] - values['distinct_zone_total']
        # Nonnegative per-group differences imply that equality of global sums
        # proves equality for EVERY group, not merely aggregate totals.
        values['count_star_exact_substitute_for_distinct'] = (
            values['source_rows'] == values['distinct_zone_total'])
        values['count_zone_exact_substitute_for_distinct'] = (
            values['nonnull_zone_rows'] == values['distinct_zone_total'])
    return values


def prove_shared_invariants(records: list[dict]) -> dict:
    if len(records) != 2 or {record['variant'] for record in records} != set(ARMS):
        raise bench.BenchmarkError('paired distinct/no-distinct records required')
    a, b = records
    comparable = OUTPUT_COLUMNS
    if any(a['values'][field] != b['values'][field] for field in comparable):
        raise bench.BenchmarkError('ablation arms disagree on shared aggregates')
    full = next(item for item in records if item['variant'] == 'with_distinct')
    quick = next(item for item in records if item['variant'] == 'no_distinct')
    a_time, b_time = full['elapsed_seconds'], quick['elapsed_seconds']
    return {'shared_aggregates_equal': True,
            'with_distinct_seconds': a_time, 'no_distinct_seconds': b_time,
            'speedup_without_distinct': round(a_time / b_time, 4) if b_time > 0 else None,
            'distinct_zone_semantics': {k: full['values'][k] for k in (
                'null_zone_rows', 'duplicate_nonnull_zone_rows',
                'count_star_exact_substitute_for_distinct',
                'count_zone_exact_substitute_for_distinct')},
            'limitation': 'sequential timings, not independent cold-cache trials'}


def measure(*, db: Path, root: Path, cycle: str, variant: str, uf: str | None,
            out: Path, memory_limit: str, threads: int, max_seconds: int,
            max_temp_gb: int) -> dict:
    import duckdb
    y, election_type = cycle.split(':')
    parquet, before = diagnostic.selected_publication(root, int(y), election_type)
    prefix = f'{cycle.replace(":", "-")}-{uf or "all"}-{variant}'
    profile = out / (prefix + '.profile.json')
    result = out / (prefix + '.result.json')
    spill = out / 'spill' / prefix
    spill.mkdir(parents=True, exist_ok=False)
    sql = build_query(variant, uf)
    con = duckdb.connect(':memory:')
    try:
        con.execute('SET memory_limit = ' + diagnostic.quote_sql(memory_limit))
        con.execute(f'SET threads = {threads}')
        con.execute('SET temp_directory = ' + diagnostic.quote_sql(str(spill)))
        con.execute('SET max_temp_directory_size = ' + diagnostic.quote_sql(f'{max_temp_gb}GB'))
        diagnostic.setup_scoped_views(con, db, parquet, int(y), election_type)
        bench.configure_json_profiler(con, profile)
        started = time.perf_counter()
        row = bench.fetchone_with_timeout(con, sql, max_seconds=max_seconds, label=prefix)
        elapsed = round(time.perf_counter() - started, 3)
    finally:
        con.close()
    values = unpack_row(variant, row)
    perf = bench.metrics(profile)
    _, after = diagnostic.selected_publication(root, int(y), election_type)
    if before != after:
        raise bench.BenchmarkError('Parquet changed during zone-count ablation')
    record = {'scope': SCOPE, 'variant': variant, 'cycle': cycle, 'uf': uf,
              'values': values, 'elapsed_seconds': elapsed,
              'profile_path': str(profile), 'profiling': perf,
              'query_sha256': hashlib.sha256(sql.encode('utf-8')).hexdigest(),
              'source_publication': after, 'duckdb_version': duckdb.__version__,
              'memory_limit': memory_limit, 'threads': threads,
              'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    bench.durable_json(result, record)
    print(f'[zone-ablation] {prefix}: seconds={elapsed}, groups={values["groups"]}, '
          f'rows={values["source_rows"]}, peak_temp={perf.get("system_peak_temp_dir_size")}', flush=True)
    return record


def execute(args: argparse.Namespace) -> dict:
    if args.db.is_symlink() or args.source_root.is_symlink():
        raise bench.BenchmarkError('database/source path must not be symlinked')
    db = args.db.expanduser().resolve(strict=True)
    root = args.source_root.expanduser().resolve(strict=True)
    if db.suffix != '.duckdb' or not root.is_dir():
        raise bench.BenchmarkError('unsafe database or source path')
    if not re.fullmatch('[0-9a-f]{64}', args.expected_main_sha256):
        raise bench.BenchmarkError('must provide full Gold SHA-256')
    if args.cycle not in bench.CYCLES:
        raise bench.BenchmarkError('cycle outside the five approved publications')
    if args.uf is not None and not re.fullmatch('[A-Z]{2}', args.uf):
        raise bench.BenchmarkError('invalid UF')
    if args.threads <= 0 or args.max_seconds <= 0 or args.max_temp_gb <= 0 or args.lock_timeout < 0:
        raise bench.BenchmarkError('invalid resource limits')
    output = args.output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink() or db.parent == output or db.parent in output.parents:
        raise bench.BenchmarkError('output must be a new directory outside the warehouse')
    bench.assert_wal_free(db)
    locks = [evidence.lock_path(db), evidence.partition_for(root, args.cycle) / '.publish.lock']
    with evidence.noncreating_shared_locks(locks, timeout=args.lock_timeout):
        if bench.sha256(db) != args.expected_main_sha256:
            raise bench.BenchmarkError('Gold SHA has changed')
        epoch = lifecycle_epoch(db)
        output.mkdir(mode=0o700, parents=False)
        records = []
        for arm in ARMS:  # Cheap arm first, then original DISTINCT; each persisted.
            print(f'[zone-ablation] {args.cycle} {args.uf or "all"}: {arm}', flush=True)
            records.append(measure(db=db, root=root, cycle=args.cycle,
                                   variant=arm, uf=args.uf, out=output,
                                   memory_limit=args.memory_limit, threads=args.threads,
                                   max_seconds=args.max_seconds, max_temp_gb=args.max_temp_gb))
        if (lifecycle_epoch(db) != epoch or bench.sha256(db) != args.expected_main_sha256
                or Path(str(db) + '.wal').exists()):
            raise bench.BenchmarkError('Gold identity drift during ablation')
        comparison = prove_shared_invariants(records)
        report = {'result': 'PASS', 'scope': SCOPE, 'cycle': args.cycle, 'uf': args.uf,
                  'gold_sha256': args.expected_main_sha256, 'lifecycle_epoch': epoch,
                  'shared_comparison': comparison, 'trials': records,
                  'limitations': ['No Gold writes, graph/model changes or certification',
                                  'Separate sequential timings subject to cache and CPU variance',
                                  'Only direct same-group source substitution is assessed; graph contract still unchanged',
                                  'Cooperative locks do not fence arbitrary DuckDB/Parquet writers']}
        bench.durable_json(output / 'summary.json', report)
        return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--expected-main-sha256', required=True)
    p.add_argument('--cycle', choices=bench.CYCLES, required=True)
    p.add_argument('--uf', help='Optional bounded region (e.g., AC) before all states')
    p.add_argument('--memory-limit', default='4GB')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--max-seconds', type=int, default=900)
    p.add_argument('--max-temp-gb', type=int, default=6)
    p.add_argument('--lock-timeout', type=float, default=5.0)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    try:
        report = execute(args)
        print(json.dumps({'result': report['result'], 'summary': str(args.output_dir / 'summary.json'),
                          'comparison': report['shared_comparison']}, indent=2))
        return 0
    except Exception as exc:
        print(f'ZONE ABLATION DENIED/FAILED: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
