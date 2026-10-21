#!/usr/bin/env python3
"""Run C3 end-to-end publication experiments using *only disposable DuckDB copies*.

Opt-in gate: requires a prepared authenticated reference, source inventory,
manifest and an intact candidate warehouse. No canonical data are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


CYCLES = ((2018, 'general'), (2020, 'municipal'), (2022, 'general'),
          (2024, 'municipal'), (2026, 'general'))
MODELS = ('silver_tally_munzona', 'fact_tally_munzona')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def snapshot(db: Path) -> dict:
    import duckdb
    con = duckdb.connect(str(db), read_only=True)
    try:
        return {model: con.execute(
            f'SELECT election_year, election_type, count(*), sum(total_votes) '
            f'FROM {model} GROUP BY 1,2 ORDER BY 1,2'
        ).fetchall() for model in MODELS}
    finally:
        con.close()


def assert_relations_equivalent(before: Path, after: Path, models=MODELS) -> None:
    """Exact bag equality over every column; never rely only on COUNT/SUM."""
    import duckdb
    con = duckdb.connect(str(before), read_only=True)
    try:
        # The candidate DB is attached read-only, and neither file is changed.
        candidate_path = str(after).replace("'", "''")
        con.execute(
            f"ATTACH '{candidate_path}' AS candidate (READ_ONLY)"
        )
        for model in models:
            if model not in MODELS:
                raise ValueError(f"unexpected C3 relation: {model}")
            bcols = con.execute(f"DESCRIBE SELECT * FROM main.{model}").fetchall()
            acols = con.execute(f"DESCRIBE SELECT * FROM candidate.main.{model}").fetchall()
            if [(r[0], r[1]) for r in bcols] != [(r[0], r[1]) for r in acols]:
                raise AssertionError(f"{model}: schema differs")
            for lhs, rhs in (("main", "candidate.main"), ("candidate.main", "main")):
                mismatch = con.execute(f"""
                    SELECT EXISTS(
                        (SELECT * FROM {lhs}.{model}
                         EXCEPT ALL
                         SELECT * FROM {rhs}.{model})
                    )
                """).fetchone()[0]
                if mismatch:
                    raise AssertionError(f"{model}: bidirectional EXCEPT ALL failed")
    finally:
        con.close()


def mutate(db: Path, scenario: str) -> None:
    import duckdb
    con = duckdb.connect(str(db))
    try:
        if scenario == 'partial':
            con.execute("DELETE FROM bronze_tally_raw WHERE election_year=2022 AND election_type='general' AND uf='GO'")
        elif scenario == 'measure':
            con.execute("""UPDATE bronze_tally_raw SET total_votes=total_votes+1 WHERE rowid=(
                SELECT min(rowid) FROM bronze_tally_raw WHERE election_year=2022
                AND election_type='general' AND total_votes IS NOT NULL)""")
        elif scenario == 'missing':
            con.execute("DELETE FROM bronze_tally_raw WHERE election_year=2022 AND election_type='general'")
        else:
            raise ValueError(f'unknown mutation: {scenario}')
    finally:
        con.close()


def run_scenario(*, gateway: Path, python: Path, dbt: Path, baseline: Path,
                 reference: Path, evidence: Path, raw_root: Path, workspace: Path,
                 scenario: str) -> dict:
    dest = workspace / f'{scenario}.duckdb'
    shutil.copy2(baseline, dest)
    if scenario in {'partial', 'measure', 'missing'}:
        mutate(dest, scenario)
    expected = snapshot(dest)
    # Freeze the exact Silver/Gold relation contents before the attempted run.
    before = workspace / f'{scenario}-before.duckdb'
    if scenario != 'intact':
        shutil.copy2(dest, before)
    local_evidence = evidence
    local_raw_root = raw_root
    if scenario == 'stale':
        obj = json.loads(evidence.read_text(encoding='utf-8'))
        obj['reference_sha256'] = '0'*64
        local_evidence = workspace/'stale.json'
        local_evidence.write_text(json.dumps(obj), encoding='utf-8')
    if scenario == 'wrong_root':
        local_raw_root = workspace/'uncertified_raw'
    vars_obj = {
        'tse_raw_root': str(local_raw_root),
        'election_years': [2018,2020,2022,2024,2026],
        'election_types': ['general','municipal'],
        'incremental_years': [2018,2020,2022,2024,2026],
        'incremental_election_types': ['general','municipal'],
    }
    command = [str(python), str(gateway), '--db', str(dest),
        '--dbt-executable', str(dbt), '--artifacts-root', str(workspace/f'{scenario}-artifacts'),
        '--c3-reference-db', str(reference), '--c3-reference-raw-root', str(raw_root),
        '--c3-candidate-raw-root', str(raw_root), '--c3-evidence', str(local_evidence),
        '--', 'run', '--project-dir', 'tse_dbt', '--profiles-dir', 'tse_dbt',
        '--select', *MODELS, '--vars', json.dumps(vars_obj)]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    (workspace/f'{scenario}.log').write_text(proc.stdout+'\n'+proc.stderr, encoding='utf-8')
    unchanged = snapshot(dest) == expected
    accepted = scenario == 'intact'
    if not accepted and unchanged:
        try:
            assert_relations_equivalent(before, dest)
        except AssertionError as exc:
            unchanged = False
            with (workspace / f'{scenario}.log').open('a', encoding='utf-8') as log:
                log.write(f"\nC3_EXACT_TARGET_DIFF: {exc}\n")
    ok = (proc.returncode == 0) if accepted else (proc.returncode != 0 and unchanged)
    return {'scenario': scenario, 'status': 'PASS' if ok else 'FAIL',
            'returncode': proc.returncode, 'target_preserved': unchanged,
            'log': str(workspace/f'{scenario}.log')}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-db', 'reference-db', 'evidence', 'raw-root', 'python', 'dbt'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    paths = (args.baseline_db,args.reference_db,args.evidence,args.raw_root,args.python,args.dbt)
    if not all(p.exists() for p in paths):
        parser.error('all input paths must exist')
    manifest = json.loads(args.evidence.read_text(encoding='utf-8'))
    if digest(args.reference_db) != manifest.get('reference_sha256'):
        parser.error('reference database does not match evidence SHA-256')
    gateway = Path(__file__).resolve().parent/'run_coordinated_dbt.py'
    if not gateway.exists():
        parser.error('gateway not found')
    # Output directory itself is unique; never overwrite an existing experiment.
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for scenario in ('intact', 'missing', 'partial', 'measure', 'stale', 'wrong_root'):
        result = run_scenario(gateway=gateway, python=args.python, dbt=args.dbt,
            baseline=args.baseline_db, reference=args.reference_db,
            evidence=args.evidence, raw_root=args.raw_root, workspace=args.output,
            scenario=scenario)
        results.append(result)
        print(f"{scenario}: {result['status']} (dbt/gateway rc={result['returncode']})", flush=True)
    (args.output/'matrix.json').write_text(json.dumps(results, indent=2)+'\n',encoding='utf-8')
    return 0 if all(row['status']=='PASS' for row in results) else 1

if __name__ == '__main__':
    raise SystemExit(main())
