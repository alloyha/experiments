#!/usr/bin/env python3
"""C3 read-only Bronze parity verifier against independently rebuilt dbt Bronze.

Never writes a DuckDB database. An external orchestration gate must verify the
reference was freshly built from authenticated immutable inputs, and must fence
all writers until the downstream dbt transaction has committed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

CYCLES = ((2018, 'general'), (2020, 'municipal'), (2022, 'general'),
          (2024, 'municipal'), (2026, 'general'))
TABLE = 'bronze_tally_raw'


def literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def reference_columns(con, catalog: str) -> dict[str, str]:
    rows = con.execute('''SELECT column_name, data_type
        FROM information_schema.columns WHERE table_catalog = ?
          AND table_schema = 'main' AND table_name = ?''', [catalog, TABLE]).fetchall()
    if not rows:
        raise ValueError(f'{catalog}.main.{TABLE} missing')
    return dict(rows)


def certified_expression(name: str, alias: str, root: Path) -> str:
    value = alias + '.' + identifier(name)
    if name == 'source_file':
        # Physical root is environment-specific; all other file components
        # including resource id and content SHA remain part of the identity.
        # Prefix match includes a trailing slash to prevent near-prefix errors.
        prefix = root.resolve().as_posix().rstrip('/') + '/'
        return (f"CASE WHEN starts_with({value}, {literal(prefix)}) "
                f"THEN substr({value}, length({literal(prefix)}) + 1) "
                f"ELSE {value} END")
    return value


def audit(reference_db: Path, candidate_db: Path, reference_raw_root: Path,
          candidate_raw_root: Path, cycles=CYCLES) -> dict:
    import duckdb
    ref = reference_db.resolve(strict=True)
    candidate = candidate_db.resolve(strict=True)
    if ref == candidate:
        raise ValueError('reference and candidate database must differ')
    con = duckdb.connect(str(ref), read_only=True)
    try:
        con.execute(f'ATTACH {literal(candidate)} AS candidate (READ_ONLY)')
        reference_columns_map = reference_columns(con, ref.stem)
        candidate_columns_map = reference_columns(con, 'candidate')
        if reference_columns_map != candidate_columns_map:
            raise ValueError('schema mismatch: ' + json.dumps({
                'reference': reference_columns_map, 'candidate': candidate_columns_map}, sort_keys=True))
        names = sorted(reference_columns_map)
        output = []
        for year, election_type in cycles:
            where = f'election_year = {int(year)} AND election_type = {literal(election_type)}'
            proj_ref = ', '.join(certified_expression(n, 'r', reference_raw_root)
                                 + ' AS ' + identifier(n) for n in names)
            proj_cand = ', '.join(certified_expression(n, 'c', candidate_raw_root)
                                  + ' AS ' + identifier(n) for n in names)
            query = f'''WITH ref_rows AS (
                SELECT {proj_ref} FROM main.{TABLE} r WHERE {where}
            ), candidate_rows AS (
                SELECT {proj_cand} FROM candidate.main.{TABLE} c WHERE {where}
            )
            SELECT (SELECT count(*) FROM ref_rows),
                   (SELECT count(*) FROM candidate_rows),
                   (SELECT count(*) FROM (SELECT * FROM ref_rows EXCEPT ALL SELECT * FROM candidate_rows)),
                   (SELECT count(*) FROM (SELECT * FROM candidate_rows EXCEPT ALL SELECT * FROM ref_rows))'''
            expected, observed, missing, unexpected = con.execute(query).fetchone()
            output.append({'cycle': f'{year}:{election_type}', 'reference': expected,
                           'candidate': observed, 'missing': missing, 'unexpected': unexpected,
                           'pass': expected > 0 and missing == 0 and unexpected == 0})
        return {'status': 'PASS' if all(r['pass'] for r in output) else 'FAIL',
                'read_only': True, 'table': TABLE, 'cycles': output,
                'note': 'Parities only; does not prove origin authenticity or writer fencing'}
    finally:
        con.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-db', required=True, type=Path)
    p.add_argument('--candidate-db', required=True, type=Path)
    p.add_argument('--reference-raw-root', required=True, type=Path)
    p.add_argument('--candidate-raw-root', required=True, type=Path)
    args = p.parse_args()
    try:
        result = audit(args.reference_db, args.candidate_db,
                       args.reference_raw_root, args.candidate_raw_root)
    except Exception as exc:
        print(json.dumps({'status': 'ERROR', 'error': str(exc)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
