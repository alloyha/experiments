#!/usr/bin/env python3
"""CCC B2: inspect persisted dim_party identity without opening a write connection.

This only audits already materialized DuckDB tables. It does not establish that
an election cycle was fully ingested, that external consumers have migrated,
or that a GG physical certificate is valid.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")
REQUIRED = frozenset({
    "election_year", "election_type", "election_code", "party_number",
    "cycle_id", "election_id", "party_id",
})
CYCLE_PATTERN = re.compile(r"^([0-9]{4}):(general|municipal)$")


def relation(schema: str) -> str:
    if not SCHEMA_PATTERN.fullmatch(schema):
        raise ValueError("schema must contain lowercase letters, digits and underscores")
    return f'"{schema}"."dim_party"'


def inspect_catalog(conn: Any, schema: str) -> None:
    """Only a persistent table in the requested schema can certify materialization."""
    result = conn.execute(
        "SELECT table_type FROM information_schema.tables "
        "WHERE table_catalog = current_database() "
        "AND table_schema = ? AND table_name = 'dim_party'", [schema],
    ).fetchall()
    if result != [("BASE TABLE",)]:
        raise ValueError(f"Expected exactly one materialized BASE TABLE: {schema}.dim_party; found {result}")
    columns = {row[0] for row in conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_catalog = current_database() "
        "AND table_schema = ? AND table_name = 'dim_party'", [schema],
    ).fetchall()}
    missing = REQUIRED - columns
    if missing:
        raise ValueError("dim_party is missing columns: " + ", ".join(sorted(missing)))


def summarize_rows(conn: Any, schema: str) -> list[dict[str, Any]]:
    """Return disjoint or independently meaningful invalid identity metrics."""
    name = relation(schema)
    sql = f"""
        WITH prepared AS (
          SELECT election_year, election_type, election_code, party_number,
                 cycle_id, election_id, party_id,
                 CAST(election_year AS VARCHAR) || ':' || CAST(election_type AS VARCHAR)
                   AS expected_cycle,
                 CAST(election_year AS VARCHAR) || ':' || CAST(election_type AS VARCHAR)
                   || ':' || CAST(election_code AS VARCHAR) AS expected_election,
                 CAST(election_id AS VARCHAR) || ':' || CAST(party_number AS VARCHAR)
                   AS expected_party,
                 CAST(election_id AS VARCHAR) || ':' || CAST(election_code AS VARCHAR)
                   || ':' || CAST(party_number AS VARCHAR) AS legacy_party
          FROM {name}
        )
        SELECT election_year, election_type, COUNT(*) AS rows,
            SUM(CASE WHEN election_year IS NULL OR election_type IS NULL
                          OR election_code IS NULL OR party_number IS NULL
                          OR cycle_id IS NULL OR election_id IS NULL OR party_id IS NULL
                     THEN 1 ELSE 0 END) AS null_key_rows,
            SUM(CASE WHEN cycle_id IS DISTINCT FROM expected_cycle
                     THEN 1 ELSE 0 END) AS invalid_cycle_ids,
            SUM(CASE WHEN election_id IS DISTINCT FROM expected_election
                     THEN 1 ELSE 0 END) AS invalid_election_ids,
            SUM(CASE WHEN party_id IS DISTINCT FROM expected_party
                     THEN 1 ELSE 0 END) AS noncanonical_party_ids,
            SUM(CASE WHEN party_id = legacy_party
                          AND party_id IS DISTINCT FROM expected_party
                     THEN 1 ELSE 0 END) AS legacy_party_ids,
            COUNT(DISTINCT party_id) AS distinct_party_ids,
            SUM(CASE WHEN party_id IS NULL THEN 1 ELSE 0 END) AS null_party_ids
        FROM prepared
        GROUP BY election_year, election_type
        ORDER BY election_year, election_type
    """
    metrics = [
        "election_year", "election_type", "rows", "null_key_rows",
        "invalid_cycle_ids", "invalid_election_ids", "noncanonical_party_ids",
        "legacy_party_ids", "distinct_party_ids", "null_party_ids",
    ]
    result = [dict(zip(metrics, row)) for row in conn.execute(sql).fetchall()]
    dup_sql = f"""
        SELECT election_year, election_type, SUM(n - 1)
        FROM (
          SELECT election_year, election_type, election_code, party_number,
                 COUNT(*) AS n
          FROM {name}
          GROUP BY election_year, election_type, election_code, party_number
          HAVING COUNT(*) > 1
        ) duplicated
        GROUP BY election_year, election_type
    """
    duplicates = {(year, kind): int(count) for year, kind, count in
                  conn.execute(dup_sql).fetchall()}
    for record in result:
        record["duplicate_grain_rows"] = duplicates.get(
            (record["election_year"], record["election_type"]), 0,
        )
        record["duplicate_party_id_rows"] = max(
            0, record["rows"] - record["distinct_party_ids"]
            - record["null_party_ids"],
        )
        record["cycle"] = f'{record["election_year"]}:{record["election_type"]}'
        bad = ("null_key_rows", "invalid_cycle_ids", "invalid_election_ids",
               "noncanonical_party_ids", "duplicate_grain_rows",
               "duplicate_party_id_rows")
        record["status"] = "DRIFT" if any(record[k] for k in bad) else "PASS"
    return result


def assess(rows: list[dict[str, Any]], expected: list[str]) -> dict[str, Any]:
    known = {row["cycle"] for row in rows}
    missing = sorted(set(expected) - known)
    anomalous = sorted(row["cycle"] for row in rows
                       if not CYCLE_PATTERN.fullmatch(row["cycle"]))
    drift = any(row["status"] == "DRIFT" for row in rows)
    if not rows:
        status = "EMPTY"
    elif missing or anomalous or drift:
        status = "FAIL"
    else:
        status = "PASS"
    return {"status": status, "total_rows": sum(r["rows"] for r in rows),
            "cycles": rows, "expected_cycles": sorted(set(expected)),
            "missing_expected_cycles": missing, "invalid_cycle_names": anomalous,
            "scope_note": "Only observed rows/cycles were inspected; external consumers, physical Parquet and source completeness are not certified."}


def audit_database(db: Path, schema: str, expected: list[str]) -> dict[str, Any]:
    db = db.expanduser().resolve(strict=True)
    if not db.is_file():
        raise ValueError(f"Not a DuckDB database file: {db}")
    import duckdb
    # read_only=True is intentional; never fall back to writable open.
    with duckdb.connect(str(db), read_only=True) as conn:
        inspect_catalog(conn, schema)
        result = assess(summarize_rows(conn, schema), expected)
    return {"warehouse": str(db), "relation": f"{schema}.dim_party", **result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path,
                        help="Existing materialized DuckDB; never automatically create a database")
    parser.add_argument("--schema", default="main")
    parser.add_argument("--expect-cycle", action="append", default=[],
                        help="Expected cycle YEAR:general or YEAR:municipal; repeat for each cycle")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    try:
        relation(args.schema)
        for cycle in args.expect_cycle:
            if not CYCLE_PATTERN.fullmatch(cycle):
                raise ValueError(f"Invalid --expect-cycle: {cycle!r}")
        report = audit_database(args.db, args.schema, args.expect_cycle)
    except (OSError, ValueError, ImportError) as exc:
        print(f"CCC B2 · UNAVAILABLE · {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        # DuckDB locks/read errors and catalog failures must not become PASS.
        print(f"CCC B2 · ERROR · {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    if args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f'CCC B2 · {report["status"]} · table={report["relation"]} rows={report["total_rows"]}')
        for r in report["cycles"]:
            print(f'  {r["cycle"]}: {r["status"]} rows={r["rows"]} '
                  f'legacy={r["legacy_party_ids"]} noncanonical={r["noncanonical_party_ids"]} '
                  f'dup_grain={r["duplicate_grain_rows"]} dup_id={r["duplicate_party_id_rows"]} '
                  f'null={r["null_key_rows"]} wrong_cycle={r["invalid_cycle_ids"]} '
                  f'wrong_election={r["invalid_election_ids"]}')
        for name in report["missing_expected_cycles"]:
            print(f'  MISSING expected cycle: {name}')
        for name in report["invalid_cycle_names"]:
            print(f'  INVALID observed cycle: {name}')
        print("  NOTE: Historical coverage, external consumers and GG certification require independent evidence.")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
