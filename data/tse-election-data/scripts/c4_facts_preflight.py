#!/usr/bin/env python3
"""CCC C4-A: read-only evidence probe for candidate votes and turnout facts.

This is NOT ODCS publication nor a historical GG certificate. Large scans are
opt-in; results distinguish PASS, FAIL and NOT_EVALUATED.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

FACTS = {
    "fact_candidate_votes": (
        "election_year", "election_type", "election_code", "round_number",
        "uf", "municipality_code", "zone", "office_code", "candidate_id",
        "is_transit_vote",
    ),
    "fact_turnout": (
        "election_year", "election_type", "election_code", "round_number",
        "uf", "municipality_code", "zone", "office_code", "is_transit_vote",
    ),
}


def entry(status: str, *, detail: str = "", **data):
    return {"status": status, "detail": detail, **data}


def _safe_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def audit(root: Path, db_path: Path | None = None, *, scan: bool = False) -> dict:
    root = root.resolve()
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CCC C4-A advisory, read-only; not ODCS or GG certification",
        "facts": {},
    }
    con = None
    connection_error = "no --db supplied"
    if db_path is not None:
        try:
            import duckdb
            con = duckdb.connect(str(db_path.expanduser().resolve()), read_only=True)
            connection_error = ""
        except Exception as exc:
            connection_error = f"read-only DuckDB connection unavailable: {type(exc).__name__}: {exc}"

    try:
        for name, grain in FACTS.items():
            sql_path = root / "tse_dbt" / "models" / "gold" / "facts" / f"{name}.sql"
            result = {"expected_grain": list(grain), "checks": {}}
            checks = result["checks"]
            if sql_path.is_file():
                sql = sql_path.read_text(encoding="utf-8")
                checks["source_model"] = entry("PASS", detail=str(sql_path.relative_to(root)))
                if name == "fact_candidate_votes":
                    ok = "read_parquet(" in sql and "hive_partitioning = true" in sql
                    checks["source_semantics"] = entry("PASS" if ok else "FAIL", detail="Parquet-backed Hive partition source")
                else:
                    ok = "partition_replace_pre_hook()" in sql and "incremental_strategy='delete+insert'" in sql
                    checks["source_semantics"] = entry("PASS" if ok else "FAIL", detail="Incremental partition replacement (static inspection only)")
            else:
                checks["source_model"] = entry("FAIL", detail=f"missing SQL: {sql_path}")
                checks["source_semantics"] = entry("NOT_EVALUATED", detail="SQL missing")

            if con is None:
                for key in ("relation", "grain_columns", "cycles", "null_grain", "duplicate_grain"):
                    checks[key] = entry("NOT_EVALUATED", detail=connection_error)
                report["facts"][name] = result
                continue
            try:
                cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {_safe_identifier(name)}").fetchall()]
                checks["relation"] = entry("PASS", detail="relation accessible in read-only DuckDB")
                missing = sorted(set(grain) - set(cols))
                checks["grain_columns"] = entry("FAIL" if missing else "PASS", missing=missing, columns=cols)
                if missing:
                    for key in ("cycles", "null_grain", "duplicate_grain"):
                        checks[key] = entry("NOT_EVALUATED", detail="grain fields missing")
                elif not scan:
                    for key in ("cycles", "null_grain", "duplicate_grain"):
                        checks[key] = entry("NOT_EVALUATED", detail="full scans require --scan")
                else:
                    qname = _safe_identifier(name)
                    cycles = con.execute(
                        f"SELECT election_year, election_type, count(*) AS n FROM {qname} "
                        "GROUP BY 1,2 ORDER BY 1,2"
                    ).fetchall()
                    checks["cycles"] = entry("PASS" if cycles else "FAIL", observed=[
                        {"year": y, "type": t, "rows": n} for y, t, n in cycles
                    ])
                    predicate = " OR ".join(f"{_safe_identifier(c)} IS NULL" for c in grain)
                    null_count = con.execute(f"SELECT count(*) FROM {qname} WHERE {predicate}").fetchone()[0]
                    checks["null_grain"] = entry("PASS" if null_count == 0 else "FAIL", null_rows=null_count)
                    keys = ", ".join(_safe_identifier(c) for c in grain)
                    duplicate_groups = con.execute(
                        f"SELECT count(*) FROM (SELECT 1 FROM {qname} GROUP BY {keys} HAVING count(*) > 1) AS duplicate_keys"
                    ).fetchone()[0]
                    checks["duplicate_grain"] = entry("PASS" if duplicate_groups == 0 else "FAIL", duplicate_groups=duplicate_groups)
            except Exception as exc:
                if "relation" not in checks:
                    checks["relation"] = entry("NOT_EVALUATED", detail=f"unavailable: {type(exc).__name__}: {exc}")
                for key in ("grain_columns", "cycles", "null_grain", "duplicate_grain"):
                    if key not in checks:
                        checks[key] = entry("NOT_EVALUATED", detail=f"query failed: {type(exc).__name__}: {exc}")
            report["facts"][name] = result
    finally:
        if con is not None:
            con.close()
    statuses = [check["status"] for fact in report["facts"].values() for check in fact["checks"].values()]
    report["summary"] = {s: statuses.count(s) for s in ("PASS", "FAIL", "NOT_EVALUATED")}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--db", type=Path)
    p.add_argument("--scan", action="store_true", help="opt in to potentially expensive full relation scans")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    result = audit(a.root, a.db, scan=a.scan)
    payload = json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(payload, encoding="utf-8")
    print(payload)
    raise SystemExit(1 if result["summary"]["FAIL"] else 0)


if __name__ == "__main__":
    main()
