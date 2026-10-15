#!/usr/bin/env python3
"""Validate the declared invariant coverage without running expensive SQL.

Normal mode rejects incomplete records for *implemented* invariants. --strict
additionally fails if any critical invariant remains planned. This distinction
keeps CI fast while preventing a false declaration of full domain coverage.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "contracts/invariants/gold_graph.yml"
STATUS = {"implemented", "planned"}
SEVERITY = {"critical", "high", "medium", "low"}


def resolve_test(reference: str, root: Path) -> bool:
    parts = reference.split("::")
    if len(parts) != 2 or "." not in parts[1]:
        return False
    filename, test_id = parts
    klass, method = test_id.split(".", 1)
    path = root / filename
    if not path.is_file() or not method.startswith("test_"):
        return False
    try:
        module = ast.parse(path.read_text(encoding="utf8"))
    except (OSError, SyntaxError):
        return False
    return any(
        isinstance(item, ast.ClassDef) and item.name == klass
        and any(isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == method for child in item.body)
        for item in module.body
    )


def check_registry(path: Path = DEFAULT, *, strict: bool = False, root: Path = ROOT) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf8"))
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError("Unsupported invariant registry format")
    records = document.get("invariants")
    if not isinstance(records, list) or not records:
        raise ValueError("Empty or invalid invariant registry")
    seen: set[str] = set()
    failures = []
    implemented = planned = 0
    referenced_sql = set()
    for record in records:
        if not isinstance(record, dict):
            failures.append("Invalid registry item")
            continue
        key = record.get("id")
        if not isinstance(key, str) or key in seen or not key:
            failures.append(f"Duplicate/invalid invariant ID: {key!r}")
            continue
        seen.add(key)
        if record.get("status") not in STATUS or record.get("severity") not in SEVERITY:
            failures.append(f"{key}: invalid status or severity")
            continue
        if not record.get("family") or not record.get("statement"):
            failures.append(f"{key}: missing family or statement")
        if record["status"] == "planned":
            planned += 1
            if not record.get("gap"):
                failures.append(f"{key}: planned invariant lacks a declared gap")
            if strict and record["severity"] == "critical":
                failures.append(f"{key}: critical invariant still planned")
            continue
        implemented += 1
        for field in ("positive_test", "falsifier_test"):
            if not resolve_test(str(record.get(field, "")), root):
                failures.append(f"{key}: invalid {field}: {record.get(field)}")
        assertion = record.get("sql_assertion")
        if not assertion or not (root / assertion).is_file():
            failures.append(f"{key}: missing SQL assertion {assertion!r}")
        else:
            referenced_sql.add(assertion)
    graph_sql = {
        path.relative_to(root).as_posix()
        for path in (root / "tse_dbt/tests").glob("assert_graph_*.sql")
    }
    for path in sorted(graph_sql - referenced_sql):
        failures.append(f"Unregistered graph SQL assertion: {path}")
    if failures:
        raise ValueError("Invariant registry FAIL:\n  " + "\n  ".join(failures))
    return {"implemented": implemented, "planned": planned,
            "critical_coverage_complete": not any(
                r["status"] == "planned" and r["severity"] == "critical"
                for r in records
            )}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--registry", type=Path, default=DEFAULT)
    args = parser.parse_args()
    try:
        result = check_registry(args.registry, strict=args.strict)
    except (ValueError, OSError, yaml.YAMLError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("Invariant registry: PASS", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
