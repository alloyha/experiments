#!/usr/bin/env python3
"""Fail-closed registry for published ODCS contracts and isolated dbt ownership.

No CLI or warehouse access: this is a filesystem, version, and metadata gate.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path, PurePosixPath

import yaml

from scripts.check_odcs_contract_version import BASELINE_SHA256_BY_CONTRACT, check_contract_version

FIELDS = frozenset({"name", "contract", "baseline", "baseline_sha256", "dbt_metadata"})
DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _file(root: Path, value: str, prefix: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid registry path: {value!r}")
    rel = PurePosixPath(value)
    if rel.is_absolute() or ".." in rel.parts or not value.startswith(prefix):
        raise ValueError(f"Registry path outside permitted prefix {prefix}: {value}")
    path = root.joinpath(*rel.parts)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root.parent):
        raise ValueError(f"Symlink forbidden in ODCS registry path: {value}")
    if not path.is_file():
        raise ValueError(f"Missing registered file: {value}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Registry path escapes project root: {value}")
    return path


def check_registry(registry: Path, root: Path) -> list[str]:
    """Validate the complete closed-world registry; return declared contract names."""
    root = root.resolve()
    data = yaml.safe_load(registry.read_bytes())
    if not isinstance(data, dict) or set(data) != {"version", "contracts"} or data["version"] != 1:
        raise ValueError("ODCS registry must have version: 1 and contracts")
    entries = data["contracts"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("ODCS registry must declare at least one contract")
    names: set[str] = set()
    candidates: set[Path] = set()
    baselines: set[Path] = set()
    owners: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != FIELDS:
            raise ValueError("ODCS entry must contain exactly name, contract, baseline, baseline_sha256 and dbt_metadata")
        name = entry["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"Invalid ODCS contract name: {name!r}")
        if name in names:
            raise ValueError(f"Duplicate ODCS contract registration: {name}")
        names.add(name)
        digest = entry["baseline_sha256"]
        if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
            raise ValueError(f"Invalid baseline_sha256 for {name}")
        if BASELINE_SHA256_BY_CONTRACT.get(name) != digest:
            raise ValueError(f"ODCS baseline digest disagrees with source-pinned trust anchor: {name}")
        current = _file(root, entry["contract"], "contracts/odcs/")
        baseline = _file(root, entry["baseline"], "contracts/odcs/baselines/")
        metadata = _file(root, entry["dbt_metadata"], "tse_dbt/models/")
        if current.name != f"{name}.odcs.yaml":
            raise ValueError(f"ODCS contract path/name mismatch: {name}")
        if baseline.name != f"{name}.v1.0.0.odcs.yaml":
            raise ValueError(f"Unexpected baseline name for {name}")
        if metadata.suffix not in {".yml", ".yaml"}:
            raise ValueError(f"dbt metadata must be YAML: {metadata}")
        if metadata in owners:
            raise ValueError(f"Shared dbt metadata ownership is forbidden: {metadata}")
        owners.add(metadata)
        candidates.add(current)
        baselines.add(baseline)
        check_contract_version(baseline, current, contract_name=name)
        document = yaml.safe_load(current.read_bytes())
        schemas = document.get("schema")
        if (document.get("name") != name or not isinstance(schemas, list) or
                len(schemas) != 1 or schemas[0].get("name") != name):
            raise ValueError(f"Contract/schema identity mismatch: {name}")
        dbt = yaml.safe_load(metadata.read_bytes())
        models = dbt.get("models") if isinstance(dbt, dict) else None
        if not isinstance(models, list) or len(models) != 1 or models[0].get("name") != name:
            raise ValueError(f"Isolated dbt model ownership mismatch: {name}")
        owner = models[0].get("config", {}).get("meta", {}).get("datacontract_cli", {}).get("contract_id")
        if owner != document.get("id"):
            raise ValueError(f"dbt model contract_id mismatch: {name}")
    if names != set(BASELINE_SHA256_BY_CONTRACT):
        raise ValueError("Registered ODCS names must match source-pinned baseline trust anchors")
    actual = set((root / "contracts/odcs").glob("*.odcs.yaml"))
    published = set((root / "contracts/odcs/baselines").glob("*.odcs.yaml"))
    if actual != candidates:
        raise ValueError(f"Unregistered ODCS contract files: {sorted(str(p.name) for p in actual ^ candidates)}")
    if published != baselines:
        raise ValueError(f"Unregistered ODCS baselines: {sorted(str(p.name) for p in published ^ baselines)}")
    return sorted(names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("contracts/odcs/registry.yaml"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        names = check_registry(args.registry, args.root)
    except (ValueError, OSError, yaml.YAMLError, TypeError, AttributeError) as exc:
        print(f"FAIL: ODCS registry: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: {len(names)} registered ODCS contract(s), baselines and isolated dbt ownership: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
