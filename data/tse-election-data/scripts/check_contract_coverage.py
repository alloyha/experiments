#!/usr/bin/env python3
"""CCC: closed-world coverage of TSE dbt resources and published ODCS surfaces.

Static/read-only. No dbt execution, database reads or GG certificate issuance.
The YAML policy classifies public surfaces individually, technical layers through
explicit default rules, and never silently classifies new Gold/Semantic models.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
LAYERS = frozenset({"bronze", "silver", "physical", "gold", "semantic"})
TECHNICAL_DEFAULTS = frozenset({"bronze", "silver", "physical", "seed", "source"})
VISIBILITIES = frozenset({"public", "internal", "excluded"})
RESOURCE_KEY = re.compile(r"^(model|seed|source|exposure):[a-zA-Z_][\w.]*$")


@dataclass(frozen=True)
class Resource:
    key: str
    kind: str
    name: str
    layer: str
    origin: str


def _load_map(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"YAML mapping required: {path}")
    return document


def discover(root: Path) -> dict[str, Resource]:
    """Enumerate SQL dbt nodes, seeds, declared source tables and exposures."""
    base = root / "tse_dbt"
    models = base / "models"
    if not models.is_dir():
        raise ValueError(f"Missing dbt model tree: {models}")
    found: dict[str, Resource] = {}

    def add(key: str, kind: str, name: str, layer: str, path: Path) -> None:
        if key in found:
            raise ValueError(f"Duplicate discovered resource {key}: {found[key].origin}, {path}")
        found[key] = Resource(key, kind, name, layer, path.relative_to(root).as_posix())

    for path in sorted(models.rglob("*.sql")):
        layer = path.relative_to(models).parts[0]
        if layer not in LAYERS:
            raise ValueError(f"Unknown dbt layer {layer}: {path}")
        add(f"model:{path.stem}", "model", path.stem, layer, path)
    for path in sorted((base / "seeds").rglob("*.csv")) if (base / "seeds").exists() else []:
        add(f"seed:{path.stem}", "seed", path.stem, "seed", path)
    for path in sorted(base.rglob("*.yml")) + sorted(base.rglob("*.yaml")):
        if any(part in {"dbt_packages", "target", "logs"} for part in path.parts):
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        for source in data.get("sources", []) or []:
            if not isinstance(source, dict) or not isinstance(source.get("name"), str):
                raise ValueError(f"Malformed dbt source: {path}")
            for table in source.get("tables", []) or []:
                if not isinstance(table, dict) or not isinstance(table.get("name"), str):
                    raise ValueError(f"Malformed dbt source table: {path}")
                name = f'{source["name"]}.{table["name"]}'
                add(f"source:{name}", "source", name, "source", path)
        for exposure in data.get("exposures", []) or []:
            if not isinstance(exposure, dict) or not isinstance(exposure.get("name"), str):
                raise ValueError(f"Malformed dbt exposure: {path}")
            name = exposure["name"]
            add(f"exposure:{name}", "exposure", name, "exposure", path)
    if not any(r.kind == "model" for r in found.values()):
        raise ValueError("No dbt SQL models discovered; coverage cannot be certified")
    return found


def manifest_members(manifest: Path, project: str = "tse_analytics") -> set[str]:
    """Optional cross-check against a freshly produced dbt manifest.json."""
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    found = set()
    for node in doc.get("nodes", {}).values():
        if node.get("package_name") != project:
            continue
        if node.get("resource_type") in {"model", "seed"}:
            found.add(f'{node["resource_type"]}:{node["name"]}')
    for source in doc.get("sources", {}).values():
        if source.get("package_name") == project:
            found.add(f'source:{source["source_name"]}.{source["name"]}')
    for exposure in doc.get("exposures", {}).values():
        if exposure.get("package_name") == project:
            found.add(f'exposure:{exposure["name"]}')
    return found


def _model_policy(root: Path) -> dict[str, dict]:
    data = _load_map(root / "tse_dbt/model_contracts.yml")
    models = data.get("models")
    if not isinstance(models, dict):
        raise ValueError("model_contracts.yml must declare models")
    return models


def audit(root: Path, *, manifest: Path | None = None, today: dt.date | None = None,
          strict_odcs: bool = True) -> dict[str, Any]:
    root = root.resolve()
    today = today or dt.date.today()
    resources = discover(root)
    policy = _load_map(root / "contracts/coverage_policy.yml")
    registry = _load_map(root / "contracts/odcs/registry.yaml")
    domain = _model_policy(root)
    errors: list[str] = []
    if policy.get("version") != 1 or set(policy) != {"version", "internal_defaults", "resources"}:
        raise ValueError("Coverage policy must contain version: 1, internal_defaults and resources")
    rules = policy["internal_defaults"]
    declared = policy["resources"]
    if not isinstance(rules, dict) or not isinstance(declared, dict):
        raise ValueError("Invalid classification rules/resources")
    if set(rules) != TECHNICAL_DEFAULTS:
        errors.append("Internal defaults must explicitly cover bronze, silver, physical, seed and source")
    for layer, reason in rules.items():
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            errors.append(f"Internal default {layer} lacks a reviewed reason")
    entries = registry.get("contracts")
    if not isinstance(entries, list):
        raise ValueError("Invalid ODCS registry")
    published: dict[str, dict] = {}
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            errors.append("Malformed ODCS registry entry")
            continue
        if item["name"] in published:
            errors.append(f'Duplicate registered contract {item["name"]}')
        published[item["name"]] = item
    if strict_odcs:
        # Reuse the independent, SHA-pinned closed-world validator.
        from scripts.check_odcs_contract_registry import check_registry
        try:
            validated = check_registry(root / "contracts/odcs/registry.yaml", root)
            if set(validated) != set(published):
                errors.append("ODCS registry validation returned a different universe")
        except (ValueError, OSError, TypeError, AttributeError, yaml.YAMLError) as exc:
            errors.append(f"Invalid ODCS registry: {exc}")
    if manifest is not None:
        try:
            from_manifest = manifest_members(manifest)
            from_sources = set(resources)
            for key in sorted(from_manifest - from_sources):
                errors.append(f"Manifest-only resource missing in source inventory: {key}")
            for key in sorted(from_sources - from_manifest):
                errors.append(f"Source-only resource missing in manifest: {key}")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"Cannot reconcile manifest: {exc}")
    stats = {"total": len(resources), "public": 0, "internal": 0,
             "excluded": 0, "unclassified": 0, "invalid_contracts": 0,
             "unmapped_semantic_obligations": 0}
    unknown: list[str] = []
    covered = set()
    for key, r in sorted(resources.items()):
        decision = declared.get(key)
        if decision is None:
            if r.layer in TECHNICAL_DEFAULTS and r.layer in rules and isinstance(rules[r.layer], str):
                stats["internal"] += 1
                continue
            stats["unclassified"] += 1
            unknown.append(key)
            errors.append(f"Unclassified {key} ({r.origin})")
            continue
        covered.add(key)
        if not isinstance(decision, dict) or decision.get("visibility") not in VISIBILITIES:
            errors.append(f"Invalid decision for {key}")
            continue
        visibility = decision["visibility"]
        stats[visibility] += 1
        if visibility == "public":
            if r.kind != "model" or r.layer not in {"gold", "semantic"}:
                errors.append(f"Public ODCS surface must be Gold/Semantic model: {key}")
            contract = decision.get("contract")
            if contract != r.name or contract not in published:
                stats["invalid_contracts"] += 1
                errors.append(f"Public {key} lacks same-name registered ODCS contract")
                continue
            grain = decision.get("grain")
            domain_grain = domain.get(r.name, {}).get("grain")
            if not isinstance(grain, list) or not grain or grain != domain_grain:
                stats["unmapped_semantic_obligations"] += 1
                errors.append(f"Public {key} grain disagrees with domain model_contracts.yml")
            expected_test = decision.get("semantic_test")
            metadata = root / published[contract]["dbt_metadata"]
            if expected_test != "dbt_utils.unique_combination_of_columns":
                stats["unmapped_semantic_obligations"] += 1
                errors.append(f"Public {key} lacks unique-composite grain evidence")
            else:
                dbt = _load_map(metadata)
                dbt_models = dbt.get("models", [])
                tests = (dbt_models[0].get("tests", []) + dbt_models[0].get("data_tests", [])) if len(dbt_models) == 1 else []
                actual_grains = [test[expected_test].get("arguments", {}).get("combination_of_columns")
                                 for test in tests if isinstance(test, dict) and expected_test in test]
                if grain not in actual_grains:
                    stats["unmapped_semantic_obligations"] += 1
                    errors.append(f"Public {key} has no dbt composite uniqueness test for its grain")
        elif visibility == "internal":
            if set(decision) != {"visibility", "reason"} or not isinstance(decision["reason"], str) or len(decision["reason"].strip()) < 20:
                errors.append(f"Internal {key} requires an explicit rationale (20+ chars)")
        elif visibility == "excluded":
            if not all(k in decision for k in ("reason", "owner", "expires")):
                errors.append(f"Excluded {key} lacks reason, owner or expiry")
            else:
                try:
                    expires = dt.date.fromisoformat(str(decision["expires"]))
                    if expires < today:
                        errors.append(f"Expired coverage exception for {key}: {expires}")
                except ValueError:
                    errors.append(f"Invalid expiry for exception {key}")
                if any(not isinstance(decision[k], str) or len(decision[k].strip()) < 3 for k in ("owner", "reason")):
                    errors.append(f"Invalid exception owner/reason for {key}")
    for key in sorted(set(declared) - covered):
        if not RESOURCE_KEY.fullmatch(key):
            errors.append(f"Invalid policy resource key: {key}")
        if key not in resources:
            errors.append(f"Stale policy entry without discovered resource: {key}")
    public_names = {r.name for key, r in resources.items()
                    if isinstance(declared.get(key), dict) and declared[key].get("visibility") == "public"}
    for missing in sorted(set(published) - public_names):
        stats["invalid_contracts"] += 1
        errors.append(f"Published ODCS contract has no public classified model: {missing}")
    # Domain policy is historically broader than the *current* dbt SQL inventory.
    # Replaced staging/intermediate models must have an explicit successor;
    # missing implemented models can never be silently reclassified or ignored.
    for model, facts in sorted(domain.items()):
        if not isinstance(facts, dict):
            errors.append(f"Invalid domain policy record: {model}")
            continue
        status = facts.get("status")
        key = f"model:{model}"
        if status == "planned":
            if key in resources:
                errors.append(f"Planned domain model is already materialized: {model}")
        elif status == "superseded":
            successor = facts.get("superseded_by")
            if (not isinstance(successor, str) or not successor
                    or successor == model):
                errors.append(f"Superseded domain model lacks a distinct successor: {model}")
            elif f"model:{successor}" not in resources:
                errors.append(f"Superseded domain model successor missing from dbt SQL: {model} -> {successor}")
            if key in resources:
                errors.append(f"Superseded domain model is still an active dbt SQL model: {model}")
            reason = facts.get("supersession_reason")
            if not isinstance(reason, str) or len(reason.strip()) < 20:
                errors.append(f"Superseded domain model lacks reviewed reason: {model}")
        elif isinstance(status, str) and status.startswith("implemented"):
            if key not in resources:
                errors.append(f"Implemented domain policy model has no dbt SQL: {model}")
        else:
            errors.append(f"Unknown domain policy status for {model}: {status!r}")
    return {"status": "PASS" if not errors else "FAIL", "stats": stats,
            "unclassified": unknown, "errors": sorted(set(errors)),
            "resources": {k: {"layer": r.layer, "origin": r.origin} for k, r in sorted(resources.items())}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "check", "report"), default="check")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, help="Optional freshly generated manifest.json")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    try:
        result = audit(args.root, manifest=args.manifest)
    except (OSError, ValueError, yaml.YAMLError, TypeError, AttributeError, KeyError) as exc:
        print(f"FAIL: contract coverage inventory: {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        counts = " ".join(f"{k}={v}" for k, v in result["stats"].items())
        print(f'CCC · {result["status"]} · {counts}')
        if args.mode == "plan":
            print("\nReview these candidate resources; do not auto-approve them:")
            for key in result["unclassified"]:
                print(f"  {key}:\n    visibility: internal  # REVIEW FIRST: may be public\n    reason: REPLACE WITH A REVIEWED 20+ CHARACTER RATIONALE")
        if result["errors"]:
            print("Coverage gaps:")
            for message in result["errors"]:
                print("  -", message)
    return int(args.mode == "check" and result["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
