#!/usr/bin/env python3
"""Guard the frozen ODCS 1.0.0 baseline and contract version progression.

This check is independent of `datacontract breaking`, which runs after it.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import yaml

BASELINE_SHA256 = "9b0aaf1ab8b7097c4ca9834a2513ea4d97bd300c23acf1158594b2cb572fd50b"
BASELINE_SHA256_BY_CONTRACT = {
    "dim_election_cycle": BASELINE_SHA256,
    "dim_election": "5e8e2be20ff4fa620b3c59d52ccad5bf11ac14a23c9c4b5b64ce1f983ce7abe2",
    "dim_candidate": "df9cf2791085d2159eb85e6df608ef64b11e3299aa719d13f5f266ff961ba96d",
    "dim_geography": "63534f1e62f347bc028e7e8147cc66a1283f86179c648e0232052f41c31a26a3",
}
SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
IDENTITY = ("apiVersion", "kind", "id", "name")


def _version(value: object) -> tuple[int, int, int]:
    match = SEMVER.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"Expected MAJOR.MINOR.PATCH contract version; received {value!r}")
    return tuple(int(part) for part in match.groups())


def check_contract_version(baseline: Path, candidate: Path, *, contract_name: str = "dim_election_cycle") -> None:
    frozen = baseline.read_bytes()
    actual_sha = hashlib.sha256(frozen).hexdigest()
    if actual_sha != BASELINE_SHA256_BY_CONTRACT[contract_name]:
        raise ValueError("Frozen ODCS baseline checksum changed: explicit migration review required")
    old = yaml.safe_load(frozen)
    new = yaml.safe_load(candidate.read_bytes())
    if not isinstance(old, dict) or not isinstance(new, dict):
        raise ValueError("Both ODCS documents must contain YAML mappings")
    if any(old.get(field) != new.get(field) for field in IDENTITY):
        raise ValueError("ODCS public contract identity changed")
    before, after = _version(old.get("version")), _version(new.get("version"))
    if after < before:
        raise ValueError("Contract version regressed below published baseline")
    old_definition = {k: v for k, v in old.items() if k != "version"}
    new_definition = {k: v for k, v in new.items() if k != "version"}
    if old_definition != new_definition and after <= before:
        raise ValueError("Contract content changed without a version increase")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--contract-name", default="dim_election_cycle", choices=sorted(BASELINE_SHA256_BY_CONTRACT))
    args = parser.parse_args()
    try:
        check_contract_version(args.baseline, args.candidate, contract_name=args.contract_name)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"FAIL: ODCS version gate: {exc}", file=sys.stderr)
        return 1
    print("PASS: frozen ODCS baseline and semantic version progression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
