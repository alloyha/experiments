#!/usr/bin/env python3
"""One registry-driven ODCS gate; all sync operations stay in a disposable tree."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import yaml

from scripts.check_odcs_contract_registry import check_registry
from scripts.check_odcs_dbt_sync import compare_model_trees

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path("contracts/odcs/registry.yaml")
MODES = ("lint", "compat", "plan", "dbt-check", "all")
CLI_SYNC = "datacontract-cli==1.2.3"   # Preserve deterministic dbt generation.
CLI_COMPAT = "datacontract-cli==1.2.7" # Preserve consumer-oriented classification.


def cli(root: Path, pin: str, *arguments: str) -> list[str]:
    return ["uv", "tool", "run", "--python", str(root / ".venv/bin/python"),
            "--from", pin, "datacontract", *map(str, arguments)]


def run_odcs(
    mode: str = "all", *, root: Path = ROOT,
    invoke: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    if mode not in MODES:
        raise ValueError(f"Unknown ODCS gate: {mode}")
    root = root.resolve()
    registry = root / REGISTRY
    names = check_registry(registry, root)  # Includes SHA/version/ownership checks.
    entries = yaml.safe_load(registry.read_bytes())["contracts"]
    if sorted(x["name"] for x in entries) != names:
        raise ValueError("ODCS registry changed after validation")

    def execute(pin: str, *args: str) -> None:
        result = invoke(cli(root, pin, *args), cwd=root, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ODCS {args[0]} rejected: exit={result.returncode}")

    for entry in entries:
        path = str(root / entry["contract"])
        execute(CLI_SYNC, "lint", path)
        if mode in ("compat", "all"):
            baseline = str(root / entry["baseline"])
            execute(CLI_COMPAT, "lint", baseline)
            execute(CLI_COMPAT, "breaking", "--perspective", "consumer", baseline, path)
        if mode == "plan":
            execute(CLI_SYNC, "dbt", "sync", path,
                    "--project-dir", str(root / "tse_dbt"), "--dry-run", "--no-prune")

    if mode in ("dbt-check", "all"):
        with tempfile.TemporaryDirectory(prefix="tse-odcs-check-") as scratch:
            sandbox = Path(scratch) / "tse_dbt"
            sandbox.mkdir()
            shutil.copy2(root / "tse_dbt/dbt_project.yml", sandbox / "dbt_project.yml")
            shutil.copytree(root / "tse_dbt/models", sandbox / "models")
            for entry in entries:
                execute(CLI_SYNC, "dbt", "sync", str(root / entry["contract"]),
                        "--project-dir", str(sandbox), "--no-prune")
            managed = {Path(e["dbt_metadata"]).relative_to("tse_dbt/models")
                       for e in entries}
            drift = compare_model_trees(root / "tse_dbt/models", sandbox / "models",
                                        managed_models=managed)
            if drift:
                raise ValueError("ODCS/dbt drift:\n" + "\n".join(drift))
    print(f"PASS: ODCS {mode}: {len(entries)} registered contract(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="all")
    args = parser.parse_args(argv)
    try:
        run_odcs(mode=args.mode)
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        print(f"FAIL: ODCS {args.mode}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
