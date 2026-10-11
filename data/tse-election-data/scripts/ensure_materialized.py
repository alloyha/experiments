#!/usr/bin/env python3
"""Ensure selected active raw versions have local physical materializations.

This command never touches the network. Missing extracted cache is rebuilt only
from the immutable source referenced by resource_state.json, after which
current_objects.jsonl is rebuilt as the physical index consumed by DuckDB/dbt.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

import tse_ingest


def matches_scope(
    row: dict,
    *,
    years: set[int],
    election_types: set[str],
    domains: set[str],
) -> bool:
    if years and int(row["year"]) not in years:
        return False
    if election_types and row.get("election_type") not in election_types:
        return False
    if domains and row.get("domain") not in domains:
        return False
    return True


def needs_tabular_materialization(row: dict) -> bool:
    return bool(
        (row.get("selected_members") or [])
        or (row.get("extracted_objects") or [])
    )


def materialize_state_row(
    root: Path,
    row: dict,
    *,
    extractor: str,
) -> tuple[dict, str]:
    """Return an updated state row and the action taken."""
    if not tse_ingest.active_version_complete(root, row):
        raise RuntimeError(
            "active version is incomplete or its immutable source is missing: "
            f"{row.get('resource_id')}"
        )

    if not needs_tabular_materialization(row):
        updated = {
            **row,
            "materialization_state": "not_required",
        }
        return updated, "not-required"

    if tse_ingest.materialization_complete(root, row):
        updated = {
            **row,
            "materialization_state": "present",
        }
        action = (
            "already-present"
            if row.get("materialization_state") == "present"
            else "reindexed"
        )
        return updated, action

    source_rel = row.get("source_object") or ""
    source = root / source_rel
    if not source.is_file():
        raise RuntimeError(
            f"immutable source missing for {row.get('resource_id')}: "
            f"{source_rel}"
        )

    # Verify bytes against durable active-version truth before opening the ZIP.
    expected_sha = row.get("source_sha256")
    if not expected_sha:
        raise RuntimeError("source_sha256 is required for offline rehydration")
    actual_sha, actual_size, _ = tse_ingest.hash_file(source)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"immutable source SHA-256 mismatch for {row.get('resource_id')}: "
            f"expected={expected_sha}, actual={actual_sha}"
        )
    expected_size = row.get("source_size_bytes")
    if expected_size is not None and int(expected_size) != actual_size:
        raise RuntimeError(
            f"immutable source size mismatch for {row.get('resource_id')}"
        )

    version_dir = source.parent.parent
    selected_members: list[str]
    expected_members = row.get("selected_members")
    expected_outputs = row.get("extracted_objects") or []

    if tse_ingest.zipfile.is_zipfile(source):
        # Stage *all* members outside extracted/: no partial result becomes
        # visible at the authoritative destination before full validation.
        staging = Path(tempfile.mkdtemp(prefix=".rehydrate-", dir=version_dir))
        try:
            staged, backend, selected_members = tse_ingest.extract_selected_members(
                source, staging, row.get("granularity", "brasil"),
                row.get("uf") or None, extractor,
            )
            if not staged:
                raise RuntimeError("no tabular members selected for rehydration")
            proposed = [
                str((version_dir / "extracted" / path.name).relative_to(root))
                for path in staged
            ]
            if expected_members is not None and list(expected_members) != selected_members:
                raise RuntimeError("selected archive members differ from active version")
            if expected_outputs and list(expected_outputs) != proposed:
                raise RuntimeError("extracted paths differ from active version")
            if not all(path.is_file() for path in staged):
                raise RuntimeError("incomplete staged extraction")

            destination = version_dir / "extracted"
            backup = version_dir / f".extracted-backup-{uuid.uuid4().hex}"
            had_destination = destination.exists()
            if had_destination:
                destination.rename(backup)
            try:
                staging.rename(destination)
            except BaseException:
                if had_destination and backup.exists() and not destination.exists():
                    backup.rename(destination)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            outputs = [destination / path.name for path in staged]
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    elif source.suffix.lower() in {".csv", ".txt"}:
        outputs = [source]
        backend = "direct"
        selected_members = [source.name]
        if expected_members is not None and list(expected_members) != selected_members:
            raise RuntimeError("selected source members differ from active version")
        if expected_outputs and list(expected_outputs) != [str(source.relative_to(root))]:
            raise RuntimeError("direct source path differs from active version")
    else:
        raise RuntimeError(
            f"unsupported immutable source for materialization: {source}"
        )

    updated = {
        **row,
        "extracted_objects": [
            str(path.relative_to(root))
            for path in outputs
        ],
        "selected_members": selected_members,
        "extractor": backend,
        "materialization_state": (
            "present" if outputs else "not_required"
        ),
        "materialized_at": tse_ingest.now_iso(),
    }
    return updated, "rehydrated"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/tse"),
    )
    parser.add_argument("--year", type=int, action="append")
    parser.add_argument(
        "--election-type",
        choices=["general", "municipal"],
        action="append",
        dest="election_types",
    )
    parser.add_argument(
        "--domain",
        action="append",
        dest="domains",
    )
    parser.add_argument(
        "--extractor",
        choices=["auto", "native", "python"],
        default="auto",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be >= 1")

    root = args.root.resolve()
    if not tse_ingest.state_path(root).is_file():
        parser.error(f"authoritative resource_state.json missing under {root}")
    state = tse_ingest.load_state(root)
    years = set(args.year or [])
    election_types = set(args.election_types or [])
    domains = set(args.domains or [])

    counts = {
        "selected": 0,
        "already-present": 0,
        "reindexed": 0,
        "rehydrated": 0,
        "not-required": 0,
        "errors": 0,
    }
    dirty = 0

    for key in sorted(state):
        row = state[key]
        if not matches_scope(
            row,
            years=years,
            election_types=election_types,
            domains=domains,
        ):
            continue

        counts["selected"] += 1
        try:
            updated, action = materialize_state_row(
                root,
                row,
                extractor=args.extractor,
            )
            state[key] = updated
            counts[action] += 1
            if updated != row:
                dirty += 1
                if dirty % args.checkpoint_every == 0:
                    tse_ingest.save_state(root, state)
        except Exception as exc:
            counts["errors"] += 1
            print(
                f"[error] {row.get('year')} :: "
                f"{row.get('domain')} :: "
                f"{row.get('resource_id')}: {exc}",
                file=sys.stderr,
            )

    # Rebuild current_objects even when no row changed. This repairs the physical
    # index after an interrupted eviction that removed index rows before deleting
    # the corresponding bytes.
    tse_ingest.save_state(root, state)

    print(
        "Materialization: "
        + ", ".join(
            f"{key}={value}"
            for key, value in counts.items()
        )
    )
    return 1 if counts["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
