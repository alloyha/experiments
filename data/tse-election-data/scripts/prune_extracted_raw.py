#!/usr/bin/env python3
"""Evict regenerable extracted cache without deleting active-version truth."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def resource_domain(path: Path) -> str | None:
    return next(
        (part.split("=", 1)[1] for part in path.parts if part.startswith("domain=")),
        None,
    )


def rewrite_current_objects(data_root: Path, deleted: set[str]) -> int:
    path = data_root / "_metadata" / "current_objects.jsonl"
    if not path.exists():
        return 0

    kept: list[dict] = []
    removed = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("object") in deleted:
                removed += 1
            else:
                kept.append(row)

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return removed


def mark_resource_state_evicted(data_root: Path, deleted: set[str]) -> int:
    """Keep the active version, but record that its derived cache is absent."""
    path = data_root / "_metadata" / "resource_state.json"
    if not path.exists():
        return 0

    state: dict[str, dict] = json.loads(
        path.read_text(encoding="utf-8")
    )
    changed = 0
    evicted_at = datetime.now(timezone.utc).isoformat()

    for row in state.values():
        objects = set(row.get("extracted_objects") or [])
        if not objects.intersection(deleted):
            continue

        row["materialization_state"] = "evicted"
        row["materialization_evicted_at"] = evicted_at
        changed += 1

    if changed:
        atomic_json(path, state)
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/tse/raw")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--domain",
        action="append",
        dest="domains",
        help="Restrict eviction to a domain; repeat for multiple domains.",
    )
    args = parser.parse_args()

    # Destructive operations must never silently expand to every domain.
    if args.apply and not args.domains:
        parser.error("--apply requires at least one explicit --domain")

    raw_root = Path(args.root).resolve()
    data_root = raw_root.parent if raw_root.name == "raw" else raw_root
    if raw_root.name != "raw":
        raw_root = data_root / "raw"

    # Invalid roots must fail, not look like an empty, successful dry-run.
    if not raw_root.is_dir():
        parser.error(f"raw directory does not exist: {raw_root}")

    if args.apply:
        # Never delete bytes without both authoritative control-plane files.
        for name in ("resource_state.json", "current_objects.jsonl"):
            metadata = data_root / "_metadata" / name
            if not metadata.is_file():
                parser.error(f"refusing eviction without {metadata}")

    candidates = []

    for extracted_dir in raw_root.rglob("extracted"):
        if not extracted_dir.is_dir():
            continue

        sha_dir = extracted_dir.parent
        source_dir = sha_dir / "source"

        # A prepared derivative is not an immutable recovery source.
        # PR34 must be able to regenerate every evicted file locally.
        has_source = source_dir.is_dir() and any(
            p.is_file() for p in source_dir.iterdir()
        )
        if not has_source:
            continue

        for p in extracted_dir.iterdir():
            if p.is_file():
                if args.domains and resource_domain(p) not in set(args.domains):
                    continue
                candidates.append(p)

    total = sum(p.stat().st_size for p in candidates)

    print(f"candidate files: {len(candidates)}")
    print(f"reclaimable: {human(total)}")

    for p in sorted(candidates, key=lambda x: x.stat().st_size, reverse=True):
        print(f"{human(p.stat().st_size):>10}  {p}")

    if not args.apply:
        print("\nDry-run only. Use --apply to delete.")
        return

    deleted_rel = {
        str(p.relative_to(data_root))
        for p in candidates
    }

    # Publish the non-reference state before removing bytes. Any crash between
    # these steps may leave harmless unindexed cache behind, but can never leave
    # current_objects pointing at a file that was deliberately evicted.
    removed_refs = rewrite_current_objects(data_root, deleted_rel)
    marked_rows = mark_resource_state_evicted(data_root, deleted_rel)

    for p in candidates:
        p.unlink()

    for d in sorted(
        {p.parent for p in candidates},
        key=lambda x: len(x.parts),
        reverse=True,
    ):
        try:
            d.rmdir()
        except OSError:
            pass

    print(f"\nDeleted {len(candidates)} file(s).")
    print(f"Removed {removed_refs} active-index reference(s).")
    print(f"Marked {marked_rows} active version(s) as evicted.")
    print(f"Freed approximately {human(total)}.")


if __name__ == "__main__":
    main()
