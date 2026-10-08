#!/usr/bin/env python3
"""Delete extracted cache while keeping current_objects physically truthful."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/tse/raw")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.root).resolve()
    data_root = raw_root.parent if raw_root.name == "raw" else raw_root
    if raw_root.name != "raw":
        raw_root = data_root / "raw"

    candidates = []

    for extracted_dir in raw_root.rglob("extracted"):
        if not extracted_dir.is_dir():
            continue

        sha_dir = extracted_dir.parent
        source_dir = sha_dir / "source"
        prepared_dir = sha_dir / "prepared"

        has_source = source_dir.exists() and any(
            p.is_file() for p in source_dir.iterdir()
        )
        has_prepared = prepared_dir.exists() and any(
            p.is_file() for p in prepared_dir.iterdir()
        )

        if not (has_source or has_prepared):
            continue

        for p in extracted_dir.iterdir():
            if p.is_file():
                candidates.append(p)

    total = sum(p.stat().st_size for p in candidates)

    print(f"candidate files: {len(candidates)}")
    print(f"reclaimable: {human(total)}")

    for p in sorted(candidates, key=lambda x: x.stat().st_size, reverse=True):
        print(f"{human(p.stat().st_size):>10}  {p}")

    if not args.apply:
        print("\nDry-run only. Use --apply to delete.")
        return

    deleted_rel: set[str] = set()
    for p in candidates:
        deleted_rel.add(str(p.relative_to(data_root)))
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

    removed_refs = rewrite_current_objects(data_root, deleted_rel)

    print(f"\nDeleted {len(candidates)} file(s).")
    print(f"Removed {removed_refs} active-index reference(s).")
    print(f"Freed approximately {human(total)}.")


if __name__ == "__main__":
    main()
