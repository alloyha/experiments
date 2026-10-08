#!/usr/bin/env python3

import argparse
from pathlib import Path


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/tse/raw")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)

    candidates = []

    for extracted_dir in root.rglob("extracted"):
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

        # Conservative rule:
        # only delete extracted files when another durable
        # representation of that exact SHA exists.
        if not (has_source or has_prepared):
            continue

        for p in extracted_dir.iterdir():
            if p.is_file():
                candidates.append(p)

    total = sum(p.stat().st_size for p in candidates)

    print(f"candidate files: {len(candidates)}")
    print(f"reclaimable: {human(total)}")

    for p in sorted(
        candidates,
        key=lambda x: x.stat().st_size,
        reverse=True,
    ):
        print(
            f"{human(p.stat().st_size):>10}  "
            f"{p}"
        )

    if not args.apply:
        print("\nDry-run only. Use --apply to delete.")
        return

    for p in candidates:
        p.unlink()

    # Remove now-empty extracted directories.
    for d in sorted(
        {p.parent for p in candidates},
        key=lambda x: len(x.parts),
        reverse=True,
    ):
        try:
            d.rmdir()
        except OSError:
            pass

    print(f"\nDeleted {len(candidates)} files.")
    print(f"Freed approximately {human(total)}.")


if __name__ == "__main__":
    main()
