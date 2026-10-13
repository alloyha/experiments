#!/usr/bin/env python3
"""Read-only TSE storage inventory and control-plane consistency report.

Never deletes, materializes, hashes or downloads payloads unless explicitly
asked to hash immutable sources with --verify-hashes (still read-only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath

SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def file_size(path: Path) -> int:
    return path.stat().st_size


def tree_size(root: Path) -> int:
    if root.is_symlink() or not root.exists():
        return 0
    if root.is_file():
        return file_size(root)
    total = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(base) / d).is_symlink()]
        total += sum(file_size(Path(base) / name) for name in files
                     if (Path(base) / name).is_file() and not (Path(base) / name).is_symlink())
    return total


def human(value: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value} B"
        value /= 1024
    return f"{value} TiB"


def read_json_object(path: Path, issues: list[str], *, required: bool = False) -> dict:
    if not path.exists():
        if required:
            issues.append(f"missing metadata: {path.name}")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        issues.append(f"unreadable metadata {path.name}: {exc}")
        return {}
    if not isinstance(data, dict):
        issues.append(f"invalid metadata {path.name}: expected JSON object")
        return {}
    return data


def safe_relative(root: Path, value: object, issues: list[str], label: str) -> Path | None:
    if not isinstance(value, str) or not value:
        issues.append(f"{label}: missing path")
        return None
    posix = PurePosixPath(value)
    if posix.is_absolute() or any(part in ("..", ".") for part in value.split("/")) or "\\" in value:
        issues.append(f"{label}: unsafe path {value!r}")
        return None
    path = root.joinpath(*posix.parts)
    if not path.resolve(strict=False).is_relative_to(root):
        issues.append(f"{label}: outside storage root {value!r}")
        return None
    return path


def physical_file(path: Path | None) -> bool:
    return path is not None and path.is_file() and not path.is_symlink()


def inventory(raw: Path) -> tuple[set[Path], set[Path], set[Path], int]:
    source: set[Path] = set()
    extracted: set[Path] = set()
    other: set[Path] = set()
    symlinks = 0
    for base, dirs, files in os.walk(raw, followlinks=False):
        parent = Path(base)
        for d in list(dirs):
            if (parent / d).is_symlink():
                dirs.remove(d)
                symlinks += 1
        for name in files:
            path = parent / name
            if path.is_symlink():
                symlinks += 1
                continue
            if not path.is_file():
                continue
            parts = path.relative_to(raw).parts
            if "source" in parts:
                source.add(path)
            elif "extracted" in parts:
                extracted.add(path)
            else:
                other.add(path)
    return source, extracted, other, symlinks


def sibling_source(extracted: Path, raw: Path, sources: set[Path]) -> Path | None:
    parts = extracted.relative_to(raw).parts
    try:
        idx = parts.index("extracted")
    except ValueError:
        return None
    version = raw.joinpath(*parts[:idx])
    matching = sorted(path for path in sources if path.parent == version / "source")
    # Source sibling is only *potential* recoverability, not a verified SHA.
    return matching[0] if len(matching) == 1 else None


def dimensions(path: Path) -> tuple[str, str]:
    domain = year = "unknown"
    for part in path.parts:
        if part.startswith("domain="):
            domain = part.split("=", 1)[1]
        if part.startswith("year="):
            year = part.split("=", 1)[1]
    return domain, year


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def produce_report(raw_root: Path, duckdb: Path, facts: Path, *, verify_hashes: bool = False) -> dict:
    root = raw_root.resolve()
    raw = root / "raw"
    if not raw.is_dir():
        raise ValueError(f"RAW directory does not exist: {raw} (no empty successful audit)")
    metadata = root / "_metadata"
    issues: list[str] = []
    warnings: list[str] = []
    state = read_json_object(metadata / "resource_state.json", issues, required=True)
    pending = read_json_object(metadata / "pending_resources.json", issues)
    source, extracted, other, symlinks = inventory(raw)
    if symlinks:
        warnings.append(f"excluded {symlinks} symlinks from payload sizes")
    active_sources: set[Path] = set()
    pending_sources: set[Path] = set()
    referenced_cache: set[Path] = set()
    missing_evicted = 0
    missing_unexpected = 0
    missing_sources = 0
    unrecognized_sources = 0
    anomaly_rows: list[tuple[str, str, dict]] = []
    expected_hashes: dict[Path, str] = {}
    for namespace, rows, found in (("active", state, active_sources), ("pending", pending, pending_sources)):
        for key, row in rows.items():
            if not isinstance(row, dict):
                issues.append(f"{namespace} {key}: expected object")
                continue
            path = safe_relative(root, row.get("source_object"), issues, f"{namespace} {key} source")
            if path is not None:
                if physical_file(path) and path in source:
                    found.add(path)
                    expected_size = row.get("source_size_bytes")
                    if expected_size is not None:
                        try:
                            if int(expected_size) != file_size(path):
                                issues.append(f"{namespace} {key}: source size mismatch")
                        except (TypeError, ValueError):
                            issues.append(f"{namespace} {key}: invalid source size")
                    expected_sha = row.get("source_sha256")
                    if isinstance(expected_sha, str) and SHA256.fullmatch(expected_sha):
                        expected_hashes[path] = expected_sha.lower()
                    else:
                        warnings.append(f"{namespace} {key}: missing/invalid source_sha256")
                else:
                    if not physical_file(path):
                        missing_sources += 1
                        issues.append(f"{namespace} {key}: immutable source missing: {row.get('source_object')}")
                    else:
                        unrecognized_sources += 1
                        issues.append(f"{namespace} {key}: source exists outside source inventory: {row.get('source_object')}")
                    anomaly_rows.append((namespace, str(key), row))
            if namespace == "active":
                for value in row.get("extracted_objects") or []:
                    obj = safe_relative(root, value, issues, f"active {key} object")
                    if obj is None:
                        continue
                    if physical_file(obj):
                        if obj in extracted:
                            referenced_cache.add(obj)
                    elif row.get("materialization_state") == "evicted":
                        missing_evicted += 1
                    else:
                        missing_unexpected += 1
                        issues.append(f"active {key}: required materialization missing: {value}")
                        anomaly_rows.append((namespace, str(key), row))

    index = metadata / "current_objects.jsonl"
    index_rows = 0
    index_objects_by_source: dict[str, set[str]] = defaultdict(set)
    if not index.is_file():
        issues.append("missing metadata: current_objects.jsonl")
    else:
        try:
            with index.open(encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        issues.append(f"current index line {line_no}: expected object")
                        continue
                    index_rows += 1
                    if isinstance(row.get("source_object"), str) and isinstance(row.get("object"), str):
                        index_objects_by_source[row["source_object"]].add(row["object"])
                    for name in ("source_object", "object"):
                        path = safe_relative(root, row.get(name), issues, f"index {line_no} {name}")
                        if path is not None and not physical_file(path):
                            issues.append(f"index line {line_no}: dangling {name}")
        except (OSError, ValueError) as exc:
            issues.append(f"unreadable current index: {exc}")

    # Diagnostics are metadata and directory listings only; never open payload bytes.
    diagnostics: list[dict] = []
    seen_anomalies: set[tuple[str, str]] = set()
    for namespace, key, row in anomaly_rows:
        if (namespace, key) in seen_anomalies:
            continue
        seen_anomalies.add((namespace, key))
        source_rel = row.get("source_object")
        source_path = safe_relative(root, source_rel, [], "diagnostic source")
        source_status = (
            "symlink" if source_path is not None and source_path.is_symlink() else
            "missing" if source_path is None or not source_path.exists() else
            "file" if source_path.is_file() else "other"
        )
        # The sha256=... directory is the physical version boundary.
        version = None
        if source_path is not None:
            for ancestor in source_path.parents:
                if ancestor.name.startswith("sha256=") and ancestor.is_relative_to(raw):
                    version = ancestor
                    break
        sibling_sources: list[dict] = []
        if version is not None and (version / "source").is_dir():
            for candidate in sorted((version / "source").iterdir())[:20]:
                if physical_file(candidate):
                    sibling_sources.append({
                        "path": candidate.relative_to(root).as_posix(),
                        "bytes": file_size(candidate),
                    })
        missing_outputs = []
        for value in row.get("extracted_objects") or []:
            obj = safe_relative(root, value, [], "diagnostic output")
            if obj is None or not physical_file(obj):
                missing_outputs.append(value)
        diagnostics.append({
            "namespace": namespace, "key": key,
            "source_object": source_rel,
            "source_status": source_status,
            "source_bytes": file_size(source_path) if physical_file(source_path) else None,
            "source_sha256_expected": row.get("source_sha256"),
            "source_size_bytes_expected": row.get("source_size_bytes"),
            "source_siblings": sibling_sources,
            "missing_extracted_objects": missing_outputs,
            "materialization_state": row.get("materialization_state"),
            "current_index_objects_for_source": sorted(index_objects_by_source.get(source_rel, set())),
        })

    verified = 0
    if verify_hashes:
        for path, expected in expected_hashes.items():
            if digest(path) != expected:
                issues.append(f"SHA-256 mismatch: {path.relative_to(root)}")
            else:
                verified += 1

    sizes = {p: file_size(p) for p in source | extracted | other}
    total = lambda group: sum(sizes[p] for p in group)
    # O(files), rather than comparing every extracted file with every source.
    version_sources: dict[Path, list[Path]] = defaultdict(list)
    for src in source:
        version_sources[src.parent.parent].append(src)
    backed: set[Path] = set()
    for path in extracted:
        parts = path.relative_to(raw).parts
        version = raw.joinpath(*parts[:parts.index("extracted")])
        if len(version_sources[version]) == 1:
            backed.add(path)
    by_domain: dict[str, int] = defaultdict(int)
    by_year: dict[str, int] = defaultdict(int)
    for path in source | extracted | other:
        domain, year = dimensions(path)
        by_domain[domain] += sizes[path]
        by_year[year] += sizes[path]
    result = {
        "schema_version": 1,
        "raw_root": str(root),
        "read_only": True,
        "source_bytes": total(source),
        "active_source_bytes": total(active_sources),
        "pending_source_bytes": total(pending_sources),
        "historical_or_orphan_source_bytes": total(source - active_sources - pending_sources),
        "extracted_bytes": total(extracted),
        "source_backed_cache_candidate_bytes": total(backed),
        "unbacked_cache_bytes": total(extracted - backed),
        "currently_referenced_cache_bytes": total(referenced_cache),
        "unreferenced_cache_bytes": total(extracted - referenced_cache),
        "other_raw_bytes": total(other),
        "metadata_bytes": tree_size(metadata),
        "warehouse_bytes": tree_size(duckdb),
        "candidate_fact_bytes": tree_size(facts),
        "active_resources": len(state),
        "pending_resources": len(pending),
        "evicted_missing_objects": missing_evicted,
        "unexpected_missing_objects": missing_unexpected,
        "unavailable_source_objects": missing_sources,
        "unrecognized_source_objects": unrecognized_sources,
        "diagnostics": diagnostics,
        "current_index_rows": index_rows,
        "source_hashes_verified": verified,
        "source_hashes_available": len(expected_hashes),
        "by_domain_bytes": dict(sorted(by_domain.items())),
        "by_year_bytes": dict(sorted(by_year.items())),
        "issues": issues,
        "warnings": warnings,
    }
    result["measured_total_bytes"] = sum(result[k] for k in (
        "source_bytes", "extracted_bytes", "other_raw_bytes",
        "metadata_bytes", "warehouse_bytes", "candidate_fact_bytes",
    ))
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--duckdb", type=Path, required=True)
    p.add_argument("--candidate-fact-root", type=Path, required=True)
    p.add_argument("--verify-hashes", action="store_true", help="Read and SHA-256 hash all active and pending source blobs; expensive")
    p.add_argument("--strict", action="store_true", help="Nonzero exit if consistency issues are detected")
    p.add_argument("--json", action="store_true", help="Machine-readable JSON on stdout")
    p.add_argument("--diagnose", action="store_true", help="Print metadata-only diagnostics for anomalous resources")
    p.add_argument("--top", type=int, default=0)
    args = p.parse_args()
    try:
        result = produce_report(args.raw_root, args.duckdb, args.candidate_fact_root, verify_hashes=args.verify_hashes)
    except (ValueError, OSError) as exc:
        print(f"storage audit: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("TSE LOCAL STORAGE — READ ONLY")
        for name, key in (
            ("Immutable sources", "source_bytes"),
            ("  active", "active_source_bytes"),
            ("  pending", "pending_source_bytes"),
            ("  historical/orphan", "historical_or_orphan_source_bytes"),
            ("Extracted cache", "extracted_bytes"),
            ("  source-backed candidate", "source_backed_cache_candidate_bytes"),
            ("  without source sibling", "unbacked_cache_bytes"),
            ("  currently referenced", "currently_referenced_cache_bytes"),
            ("  unreferenced", "unreferenced_cache_bytes"),
            ("Other raw (prepared/manifests)", "other_raw_bytes"),
            ("Control-plane metadata", "metadata_bytes"),
            ("DuckDB", "warehouse_bytes"),
            ("Physical candidate facts", "candidate_fact_bytes"),
            ("Measured logical footprint", "measured_total_bytes"),
        ):
            print(f"  {name:<33} {human(result[key]):>12}")
        print(f"  Active/pending resources             {result['active_resources']}/{result['pending_resources']}")
        print(f"  Physically indexed rows              {result['current_index_rows']}")
        print(f"  Expected cache intentionally evicted {result['evicted_missing_objects']}")
        print(f"  Unexpectedly missing cache objects   {result['unexpected_missing_objects']}")
        print(f"  Unavailable immutable sources        {result['unavailable_source_objects']}")
        print(f"  Unrecognized immutable sources       {result['unrecognized_source_objects']}")
        print(f"  SHA sources verified                 {result['source_hashes_verified']}/{result['source_hashes_available']}")
        for label, field in (("By domain", "by_domain_bytes"), ("By year", "by_year_bytes")):
            print(f"\n{label}")
            pairs = sorted(result[field].items(), key=lambda x: (-x[1], x[0]))
            for key, size in pairs[:args.top or None]:
                print(f"  {key:<30} {human(size):>12}")
        if args.diagnose and result["diagnostics"]:
            print("\nAnomalous resources (metadata only)")
            for row in result["diagnostics"]:
                print(json.dumps(row, ensure_ascii=False, sort_keys=True))
        for warning in result["warnings"]:
            print(f"[WARN] {warning}", file=sys.stderr)
        for issue in result["issues"]:
            print(f"[ISSUE] {issue}", file=sys.stderr)
        print("\nSource-backed cache is a *candidate*, not a deletion authorization.")
        if not args.verify_hashes:
            print("Source SHA-256 not checked; use --verify-hashes for a full read-only scan.")
    return 2 if args.strict and result["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
