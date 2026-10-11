#!/usr/bin/env python3
"""PR34: certify one small real TSE archive in a TEMPORARY isolated copy.

Never prunes canonical data. Requires the project-local tse_ingest and PR33/34
scripts. Launch from the project root after copying this file to scripts/:

  .venv/bin/python scripts/certify_pr34_real_resource.py \
    --canonical-root "$HOME/github/experiments/data/tse-election-data/data/tse"

No production data is modified. Network attempts during the isolated round-trip
raise an error. The test compares exact extracted-byte hashes and physical index.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import socket
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tse_ingest
from scripts import ensure_materialized, prune_extracted_raw


def within(root: Path, relative: str) -> Path:
    """Allow only existing, non-symlinked relative paths strictly under root."""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not relative:
        raise ValueError(f"unsafe relative object path: {relative!r}")
    candidate = root / rel
    resolved_root = root.resolve()
    if not candidate.resolve().is_relative_to(resolved_root):
        raise ValueError(f"object escapes canonical root: {relative!r}")
    if any(part.is_symlink() for part in (candidate, *candidate.parents) if part != root and part.is_relative_to(root)):
        raise ValueError(f"symlinked object path: {relative!r}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def checksum(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    size = 0
    linebreaks = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
            linebreaks += chunk.count(b"\n")
    return digest.hexdigest(), size, linebreaks


def choose_resource(root: Path, *, max_mib: int, year: int, domain: str):
    state = tse_ingest.load_state(root)
    options = []
    for key, row in state.items():
        if int(row.get("year", 0)) != year or row.get("domain") != domain:
            continue
        if row.get("election_type") not in ("general", "municipal"):
            continue
        if not row.get("selection_complete") or not row.get("selected_members"):
            continue
        objects = row.get("extracted_objects") or []
        if len(objects) != len(row["selected_members"]):
            continue
        if not tse_ingest.materialization_complete(root, row):
            continue
        try:
            source_rel = row["source_object"]
            src = within(root, source_rel)
            if src.suffix.lower() != ".zip":
                continue
            sources = [within(root, item) for item in objects]
        except (KeyError, FileNotFoundError, ValueError):
            continue
        # Sum bytes copied to the sandbox, not just the compressed archive.
        total = src.stat().st_size + sum(obj.stat().st_size for obj in sources)
        if total > max_mib * 1024 * 1024:
            continue
        options.append((total, key, row, source_rel, objects))
    if not options:
        raise RuntimeError(
            f"No complete {domain}/{year} ZIP fits within {max_mib} MiB "
            "including extracted files. Try --max-mib 512 or a different --domain."
        )
    options.sort(key=lambda x: (x[0], x[1]))
    return options[0]


def invoke(entrypoint, argv: list[str]):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = entrypoint()
    if stderr.getvalue():
        print(stderr.getvalue().strip(), file=sys.stderr)
    return result, stdout.getvalue()


def index_objects(root: Path) -> list[str]:
    index = tse_ingest.current_objects_path(root)
    return sorted(
        json.loads(line)["object"]
        for line in index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def run(root: Path, *, max_mib: int, year: int, domain: str, extractor: str) -> None:
    canonical = root.resolve(strict=True)
    state_file = tse_ingest.state_path(canonical)
    if not state_file.is_file():
        raise RuntimeError(f"Authoritative state missing: {state_file}")
    total, key, row, source_rel, extracted_rels = choose_resource(
        canonical, max_mib=max_mib, year=year, domain=domain
    )
    if len(set(extracted_rels)) != len(extracted_rels):
        raise RuntimeError("Duplicate active physical object path")
    source_original = within(canonical, source_rel)
    source_baseline = checksum(source_original)
    if not row.get("source_sha256") or source_baseline[0] != row["source_sha256"]:
        raise RuntimeError("Canonical immutable ZIP does not match recorded SHA-256; aborting")
    baseline = {rel: checksum(within(canonical, rel)) for rel in extracted_rels}

    print(f"Source resource: {row['resource_id']} ({domain}, {year})")
    print(f"Selected files: {len(extracted_rels)}")
    print(f"Sandbox source + extracted: {total / (1024 ** 2):.2f} MiB")
    print(f"Immutable source SHA-256: {source_baseline[0]}")

    with tempfile.TemporaryDirectory(prefix="tse-pr34-real-") as temporary:
        sandbox = Path(temporary).resolve() / "tse"
        if sandbox == canonical or sandbox.is_relative_to(canonical):
            raise RuntimeError("Sandbox unexpectedly aliases canonical data")
        copied = [source_rel, *extracted_rels]
        for rel in copied:
            destination = sandbox / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(within(canonical, rel), destination)
        # Create isolated metadata from exactly one selected active resource.
        tse_ingest.save_state(sandbox, {key: dict(row)})
        if index_objects(sandbox) != sorted(extracted_rels):
            raise AssertionError("Incorrect initial physical index in sandbox")
        for rel, before in baseline.items():
            if checksum(sandbox / rel) != before:
                raise AssertionError(f"Initial copy changed bytes: {rel}")

        def network_forbidden(*_args, **_kwargs):
            raise AssertionError("Unexpected network access during offline PR34 certification")

        with mock.patch.object(socket.socket, "connect", side_effect=network_forbidden), \
             mock.patch.object(socket.socket, "connect_ex", side_effect=network_forbidden), \
             mock.patch("requests.sessions.Session.request", side_effect=network_forbidden):
            _, eviction = invoke(prune_extracted_raw.main, [
                "prune_extracted_raw.py", "--root", str(sandbox / "raw"),
                "--domain", domain, "--apply",
            ])
            if f"Deleted {len(extracted_rels)} file(s)." not in eviction:
                raise AssertionError(f"Unexpected eviction outcome:\n{eviction}")
            if index_objects(sandbox):
                raise AssertionError("Evicted files remain advertised in physical index")
            if any((sandbox / rel).exists() for rel in extracted_rels):
                raise AssertionError("Some extracted files survived sandbox eviction")
            if checksum(sandbox / source_rel) != source_baseline:
                raise AssertionError("Sandbox source changed during eviction")

            code, recovered = invoke(ensure_materialized.main, [
                "ensure_materialized.py", "--root", str(sandbox),
                "--year", str(year), "--election-type", row["election_type"],
                "--domain", domain, "--extractor", extractor,
            ])
            if code != 0 or "rehydrated=1" not in recovered:
                raise AssertionError(f"Real-resource recovery failed:\n{recovered}")
            if index_objects(sandbox) != sorted(extracted_rels):
                raise AssertionError("Recovered physical index differs from baseline")
            for rel, before in baseline.items():
                if checksum(sandbox / rel) != before:
                    raise AssertionError(f"Byte/line-count mismatch after rehydration: {rel}")
            if checksum(sandbox / source_rel) != source_baseline:
                raise AssertionError("Immutable source changed during recovery")

            with mock.patch.object(tse_ingest, "extract_selected_members", side_effect=AssertionError("unexpected re-extraction")):
                code, repeated = invoke(ensure_materialized.main, [
                    "ensure_materialized.py", "--root", str(sandbox),
                    "--year", str(year), "--election-type", row["election_type"],
                    "--domain", domain, "--extractor", extractor,
                ])
            if code != 0 or "already-present=1" not in repeated:
                raise AssertionError(f"Idempotency failed:\n{repeated}")
            if index_objects(sandbox) != sorted(extracted_rels):
                raise AssertionError("Idempotent re-run changed index")

        print("PASS: real-resource sandbox eviction, offline rehydration, byte identity, physical index, idempotency")
        for rel, (sha, size, lines) in baseline.items():
            print(f"  {rel}: sha256={sha}; bytes={size}; newline_count={lines}")
    print("PASS: canonical files unchanged; isolated sandbox removed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--domain", choices=["vote_result", "electorate"], default="vote_result")
    parser.add_argument("--max-mib", type=int, default=256)
    parser.add_argument("--extractor", choices=["python", "native", "auto"], default="python")
    args = parser.parse_args()
    if args.max_mib < 1:
        parser.error("--max-mib must be >= 1")
    run(args.canonical_root, max_mib=args.max_mib, year=args.year,
        domain=args.domain, extractor=args.extractor)


if __name__ == "__main__":
    main()
