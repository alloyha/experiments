#!/usr/bin/env python3
"""Promote a legacy candidate Parquet without rewriting its bytes.

POSIX only. All writers must cooperate with .publish.lock.
This is physical migration evidence, NOT a GG certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid

import duckdb
from pathlib import Path

from scripts.write_candidate_fact_partition import (
    current_generation,
    fsync_directory,
    fsync_file,
    partition_write_lock,
    prove_staged_zone_uniqueness,
    ZONE_UNIQUENESS_SQL,
    persist_immutable_zone_proof,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def migrate(
    root: Path,
    year: int,
    election_type: str,
    *,
    expected_sha256: str,
    lock_timeout: float = 30,
    fault: str | None = None,
) -> dict:
    if year <= 0 or election_type not in {"general", "municipal"}:
        raise ValueError("Invalid election cycle")
    if len(expected_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in expected_sha256
    ):
        raise ValueError("Expected SHA-256 must be lowercase hexadecimal")

    partition = root / f"election_type={election_type}" / f"year={year}"
    public = partition / "data.parquet"
    generation = f"generation-legacy-{expected_sha256}.parquet"
    immutable = partition / generation
    evidence = partition / f"generation-legacy-{expected_sha256}.migration.json"

    def fail(point: str) -> None:
        if fault == point:
            raise RuntimeError(f"Injected failure: {point}")

    with partition_write_lock(partition, timeout=lock_timeout):
        active = current_generation(partition)
        if active not in ("legacy", generation):
            raise RuntimeError(f"Unexpected active generation: {active}")

        source = public if active == "legacy" else immutable
        if sha256_file(source) != expected_sha256:
            raise RuntimeError("Published bytes differ from expected SHA-256")

        if immutable.exists():
            if immutable.is_symlink() or not immutable.is_file():
                raise RuntimeError("Invalid immutable generation")
            if sha256_file(immutable) != expected_sha256:
                raise RuntimeError("Existing generation checksum mismatch")
        elif active == "legacy":
            os.link(public, immutable)
            fsync_directory(partition)
        else:
            raise RuntimeError("Active generation missing")



        # Reavaliar a unicidade nos bytes efetivos da geração.
        # Uma prova persistida nunca substitui a validação do conteúdo.
        zone_proof_path = immutable.with_suffix(".zone-uniqueness.json")

        with duckdb.connect() as con:
            count = con.execute(
                "SELECT count(*) FROM read_parquet(?, hive_partitioning=false)",
                [str(immutable)],
            ).fetchone()[0]

            inventory = prove_staged_zone_uniqueness(
                con, immutable, count
            )

        expected_proof = {
            "schema_version": 1,
            "scope": "candidate_physical_zone_uniqueness_not_graph_freshness",
            "result": "PASS",
            "cycle": f"{year}:{election_type}",
            "generation": immutable.name,
            "parquet_sha256": expected_sha256,
            "guard_sql_sha256": hashlib.sha256(
                ZONE_UNIQUENESS_SQL.encode()
            ).hexdigest(),
            **inventory,
        }

        if zone_proof_path.exists() or zone_proof_path.is_symlink():
            if zone_proof_path.is_symlink() or not zone_proof_path.is_file():
                raise RuntimeError("Invalid zone proof file")

            saved = json.loads(
                zone_proof_path.read_text(encoding="utf-8")
            )

            if (
                set(saved) != set(expected_proof) | {"checked_at_utc"}
                or any(
                    saved.get(key) != value
                    for key, value in expected_proof.items()
                )
                or not isinstance(saved.get("checked_at_utc"), str)
                or not saved["checked_at_utc"]
            ):
                raise RuntimeError(
                    "Persisted zone proof differs from verified Parquet"
                )
        else:
            persist_immutable_zone_proof(
                partition,
                immutable,
                year=year,
                election_type=election_type,
                inventory=inventory,
            )

        fail("after_zone_proof")

        fail("after_generation")

        if active == "legacy":
            if os.stat(public).st_ino != os.stat(immutable).st_ino:
                raise RuntimeError("Generation does not share source inode")

        record = {
            "schema_version": 1,
            "scope": "legacy_candidate_parquet_byte_preservation_not_GG",
            "cycle": f"{year}:{election_type}",
            "generation": generation,
            "parquet_sha256": expected_sha256,
            "bytes": immutable.stat().st_size,
            "result": "PASS",
        }

        if evidence.exists() or evidence.is_symlink():
            if evidence.is_symlink() or not evidence.is_file():
                raise RuntimeError("Invalid migration evidence file")
            try:
                saved = json.loads(evidence.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError) as exc:
                raise RuntimeError("Unreadable migration evidence") from exc
            if saved != record:
                raise RuntimeError("Existing migration evidence mismatch")
        else:
            temp = partition / f".migration-{uuid.uuid4().hex}.tmp"
            try:
                with temp.open("x", encoding="utf-8") as stream:
                    json.dump(record, stream, sort_keys=True, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temp, evidence)
                fsync_directory(partition)
            finally:
                temp.unlink(missing_ok=True)

        fail("after_evidence")

        if active == "legacy":
            temp_link = partition / f".data-parquet-{uuid.uuid4().hex}.link"
            try:
                temp_link.symlink_to(generation)
                fsync_directory(partition)
                fail("before_swap")
                os.replace(temp_link, public)
                fsync_directory(partition)
            finally:
                temp_link.unlink(missing_ok=True)

        fail("after_swap")

        if current_generation(partition) != generation:
            raise RuntimeError("Published generation mismatch")
        if sha256_file(public) != expected_sha256:
            raise RuntimeError("Post-migration checksum mismatch")

        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--election-type", choices=["general", "municipal"], required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--lock-timeout", type=float, default=30)
    args = parser.parse_args()

    record = migrate(
        args.root, args.year, args.election_type,
        expected_sha256=args.expected_sha256,
        lock_timeout=args.lock_timeout,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
