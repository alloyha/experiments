"""Fail-closed freshness attestations for the persisted candidate performance cube.

The digest combines the published source Parquet, full scoped dimension rows,
and a non-cryptographic DuckDB cube fingerprint. Certificates belong to an
explicit database, publication root and cycle and are published only after dbt
conservation and ranking contracts succeed. Old certificates are revoked before
any re-materialization attempt so an interrupted refresh never remains trusted.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from scripts.warehouse_lifecycle import lifecycle_epoch

MODEL = "cube_candidate_performance_municipality"
DIMENSIONS = ("dim_candidate", "dim_geography")
FORMAT_VERSION = 1
CYCLE_PATTERN = re.compile(r"^(\d{4}):(general|municipal)$")


class FreshnessError(RuntimeError):
    pass


def cycle_parts(cycle: str) -> tuple[int, str]:
    match = CYCLE_PATTERN.fullmatch(cycle)
    if not match:
        raise FreshnessError(f"Invalid election cycle: {cycle!r}")
    return int(match.group(1)), match.group(2)


def certificate_path(directory: Path, cycle: str) -> Path:
    year, kind = cycle_parts(cycle)
    return Path(directory) / f"{year}-{kind}.json"


def publication_signature(root: Path, cycle: str) -> dict:
    """Hash pinned bytes and verify the public symlink has not switched mid-read."""
    year, kind = cycle_parts(cycle)
    partition = Path(root) / f"election_type={kind}" / f"year={year}"
    pointer = partition / "data.parquet"

    def token() -> str:
        if pointer.is_symlink():
            link = os.readlink(pointer)
            if Path(link).name != link or link in {".", ".."}:
                raise FreshnessError(f"Unsafe published generation pointer: {pointer}")
            if not (partition / link).is_file():
                raise FreshnessError(f"Broken published generation: {pointer}")
            return link
        if pointer.is_file():
            return "legacy-regular-file"
        raise FreshnessError(f"Published partition not found: {pointer}")

    before = token()
    digest = hashlib.sha256()
    count = 0
    with pointer.open("rb") as file:
        while data := file.read(2 * 1024 * 1024):
            digest.update(data)
            count += len(data)
    if not count:
        raise FreshnessError(f"Published Parquet is empty: {pointer}")
    if token() != before:
        raise FreshnessError(f"Parquet generation changed during fingerprint: {cycle}")
    return {"generation": before, "sha256": digest.hexdigest(), "bytes": count}


def _encode_row(row: tuple) -> bytes:
    """Canonical enough for stable native DuckDB scalar data across snapshots."""
    return json.dumps(row, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"), default=str).encode("utf-8")


def dimension_signature(con, table: str, cycle: str) -> dict:
    if table not in DIMENSIONS:
        raise FreshnessError(f"Unsupported dimension: {table}")
    kind = con.execute("""
        select table_type from information_schema.tables
        where table_schema='main' and table_name=?
    """, [table]).fetchone()
    if kind != ("BASE TABLE",):
        raise FreshnessError(f"Missing persisted dimension {table}: {kind}")
    cursor = con.execute(f"SELECT * FROM main.{table} WHERE cycle_id = ? ORDER BY ALL", [cycle])
    schema = [(column[0], str(column[1])) for column in cursor.description]
    hasher = hashlib.sha256()
    header = _encode_row(tuple(schema))
    hasher.update(len(header).to_bytes(8, "big") + header)
    count = 0
    while batch := cursor.fetchmany(4096):
        for row in batch:
            data = _encode_row(row)
            hasher.update(len(data).to_bytes(8, "big") + data)
            count += 1
    return {"rows": count, "sha256": hasher.hexdigest()}


def cube_signature(con, cycle: str) -> dict:
    kind = con.execute("""
        select table_type from information_schema.tables
        where table_schema='main' and table_name=?
    """, [MODEL]).fetchone()
    if kind != ("BASE TABLE",):
        raise FreshnessError(f"Candidate cube must be a persisted BASE TABLE: {kind}")
    columns = [row[1] for row in con.execute(f"PRAGMA table_info('{MODEL}')").fetchall()]
    if "cycle_id" not in columns or "nominal_valid_votes" not in columns:
        raise FreshnessError("Candidate cube lacks mandatory columns")
    quoted = ", ".join('"' + column.replace('"', '""') + '"' for column in columns)
    query = f"""
        with row_hashes as (
            select nominal_valid_votes, hash({quoted}) as row_hash
            from main.{MODEL} where cycle_id = ?
        )
        select count(*), sum(nominal_valid_votes), bit_xor(row_hash),
               sum(cast(row_hash as hugeint)) from row_hashes
    """
    rows, votes, xor_digest, sum_digest = con.execute(query, [cycle]).fetchone()
    if rows == 0 or votes is None:
        raise FreshnessError(f"Materialized cycle missing or empty: {cycle}")
    return {"rows": rows, "nominal_votes": int(votes),
            "row_hash_xor": str(xor_digest), "row_hash_sum": str(sum_digest),
            "columns": columns}


def input_signature(db_path: Path, root: Path, cycle: str) -> dict:
    """Capture all causal inputs without modifying the DuckDB or the Parquets."""
    source_before = publication_signature(root, cycle)
    import duckdb
    with duckdb.connect(str(db_path), read_only=True) as con:
        dims = {name: dimension_signature(con, name, cycle) for name in DIMENSIONS}
    source_after = publication_signature(root, cycle)
    if source_before != source_after:
        raise FreshnessError(f"Published Parquet changed during dimension read: {cycle}")
    return {"source": source_before, "dimensions": dims}


def output_signature(db_path: Path, cycle: str) -> dict:
    import duckdb
    with duckdb.connect(str(db_path), read_only=True) as con:
        return {"duckdb_version": duckdb.__version__,
                "cube": cube_signature(con, cycle)}


def revoke_certificate(directory: Path, cycle: str) -> None:
    """Move old authority out of the active directory before a mutable run."""
    active = certificate_path(directory, cycle)
    if not active.exists() and not active.is_symlink():
        return
    archive = Path(directory) / "revoked"
    archive.mkdir(parents=True, exist_ok=True)
    destination = archive / f"{active.stem}-{uuid.uuid4().hex}.json"
    os.replace(active, destination)


def issue_certificate(directory: Path, db_path: Path, root: Path, cycle: str,
                      inputs: dict, output: dict) -> dict:
    """Atomically publish new authority only after all tests have succeeded."""
    active = certificate_path(directory, cycle)
    active.parent.mkdir(parents=True, exist_ok=True)
    if active.exists() or active.is_symlink():
        raise FreshnessError(f"Refusing to overwrite active freshness certificate: {active}")
    record = {
        "version": FORMAT_VERSION, "cycle": cycle,
        "db": str(Path(db_path).resolve()), "source_root": str(Path(root).resolve()),
        "lifecycle_epoch": lifecycle_epoch(db_path),
        "certified_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": inputs, "output": output,
        "contracts": ["assert_cube_candidate_vote_conservation",
                      "assert_cube_candidate_share_and_rank"],
    }
    temporary = active.with_name(f".{active.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf8") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, active)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def audit_freshness(db_path: Path, root: Path, directory: Path,
                    cycles: set[str]) -> dict:
    """Fail closed on missing/stale/cross-warehouse attestations."""
    errors = []
    fresh = []
    for cycle in sorted(cycles):
        path = certificate_path(directory, cycle)
        try:
            if not path.is_file() or path.is_symlink():
                raise FreshnessError("missing attestation or invalid certificate path")
            with path.open(encoding="utf8") as stream:
                certificate = json.load(stream)
            if certificate.get("version") != FORMAT_VERSION:
                raise FreshnessError("unknown certificate version")
            if (certificate.get("db") != str(Path(db_path).resolve()) or
                certificate.get("source_root") != str(Path(root).resolve()) or
                certificate.get("cycle") != cycle):
                raise FreshnessError("attestation belongs to a different database/root/cycle")
            if certificate.get("lifecycle_epoch") != lifecycle_epoch(db_path):
                raise FreshnessError("warehouse lifecycle changed; recertification required")
            if certificate.get("input") != input_signature(db_path, root, cycle):
                raise FreshnessError("source Parquet or dimensions changed")
            if certificate.get("output") != output_signature(db_path, cycle):
                raise FreshnessError("materialized cube contents changed")
            fresh.append(cycle)
        except (FreshnessError, OSError, ValueError, TypeError, KeyError) as error:
            errors.append(f"{cycle}: {error}")
    if errors:
        raise FreshnessError("Freshness gate failed: " + "; ".join(errors))
    return {"fresh_cycles": fresh, "freshness": "certified",
            "checked_at_utc": datetime.now(timezone.utc).isoformat()}
