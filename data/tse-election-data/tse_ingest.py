#!/usr/bin/env python3
"""TSE public-data ingestion into a local immutable raw lake.

Two-phase pipeline:

* Phase 1 (network): metadata fingerprint check, resilient download, SHA-256,
  atomic materialization of the immutable source. Every freshly materialized
  source is journaled in _metadata/pending_sources.json before the barrier, so
  a crash after download never requires downloading it again.
* Phase 2 (local): extraction of selected CSV/TXT members and manifest
  publication. Active state (_metadata/resource_state.json and
  _metadata/current_objects.jsonl) changes only here, so consumers never see
  an incomplete version as active.

The immutable payload lives under
raw/election_type=.../year=.../domain=.../dataset=.../resource=.../sha256=...
and historical versions are never overwritten.

--mode is a provenance label recorded in manifests and profiles. Resource
idempotency is identical in both modes: unchanged resources with complete local
state are skipped regardless of mode. The operational difference between
backfill and incremental runs is the year scope chosen by the caller.
"""

from __future__ import annotations

import argparse
import os
import random
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import threading
import unicodedata
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests

from resource_lifecycle import (
    ResourceLifecycle,
    drive,
    fail_lifecycle,
    initial_plan_event,
    post_download_event,
    state_name,
)


class ResourceFailed(RuntimeError):
    """A resource raised during its lifecycle; from_state is where it stopped."""

    def __init__(self, from_state: str, cause: BaseException):
        super().__init__(f"failed while {from_state}: {cause}")
        self.from_state = from_state

BASE_URL = "https://dadosabertos.tse.jus.br"
API_URL = f"{BASE_URL}/api/3/action"
REQUEST_TIMEOUT = 120
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
USER_AGENT = "tse-local-lake-ingestor/3.0 (public-data research; respectful downloader)"
HEARTBEAT_SECONDS = 20.0

OPERATIONAL_RESOURCE_PATTERNS = (
    r"(?i)\bgedai\b", r"(?i)\bhash\b", r"(?i)\bsha2\b",
    r"(?i)correspond[eê]ncia", r"(?i)prepara[cç][aã]o",
)
NON_TABULAR_RESOURCE_PATTERNS = (
    r"(?i)foto", r"(?i)certid", r"(?i)proposta\s+de\s+governo",
    r"(?i)\bnotas?\s+fisc(?:al|ais)\b", r"(?i)question[aá]rio",
)

ANALYTICS_RESOURCE_DENY_PATTERNS = (
    r"(?i)motivo.*cassa[cç][aã]o",
    r"(?i)\bnotas?\s+fisc(?:al|ais)\b",
)

# Canonical regular Brazilian election calendar. We intentionally do not infer
# cycle type with a modulo rule: unknown/exceptional years must be explicit.
ELECTION_CALENDAR: dict[int, str] = {
    1994: "general", 1996: "municipal", 1998: "general", 2000: "municipal",
    2002: "general", 2004: "municipal", 2006: "general", 2008: "municipal",
    2010: "general", 2012: "municipal", 2014: "general", 2016: "municipal",
    2018: "general", 2020: "municipal", 2022: "general", 2024: "municipal",
    2026: "general",
}

def election_type_for_year(year: int, explicit: str | None = None) -> str:
    if explicit:
        known = ELECTION_CALENDAR.get(year)
        if known and known != explicit:
            raise ValueError(
                f"election type mismatch for {year}: calendar={known}, explicit={explicit}"
            )
        return explicit
    try:
        return ELECTION_CALENDAR[year]
    except KeyError as exc:
        raise ValueError(
            f"unknown election cycle for {year}; pass --election-type general|municipal "
            "for exceptional or future years"
        ) from exc

def election_scope_for_type(election_type: str) -> str:
    return "federal_state" if election_type == "general" else "municipal"

CANDIDATE_HISTORY_PATTERN = (
    r"(?i)historico[_\s-]*candidatura|hist[oó]rico\s+de\s+candidaturas"
)

DOMAIN_RULES = (
    (CANDIDATE_HISTORY_PATTERN, "candidate_history"),
    (r"(?i)consulta[_\s-]*cand[_\s-]*complement", "candidate_complement"),
    (r"(?i)bem[_\s-]*candidato|bens?\s+de\s+candidatos", "candidate_assets"),
    (r"(?i)rede[_\s-]*social|redes?\s+sociais", "candidate_social"),
    (r"(?i)consulta[_\s-]*colig|coliga[cç]", "coalition"),
    (r"(?i)consulta[_\s-]*vagas|\bvagas\b", "seats"),
    (r"(?i)consulta[_\s-]*cand|\bcandidatos?\b", "candidate"),
    (r"(?i)perfil.*se[cç][aã]o", "electorate_section"),
    (r"(?i)local\s+de\s+vota[cç][aã]o", "polling_place"),
    (r"(?i)defici[eê]ncia", "electorate_disability"),
    (r"(?i)eleitorado", "electorate"),
    (r"(?i)presta[cç][aã]o.*contas|receita|despesa|extrato", "campaign_finance"),
    (r"(?i)pesquis", "polls"),
    (r"(?i)den[uú]ncia", "complaints"),
    (r"(?i)vota[cç][aã]o|resultado|boletim.*urna", "vote_result"),
)


@dataclass(frozen=True)
class ManifestRecord:
    year: int
    election_type: str
    election_scope: str
    mode: str
    dataset_id: str
    dataset_title: str
    resource_id: str
    resource_name: str
    resource_format: str
    resource_fingerprint: str
    domain: str
    partition: str
    source_url: str
    source_sha256: str
    source_size_bytes: int
    source_object: str
    extracted_objects: list[str]
    ingested_at: str


@dataclass(frozen=True)
class ResourceProfile:
    started_at: str
    finished_at: str
    worker: str
    year: int
    election_type: str
    dataset_id: str
    resource_id: str
    resource_name: str
    result: str
    source_size_bytes: int
    extracted_objects: int
    total_seconds: float
    metadata_seconds: float
    download_seconds: float
    hash_seconds: float
    materialize_seconds: float
    extract_seconds: float
    publish_prep_seconds: float
    retry_seconds: float
    attempts: int
    resumed_bytes: int
    network_errors: int
    download_mib_per_second: float | None
    extractor: str
    selected_members: int
    lifecycle: str = ""
    error: str = ""


@dataclass
class PendingPreparation:
    root: Path
    key: str
    label: str
    started_at: str
    wall_start: float
    download_worker: str
    year: int
    election_type: str
    mode: str
    package: dict
    resource: dict
    granularity: str
    uf: str | None
    checked_at: str
    fingerprint: str
    domain: str
    source_url: str
    source_sha256: str
    source_size_bytes: int
    source_target: Path
    version_dir: Path
    metadata_seconds: float
    download_seconds: float
    hash_seconds: float
    materialize_seconds: float
    retry_seconds: float
    attempts: int
    resumed_bytes: int
    network_errors: int
    selected_member_names: list[str]
    selected_uncompressed_bytes: int
    lifecycle: object = None



def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str, *, stream=None) -> None:
    print(f"[{now_iso()}] {message}", file=stream or sys.stdout, flush=True)


_STATUS_LOCK = threading.Lock()
_WORKER_STATUS: dict[str, dict] = {}


def set_worker_status(key: str, *, label: str, phase: str) -> None:
    with _STATUS_LOCK:
        previous = _WORKER_STATUS.get(key) or {}
        _WORKER_STATUS[key] = {
            "label": label,
            "phase": phase,
            "phase_started": time.monotonic(),
            "started": previous.get("started", time.monotonic()),
        }


def clear_worker_status(key: str) -> None:
    with _STATUS_LOCK:
        _WORKER_STATUS.pop(key, None)


def worker_status_snapshot() -> list[dict]:
    now = time.monotonic()
    with _STATUS_LOCK:
        rows = [dict(v) for v in _WORKER_STATUS.values()]
    for row in rows:
        row["phase_elapsed"] = now - float(row["phase_started"])
        row["total_elapsed"] = now - float(row["started"])
    return rows


def slugify(value: str, max_len: int = 120) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return (re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "resource")[:max_len]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)



def normalize_source_url(value: str) -> str:
    """
    Normalize CKAN resource URLs.

    Some historical TSE metadata contains values like:
        "URL: https://cdn.tse.jus.br/..."

    requests treats that literally and raises
    "No connection adapters were found". Strip known presentation prefixes
    and surrounding whitespace before any URL parsing/downloading.
    """
    value = (value or "").strip()
    value = re.sub(r"(?i)^url\s*:\s*", "", value).strip()
    return value


def ckan_get(session: requests.Session, action: str, **params) -> dict:
    response = session.get(f"{API_URL}/{action}", params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN action failed: {action}: {payload}")
    return payload["result"]


def discover_packages(session: requests.Session, year: int, explicit: list[str] | None) -> list[dict]:
    if explicit:
        ids = [x.format(year=year) for x in explicit]
        return [ckan_get(session, "package_show", id=x) for x in ids]
    try:
        result = ckan_get(session, "package_search", fq=f'tags:"Ano {year}"', rows=1000)
        if result.get("results"):
            return result["results"]
    except Exception as exc:
        print(f"[warn] tagged discovery failed for {year}: {exc}", file=sys.stderr)
    result = ckan_get(session, "package_search", q=str(year), rows=1000)
    return [p for p in result.get("results", []) if str(year) in json.dumps(p, ensure_ascii=False)]


def dataset_family(dataset_id: str) -> str:
    """Classify a CKAN package before looking at individual resource names.

    Package-level classification has precedence. This prevents resources named
    "Candidatos" inside campaign-finance datasets from leaking into the analytics
    profile as candidate master data.
    """
    value = slugify(dataset_id)
    value = re.sub(r"_(19|20)\d{2}(?:_subtemas)?$", "", value)

    if value.startswith("candidatos"):
        return "candidates"
    if value.startswith("eleitorado"):
        return "electorate"
    if value.startswith("resultados") or value.startswith("votacao"):
        return "results"
    if value.startswith("comparecimento_e_abstencao"):
        return "turnout"
    if value.startswith("prestacao_de_contas"):
        return "campaign_finance"
    if value.startswith("pesquis"):
        return "polls"
    if value.startswith("denunc"):
        return "complaints"
    return "other"


def normalize_resource_domain(
    domain: str,
    *,
    resource_name: str = "",
    source_url: str = "",
) -> str:
    # Normalize semantic domains, including legacy control-plane state.
    if domain != "candidate":
        return domain
    text = " ".join([resource_name or "", normalize_source_url(source_url or "")])
    if re.search(CANDIDATE_HISTORY_PATTERN, text):
        return "candidate_history"
    return domain


def infer_domain(package: dict, resource: dict) -> str:
    family = dataset_family(package.get("name") or "")

    # Dataset family wins for semantically strong package types.
    if family == "campaign_finance":
        return "campaign_finance"
    if family == "polls":
        return "polls"
    if family == "complaints":
        return "complaints"
    if family == "results":
        return "vote_result"

    text = " ".join([
        resource.get("name") or "",
        normalize_source_url(resource.get("url") or ""),
        resource.get("description") or "",
    ])
    for pattern, domain in DOMAIN_RULES:
        if re.search(pattern, text):
            return normalize_resource_domain(
                domain,
                resource_name=resource.get("name") or "",
                source_url=resource.get("url") or "",
            )

    # Known core families remain meaningful even when a historical resource name
    # does not match one of the fine-grained rules.
    if family == "candidates":
        return normalize_resource_domain(
            "candidate",
            resource_name=resource.get("name") or "",
            source_url=resource.get("url") or "",
        )
    if family in {"electorate", "turnout"}:
        return "electorate"
    return "other"


def is_tabular(resource: dict) -> bool:
    fmt = (resource.get("format") or "").lower()
    path = urlparse(normalize_source_url(resource.get("url") or "")).path.lower()
    return fmt in {"csv", "txt", "zip"} or path.endswith((".csv", ".txt", ".zip"))


def matches_any(patterns: Iterable[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def resource_partition(resource: dict) -> str:
    """Return an explicit CKAN-level partition when the resource itself is UF-scoped.

    GLOBAL means the CKAN resource is not explicitly partitioned. A GLOBAL ZIP may
    still contain BRASIL/UF members, which are filtered later by select_zip_members().
    """
    name = (resource.get("name") or "").strip().upper()
    url_path = urlparse(normalize_source_url(resource.get("url") or "")).path.upper()

    # TSE commonly names already-partitioned resources as `GO - ...`.
    match = re.match(
        r"^(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MG|MS|MT|PA|PB|PE|PI|PR|RJ|RN|RO|RR|RS|SC|SE|SP|TO|ZZ)\s*[-–—]",
        name,
    )
    if match:
        return match.group(1)

    if re.match(r"^BRASIL\s*[-–—]", name):
        return "BRASIL"
    if re.match(r"^BR\s*[-–—]", name):
        return "BR"

    # Some resources encode the partition only in the physical filename.
    filename = Path(url_path).name
    match = re.search(
        r"(?:^|_)(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MG|MS|MT|PA|PB|PE|PI|PR|RJ|RN|RO|RR|RS|SC|SE|SP|TO|ZZ)(?:_|\.|$)",
        filename,
    )
    if match:
        return match.group(1)
    if re.search(r"(?:^|_)BRASIL(?:_|\.|$)", filename):
        return "BRASIL"
    return "GLOBAL"


def is_archive_resource(resource: dict) -> bool:
    path = urlparse(normalize_source_url(resource.get("url") or "")).path.lower()
    fmt = (resource.get("format") or "").lower()
    return fmt == "zip" or path.endswith(".zip")


def resource_matches_granularity(resource: dict, granularity: str, uf: str | None) -> bool:
    """Filter CKAN resources before download.

    ZIP-member filtering still happens later. This function handles the important
    case where TSE publishes one CKAN resource per UF (for example electorate by
    section), which otherwise bypasses select_zip_members entirely.
    """
    partition = resource_partition(resource)

    if granularity == "all":
        return True

    if granularity == "brasil":
        # Keep national/unpartitioned resources; reject already UF-scoped ones.
        return partition in {"GLOBAL", "BRASIL", "BR"}

    if granularity in {"uf", "section"}:
        if uf:
            wanted = uf.upper()
            if partition == wanted:
                return True
            # An unpartitioned archive may contain a member for the requested UF.
            return partition == "GLOBAL" and is_archive_resource(resource)

        # No UF specified: accept explicit UF resources and generic archives whose
        # members can be split by UF, but not national flat files.
        if partition not in {"GLOBAL", "BRASIL", "BR", "ZZ"}:
            return True
        return partition == "GLOBAL" and is_archive_resource(resource)

    raise ValueError(granularity)


def resource_allowed(package: dict, resource: dict, profile: str, regex: re.Pattern | None) -> bool:
    text = " ".join([
        resource.get("name") or "",
        normalize_source_url(resource.get("url") or ""),
        resource.get("format") or "",
    ])
    if not is_tabular(resource) or (regex and not regex.search(text)):
        return False

    if profile == "mirror":
        return True

    if matches_any(OPERATIONAL_RESOURCE_PATTERNS, text) or matches_any(NON_TABULAR_RESOURCE_PATTERNS, text):
        return False

    family = dataset_family(package.get("name") or "")
    domain = infer_domain(package, resource)

    core_families = {"candidates", "electorate", "turnout", "results"}
    analytics_core_domains = {
        "candidate", "candidate_complement", "candidate_assets", "candidate_social",
        "coalition", "seats", "electorate", "electorate_disability",
        "polling_place", "electorate_section", "vote_result",
    }
    extended_core_domains = analytics_core_domains | {"candidate_history"}

    if profile == "analytics":
        if family not in core_families or domain not in analytics_core_domains:
            return False
        if matches_any(ANALYTICS_RESOURCE_DENY_PATTERNS, text):
            return False

        # `comparecimento-e-abstencao-*` contains several auxiliary resources
        # that duplicate the electorate family. For the lean analytical core we
        # retain only the canonical turnout/abstention resource.
        if family == "turnout":
            name = resource.get("name") or ""
            return bool(re.search(r"(?i)comparecimento.*absten[cç][aã]o", name))

        return True

    if profile == "extended":
        return (
            (family in core_families and domain in extended_core_domains)
            or family in {"campaign_finance", "polls", "complaints"}
        )

    raise ValueError(profile)


def resource_fingerprint(resource: dict) -> str:
    # Fields chosen to avoid a GET when CKAN says nothing relevant changed.
    payload = {
        key: resource.get(key)
        for key in ("id", "url", "hash", "size", "last_modified", "metadata_modified", "created", "format")
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def hash_file(path: Path) -> tuple[str, int, float]:
    started = time.perf_counter()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(DOWNLOAD_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size, time.perf_counter() - started


def _retry_delay(attempt: int, response: requests.Response | None = None) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(30.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
    base = min(8.0, 0.75 * (2 ** max(0, attempt - 1)))
    return base + random.uniform(0.0, min(1.0, base * 0.25))


def download_resilient(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    max_attempts: int = 4,
) -> tuple[str, int, float, int, float, int, int]:
    """Download with retry/backoff and best-effort HTTP Range resume.

    Returns:
      sha256, size, hash_seconds, attempts, retry_seconds, resumed_bytes, network_errors
    """
    part = destination.with_suffix(destination.suffix + ".part")
    attempts = 0
    retry_seconds = 0.0
    resumed_bytes = 0
    network_errors = 0

    while attempts < max_attempts:
        attempts += 1
        existing = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        response = None
        try:
            response = session.get(
                url, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=True, headers=headers
            )

            if response.status_code in RETRYABLE_STATUS:
                delay = _retry_delay(attempts, response)
                response.close()
                if attempts >= max_attempts:
                    response.raise_for_status()
                time.sleep(delay)
                retry_seconds += delay
                continue

            response.raise_for_status()

            append = existing > 0 and response.status_code == 206
            if append:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {existing}-"):
                    append = False
            if existing and not append:
                existing = 0
                part.unlink(missing_ok=True)

            mode = "ab" if append else "wb"
            if append:
                resumed_bytes += existing

            with part.open(mode) as fh:
                for chunk in response.iter_content(DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
            response.close()
            part.replace(destination)
            sha256, size, hash_seconds = hash_file(destination)
            return sha256, size, hash_seconds, attempts, retry_seconds, resumed_bytes, network_errors

        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
            network_errors += 1
            if response is not None:
                response.close()
            if attempts >= max_attempts:
                raise
            delay = _retry_delay(attempts)
            time.sleep(delay)
            retry_seconds += delay
        except requests.HTTPError:
            if response is not None and response.status_code not in RETRYABLE_STATUS:
                raise
            if attempts >= max_attempts:
                raise
            delay = _retry_delay(attempts, response)
            if response is not None:
                response.close()
            time.sleep(delay)
            retry_seconds += delay

    raise RuntimeError(f"download failed after {max_attempts} attempts: {url}")


def member_partition(filename: str) -> str:
    upper = filename.upper()
    if "BRASIL" in upper:
        return "BRASIL"
    match = re.search(r"(?:^|_)(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MG|MS|MT|PA|PB|PE|PI|PR|RJ|RN|RO|RR|RS|SC|SE|SP|TO|ZZ)(?:_|\.|$)", upper)
    return match.group(1) if match else "GLOBAL"


def select_zip_members(members: list[zipfile.ZipInfo], granularity: str, uf: str | None) -> list[zipfile.ZipInfo]:
    by_partition: dict[str, list[zipfile.ZipInfo]] = {}
    for info in members:
        by_partition.setdefault(member_partition(info.filename), []).append(info)

    if granularity == "all":
        return members

    if granularity == "brasil":
        # Never explode a UF-only archive into every state when the caller asked
        # for Brasil granularity. Prefer explicit national aggregates, then a
        # genuinely unpartitioned member. If none exists, select nothing.
        for p in ("BRASIL", "BR", "GLOBAL"):
            if p in by_partition:
                return by_partition[p]
        return []

    if granularity in {"uf", "section"}:
        if uf:
            wanted = uf.upper()
            if wanted in by_partition:
                return by_partition[wanted]
            # Historical archives sometimes contain one flat national file rather
            # than UF members. Keep it so downstream models may filter the UF.
            return by_partition.get("GLOBAL", [])

        return [
            info
            for partition, infos in by_partition.items()
            if partition not in {"BRASIL", "BR", "GLOBAL", "ZZ"}
            for info in infos
        ]

    raise ValueError(granularity)


def _unique_target(destination: Path, filename: str, used: set[str]) -> Path:
    base = Path(filename).name
    candidate = base
    stem = Path(base).stem
    suffix = Path(base).suffix
    n = 2
    while candidate.lower() in used:
        candidate = f"{stem}_{n}{suffix}"
        n += 1
    used.add(candidate.lower())
    return destination / candidate


def native_member_is_literal_safe(member: str) -> bool:
    """`unzip` interprets *, ? and [ in member names as wildcard patterns.

    A member whose literal name contains these characters could match nothing,
    or the wrong member, so such names must not go through the native binary.
    """
    return not any(char in member for char in "*?[")


def _extract_member_native(source: Path, member: str, target: Path) -> None:
    part = target.with_suffix(target.suffix + ".part")
    part.unlink(missing_ok=True)
    with part.open("wb") as out:
        cp = subprocess.run(
            ["unzip", "-qq", "-p", str(source), member],
            stdout=out,
            stderr=subprocess.PIPE,
            check=False,
        )
    if cp.returncode != 0:
        part.unlink(missing_ok=True)
        stderr = cp.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"native unzip failed for {Path(source).name}:{member} "
            f"(exit={cp.returncode}): {stderr}"
        )
    part.replace(target)


def _extract_member_python(source: Path, info: zipfile.ZipInfo, target: Path) -> None:
    part = target.with_suffix(target.suffix + ".part")
    part.unlink(missing_ok=True)
    with zipfile.ZipFile(source) as zf:
        with zf.open(info) as src, part.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    part.replace(target)


def extract_selected_members(
    source: Path,
    destination: Path,
    granularity: str,
    uf: str | None,
    extractor: str,
) -> tuple[list[Path], str, list[str]]:
    """Extract selected CSV/TXT members, preferring the native `unzip` binary.

    Python's zipfile remains useful for cheap archive inspection and as a portable
    fallback, but large TSE members are materially faster through native unzip on
    WSL/Linux. Members whose names contain wildcard syntax always go through
    zipfile under `auto`, and fail explicitly under `native`.
    """
    if not zipfile.is_zipfile(source):
        return [], "none", []

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as zf:
        members = [
            i for i in zf.infolist()
            if not i.is_dir() and Path(i.filename).suffix.lower() in {".csv", ".txt"}
        ]
        selected = select_zip_members(members, granularity, uf)

    if extractor == "native" and shutil.which("unzip") is None:
        raise RuntimeError("--extractor native requested but `unzip` is not installed")

    backend = extractor
    if extractor == "auto":
        backend = "native" if shutil.which("unzip") else "python"

    unsafe_native_members = [
        info.filename
        for info in selected
        if not native_member_is_literal_safe(info.filename)
    ]
    if backend == "native" and unsafe_native_members:
        if extractor == "auto":
            backend = "python"
        else:
            raise RuntimeError(
                "native unzip cannot safely extract member names "
                "containing wildcard syntax; use --extractor python: "
                + ", ".join(unsafe_native_members)
            )

    used: set[str] = set()
    outputs: list[Path] = []
    selected_names: list[str] = []

    for info in selected:
        target = _unique_target(destination, info.filename, used)
        if backend == "native":
            _extract_member_native(source, info.filename, target)
        elif backend == "python":
            _extract_member_python(source, info, target)
        else:
            raise ValueError(f"unsupported extractor: {backend}")
        outputs.append(target)
        selected_names.append(info.filename)

    return outputs, backend, selected_names


def metadata_dir(root: Path) -> Path:
    return root / "_metadata"


def manifest_path(root: Path) -> Path:
    return metadata_dir(root) / "ingest_manifest.jsonl"


def state_path(root: Path) -> Path:
    return metadata_dir(root) / "resource_state.json"


def pending_path(root: Path) -> Path:
    return metadata_dir(root) / "pending_sources.json"


def current_objects_path(root: Path) -> Path:
    return metadata_dir(root) / "current_objects.jsonl"


def profile_path(root: Path) -> Path:
    return metadata_dir(root) / "ingest_profile.jsonl"


def append_profile(root: Path, profile: ResourceProfile) -> None:
    path = profile_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(profile), ensure_ascii=False) + "\n")


def record_failure(
    root: Path,
    *,
    no_profile: bool,
    exc: BaseException,
    year: int,
    election_type: str,
    package: dict,
    resource: dict,
    worker: str,
) -> None:
    """Persist a failed resource to ingest_profile.jsonl (audit trail across runs).

    Honors --no-profile like every other profile row. The lifecycle field records
    the state the resource stopped in; error keeps the message for diagnosis.
    """
    if no_profile:
        return
    now = now_iso()
    rid = resource.get("id") or slugify(resource.get("url") or "resource")
    append_profile(root, ResourceProfile(
        started_at=now, finished_at=now, worker=worker, year=year,
        election_type=election_type, dataset_id=package.get("name") or "",
        resource_id=rid, resource_name=resource.get("name") or rid, result="failed",
        source_size_bytes=0, extracted_objects=0, total_seconds=0.0, metadata_seconds=0.0,
        download_seconds=0.0, hash_seconds=0.0, materialize_seconds=0.0, extract_seconds=0.0,
        publish_prep_seconds=0.0, retry_seconds=0.0, attempts=0, resumed_bytes=0,
        network_errors=0, download_mib_per_second=None, extractor="none", selected_members=0,
        lifecycle=getattr(exc, "from_state", "unknown"), error=str(exc),
    ))


def state_key(year: int, election_type: str, resource_id: str, granularity: str, uf: str | None) -> str:
    return f"{election_type}|{year}|{resource_id}|{granularity}|{(uf or '').upper()}"


def load_state(root: Path) -> dict[str, dict]:
    path = state_path(root)
    if not path.exists():
        return {}
    raw: dict[str, dict] = json.loads(path.read_text(encoding="utf-8"))
    migrated: dict[str, dict] = {}
    for row in raw.values():
        year = int(row["year"])
        election_type = row.get("election_type") or election_type_for_year(year)
        row = {
            **row,
            "election_type": election_type,
            "election_scope": row.get("election_scope") or election_scope_for_type(election_type),
            "domain": normalize_resource_domain(
                row.get("domain") or "other",
                resource_name=row.get("resource_name") or "",
                source_url=row.get("source_url") or "",
            ),
        }
        key = state_key(
            year, election_type, row["resource_id"],
            row.get("granularity", "brasil"), row.get("uf") or None,
        )
        previous = migrated.get(key)
        if previous is None or row.get("checked_at", "") >= previous.get("checked_at", ""):
            migrated[key] = row
    return migrated


def save_state(root: Path, state: dict[str, dict]) -> None:
    atomic_json(state_path(root), state)
    path = current_objects_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for key in sorted(state):
            row = state[key]
            source_obj = row.get("source_object") or ""
            domain = normalize_resource_domain(
                row.get("domain") or "other",
                resource_name=row.get("resource_name") or "",
                source_url=row.get("source_url") or "",
            )
            # Resolve the cycle once: the persisted type is authoritative, and the
            # calendar is only a fallback for legacy rows that never stored it.
            election_type = (
                row.get("election_type")
                or election_type_for_year(int(row["year"]))
            )
            election_scope = (
                row.get("election_scope")
                or election_scope_for_type(election_type)
            )
            for obj in row.get("extracted_objects") or []:
                if not (root / obj).is_file():
                    continue
                if source_obj and not (root / source_obj).is_file():
                    continue
                fh.write(json.dumps({
                    "year": row["year"],
                    "election_type": election_type,
                    "election_scope": election_scope,
                    "dataset_id": row["dataset_id"],
                    "dataset_title": row.get("dataset_title") or "",
                    "resource_id": row["resource_id"],
                    "resource_name": row.get("resource_name") or "",
                    "resource_format": row.get("resource_format") or "",
                    "domain": domain,
                    "partition": row["partition"],
                    "source_sha256": row["source_sha256"],
                    "source_object": row.get("source_object") or "",
                    "source_url": row.get("source_url") or "",
                    "extractor": row.get("extractor") or "",
                    "object": obj,
                    "updated_at": row["checked_at"],
                }, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_pending(root: Path) -> dict[str, dict]:
    """Downloaded-but-not-yet-published sources, keyed like the active state."""
    path = pending_path(root)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_pending(root: Path, pending: dict[str, dict]) -> None:
    atomic_json(pending_path(root), pending)


def append_manifest(root: Path, record: ManifestRecord) -> None:
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def resource_output_dir(root: Path, year: int, election_type: str, package: dict, resource: dict, domain: str) -> Path:
    return root / "raw" / f"election_type={election_type}" / f"year={year}" / f"domain={slugify(domain)}" / f"dataset={slugify(package.get('name') or 'dataset')}" / f"resource={slugify(resource.get('id') or resource.get('name') or 'resource')}"


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session



def local_state_complete(root: Path, previous: dict | None) -> bool:
    if not previous:
        return False
    source = previous.get("source_object")
    if not source or not (root / source).exists():
        return False
    extracted = previous.get("extracted_objects") or []
    if not extracted:
        # A Brasil-granularity archive may legitimately have no national member.
        # Once inspected, that is still a complete local decision for this resource.
        return bool(previous.get("selection_complete"))
    return all((root / obj).exists() for obj in extracted)


def pending_state_row(pending: PendingPreparation) -> dict:
    """Journal row for a source that is durable on disk but not yet published."""
    rid = pending.resource.get("id") or slugify(pending.resource.get("url") or "resource")
    return {
        "year": pending.year,
        "election_type": pending.election_type,
        "election_scope": election_scope_for_type(pending.election_type),
        "mode": pending.mode,
        "dataset_id": pending.package.get("name") or "",
        "dataset_title": pending.package.get("title") or "",
        "resource_id": rid,
        "resource_name": pending.resource.get("name") or rid,
        "resource_format": pending.resource.get("format") or "",
        "resource_fingerprint": pending.fingerprint,
        "domain": pending.domain,
        "partition": "PENDING",
        "source_url": pending.source_url,
        "source_sha256": pending.source_sha256,
        "source_size_bytes": pending.source_size_bytes,
        "source_object": str(pending.source_target.relative_to(pending.root)),
        "extracted_objects": [],
        "granularity": pending.granularity,
        "uf": (pending.uf or "").upper(),
        "checked_at": pending.checked_at,
        "selection_complete": False,
        "extractor": "",
        "selected_members": list(pending.selected_member_names),
    }


def inspect_selected_archive_members(
    source: Path, granularity: str, uf: str | None
) -> tuple[list[str], int]:
    """Return selected member names and their total uncompressed bytes."""
    if not zipfile.is_zipfile(source):
        return [], 0
    with zipfile.ZipFile(source) as zf:
        members = [
            i for i in zf.infolist()
            if not i.is_dir() and Path(i.filename).suffix.lower() in {".csv", ".txt"}
        ]
        selected = select_zip_members(members, granularity, uf)
    return [i.filename for i in selected], sum(int(i.file_size) for i in selected)


def download_resource_worker(**kwargs) -> dict:
    """Run one resource's Phase-1 lifecycle, recording failures in the machine.

    Any exception leaves the machine in `failed` (or keeps it final) and is
    re-raised as ResourceFailed, which names the state the resource stopped in.
    The outcome carries the final lifecycle state; a pending outcome carries the
    live machine so Phase 2 can continue the same lifecycle.
    """
    life = ResourceLifecycle()
    try:
        outcome = _download_resource_body(life=life, **kwargs)
    except Exception as exc:
        raise ResourceFailed(fail_lifecycle(life), exc) from exc
    if "profile" in outcome:
        outcome["profile"] = replace(outcome["profile"], lifecycle=state_name(life))
    outcome["lifecycle"] = state_name(life)
    if outcome["kind"] == "pending":
        outcome["pending"].lifecycle = life
    return outcome


def _download_resource_body(
    *, life: ResourceLifecycle, root: Path, year: int, election_type: str, mode: str,
    package: dict, resource: dict, granularity: str, uf: str | None,
    previous: dict | None, force: bool, pending_row: dict | None = None,
) -> dict:
    """Phase 1: metadata check, resume-from-local-source or download, atomic materialization.

    No ZIP extraction occurs here. Fresh resources return a PendingPreparation that
    is consumed only after every download future has finished.

    A local source recorded in the pending journal (or an incomplete active row)
    with an unchanged CKAN fingerprint is reused without any network request.
    """
    wall_start = time.perf_counter()
    started_at = now_iso()
    worker_name = threading.current_thread().name

    rid = resource.get("id") or slugify(resource.get("url") or "resource")
    key = state_key(year, election_type, rid, granularity, uf)
    label = f"{year} :: {package.get('name')} :: {resource.get('name')}"
    set_worker_status(key, label=label, phase="metadata")

    phase_start = time.perf_counter()
    fingerprint = resource_fingerprint(resource)
    checked_at = now_iso()
    metadata_seconds = time.perf_counter() - phase_start

    # Lifecycle decision: the machine enforces legal transitions, while the
    # booleans below are computed here from CKAN fingerprints and local state.
    prev_match = bool(previous) and previous.get("resource_fingerprint") == fingerprint
    pending_match = bool(pending_row) and pending_row.get("resource_fingerprint") == fingerprint
    if force:
        resume_row = None
    elif pending_match:
        resume_row = pending_row
    elif prev_match:
        resume_row = previous
    else:
        resume_row = None
    resume_present = bool(
        resume_row
        and resume_row.get("source_object")
        and (root / resume_row["source_object"]).is_file()
    )

    plan = initial_plan_event(
        force=force,
        fingerprint_matches=prev_match or pending_match,
        local_complete=prev_match and local_state_complete(root, previous),
        local_source_present=resume_present,
    )
    drive(life, plan)

    if plan == "plan_metadata_skip":
        updated = {**previous, "checked_at": checked_at}
        total_seconds = time.perf_counter() - wall_start
        profile = ResourceProfile(
            started_at=started_at, finished_at=now_iso(), worker=worker_name,
            year=year, election_type=election_type, dataset_id=package.get("name") or "",
            resource_id=rid, resource_name=resource.get("name") or rid,
            result="metadata-skip", source_size_bytes=int(previous.get("source_size_bytes") or 0),
            extracted_objects=len(previous.get("extracted_objects") or []),
            total_seconds=total_seconds, metadata_seconds=metadata_seconds,
            download_seconds=0.0, hash_seconds=0.0, materialize_seconds=0.0, extract_seconds=0.0,
            publish_prep_seconds=0.0, retry_seconds=0.0, attempts=0, resumed_bytes=0, network_errors=0,
            download_mib_per_second=None, extractor="none",
            selected_members=len(previous.get("selected_members") or previous.get("extracted_objects") or []),
        )
        return {"kind": "metadata-skip", "key": key, "state_row": updated, "profile": profile}

    # Resume from a durable local source without touching the network. The pending
    # journal takes precedence; an incomplete active row (for example after cache
    # pruning removed extracted objects) is the fallback.
    resume = resume_row if plan == "plan_rehydrate" else None

    if resume is not None:
        source_rel = resume.get("source_object") or ""
        source_target = root / source_rel if source_rel else None
        if source_target is not None and source_target.is_file():
            source_sha256 = resume.get("source_sha256") or ""
            source_size = int(resume.get("source_size_bytes") or source_target.stat().st_size)
            rehydrate_hash_seconds = 0.0
            if not source_sha256:
                source_sha256, source_size, rehydrate_hash_seconds = hash_file(source_target)

            domain = infer_domain(package, resource)
            version_dir = source_target.parent.parent
            set_worker_status(key, label=label, phase="inspect-local-source")
            if zipfile.is_zipfile(source_target):
                selected_names, selected_bytes = inspect_selected_archive_members(
                    source_target, granularity, uf
                )
            elif source_target.suffix.lower() in {".csv", ".txt"}:
                selected_names = [source_target.name]
                selected_bytes = int(source_target.stat().st_size)
            else:
                selected_names, selected_bytes = [], 0

            pending = PendingPreparation(
                root=root, key=key, label=label, started_at=started_at,
                wall_start=wall_start, download_worker=worker_name, year=year,
                election_type=election_type, mode=mode, package=package,
                resource=resource, granularity=granularity, uf=uf,
                checked_at=checked_at, fingerprint=fingerprint, domain=domain,
                source_url=normalize_source_url(resource["url"]),
                source_sha256=source_sha256, source_size_bytes=source_size,
                source_target=source_target, version_dir=version_dir,
                metadata_seconds=metadata_seconds, download_seconds=0.0,
                hash_seconds=rehydrate_hash_seconds, materialize_seconds=0.0,
                retry_seconds=0.0, attempts=0, resumed_bytes=0,
                network_errors=0, selected_member_names=selected_names,
                selected_uncompressed_bytes=selected_bytes,
            )
            drive(life, "rehydrated")
            return {"kind": "pending", "key": key, "pending": pending}

    if resume is not None:
        # Planned as a rehydrate, but the local source disappeared before it was read.
        # The wrapper records the failure; the journal row still points at the
        # missing file, so the next run plans a download instead of looping here.
        raise RuntimeError(f"rehydrate source vanished: {resume.get('source_object')}")

    with make_session() as session:
        source_url = normalize_source_url(resource["url"])
        suffix = Path(urlparse(source_url).path).suffix or ".bin"
        staging_dir = root / "_staging" / "downloads"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging_name = f"{year}_{slugify(rid, 80)}_{hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:10]}{suffix}"
        tmp_source = staging_dir / staging_name

        set_worker_status(key, label=label, phase="download")
        phase_start = time.perf_counter()
        sha256, size, hash_seconds, attempts, retry_seconds, resumed_bytes, network_errors = download_resilient(
            session, source_url, tmp_source
        )
        download_seconds = max(0.0, time.perf_counter() - phase_start - retry_seconds - hash_seconds)

        after_download = post_download_event(
            force=force,
            same_sha256_as_previous=bool(previous) and previous.get("source_sha256") == sha256,
        )
        drive(life, after_download)
        if after_download == "same_content":
            updated = {**previous, "resource_fingerprint": fingerprint, "checked_at": checked_at}
            tmp_source.unlink(missing_ok=True)
            total_seconds = time.perf_counter() - wall_start
            mib = size / (1024 * 1024)
            profile = ResourceProfile(
                started_at=started_at, finished_at=now_iso(), worker=worker_name,
                year=year, election_type=election_type, dataset_id=package.get("name") or "",
                resource_id=rid, resource_name=resource.get("name") or rid, result="content-skip",
                source_size_bytes=size, extracted_objects=len(previous.get("extracted_objects") or []),
                total_seconds=total_seconds, metadata_seconds=metadata_seconds,
                download_seconds=download_seconds, hash_seconds=hash_seconds, materialize_seconds=0.0,
                extract_seconds=0.0, publish_prep_seconds=0.0, retry_seconds=retry_seconds, attempts=attempts,
                resumed_bytes=resumed_bytes, network_errors=network_errors,
                download_mib_per_second=(mib / download_seconds) if download_seconds > 0 else None,
                extractor="none", selected_members=len(previous.get("selected_members") or previous.get("extracted_objects") or []),
            )
            return {"kind": "content-skip", "key": key, "state_row": updated, "profile": profile}

        domain = infer_domain(package, resource)
        version_dir = resource_output_dir(root, year, election_type, package, resource, domain) / f"sha256={sha256}"
        source_dir = version_dir / "source"

        set_worker_status(key, label=label, phase="materialize")
        phase_start = time.perf_counter()
        source_dir.mkdir(parents=True, exist_ok=True)
        original_name = Path(urlparse(source_url).path).name or f"{rid}{suffix}"
        source_target = source_dir / original_name
        if source_target.exists():
            tmp_source.unlink(missing_ok=True)
        else:
            tmp_source.replace(source_target)
        materialize_seconds = time.perf_counter() - phase_start

        set_worker_status(key, label=label, phase="inspect")
        if zipfile.is_zipfile(source_target):
            selected_names, selected_bytes = inspect_selected_archive_members(
                source_target, granularity, uf
            )
        elif suffix.lower() in {".csv", ".txt"}:
            selected_names, selected_bytes = [original_name], int(size)
        else:
            selected_names, selected_bytes = [], 0

        pending = PendingPreparation(
            root=root, key=key, label=label, started_at=started_at, wall_start=wall_start,
            download_worker=worker_name, year=year, election_type=election_type, mode=mode,
            package=package, resource=resource, granularity=granularity, uf=uf,
            checked_at=checked_at, fingerprint=fingerprint, domain=domain, source_url=source_url,
            source_sha256=sha256, source_size_bytes=size, source_target=source_target, version_dir=version_dir,
            metadata_seconds=metadata_seconds, download_seconds=download_seconds, hash_seconds=hash_seconds,
            materialize_seconds=materialize_seconds, retry_seconds=retry_seconds, attempts=attempts,
            resumed_bytes=resumed_bytes, network_errors=network_errors,
            selected_member_names=selected_names, selected_uncompressed_bytes=selected_bytes,
        )
        return {"kind": "pending", "key": key, "pending": pending}


def prepare_resource_worker(*, pending: PendingPreparation, extractor: str) -> tuple[dict, ManifestRecord, ResourceProfile]:
    """Phase 2: extract/publish one already-downloaded immutable source."""
    key = pending.key
    set_worker_status(key, label=pending.label, phase="prepare")
    prepare_worker = threading.current_thread().name

    extracted: list[Path] = []
    extraction_backend = "none"
    extract_seconds = 0.0
    selected_member_names = list(pending.selected_member_names)

    if zipfile.is_zipfile(pending.source_target):
        set_worker_status(key, label=pending.label, phase="extract")
        phase_start = time.perf_counter()
        extracted_dir = pending.version_dir / "extracted"
        extracted, extraction_backend, selected_member_names = extract_selected_members(
            pending.source_target, extracted_dir, pending.granularity, pending.uf, extractor
        )
        extract_seconds = time.perf_counter() - phase_start
    elif pending.source_target.suffix.lower() in {".csv", ".txt"}:
        extracted = [pending.source_target]
        extraction_backend = "direct"

    set_worker_status(key, label=pending.label, phase="publish")
    phase_start = time.perf_counter()
    partition = member_partition(extracted[0].name) if len(extracted) == 1 else ("MULTI" if extracted else "GLOBAL")
    rid = pending.resource.get("id") or slugify(pending.resource.get("url") or "resource")
    record = ManifestRecord(
        year=pending.year, election_type=pending.election_type,
        election_scope=election_scope_for_type(pending.election_type), mode=pending.mode,
        dataset_id=pending.package.get("name") or "", dataset_title=pending.package.get("title") or "",
        resource_id=rid, resource_name=pending.resource.get("name") or rid,
        resource_format=pending.resource.get("format") or "", resource_fingerprint=pending.fingerprint,
        domain=pending.domain, partition=partition, source_url=pending.source_url,
        source_sha256=pending.source_sha256, source_size_bytes=pending.source_size_bytes,
        source_object=str(pending.source_target.relative_to(pending.root)),
        extracted_objects=[str(p.relative_to(pending.root)) for p in extracted],
        ingested_at=now_iso(),
    )
    (pending.version_dir / "manifest.json").write_text(
        json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    state_row = {
        **asdict(record), "granularity": pending.granularity, "uf": (pending.uf or "").upper(),
        "checked_at": pending.checked_at, "selection_complete": True, "extractor": extraction_backend,
        "selected_members": selected_member_names,
    }
    publish_prep_seconds = time.perf_counter() - phase_start

    total_seconds = time.perf_counter() - pending.wall_start
    mib = pending.source_size_bytes / (1024 * 1024)
    profile = ResourceProfile(
        started_at=pending.started_at, finished_at=now_iso(),
        worker=f"{pending.download_worker}->{prepare_worker}",
        year=pending.year, election_type=pending.election_type, dataset_id=pending.package.get("name") or "",
        resource_id=rid, resource_name=pending.resource.get("name") or rid, result="ingested",
        source_size_bytes=pending.source_size_bytes, extracted_objects=len(extracted),
        total_seconds=total_seconds, metadata_seconds=pending.metadata_seconds,
        download_seconds=pending.download_seconds, hash_seconds=pending.hash_seconds,
        materialize_seconds=pending.materialize_seconds, extract_seconds=extract_seconds,
        publish_prep_seconds=publish_prep_seconds, retry_seconds=pending.retry_seconds,
        attempts=pending.attempts, resumed_bytes=pending.resumed_bytes, network_errors=pending.network_errors,
        download_mib_per_second=(mib / pending.download_seconds) if pending.download_seconds > 0 else None,
        extractor=extraction_backend, selected_members=len(selected_member_names),
    )
    return state_row, record, profile

def prepare_resource_worker_guarded(*, pending: PendingPreparation, extractor: str):
    """Phase-2 entry point: a preparation failure is recorded in the machine."""
    try:
        return prepare_resource_worker(pending=pending, extractor=extractor)
    except Exception as exc:
        raise ResourceFailed(fail_lifecycle(pending.lifecycle), exc) from exc


def resolve_years(args: argparse.Namespace) -> list[int]:
    if args.from_year is not None or args.to_year is not None:
        if args.from_year is None or args.to_year is None:
            raise SystemExit("--from-year and --to-year must be used together")
        if args.from_year > args.to_year:
            raise SystemExit("--from-year cannot be greater than --to-year")
        return list(range(args.from_year, args.to_year + 1))
    return sorted(set(args.year or [2026]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill/incremental ingestion of TSE public data into a local raw lake.")
    p.add_argument(
        "--mode",
        choices=["backfill", "incremental"],
        default="incremental",
        help=(
            "Run provenance label written to manifests and profiles. Resource "
            "idempotency is identical in both modes; callers choose the year scope."
        ),
    )
    p.add_argument("--year", type=int, action="append", help="Year to ingest; repeat for multiple years. Default: 2026")
    p.add_argument("--election-type", choices=["general", "municipal"], help="Explicit cycle type; required for years not in the canonical calendar")
    p.add_argument("--from-year", type=int)
    p.add_argument("--to-year", type=int)
    p.add_argument("--root", type=Path, default=Path("./data/tse"))
    p.add_argument("--profile", choices=["analytics", "extended", "mirror"], default="analytics")
    p.add_argument("--granularity", choices=["brasil", "uf", "section", "all"], default="brasil")
    p.add_argument("--uf")
    p.add_argument("--dataset", action="append", dest="datasets", help="Explicit CKAN id; {year} placeholder is supported")
    p.add_argument("--resource-regex")
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--force", action="store_true", help="Ignore state/fingerprint checks and re-download")
    p.add_argument("--workers", type=int, default=min(8, max(4, (os.cpu_count() or 2) * 2)), help="Parallel resource workers (default: %(default)s). Use 1 for serial execution.")
    p.add_argument("--checkpoint-every", type=int, default=10, help="Persist control-plane state every N completed resources (default: %(default)s).")
    p.add_argument("--extractor", choices=["auto", "native", "python"], default="auto",
                   help="ZIP extraction backend. auto prefers the native `unzip` binary (default: %(default)s).")
    p.add_argument("--extract-workers", type=int, default=1,
                   help="Preparation workers after the download barrier (default: %(default)s).")
    p.add_argument("--heartbeat-seconds", type=float, default=HEARTBEAT_SECONDS,
                   help="Log in-flight phases while long resources run; 0 disables (default: %(default)s).")
    p.add_argument("--no-profile", action="store_true",
                   help="Do not persist per-resource profiling JSONL. Run accounting is unaffected.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.uf and args.granularity not in {"uf", "section"}:
        raise SystemExit("--uf requires --granularity uf or section")
    years = resolve_years(args)
    regex = re.compile(args.resource_regex) if args.resource_regex else None

    discovery_session = make_session()
    selected: list[tuple[int, str, dict, dict]] = []
    for year in years:
        try:
            election_type = election_type_for_year(year, args.election_type)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        try:
            packages = discover_packages(discovery_session, year, args.datasets)
        except Exception as exc:
            print(f"[error] discovery {year}: {exc}", file=sys.stderr)
            continue
        for package in sorted(packages, key=lambda p: p.get("name", "")):
            for resource in package.get("resources", []):
                if (
                    resource_allowed(package, resource, args.profile, regex)
                    and resource_matches_granularity(resource, args.granularity, args.uf)
                ):
                    selected.append((year, election_type, package, resource))

    print(
        f"Selected {len(selected)} resource(s) across {len(years)} year(s); "
        f"mode={args.mode}, profile={args.profile}."
    )
    for year, election_type, package, resource in selected:
        print(
            f"  {year} [{election_type}] :: {package.get('name')} :: "
            f"{resource.get('name')} [{infer_domain(package, resource)}]"
        )
    if args.list_only:
        return 0

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.extract_workers < 1:
        raise SystemExit("--extract-workers must be >= 1")
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be >= 1")
    if args.heartbeat_seconds < 0:
        raise SystemExit("--heartbeat-seconds must be >= 0")
    if args.extractor == "native" and shutil.which("unzip") is None:
        raise SystemExit("--extractor native requires the `unzip` executable")

    resolved_extractor = args.extractor
    if resolved_extractor == "auto":
        resolved_extractor = "native" if shutil.which("unzip") else "python"

    pipeline_started_at = now_iso()
    pipeline_start = time.perf_counter()
    args.root.mkdir(parents=True, exist_ok=True)
    state_load_start = time.perf_counter()
    state = load_state(args.root)
    pending_journal = load_pending(args.root)
    state_load_seconds = time.perf_counter() - state_load_start

    counts = {"ingested": 0, "metadata-skip": 0, "content-skip": 0, "errors": 0}
    completed = 0   # Phase-1 outcomes durably known (skip, error, or source journaled)
    prepared = 0    # Phase-2 outcomes (published or failed)
    transferred_bytes = 0
    run_profiles: list[ResourceProfile] = []
    pending_prepare: list[PendingPreparation] = []

    jobs = []
    for year, election_type, package, resource in selected:
        rid = resource.get("id") or slugify(resource.get("url") or "resource")
        key = state_key(year, election_type, rid, args.granularity, args.uf)
        jobs.append((year, election_type, package, resource, state.get(key), pending_journal.get(key)))

    # ------------------------------------------------------------------
    # PHASE 1: NETWORK / RAW MATERIALIZATION
    # ------------------------------------------------------------------
    download_phase_start = time.perf_counter()
    log(
        f"Phase 1/2 download: workers={args.workers}; resources={len(jobs)}; "
        "ZIP extraction is disabled until the download barrier."
    )

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="tse-download") as pool:
        future_map = {
            pool.submit(
                download_resource_worker,
                root=args.root,
                year=year,
                election_type=election_type,
                mode=args.mode,
                package=package,
                resource=resource,
                granularity=args.granularity,
                uf=args.uf,
                previous=previous,
                force=args.force,
                pending_row=pending_row,
            ): (year, election_type, package, resource)
            for year, election_type, package, resource, previous, pending_row in jobs
        }

        futures_pending = set(future_map)
        phase_done = 0
        while futures_pending:
            timeout = args.heartbeat_seconds if args.heartbeat_seconds > 0 else None
            done, futures_pending = wait(
                futures_pending, timeout=timeout, return_when=FIRST_COMPLETED
            )

            if not done:
                statuses = sorted(
                    worker_status_snapshot(),
                    key=lambda row: float(row.get("total_elapsed") or 0),
                    reverse=True,
                )
                log(
                    f"[heartbeat:download] finished={phase_done}/{len(jobs)} "
                    f"active={len(statuses)} pending={len(futures_pending)}"
                )
                for row in statuses[: min(5, len(statuses))]:
                    log(
                        f"[running] phase={row['phase']} "
                        f"phase_elapsed={row['phase_elapsed']:.1f}s "
                        f"total={row['total_elapsed']:.1f}s :: {row['label']}"
                    )
                continue

            for future in done:
                year, election_type, package, resource = future_map[future]
                rid = resource.get("id") or slugify(resource.get("url") or "resource")
                status_key = state_key(year, election_type, rid, args.granularity, args.uf)
                label = f"{year} :: {package.get('name')} :: {resource.get('name')}"
                try:
                    outcome = future.result()
                    kind = outcome["kind"]
                    if kind == "pending":
                        item: PendingPreparation = outcome["pending"]
                        pending_prepare.append(item)
                        # Journal before the barrier: the source is durable, so a crash
                        # from here on resumes locally. The active state is untouched.
                        pending_journal[item.key] = pending_state_row(item)
                        save_pending(args.root, pending_journal)
                        log(
                            f"[downloaded] {label} :: {item.source_size_bytes / (1024 * 1024):.1f} MiB :: "
                            f"download={item.download_seconds:.2f}s hash={item.hash_seconds:.2f}s "
                            f"materialize={item.materialize_seconds:.3f}s "
                            f"prepare_raw={item.selected_uncompressed_bytes / (1024 * 1024):.1f} MiB "
                            f"members={len(item.selected_member_names)}"
                        )
                    else:
                        counts[kind] += 1
                        state[outcome["key"]] = outcome["state_row"]
                        if pending_journal.pop(outcome["key"], None) is not None:
                            save_pending(args.root, pending_journal)
                        profile: ResourceProfile = outcome["profile"]
                        run_profiles.append(profile)
                        if profile.attempts > 0:
                            transferred_bytes += profile.source_size_bytes
                        if kind == "metadata-skip":
                            log(f"[unchanged-metadata] {label} :: total={profile.total_seconds:.3f}s")
                        else:
                            log(
                                f"[unchanged-content] {label} :: total={profile.total_seconds:.2f}s "
                                f"download={profile.download_seconds:.2f}s hash={profile.hash_seconds:.2f}s "
                                f"retries={max(0, profile.attempts - 1)}"
                            )
                        if not args.no_profile:
                            append_profile(args.root, profile)
                    completed += 1
                except Exception as exc:
                    counts["errors"] += 1
                    completed += 1
                    log(f"[error:download] {label}: {exc}", stream=sys.stderr)
                    record_failure(
                        args.root, no_profile=args.no_profile, exc=exc, year=year,
                        election_type=election_type, package=package, resource=resource,
                        worker=threading.current_thread().name,
                    )
                finally:
                    clear_worker_status(status_key)

                phase_done += 1
                if completed % args.checkpoint_every == 0:
                    checkpoint_start = time.perf_counter()
                    save_state(args.root, state)
                    log(
                        f"[checkpoint] completed={completed}/{len(jobs)} :: "
                        f"{time.perf_counter() - checkpoint_start:.3f}s"
                    )

    download_phase_seconds = time.perf_counter() - download_phase_start
    save_state(args.root, state)
    save_pending(args.root, pending_journal)
    log(
        f"Download barrier reached: wall={download_phase_seconds:.2f}s; "
        f"to_prepare={len(pending_prepare)}; skips={counts['metadata-skip'] + counts['content-skip']}; "
        f"download_errors={counts['errors']}."
    )

    # ------------------------------------------------------------------
    # PHASE 2: LOCAL PREPARATION / EXTRACTION
    # ------------------------------------------------------------------
    # Smallest selected uncompressed payload first. Flat CSV/TXT and ZIPs with no
    # national member naturally sort to the front. This improves time-to-first-ready
    # and avoids a multi-GiB member monopolizing the preparation stage immediately.
    pending_prepare.sort(
        key=lambda item: (item.selected_uncompressed_bytes, item.source_size_bytes, item.label)
    )

    prepare_phase_start = time.perf_counter()
    if pending_prepare:
        log(
            f"Phase 2/2 prepare: workers={args.extract_workers}; extractor={resolved_extractor}; "
            f"resources={len(pending_prepare)}; order=smallest-uncompressed-first."
        )
        for i, item in enumerate(pending_prepare[:10], 1):
            log(
                f"[prepare-queue #{i}] raw={item.selected_uncompressed_bytes / (1024 * 1024):.1f} MiB :: "
                f"{item.label}"
            )

        with ThreadPoolExecutor(
            max_workers=args.extract_workers, thread_name_prefix="tse-prepare"
        ) as pool:
            future_map = {
                pool.submit(
                    prepare_resource_worker_guarded,
                    pending=item,
                    extractor=args.extractor,
                ): item
                for item in pending_prepare
            }
            futures_pending = set(future_map)
            phase_done = 0

            while futures_pending:
                timeout = args.heartbeat_seconds if args.heartbeat_seconds > 0 else None
                done, futures_pending = wait(
                    futures_pending, timeout=timeout, return_when=FIRST_COMPLETED
                )

                if not done:
                    statuses = sorted(
                        worker_status_snapshot(),
                        key=lambda row: float(row.get("total_elapsed") or 0),
                        reverse=True,
                    )
                    log(
                        f"[heartbeat:prepare] finished={phase_done}/{len(pending_prepare)} "
                        f"active={len(statuses)} pending={len(futures_pending)}"
                    )
                    for row in statuses[: min(5, len(statuses))]:
                        log(
                            f"[running] phase={row['phase']} "
                            f"phase_elapsed={row['phase_elapsed']:.1f}s "
                            f"total={row['total_elapsed']:.1f}s :: {row['label']}"
                        )
                    continue

                for future in done:
                    item = future_map[future]
                    try:
                        state_row, record, profile = future.result()
                        state[item.key] = state_row
                        pending_journal.pop(item.key, None)
                        append_manifest(args.root, record)
                        # Publish only after the manifest is durable; a failure above
                        # is recorded as a failed lifecycle in the handler below.
                        drive(item.lifecycle, "publish")
                        profile = replace(profile, lifecycle=state_name(item.lifecycle))
                        counts["ingested"] += 1
                        run_profiles.append(profile)
                        if profile.attempts > 0:
                            transferred_bytes += profile.source_size_bytes
                        log(
                            f"[ingested] {item.label} :: {record.source_size_bytes / (1024 * 1024):.1f} MiB :: "
                            f"{len(record.extracted_objects)} object(s) :: total={profile.total_seconds:.2f}s "
                            f"download={profile.download_seconds:.2f}s hash={profile.hash_seconds:.2f}s "
                            f"materialize={profile.materialize_seconds:.3f}s extract={profile.extract_seconds:.2f}s "
                            f"extractor={profile.extractor} retries={max(0, profile.attempts - 1)} "
                            f"retry_wait={profile.retry_seconds:.2f}s "
                            f"resumed={profile.resumed_bytes / (1024 * 1024):.1f}MiB"
                        )
                        if not args.no_profile:
                            append_profile(args.root, profile)
                    except Exception as exc:
                        fail_lifecycle(item.lifecycle)
                        # The journal entry is kept, so the next run retries extraction
                        # from the local source without touching the network.
                        counts["errors"] += 1
                        log(f"[error:prepare] {item.label}: {exc}", stream=sys.stderr)
                        record_failure(
                            args.root, no_profile=args.no_profile, exc=exc, year=item.year,
                            election_type=item.election_type, package=item.package,
                            resource=item.resource, worker=item.download_worker,
                        )
                    finally:
                        clear_worker_status(item.key)

                    phase_done += 1
                    prepared += 1
                    if prepared % args.checkpoint_every == 0:
                        checkpoint_start = time.perf_counter()
                        save_state(args.root, state)
                        save_pending(args.root, pending_journal)
                        log(
                            f"[checkpoint] prepared={prepared}/{len(pending_prepare)} :: "
                            f"{time.perf_counter() - checkpoint_start:.3f}s"
                        )

    prepare_phase_seconds = time.perf_counter() - prepare_phase_start

    final_save_start = time.perf_counter()
    save_state(args.root, state)
    save_pending(args.root, pending_journal)
    final_save_seconds = time.perf_counter() - final_save_start
    pipeline_finished_at = now_iso()
    pipeline_seconds = time.perf_counter() - pipeline_start

    if run_profiles:
        phase_totals = {
            "metadata": sum(p.metadata_seconds for p in run_profiles),
            "download": sum(p.download_seconds for p in run_profiles),
            "hash": sum(p.hash_seconds for p in run_profiles),
            "materialize": sum(p.materialize_seconds for p in run_profiles),
            "extract": sum(p.extract_seconds for p in run_profiles),
            "publish_prep": sum(p.publish_prep_seconds for p in run_profiles),
            "retry": sum(p.retry_seconds for p in run_profiles),
        }
        log(
            "Profile totals (summed worker time): "
            + ", ".join(f"{k}={v:.2f}s" for k, v in phase_totals.items())
        )
        slowest = sorted(run_profiles, key=lambda p: p.total_seconds, reverse=True)[:5]
        for i, row in enumerate(slowest, 1):
            log(
                f"[slowest #{i}] {row.year} :: {row.dataset_id} :: {row.resource_name} :: "
                f"total={row.total_seconds:.2f}s download={row.download_seconds:.2f}s "
                f"hash={row.hash_seconds:.2f}s extract={row.extract_seconds:.2f}s "
                f"attempts={row.attempts} retry_wait={row.retry_seconds:.2f}s"
            )

    log("Summary: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    log(
        f"Pipeline phases: download_wall={download_phase_seconds:.2f}s "
        f"prepare_wall={prepare_phase_seconds:.2f}s"
    )
    log(
        f"Pipeline: started={pipeline_started_at} finished={pipeline_finished_at} "
        f"wall={pipeline_seconds:.2f}s state_load={state_load_seconds:.3f}s "
        f"final_save={final_save_seconds:.3f}s "
        f"transferred={transferred_bytes / (1024 * 1024):.1f} MiB "
        f"unfinished_sources={len(pending_journal)}"
    )
    log(f"State: {state_path(args.root).resolve()}")
    log(f"Current objects: {current_objects_path(args.root).resolve()}")
    if not args.no_profile:
        log(f"Profile: {profile_path(args.root).resolve()}")
    return 1 if counts["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
