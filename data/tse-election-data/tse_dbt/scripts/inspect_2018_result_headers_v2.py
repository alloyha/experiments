#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("data/tse")
RAW = ROOT / "raw"

TARGETS = (
    "Votação em partido por município e zona",
    "Detalhe da apuração por município e zona",
)

def norm(s: str) -> str:
    return " ".join((s or "").casefold().split())

def manifest_name(obj: dict) -> str:
    # Tolerate manifest schema evolution.
    for key in ("resource_name", "name", "resource_title", "title"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value
    resource = obj.get("resource")
    if isinstance(resource, dict):
        for key in ("name", "title"):
            value = resource.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""

matches = []

for manifest in RAW.rglob("manifest.json"):
    try:
        obj = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        continue

    name = manifest_name(obj)
    if not name:
        continue

    # Scope explicitly to year 2018, either from manifest or path.
    year = obj.get("year")
    if year not in (2018, "2018") and "year=2018" not in manifest.as_posix():
        continue

    if any(norm(target) in norm(name) for target in TARGETS):
        extracted_dir = manifest.parent / "extracted"
        files = sorted(p for p in extracted_dir.glob("*") if p.is_file())
        matches.append((name, manifest, files))

if not matches:
    print("No matching manifests found.")
    print("\nUseful diagnostics:")
    print(f"  manifests under raw: {sum(1 for _ in RAW.rglob('manifest.json'))}")
    print("\n2018 vote_result-like manifests:")
    for manifest in RAW.rglob("manifest.json"):
        if "year=2018" not in manifest.as_posix():
            continue
        try:
            obj = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = manifest_name(obj)
        if any(word in norm(name) for word in ("votação", "apur", "partido")):
            print(" -", name, "::", manifest)
    raise SystemExit(1)

for name, manifest, files in matches:
    print(f"\n=== {name} ===")
    print("manifest:", manifest)

    if not files:
        print("No files under:", manifest.parent / "extracted")
        continue

    for path in files:
        print("\nfile:", path)
        try:
            with path.open("r", encoding="latin-1", newline="") as f:
                reader = csv.reader(f, delimiter=";")
                header = next(reader)
        except Exception as exc:
            print("ERROR reading header:", exc)
            continue

        print(f"columns ({len(header)}):")
        for col in header:
            print(" ", col)
