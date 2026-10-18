#!/usr/bin/env python3
"""Read-only forensic diagnosis of missing TSE histórico candidatura 2026 source.

Reads resource_state, pending journal, manifest, profiles, physical index and
filesystem candidates. Does not perform network I/O, downloads, extraction or writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath

RESOURCE_ID = '6c3a8826-32da-460f-b306-97474b580efb'
YEAR = 2026
META_FILES = ('resource_state.json', 'pending_resources.json', 'ingest_manifest.jsonl',
              'ingest_profile.jsonl', 'current_objects.jsonl')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as reader:
        for block in iter(lambda: reader.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def safe_resolve(root: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative or '\\' in relative:
        return None
    p = PurePosixPath(relative)
    if p.is_absolute() or any(part in {'.', '..'} for part in relative.split('/')):
        return None
    result = root.joinpath(*p.parts)
    if not result.resolve(strict=False).is_relative_to(root):
        return None
    if result.is_symlink():
        return None
    return result


def matching_row(row: object, resource_id: str, source_sha: str) -> bool:
    if not isinstance(row, dict):
        return False
    return str(row.get('resource_id') or '') == resource_id or (bool(source_sha) and
           (row.get('source_sha256') == source_sha or
            source_sha in str(row.get('source_object') or '')))


def jsonl_events(path: Path, resource_id: str, sha: str) -> list[dict]:
    if not path.exists():
        return []
    result = []
    with path.open(encoding='utf-8') as file:
        for lineno, line in enumerate(file, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if matching_row(item, resource_id, sha):
                columns = ('ingested_at', 'started_at', 'finished_at', 'result',
                           'lifecycle', 'mode', 'domain', 'resource_id', 'source_object',
                           'source_sha256', 'source_size_bytes', 'source_url',
                           'resource_fingerprint', 'selected_members')
                result.append({'line': lineno, **{k: item[k] for k in columns if k in item}})
    return result[-12:]


def collect_paths(roots: list[Path], expected_name: str, expected_size: int | None,
                  expected_sha: str, scan_by_size: bool) -> tuple[dict, list[dict]]:
    counters = {'files_examined': 0, 'hashes_computed': 0, 'scan_errors': 0}
    findings = []
    visited = set()
    for root in roots:
        if not root.is_dir():
            counters['scan_errors'] += 1
            continue
        def onerror(_error: OSError) -> None:
            counters['scan_errors'] += 1
        for directory, subdirs, names in os.walk(root, followlinks=False, onerror=onerror):
            subdirs[:] = [d for d in subdirs if not (Path(directory) / d).is_symlink()
                          and d not in {'.git', '.venv', 'dbt_packages', 'node_modules'}]
            for name in names:
                candidate = Path(directory) / name
                if candidate.is_symlink():
                    continue
                try:
                    stat = candidate.stat()
                    key = (stat.st_dev, stat.st_ino)
                    if key in visited:
                        continue
                    visited.add(key)
                    counters['files_examined'] += 1
                    by_name = name == expected_name
                    by_size = bool(scan_by_size and expected_size is not None
                                   and stat.st_size == expected_size)
                    if not by_name and not by_size:
                        continue
                    counters['hashes_computed'] += 1
                    actual = digest(candidate)
                    findings.append({'path': str(candidate), 'bytes': stat.st_size,
                                     'match_sha256': actual == expected_sha,
                                     'matched_filename': by_name,
                                     'matched_expected_size': expected_size == stat.st_size})
                except OSError:
                    counters['scan_errors'] += 1
    return counters, findings


def audit(raw_root: Path, extra_roots: list[Path] | None = None, *,
          scan_by_size: bool = False, resource_id: str = RESOURCE_ID) -> dict:
    root = raw_root.resolve(strict=True)
    meta = root / '_metadata'
    state_path = meta / 'resource_state.json'
    if not state_path.is_file():
        raise FileNotFoundError(f'Authoritative state missing: {state_path}')
    state = json.loads(state_path.read_text(encoding='utf-8'))
    if not isinstance(state, dict):
        raise ValueError('resource_state.json must contain an object')
    matches = [(key, row) for key, row in state.items() if isinstance(row, dict)
               and row.get('resource_id') == resource_id and int(row.get('year', 0)) == YEAR]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one 2026 active state row, got {len(matches)}')
    key, row = matches[0]
    sha = str(row.get('source_sha256') or '')
    if not re.fullmatch('[0-9a-f]{64}', sha):
        raise ValueError('Expected source SHA-256 missing or invalid')
    expected_size = int(row['source_size_bytes']) if row.get('source_size_bytes') is not None else None
    source_rel = row.get('source_object')
    source = safe_resolve(root, source_rel)
    if source is None:
        raise ValueError('Recorded source path invalid or escapes raw root')
    source_exists = source.is_file()
    state_stats_before = {f: ((meta / f).stat().st_mtime_ns, (meta / f).stat().st_size)
                          for f in META_FILES if (meta / f).is_file()}
    candidate_roots = [root / 'raw'] + [p.resolve() for p in extra_roots or []]
    names = Path(str(source_rel)).name
    counts, alternatives = collect_paths(candidate_roots, names, expected_size, sha, scan_by_size)
    pending_path = meta / 'pending_resources.json'
    pending = json.loads(pending_path.read_text(encoding='utf-8')) if pending_path.is_file() else {}
    if not isinstance(pending, dict):
        raise ValueError('pending_resources.json must contain an object')
    pending_rows = [{'key': k, 'row': v} for k, v in pending.items()
                    if k == key or matching_row(v, resource_id, sha)]
    current_path = meta / 'current_objects.jsonl'
    current_count = 0
    if current_path.exists():
        with current_path.open(encoding='utf-8') as file:
            for line in file:
                if line.strip() and matching_row(json.loads(line), resource_id, sha):
                    current_count += 1
    outputs = []
    for value in row.get('extracted_objects') or []:
        path = safe_resolve(root, value)
        outputs.append({'path': value, 'exists': bool(path and path.is_file())})
    state_stats_after = {f: ((meta / f).stat().st_mtime_ns, (meta / f).stat().st_size)
                         for f in META_FILES if (meta / f).is_file()}
    exact_elsewhere = [a for a in alternatives if a['match_sha256'] and
                       Path(a['path']).resolve(strict=False) != source.resolve(strict=False)]
    actual_source_hash = digest(source) if source_exists else None
    return {
        'read_only': True,
        'active_key': key,
        'logical_domain': ('candidate_history' if row.get('domain') == 'candidate'
                           and ('historico' in str(row.get('resource_name', '')).lower()
                                or 'histórico' in str(row.get('resource_name', '')).lower())
                           else row.get('domain')),
        'stored_domain': row.get('domain'),
        'resource_name': row.get('resource_name'),
        'source_url': row.get('source_url'),
        'dataset_id': row.get('dataset_id'),
        'checked_at': row.get('checked_at'),
        'selection_complete': row.get('selection_complete'),
        'source_object': source_rel,
        'source_exists': source_exists,
        'source_sha256_expected': sha,
        'source_size_expected': expected_size,
        'source_hash_matches': actual_source_hash == sha if actual_source_hash else None,
        'missing_outputs': [x['path'] for x in outputs if not x['exists']],
        'pending_rows': pending_rows,
        'physical_index_rows': current_count,
        'manifest_recent': jsonl_events(meta / 'ingest_manifest.jsonl', resource_id, sha),
        'profiles_recent': jsonl_events(meta / 'ingest_profile.jsonl', resource_id, sha),
        'scanned_roots': [str(p) for p in candidate_roots],
        'scan_by_size': scan_by_size,
        'scan_counts': counts,
        'possible_alternatives': alternatives[:30],
        'matching_hash_elsewhere': exact_elsewhere[:30],
        'metadata_changed_while_scanning': state_stats_before != state_stats_after,
        'conclusion': ('source_exists_and_sha_matches' if actual_source_hash == sha else
                       'source_relocated_exact_sha_found' if exact_elsewhere else
                       'source_not_found_in_scanned_locations' if not source_exists else
                       'source_present_but_sha_mismatch'),
        'next_step': ('Do not alter active state. Stop concurrent writers; verify provenance and plan restore.'
                      if exact_elsewhere else
                      'No authorized deletion. Examine ingest history/source_url and profile selection; plan reacquisition separately.'
                      if not source_exists else
                      'Existing source needs consistency review.'),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--search-root', type=Path, action='append', default=[])
    parser.add_argument('--scan-by-size', action='store_true', help='hash any scanned file with expected byte count')
    parser.add_argument('--resource-id', default=RESOURCE_ID)
    parser.add_argument('--output', choices=('json', 'text'), default='text')
    args = parser.parse_args()
    try:
        result = audit(args.raw_root, args.search_root, scan_by_size=args.scan_by_size,
                       resource_id=args.resource_id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f'Cannot complete read-only diagnosis: {exc}', file=sys.stderr)
        return 2
    if args.output == 'json':
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print('TSE 2026 HISTORY — READ ONLY')
        for k in ('conclusion', 'logical_domain', 'stored_domain', 'source_exists',
                  'source_size_expected', 'source_sha256_expected', 'selection_complete',
                  'physical_index_rows', 'metadata_changed_while_scanning'):
            print(f'{k}: {result[k]}')
        print(f'missing_outputs: {len(result["missing_outputs"])}')
        print(f'pending_rows: {len(result["pending_rows"])}')
        print(f'manifest_history: {len(result["manifest_recent"])} record(s)')
        print(f'profile_history: {len(result["profiles_recent"])} record(s)')
        print(f'scanned: {result["scan_counts"]}')
        print(f'matching_hash_elsewhere: {result["matching_hash_elsewhere"]}')
        print(f'source_url: {result["source_url"]}')
        print('Use --output json for historical record details.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
