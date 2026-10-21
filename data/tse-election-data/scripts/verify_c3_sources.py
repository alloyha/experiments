"""Offline C3 source inventory attestation, without warehouse writes.

Authenticates the immutable source bytes against durable active-version state,
checks zip CRC, selected member bytes against extracted objects, and ensures
current_objects.jsonl matches the required source selection. NOT a proof that
an independently supplied reference DuckDB was built from those bytes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

YEARS = {2018: 'general', 2020:'municipal', 2022:'general', 2024:'municipal', 2026:'general'}

def _safe(root: Path, rel: str) -> Path:
    if not rel or PurePosixPath(rel).is_absolute() or '..' in PurePosixPath(rel).parts:
        raise ValueError(f'unsafe relative path: {rel}')
    p = root / rel
    if not p.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'path escapes root: {rel}')
    return p

def _digest(path: Path) -> tuple[str,int]:
    h=hashlib.sha256(); size=0
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b);size+=len(b)
    return h.hexdigest(),size

def _compare_zip_member(zf, name, target):
    # Stream-decompress and compare bytes, not merely CRC32 metadata.
    with zf.open(name) as a, target.open('rb') as b:
        while True:
            x=a.read(1024*1024);y=b.read(1024*1024)
            if x!=y: raise ValueError(f'extracted content differs from verified source: {target}')
            if not x: break

def attest(root: Path):
    root=root.resolve(strict=True)
    state_path=root/'_metadata/resource_state.json'
    index_path=root/'_metadata/current_objects.jsonl'
    state=json.loads(state_path.read_text())
    index=[json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    selected=[]
    for row in state.values():
        year=int(row['year']); kind=row.get('election_type')
        if year not in YEARS or kind!=YEARS[year] or row.get('domain')!='vote_result' or not row.get('resource_name','').casefold().startswith('detalhe da apuração por município e zona'):
            continue
        selected.append(row)
    if len(selected)!=5 or {int(x['year']) for x in selected}!=set(YEARS):
        raise ValueError(f'exactly five active tally resource versions required, got {len(selected)}')
    expected_index=[]; evidence=[]
    for row in sorted(selected,key=lambda x:int(x['year'])):
        if not row.get('selection_complete') or not row.get('source_sha256'):
            raise ValueError('incomplete authoritative active version')
        source=_safe(root,row['source_object'])
        actual,size=_digest(source)
        if actual!=row['source_sha256'] or (row.get('source_size_bytes') is not None and size!=int(row['source_size_bytes'])):
            raise ValueError(f'ZIP SHA/size mismatch for {row["year"]}')
        members=row.get('selected_members') or []
        outputs=row.get('extracted_objects') or []
        if not members or len(members)!=len(outputs):
            raise ValueError(f'incomplete selected-member mapping: {row["year"]}')
        if not zipfile.is_zipfile(source):
            raise ValueError('C3 ZIP format required')
        with zipfile.ZipFile(source) as z:
            for member,output in zip(members,outputs):
                if member not in z.namelist(): raise ValueError(f'member missing: {member}')
                _compare_zip_member(z,member,_safe(root,output))
                expected_index.append((int(row['year']),row['election_type'],row['resource_id'],row['source_sha256'],output))
        evidence.append({'cycle':f'{row["year"]}:{row["election_type"]}','sha256':actual,'members':len(members)})
    actual_index=[(int(i['year']),i['election_type'],i['resource_id'],i['source_sha256'],i['object']) for i in index if i.get('domain')=='vote_result' and str(i.get('resource_name','')).casefold().startswith('detalhe da apuração por município e zona') and int(i.get('year',0)) in YEARS]
    if Counter(actual_index)!=Counter(expected_index):
        raise ValueError('current_objects tally index does not match durable active state')
    return {'status':'PASS','cycles':evidence,'note':'Source bytes verified; no reference-DuckDB build provenance established'}

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args(argv)
    try: print(json.dumps(attest(a.root),ensure_ascii=False,indent=2));return 0
    except (OSError,ValueError,KeyError,zipfile.BadZipFile,json.JSONDecodeError) as e:
        print(json.dumps({'status':'FAIL','error':str(e)}));return 1
if __name__=='__main__':raise SystemExit(main())
