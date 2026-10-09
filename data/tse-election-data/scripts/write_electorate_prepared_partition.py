#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import duckdb

def q(v:str)->str: return "'"+v.replace("'","''")+"'"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-root',type=Path,required=True); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--year',type=int,required=True); ap.add_argument('--election-type',required=True); ap.add_argument('--encoding',default='latin-1'); a=ap.parse_args()
    raw=a.raw_root.resolve(); idx=raw/'_metadata/current_objects.jsonl'
    rows=[json.loads(x) for x in idx.read_text(encoding='utf-8').splitlines() if x.strip()]
    selected=[r for r in rows if int(r.get('year',-1))==a.year and r.get('election_type')==a.election_type and r.get('domain')=='electorate' and str(r.get('resource_name','')).startswith('Eleitorado - ')]
    if len(selected)!=1: raise RuntimeError(f'Expected exactly one canonical electorate object, found {len(selected)}')
    r=selected[0]; src=raw/r['object']
    if not src.is_file(): raise RuntimeError(f'Missing current electorate object: {src}')
    scope=str(r.get('election_scope') or '')
    part=a.root.resolve()/f'election_type={a.election_type}'/f'year={a.year}'; part.mkdir(parents=True,exist_ok=True)
    final=part/'data.parquet'; tmp=part/'data.parquet.tmp'; tmp.unlink(missing_ok=True)
    scan=f"""read_csv({q(str(src))}, delim=';', quote='"', escape='"', header=true, all_varchar=true, union_by_name=true, filename=true, sample_size=20480, encoding={q(a.encoding)}, strict_mode=false, null_padding=true, ignore_errors=false, parallel=false)"""
    con=duckdb.connect()
    try:
        source_count=con.execute(f'select count(*) from {scan}').fetchone()[0]
        if source_count==0: raise RuntimeError('Refusing to publish an empty electorate partition')
        con.execute(f"""copy (select * exclude(filename), {q(a.election_type)}::varchar as _election_type, {q(scope)}::varchar as _election_scope, filename::varchar as source_file from {scan}) to {q(str(tmp))} (format parquet, compression zstd)""")
    finally: con.close()
    check=duckdb.connect()
    try: written=check.execute('select count(*) from read_parquet(?)',[str(tmp)]).fetchone()[0]
    finally: check.close()
    if written!=source_count: raise RuntimeError(f'Prepared row-count mismatch: source={source_count} parquet={written}')
    os.replace(tmp,final); print('prepared electorate partition:',f'{a.year}/{a.election_type}'); print('rows:',written); print('path:',final)
    return 0
if __name__=='__main__': raise SystemExit(main())
