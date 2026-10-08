#!/usr/bin/env python3
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODELS=ROOT/'tse_dbt/models'
LAYERS={'bronze','silver','physical','gold','semantic'}
ALLOWED={
 'bronze':{'bronze'},
 'silver':{'bronze','silver'},
 'physical':{'bronze','silver','physical'},
 'gold':{'bronze','silver','physical','gold'},
 'semantic':{'gold','semantic'},
}
REF_RE=re.compile(r"ref\(\s*['\"]([^'\"]+)['\"]\s*\)")
models={}
for p in MODELS.rglob('*.sql'):
    rel=p.relative_to(MODELS)
    if rel.parts[0] not in LAYERS:
        raise SystemExit(f'unclassified dbt model path: {rel}')
    name=p.stem
    if name in models: raise SystemExit(f'duplicate model name: {name}')
    models[name]=(rel.parts[0],p)
fail=[]; edges=0
for name,(layer,p) in sorted(models.items()):
    for dep in REF_RE.findall(p.read_text(encoding='utf-8')):
        if dep not in models: continue
        dep_layer,dep_p=models[dep]; edges+=1
        if dep_layer not in ALLOWED[layer]:
            fail.append(f"{p.relative_to(ROOT)} [{layer}] -> {dep_p.relative_to(ROOT)} [{dep_layer}] via ref('{dep}')")
if fail:
    print('Medallion DAG contract: FAIL')
    for x in fail: print('  ',x)
    raise SystemExit(1)
counts={k:0 for k in LAYERS}
for layer,_ in models.values(): counts[layer]+=1
print('Medallion DAG contract: PASS')
print('models:', ', '.join(f'{k}={counts[k]}' for k in sorted(counts)))
print('resolved model edges:',edges)
