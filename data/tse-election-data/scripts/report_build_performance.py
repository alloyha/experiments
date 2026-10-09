#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path


def load(path: Path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    results = payload.get('results', [])
    if not isinstance(results, list):
        raise SystemExit(f'Invalid run_results.json: {path}')
    return results


def seconds(row):
    value = row.get('execution_time')
    return float(value) if isinstance(value, (int,float)) else 0.0


def uid(row):
    return str(row.get('unique_id') or (row.get('node') or {}).get('unique_id') or '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-results', type=Path, default=Path('tse_dbt/target/run_results.json'))
    ap.add_argument('--output', type=Path)
    ap.add_argument('--baseline', type=Path)
    ap.add_argument('--write-baseline', type=Path)
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--max-regression-pct', type=float)
    ap.add_argument('--min-seconds', type=float, default=1.0)
    args = ap.parse_args()

    rows=[]
    for row in load(args.run_results):
        unique_id=uid(row)
        if not unique_id:
            continue
        s=seconds(row)
        rows.append({
            'unique_id': unique_id,
            'name': unique_id.split('.')[-1],
            'resource_type': unique_id.split('.')[0],
            'status': str(row.get('status','unknown')),
            'seconds': round(s,6),
        })
    rows.sort(key=lambda r:(-r['seconds'], r['unique_id']))
    total=sum(r['seconds'] for r in rows)

    baseline={}
    if args.baseline and args.baseline.exists():
        payload=json.loads(args.baseline.read_text(encoding='utf-8'))
        entries=payload.get('nodes', payload)
        if isinstance(entries, list):
            baseline={r['unique_id']:float(r['seconds']) for r in entries}
        else:
            baseline={str(k):float(v) for k,v in entries.items()}

    print('=== DBT PERFORMANCE REPORT ===')
    print('nodes:', len(rows))
    print(f'summed node execution: {total:.2f}s')
    print(f"{'#':>3} {'seconds':>9} {'share':>7} {'type':>6}  node")
    for i,row in enumerate(rows[:args.top],1):
        share=(100.0*row['seconds']/total) if total else 0.0
        print(f"{i:>3} {row['seconds']:>9.2f} {share:>6.1f}% {row['resource_type']:>6}  {row['name']}")

    regressions=[]
    if baseline:
        print('\n=== BASELINE DELTAS ===')
        for row in rows:
            old=baseline.get(row['unique_id'])
            if old is None or old <= 0 or row['seconds'] < args.min_seconds:
                continue
            pct=100.0*(row['seconds']-old)/old
            if pct > 0:
                print(f"{pct:+7.1f}% {old:8.2f}s -> {row['seconds']:8.2f}s  {row['name']}")
            if args.max_regression_pct is not None and pct > args.max_regression_pct:
                regressions.append((row['unique_id'],old,row['seconds'],pct))

    report={'schema_version':1,'total_node_seconds':round(total,6),'nodes':rows}
    for target in [args.output, args.write_baseline]:
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
            print(('baseline' if target==args.write_baseline else 'report')+':', target)

    if regressions:
        print('\nPerformance regressions:')
        for unique_id,old,new,pct in regressions:
            print(f'  {unique_id}: {old:.2f}s -> {new:.2f}s ({pct:+.1f}%)')
        return 1
    return 0

if __name__=='__main__':
    raise SystemExit(main())
