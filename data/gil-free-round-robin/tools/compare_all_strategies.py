#!/usr/bin/env python3
"""
tools/compare_all_strategies.py

Read per-strategy JSON pairs in results/ and produce a combined CSV + table + PNG plot.

Usage:
  python tools/compare_all_strategies.py --results-dir results --out results/compare_all.csv --plot results/compare_all.png
"""
from __future__ import annotations
import json, math, csv, argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

def load_json(path: Path):
    if not path.exists():
        return None
    try:
        txt = path.read_text()
        if not txt.strip():
            return None
        return json.loads(txt)
    except Exception:
        return None

def best_entry_from_summary(j: Dict[str, Any]) -> Dict[str, Any]:
    # If the JSON contains multiple "results", pick first (we expect one strategy per run)
    if not j or "results" not in j or not j["results"]:
        return {}
    # If multiple, choose the one with highest throughput
    best = max(j["results"], key=lambda r: float(r.get("throughput") or 0.0))
    return best

def gather(results_dir: Path, strategies: List[str]) -> List[Dict[str,Any]]:
    rows = []
    for strat in strategies:
        nogil_path = results_dir / f"{strat}_nogil.json"
        gil_path = results_dir / f"{strat}_gil.json"
        nogil_j = load_json(nogil_path)
        gil_j = load_json(gil_path)
        nog = best_entry_from_summary(nogil_j) if nogil_j else {}
        gil = best_entry_from_summary(gil_j) if gil_j else {}
        nog_thr = float(nog.get("throughput") or math.nan)
        gil_thr = float(gil.get("throughput") or math.nan)
        nog_time = float(nog.get("total_time") or math.nan)
        gil_time = float(gil.get("total_time") or math.nan)
        row = {
            "strategy": strat,
            "nogil_throughput": nog_thr,
            "gil_throughput": gil_thr,
            "throughput_speedup": (nog_thr / gil_thr) if (gil_thr and not math.isnan(gil_thr)) else math.nan,
            "pct_improvement": ((nog_thr - gil_thr) / gil_thr * 100) if (gil_thr and not math.isnan(gil_thr)) else math.nan,
            "nogil_time": nog_time,
            "gil_time": gil_time,
            "nogil_items": int(nog.get("items_processed") or 0),
            "gil_items": int(gil.get("items_processed") or 0),
            "nogil_avg_latency": float(nog.get("avg_latency") or math.nan),
            "gil_avg_latency": float(gil.get("avg_latency") or math.nan),
        }
        rows.append(row)
    return rows

def write_csv(rows: List[Dict[str,Any]], outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with outpath.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def print_table(rows: List[Dict[str,Any]]):
    rows_sorted = sorted(rows, key=lambda r: (-(r["nogil_throughput"] if not math.isnan(r["nogil_throughput"]) else -1)))
    print()
    print(f"{'Strategy':<20} {'nogil_thr':>10} {'gil_thr':>10} {'speedup':>9} {'%impr':>8} {'nogil_t(s)':>10} {'gil_t(s)':>10}")
    print("-"*80)
    for r in rows_sorted:
        def f(v):
            return "-" if (v is None or (isinstance(v,float) and math.isnan(v))) else (f"{v:.1f}" if isinstance(v,float) else str(v))
        print(f"{r['strategy']:<20} {f(r['nogil_throughput']):>10} {f(r['gil_throughput']):>10} {f(r['throughput_speedup']):>9} {f(r['pct_improvement']):>8} {f(r['nogil_time']):>10} {f(r['gil_time']):>10}")

def plot(rows: List[Dict[str,Any]], outpath: Path):
    strategies = [r["strategy"] for r in rows]
    nog = [0.0 if math.isnan(r["nogil_throughput"]) else r["nogil_throughput"] for r in rows]
    gil = [0.0 if math.isnan(r["gil_throughput"]) else r["gil_throughput"] for r in rows]
    x = range(len(strategies))
    width = 0.35
    plt.figure(figsize=(max(6, len(strategies)*0.6), 4))
    plt.bar([i - width/2 for i in x], nog, width, label="NO-GIL")
    plt.bar([i + width/2 for i in x], gil, width, label="GIL")
    plt.xticks(list(x), strategies, rotation=45, ha="right")
    plt.ylabel("Throughput (items/sec)")
    plt.title("Strategy Throughput: NO-GIL vs GIL")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close()
    print(f"Saved plot to {outpath}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("results/compare_all.csv"))
    p.add_argument("--plot", type=Path, default=Path("results/compare_all.png"))
    p.add_argument("--strategies", type=str, default="static workstealing central sharded dynamic throttled dropoldest creditbased adaptive pushpull")
    args = p.parse_args()
    strategies = args.strategies.split()
    rows = gather(args.results_dir, strategies)
    write_csv(rows, args.out)
    print_table(rows)
    try:
        plot(rows, args.plot)
    except Exception as e:
        print(f"Plot failed: {e}")

if __name__ == "__main__":
    main()
