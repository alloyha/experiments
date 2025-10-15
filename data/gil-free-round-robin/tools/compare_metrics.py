#!/usr/bin/env python3
"""
tools/compare_metrics.py - nicer comparison display + optional plot

Usage:
    python tools/compare_metrics.py results/nogil.json results/gil.json --out results/compare.csv [--plot results/compare.png]

If one of the JSON files is missing/empty, the script will attempt to parse the
corresponding run log (results_nogil.txt / results_gil.txt) and extract the
table row using a tolerant regex.

Output:
 - pretty terminal table with colors
 - ASCII throughput bars
 - CSV file (default: results/compare.csv)
 - optional PNG plot when --plot <file> provided and matplotlib is installed
"""
from __future__ import annotations
import sys, json, csv, math, re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import glob

# ANSI colors
CSI = "\x1b["
RESET = CSI + "0m"
BOLD = CSI + "1m"
GREEN = CSI + "32m"
YELLOW = CSI + "33m"
RED = CSI + "31m"
CYAN = CSI + "36m"

# regex to parse the human table line from run logs:
# Example format (columns separated by spaces):
# Work Stealing             14.095     709.5           283.929            241.027
ROW_RE = re.compile(r"^\s*(?P<strategy>[A-Za-z0-9 _-]{1,40}?)\s+(?P<time>[0-9]*\.?[0-9]+)\s+(?P<throughput>[0-9]*\.?[0-9]+)", re.IGNORECASE)

def find_candidate_logs(json_path: Path) -> List[Path]:
    """Return likely run log files associated with a JSON path."""
    name = json_path.name
    base = name.replace(".json", "")
    candidates = []
    patterns = [
        f"results_{base}.txt",
        f"results/{base}.txt",
        f"results/{base}_*.txt",
        f"{base}.txt",
        f"results/{base}.log",
        f"results/*{base}*.txt",
    ]
    for p in patterns:
        for m in glob.glob(p):
            candidates.append(Path(m))
    # also include any results/*.txt
    for p in Path("results").glob("*.txt"):
        candidates.append(p)
    # dedupe
    seen = set()
    out = []
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out

def try_parse_log_for_results(log_path: Path) -> Optional[Dict[str, Dict[str, float]]]:
    """Try to extract strategy rows (strategy -> {total_time, throughput}) from a textual log file."""
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    results = {}
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if m:
            strategy = m.group("strategy").strip()
            time_v = float(m.group("time"))
            thr_v = float(m.group("throughput"))
            results[strategy] = {"total_time": time_v, "throughput": thr_v}
    return results if results else None

def load_json_or_fallback(json_path: Path) -> Dict[str, Any]:
    """Load JSON; if missing/empty/invalid, try to parse run logs to build a minimal summary."""
    if not json_path.exists():
        print(f"NOTE: JSON file {json_path} missing — will attempt to parse run logs.")
    else:
        txt = json_path.read_text()
        if txt.strip():
            try:
                j = json.loads(txt)
                # minimal validation
                if "results" in j and isinstance(j["results"], list):
                    return j
                # fallthrough to try logs
            except Exception as e:
                print(f"NOTE: Failed to parse JSON {json_path}: {e} — will attempt to parse run logs.")
    # fallback: try candidate logs
    logs = find_candidate_logs(json_path)
    parsed_any = {}
    for p in logs:
        parsed = try_parse_log_for_results(p)
        if parsed:
            # Merge results; parsed contains strategy -> time/throughput
            for strat, vals in parsed.items():
                parsed_any[strat] = {
                    "strategy": strat,
                    "total_time": vals["total_time"],
                    "throughput": vals["throughput"],
                    "avg_latency": None,
                    "std_latency": None,
                    "items_processed": None,
                    "num_consumers": None
                }
            # break on first successful parse (we'll show candidate logs elsewhere)
            break
    if parsed_any:
        return {"results": list(parsed_any.values())}
    # if all failed, print helpful diagnostics and raise
    print(f"ERROR: Could not load JSON or parse run logs for {json_path}")
    # show candidate logs if any
    if logs:
        print("Candidate logs found; showing last 4000 chars of each to help debugging:")
        for p in logs:
            try:
                print(f"\n--- {p} ({p.stat().st_size} bytes) ---")
                print(p.read_text(errors="ignore")[-4000:])
            except Exception as ex:
                print(f"(couldn't read {p}: {ex})")
    raise FileNotFoundError(f"No usable data for {json_path}")

def build_index(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for r in results:
        name = r.get("strategy") or r.get("strategy_name") or "unknown"
        idx[name] = r
    return idx

def safe(v, default=float("nan")):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def compute_rows(nogil_idx, gil_idx) -> List[Dict[str, Any]]:
    names = sorted(set(list(nogil_idx.keys()) + list(gil_idx.keys())))
    rows = []
    for name in names:
        a = nogil_idx.get(name, {})
        b = gil_idx.get(name, {})
        row = {"strategy": name}
        row["nogil_total_time"] = safe(a.get("total_time"))
        row["nogil_throughput"] = safe(a.get("throughput"))
        row["nogil_avg_latency"] = safe(a.get("avg_latency"))
        row["nogil_std_latency"] = safe(a.get("std_latency"))
        row["nogil_items"] = int(a.get("items_processed", 0) or 0)

        row["gil_total_time"] = safe(b.get("total_time"))
        row["gil_throughput"] = safe(b.get("throughput"))
        row["gil_avg_latency"] = safe(b.get("avg_latency"))
        row["gil_std_latency"] = safe(b.get("std_latency"))
        row["gil_items"] = int(b.get("items_processed", 0) or 0)

        # ratios: throughput_speedup = nogil_throughput / gil_throughput
        try:
            row["throughput_speedup"] = (row["nogil_throughput"] / row["gil_throughput"]) if row["gil_throughput"] and not math.isnan(row["gil_throughput"]) else math.nan
        except Exception:
            row["throughput_speedup"] = math.nan
        try:
            row["time_ratio"] = (row["gil_total_time"] / row["nogil_total_time"]) if row["nogil_total_time"] and not math.isnan(row["nogil_total_time"]) else math.nan
        except Exception:
            row["time_ratio"] = math.nan
        # percent improvement: (nogil - gil)/gil * 100  (positive means nogil better)
        try:
            row["pct_improvement"] = ((row["nogil_throughput"] - row["gil_throughput"]) / row["gil_throughput"]) * 100 if row["gil_throughput"] and not math.isnan(row["gil_throughput"]) else math.nan
        except Exception:
            row["pct_improvement"] = math.nan

        rows.append(row)
    return rows

def color_for_speedup(speedup: float, neutral_tol: float = 0.03) -> str:
    """Return color code for a speedup (1.0 = tie). neutral_tol = 3% tolerance."""
    if speedup is None or math.isnan(speedup):
        return RESET
    if abs(speedup - 1.0) <= neutral_tol:
        return YELLOW  # near tie
    if speedup > 1.0:
        return GREEN   # nogil faster
    return RED        # nogil slower

def print_comparison(rows: List[Dict[str, Any]]):
    # header
    hdr = f"{BOLD}{'Strategy':<28} {'nogil_thr':>12} {'gil_thr':>10} {'speedup':>9} {'%impr':>8} {'nogil_t(s)':>12} {'gil_t(s)':>10} {'items':>8}{RESET}"
    print(hdr)
    print("-" * 95)
    # compute max throughput for bar chart
    max_thr = max((r["nogil_throughput"] if not math.isnan(r["nogil_throughput"]) else 0,
                   r["gil_throughput"] if not math.isnan(r["gil_throughput"]) else 0)
                  for r in rows)
    max_thr_val = max_thr[0] if isinstance(max_thr, tuple) else max_thr  # safe
    # Actually compute a single numeric maximum
    numeric_max = 0.0
    for r in rows:
        numeric_max = max(numeric_max,
                          0.0 if math.isnan(r["nogil_throughput"]) else r["nogil_throughput"],
                          0.0 if math.isnan(r["gil_throughput"]) else r["gil_throughput"])
    for r in rows:
        sp = r.get("throughput_speedup", math.nan)
        col = color_for_speedup(sp)
        nog = "-" if math.isnan(r["nogil_throughput"]) else f"{r['nogil_throughput']:.1f}"
        gil = "-" if math.isnan(r["gil_throughput"]) else f"{r['gil_throughput']:.1f}"
        spf = "-" if math.isnan(sp) else f"{sp:.2f}x"
        pct = "-" if math.isnan(r.get("pct_improvement", math.nan)) else f"{r['pct_improvement']:.1f}%"
        nt = "-" if math.isnan(r["nogil_total_time"]) else f"{r['nogil_total_time']:.3f}"
        gt = "-" if math.isnan(r["gil_total_time"]) else f"{r['gil_total_time']:.3f}"
        items = f"{r.get('nogil_items',0)}/{r.get('gil_items',0)}"
        print(f"{col}{r['strategy']:<28} {nog:>12} {gil:>10} {spf:>9} {pct:>8} {nt:>12} {gt:>10} {items:>8}{RESET}")
    print("\nThroughput comparison (normalized bars):")
    # ASCII bar chart showing nogil and gil bars
    scale_width = 40
    for r in rows:
        nogval = 0.0 if math.isnan(r["nogil_throughput"]) else r["nogil_throughput"]
        gilval = 0.0 if math.isnan(r["gil_throughput"]) else r["gil_throughput"]
        scale = numeric_max or 1.0
        nog_len = int((nogval / scale) * scale_width)
        gil_len = int((gilval / scale) * scale_width)
        print(f"{r['strategy']:<28} NO-GIL |{GREEN}{'█'*nog_len}{RESET}{' '*(scale_width-nog_len)}| {nogval:.1f}")
        print(f"{'':28} GIL    |{CYAN}{'█'*gil_len}{RESET}{' '*(scale_width-gil_len)}| {gilval:.1f}")
        print("-" * 95)

def write_csv(rows: List[Dict[str, Any]], outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with outpath.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def try_make_plot(rows: List[Dict[str, Any]], plot_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available — skipping PNG plot. Install matplotlib to enable plotting.")
        return
    strategies = [r["strategy"] for r in rows]
    nog = [0.0 if math.isnan(r["nogil_throughput"]) else r["nogil_throughput"] for r in rows]
    gil = [0.0 if math.isnan(r["gil_throughput"]) else r["gil_throughput"] for r in rows]
    x = range(len(strategies))
    width = 0.35
    plt.figure(figsize=(max(6, len(strategies)*1.2), 4))
    plt.bar([i - width/2 for i in x], nog, width, label="NO-GIL")
    plt.bar([i + width/2 for i in x], gil, width, label="GIL")
    plt.xticks(list(x), strategies, rotation=30, ha="right")
    plt.ylabel("Throughput (items/sec)")
    plt.title("NO-GIL vs GIL Throughput Comparison")
    plt.legend()
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

def main(argv: Optional[List[str]] = None):
    import argparse
    p = argparse.ArgumentParser(description="Compare benchmark JSON outputs (nogil vs gil) with nicer display")
    p.add_argument("nogil_json", type=Path, help="NO-GIL JSON file (or will fallback to logs)")
    p.add_argument("gil_json", type=Path, help="GIL JSON file (or will fallback to logs)")
    p.add_argument("--out", type=Path, default=Path("results/compare.csv"), help="CSV output path")
    p.add_argument("--plot", type=Path, default=None, help="Optional PNG output path (requires matplotlib)")
    args = p.parse_args(argv)

    nogil_summary = load_json_or_fallback(args.nogil_json)
    gil_summary = load_json_or_fallback(args.gil_json)

    nogil_idx = build_index(nogil_summary["results"])
    gil_idx = build_index(gil_summary["results"])
    rows = compute_rows(nogil_idx, gil_idx)
    print()
    print(f"{BOLD}Comparison results:{RESET}")
    print_comparison(rows)
    write_csv(rows, args.out)
    print(f"\nCSV written to: {args.out}")
    if args.plot:
        try_make_plot(rows, args.plot)

if __name__ == "__main__":
    main()
