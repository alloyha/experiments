#!/usr/bin/env python3
import argparse, json, os, re
from datetime import datetime, timezone
from pathlib import Path
import duckdb

CANDIDATE_RE = re.compile(r"votacao_candidato_munzona_.*\.csv$", re.I)

def part(path, key):
    prefix = key + "="
    for p in path.parts:
        if p.startswith(prefix):
            return p[len(prefix):]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="data/tse")
    ap.add_argument("--year", type=int, action="append", dest="years")
    ap.add_argument("--election-type", action="append", dest="election_types")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--compression", default="zstd")
    a = ap.parse_args()

    root = Path(a.raw_root)
    pattern = "raw/election_type=*/year=*/domain=*/dataset=*/resource=*/sha256=*/extracted/*.csv"
    sources = []
    for csv in root.glob(pattern):
        if not CANDIDATE_RE.search(csv.name):
            continue
        year = part(csv, "year")
        et = part(csv, "election_type")
        if a.years and int(year) not in a.years:
            continue
        if a.election_types and et not in a.election_types:
            continue
        sources.append(csv)

    if not sources:
        raise SystemExit("No candidate vote CSV matched.")

    con = duckdb.connect()
    con.execute(f"set threads={a.threads}")
    con.execute("set preserve_insertion_order=false")

    for csv in sources:
        prepared_dir = csv.parent.parent / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        parquet = prepared_dir / (csv.stem + ".parquet")
        manifest = prepared_dir / (csv.stem + ".prepared.json")
        sha = part(csv, "sha256")

        if parquet.exists() and manifest.exists() and not a.force:
            try:
                meta = json.loads(manifest.read_text())
                if (meta.get("source_sha256") == sha
                    and meta.get("source_size_bytes") == csv.stat().st_size
                    and meta.get("schema_version") == 1):
                    print("SKIP", parquet)
                    continue
            except Exception:
                pass

        tmp = parquet.with_suffix(".parquet.tmp")
        tmp.unlink(missing_ok=True)
        src = str(csv.resolve()).replace("'", "''")
        dst = str(tmp.resolve()).replace("'", "''")
        compression = a.compression.upper().replace("'", "")

        print("PREPARE", csv)
        con.execute(f"""
            COPY (
                SELECT *
                FROM read_csv(
                    '{src}',
                    delim=';',
                    quote='"',
                    header=true,
                    all_varchar=true,
                    encoding='latin-1',
                    sample_size=20480
                )
            )
            TO '{dst}'
            (
                FORMAT PARQUET,
                COMPRESSION {compression},
                ROW_GROUP_SIZE 122880
            )
        """)

        rows = con.execute(f"select count(*) from read_parquet('{dst}')").fetchone()[0]
        tmp.replace(parquet)
        meta = {
            "schema_version": 1,
            "source_path": str(csv),
            "source_sha256": sha,
            "source_size_bytes": csv.stat().st_size,
            "prepared_path": str(parquet),
            "prepared_size_bytes": parquet.stat().st_size,
            "row_count": rows,
            "compression": a.compression.lower(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"DONE rows={rows:,} parquet={parquet}")

    con.close()

if __name__ == "__main__":
    main()
