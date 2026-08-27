import sys, pathlib
sys.path.insert(0, "/home/pingu/github/experiments/data/meta_metrics")
base = pathlib.Path("/home/pingu/github/experiments/data/meta_metrics")

import runpy, os
os.chdir(base)
runpy.run_path(str(base / "src" / "main.py"))

from src.to_duckdb import load_catalog, populate, create_views, write_mermaid
from src.cubes import ensure_entity_relations, generate_cube_cover, persist_cube_cover
import duckdb

db   = base / "data/metric_catalog.duckdb"
erd  = base / "data/metric_catalog.md"
json = base / "data/metric_catalog_v1.json"

db.unlink(missing_ok=True)
for wal in (base / "data").glob("metric_catalog.duckdb*"):
    wal.unlink(missing_ok=True)

catalog = load_catalog(json)
con = duckdb.connect(str(db))

populate(con, catalog)
create_views(con)
write_mermaid(con, erd)

n_relations = ensure_entity_relations(con)
print(f"  entity_relation seeded: {n_relations} rows")

cover = generate_cube_cover(con)
persist_cube_cover(con, cover)
summary = cover.summary()
print(f"  analytical_cube        {summary['cube_count']:5d}")
print(f"  metrics_covered        {summary['metrics_covered']:5d}")
print(f"  metrics_uncovered      {summary['metrics_uncovered']:5d}")
print(f"  cost.score             {summary['cost']['score']:.0f}")
for c in summary["cubes"]:
    print(f"    {c['cube_id']:35s} {c['metric_count']:4d} metrics ({c['type']})")

for tbl in ("entity", "entity_relation", "dimension", "dataset",
            "metric_definition", "metric_implementation",
            "impl_column", "metric_dependency", "quality_contract",
            "metric_dimension", "analytical_cube", "cube_metric"):
    n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"  {tbl:35s} {n:5d}")

con.close()
print("DONE")

