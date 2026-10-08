#!/usr/bin/env python3
from pathlib import Path
import csv, json

root = Path("data/tse")
index = root / "_metadata" / "current_objects.jsonl"

targets = (
    "Votação em partido por município e zona",
    "Detalhe da apuração por município e zona",
)

rows = []
for line in index.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    name = obj.get("resource_name","")
    if any(t.lower() in name.lower() for t in targets) and obj.get("year") == 2018:
        rows.append(obj)

for obj in rows:
    print(f"\n=== {obj.get('resource_name')} ===")
    # current_objects schemas evolved; find an extracted path robustly.
    candidates = []
    for key in ("extracted_path","selected_path","path"):
        value = obj.get(key)
        if value:
            candidates.append(Path(value))
    resource_id = obj.get("resource_id")
    if resource_id:
        candidates.extend(root.glob(f"raw/election_type=*/year=2018/domain=*/dataset=*/resource={resource_id}/sha256=*/extracted/*"))

    path = next((p for p in candidates if p.exists() and p.is_file()), None)
    if not path:
        print("Could not resolve extracted file path")
        continue

    print(path)
    with path.open("r", encoding="latin-1", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
    for col in header:
        print(col)
