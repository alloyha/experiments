#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

path = Path("data/tse/_metadata/current_objects.jsonl")
keys = []

for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue

    row = json.loads(line)

    if row.get("year") == 2018 and row.get("election_type") == "general":
        keys.append(
            (
                row.get("object"),
                row.get("year"),
                row.get("election_type"),
                row.get("election_scope"),
            )
        )

counts = Counter(keys)
dupes = [(key, count) for key, count in counts.items() if count > 1]

print(f"active records: {len(keys)}")
print(f"unique active objects: {len(counts)}")
print(f"duplicate active object keys: {len(dupes)}")

for (obj, year, election_type, election_scope), count in sorted(
    dupes,
    key=lambda item: (-item[1], str(item[0][0])),
)[:50]:
    print(count, year, election_type, election_scope, obj)
