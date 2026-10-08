#!/usr/bin/env python3
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DBT_ROOT = ROOT / "tse_dbt"
MODELS = DBT_ROOT / "models"
SEEDS = DBT_ROOT / "seeds"
LAYERS = ("bronze", "silver", "physical", "gold", "semantic")

REF_RE = re.compile(r'''ref\(\s*['\"]([^'\"]+)['\"]\s*\)''')
PHYSICAL_PUBLISH_RE = re.compile(r'''['\"]physical_publish['\"]\s*:\s*['\"]([^'\"]+)['\"]''')
PHYSICAL_SOURCE_RE = re.compile(r'''['\"]physical_source['\"]\s*:\s*['\"]([^'\"]+)['\"]''')
ARCH_STATUS_RE = re.compile(r'''['\"]architecture_status['\"]\s*:\s*['\"]([^'\"]+)['\"]''')
ARCH_REASON_RE = re.compile(r'''['\"]architecture_reason['\"]\s*:\s*['\"]([^'\"]+)['\"]''')


def extract(pattern, text):
    m = pattern.search(text)
    return m.group(1) if m else None


def rel(path):
    return str(path.relative_to(ROOT))

models = {}
for path in sorted(MODELS.rglob("*.sql")):
    rp = path.relative_to(MODELS)
    if not rp.parts or rp.parts[0] not in LAYERS:
        continue
    text = path.read_text(encoding="utf-8")
    models[path.stem] = {
        "name": path.stem,
        "layer": rp.parts[0],
        "path": path,
        "refs": REF_RE.findall(text),
        "physical_publish": extract(PHYSICAL_PUBLISH_RE, text),
        "physical_source": extract(PHYSICAL_SOURCE_RE, text),
        "architecture_status": extract(ARCH_STATUS_RE, text),
        "architecture_reason": extract(ARCH_REASON_RE, text),
    }

seeds = {p.stem: p for p in sorted(SEEDS.rglob("*.csv"))} if SEEDS.exists() else {}
edges, unresolved = [], []
for name, model in sorted(models.items()):
    for dep in model["refs"]:
        if dep in models:
            edges.append((models[dep]["layer"], dep, model["layer"], name))
        elif dep in seeds:
            edges.append(("seed", dep, model["layer"], name))
        else:
            unresolved.append((model["layer"], name, dep))

publishers = {}
consumers = defaultdict(list)
orphans = []
for name, model in sorted(models.items()):
    if model["physical_publish"]:
        publishers[model["physical_publish"]] = name
    if model["physical_source"]:
        consumers[model["physical_source"]].append(name)
    if model["architecture_status"] == "orphan":
        orphans.append(model)

print("=== MEDALLION DAG STATE ===\n")
counts = Counter(m["layer"] for m in models.values())
print("Models by layer:")
for layer in LAYERS:
    print(f"  {layer:10s} {counts[layer]:3d}")
print(f"  {'TOTAL':10s} {len(models):3d}")
print(f"  {'seeds':10s} {len(seeds):3d}")

grouped = defaultdict(list)
for model in models.values():
    grouped[model["layer"]].append(model)

print("\n=== MODELS ===")
for layer in LAYERS:
    print(f"\n[{layer.upper()}]")
    for model in sorted(grouped[layer], key=lambda x: x["name"]):
        print(f"  {model['name']:<40s} {rel(model['path'])}")

print("\n=== RESOLVED EDGES ===")
for a,b,c,d in sorted(edges):
    print(f"  [{a:8s}] {b} -> [{c:8s}] {d}")
print(f"\nResolved edges: {len(edges)}")

print("\n=== PHYSICAL LINEAGE ===")
if not publishers and not consumers:
    print("  none")
for artifact in sorted(set(publishers) | set(consumers)):
    publisher = publishers.get(artifact, "<missing publisher>")
    print(f"  [physical] {publisher}")
    print(f"      publishes {artifact}")
    for consumer in sorted(consumers.get(artifact, [])):
        print(f"          -> [{models[consumer]['layer']}] {consumer}")
    if not consumers.get(artifact):
        print("          -> <missing consumer>")

print("\n=== DECLARED ORPHANS ===")
if not orphans:
    print("  none")
for model in orphans:
    print(f"  [{model['layer']}] {model['name']}")
    print(f"      {model['architecture_reason'] or '<no reason>'}")

print("\n=== EDGE COUNTS BY LAYER ===")
ec = Counter((a,c) for a,_,c,_ in edges)
for (a,c), n in sorted(ec.items()):
    print(f"  {a:10s} -> {c:10s}: {n}")

incoming = Counter(d for _,_,_,d in edges)
print("\n=== ROOT MODELS ===")
for name, model in sorted(models.items()):
    if incoming[name] == 0 and not model["physical_source"]:
        print(f"  [{model['layer']:8s}] {name}")

outgoing = Counter(b for _,b,_,_ in edges)
print("\n=== LEAF MODELS ===")
for name, model in sorted(models.items()):
    if outgoing[name] == 0 and not model["physical_publish"]:
        suffix = " (declared orphan)" if model["architecture_status"] == "orphan" else ""
        print(f"  [{model['layer']:8s}] {name}{suffix}")

print("\n=== UNRESOLVED REFS ===")
if unresolved:
    for layer, model, dep in sorted(unresolved):
        print(f"  [{layer:8s}] {model} -> ref('{dep}')")
else:
    print("  none")

print("\n=== MERMAID ===\n")
print("flowchart LR")
for layer in LAYERS:
    print(f"  subgraph {layer.upper()}[{layer.capitalize()}]")
    for model in sorted(grouped[layer], key=lambda x: x["name"]):
        print(f'    {model["name"]}["{model["name"]}"]')
    print("  end")
for a,b,c,d in sorted(edges):
    if a != "seed":
        print(f"  {b} --> {d}")
for artifact in sorted(set(publishers) | set(consumers)):
    node = "physical_artifact_" + re.sub(r"[^A-Za-z0-9_]", "_", artifact)
    print(f'  {node}[("{artifact}")]')
    if artifact in publishers:
        print(f"  {publishers[artifact]} -. publish .-> {node}")
    for consumer in sorted(consumers.get(artifact, [])):
        print(f"  {node} -. read .-> {consumer}")
