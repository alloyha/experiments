#!/usr/bin/env python3
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DBT_ROOT = ROOT / "tse_dbt"
MODELS = DBT_ROOT / "models"
TESTS = DBT_ROOT / "tests"
SEEDS = DBT_ROOT / "seeds"

LAYERS = {"bronze", "silver", "physical", "gold", "semantic"}
ALLOWED = {
    "bronze": {"bronze"},
    "silver": {"bronze", "silver"},
    "physical": {"bronze", "silver", "physical"},
    "gold": {"bronze", "silver", "physical", "gold"},
    "semantic": {"gold", "semantic"},
}

REF_RE = re.compile(r'''ref\(\s*['\"]([^'\"]+)['\"]\s*\)''')
PHYSICAL_PUBLISH_RE = re.compile(r'''['\"]physical_publish['\"]\s*:\s*['\"]([^'\"]+)['\"]''')
PHYSICAL_SOURCE_RE = re.compile(r'''['\"]physical_source['\"]\s*:\s*['\"]([^'\"]+)['\"]''')
ARCH_STATUS_RE = re.compile(r'''['\"]architecture_status['\"]\s*:\s*['\"]([^'\"]+)['\"]''')
ARCH_REASON_RE = re.compile(r'''['\"]architecture_reason['\"]\s*:\s*['\"]([^'\"]+)['\"]''')


def relative(path: Path) -> Path:
    return path.relative_to(ROOT)


def extract(pattern, text):
    m = pattern.search(text)
    return m.group(1) if m else None


def discover_models():
    models = {}
    for path in sorted(MODELS.rglob("*.sql")):
        rel = path.relative_to(MODELS)
        if not rel.parts:
            raise SystemExit(f"invalid dbt model path: {rel}")
        layer = rel.parts[0]
        if layer not in LAYERS:
            raise SystemExit(f"unclassified dbt model path: {rel}")
        name = path.stem
        if name in models:
            previous = models[name]
            raise SystemExit(
                "duplicate model name: "
                f"{name}: {relative(previous['path'])} [{previous['layer']}] and "
                f"{relative(path)} [{layer}]"
            )
        text = path.read_text(encoding="utf-8")
        models[name] = {
            "name": name,
            "layer": layer,
            "path": path,
            "text": text,
            "refs": REF_RE.findall(text),
            "physical_publish": extract(PHYSICAL_PUBLISH_RE, text),
            "physical_source": extract(PHYSICAL_SOURCE_RE, text),
            "architecture_status": extract(ARCH_STATUS_RE, text),
            "architecture_reason": extract(ARCH_REASON_RE, text),
        }
    return models


def discover_seeds():
    seeds = {}
    if not SEEDS.exists():
        return seeds
    for path in sorted(SEEDS.rglob("*.csv")):
        if path.stem in seeds:
            raise SystemExit(f"duplicate seed name: {path.stem}")
        seeds[path.stem] = path
    return seeds


parser = argparse.ArgumentParser(description="Validate and inspect the TSE dbt DAG")
parser.add_argument("--format", choices=("summary", "detail", "mermaid"),
                    default="summary", help="Validated output representation")
args = parser.parse_args()

models = discover_models()
seeds = discover_seeds()
known_nodes = set(models) | set(seeds)
failures = []
edges = []
outgoing = Counter()

for name, model in sorted(models.items()):
    for dep in model["refs"]:
        if dep not in known_nodes:
            failures.append(
                f"{relative(model['path'])} [{model['layer']}] has unresolved ref('{dep}')"
            )
            continue
        if dep in seeds:
            continue
        dep_model = models[dep]
        edges.append((dep, name))
        outgoing[dep] += 1
        if dep_model["layer"] not in ALLOWED[model["layer"]]:
            failures.append(
                f"{relative(model['path'])} [{model['layer']}] -> "
                f"{relative(dep_model['path'])} [{dep_model['layer']}] via ref('{dep}')"
            )

if TESTS.exists():
    for path in sorted(TESTS.rglob("*.sql")):
        text = path.read_text(encoding="utf-8")
        for dep in REF_RE.findall(text):
            if dep not in known_nodes:
                failures.append(f"{relative(path)} [test] has unresolved ref('{dep}')")

publishers = {}
consumers = defaultdict(list)
for name, model in sorted(models.items()):
    publish = model["physical_publish"]
    source = model["physical_source"]
    if publish:
        if model["layer"] != "physical":
            failures.append(
                f"{relative(model['path'])} [{model['layer']}] declares "
                f"physical_publish='{publish}' outside physical layer"
            )
        if publish in publishers:
            failures.append(
                f"duplicate physical publisher '{publish}': {publishers[publish]} and {name}"
            )
        publishers[publish] = name
    if source:
        if model["layer"] != "gold":
            failures.append(
                f"{relative(model['path'])} [{model['layer']}] declares "
                f"physical_source='{source}' outside gold layer"
            )
        consumers[source].append(name)

for physical_name, publisher in sorted(publishers.items()):
    if not consumers.get(physical_name):
        failures.append(
            f"physical publish '{physical_name}' from {publisher} has no consumer"
        )
for physical_name, names in sorted(consumers.items()):
    if physical_name not in publishers:
        failures.append(
            f"physical source '{physical_name}' consumed by {', '.join(sorted(names))} has no publisher"
        )

declared_orphans = []
for name, model in sorted(models.items()):
    is_leaf = outgoing[name] == 0
    status = model["architecture_status"]
    if status == "orphan":
        declared_orphans.append(name)
        if not is_leaf or model["physical_publish"]:
            failures.append(
                f"{relative(model['path'])} [{model['layer']}] is marked orphan "
                "but has outgoing logical or physical lineage"
            )
        if not model["architecture_reason"]:
            failures.append(
                f"{relative(model['path'])} [{model['layer']}] is marked orphan without architecture_reason"
            )
        continue
    if model["layer"] in {"bronze", "silver"} and is_leaf:
        failures.append(
            f"{relative(model['path'])} [{model['layer']}] is an undeclared leaf/orphan"
        )

if failures:
    print("Medallion DAG contract: FAIL")
    for failure in failures:
        print("  ", failure)
    raise SystemExit(1)

counts = Counter(model["layer"] for model in models.values())


def emit_summary():
    # Preserve the existing CI output exactly.
    print("Medallion DAG contract: PASS")
    print("models:", ", ".join(f"{layer}={counts[layer]}" for layer in sorted(LAYERS)))
    print("resolved model edges:", len(edges))
    print("physical lineages:", len(publishers))
    print("declared orphans:", len(declared_orphans))
    print("seeds:", len(seeds))


def emit_details():
    # All data comes from the same validated inventory as the CI gate.
    emit_summary()
    print("\n=== MODELS ===")
    for layer in sorted(LAYERS):
        print(f"[{layer.upper()}]")
        for name, model in sorted(models.items()):
            if model["layer"] == layer:
                print(f"  {name}: {relative(model['path'])}")

    print("\n=== RESOLVED EDGES ===")
    for source, target in sorted(edges):
        print(f"  {models[source]['layer']}:{source} -> {models[target]['layer']}:{target}")
    for name, model in sorted(models.items()):
        for dependency in model["refs"]:
            if dependency in seeds:
                print(f"  seed:{dependency} -> {model['layer']}:{name}")

    print("\n=== PHYSICAL LINEAGE ===")
    for artifact in sorted(set(publishers) | set(consumers)):
        print(f"  {artifact}: publisher={publishers.get(artifact, '<missing>')}")
        for name in sorted(consumers.get(artifact, [])):
            print(f"    -> {name}")

    print("\n=== DECLARED ORPHANS ===")
    for name in sorted(declared_orphans):
        model = models[name]
        print(f"  {model['layer']}:{name}: {model['architecture_reason']}")

    print("\n=== ROOT MODELS ===")
    all_references = {name for model in models.values() for name in model["refs"]}
    for name, model in sorted(models.items()):
        if not model["refs"] and not model["physical_source"]:
            print(f"  {model['layer']}:{name}")

    print("\n=== LEAF MODELS ===")
    for name, model in sorted(models.items()):
        if name not in all_references and not model["physical_publish"]:
            print(f"  {model['layer']}:{name}")


def emit_mermaid():
    # A valid Mermaid graph without a separate unvalidated parser.
    print("flowchart LR")
    for layer in sorted(LAYERS):
        print(f"  subgraph {layer.upper()}[{layer.capitalize()}]")
        for name, model in sorted(models.items()):
            if model["layer"] == layer:
                print(f'    model_{name}["{name}"]')
        print("  end")
    for name in sorted(seeds):
        print(f'  seed_{name}["seed: {name}"]')
    for source, target in sorted(edges):
        print(f"  model_{source} --> model_{target}")
    for name, model in sorted(models.items()):
        for dep in model["refs"]:
            if dep in seeds:
                print(f"  seed_{dep} --> model_{name}")
    for artifact in sorted(set(publishers) | set(consumers)):
        node = "physical_" + re.sub(r"[^A-Za-z0-9_]", "_", artifact)
        print(f'  {node}[("{artifact}")]')
        if artifact in publishers:
            print(f"  model_{publishers[artifact]} -. publish .-> {node}")
        for name in sorted(consumers.get(artifact, [])):
            print(f"  {node} -. read .-> model_{name}")


if args.format == "summary":
    emit_summary()
elif args.format == "detail":
    emit_details()
else:
    emit_mermaid()
