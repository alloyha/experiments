#!/usr/bin/env python3

import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DBT_ROOT = ROOT / "tse_dbt"
MODELS = DBT_ROOT / "models"
TESTS = DBT_ROOT / "tests"
SEEDS = DBT_ROOT / "seeds"

LAYERS = {
    "bronze",
    "silver",
    "physical",
    "gold",
    "semantic",
}

ALLOWED = {
    "bronze": {"bronze"},
    "silver": {"bronze", "silver"},
    "physical": {"bronze", "silver", "physical"},
    "gold": {"bronze", "silver", "physical", "gold"},
    "semantic": {"gold", "semantic"},
}

# Local one-argument dbt refs:
#   ref('model_name')
#   ref("model_name")
REF_RE = re.compile(
    r"""ref\(\s*['"]([^'"]+)['"]\s*\)"""
)


def relative(path: Path) -> Path:
    return path.relative_to(ROOT)


def discover_models():
    models = {}

    for path in MODELS.rglob("*.sql"):
        rel = path.relative_to(MODELS)

        if not rel.parts:
            raise SystemExit(f"invalid dbt model path: {rel}")

        layer = rel.parts[0]

        if layer not in LAYERS:
            raise SystemExit(
                f"unclassified dbt model path: {rel}"
            )

        name = path.stem

        if name in models:
            previous_layer, previous_path = models[name]
            raise SystemExit(
                "duplicate model name: "
                f"{name}: "
                f"{relative(previous_path)} [{previous_layer}] and "
                f"{relative(path)} [{layer}]"
            )

        models[name] = (layer, path)

    return models


def discover_seeds():
    if not SEEDS.exists():
        return {}

    seeds = {}

    for path in SEEDS.rglob("*.csv"):
        name = path.stem

        if name in seeds:
            raise SystemExit(
                f"duplicate seed name: {name}"
            )

        seeds[name] = path

    return seeds


def refs_in(path: Path):
    return REF_RE.findall(
        path.read_text(encoding="utf-8")
    )


models = discover_models()
seeds = discover_seeds()

known_nodes = set(models) | set(seeds)

failures = []
edges = 0


# Model DAG:
# validate both existence and medallion direction.
for name, (layer, path) in sorted(models.items()):
    for dep in refs_in(path):
        if dep not in known_nodes:
            failures.append(
                f"{relative(path)} [{layer}] "
                f"has unresolved ref('{dep}')"
            )
            continue

        # Seeds are legitimate dbt nodes but have no medallion layer.
        if dep in seeds:
            continue

        dep_layer, dep_path = models[dep]
        edges += 1

        if dep_layer not in ALLOWED[layer]:
            failures.append(
                f"{relative(path)} [{layer}] -> "
                f"{relative(dep_path)} [{dep_layer}] "
                f"via ref('{dep}')"
            )


# Singular tests are not medallion nodes themselves, but every local
# ref they declare must still resolve.
if TESTS.exists():
    for path in sorted(TESTS.rglob("*.sql")):
        for dep in refs_in(path):
            if dep not in known_nodes:
                failures.append(
                    f"{relative(path)} [test] "
                    f"has unresolved ref('{dep}')"
                )


if failures:
    print("Medallion DAG contract: FAIL")

    for failure in failures:
        print("  ", failure)

    raise SystemExit(1)


counts = Counter(
    layer
    for layer, _ in models.values()
)

print("Medallion DAG contract: PASS")
print(
    "models:",
    ", ".join(
        f"{layer}={counts[layer]}"
        for layer in sorted(LAYERS)
    ),
)
print("resolved model edges:", edges)
print("seeds:", len(seeds))

