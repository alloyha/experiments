# PR44 — Gold incremental / Semantic persisted recovery certification

## Objective

Prove equivalence of **persisted** downstream results for one 2026 election slice,
not merely identical recovered ZIP/CSV bytes. The actual dependency graph is:

`candidate` + `candidate_assets` → `bronze_candidates` + `bronze_candidate_assets` →
`silver_candidate_assets` → **`dim_candidate` (Gold, incremental partition replacement)** →
**`candidate_summary` (Semantic, view)**.

`candidate_summary` is normally a view. This gate materializes an explicit,
**scratch-only persistent table** (`pr44_candidate_summary_persisted`) to capture
its rows. It does **not** claim that the production semantic model is a table.

## Controlled arms

1. Validate **two** published current resource versions, source SHA-256, selected
   ZIP members, extracted files and exact index references. Legacy rows without
   `materialization_state` are permitted **only** with the PR43 opt-in and full
   byte+manifest+index checks. Explicit `null`/`evicted` states fail closed.
2. Independent-byte-copy both resources into one disposable raw lake with a
   minimal, correct state/index control plane. Protect the canonical sources by
   before/after SHA-256 checks.
3. Run `dbt run --select +candidate_summary` using an isolated DuckDB database,
   fresh dbt artifacts and Python socket guard. Require a non-empty Gold relation,
   a non-empty Semantic output and at least one candidate with declared assets.
   Persist Semantic into `pr44_candidate_summary_persisted`.
4. **Close DuckDB connections** and copy the same baseline warehouse into a
   *control* database and a *recovery* database. This keeps the same starting
   persisted state, not independently initialized databases.
5. In **both copies**, insert synthetic probe rows into the Gold table:
   - selected election year: a deliberately stale row that must be removed by
     the official `partition_replace_pre_hook()` on incremental execution;
   - other election year: an untouched row whose entire typed-row SHA-256 must
     remain identical.
6. Run the incremental dbt graph in the control database, with no eviction.
7. Run the official `prune_extracted_raw.py --apply --domain candidate_assets`
   **on scratch only**; verify only the assets index/state were evicted, leaving
   the candidate source and index intact. Run the official
   `ensure_materialized.py` offline and prove the exact two-resource index and
   all bytes are restored.
8. Run the same incremental dbt graph on the recovered warehouse copy.
9. Persist Semantic snapshots in both copies. Verify the selected probe was
   deleted, the outside row survived byte-for-byte, and Gold/Semantic schemas,
   row counts, and order-insensitive duplicate-sensitive row multiset digests
   match **baseline = uninterrupted control = recovered** for the selected year.
10. Run strict physical storage audit + raw contracts on scratch, verify
    canonical hashes and manifest identity, and destroy scratch on all exits.

**The three dbt runs are real, not mocked, when executed on the WSL lake.** The
unit tests cover fail-closed configuration, SQL scope, fingerprint consistency,
probe logic and isolation. The WSL run is the physical evidence; do not label it
certified until `result=PASS` is emitted.

## WSL usage

```bash
cd ~/github/experiments-gold-semantic/data/tse-election-data
make unit
make medallion-contracts
make dbt-deps
RAW="$HOME/github/experiments/data/tse-election-data/data/tse"

# Diagnostic only (does not hash or mutate source files)
.venv/bin/python scripts/certify_gold_semantic_roundtrip.py \
  --raw-root "$RAW" --year 2026 --inspect-sources \
  --output-json /tmp/tse-pr44-source-inventory.json

# Actual physical experiment; never call prune on $RAW
make gold-semantic-cache-roundtrip DATA_ROOT="$RAW" \
  GS_ROUNDTRIP_YEAR=2026 \
  GS_ROUNDTRIP_JSON=/tmp/tse-pr44-gold-semantic.json
python3 -m json.tool /tmp/tse-pr44-gold-semantic.json
```

If multiple 2026 `candidate` resources qualify, inspect candidate identifiers
first and set `GS_ROUNDTRIP_CANDIDATE_RESOURCE_ID`. Set
`GS_ROUNDTRIP_ASSETS_RESOURCE_ID` to explicitly select the assets source.
Defaults: maximum **128 MiB compressed, 512 MiB extracted per resource**, 500k
rows per selected relation, and at least **3 GiB free** in `/tmp`. All limits
are fail-closed and adjustable only with explicit Make variables.

The runner does **not** alter your canonical `DATA_ROOT`, original warehouse,
physical candidate votes or existing Git worktrees. `prune --apply` always sees
only the internal disposable lake; no `prune-cache APPLY=1` is used.

## Pass criteria and non-claims

A PASS certifies the selected `dim_candidate` 2026 partition and the
`candidate_summary` persistent snapshot, including Gold partition replacement,
out-of-scope row preservation, and correspondence of the two isolated arms.

It does **not** certify all Gold models, all Semantic models, candidate vote
Parquet publications, alternative years, the full warehouse, multi-resource
concurrent accesses or global eviction of ~39.2 GiB. The Python socket guard is
**not** an OS/kernel-level network sandbox. These need independent gates before
production-cache cleanup can be considered.


## Physical certification — 2026-10-13

**Result: PASS** — real 2026 candidate and candidate-assets sources.

- Unit tests: **224 PASS**
- Medallion DAG: **PASS**
- dbt baseline/control/recovered: **5 PASS each**
- Gold `dim_candidate`: **20,989 rows**
- Semantic `candidate_summary`: **251 rows**
- Candidates enriched with assets: **13,917**
- Incremental selected partition replacement: **PASS**
- Unselected partition preservation: **PASS**
- Persisted Gold/Semantic equivalence: **PASS**
- Offline recovery, source integrity and strict audit: **PASS**
- Canonical data preservation: **PASS**

Source identities and SHA-256 values are recorded in
`contracts/certification/pr44_gold_semantic_2026.json`.

This is a bounded certification of the 2026 candidate subgraph,
not approval for warehouse-wide cache eviction.
