# Historical TSE research and forensic tools

These read-only research tools are preserved for reproducibility of the
2026 candidate vote-conservation and distinct-zone ablation studies.
They are **not** supported production graph certification commands.

- `benchmark_graph_candidate_conservation.py` deliberately accepts the
  **pre-promotion** conservation assertion; the active signed-multiset
  canonical SQL no longer matches its checked tail. Do not weaken that
  fail-closed behavior or interpret historical times as current speedups.
- `benchmark_graph_zone_count_ablation.py` is an isolated physical
  source benchmark, not a certification or deployable correctness proof.

Both tools preserve create-only/read-only evidence semantics and must
never be inserted into the ingestion, dbt or production gate path.
From the TSE project root, run `python research/<script>.py --help`
to inspect their options. Historical docs live under `docs/analytics/`.

## One-off historical source forensics

`diagnose_history_source.py` is a **read-only investigation** of one specific
2026 candidatura-history source resource. It inspects source identities,
ingestion journal/state and filesystem candidates but performs no rehydration,
publication, download, or authoritative-state mutation. It is not an ingestion
entrypoint or general recovery tool.

From `data/tse-election-data`:

```bash
.venv/bin/python research/diagnose_history_source.py \
  --raw-root data/tse --output json
```

Its six fixture-based tests remain in `tests/test_history_source_diagnosis.py`.
