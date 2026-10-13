# PR45 — Physical Parquet publication and downstream recovery gate

## Scientific scope

PR44 certified real TSE `candidate` + `candidate_assets` for 2026 with a persisted Gold/Semantic slice. PR45 addresses a **different physical boundary**: the external Parquet generations consumed by `fact_candidate_votes`, and a persisted downstream `candidate_ranking` built by the **real dbt models**.

This first gate is a **small synthetic, real-publisher integration experiment**. It runs `write_candidate_fact_partition.publish_partition()` and dbt with a disposable DuckDB database, rather than consuming production TSE vote CSVs. The file publisher, symlink generation scheme, CAS fencing, and dbt consumer models are real. A separately scoped live-source test and full coordinated `build_candidate_fact_partition.py` / `int_candidate_fact_partition` CTAS chain remain necessary before global cache eviction.

### Three physical/analytical arms

1. Create a small synthetic `main.int_candidate_fact_partition` for 2024/municipal and publish the unrelated Parquet partition using the real publisher.
2. Create 2026/general votes, publish their immutable generation, and seed a scratch `main.dim_candidate` table containing exactly the required real-model join fields.
3. Execute `dbt run --select fact_candidate_votes candidate_ranking` with an isolated profile and physical root. Verify that `candidate_ranking` is a persisted DuckDB table and fingerprint its rows.
4. Copy the **closed** baseline DuckDB database and facts directory into independent control and recovery arms, preserving **relative symlinks**.
5. Change the synthetic physical source values in both arms. The control publishes normally, then rebuilds and captures the actual dbt consumer output.
6. The recovery arm starts the **real publisher in a child process**, injects a real Linux/WSL `SIGKILL` **after the immutable Parquet generation is durable but before the `data.parquet` symlink swap**, and proves that the previously published generation is still readable with identical bytes/rows.
7. Retry publication under the real lock; verify pointer switch, retained old immutable generation, abandoned symlink cleanup, and rejection of a stale writer using `expected_generation` CAS.
8. Re-run `fact_candidate_votes` + `candidate_ranking` in the recovered DuckDB. Compare schema, cardinality, duplicate-sensitive row multiset and semantic change from the previous generation.
9. Verify the unrelated 2024/municipal partition was not affected. Delete the scratch with `TemporaryDirectory` even if the test fails.

## Safety

- **No canonical TSE data are read or written.** No production raw/warehouse/fact-root arguments are accepted.
- The `SIGKILL` targets **only the synthetic publisher child**, never the caller's shell, dbt worker, production ingestion or other processes.
- Must run under Linux/WSL with POSIX filesystem primitives. The scratch root is checked and free temporary disk space is required.
- Python socket interception and an isolated dbt profile without `httpfs` reduce accidental networking but are **not an OS-level network sandbox**.
- The official real-source certification and failure matrix remain separate gates.

## WSL commands

```bash
cd ~/github/experiments-physical-recovery/data/tse-election-data
make setup
make dbt-deps
make unit
make medallion-contracts

# Runs ONLY synthetic sample data in a disposable tree:
make physical-publication-recovery \
  PHYSICAL_RECOVERY_JSON=/tmp/tse-pr45-physical-recovery.json
python3 -m json.tool /tmp/tse-pr45-physical-recovery.json
```

`result=PASS` must be reported by the real WSL run before PR45 can be called physically certified. A PR/merge is not created by the installer. Unit tests alone do not certify the dbt/Parquet pipeline.

### Output and gates

The JSON records independent before/control/recovery Parquet and `candidate_ranking` signatures, process `SIGKILL` exit status, boolean gates, and tails of the three dbt runs. A successful gate proves real physical publication and recovery semantics for this synthetic boundary, **not** equivalence of full TSE 2026 vote results or historical partitions.

## Follow-up before global cache eviction

1. Real raw `vote_result` recovery and CTAS recomputation (with materialization budgets and a bounded representative sample or isolated real cycle).
2. Complete coordinated `build_candidate_fact_partition.py` and physical `int_candidate_fact_partition` path, including interrupted source preparation and worker death during dbt.
3. Crash matrix before/after fsync and symlink swap; readers across old/new generations; durable restart; concurrent independent partitions.
4. Expanded persisted Gold/Semantic outcomes for actual vote data, multi-year lineage and warehouse equivalence.
5. Formal scoped eviction policy with per-resource reversible rollout and evidence-backed budget.

No production/global cache pruning is authorized by this PR.


## Physical certification — 2026-10-13

**Result: PASS** — synthetic fixture using real Parquet publisher
and real dbt `fact_candidate_votes` / `candidate_ranking`.

- SIGKILL return code: **-9**
- Failure boundary: after immutable fsync, before symlink swap
- Old immutable generation retained: **PASS**
- Stale writer fenced by CAS: **PASS**
- Unrelated 2024 partition preserved: **PASS**
- Control/recovery Parquet equivalence: **PASS**
- Control/recovery persisted ranking equivalence: **PASS**
- Published update detected: **PASS**
- dbt initial/control/recovered: **2/2 PASS each**
- No canonical TSE lake access: **PASS**

Evidence:
`contracts/certification/pr45_physical_recovery_2026.json`.

This certification does not cover real TSE vote ingestion or the
complete physical CTAS pipeline. Global cache eviction remains blocked.
