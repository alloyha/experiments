# PR42 — Cache eviction / offline rehydration certification

## Intent

Turn prior synthetic coverage into a **falsifiable, real-source, disposable-copy**
certification. Neither the canonical raw lake nor the local DuckDB warehouse is
written to by the certification command. This is an operational roundtrip gate;
it does not restore missing historical immutable source versions.

## Implementation

`python scripts/certify_cache_roundtrip.py` performs:

1. Select exactly one active resource by resource ID + year + logical domain.
   Reject ambiguous records, stale/partial materializations, missing source,
   path escapes, symlinks, missing hash or mismatching content/size, unexpected
   physical index entries, and inputs above configurable size limits.
2. Calculate original payload hashes and a downstream tabular reader signature.
3. Create an isolated temporary copy of that source, its extracted members and
   a minimal compatible control plane. Never hard-link files into the scratch
   directory: source and outputs must be copied as independent bytes.
4. Block socket connections for child Python interpreters with an explicit
   pre-execution Python bootstrap plus an isolated `sitecustomize.py` and invoke the repository's **real**
   `prune_extracted_raw.py --apply --domain DOMAIN` against the temporary raw
   directory only.
5. Prove that the scratch physical index has no references to the evicted
   objects; the scratch state records `evicted`; source SHA is unchanged.
6. Invoke the repository's **real** `ensure_materialized.py` in offline mode,
   with the exact year/election type/domain. Require `rehydrated=1, errors=0`.
7. Validate post-recovery byte equality, tabular reader equality (DuckDB's
   latin-1 semicolon reader), restored active state, exact index object-pairs,
   unchanged append-only manifest and source, strict `storage-report`
   checks **both** before and after rehydration, and `check_raw_contracts.py`.
8. Recheck hashes of the canonical control-plane files and source/extracted
   payloads. Destroy temporary copy even on errors, and output JSON evidence.

The command does **not** run the entire dbt graph. Exact source and extracted
bytes plus DuckDB raw-reader signatures demonstrate byte and raw-table
reproducibility; compiled/dbt model materializations and warehouse equivalence
are a **separate** downstream gate before global eviction approval.

## Real-resource gate in WSL (PR42 branch)

```bash
cd ~/github/experiments-cache-roundtrip/data/tse-election-data
RAW="$HOME/github/experiments/data/tse-election-data/data/tse"

.venv/bin/python scripts/certify_cache_roundtrip.py \
  --raw-root "$RAW" \
  --resource-id 6c3a8826-32da-460f-b306-97474b580efb \
  --year 2026 \
  --domain candidate_history \
  --reader duckdb \
  --output-json /tmp/tse-pr42-history-roundtrip.json
```

The program has **no destructive mode for canonical data**, and calls `--apply`
only on the freshly copied, disposable lake. Its limit defaults to 64 MiB
compressed source and 512 MiB extracted payload. Do not repoint the tool at
somebody else's active data or execute standalone prune on the canonical tree.

For a complete offline fixture and regression suite:

```bash
make unit
make medallion-contracts
.venv/bin/python -m unittest discover -s tests \
  -p 'test_cache_roundtrip_integration.py' -v
```

## Evidence and acceptance

- `result = PASS` for one physical source and at least one CSV object;
  `offline_enforced = true`; `read_only_canonical = true`;
  `payload_and_reader_equivalent = true`;
  `index_rebuilt = true`; `manifest_unchanged = true`.
- `source_sha256` matches current active source exactly.
- JSON report kept under `/tmp` or audit-backups, not in raw metadata.
- Failures in source hash, extraction, source/path isolation, index mutation,
  downstream signature or control-plane preservation fail closed.
- Only after this gate consider a **separate** disposable DuckDB/dbt evaluation
  and policy proposal for high-volume cache eviction.

## Not yet established

No attempt is made here to recover the historic missing 2026-10-08 source or to
infer the cause of that loss. No actual 39.2 GiB eviction is authorized.


## Observed physical certification — 2026-10-13

Execution on an isolated WSL worktree against the current
`candidate_history` resource for the 2026 election:

- Unit tests: **193 PASS**
- Medallion DAG contract: **PASS**
- Roundtrip certification: **PASS**
- Source SHA-256: `16fd5803af84fcb2d02dbecda70999d766c11264e833d354b6c0e88f0f1e0338`
- Immutable ZIP: 4,423,762 bytes
- Extracted CSV: 17,415,469 bytes
- Evicted and recovered objects: 1/1
- Offline recovery, index reconstruction, payload and DuckDB reader equivalence: **PASS**
- Strict audit and physical contracts after recovery: **PASS**
- Canonical source and control-plane hashes preserved
- Disposable copy destroyed

This establishes recoverability for the selected resource only.
Full downstream dbt/warehouse equivalence and safe eviction
of the 39.2 GiB production cache remain uncertified.
