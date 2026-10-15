# GG-009 — Cooperative guarded reads (foundation, not production promotion)

The existing physical publisher serializes one partition using `flock` on
`election_type=<kind>/year=<year>/.publish.lock`. The CTAS coordinator holds
`<warehouse>.build.lock` **exclusive** and then takes the partition lock.

`graph_guarded_read.py` takes the **same** locks **shared, in that order**, and
holds both across (a) graph GG-010 freshness audit, (b) a physically scoped
query over `graph_vertices` or `graph_edges`, and (c) a second audit. It rejects
missing or changed authority before returning rows. No arbitrary SQL can be
submitted and results are capped at 1–1,000 rows. Query output is JSON.

```bash
PY="$HOME/github/experiments/data/tse-election-data/.venv/bin/python"
MAIN="$HOME/github/experiments/data/tse-election-data"
"$PY" scripts/graph_guarded_read.py \
  --db "$MAIN/data/warehouse/tse_event_registry_preview.duckdb" \
  --source-root "$MAIN/data/warehouse/fact_candidate_votes" \
  --attestations-dir "$MAIN/data/warehouse/graph_attestations" \
  --cycle 2024:municipal --relation edges \
  --type VOTED_FOR_CANDIDATE --limit 3 --lock-timeout 5 --memory-limit 4GB
```

## Scope and non-goals

- **Cooperative only**: direct dbt/SQL writes and ad hoc clients do not honor
  these sidecar locks automatically. They may change Gold or the certificate
  during a read. Pre/post audit can detect many such changes but cannot make an
  uncoordinated read atomic, nor detect a change-and-restore ABA sequence.
- GG-010 `graph_freshness.py certify` now takes **exclusive** warehouse +
  partition locks before revoking or issuing a certificate. Existing guarded
  readers use matching **shared** locks, so a recertification cannot revoke
  during a cooperating read. Pass `--lock-timeout` to limit issuer waiting.
- This does **not** fence direct DuckDB SQL or uncoordinated dimension dbt
  writes. GG-009 remains planned until all publication paths are audited.
- A shared warehouse lock also blocks coordinated builds for **other cycles in
  the same DuckDB**, while still allowing shared readers. This intentionally
  preserves the existing single-writer warehouse scope.
- A certificate is bound to one DuckDB and one published source root; it is
  not a cryptographic signature. `hash()` is a change detector.
- Every read currently recomputes source, Gold and graph hashes **twice**.
  Benchmark before adding API/interactive use. The cost is intentional at this
  stage to expose and falsify race windows.
- The certifier/source-code fingerprint includes `graph_freshness.py`. This
  change invalidates existing GG-010 certificates. Reissue via `certify`
  before the next guarded read; do not edit certificates in place.
- No promotion of GG-009 or GG-012 to `implemented` from this partial guard.
  A full GG-009 change must migrate or fence **all relevant writers** and the
  certificate issuer, test kill/restart around locks and prove that no
  uncoordinated path is presented as certified.
