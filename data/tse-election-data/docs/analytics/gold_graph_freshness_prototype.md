# Gold graph — GG-010: scoped freshness evidence (prototype)

This is **not** a concurrency-safe publication or certified consumer interface.
The existing Gold graph consists of live views. Freshness evidence captures the
published candidate Parquet generation and SHA-256, fingerprints of **all six**
cycle-scoped Gold relations consumed by the graph, and fingerprints of both
cycle-scoped derived graph views. It includes the database path, source root,
cycle and DuckDB version. Gold and graph row fingerprints use DuckDB's `hash`
plus count/XOR/sum and are **not cryptographic integrity guarantees**.

`certify` takes a before snapshot, revokes the previous active authority, runs
**all eight** SQL graph invariants using the existing physically pinned,
read-only diagnostic, takes an after snapshot, and creates a certificate only if
the snapshots match. `audit` independently recomputes the complete snapshot;
missing or stale evidence causes nonzero exit. No mutation of the DuckDB
warehouse, facts or dimensions is performed by either command.

```bash
MAIN="$HOME/github/experiments/data/tse-election-data"
PY="$MAIN/.venv/bin/python"
CANARY="$MAIN/data/warehouse/tse_event_registry_preview.duckdb"
FACTS="$MAIN/data/warehouse/fact_candidate_votes"
CERTS="$MAIN/data/warehouse/graph_attestations"

"$PY" scripts/graph_freshness.py --db "$CANARY" --source-root "$FACTS" \
  --attestations-dir "$CERTS" --cycle 2024:municipal audit
# Expected FAIL before any certified 2024 graph proof.

# Expensive, opt-in, single-cycle certification only:
"$PY" scripts/graph_freshness.py --db "$CANARY" --source-root "$FACTS" \
  --attestations-dir "$CERTS" --cycle 2024:municipal --memory-limit 4GB certify
"$PY" scripts/graph_freshness.py --db "$CANARY" --source-root "$FACTS" \
  --attestations-dir "$CERTS" --cycle 2024:municipal audit
```

Do not run on all cycles in CI. The diagnostic may scan materialized Gold
partitions but accesses only the selected candidate-fact Parquet path. Limit
memory/time externally. Existing certificates cannot be reused for another
warehouse, and certification is create-only (old proof archived before a retry).

**Limitations:** No writer-lock or read/verify transaction. A writer can change
Gold after the last check, and an uncoordinated writer can create inconsistent
intermediate reads. The absence of GG-009 and GG-012 means this proof must not
be used to publish a production-certified graph or advertised as complete
critical-invariant coverage. Protect the attestation directory against tampering;
these are unsigned change-detection records, not authenticated signatures.


## Cross-process JSON schema regression

DuckDB PRAGMA column metadata consists of `(column_name, type)` tuples. JSON
serializes each tuple as an array. The in-memory fingerprint must use arrays
as well, or a separate `audit` process rejects a newly issued certificate even
with unchanged sources. Tests cover JSON normalization and synthetic real-DuckDB
`certify -> audit -> mutate Gold -> audit fails`, in distinct processes.

Because `graph_freshness.py` is itself included in the versioned contract hash,
this correction deliberately invalidates certificates produced before the fix.
Reissue with `certify`; never modify a certificate to make it pass.


## Invariant registry promotion (GG-010)

`GG-010` is implemented with the runtime verifier
`scripts/graph_freshness.py::audit_certificate`, rather than a fabricated SQL
assertion. The registry validates the verifier's AST symbol and links a
cross-process positive test and a real Parquet-republication falsifier. The
synthetic regression also restores the original published Parquet bytes and
verifies that the unchanged certificate passes again, without silently
rewriting evidence. On the real 2024 canary, eight scoped SQL assertions,
`certify` and an independent `audit` have all passed.

This is freshness **at audit time**, not a consumer/producer transaction.
`GG-009` and `GG-012` are intentionally `planned` and `--strict` must remain
red until coordination and recovery can be demonstrated. No implicit global
read certification or safety against concurrent writes is claimed.
