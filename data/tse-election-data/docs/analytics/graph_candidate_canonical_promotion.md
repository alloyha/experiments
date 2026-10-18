# Candidate graph canonical promotion

This change promotes `COUNT(*)` **only** in the candidate zone rollup,
leaving the independent party rollup's `COUNT(DISTINCT zone)` intact.
The candidate conservation assertion now compares signed row multiplicities
on all ten typed projection columns. Any nonzero signed difference remains a
DBT violation; the result also reports the absolute differing multiplicity.

`certify_gold_graph_cycle.run(..., execute=True)` first executes the **exact**
active source scope and group/zone uniqueness guards on the pinned Parquet bytes.
The GG-010 `graph_freshness.py certify` path holds the existing warehouse and
partition locks while calling that diagnostic, and its post-snapshot fingerprint
binds source bytes, model SQL and assertion SQL to the issued certificate.
Historical V1 reports are not accepted as runtime authority; changed source
bytes or changed SQL require independent recertification.

Do not equate CI unit coverage or a five-cycle historical PASS with GG-010 graph
freshness certification. Promote only after five real cycle certifications
have succeeded on the active canonical warehouse using reviewed resources,
and after a guarded read rejects stale/modified certificates.

The preflight scans the physical Parquet; its cost is an explicit certification
overhead, not an inference about faster end-to-end execution. The older
`benchmark_graph_candidate_conservation.py` expects the **pre-promotion** SQL
tail and intentionally refuses changed canonical contracts. Preserve its
historical evidence; do not report new speedups without paired measurements.
