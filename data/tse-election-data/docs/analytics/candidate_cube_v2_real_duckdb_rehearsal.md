# Promotion V2 — Real DuckDB rehearsal (no live warehouse mutation)

The existing V2 tests use synthetic byte files. Before executing `activate` on
`data/warehouse/tse_analytics.duckdb`, run the additional **real DuckDB**
integration tests on disposable fixtures.

```bash
cd ~/github/experiments/data/tse-election-data
.venv/bin/python -m unittest discover -s tests \
  -p 'test_promote_candidate_cube_v2_real_duckdb.py' -v
```

The four tests construct **new DuckDB files in isolated temporary directories**
with 198 synthetic election events, five cycles, and a tiny candidate cube.
They exercise `prepare`, `status`, actual live-path DuckDB smoke, activation,
explicit recovery, automatic rollback after failed live smoke, and real Linux
`SIGKILL` followed by recovery at seven durable boundaries. The original V1
journal and backup hashes must remain unchanged, and the baseline must remain
queryable after every rollback.

The upstream manifest, Parquet partitions and contract proofs are **synthetic
fixtures** re-generated with the test database SHA-256 values. These tests are
therefore **state-machine, filesystem and DuckDB compatibility tests**, not a
new candidate-cube freshness certificate and not a substitute for the 5/5
real-source offline SQL contract report.

**Never pass a production path to the tests, and never run `activate` against the
existing `PREPARED` production V2 state as part of this rehearsal.** Approval of
production activation additionally requires a reader/writer drain, live-path
certificate issuance procedure and an explicit operator decision. A `PREPARED`
state with `recovery_required=true` is pending, not necessarily corrupt.
