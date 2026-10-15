# GG-009 — Candidate cube certifier writer fencing

Status: proposed. This amendment removes the direct unfenced `dbt run` and
`dbt test` path inside `scripts/certify_candidate_cube.py`.

`certify` acquires the existing `<warehouse>.build.lock` **before** reading
source signatures, revoking the prior cube certificate, running the cycle-scoped
dbt model, running both SQL tests or issuing the replacement certificate.
Both dbt subprocesses inherit the same POSIX lock FD, via `pass_fds`, so a
`SIGKILL` of the Python parent alone cannot free the lock while a child is
writing. The lock is not reacquired inside either dbt subprocess: nesting the
new dbt gateway under it would self-deadlock.

Reader/issuer contention and the pre-revocation fail-closed behavior are
covered by `test_candidate_cube_writer_fencing.py`. This remains **cooperative**:
ad-hoc DuckDB SQL clients, direct dbt callers outside the gateway, and
warehouse deletion/replacement require separate gates. GG-009 is not promoted.

Validation: `python -m unittest discover -s tests -p 'test_candidate_cube_writer_fencing.py' -v`
and the existing `test_candidate_cube_certifier.py` tests, followed by CI.
No historical cycles or canonical TSE data need to be rebuilt.
