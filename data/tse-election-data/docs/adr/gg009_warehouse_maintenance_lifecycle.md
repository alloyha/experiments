# GG-009: Fenced warehouse maintenance and persistent lifecycle identity

**Status:** Incremental contract, not GG-009 general certification.

`clean-warehouse` and `clean-all` execute inside the same stable
`<warehouse>.build.lock` used by coordinated DuckDB writers and guarded graph
readers. They leave that lock file in place and never remove its parent.

Before unlinking database or derived state, cleanup durably rotates
`<warehouse>.lifecycle` (fsync of file and directory). The epoch survives the
warehouse deletion. Both graph and candidate-cube freshness attestations
include the current epoch. An older attestation lacking the field must fail
closed; if identical DuckDB content is recreated at the same path, an old
attestation cannot become valid again. Epoch rotation happens **before**
any destructive action, so SIGKILL after the rotation invalidates prior proof.

The maintenance entry point uses only Python's standard library because
`clean-all` is allowed to delete `.venv`. Destructive paths outside the
explicit project root or via symlinks are rejected; `CONFIRM=1` is mandatory
for `clean-all`. The lock and lifecycle sidecars are intentionally retained.

**Migration:** all existing graph/candidate-cube certificates need fresh
certification after this change, even if their underlying contents are
unchanged. This is a versionless fail-closed authorization migration, not a
silent acceptance of older schema certificates.

**Limitations:** POSIX cooperative locking cannot protect against ad-hoc
DuckDB writers or manual `rm`. Only `make clean-warehouse`/`make clean-all`
are protected. GG-009 remains planned; GG-012 restart/uncertain publication
remains independent.
