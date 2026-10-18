# Candidate cube freshness certificate: POSIX durability hardening

The candidate cube certificate is **authority** for one persisted DuckDB
warehouse, Parquet publication root and election cycle; it is not a signed
cryptographic proof. Issuance follows successful dbt tests under the existing
coordinated exclusive warehouse writer lock. Reads must use the coordinated
shared-lock audit entrypoint.

## Publication and revocation boundaries

- Revoke by moving active authority into `revoked/`, then `fsync` the active
  parent directory and the archive directory **before** mutating dbt state.
- Issue by writing a fresh same-directory temporary JSON, flushing and
  `fsync`ing its bytes, then `os.link`ing it to the active name **create-only**.
  This avoids the check-then-`os.replace` race where another issuer's
  certificate could be overwritten. `fsync` the active parent directory
  before acknowledging issuance.
- On directory-sync failure after linking, retract the visible active name
  under the writer lock, sync the retraction and propagate the failure.
  Cleanup of the unpublished temporary is best effort via `finally`.
- Preserve the existing JSON format/version and audit comparison logic;
  already-issued certificates remain readable.

## Explicit limitations and follow-up falsifiers

Tests inject a racing issuer and failed directory `fsync` and verify that
no successful new authority is reported. These tests are *not* a demonstration
of power-loss durability or recovery equivalence after SIGKILL at every
syscall boundary. They also cannot protect uncoordinated external DuckDB or
Parquet writers, unsafe filesystems, or prevent unguarded reads during the
small publish-before-sync interval. Keep writer locks through issue/revoke,
and use guarded readers. A further crash matrix must test real worker death
before/after rename, link, and `fsync` against the target filesystem.

The graph certifier remains unchanged. The helper `fsync_directory` is reused
from the already operational physical partition publisher, rather than adding
another independent POSIX flush implementation.
