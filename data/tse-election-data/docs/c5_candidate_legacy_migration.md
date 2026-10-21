# C5-P — Legacy candidate Parquet migration

## Scope

Promote existing candidate fact Parquet files from regular-file publication
to immutable-generation naming, without rewriting their content.

This operation is separate from ODCS publication and GG certification.

## Preconditions

- Operate on a local POSIX filesystem supporting flock, hard links,
  atomic rename and directory fsync.
- All candidate partition publishers must cooperate with `.publish.lock`.
- Published Parquet content must never be modified in-place.
- The expected SHA-256 must come from independently verified inventory.
- The source must pass exact candidate/zone uniqueness checks.
- The warehouse and historical partition roots must not be rebuilt
  during migration.

## Guarantees

- Byte identity is checked against the expected SHA-256.
- Existing bytes are preserved using a hard link.
- Zone uniqueness evidence is persisted before pointer publication.
- `data.parquet` is replaced atomically with a generation symlink.
- Interrupted operations can be resumed idempotently.
- The normal candidate publisher can replace the migrated generation
  using its existing compare-and-swap protocol.

## Limitations

- Hard links share an inode; they do not enforce immutable bytes.
- Non-cooperating writers can invalidate the migration evidence.
- SHA-256 evidence and zone proofs do not constitute GG certification.
- Readers requiring certified provenance must validate the relevant
  physical evidence.
- This migration does not certify upstream source completeness.

## Production rollout

Publication is a separate, explicitly authorized operation.

Before rollout, verify the five audited cycle hashes, review active
writers and readers, and establish a recovery procedure.

Promote one cycle at a time, verify the published pointer and evidence,
and stop immediately on any mismatch. Never bulk-migrate on failure.
