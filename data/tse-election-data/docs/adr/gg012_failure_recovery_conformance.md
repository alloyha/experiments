# GG-012 — Bounded failure/recovery conformance

**Decision:** Implemented for the cooperating TSE project certifier, guarded
consumer, physical publisher, coordinated dbt gateway and maintenance entrypoints
on POSIX-local filesystems supporting `flock`, atomic link/rename and `fsync`.
**Baseline:** `f29957f` (PR #64 merged). **Full production certification:**
not implied. This decision supersedes *status* references in historical
prototype and intermediate GG-009 ADRs, without changing their historical
experiment observations.

## Exact authority and stop condition

The Gold graph is a **live projection**, not a persistently materialized graph.
What is durable is its cycle-scoped, create-only **publication certificate**.
The authorized read interface is `graph_guarded_read.py::read_graph`, never
`graph_freshness.py audit` or the diagnostic SQL alone.

A cooperating issuer takes the EX warehouse lock and then the EX physical
partition lock. It revokes the previous certificate before computing two
freshness snapshots around all eight graph SQL assertions. Only identical
before/after evidence allows create-only certificate publication, whose file
and containing directory are synchronized. Guarded consumers hold matching
SH locks through initial evidence, query and post-read revalidation.

This establishes the GG-012 statement for the specified cooperative interface:
interruption does not authorize an unproven graph; a publication by a competing
cooperating writer cannot bypass the certifier, and any later source change
invalidates the old certificate.

## Existing evidence and explicit falsifiers

| Boundary | Existing executable evidence | Expected authority |
|---|---|---|
| Worker death after revoke | `test_gg012_graph_crash_falsifiers.py::RealGraphCrashRecoveryTests.test_death_after_revocation_denies_old_authority_then_restart` | No active proof until successful recertification; recovered rows match baseline |
| Worker death before hard-link | `test_gg012_graph_crash_falsifiers.py::RealGraphCrashRecoveryTests.test_death_before_hard_link_never_exposes_partial_certificate` | Temporary record is not active; retry succeeds |
| Worker death after hard-link | `test_gg012_graph_crash_falsifiers.py::RealGraphCrashRecoveryTests.test_death_after_hard_link_preserves_committed_authority` | Completed diagnostics' active proof remains visible after process death; create-only prevents overwrite |
| Concurrent physical writer | `test_gg012_graph_crash_falsifiers.py::RealGraphCrashRecoveryTests.test_publisher_waits_for_certifier_and_recovery_denies_stale_proof` | Writer waits for EX locks; stale proof denied; recovery matches continuous control |
| Post-link `fsync` failure | `test_gg012_uncertain_durability.py::UncertainGraphCertificateDurabilityTests.test_failed_post_link_fsync_must_rollback_active_authority` | Active proof retracted and directory resynchronized before retry if rollback I/O succeeds |
| Successful publication/revocation directory sync | `test_gg012_graph_crash_falsifiers.py::GraphCertificateDurabilityTests` | File and directory durability calls occur in order |
| Mid-certification source/graph changes or SQL failures | `test_graph_freshness.py::GraphFreshnessTests` | No new active certificate |
| Warehouse rebuild after death | `test_warehouse_maintenance_lifecycle.py::WarehouseMaintenanceTests.test_sigkill_during_cleanup_cannot_leave_prior_authority_valid` | Lifecycle epoch fences resurrection |
| Real dbt writer handoff | `test_graph_dbt_writer_handoff.py::RealDbtGraphHandoffTests.test_guarded_reader_handoff_real_dbt_commit_stales_certificate` | Changed Gold cannot be returned using old proof |

No new lock, persistence adapter, global coordinator, compatibility layer or
additional failure-injection scenario is required to ratify these existing
contracts. Registry GG-012 names `certify_with_locks` as the runtime check;
positive and negative oracle references resolve to existing executable tests.

## Nonclaims and separate promotion gate

- **Cooperative only:** direct SQL, external writers or processes bypassing the
  stable POSIX advisory lock inodes are not isolated by this design.
- **Not a hardware power-loss experiment:** `SIGKILL` is worker/process death,
  not a sudden loss of storage power. Durable directory operations are tested
  for call ordering, not under a physical power cut or NFS/object-store faults.
- **Persistent I/O failure:** repeated `EIO` during rollback/revocation does
  not guarantee that the filesystem has durably removed active authority.
  Stop serving data and repair storage before trusting that environment again.
- **No adversarial authenticity:** DuckDB row hashes are change detectors,
  not signed proofs resistant to crafted collisions.
- **Not complete electoral-data certification:** `--strict` validates that
  implemented invariants have executable positive and negative oracles.
  The separate registry policy `full_promotion` additionally requires
  real-cycle certification. The existing synthetic fixtures do not fulfill it.

**Final acceptance:** registry = 12 implemented, 0 planned, strict PASS; full
Python regression, Medallion DAG and compilation PASS; GitHub PR CI green;
real-cycle production promotion remains a distinct decision. Reopen this
contract only for a reproducible violation of the bounded guarantees above.
