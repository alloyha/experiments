"""GG-009 conformance inventory: all referenced falsifiers stay executable.

Functional conformance lives in the referenced real-process test suites. This
inventory catches accidental deletion/renaming of one surface without creating
a second implementation of locking.
"""
from __future__ import annotations

import unittest

import yaml

from scripts.check_invariant_registry import (
    DEFAULT, ROOT, check_registry, resolve_runtime_check, resolve_test,
)

# Evidence for distinct code paths is run by `make unit`, not mocked here.
# Each item links directly to a real regression; this test enforces references.
EVIDENCE = {
    "dbt entrypoint and SIGKILL fd inheritance":
        "tests/test_coordinated_dbt_gateway.py::CoordinatedDbtGatewayTests.test_sigkill_parent_does_not_release_live_dbt_child_lock",
    "Makefile routes every dbt writer through gateway":
        "tests/test_coordinated_dbt_gateway.py::CoordinatedDbtGatewayTests.test_makefile_db_is_single_gateway_for_all_targets",
    "real dbt-to-graph reader handoff":
        "tests/test_graph_dbt_writer_handoff.py::RealDbtGraphHandoffTests.test_guarded_reader_handoff_real_dbt_commit_stales_certificate",
    "standalone physical publisher blocked by guarded reader":
        "tests/test_physical_publisher_warehouse_fencing.py::WarehouseFencedPublisherTests.test_standalone_publisher_cannot_bypass_graph_reader",
    "independent warehouse physical writers":
        "tests/test_physical_publisher_warehouse_fencing.py::WarehouseFencedPublisherTests.test_independent_warehouses_can_publish_concurrently",
    "candidate-cube writer full certification lock":
        "tests/test_candidate_cube_writer_fencing.py::CubeWriterFencingTests.test_lock_spans_snapshots_revoke_dbt_and_issue",
    "Gold guarded read post-read invalidation":
        "tests/test_graph_guarded_read.py::GraphReadGuardTests.test_post_read_fingerprint_change_denies_results",
    "Gold proof issuer blocked by reader":
        "tests/test_graph_guarded_read.py::GraphReadGuardTests.test_issuer_cannot_revoke_while_reader_holds_locks",
    "candidate cube audit real publisher interference":
        "tests/test_candidate_cube_guarded_audit.py::RealCubeAuditTests.test_real_fingerprint_audit_rejects_publication_during_held_locks",
    "candidate cube audit partial acquisition timeout":
        "tests/test_candidate_cube_guarded_audit.py::CubeAuditProcessLockTests.test_timeout_when_independent_writer_holds_partition_lock",
    "warehouse destructive maintenance and epoch rotation":
        "tests/test_warehouse_maintenance_lifecycle.py::WarehouseMaintenanceTests.test_shared_reader_blocks_deletion_before_epoch_rotation",
    "warehouse maintenance after SIGKILL":
        "tests/test_warehouse_maintenance_lifecycle.py::WarehouseMaintenanceTests.test_sigkill_during_cleanup_cannot_leave_prior_authority_valid",
}


class GG009ConformanceRegistryTests(unittest.TestCase):
    def test_conformance_matrix_only_references_existing_executable_tests(self):
        for surface, test in EVIDENCE.items():
            with self.subTest(surface=surface):
                self.assertTrue(resolve_test(test, ROOT), f"missing evidence: {test}")

    def test_runtime_authority_is_guarded_reader_not_unfenced_diagnostic(self):
        document = yaml.safe_load(DEFAULT.read_text(encoding="utf8"))
        record = next(r for r in document["invariants"] if r["id"] == "GG-009")
        self.assertEqual(record["status"], "implemented")
        self.assertEqual(record["runtime_check"],
                         "scripts/graph_guarded_read.py::read_graph")
        self.assertTrue(resolve_runtime_check(record["runtime_check"], ROOT))
        self.assertTrue(resolve_test(record["positive_test"], ROOT))
        self.assertIn(record["falsifier_test"], EVIDENCE.values())
        self.assertEqual(check_registry()["planned"], 1)


if __name__ == "__main__":
    unittest.main()
