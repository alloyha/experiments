"""C5-P coordinator recovery tests. No production warehouse access."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

from scripts import coordinate_candidate_legacy_migration as coordinator
from scripts.certify_gold_graph_cycle import selected_publication


class CoordinatorTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="c5p-coordinator-")
        self.addCleanup(self.temp.cleanup)

        base = Path(self.temp.name)
        self.db = base / "warehouse.duckdb"
        self.root = base / "facts"
        self.attestations = base / "graph_attestations"
        self.attestations.mkdir()

        self.partition = self.root / "election_type=general/year=2026"
        self.partition.mkdir(parents=True)

        self.public = self.partition / "data.parquet"
        self.cert_path = self.attestations / "2026-general.json"

        with duckdb.connect(str(self.db)):
            pass

        with duckdb.connect(":memory:") as con:
            con.execute("""
                COPY (
                    SELECT
                        '2026:general:1'::VARCHAR AS election_id,
                        1::INTEGER AS round_number,
                        'GO'::VARCHAR AS uf,
                        '93734'::VARCHAR AS municipality_code,
                        '10'::VARCHAR AS office_code,
                        FALSE AS is_transit_vote,
                        '123'::VARCHAR AS candidate_id,
                        10::INTEGER AS zone,
                        42::BIGINT AS nominal_valid_votes
                ) TO ? (FORMAT PARQUET)
            """, [str(self.public)])

        self.original = self.public.read_bytes()
        self.sha = hashlib.sha256(self.original).hexdigest()
        self.publication = selected_publication(
            self.root, 2026, "general"
        )[1]

        self.identity = {
            "warehouse": str(self.db.resolve()),
            "source_root": str(self.root.resolve()),
            "cycle": "2026:general",
            "lifecycle_epoch": "fixture-epoch",
        }

        self.cert_path.write_text(json.dumps({
            "version": 1,
            **self.identity,
            "snapshot": {
                "publication": self.publication
            },
        }))

        self.identity_patch = patch.object(
            coordinator.graph_freshness,
            "identity",
            return_value=self.identity,
        )
        self.snapshot_patch = patch.object(
            coordinator.graph_freshness,
            "snapshot",
            side_effect=lambda *args, **kwargs: {
                "publication": selected_publication(
                    self.root, 2026, "general"
                )[1]
            },
        )
        self.audit_patch = patch.object(
            coordinator.graph_freshness,
            "audit_certificate",
            side_effect=self.check_certificate,
        )

        self.identity_patch.start()
        self.snapshot_patch.start()
        self.audit_patch.start()

        self.addCleanup(self.identity_patch.stop)
        self.addCleanup(self.snapshot_patch.stop)
        self.addCleanup(self.audit_patch.stop)

    def check_certificate(
        self, attestations_dir, db, root, cycle, current
    ):
        record = json.loads(self.cert_path.read_text())
        if record["snapshot"] != current:
            raise RuntimeError("Stale GG snapshot")
        return record

    def run_migration(self, **kwargs):
        return coordinator.coordinate(
            db=self.db,
            root=self.root,
            attestations_dir=self.attestations,
            year=2026,
            election_type="general",
            expected_sha256=self.sha,
            **kwargs,
        )

    def assert_preserved(self):
        self.assertEqual(self.public.read_bytes(), self.original)
        self.assertEqual(
            hashlib.sha256(self.public.read_bytes()).hexdigest(),
            self.sha,
        )

    def test_success_revoke_before_physical_publication(self):
        result = self.run_migration()

        self.assertEqual(
            result["status"], "AWAITING_GG_CERTIFICATION"
        )
        self.assertTrue(self.public.is_symlink())
        self.assertFalse(self.cert_path.exists())
        self.assertTrue(
            (self.partition / coordinator.JOURNAL_NAME).is_file()
        )
        self.assert_preserved()

    def test_wrong_hash_preserves_authority(self):
        with self.assertRaises(RuntimeError):
            coordinator.coordinate(
                db=self.db,
                root=self.root,
                attestations_dir=self.attestations,
                year=2026,
                election_type="general",
                expected_sha256="0" * 64,
            )

        self.assertTrue(self.cert_path.exists())
        self.assertFalse(self.public.is_symlink())
        self.assert_preserved()

    def test_after_journal_recovery(self):
        with self.assertRaisesRegex(
            RuntimeError, "Injected failure: after_journal"
        ):
            self.run_migration(fault="after_journal")

        self.assertTrue(self.cert_path.exists())
        self.assertFalse(self.public.is_symlink())

        self.run_migration()
        self.assertFalse(self.cert_path.exists())
        self.assertTrue(self.public.is_symlink())
        self.assert_preserved()

    def test_after_revoke_recovery(self):
        with self.assertRaisesRegex(
            RuntimeError, "Injected failure: after_revoke"
        ):
            self.run_migration(fault="after_revoke")

        self.assertFalse(self.cert_path.exists())
        self.assertFalse(self.public.is_symlink())

        self.run_migration()
        self.assertTrue(self.public.is_symlink())
        self.assert_preserved()

    def test_after_swap_recovery(self):
        with self.assertRaisesRegex(
            RuntimeError, "Injected failure: after_swap"
        ):
            self.run_migration(fault="after_swap")

        self.assertFalse(self.cert_path.exists())
        self.assertTrue(self.public.is_symlink())

        result = self.run_migration()
        self.assertEqual(
            result["status"], "AWAITING_GG_CERTIFICATION"
        )
        self.assert_preserved()

    def test_idempotent_reexecution(self):
        first = self.run_migration()
        second = self.run_migration()

        self.assertEqual(
            first["migration"], second["migration"]
        )
        self.assertFalse(self.cert_path.exists())
        self.assert_preserved()

    def test_conflicting_certificate_after_journal(self):
        with self.assertRaisesRegex(
            RuntimeError, "Injected failure: after_journal"
        ):
            self.run_migration(fault="after_journal")

        record = json.loads(self.cert_path.read_text())
        record["certified_at_utc"] = "changed"
        self.cert_path.write_text(json.dumps(record))

        with self.assertRaisesRegex(
            RuntimeError, "GG certificate changed"
        ):
            self.run_migration()

        self.assertTrue(self.cert_path.exists())
        self.assertFalse(self.public.is_symlink())


    def test_warehouse_lock_contention(self):
        import fcntl

        lock = self.db.with_name(self.db.name + ".build.lock")
        with lock.open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                with self.assertRaises(TimeoutError):
                    self.run_migration(lock_timeout=0.05)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

        self.assertTrue(self.cert_path.exists())
        self.assertFalse(self.public.is_symlink())
        self.assertFalse(
            (self.partition / coordinator.JOURNAL_NAME).exists()
        )

    def test_partition_lock_contention(self):
        import fcntl

        lock = self.partition / ".publish.lock"
        with lock.open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                with self.assertRaises(TimeoutError):
                    self.run_migration(lock_timeout=0.05)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

        self.assertTrue(self.cert_path.exists())
        self.assertFalse(self.public.is_symlink())
        self.assertFalse(
            (self.partition / coordinator.JOURNAL_NAME).exists()
        )


    def assert_recovery_from_physical_fault(self, fault):
        with self.assertRaisesRegex(
            RuntimeError, f"Injected failure: {fault}"
        ):
            self.run_migration(fault=fault)

        journal = self.partition / coordinator.JOURNAL_NAME

        self.assertTrue(journal.is_file())
        self.assertFalse(self.cert_path.exists())
        self.assertFalse(self.public.is_symlink())
        self.assert_preserved()

        result = self.run_migration()

        self.assertEqual(
            result["status"], "AWAITING_GG_CERTIFICATION"
        )
        self.assertTrue(self.public.is_symlink())
        self.assertFalse(self.cert_path.exists())
        self.assert_preserved()

        # Recovery must remain idempotent.
        again = self.run_migration()
        self.assertEqual(result["migration"], again["migration"])

    def test_recovery_after_zone_proof(self):
        self.assert_recovery_from_physical_fault("after_zone_proof")

    def test_recovery_after_generation(self):
        self.assert_recovery_from_physical_fault("after_generation")

    def test_recovery_after_evidence(self):
        self.assert_recovery_from_physical_fault("after_evidence")

    def test_recovery_before_swap(self):
        self.assert_recovery_from_physical_fault("before_swap")

    def test_missing_certificate_rejects_new_intent(self):
        self.cert_path.unlink()

        with self.assertRaisesRegex(
            RuntimeError, "Missing active GG certificate"
        ):
            self.run_migration()

        self.assertFalse(self.public.is_symlink())
        self.assertFalse(
            (self.partition / coordinator.JOURNAL_NAME).exists()
        )


    def test_reject_journal_with_wrong_epoch(self):
        with self.assertRaisesRegex(
            RuntimeError, "Injected failure: after_journal"
        ):
            self.run_migration(fault="after_journal")

        journal = self.partition / coordinator.JOURNAL_NAME
        record = json.loads(journal.read_text())
        record["lifecycle_epoch"] = "foreign-epoch"
        journal.write_text(json.dumps(record))

        with self.assertRaisesRegex(
            RuntimeError, "Migration journal identity mismatch"
        ):
            self.run_migration()

        self.assertTrue(self.cert_path.exists())
        self.assertFalse(self.public.is_symlink())

    def test_reject_recreated_certificate_after_revoke(self):
        with self.assertRaisesRegex(
            RuntimeError, "Injected failure: after_revoke"
        ):
            self.run_migration(fault="after_revoke")

        self.cert_path.write_text(json.dumps({
            "version": 1,
            **self.identity,
            "snapshot": {"publication": self.publication},
            "certified_at_utc": "unauthorized-reissue",
        }))

        with self.assertRaises(RuntimeError):
            self.run_migration()

        self.assertTrue(self.cert_path.exists())
        self.assertFalse(self.public.is_symlink())

    def test_reject_missing_revocation_evidence(self):
        with self.assertRaisesRegex(
            RuntimeError, "Injected failure: after_revoke"
        ):
            self.run_migration(fault="after_revoke")

        revoked = list(
            (self.attestations / "revoked").glob(
                "2026-general-*.json"
            )
        )
        self.assertEqual(len(revoked), 1)

        revoked[0].unlink()

        # Merely observing an absent active certificate
        # must not authorize migration.
        with self.assertRaisesRegex(
            RuntimeError, "revocation evidence"
        ):
            self.run_migration()

        self.assertFalse(self.public.is_symlink())
        self.assertFalse(self.cert_path.exists())


    def test_journal_fsync_failure_preserves_gg(self):
        original = coordinator.fsync_directory

        def fail_journal_sync(directory):
            if Path(directory) == self.partition:
                raise OSError("injected journal directory fsync failure")
            return original(directory)

        with patch.object(
            coordinator, "fsync_directory",
            side_effect=fail_journal_sync,
        ):
            with self.assertRaisesRegex(OSError, "journal directory fsync"):
                self.run_migration()

        self.assertTrue(self.cert_path.exists())
        self.assertFalse(self.public.is_symlink())
        self.assert_preserved()

    def test_reject_tampered_certificate_digest_in_journal(self):
        with self.assertRaisesRegex(
            RuntimeError, "Injected failure: after_revoke"
        ):
            self.run_migration(fault="after_revoke")

        journal = self.partition / coordinator.JOURNAL_NAME
        record = json.loads(journal.read_text())
        record["certificate_sha256"] = "0" * 64
        journal.write_text(json.dumps(record))

        with self.assertRaisesRegex(
            RuntimeError, "revocation evidence"
        ):
            self.run_migration()

        self.assertFalse(self.cert_path.exists())
        self.assertFalse(self.public.is_symlink())

    def test_reject_recreated_gg_after_swap(self):
        with self.assertRaisesRegex(
            RuntimeError, "Injected failure: after_swap"
        ):
            self.run_migration(fault="after_swap")

        self.cert_path.write_text(json.dumps({
            "version": 1,
            **self.identity,
            "snapshot": {"publication": self.publication},
            "certified_at_utc": "unauthorized-reissue",
        }))

        with self.assertRaisesRegex(
            RuntimeError, "Active GG certificate after pointer migration"
        ):
            self.run_migration()

        self.assertTrue(self.public.is_symlink())
        self.assertTrue(self.cert_path.exists())
        self.assert_preserved()

    def test_reject_mutated_physical_generation_on_recovery(self):
        with self.assertRaisesRegex(
            RuntimeError, "Injected failure: after_swap"
        ):
            self.run_migration(fault="after_swap")

        generation = self.public.resolve()
        generation.write_bytes(b"corrupted")

        with self.assertRaises(RuntimeError):
            self.run_migration()

        self.assertFalse(self.cert_path.exists())
        self.assertTrue(self.public.is_symlink())



    def test_sigkill_after_gg_revoke_then_recover(self):
        import os
        import selectors
        import signal
        import subprocess
        import sys

        worker = r"""
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

from scripts import coordinate_candidate_legacy_migration as c
from scripts.certify_gold_graph_cycle import selected_publication

db, root, attestations, sha = (
    Path(sys.argv[1]), Path(sys.argv[2]),
    Path(sys.argv[3]), sys.argv[4]
)
cycle = "2026:general"
cert = attestations / "2026-general.json"
identity = {
    "warehouse": str(db.resolve()),
    "source_root": str(root.resolve()),
    "cycle": cycle,
    "lifecycle_epoch": "fixture-epoch",
}

def snapshot(*args, **kwargs):
    return {
        "publication": selected_publication(root, 2026, "general")[1]
    }

def audit(*args, **kwargs):
    record = json.loads(cert.read_text())
    if record["snapshot"] != args[-1]:
        raise RuntimeError("Stale GG snapshot")
    return record

def stop_before_physical_migration(*args, **kwargs):
    print("GG_REVOKED", flush=True)
    while True:
        time.sleep(1)

with (
    patch.object(c.graph_freshness, "identity", return_value=identity),
    patch.object(c.graph_freshness, "snapshot", side_effect=snapshot),
    patch.object(c.graph_freshness, "audit_certificate", side_effect=audit),
    patch.object(c, "_migrate_locked",
                 side_effect=stop_before_physical_migration),
):
    c.coordinate(
        db=db, root=root, attestations_dir=attestations,
        year=2026, election_type="general",
        expected_sha256=sha, lock_timeout=2,
    )
"""

        repo = Path(__file__).resolve().parents[1]
        process = subprocess.Popen(
            [
                sys.executable, "-u", "-c", worker,
                str(self.db), str(self.root),
                str(self.attestations), self.sha,
            ],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        try:
            selector = selectors.DefaultSelector()
            try:
                selector.register(process.stdout, selectors.EVENT_READ)
                ready = selector.select(timeout=10)
            finally:
                selector.close()

            self.assertTrue(ready, "Worker did not signal readiness")
            self.assertEqual(process.stdout.readline().strip(), "GG_REVOKED")
            self.assertIsNone(process.poll())

            # At this point the worker holds both exclusive locks.
            self.assertFalse(self.cert_path.exists())
            self.assertFalse(self.public.is_symlink())

            process.send_signal(signal.SIGKILL)
            process.communicate(timeout=10)
            self.assertEqual(process.returncode, -signal.SIGKILL)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)

        # A new process execution can reacquire the released locks.
        result = self.run_migration(lock_timeout=2)
        self.assertEqual(
            result["status"], "AWAITING_GG_CERTIFICATION"
        )
        self.assertTrue(self.public.is_symlink())
        self.assertFalse(self.cert_path.exists())
        self.assert_preserved()

    def test_journal_fsync_uncertain_then_recover(self):
        original = coordinator.fsync_directory
        injected = False

        def fail_once(directory):
            nonlocal injected
            if Path(directory) == self.partition and not injected:
                injected = True
                raise OSError("injected uncertain journal fsync")
            return original(directory)

        with patch.object(
            coordinator,
            "fsync_directory",
            side_effect=fail_once,
        ):
            with self.assertRaisesRegex(
                OSError, "uncertain journal fsync"
            ):
                self.run_migration()

        journal = self.partition / coordinator.JOURNAL_NAME

        # The hard link can remain visible despite failed fsync.
        self.assertTrue(journal.is_file())
        self.assertTrue(self.cert_path.is_file())
        self.assertFalse(self.public.is_symlink())

        result = self.run_migration()

        self.assertEqual(
            result["status"],
            "AWAITING_GG_CERTIFICATION",
        )
        self.assertFalse(self.cert_path.exists())
        self.assertTrue(self.public.is_symlink())
        self.assert_preserved()


    def test_revoke_rename_failure_preserves_authority(self):
        from scripts import graph_freshness
        from unittest.mock import patch

        original_replace = graph_freshness.os.replace

        def fail_certificate_move(src, dst):
            if Path(src) == self.cert_path:
                raise OSError("injected revoke rename failure")
            return original_replace(src, dst)

        with patch.object(
            graph_freshness.os,
            "replace",
            side_effect=fail_certificate_move,
        ):
            with self.assertRaisesRegex(
                OSError, "revoke rename failure"
            ):
                self.run_migration()

        self.assertTrue(self.cert_path.is_file())
        self.assertFalse(self.public.is_symlink())

        journal = self.partition / coordinator.JOURNAL_NAME
        self.assertTrue(journal.is_file())

        result = self.run_migration()
        self.assertEqual(
            result["status"], "AWAITING_GG_CERTIFICATION"
        )
        self.assertFalse(self.cert_path.exists())
        self.assertTrue(self.public.is_symlink())
        self.assert_preserved()

    def test_revoke_fsync_uncertain_then_recover(self):
        from scripts import graph_freshness
        from unittest.mock import patch

        original_sync = graph_freshness.fsync_directory
        injected = False

        def fail_once(directory):
            nonlocal injected
            if Path(directory) == self.attestations and not injected:
                injected = True
                raise OSError("injected revoke fsync uncertainty")
            return original_sync(directory)

        with patch.object(
            graph_freshness,
            "fsync_directory",
            side_effect=fail_once,
        ):
            with self.assertRaisesRegex(
                OSError, "revoke fsync uncertainty"
            ):
                self.run_migration()

        self.assertFalse(self.cert_path.exists())
        self.assertFalse(self.public.is_symlink())

        revoked = list(
            (self.attestations / "revoked").glob(
                "2026-general-*.json"
            )
        )
        self.assertEqual(len(revoked), 1)

        result = self.run_migration()

        self.assertEqual(
            result["status"], "AWAITING_GG_CERTIFICATION"
        )
        self.assertTrue(self.public.is_symlink())
        self.assertFalse(self.cert_path.exists())
        self.assert_preserved()


class RealGGIntegrationTests(unittest.TestCase):
    """Real GG certification and C5-P migration, isolated on disk."""

    def test_certify_migrate_recertify(self):
        import subprocess
        import sys

        from tests.test_gold_graph_invariants import GoldGraphTests
        from scripts.certify_gold_graph_cycle import (
            SCOPED_SOURCES, quote_sql,
        )

        fixture = GoldGraphTests(
            "test_graph_projection_conserves_relationships"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)

        with tempfile.TemporaryDirectory(
            prefix="c5p-real-gg-"
        ) as directory:
            base = Path(directory)
            db = base / "real.duckdb"
            root = base / "facts"
            certs = base / "graph_attestations"
            cycle = "2024:municipal"

            part = (
                root / "election_type=municipal/year=2024"
            )
            part.mkdir(parents=True)

            public = part / "data.parquet"

            fixture.db.execute(
                f"ATTACH {quote_sql(str(db))} AS evidence"
            )
            try:
                for name in SCOPED_SOURCES:
                    fixture.db.execute(
                        f"CREATE TABLE evidence.{name} AS "
                        f"SELECT * FROM main.{name}"
                    )
            finally:
                fixture.db.execute("DETACH evidence")

            fixture.db.execute(
                "COPY (SELECT * EXCLUDE (cycle_id) "
                "FROM fact_candidate_votes) "
                f"TO {quote_sql(str(public))} (FORMAT PARQUET)"
            )

            before = public.read_bytes()
            digest = hashlib.sha256(before).hexdigest()

            project = Path(__file__).resolve().parents[1]

            command = [
                sys.executable,
                str(project / "scripts/graph_freshness.py"),
                "--db", str(db),
                "--source-root", str(root),
                "--attestations-dir", str(certs),
                "--cycle", cycle,
                "--memory-limit", "512MB",
                "--lock-timeout", "10",
            ]

            def invoke(action):
                return subprocess.run(
                    command + [action],
                    cwd=project,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

            issued = invoke("certify")
            self.assertEqual(
                issued.returncode, 0,
                issued.stdout + issued.stderr
            )

            audited = invoke("audit")
            self.assertEqual(
                audited.returncode, 0,
                audited.stdout + audited.stderr
            )

            from scripts.graph_guarded_read import read_graph

            def guarded_read():
                return read_graph(
                    db=db,
                    source_root=root,
                    attestations_dir=certs,
                    cycle=cycle,
                    relation="edges",
                    kind="VOTED_FOR_CANDIDATE",
                    limit=3,
                    lock_timeout=10,
                    memory_limit="512MB",
                )

            initial_read = guarded_read()
            self.assertTrue(initial_read["rows"])

            result = coordinator.coordinate(
                db=db,
                root=root,
                attestations_dir=certs,
                year=2024,
                election_type="municipal",
                expected_sha256=digest,
                lock_timeout=10,
            )

            self.assertEqual(
                result["status"],
                "AWAITING_GG_CERTIFICATION"
            )
            self.assertTrue(public.is_symlink())
            self.assertEqual(public.read_bytes(), before)

            # The old authority must not be usable after migration.
            rejected = invoke("audit")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(
                "Missing/invalid graph attestation",
                rejected.stdout + rejected.stderr
            )

            from scripts.graph_guarded_read import GuardError

            with self.assertRaisesRegex(
                GuardError, "Missing active graph certificate"
            ):
                guarded_read()

            # New authority is explicitly issued, not fabricated
            # by the migration coordinator.
            recertified = invoke("certify")
            self.assertEqual(
                recertified.returncode, 0,
                recertified.stdout + recertified.stderr
            )

            verified = invoke("audit")
            self.assertEqual(
                verified.returncode, 0,
                verified.stdout + verified.stderr
            )

            self.assertEqual(public.read_bytes(), before)

            final_read = guarded_read()
            self.assertTrue(final_read["rows"])
            self.assertEqual(initial_read["rows"], final_read["rows"])


if __name__ == "__main__":
    unittest.main()
