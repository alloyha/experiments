import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.migrate_candidate_legacy import migrate


class LegacyMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.partition = self.root / "election_type=general/year=2026"
        self.partition.mkdir(parents=True)
        self.public = self.partition / "data.parquet"

        import duckdb

        escaped = str(self.public).replace("'", "''")
        with duckdb.connect() as con:
            con.execute(f"""
                COPY (
                    SELECT
                        '2026:general:1'::VARCHAR AS election_id,
                        1::INTEGER AS round_number,
                        'GO'::VARCHAR AS uf,
                        '93734'::VARCHAR AS municipality_code,
                        '10'::VARCHAR AS office_code,
                        FALSE AS is_transit_vote,
                        '123'::VARCHAR AS candidate_id,
                        10::INTEGER AS zone
                ) TO '{escaped}' (FORMAT PARQUET)
            """)

        self.content = self.public.read_bytes()
        self.digest = hashlib.sha256(self.content).hexdigest()

    def promote(self, **kwargs):
        return migrate(
            self.root, 2026, "general",
            expected_sha256=self.digest, **kwargs
        )

    def test_byte_preservation_and_idempotence(self):
        first = self.promote()
        self.assertTrue(self.public.is_symlink())
        self.assertEqual(self.public.read_bytes(), self.content)
        self.assertEqual(first, self.promote())

    def test_wrong_checksum_preserves_legacy(self):
        with self.assertRaises(RuntimeError):
            migrate(
                self.root, 2026, "general",
                expected_sha256="0" * 64,
            )
        self.assertFalse(self.public.is_symlink())

    def test_recovery_after_generation(self):
        with self.assertRaises(RuntimeError):
            self.promote(fault="after_generation")
        self.assertFalse(self.public.is_symlink())
        self.promote()
        self.assertTrue(self.public.is_symlink())

    def test_recovery_after_evidence(self):
        with self.assertRaises(RuntimeError):
            self.promote(fault="after_evidence")
        self.assertFalse(self.public.is_symlink())
        self.promote()
        self.assertTrue(self.public.is_symlink())

    def test_recovery_before_swap(self):
        with self.assertRaises(RuntimeError):
            self.promote(fault="before_swap")
        self.assertFalse(self.public.is_symlink())
        self.promote()
        self.assertTrue(self.public.is_symlink())

    def test_recovery_after_swap(self):
        with self.assertRaises(RuntimeError):
            self.promote(fault="after_swap")
        self.assertTrue(self.public.is_symlink())
        self.assertEqual(self.promote()["parquet_sha256"], self.digest)

    def test_reject_conflicting_generation(self):
        self.promote()
        self.public.unlink()
        self.public.symlink_to("generation-unrelated.parquet")
        (self.partition / "generation-unrelated.parquet").write_bytes(b"x")
        with self.assertRaises(RuntimeError):
            self.promote()




    def test_real_parquet_preserves_schema_and_rows(self):
        import duckdb

        # Substituir a fixture inicial por um Parquet real.
        self.public.unlink()
        escaped = str(self.public).replace("'", "''")

        with duckdb.connect() as con:
            con.execute(f"""
                COPY (
                    SELECT
                        2026::INTEGER AS election_year,
                        'general'::VARCHAR AS election_type,
                        '2026:general'::VARCHAR AS cycle_id,
                        '2026:general:1'::VARCHAR AS election_id,
                        1::INTEGER AS round_number,
                        'GO'::VARCHAR AS uf,
                        '93734'::VARCHAR AS municipality_code,
                        '10'::VARCHAR AS office_code,
                        FALSE AS is_transit_vote,
                        '123'::VARCHAR AS candidate_id,
                        10::INTEGER AS zone,
                        42::BIGINT AS nominal_votes
                ) TO '{escaped}' (FORMAT PARQUET)
            """)

            before = con.execute(
                "SELECT * FROM read_parquet(?)",
                [str(self.public)],
            ).fetchall()

            schema_before = con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(self.public)],
            ).fetchall()

        digest = hashlib.sha256(self.public.read_bytes()).hexdigest()

        result = migrate(
            self.root,
            2026,
            "general",
            expected_sha256=digest,
        )

        with duckdb.connect() as con:
            after = con.execute(
                "SELECT * FROM read_parquet(?)",
                [str(self.public)],
            ).fetchall()

            schema_after = con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(self.public)],
            ).fetchall()

        self.assertEqual(before, after)
        self.assertEqual(schema_before, schema_after)
        self.assertTrue(self.public.is_symlink())
        self.assertEqual(result["parquet_sha256"], digest)

    def test_reject_mutated_existing_generation(self):
        self.promote()
        immutable = self.public.resolve()
        immutable.write_bytes(b"corrupted")

        with self.assertRaises(RuntimeError):
            self.promote()

    def test_reject_invalid_cycle(self):
        with self.assertRaises(ValueError):
            migrate(
                self.root,
                2026,
                "unknown",
                expected_sha256=self.digest,
            )


    def test_reject_tampered_zone_proof(self):
        import json

        result = self.promote()
        proof = (
            self.partition /
            result["generation"].replace(
                ".parquet", ".zone-uniqueness.json"
            )
        )

        saved = json.loads(proof.read_text())
        saved["distinct_group_zone_keys"] = 999
        proof.write_text(json.dumps(saved))

        with self.assertRaises(RuntimeError):
            self.promote()

    def test_recovery_after_zone_proof(self):
        with self.assertRaisesRegex(
            RuntimeError, "Injected failure: after_zone_proof"
        ):
            self.promote(fault="after_zone_proof")

        self.assertFalse(self.public.is_symlink())
        self.promote()
        self.assertTrue(self.public.is_symlink())

    def test_reject_duplicate_candidate_zone(self):
        import duckdb

        self.public.unlink()
        escaped = str(self.public).replace("'", "''")

        with duckdb.connect() as con:
            con.execute(f"""
                COPY (
                    SELECT
                        '2026:general:1'::VARCHAR AS election_id,
                        1::INTEGER AS round_number,
                        'GO'::VARCHAR AS uf,
                        '93734'::VARCHAR AS municipality_code,
                        '10'::VARCHAR AS office_code,
                        FALSE AS is_transit_vote,
                        '123'::VARCHAR AS candidate_id,
                        10::INTEGER AS zone
                    FROM range(2)
                ) TO '{escaped}' (FORMAT PARQUET)
            """)

        digest = hashlib.sha256(self.public.read_bytes()).hexdigest()

        with self.assertRaises(RuntimeError):
            migrate(
                self.root, 2026, "general",
                expected_sha256=digest,
            )

        self.assertFalse(self.public.is_symlink())



    def test_process_fencing_and_sigkill_recovery(self):
        import os
        import signal
        import subprocess
        import sys
        import time

        repo = Path(__file__).resolve().parents[1]
        ready = self.root / "lock-ready"

        holder = """
import fcntl, os, sys, time
from pathlib import Path

partition = Path(sys.argv[1])
ready = Path(sys.argv[2])
lock = partition / ".publish.lock"

with lock.open("a+b") as stream:
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    ready.write_text("READY")
    while True:
        time.sleep(1)
"""

        process = subprocess.Popen(
            [
                sys.executable, "-c", holder,
                str(self.partition), str(ready),
            ],
            cwd=repo,
        )

        try:
            deadline = time.monotonic() + 5
            while not ready.exists():
                if process.poll() is not None:
                    self.fail("Lock holder terminated before readiness")
                if time.monotonic() > deadline:
                    self.fail("Lock holder did not become ready")
                time.sleep(0.01)

            contender = subprocess.run(
                [
                    sys.executable, "-m",
                    "scripts.migrate_candidate_legacy",
                    "--root", str(self.root),
                    "--year", "2026",
                    "--election-type", "general",
                    "--expected-sha256", self.digest,
                    "--lock-timeout", "0.1",
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=10,
            )

            self.assertNotEqual(contender.returncode, 0)
            self.assertIn(
                "Timed out acquiring partition lock",
                contender.stderr,
            )
            self.assertFalse(self.public.is_symlink())
            self.assertEqual(self.public.read_bytes(), self.content)

        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGKILL)
            process.wait(timeout=5)

        # SIGKILL libera o flock no kernel; nenhuma remoção manual
        # do arquivo .publish.lock é necessária ou permitida.
        result = self.promote()
        self.assertTrue(self.public.is_symlink())
        self.assertEqual(result["parquet_sha256"], self.digest)



    def test_handoff_to_normal_publisher(self):
        import json
        import duckdb

        from scripts.write_candidate_fact_partition import (
            current_generation,
            publish_partition,
        )

        # Migrar o Parquet legado preparado pela fixture.
        migrated = self.promote()
        legacy_generation = migrated["generation"]
        legacy_path = self.partition / legacy_generation
        legacy_bytes = legacy_path.read_bytes()

        self.assertEqual(
            current_generation(self.partition),
            legacy_generation,
        )

        # Preparar uma fonte física descartável para o publicador normal.
        db_path = self.root / "fixture.duckdb"

        with duckdb.connect(str(db_path)) as con:
            con.execute("""
                CREATE TABLE main.int_candidate_fact_partition AS
                SELECT
                    2026::INTEGER AS election_year,
                    'general'::VARCHAR AS election_type,
                    *
                FROM read_parquet(?, hive_partitioning=false)
            """, [str(legacy_path)])

        publish_partition(
            db_path,
            self.root,
            2026,
            "general",
            expected_generation=legacy_generation,
        )

        active = current_generation(self.partition)

        self.assertNotEqual(active, legacy_generation)
        self.assertTrue(active.startswith("generation-"))
        self.assertTrue(self.public.is_symlink())
        self.assertEqual(self.public.read_bytes()[:4], b"PAR1")

        # A geração histórica deve sobreviver à publicação posterior.
        self.assertEqual(legacy_path.read_bytes(), legacy_bytes)

        new_proof = (
            self.partition /
            active.replace(".parquet", ".zone-uniqueness.json")
        )
        self.assertTrue(new_proof.is_file())
        proof = json.loads(new_proof.read_text())
        self.assertEqual(proof["result"], "PASS")
        self.assertEqual(proof["generation"], active)

        # Um orquestrador atrasado não pode sobrescrever a geração nova.
        with self.assertRaisesRegex(RuntimeError, "generation changed"):
            publish_partition(
                db_path,
                self.root,
                2026,
                "general",
                expected_generation=legacy_generation,
            )

        self.assertEqual(current_generation(self.partition), active)


if __name__ == "__main__":
    unittest.main()
