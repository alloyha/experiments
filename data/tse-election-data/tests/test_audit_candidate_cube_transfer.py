"""Read-only cube-transfer eligibility unit tests; no external data needed."""
from __future__ import annotations

from pathlib import Path
from collections import Counter
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

from scripts import audit_candidate_cube_transfer as gate


class CandidateCubeTransferGateUnitTests(unittest.TestCase):
    def test_strict_cycle_format(self):
        self.assertEqual(gate.parse_cycle("2024:municipal"), (2024, "municipal"))
        for invalid in ("2024;DROP", "2024:general--", "2024:municipal ", "1800:general"):
            with self.subTest(invalid=invalid), self.assertRaises(gate.GateError):
                gate.parse_cycle(invalid)

    def test_exact_model_column_projection(self):
        self.assertEqual(len(gate.COLUMNS), 29)
        self.assertEqual(len(set(gate.COLUMNS)), 29)
        self.assertEqual(gate.COLUMNS[0], "cycle_id")
        self.assertEqual(gate.COLUMNS[-1], "candidate_share_of_observed_nominal_votes")

    def test_sql_recomputes_direct_from_pinned_source_and_main_dimensions(self):
        sql = gate.expected_cube_sql(Path("/tmp/fact/data.parquet"), "2024:municipal")
        self.assertIn("read_parquet('/tmp/fact/data.parquet'", sql)
        self.assertIn("tse_analytics.main.dim_candidate", sql)
        self.assertIn("tse_analytics.main.dim_geography", sql)
        self.assertIn("ROW_NUMBER() OVER", sql)
        self.assertIn("sum(nominal_valid_votes)".upper(), sql.upper())
        self.assertNotIn("tse_event_registry_preview", sql)

    def test_exact_comparison_retains_original_except_all_fallback(self):
        sql = gate.multiset_comparison_sql("2022:general", mode="except-all")
        self.assertEqual(sql.count("EXCEPT ALL"), 2)
        self.assertIn("UNION ALL", sql)

    def test_signed_mode_groups_all_29_columns_without_lossy_hash(self):
        sql = gate.multiset_comparison_sql("2022:general", mode="signed")
        self.assertIn("SUM(_transfer_delta)", sql)
        self.assertIn("GROUP BY", sql)
        self.assertIn("COALESCE(SUM(ABS(_balance)), 0)", sql)
        self.assertNotIn("hash(", sql.lower())
        self.assertNotIn("EXCEPT ALL", sql)
        for col in gate.COLUMNS:
            self.assertIn(f'"{col}"', sql)

    def test_signed_sql_matches_python_counter_with_nulls_duplicates_and_changes(self):
        # SQLite also supports UNION ALL, GROUP BY and SUM; this is a concrete
        # execution test without depending on DuckDB or external electoral data.
        cols = ", ".join(f'"{c}"' for c in gate.COLUMNS)
        with sqlite3.connect(":memory:") as con:
            for relation in ("expected_cube", "preview_cube"):
                con.execute(f"CREATE TABLE {relation} ({', '.join(chr(34)+x+chr(34)+' TEXT' for x in gate.COLUMNS)})")
            def row(candidate, party, municipality=None):
                values = [None] * len(gate.COLUMNS)
                values[0] = "2022:general"
                values[gate.COLUMNS.index("candidate_id")] = candidate
                values[gate.COLUMNS.index("party")] = party
                values[gate.COLUMNS.index("municipality")] = municipality
                return tuple(values)
            a, b, c = row("A", "x", "São Paulo"), row("B", "y"), row("C", None)
            samples = [
                ([a, a, b], [b, a, a]),
                ([a, a, b], [a, b]),
                ([a, a, b], [a, b, b]),
                ([a, a, b], [a, b, c]),
                ([a, b], [c, a, b]),
                ([], []),
            ]
            sql = gate.multiset_comparison_sql("2022:general", mode="signed",
                          expected_relation="expected_cube", preview_relation="preview_cube")
            for original, preview in samples:
                con.execute("DELETE FROM expected_cube")
                con.execute("DELETE FROM preview_cube")
                if original:
                    con.executemany(f"INSERT INTO expected_cube ({cols}) VALUES ({', '.join('?' for _ in gate.COLUMNS)})", original)
                if preview:
                    con.executemany(f"INSERT INTO preview_cube ({cols}) VALUES ({', '.join('?' for _ in gate.COLUMNS)})", preview)
                actual = con.execute(sql).fetchone()[0]
                left, right = Counter(original), Counter(preview)
                expected = sum(abs(left[v] - right[v]) for v in left.keys() | right.keys())
                self.assertEqual(actual, expected)

    def test_duckdb_signed_matches_original_except_all_on_real_engine(self):
        try:
            import duckdb
        except ImportError:
            self.skipTest("DuckDB not installed in this test environment")
        columns = ", ".join(f'"{c}" VARCHAR' for c in gate.COLUMNS)
        names = ", ".join(f'"{c}"' for c in gate.COLUMNS)
        params = ", ".join("?" for _ in gate.COLUMNS)
        with duckdb.connect(":memory:") as con:
            for table in ("expected_cube", "preview_cube"):
                con.execute(f"CREATE TABLE {table} ({columns})")
            def make_row(i):
                row = [None] * len(gate.COLUMNS)
                row[0] = "2022:general"
                row[gate.COLUMNS.index("candidate_id")] = str(i % 13)
                row[gate.COLUMNS.index("candidate_name")] = f"Nome {i % 7}"
                row[gate.COLUMNS.index("municipality")] = None if i % 3 == 0 else "São Paulo"
                return tuple(row)
            original = [make_row(i) for i in range(80)]
            other = original.copy()
            other.pop(17)
            other.extend([make_row(3), make_row(8)])
            for table, rows in (("expected_cube", original), ("preview_cube", other)):
                con.executemany(f"INSERT INTO {table} ({names}) VALUES ({params})", rows)
            statements = [gate.multiset_comparison_sql(
                "2022:general", mode=mode,
                expected_relation="expected_cube", preview_relation="preview_cube")
                for mode in ("signed", "except-all")]
            signed, old = [con.execute(sql).fetchone()[0] for sql in statements]
            self.assertEqual(signed, old)
            self.assertGreater(signed, 0)

    def test_invalid_or_missing_inputs_fail_before_opening_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main = root / "tse_analytics.duckdb"
            preview = root / "preview.duckdb"
            main.write_bytes(b"a")
            preview.write_bytes(b"b")
            with self.assertRaisesRegex(gate.GateError, "Missing main, preview"):
                gate.validate_locations(main, preview, root / "missing", "2024:municipal")
            with self.assertRaisesRegex(gate.GateError, "Published Parquet missing"):
                gate.validate_locations(main, preview, root, "2024:municipal")

    def test_symlink_warehouse_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main = root / "original.duckdb"
            other = root / "preview.duckdb"
            main.write_bytes(b"a")
            (root / "alias.duckdb").symlink_to(main)
            other.write_bytes(b"b")
            with self.assertRaisesRegex(gate.GateError, "symlink"):
                gate.validate_locations(root / "alias.duckdb", other, root, "2024:municipal")

    def test_missing_lock_does_not_create_lock_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.build.lock"
            with self.assertRaisesRegex(gate.GateError, "Required stable lock"):
                gate.require_existing_lock(path)
            self.assertFalse(path.exists())

    def _lock_scope_fixture(self, directory: Path):
        main = directory / "tse_analytics.duckdb"
        preview = directory / "tse_event_registry_preview.duckdb"
        physical = directory / "fact_candidate_votes"
        for db in (main, preview):
            db.write_bytes(b"test-only-db-bytes")
            (directory / (db.name + ".build.lock")).touch()
        for year, kind in ((2018, "general"), (2024, "municipal")):
            partition = physical / f"election_type={kind}" / f"year={year}"
            partition.mkdir(parents=True)
            (partition / "data.parquet").write_bytes(b"test-only-parquet-bytes")
        return main, preview, physical

    def test_older_partition_without_publish_lock_does_not_block_selected_cycle(self):
        # Both warehouses and 2024 have locks. The unrelated older 2018
        # partition is published but predates the physical sidecar protocol.
        # Stop at sha256 so no real DuckDB read or fixture SQL is necessary.
        with tempfile.TemporaryDirectory() as tmp:
            main, preview, physical = self._lock_scope_fixture(Path(tmp))
            selected_lock = physical / "election_type=municipal/year=2024/.publish.lock"
            selected_lock.touch()
            historic_lock = physical / "election_type=general/year=2018/.publish.lock"
            with mock.patch.dict(sys.modules, {"duckdb": mock.MagicMock()}), \
                 mock.patch.object(gate, "sha256", side_effect=gate.GateError("reached hash stage")):
                with self.assertRaisesRegex(gate.GateError, "reached hash stage"):
                    gate.audit(main, preview, physical, "2024:municipal",
                               expected_main_sha256="unused", lock_timeout=2,
                               memory_limit="4GB", max_rows=1_000_000)
            self.assertFalse(historic_lock.exists())
            self.assertTrue(selected_lock.exists())

    def test_selected_publish_lock_is_still_required_and_never_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            main, preview, physical = self._lock_scope_fixture(Path(tmp))
            selected_lock = physical / "election_type=municipal/year=2024/.publish.lock"
            with mock.patch.dict(sys.modules, {"duckdb": mock.MagicMock()}):
                with self.assertRaisesRegex(gate.GateError, "Required stable lock missing"):
                    gate.audit(main, preview, physical, "2024:municipal",
                               expected_main_sha256="unused", lock_timeout=2,
                               memory_limit="4GB", max_rows=1_000_000)
            self.assertFalse(selected_lock.exists())

    def test_preflight_declines_same_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "main.duckdb"
            db.write_bytes(b"x")
            with self.assertRaisesRegex(gate.GateError, "distinct"):
                gate.validate_locations(db, db, root, "2024:municipal")

    def test_cli_requires_all_identity_flags(self):
        parser = gate.main
        with self.assertRaises(SystemExit) as e:
            parser(["--cycle", "2024:municipal"])
        self.assertEqual(e.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
