"""Falsify the mandatory staged physical group/zone publication gate.

Only synthetic DuckDB+Parquet bytes under TemporaryDirectory are used.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts import write_candidate_fact_partition as publisher


class ZonePublicationGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="zone-publish-guard-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.db = self.root / "warehouse.duckdb"
        self.facts = self.root / "facts"
        self.part = self.facts / "election_type=general/year=2022"
        self.pointer = self.part / "data.parquet"
        self.seed([(10, 17)])

    def seed(self, values, *, missing_zone=False):
        with duckdb.connect(str(self.db)) as con:
            con.execute("""create or replace table int_candidate_fact_partition as
              select 2022::integer as election_year, 'general'::varchar as election_type,
                     1::integer as election_id, 1::integer as round_number,
                     'GO'::varchar as uf, '5208707'::varchar as municipality_code,
                     '13'::varchar as office_code, false as is_transit_vote,
                     'C1'::varchar as candidate_id,
                     10::integer as zone, 0::integer as nominal_votes
              where false""")
            if missing_zone:
                con.execute('alter table int_candidate_fact_partition drop column zone')
            for zone, votes in values:
                if missing_zone:
                    con.execute("""insert into int_candidate_fact_partition
                        (election_year, election_type, election_id, round_number,
                         uf, municipality_code, office_code, is_transit_vote,
                         candidate_id, nominal_votes)
                        values (2022,'general',1,1,'GO','5208707','13',false,'C1',?)""",
                        [votes])
                else:
                    con.execute("""insert into int_candidate_fact_partition
                        values (2022,'general',1,1,'GO','5208707','13',false,'C1',?,?)""",
                        [zone, votes])

    def publish(self):
        return publisher.publish_partition(self.db, self.facts, 2022, 'general')

    def _snapshot(self):
        return os.readlink(self.pointer) if self.pointer.is_symlink() else None

    def test_exact_unique_and_distinct_zone_rows_publish_with_sha_evidence(self):
        self.seed([(10,17), (20,31)])
        self.publish()
        self.assertTrue(self.pointer.is_symlink())
        generation = self.part / self._snapshot()
        evidence_path = generation.with_suffix('.zone-uniqueness.json')
        self.assertTrue(evidence_path.is_file())
        data = json.loads(evidence_path.read_text())
        self.assertEqual(data['result'], 'PASS')
        self.assertEqual(data['cycle'], '2022:general')
        self.assertEqual(data['scope'], 'candidate_physical_zone_uniqueness_not_graph_freshness')
        self.assertEqual(data['source_rows'], 2)
        self.assertEqual(data['distinct_group_zone_keys'], 2)
        self.assertEqual(data['null_zone_rows'], 0)
        self.assertEqual(data['duplicate_nonnull_zone_rows'], 0)
        self.assertEqual(data['generation'], generation.name)
        self.assertEqual(data['parquet_sha256'], hashlib.sha256(generation.read_bytes()).hexdigest())
        self.assertEqual(data['guard_sql_sha256'], hashlib.sha256(
            publisher.ZONE_UNIQUENESS_SQL.encode()).hexdigest())

    def test_duplicate_nonnull_zone_rejected_before_pointer_swap(self):
        self.publish()
        original = self._snapshot()
        self.seed([(10,1),(10,2)])
        with self.assertRaisesRegex(RuntimeError, 'duplicate_nonnull_zone_rows=1'):
            self.publish()
        self.assertEqual(self._snapshot(), original)
        self.assertFalse(list(self.part.glob('.generation-*.tmp')))

    def test_null_zone_rejected_before_pointer_swap(self):
        self.publish()
        original = self._snapshot()
        self.seed([(None,4)])
        with self.assertRaisesRegex(RuntimeError, 'null_zone_rows=1'):
            self.publish()
        self.assertEqual(self._snapshot(), original)

    def test_missing_zone_column_fails_closed(self):
        self.publish()
        original = self._snapshot()
        self.seed([(None,9)], missing_zone=True)
        with self.assertRaisesRegex(RuntimeError, 'schema missing'):
            self.publish()
        self.assertEqual(self._snapshot(), original)

    def test_distinct_groups_may_share_zone(self):
        self.seed([(10,17)])
        with duckdb.connect(str(self.db)) as con:
            con.execute("""insert into int_candidate_fact_partition values
                (2022,'general',1,1,'GO','5208707','13',false,'C2',10,31)""")
        self.publish()
        data=json.loads((self.part / self._snapshot()).with_suffix('.zone-uniqueness.json').read_text())
        self.assertEqual(data['source_rows'], 2)
        self.assertEqual(data['duplicate_nonnull_zone_rows'], 0)

    def test_staged_guard_failure_never_creates_active_evidence(self):
        self.seed([(10,1),(10,2)])
        with self.assertRaisesRegex(RuntimeError, 'uniqueness rejected'):
            self.publish()
        self.assertFalse(self.pointer.exists())
        self.assertFalse(list(self.part.glob('*.zone-uniqueness.json')))


if __name__ == '__main__':
    unittest.main()
