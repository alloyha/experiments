"""Promotion gate: exact active-source uniqueness and signed multiset assertion.

All DuckDB data is synthetic and disposable. No canonical TSE paths accessed.
"""
from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts import certify_gold_graph_cycle as diagnostic
from scripts.audit_graph_candidate_zone_history import HistoryGuardError
from test_gold_graph_invariants import GoldGraphTests


class CanonicalPromotionTests(unittest.TestCase):
    def setUp(self):
        fixture = GoldGraphTests('test_graph_projection_conserves_relationships')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.synthetic = fixture.db
        temp = tempfile.TemporaryDirectory(prefix='graph-canonical-promotion-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db = self.root / 'test.duckdb'
        self.facts = self.root / 'facts'
        self.partition = self.facts / 'election_type=municipal/year=2024'
        self.partition.mkdir(parents=True)
        self.parquet = self.partition / 'data.parquet'
        escaped = diagnostic.quote_sql(str(self.db))
        self.synthetic.execute(f'ATTACH {escaped} AS persisted')
        try:
            for source in diagnostic.SCOPED_SOURCES:
                self.synthetic.execute(
                    f'CREATE TABLE persisted.{source} AS SELECT * FROM main.{source}'
                )
        finally:
            self.synthetic.execute('DETACH persisted')
        self.export()

    def export(self):
        if self.parquet.exists():
            self.parquet.unlink()
        self.synthetic.execute(
            'COPY (SELECT * EXCLUDE (cycle_id) FROM fact_candidate_votes) '
            f'TO {diagnostic.quote_sql(str(self.parquet))} (FORMAT PARQUET)'
        )

    def check(self):
        return diagnostic.run(argparse.Namespace(
            db=self.db, source_root=self.facts, cycle='2024:municipal',
            memory_limit='512MB', temp_directory=None, execute=True,
            tests=['candidate_vote_conservation'],
        ))

    def test_canonical_graph_has_only_candidate_countstar(self):
        model = (diagnostic.MODELS / 'graph_edges.sql').read_text()
        before, separator, after = model.partition('party_vote_rollup as (')
        self.assertTrue(separator)
        self.assertIn('count(*) as zone_rows, count(*) as zone_count', before)
        self.assertNotIn('count(distinct zone)', before.lower())
        self.assertIn('count(distinct zone) as zone_count', after.lower())

    def test_canonical_assertion_is_signed_not_except_all(self):
        sql = (diagnostic.TESTS / 'assert_graph_candidate_vote_conservation.sql').read_text().lower()
        self.assertIn('sum(multiplicity_delta)', sql)
        self.assertIn('abs(signed_multiplicity)', sql)
        self.assertIn('count(*) as zones', sql)
        self.assertNotIn('except all', sql)

    def test_positive_active_source_passes(self):
        result = self.check()
        self.assertEqual(len(result['assertions']), 1)
        self.assertTrue(result['assertions'][0]['pass'])
        self.assertEqual(result['zone_uniqueness']['source_rows'], 4)

    def test_duplicate_zone_rejected_even_if_rollups_would_agree(self):
        self.synthetic.execute('INSERT INTO fact_candidate_votes SELECT * FROM fact_candidate_votes LIMIT 1')
        self.export()
        with self.assertRaisesRegex(HistoryGuardError, 'duplicate_nonnull_zone_rows=1'):
            self.check()

    def test_null_zone_rejected_before_assertions(self):
        self.synthetic.execute("UPDATE fact_candidate_votes SET zone=NULL WHERE candidate_id='C2'")
        self.export()
        with self.assertRaisesRegex(HistoryGuardError, 'null_zone_rows=1'):
            self.check()

    def test_missing_zone_column_fails_closed(self):
        self.synthetic.execute('CREATE TABLE incomplete AS SELECT * EXCLUDE(zone) FROM fact_candidate_votes')
        self.parquet.unlink()
        self.synthetic.execute('COPY (SELECT * EXCLUDE (cycle_id) FROM incomplete) '
                               f'TO {diagnostic.quote_sql(str(self.parquet))} (FORMAT PARQUET)')
        with self.assertRaisesRegex(HistoryGuardError, 'missing mandatory columns'):
            self.check()


if __name__ == '__main__':
    unittest.main()
