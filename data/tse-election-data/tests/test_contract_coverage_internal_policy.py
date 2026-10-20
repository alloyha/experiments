"""Regression tests for reviewed diagnostic/graph classifications (CCC v1.2).

No warehouse data, dbt execution or GG certificate access.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from scripts.check_contract_coverage import discover

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'candidate_result_coverage': ('gold/reconciliation', 'select distinct', "ref('fact_candidate_votes')"),
    'candidate_tally_coverage_gaps': ('gold/reconciliation', 'missing_candidate_result_coverage', "ref('candidate_result_coverage')"),
    'candidate_tally_reconciliation': ('gold/reconciliation', 'nominal_valid_delta', "ref('candidate_result_coverage')"),
    'party_tally_coverage_gaps': ('gold/reconciliation', 'converted_vote_overlap', "ref('fact_party_votes')"),
    'party_tally_reconciliation': ('gold/reconciliation', 'total_valid_delta', "ref('party_tally_coverage_gaps')"),
    'graph_edges': ('gold/graph', "'graph_role': 'edges'", "materialized='view'"),
    'graph_vertices': ('gold/graph', "'graph_role': 'vertices'", "materialized='view'"),
}


class CccInternalDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resources = discover(ROOT)
        cls.policy = yaml.safe_load((ROOT / 'contracts/coverage_policy.yml').read_text())
        cls.registry = yaml.safe_load((ROOT / 'contracts/odcs/registry.yaml').read_text())

    def test_all_seven_internal_resources_remain_explicit(self):
        for name in EXPECTED:
            with self.subTest(model=name):
                decision = self.policy['resources'].get(f'model:{name}')
                self.assertIsInstance(decision, dict)
                self.assertEqual(set(decision), {'visibility', 'reason'})
                self.assertEqual(decision['visibility'], 'internal')
                self.assertGreaterEqual(len(decision['reason'].strip()), 20)

    def test_reviewed_sql_structure_has_not_changed_silently(self):
        for name, (folder, *tokens) in EXPECTED.items():
            with self.subTest(model=name):
                record = self.resources[f'model:{name}']
                self.assertEqual(record.layer, 'gold')
                path = ROOT / 'tse_dbt/models' / folder / f'{name}.sql'
                self.assertEqual(record.origin, path.relative_to(ROOT).as_posix())
                source = path.read_text(encoding='utf-8')
                for token in tokens:
                    self.assertIn(token, source)

    def test_no_internal_diagnostic_is_also_published_odcs(self):
        names = {entry['name'] for entry in self.registry['contracts']}
        for name in EXPECTED:
            self.assertNotIn(name, names)


if __name__ == '__main__':
    unittest.main()
