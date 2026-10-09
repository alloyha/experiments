import unittest
from pathlib import Path
class AnalyticalProductTests(unittest.TestCase):
    def test_products_and_decision_metrics_exist(self):
        root=Path(__file__).resolve().parents[1]; sem=root/'tse_dbt/models/semantic'
        expected={'municipality_election_results.sql','candidate_ranking.sql','election_results.sql'}
        self.assertTrue(expected.issubset({p.name for p in sem.glob('*.sql')}))
        ranking=(sem/'candidate_ranking.sql').read_text()
        for token in ('candidate_rank','vote_share','winner','municipalities_won'): self.assertIn(token,ranking)
if __name__=='__main__': unittest.main()
