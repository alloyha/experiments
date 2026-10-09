from __future__ import annotations

import unittest
from pathlib import Path


class AnalyticalProductTests(unittest.TestCase):
    def test_products_and_decision_metrics_exist(self):
        root = Path(__file__).resolve().parents[1]
        semantic = root / "tse_dbt/models/semantic"

        expected = {
            "candidate_ranking.sql",
            "election_results.sql",
        }
        self.assertTrue(expected.issubset({path.name for path in semantic.glob("*.sql")}))
        self.assertFalse((semantic / "municipality_election_results.sql").exists())

        ranking = (semantic / "candidate_ranking.sql").read_text(encoding="utf-8")
        for token in (
            "candidate_rank",
            "candidate_nominal_vote_share",
            "is_top_ranked",
            "votes_behind_previous",
            "lead_over_next_votes",
        ):
            self.assertIn(token, ranking)

        self.assertNotIn("winner", ranking.lower())
        self.assertNotIn("municipalities_won", ranking)

    def test_election_results_does_not_claim_elected_status(self):
        root = Path(__file__).resolve().parents[1]
        sql = (root / "tse_dbt/models/semantic/election_results.sql").read_text(encoding="utf-8")
        self.assertIn("top_candidate_id", sql)
        self.assertIn("second_candidate_id", sql)
        self.assertIn("lead_margin_votes", sql)
        self.assertNotIn("winner", sql.lower())
        self.assertNotIn("elected", sql.lower())


if __name__ == "__main__":
    unittest.main()
