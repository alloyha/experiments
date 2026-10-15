"""PR54: Gold graph projections and deliberate falsifiers (no real TSE reads).

The fixture SQL is the actual dbt model/test SQL with only refs/config and
partition predicates rendered. CI runs in memory; production scans remain
explicit, cycle-scoped, expensive dbt tests.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from jinja2 import Environment, StrictUndefined

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "tse_dbt/models/gold/graph"
TESTS = ROOT / "tse_dbt/tests"


def render(path: Path) -> str:
    env = Environment(undefined=StrictUndefined)
    return env.from_string(path.read_text(encoding="utf8")).render(
        config=lambda **kwargs: "",
        ref=lambda name: name,
        incremental_partition_predicate=lambda year, kind: (
            f"{year} in (2024) and {kind} in ('municipal')"
        ),
    )


class GraphStaticContracts(unittest.TestCase):
    def test_graph_reads_only_gold_and_no_semantic_cubes(self):
        for name in ("graph_vertices", "graph_edges"):
            sql = (GOLD / f"{name}.sql").read_text()
            refs = re.findall(r"ref\('([^']+)'\)", sql)
            self.assertTrue(refs)
            self.assertTrue(set(refs) <= {
                "dim_candidate", "dim_party", "dim_election", "dim_election_cycle",
                "dim_geography", "fact_candidate_votes", "fact_party_votes",
            })
            self.assertNotIn("cube_candidate_performance", sql)
            self.assertIn("materialized='view'", sql)
            self.assertIn("'expensive'", sql)

    def test_identity_is_typed_and_contains_contest_scope(self):
        vertices = (GOLD / "graph_vertices.sql").read_text()
        edges = (GOLD / "graph_edges.sql").read_text()
        self.assertIn("'contest'", vertices)
        self.assertIn("'VOTED_FOR_CANDIDATE'", edges)
        self.assertIn("'VOTED_FOR_PARTY'", edges)
        for name in ("election_id", "round_number", "municipality_code",
                     "office_code", "is_transit_vote"):
            self.assertIn(f"cast({name} as varchar)", edges)
        self.assertNotIn("+1000", vertices)
        self.assertNotIn("+2000", vertices)

    def test_graph_sql_compiles_as_jinja_without_external_services(self):
        for path in (*GOLD.glob("*.sql"), *TESTS.glob("assert_graph_*.sql")):
            with self.subTest(file=path.name):
                sql = render(path)
                self.assertNotIn("{{", sql)
                self.assertNotIn("{%", sql)
                self.assertIn("select", sql.lower())


class GoldGraphTests(unittest.TestCase):
    def setUp(self):
        try:
            import duckdb
        except ImportError:
            self.skipTest("DuckDB required; present in TSE project environment")
        self.db = duckdb.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.execute("""create table dim_election_cycle (
            cycle_id varchar, election_year integer, election_type varchar)""")
        self.db.execute("""create table dim_election (
            cycle_id varchar, election_id varchar)""")
        self.db.execute("""create table dim_geography (
            cycle_id varchar, uf varchar, municipality_code varchar,
            municipality varchar)""")
        self.db.execute("""create table dim_candidate (
            cycle_id varchar, election_id varchar, candidate_id varchar,
            ballot_name varchar, party_number varchar)""")
        self.db.execute("""create table dim_party (
            cycle_id varchar, election_id varchar, party_number varchar,
            party_name varchar)""")
        self.db.execute("""create table fact_candidate_votes (
            cycle_id varchar, election_id varchar, election_year integer,
            election_type varchar, year integer, round_number integer,
            uf varchar, municipality_code varchar, office_code varchar,
            is_transit_vote boolean, zone integer, candidate_id varchar,
            nominal_valid_votes bigint)""")
        self.db.execute("""create table fact_party_votes (
            cycle_id varchar, election_id varchar, election_year integer,
            election_type varchar, round_number integer, uf varchar,
            municipality_code varchar, office_code varchar,
            is_transit_vote boolean, zone integer, party_number varchar,
            nominal_valid_votes bigint, legend_valid_votes bigint,
            total_legend_valid_votes bigint, party_valid_votes bigint)""")
        self.db.execute("insert into dim_election_cycle values ('2024:municipal',2024,'municipal')")
        self.db.execute("insert into dim_election values ('2024:municipal','2024:municipal:620')")
        self.db.execute("insert into dim_geography values ('2024:municipal','GO','5208707','Goiânia')")
        self.db.execute("""insert into dim_candidate values
            ('2024:municipal','2024:municipal:620','C1','Ana','45'),
            ('2024:municipal','2024:municipal:620','C2','Bia','18')""")
        self.db.execute("""insert into dim_party values
            ('2024:municipal','2024:municipal:620','45','Partido 45'),
            ('2024:municipal','2024:municipal:620','18','Partido 18')""")
        for candidate, zone, transit, votes in (
            ('C1', 1, False, 40), ('C1', 2, False, 60),
            ('C2', 1, False, 25), ('C1', 1, True, 3),
        ):
            self.db.execute("""insert into fact_candidate_votes values
                ('2024:municipal','2024:municipal:620',2024,'municipal',2024,2,
                 'GO','5208707','11',?,?,?,?)""", [transit, zone, candidate, votes])
        for party, zone, nominal, legend, total in (
            ('45', 1, 40, 2, 3), ('45', 2, 60, 5, 6), ('18', 1, 20, 0, 0),
        ):
            self.db.execute("""insert into fact_party_votes values
                ('2024:municipal','2024:municipal:620',2024,'municipal',2,
                 'GO','5208707','11',false,?,?,?,?,?,?)""",
                [zone, party, nominal, legend, total, nominal + total])
        self.db.execute("create view graph_vertices as " + render(GOLD / "graph_vertices.sql"))
        self.db.execute("create view graph_edges as " + render(GOLD / "graph_edges.sql"))

    def violations(self, test: str):
        return self.db.execute(render(TESTS / f"assert_graph_{test}.sql")).fetchall()

    def check_all(self):
        for name in ("node_identity", "edge_identity", "edge_endpoints",
                     "edge_measures", "edge_context", "affiliation_conservation",
                     "candidate_vote_conservation", "party_vote_conservation"):
            with self.subTest(assertion=name):
                self.assertEqual(self.violations(name), [])

    def test_graph_projection_conserves_relationships(self):
        self.check_all()
        rows = self.db.execute("""select edge_type, sum(nominal_valid_votes),
            sum(party_valid_votes) from graph_edges
            where edge_type like 'VOTED_FOR_%' group by 1 order by 1""").fetchall()
        self.assertEqual(rows, [('VOTED_FOR_CANDIDATE', 128, None),
                                ('VOTED_FOR_PARTY', 120, 129)])

    def test_graph_event_round_isolation(self):
        self.db.execute("""insert into fact_candidate_votes values
            ('2024:municipal','2024:municipal:620',2024,'municipal',2024,1,
             'GO','5208707','11',false,1,'C1',5)""")
        self.assertEqual(self.violations("candidate_vote_conservation"), [])
        self.assertEqual(self.violations("node_identity"), [])
        contest_count = self.db.execute("""select count(*) from graph_vertices
            where node_type='contest'""").fetchone()[0]
        self.assertEqual(contest_count, 3)  # two rounds, transit separated

    def test_duplicate_dimension_members_fail_identity(self):
        self.db.execute("insert into dim_candidate select * from dim_candidate where candidate_id='C1'")
        self.assertTrue(self.violations("node_identity"))
        self.assertTrue(self.violations("edge_identity"))

    def test_missing_election_is_detected(self):
        self.db.execute("delete from dim_election")
        self.assertTrue(self.violations("edge_endpoints"))

    def test_missing_dimension_keeps_fact_edges(self):
        self.db.execute("delete from dim_candidate where candidate_id='C2'")
        self.db.execute("delete from dim_geography")
        self.assertEqual(self.violations("candidate_vote_conservation"), [])
        self.assertEqual(self.violations("node_identity"), [])
        self.assertEqual(self.violations("edge_endpoints"), [])
        unmatched = self.db.execute("""select node_type, count(*)
            from graph_vertices where not metadata_matched
            group by 1 order by 1""").fetchall()
        self.assertEqual(unmatched, [('candidate', 1), ('municipality', 1)])

    def test_cross_election_endpoint_same_cycle_is_rejected(self):
        # Same cycle, different election: endpoint FK alone must not authorize it.
        self.db.execute("insert into dim_election values ('2024:municipal','2024:municipal:621')")
        self.db.execute("""insert into dim_candidate values
            ('2024:municipal','2024:municipal:621','OTHER','Other','45')""")
        self.corrupt_edge(
            "edge_type='VOTED_FOR_CANDIDATE' and nominal_valid_votes=25",
            "target_node_id=to_json(list_value('candidate','2024:municipal:621','OTHER'))",
        )
        self.assertEqual(self.violations("edge_endpoints"), [])
        self.assertTrue(self.violations("edge_context"))

    def test_wrong_round_annotation_is_rejected(self):
        # Edge remains connected to the correct contest, but lies about round.
        self.corrupt_edge("edge_type='HAS_CONTEST'", "round_number=99")
        self.assertEqual(self.violations("edge_endpoints"), [])
        self.assertTrue(self.violations("edge_context"))

    def test_party_affiliation_mutation_is_rejected(self):
        # Wrong same-event party: endpoints exist and types match, but the
        # authoritative candidate dimension does not support this association.
        self.corrupt_edge(
            "edge_type='AFFILIATED_WITH' and "
            "source_node_id=to_json(list_value('candidate','2024:municipal:620','C1'))",
            "target_node_id=to_json(list_value('party','2024:municipal:620','18'))",
        )
        self.assertEqual(self.violations("edge_endpoints"), [])
        self.assertEqual(self.violations("edge_context"), [])
        self.assertTrue(self.violations("affiliation_conservation"))

    def corrupt_edge(self, where: str, assignment: str):
        self.db.execute("create table corrupted as select * from graph_edges")
        self.db.execute(f"update corrupted set {assignment} where {where}")
        self.db.execute("drop view graph_edges")
        self.db.execute("create view graph_edges as select * from corrupted")

    def test_candidate_measure_mutation_detected(self):
        self.corrupt_edge("edge_type='VOTED_FOR_CANDIDATE'", "nominal_valid_votes=nominal_valid_votes+1")
        self.assertTrue(self.violations("candidate_vote_conservation"))

    def test_party_measure_mutation_detected(self):
        self.corrupt_edge("edge_type='VOTED_FOR_PARTY'", "party_valid_votes=party_valid_votes+1")
        self.assertTrue(self.violations("party_vote_conservation"))
        self.assertTrue(self.violations("edge_measures"))

    def test_missing_fact_edge_detected(self):
        self.db.execute("create table corrupted as select * from graph_edges where edge_type <> 'VOTED_FOR_CANDIDATE'")
        self.db.execute("drop view graph_edges")
        self.db.execute("create view graph_edges as select * from corrupted")
        self.assertTrue(self.violations("candidate_vote_conservation"))

    def test_nonvote_carries_measures_detected(self):
        self.corrupt_edge("edge_type='CONTAINS_ELECTION'", "nominal_valid_votes=1")
        self.assertTrue(self.violations("edge_measures"))


if __name__ == "__main__":
    unittest.main()
