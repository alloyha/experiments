"""Regression tests: election-local party_id and downstream semantic consumers.

These tests evaluate the actual dbt model SQL against a small DuckDB fixture;
no real warehouse, input download, or GG attestation is involved.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import duckdb
import jinja2
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "tse_dbt/models"


def render(path: Path) -> str:
    environment = jinja2.Environment(undefined=jinja2.StrictUndefined)
    return environment.from_string(path.read_text(encoding="utf-8")).render(
        config=lambda *args, **kwargs: "",
        ref=lambda name: name,
        is_incremental=lambda: False,
        incremental_partition_predicate=lambda *args: "TRUE",
    )


class CanonicalPartyIdentity(unittest.TestCase):
    def setUp(self):
        self.db = duckdb.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.execute("""
            create table silver_party_votes_munzona (
                election_year integer, election_type varchar,
                election_scope varchar, election_code varchar,
                party_number varchar, party varchar, party_name varchar,
                generated_at timestamp
            )
        """)
        self.db.execute("""
            insert into silver_party_votes_munzona values
              (2024,'municipal','municipal','620','45','OLD','Old Name', '2024-10-06'),
              (2024,'municipal','municipal','620','45','NEW','New Name', '2024-10-07'),
              (2024,'municipal','municipal','620','18','P18','Party 18', '2024-10-07'),
              (2024,'municipal','municipal','621','45','P45','Party 45', '2024-10-07'),
              (2026,'general','federal_state','700','45','P45','Party 45', '2026-10-10')
        """)
        self.db.execute(
            "create view dim_party as " + render(MODELS / "gold/dimensions/dim_party.sql")
        )

    def test_party_id_is_exactly_event_plus_party_number(self):
        rows = self.db.execute("""
            select election_id, party_number, party_id, party_name
            from dim_party order by election_id, party_number
        """).fetchall()
        self.assertEqual(rows, [
            ('2024:municipal:620', '18', '2024:municipal:620:18', 'Party 18'),
            ('2024:municipal:620', '45', '2024:municipal:620:45', 'New Name'),
            ('2024:municipal:621', '45', '2024:municipal:621:45', 'Party 45'),
            ('2026:general:700', '45', '2026:general:700:45', 'Party 45'),
        ])
        stats = self.db.execute("""
            select count(*), count(distinct party_id), count(distinct (election_id, party_number))
            from dim_party
        """).fetchone()
        self.assertEqual(stats, (4, 4, 4))

    def test_consumer_semantic_cube_propagates_canonical_party_id(self):
        self.db.execute("""create table fact_party_votes (
            cycle_id varchar, election_id varchar, election_year integer,
            election_type varchar, election_scope varchar, election_code varchar,
            round_number integer, uf varchar, municipality_code varchar,
            office_code varchar, is_transit_vote boolean, party_number varchar,
            zone integer, nominal_valid_votes bigint, legend_valid_votes bigint,
            total_legend_valid_votes bigint, party_valid_votes bigint,
            generated_at timestamp, office_scope varchar
        )""")
        self.db.execute("""insert into fact_party_votes values
            ('2024:municipal','2024:municipal:620',2024,'municipal','municipal',
             '620',2,'GO','5208707','11',false,'45',1,40,3,3,43,'2024-10-07','municipal'),
            ('2024:municipal','2024:municipal:620',2024,'municipal','municipal',
             '620',2,'GO','5208707','11',false,'45',2,60,6,6,66,'2024-10-07','municipal')
        """)
        self.db.execute("""create table dim_geography (
            cycle_id varchar, uf varchar, municipality_code varchar,
            municipality varchar
        )""")
        self.db.execute("insert into dim_geography values ('2024:municipal','GO','5208707','Goiânia')")
        cube_path = MODELS / "semantic/cube_party_performance_municipality.sql"
        self.db.execute("create view party_cube as " + render(cube_path))
        self.assertEqual(self.db.execute("""
            select party_id, nominal_valid_votes, party_valid_votes
            from party_cube
        """).fetchall(), [('2024:municipal:620:45', 100, 109)])

        # The other declared consumer projects the same key from dim_party.
        other_sql = (MODELS / "semantic/party_performance.sql").read_text()
        self.assertIn("d.party_id", other_sql)
        self.assertIn("ref('dim_party')", other_sql)

    def test_domain_contract_documents_non_persistent_party_identity(self):
        policy = yaml.safe_load((ROOT / "tse_dbt/model_contracts.yml").read_text())
        party = policy["identity_conformance"]["party_id"]
        self.assertEqual(party["expression"], "election_id:party_number")
        self.assertEqual(party["grain"], "election_party_snapshot")
        self.assertIn("dim_party", party["applies_to"])
        self.assertIn("party_performance", party["applies_to"])
        self.assertIn("cube_party_performance_municipality", party["applies_to"])


if __name__ == "__main__":
    unittest.main()
