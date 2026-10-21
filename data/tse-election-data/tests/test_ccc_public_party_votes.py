"""CCC C2: registry/owner, public grain, and vote conservation invariants."""
from __future__ import annotations
import unittest
from pathlib import Path
import yaml
from scripts.check_odcs_contract_version import check_contract_version
from scripts.check_odcs_contract_registry import check_registry
from scripts.check_contract_coverage import audit
ROOT = Path(__file__).resolve().parents[1]
NAME = 'fact_party_votes'
GRAIN = ['election_year','election_type','election_code','round_number','uf',
         'municipality_code','zone','office_code','party_number','is_transit_vote']

class PartyVotesPublicContract(unittest.TestCase):
    def test_odcs_registered_and_frozen(self):
        names = check_registry(ROOT/'contracts/odcs/registry.yaml', ROOT)
        self.assertIn(NAME, names)
        contract=ROOT/f'contracts/odcs/{NAME}.odcs.yaml'
        baseline=ROOT/f'contracts/odcs/baselines/{NAME}.v1.0.0.odcs.yaml'
        check_contract_version(baseline, contract, contract_name=NAME)
        doc=yaml.safe_load(contract.read_bytes())
        self.assertEqual(doc['version'],'1.0.0')
        props={p['name']:p for p in doc['schema'][0]['properties']}
        self.assertFalse(any(p.get('primaryKey') for p in props.values()))
        self.assertEqual(props['is_transit_vote']['logicalType'],'boolean')
        self.assertEqual(props['party_valid_votes']['physicalType'],'bigint')
        self.assertFalse(props['nominal_converted_to_legend_votes']['required'])
        self.assertIn('NOT an additional term',doc['description']['purpose'])

    def test_owned_metadata_and_composite_grain(self):
        meta=yaml.safe_load((ROOT/f'tse_dbt/models/gold/facts/{NAME}.yml').read_bytes())['models'][0]
        shared=yaml.safe_load((ROOT/'tse_dbt/models/schema.yml').read_bytes())
        self.assertNotIn(NAME,{r['name'] for r in shared['models']})
        self.assertEqual(meta['config']['meta']['datacontract_cli']['contract_id'],
                         'urn:datacontract:tse:gold:fact_party_votes')
        tests=meta.get('tests',[])+meta.get('data_tests',[])
        combos=[t['dbt_utils.unique_combination_of_columns']['arguments']['combination_of_columns']
                for t in tests if isinstance(t,dict) and 'dbt_utils.unique_combination_of_columns' in t]
        self.assertEqual(combos,[GRAIN])
        cols={c['name']:c for c in meta['columns']}
        self.assertTrue({'is_transit_vote','party_valid_votes','nominal_converted_to_legend_votes'} <= set(cols))

    def test_business_rule_is_executable(self):
        sql=(ROOT/'tse_dbt/tests/assert_fact_party_votes_public_conservation.sql').read_text()
        self.assertIn("coalesce(nominal_valid_votes, 0) + coalesce(total_legend_valid_votes, 0)",sql)
        self.assertNotIn('nominal_converted_to_legend_votes',sql)
        domain=yaml.safe_load((ROOT/'tse_dbt/model_contracts.yml').read_text())['models']
        self.assertEqual(domain[NAME]['grain'],GRAIN)

    def test_coverage_stays_advisory(self):
        stats=audit(ROOT)['stats']
        self.assertEqual(stats['total'],42)
        self.assertGreaterEqual(stats['public'],7)
        self.assertLessEqual(stats['unclassified'],13)
        self.assertEqual(stats['invalid_contracts'],0)
        self.assertEqual(stats['unmapped_semantic_obligations'],0)

if __name__=='__main__': unittest.main()
