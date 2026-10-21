"""CCC B2: public party snapshot ID, isolated ODCS ownership and coverage."""
from __future__ import annotations
import unittest
from pathlib import Path
import yaml
from scripts.check_contract_coverage import audit
from scripts.check_odcs_contract_registry import check_registry
from scripts.check_odcs_contract_version import check_contract_version

ROOT=Path(__file__).resolve().parents[1]
GRAIN=['election_year','election_type','election_code','party_number']

class PublicPartyContract(unittest.TestCase):
    def test_registry_and_baseline(self):
        names=check_registry(ROOT/'contracts/odcs/registry.yaml',ROOT)
        self.assertIn('dim_party',names)
        check_contract_version(ROOT/'contracts/odcs/baselines/dim_party.v1.0.0.odcs.yaml',
                               ROOT/'contracts/odcs/dim_party.odcs.yaml',
                               contract_name='dim_party')
        doc=yaml.safe_load((ROOT/'contracts/odcs/dim_party.odcs.yaml').read_text())
        self.assertEqual(doc['id'],'urn:datacontract:tse:gold:dim_party')
        self.assertEqual(doc['version'],'1.0.0')
        self.assertEqual(doc['schema'][0]['name'],'dim_party')
        cols={x['name']:x for x in doc['schema'][0]['properties']}
        self.assertEqual([x for x,v in cols.items() if v.get('primaryKey')],['party_id'])
        self.assertTrue(all(x.get('required') for x in cols.values()))

    def test_exact_owner_grain_and_no_duplicate_schema(self):
        model=yaml.safe_load((ROOT/'tse_dbt/models/gold/dimensions/dim_party.yml').read_text())['models'][0]
        shared=yaml.safe_load((ROOT/'tse_dbt/models/schema.yml').read_text())
        self.assertNotIn('dim_party',{x['name'] for x in shared['models']})
        self.assertEqual(model['config']['meta']['datacontract_cli']['contract_id'],
                         'urn:datacontract:tse:gold:dim_party')
        tests=model.get('tests',[])+model.get('data_tests',[])
        grains=[x['dbt_utils.unique_combination_of_columns']['arguments']['combination_of_columns']
                for x in tests if isinstance(x,dict) and 'dbt_utils.unique_combination_of_columns' in x]
        self.assertEqual(grains,[GRAIN])
        cols={x['name']:x for x in model['columns']}
        self.assertIn('unique',[next(iter(x)) if isinstance(x,dict) else x
                                for x in cols['party_id'].get('data_tests',[])])

    def test_coverage_still_advisory(self):
        report=audit(ROOT)
        stats=report['stats']
        self.assertEqual(stats['total'],42)
        self.assertEqual(stats['public'],5)
        self.assertEqual(stats['internal'],22)
        self.assertEqual(stats['unclassified'],15)
        self.assertEqual(stats['invalid_contracts'],0)
        self.assertEqual(stats['unmapped_semantic_obligations'],0)
        self.assertEqual(report['status'],'FAIL')
        self.assertNotIn('model:dim_party',report['unclassified'])

if __name__=='__main__':unittest.main()
