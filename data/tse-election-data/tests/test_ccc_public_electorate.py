"""CCC C1: public electorate grain, source-backed nullability, isolated ODCS owner."""
from __future__ import annotations
import unittest
from pathlib import Path
import yaml
from scripts.check_odcs_contract_version import check_contract_version
from scripts.check_odcs_contract_registry import check_registry
from scripts.check_contract_coverage import audit

ROOT=Path(__file__).resolve().parents[1]
NAME='fact_electorate_municipality'
GRAIN=['election_year','election_type','uf','municipality_code']

class PublicElectorate(unittest.TestCase):
    def test_registry_and_immutable_contract(self):
        names=check_registry(ROOT/'contracts/odcs/registry.yaml',ROOT)
        self.assertIn(NAME,names)
        old=ROOT/f'contracts/odcs/baselines/{NAME}.v1.0.0.odcs.yaml'
        current=ROOT/f'contracts/odcs/{NAME}.odcs.yaml'
        check_contract_version(old,current,contract_name=NAME)
        contract=yaml.safe_load(current.read_bytes())
        self.assertEqual(contract['version'],'1.0.0')
        self.assertEqual(contract['schema'][0]['name'],NAME)
        self.assertFalse(any(x.get('primaryKey') for x in contract['schema'][0]['properties']))
        self.assertIn('disjoint',contract['description']['purpose'])

    def test_grain_and_owned_metadata(self):
        domain=yaml.safe_load((ROOT/'tse_dbt/model_contracts.yml').read_text())['models']
        self.assertEqual(domain[NAME]['grain'],GRAIN)
        metadata=yaml.safe_load((ROOT/f'tse_dbt/models/gold/facts/{NAME}.yml').read_text())['models'][0]
        shared=yaml.safe_load((ROOT/'tse_dbt/models/schema.yml').read_text())
        self.assertNotIn(NAME,{x['name'] for x in shared['models']})
        self.assertEqual(metadata['config']['meta']['datacontract_cli']['contract_id'],
                         f'urn:datacontract:tse:gold:{NAME}')
        tests=metadata.get('tests',[])+metadata.get('data_tests',[])
        combos=[t['dbt_utils.unique_combination_of_columns']['arguments']['combination_of_columns']
                for t in tests if isinstance(t,dict) and 'dbt_utils.unique_combination_of_columns' in t]
        self.assertEqual(combos,[GRAIN])
        columns={c['name']:c for c in metadata['columns']}
        self.assertIn('electorate',columns)

    def test_coverage_closed_world(self):
        report=audit(ROOT)
        stats=report['stats']
        self.assertEqual(stats['public'],14)
        self.assertEqual(stats['internal'],28)
        self.assertEqual(stats['total'],42)
        self.assertEqual(stats['unclassified'],0)
        self.assertEqual(stats['invalid_contracts'],0)
        self.assertEqual(stats['unmapped_semantic_obligations'],0)
        self.assertNotIn(f'model:{NAME}',report['unclassified'])
        self.assertEqual(report['status'],'PASS')

if __name__=='__main__': unittest.main()
