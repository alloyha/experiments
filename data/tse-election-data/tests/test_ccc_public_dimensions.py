"""CCC public dimension wave: baseline, owners, grain, and fail-closed decisions."""
from __future__ import annotations

import unittest
from pathlib import Path
import yaml
from scripts.check_odcs_contract_version import check_contract_version
from scripts.check_odcs_contract_registry import check_registry
from scripts.check_contract_coverage import audit

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={
 'dim_candidate':['election_year','election_type','election_code','candidate_id'],
 'dim_geography':['election_year','election_type','uf','municipality_code'],
}

class PublicDimensionCoverage(unittest.TestCase):
 def test_baselines_owners_grain(self):
  registry=yaml.safe_load((ROOT/'contracts/odcs/registry.yaml').read_bytes())
  self.assertEqual(len(registry['contracts']),4)
  names=check_registry(ROOT/'contracts/odcs/registry.yaml',ROOT)
  self.assertEqual(names, sorted(['dim_election_cycle','dim_election',*EXPECTED]))
  policy=yaml.safe_load((ROOT/'contracts/coverage_policy.yml').read_bytes())['resources']
  domain=yaml.safe_load((ROOT/'tse_dbt/model_contracts.yml').read_bytes())['models']
  shared=yaml.safe_load((ROOT/'tse_dbt/models/schema.yml').read_bytes())
  for name,grain in EXPECTED.items():
   with self.subTest(name=name):
    record=next(x for x in registry['contracts'] if x['name']==name)
    check_contract_version(ROOT/record['baseline'],ROOT/record['contract'],contract_name=name)
    metadata=yaml.safe_load((ROOT/record['dbt_metadata']).read_bytes())['models'][0]
    self.assertEqual(metadata['name'],name)
    self.assertEqual(metadata['config']['meta']['datacontract_cli']['contract_id'],f'urn:datacontract:tse:gold:{name}')
    self.assertEqual(policy[f'model:{name}']['visibility'],'public')
    self.assertEqual(policy[f'model:{name}']['grain'],grain)
    self.assertEqual(domain[name]['grain'],grain)
    self.assertNotIn(name,{m['name'] for m in shared['models']})
    tests=metadata.get('tests',[])+metadata.get('data_tests',[])
    composites=[t['dbt_utils.unique_combination_of_columns']['arguments']['combination_of_columns']
     for t in tests if isinstance(t,dict) and 'dbt_utils.unique_combination_of_columns' in t]
    self.assertEqual(composites,[grain])

 def test_other_public_decisions_remain_unresolved(self):
  report=audit(ROOT)
  self.assertEqual(report['stats']['total'],42)
  self.assertEqual(report['stats']['public'],4)
  self.assertEqual(report['stats']['internal'],22)
  self.assertEqual(report['stats']['unclassified'],16)
  self.assertEqual(report['stats']['invalid_contracts'],0)
  self.assertEqual(report['stats']['unmapped_semantic_obligations'],0)
  self.assertEqual(report['status'],'FAIL')  # 16 genuine pending decisions

if __name__=='__main__':
 unittest.main()
