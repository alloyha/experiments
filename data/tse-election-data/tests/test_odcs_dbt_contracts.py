"""A minimal anti-drift gate between an ODCS public interface and existing dbt declarations.

Does not invoke dbt, scan real data, edit YAML, or issue freshness certificates.
ODCS schema validation is additionally performed by `datacontract lint` in CI.
"""
from __future__ import annotations

from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'contracts/odcs/dim_election_cycle.odcs.yaml'
DBT_SCHEMA = ROOT / 'tse_dbt/models/schema.yml'
DBT_CYCLE = ROOT / 'tse_dbt/models/gold/dimensions/dim_election_cycle.yml'
GRAIN = ROOT / 'tse_dbt/model_contracts.yml'
MODEL = ROOT / 'tse_dbt/models/gold/dimensions/dim_election_cycle.sql'


class OdcsDbtIntegrationTests(unittest.TestCase):
    def test_public_cycle_contract_tracks_existing_dbt_identity(self):
        contract = yaml.safe_load(CONTRACT.read_text(encoding='utf-8'))
        self.assertEqual(contract['kind'], 'DataContract')
        self.assertEqual(contract['apiVersion'], 'v3.2.0')
        self.assertEqual(contract['status'], 'active')
        self.assertEqual(contract['name'], 'dim_election_cycle')
        (schema,) = contract['schema']
        self.assertEqual(schema['name'], 'dim_election_cycle')

        fields = {f['name']: f for f in schema['properties']}
        self.assertEqual(set(fields), {'cycle_id', 'election_year', 'election_type'})
        self.assertTrue(all(f.get('required') is True for f in fields.values()))
        self.assertEqual([name for name, f in fields.items() if f.get('primaryKey')], ['cycle_id'])
        self.assertEqual(fields['cycle_id']['logicalType'], 'string')
        self.assertEqual(fields['election_year']['logicalType'], 'integer')

        # ODCS owns this one model; an independent file prevents unrelated
        # schema.yml descriptions from being rewritten by the CLI.
        shared = yaml.safe_load(DBT_SCHEMA.read_text(encoding='utf-8'))
        self.assertNotIn(schema['name'], {m['name'] for m in shared['models']})
        metadata = yaml.safe_load(DBT_CYCLE.read_text(encoding='utf-8'))
        self.assertEqual(metadata['version'], 2)
        (dbt,) = metadata['models']
        self.assertEqual(dbt['name'], schema['name'])
        self.assertEqual(dbt['config']['meta']['datacontract_cli']['contract_id'], contract['id'])
        self.assertEqual({col['name'] for col in dbt['columns']}, set(fields))
        test_names = {}
        for column in dbt['columns']:
            checks = [next(iter(test)) if isinstance(test, dict) else test
                      for test in column.get('data_tests', [])]
            self.assertEqual(len(checks), len(set(checks)), column['name'])
            test_names[column['name']] = set(checks)
        self.assertEqual(test_names['cycle_id'], {'not_null', 'unique'})
        self.assertEqual(test_names['election_year'], {'not_null'})
        self.assertEqual(test_names['election_type'], {'not_null'})
        self.assertEqual(len(dbt['tests']), 1)
        self.assertIn('dbt_utils.unique_combination_of_columns', dbt['tests'][0])

        declared = yaml.safe_load(GRAIN.read_text(encoding='utf-8'))['models']['dim_election_cycle']
        self.assertEqual(declared['grain'], ['election_year', 'election_type'])
        sql = MODEL.read_text(encoding='utf-8').lower()
        self.assertIn('select distinct', sql)
        self.assertIn('election_year', sql)
        self.assertIn('election_type', sql)
        self.assertIn('as cycle_id', sql)

    def test_cli_is_preview_only_in_ci(self):
        makefile = (ROOT / 'Makefile').read_text(encoding='utf-8')
        self.assertIn('$(DATACONTRACT) dbt sync', makefile)
        self.assertIn('--dry-run', makefile)
        self.assertIn('--no-prune', makefile)
        self.assertNotIn('--run-tests', makefile)


if __name__ == '__main__':
    unittest.main()
