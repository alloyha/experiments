from pathlib import Path
import json
import tempfile
import unittest
from scripts.c3_scope_policy import validate_c3_scope


class C3ScopePolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        p = Path(self.tmp.name)
        self.manifest = p / 'evidence.json'
        self.calendar = p / 'calendar.csv'
        self.root = p / 'raw'
        self.root.mkdir()
        self.calendar.write_text('election_year,election_type\n2022,general\n2024,municipal\n')
        self.manifest.write_text(json.dumps({'cycles':{'2022:general': {}, '2024:municipal':{}}}))
        self.v = {'tse_raw_root':str(self.root),'election_years':[2022,2024],
                  'election_types':['general','municipal'],
                  'incremental_years':[2022,2024],
                  'incremental_election_types':['general','municipal']}

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, v):
        return validate_c3_scope(['run','--vars',json.dumps(v)],
                                 evidence_path=self.manifest,raw_root=self.root,
                                 calendar_csv=self.calendar)

    def test_valid_scope(self):
        self.assertEqual(self.check(self.v), {'2022:general','2024:municipal'})

    def test_changed_root(self):
        v={**self.v,'tse_raw_root':'/tmp/alternate'}
        with self.assertRaisesRegex(ValueError,'differs'):
            self.check(v)

    def test_uncertified_cycle(self):
        self.manifest.write_text(json.dumps({'cycles':{'2022:general':{}}}))
        with self.assertRaisesRegex(ValueError,'uncertified'):
            self.check(self.v)

    def test_missing_scope(self):
        v=dict(self.v)
        del v['incremental_years']
        with self.assertRaisesRegex(ValueError,'explicit'):
            self.check(v)

    def test_wrong_type(self):
        v={**self.v,'incremental_election_types':['general']}
        with self.assertRaisesRegex(ValueError,'mismatches'):
            self.check(v)

    def test_duplicate_vars_flag(self):
        with self.assertRaisesRegex(ValueError,'exactly one'):
            validate_c3_scope(['run','--vars',json.dumps(self.v),'--vars',json.dumps(self.v)],
                              evidence_path=self.manifest,raw_root=self.root,
                              calendar_csv=self.calendar)

if __name__ == '__main__':
    unittest.main()
