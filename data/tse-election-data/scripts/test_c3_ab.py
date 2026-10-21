import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from c3_policy import selection,require_protected_run
from verify_c3_sources import attest

class PolicyTests(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(selection(['run','--select','silver_tally_munzona']),{'silver_tally_munzona'})
    def test_ambiguous(self):
        for v in ('+silver_tally_munzona','tag:silver','fact_tally_munzona+'):
            with self.assertRaises(ValueError):selection(['build','--select',v])
    def test_fail_closed(self):
        with self.assertRaises(ValueError):require_protected_run(['build'],enabled=False,configured=False)
        require_protected_run(['run','--select','fact_tally_munzona'],enabled=True,configured=True)
    def test_nonwriters(self):require_protected_run(['compile'],enabled=False,configured=False)

class SourceTests(unittest.TestCase):
    def _fixture(self,root):
        meta=root/'_metadata';meta.mkdir()
        rows={};index=[]
        for year,typ in ((2018,'general'),(2020,'municipal'),(2022,'general'),(2024,'municipal'),(2026,'general')):
            zrel=f'{year}/source.zip';orel=f'{year}/extracted/file.csv'
            z=root/zrel;z.parent.mkdir(parents=True)
            with zipfile.ZipFile(z,'w') as archive: archive.writestr('file.csv',f'year\n{year}\n')
            import hashlib
            sha=hashlib.sha256(z.read_bytes()).hexdigest()
            out=root/orel;out.parent.mkdir(parents=True);out.write_text(f'year\n{year}\n')
            rows[str(year)]={'year':year,'election_type':typ,'domain':'vote_result','resource_name':'Detalhe da apuração por município e zona','resource_id':str(year),'selection_complete':True,'source_object':zrel,'source_sha256':sha,'selected_members':['file.csv'],'extracted_objects':[orel]}
            index.append({'year':year,'election_type':typ,'domain':'vote_result','resource_name':'Detalhe da apuração por município e zona','resource_id':str(year),'source_sha256':sha,'object':orel})
        (meta/'resource_state.json').write_text(json.dumps(rows))
        (meta/'current_objects.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in index))
    def test_source_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self._fixture(p)
            self.assertEqual(attest(p)['status'],'PASS')
            (p/'2022/extracted/file.csv').write_text('tampered')
            with self.assertRaises(ValueError):attest(p)
    def test_missing_index(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self._fixture(p)
            (p/'_metadata/current_objects.jsonl').write_text('')
            with self.assertRaises(ValueError):attest(p)
if __name__=='__main__':unittest.main()
