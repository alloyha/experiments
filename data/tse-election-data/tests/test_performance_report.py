from __future__ import annotations
import json, subprocess, sys, tempfile, unittest
from pathlib import Path

class PerformanceReportTests(unittest.TestCase):
    def test_orders_nodes_by_execution_time(self):
        root=Path(__file__).resolve().parents[1]
        script=root/'scripts/report_build_performance.py'
        payload={'results':[
            {'unique_id':'model.tse_analytics.fast','status':'success','execution_time':1.0},
            {'unique_id':'test.tse_analytics.slow','status':'pass','execution_time':4.0},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); src=tmp/'run_results.json'; out=tmp/'report.json'
            src.write_text(json.dumps(payload), encoding='utf-8')
            subprocess.run([sys.executable,str(script),'--run-results',str(src),'--output',str(out)],cwd=root,check=True)
            report=json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(report['nodes'][0]['name'],'slow')
            self.assertEqual(report['total_node_seconds'],5.0)

if __name__=='__main__': unittest.main()
