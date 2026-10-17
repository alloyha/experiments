"""Real DuckDB integration of V2 snapshot adoption and existing freshness format.

Works only on disposable synthetic election fixtures; never opens main Gold.
Proof-chain records are synthetic test fixtures, not genuine electoral tests.
"""
from __future__ import annotations

from pathlib import Path
import unittest

from scripts import adopt_candidate_cube_snapshot_v2 as adopt
from scripts import consolidate_candidate_cube_transfer as evidence
from scripts import promote_candidate_cube_v2 as v2
from scripts import candidate_cube_freshness as freshness
import test_promote_candidate_cube_v2_real_duckdb as fixtures

try:
    import duckdb
except ImportError:
    duckdb = None


@unittest.skipUnless(duckdb is not None, 'requires real DuckDB')
class RealDuckDBSnapshotAdoptionTests(unittest.TestCase):
    def test_actual_freshness_certificates_and_recovery_invalidates(self):
        with fixtures.real_v2_fixture() as (f, _old_approval):
            # Ensure the actual candidate_cube_freshness module can fingerprint
            # all published dimensions and a true 29-column persisted cube.
            with duckdb.connect(str(f.clone)) as con:
                for name in ('dim_candidate', 'dim_geography'):
                    con.execute(f'CREATE TABLE {name} (cycle_id VARCHAR, label VARCHAR)')
                    con.executemany(f'INSERT INTO {name} VALUES (?, ?)',
                                    [(c, c) for c in fixtures.CYCLES])
                for n in range(27):
                    con.execute('ALTER TABLE cube_candidate_performance_municipality '
                                f'ADD COLUMN extra_{n:02d} INTEGER DEFAULT {n}')
            f.stage_data['output_sha256'] = evidence.digest(f.clone)
            fixtures.dump(f.stage, f.stage_data)
            f.contract_data['clone_sha256'] = evidence.digest(f.clone)
            f.contract_data['stage_report_sha256'] = evidence.digest(f.stage)
            fixtures.dump(f.contracts, f.contract_data)
            approval = f.dir / 'revised-approval.json'
            evidence.write_create_only(approval, f.run_gate())
            v2.prepare(main=f.main, approval_path=approval,
                       approval_sha=evidence.digest(approval), timeout=3)
            with self.assertRaisesRegex(adopt.AdoptionError, 'phase'):
                adopt.adopt(main=f.main, cert_dir=f.dir/'attestations', timeout=3,
                            ack_quiescent=True, confirm_adopt=True)
            v2.activate(main=f.main, timeout=3)  # Real DuckDB live smoke
            clone_hash = evidence.digest(f.clone)
            certs = f.dir / 'attestations'
            issued = adopt.adopt(main=f.main, cert_dir=certs, timeout=3,
                                 ack_quiescent=True, confirm_adopt=True)
            self.assertEqual(issued['cycle_count'], 5)
            self.assertEqual(evidence.digest(f.main), clone_hash)
            checked = freshness.audit_freshness(f.main, f.root, certs, set(fixtures.CYCLES))
            self.assertEqual(set(checked['fresh_cycles']), set(fixtures.CYCLES))
            v2.recover(main=f.main, timeout=3)
            with self.assertRaises(freshness.FreshnessError):
                freshness.audit_freshness(f.main, f.root, certs, set(fixtures.CYCLES))


if __name__ == '__main__':
    unittest.main()
