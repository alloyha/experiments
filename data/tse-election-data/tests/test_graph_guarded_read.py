"""GG-009 foundation: cooperative locking, fail-closed serving, no global lock."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import graph_guarded_read as guarded


class GraphReadGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / 'warehouse.duckdb'
        self.db.write_bytes(b'synthetic')
        self.source = self.root / 'facts'
        self.part = self.source / 'election_type=municipal/year=2024'
        self.part.mkdir(parents=True)
        self.certs = self.root / 'certs'
        self.cycle = '2024:municipal'

    @staticmethod
    def run_exclusive_probe(path):
        script = '''import fcntl,sys\nwith open(sys.argv[1], 'a+b') as f:\n try:\n  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n except BlockingIOError:\n  sys.exit(7)\n'''
        return subprocess.run([sys.executable, '-c', script, str(path)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode

    def test_shared_locks_block_existing_writer_lock_inodes(self):
        warehouse = self.db.with_name(self.db.name + '.build.lock')
        partition = self.part / '.publish.lock'
        with guarded.graph_read_locks(self.db, self.source, self.cycle, timeout=0):
            self.assertEqual(self.run_exclusive_probe(warehouse), 7)
            self.assertEqual(self.run_exclusive_probe(partition), 7)
        self.assertEqual(self.run_exclusive_probe(warehouse), 0)
        self.assertEqual(self.run_exclusive_probe(partition), 0)

    def test_existing_writer_prevents_read(self):
        lock = self.part / '.publish.lock'
        with lock.open('a+b') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                with self.assertRaisesRegex(TimeoutError, 'graph read lock'):
                    with guarded.graph_read_locks(self.db, self.source, self.cycle, timeout=0):
                        self.fail('read acquired lock held by writer')
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def test_nonexistent_source_does_not_create_partition(self):
        other = self.source / 'election_type=general/year=2026'
        with self.assertRaisesRegex(guarded.GuardError, 'partition does not exist'):
            with guarded.graph_read_locks(self.db, self.source, '2026:general', timeout=0):
                pass
        self.assertFalse(other.exists())

    def test_invalid_query_rejected_before_read(self):
        for opts in ({'relation': 'raw_sql'}, {'relation': 'edges', 'limit': 0},
                     {'relation': 'edges', 'limit': 1001}, {'relation': 'edges', 'kind': ''}):
            values = {'relation': 'edges', **opts}
            with self.subTest(values=values), patch.object(guarded.freshness, 'snapshot') as snap:
                with self.assertRaises(ValueError):
                    guarded.read_graph(db=self.db, source_root=self.source,
                                      attestations_dir=self.certs,
                                      cycle=self.cycle, **values)
                snap.assert_not_called()

    def test_missing_certificate_denies_without_expensive_scans(self):
        with patch.object(guarded.freshness, 'snapshot') as scan:
            with self.assertRaisesRegex(guarded.GuardError, 'Missing active graph certificate'):
                guarded.read_graph(db=self.db, source_root=self.source,
                                   attestations_dir=self.certs, cycle=self.cycle,
                                   relation='edges', lock_timeout=0)
            scan.assert_not_called()

    def test_different_warehouses_have_independent_locks(self):
        other = self.root / 'another.duckdb'
        other.write_bytes(b'other')
        with guarded.graph_read_locks(self.db, self.source, self.cycle, timeout=0):
            # Warehouse locks remain separate. The shared partition lock is
            # also shared, so independent read operations can coexist.
            with guarded.graph_read_locks(other, self.source, self.cycle, timeout=0):
                pass

    def test_issuer_cannot_revoke_while_reader_holds_locks(self):
        """The authority survives a timed-out certifier without a SQL scan."""
        fresh = guarded.freshness
        original = fresh.issue_certificate(self.certs, self.db, self.source,
                                           self.cycle, {'snapshot': 'old'})
        before = original.read_bytes()
        args = ['--db', str(self.db), '--source-root', str(self.source),
                '--attestations-dir', str(self.certs), '--cycle', self.cycle,
                '--lock-timeout', '0', 'certify']
        with guarded.graph_read_locks(self.db, self.source, self.cycle, timeout=0):
            with patch.object(fresh, 'snapshot') as scan:
                self.assertEqual(fresh.main(args), 1)
                scan.assert_not_called()
        self.assertEqual(original.read_bytes(), before)
        self.assertFalse((self.certs / 'revoked').exists())

    def test_issuer_exclusive_locks_block_new_reader(self):
        """Read acquisition cannot overlap a cooperating certifier."""
        with guarded.freshness.graph_certification_locks(
                self.db, self.source, self.cycle, timeout=0):
            with self.assertRaisesRegex(TimeoutError, 'graph read lock'):
                with guarded.graph_read_locks(self.db, self.source,
                                              self.cycle, timeout=0):
                    self.fail('reader bypassed exclusive certificate lock')

    def test_sigkill_certifier_releases_both_os_locks(self):
        """Abrupt process death cannot strand the two flock inodes."""
        marker = self.root / 'issuer-acquired'
        script = '''import sys, time
from pathlib import Path
from scripts.graph_freshness import graph_certification_locks
with graph_certification_locks(Path(sys.argv[1]), Path(sys.argv[2]),
                               '2024:municipal', timeout=5):
    Path(sys.argv[3]).write_text('locked')
    time.sleep(30)
'''
        issuer = subprocess.Popen(
            [sys.executable, '-c', script, str(self.db), str(self.source), str(marker)],
            cwd=str(guarded.freshness.ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 6
            while not marker.exists() and issuer.poll() is None and time.monotonic() < deadline:
                time.sleep(.025)
            if not marker.exists():
                out, err = issuer.communicate(timeout=5)
                self.fail(f'issuer did not acquire locks: {out} {err}')
            with self.assertRaises(TimeoutError):
                with guarded.graph_read_locks(self.db, self.source,
                                              self.cycle, timeout=0):
                    pass
        finally:
            if issuer.poll() is None:
                issuer.kill()
            issuer.communicate(timeout=5)
        with guarded.graph_read_locks(self.db, self.source, self.cycle, timeout=0):
            pass

    def _fake_scoped_query(self, *, before: dict, after: dict, mutate_certificate=False):
        """Exercise post-read guard without needing a physical DuckDB scan."""
        from unittest.mock import MagicMock
        import types
        self.certs.mkdir(exist_ok=True)
        cert_file = self.certs / '2024-municipal.json'
        cert_file.write_text('{"test": 1}')
        published = self.part / 'data.parquet'
        published.write_bytes(b'synthetic source fixture')
        cursor = MagicMock()
        cursor.description = [('edge_type', 'VARCHAR')]
        cursor.fetchall.return_value = [('VOTED_FOR_CANDIDATE',)]
        connection = MagicMock()
        connection.execute.return_value = cursor
        connection.__enter__.return_value = connection
        fake_duckdb = types.SimpleNamespace(connect=lambda _: connection)
        original_sys_module = sys.modules.get('duckdb')
        sys.modules['duckdb'] = fake_duckdb
        if mutate_certificate:
            def capture(*args, **kwargs):
                cert_file.write_text('{"test": 2}')
        else:
            def capture(*args, **kwargs):
                return None
        try:
            with patch.object(guarded.freshness, 'snapshot', side_effect=[before, after]), \
                 patch.object(guarded.freshness, 'audit_certificate'), \
                 patch.object(guarded, 'selected_publication', return_value=(published, {'generation': 1})), \
                 patch.object(guarded, 'setup_scoped_views', side_effect=capture):
                return guarded.read_graph(db=self.db, source_root=self.source,
                                          attestations_dir=self.certs, cycle=self.cycle,
                                          relation='edges', limit=1, lock_timeout=0)
        finally:
            if original_sys_module is None:
                sys.modules.pop('duckdb', None)
            else:
                sys.modules['duckdb'] = original_sys_module

    def test_post_read_fingerprint_change_denies_results(self):
        with self.assertRaisesRegex(guarded.GuardError, 'changed between'):
            self._fake_scoped_query(before={'hash': 'one'}, after={'hash': 'two'})

    def test_certificate_rotation_during_query_denies_results(self):
        with self.assertRaisesRegex(guarded.GuardError, 'certificate changed'):
            self._fake_scoped_query(before={'hash': 'one'}, after={'hash': 'one'},
                                    mutate_certificate=True)

    def test_post_read_recheck_allows_stable_result(self):
        result = self._fake_scoped_query(before={'hash': 'one'}, after={'hash': 'one'})
        self.assertEqual(len(result['rows']), 1)
        self.assertEqual(result['rows'][0]['edge_type'], 'VOTED_FOR_CANDIDATE')

    def test_real_cross_process_certified_read(self):
        """Runs only when project DuckDB is installed; distinct proof subprocess."""
        try:
            import duckdb
            from test_gold_graph_invariants import GoldGraphTests
            from scripts.certify_gold_graph_cycle import SCOPED_SOURCES, quote_sql
        except ImportError:
            self.skipTest('DuckDB provided by project .venv')
        fixture = GoldGraphTests('test_graph_projection_conserves_relationships')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        disk_db = self.root / 'real.duckdb'
        source_path = self.part / 'data.parquet'
        fixture.db.execute(f'ATTACH {quote_sql(str(disk_db))} AS evidence')
        try:
            for table in SCOPED_SOURCES:
                fixture.db.execute(f'CREATE TABLE evidence.{table} AS SELECT * FROM main.{table}')
        finally:
            fixture.db.execute('DETACH evidence')
        fixture.db.execute(
            f'COPY (SELECT * EXCLUDE (cycle_id) FROM fact_candidate_votes) '
            f'TO {quote_sql(str(source_path))} (FORMAT PARQUET)')
        base = [sys.executable, str(guarded.freshness.ROOT / 'scripts/graph_freshness.py'),
                '--db', str(disk_db), '--source-root', str(self.source),
                '--attestations-dir', str(self.certs), '--cycle', self.cycle,
                '--memory-limit', '512MB']
        certified = subprocess.run(base + ['certify'], cwd=guarded.freshness.ROOT,
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(certified.returncode, 0, certified.stdout + certified.stderr)
        result = guarded.read_graph(db=disk_db, source_root=self.source,
                                    attestations_dir=self.certs, cycle=self.cycle,
                                    relation='edges', kind='VOTED_FOR_CANDIDATE',
                                    limit=3, memory_limit='512MB')
        self.assertTrue(result['rows'])
        self.assertLessEqual(len(result['rows']), 3)
        self.assertTrue(all(row['edge_type'] == 'VOTED_FOR_CANDIDATE'
                            for row in result['rows']))
        with duckdb.connect(str(disk_db)) as connection:
            connection.execute('UPDATE dim_party SET party_name = party_name || \'_x\'')
        with self.assertRaisesRegex(Exception, 'freshness failure'):
            guarded.read_graph(db=disk_db, source_root=self.source,
                               attestations_dir=self.certs, cycle=self.cycle,
                               relation='edges', limit=1, memory_limit='512MB')


if __name__ == '__main__':
    unittest.main()
