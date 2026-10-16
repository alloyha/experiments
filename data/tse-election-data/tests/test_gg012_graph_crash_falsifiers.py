"""GG-012: falsify graph publication authority under real worker death.

All DuckDB and Parquet inputs are disposable local fixtures. This deliberately
checks BOTH sides of the atomic hard-link's linearization point. A certificate
linked after successful diagnostics is valid even if the worker dies before ACK;
prior to linking, no certificate can be served.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import graph_freshness as freshness
from scripts import graph_guarded_read as guarded
from scripts import write_candidate_fact_partition as publisher

ROOT = freshness.ROOT
CYCLE = '2024:municipal'

# Child injection acts at REAL filesystem/commit boundaries, not by raising
# Python exceptions in the parent. Nothing touches canonical TSE datasets.
INJECTED_CERTIFIER = r'''
import os, signal, sys, time
from pathlib import Path
from scripts import graph_freshness as graph
stage, marker = sys.argv[1], Path(sys.argv[2])

def crash(where):
    marker.write_text(where, encoding='utf8')
    os.kill(os.getpid(), signal.SIGKILL)

if stage == 'after_revoke':
    original = graph.revoke_certificate
    def injected(*a, **kw):
        original(*a, **kw)
        crash('after_revoke')
    graph.revoke_certificate = injected
elif stage in ('before_link', 'after_link'):
    original = os.link
    def injected(*a, **kw):
        if stage == 'before_link':
            crash('before_link')
        result = original(*a, **kw)
        crash('after_link')
        return result
    graph.os.link = injected
elif stage == 'hold_before_issue':
    original = graph.issue_certificate
    def injected(*a, **kw):
        marker.write_text('certifier_holds_both_locks', encoding='utf8')
        time.sleep(30)
        return original(*a, **kw)
    graph.issue_certificate = injected
else:
    raise SystemExit('unknown fault injection')
raise SystemExit(graph.main(sys.argv[3:]))
'''


class GraphCertificateDurabilityTests(unittest.TestCase):
    """Strict order of durable commit/revocation, without DuckDB scans."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix='gg012-durable-')
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.db = self.root / 'warehouse.duckdb'
        self.db.write_bytes(b'synthetic warehouse identity')
        self.sources = self.root / 'facts'
        self.sources.mkdir()
        self.certs = self.root / 'proofs'

    def test_issue_fsyncs_parent_after_atomic_link(self):
        """A successful return implies directory-persisted authority."""
        events = []
        real_link = os.link
        def linked(*a, **kw):
            result = real_link(*a, **kw)
            events.append(('link', Path(a[1]).parent))
            return result
        def synced(path):
            events.append(('fsync', Path(path)))
        # create=True is intentional: this test turns RED on the pre-GG-012
        # implementation, which has no directory sync helper at all.
        with patch.object(freshness, 'fsync_directory', side_effect=synced, create=True), \
             patch.object(freshness.os, 'link', side_effect=linked):
            path = freshness.issue_certificate(
                self.certs, self.db, self.sources, CYCLE, {'fixture': 'ready'})
        self.assertTrue(path.is_file())
        self.assertIn(('link', self.certs), events)
        link_index = events.index(('link', self.certs))
        self.assertIn(('fsync', self.certs), events[link_index + 1:],
                      'Certificate publication did not durably fsync its parent directory')

    def test_revoke_fsyncs_both_directories_after_rename(self):
        """Previously active proof must not reappear after a crash/restart."""
        path = freshness.issue_certificate(
            self.certs, self.db, self.sources, CYCLE, {'fixture': 'original'})
        events = []
        real_replace = os.replace
        def replaced(*a, **kw):
            result = real_replace(*a, **kw)
            events.append(('replace', Path(a[1]).parent))
            return result
        def synced(directory):
            events.append(('fsync', Path(directory)))
        with patch.object(freshness, 'fsync_directory', side_effect=synced, create=True), \
             patch.object(freshness.os, 'replace', side_effect=replaced):
            freshness.revoke_certificate(self.certs, CYCLE)
        self.assertFalse(path.exists())
        self.assertEqual(len(list((self.certs / 'revoked').glob('*.json'))), 1)
        self.assertIn(('replace', self.certs / 'revoked'), events)
        replacement_index = events.index(('replace', self.certs / 'revoked'))
        self.assertIn(('fsync', self.certs), events[replacement_index + 1:],
                      'Revocation did not persist removal from active directory')
        self.assertIn(('fsync', self.certs / 'revoked'), events[replacement_index + 1:],
                      'Revocation did not persist evidence archive directory')


@unittest.skipUnless(os.name == 'posix' and hasattr(signal, 'SIGKILL'),
                     'Requires Linux/WSL SIGKILL and POSIX locking')
class RealGraphCrashRecoveryTests(unittest.TestCase):
    """Validate real DuckDB+Parquet certification after process SIGKILL."""

    def setUp(self):
        try:
            import duckdb  # noqa: F401
            from test_gold_graph_invariants import GoldGraphTests
            from scripts.certify_gold_graph_cycle import SCOPED_SOURCES, quote_sql
        except ImportError:
            self.skipTest('Project DuckDB/Jinja fixture dependencies required')
        temp = tempfile.TemporaryDirectory(prefix='gg012-real-graph-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db = self.root / 'warehouse.duckdb'
        self.facts = self.root / 'facts'
        self.part = self.facts / 'election_type=municipal/year=2024'
        self.part.mkdir(parents=True)
        self.certs = self.root / 'proofs'
        self.active = freshness.certificate_path(self.certs, CYCLE)
        fixture = GoldGraphTests('test_graph_projection_conserves_relationships')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.db.execute(f'ATTACH {quote_sql(str(self.db))} AS recovery')
        try:
            for source in SCOPED_SOURCES:
                fixture.db.execute(
                    f'CREATE TABLE recovery.{source} AS SELECT * FROM main.{source}')
            # Real physical publisher reads this persisted staging relation.
            fixture.db.execute(
                'CREATE TABLE recovery.int_candidate_fact_partition AS '
                'SELECT * EXCLUDE(cycle_id) FROM main.fact_candidate_votes')
        finally:
            fixture.db.execute('DETACH recovery')
        fixture.db.execute(
            'COPY (SELECT * EXCLUDE(cycle_id) FROM fact_candidate_votes) '
            f'TO {quote_sql(str(self.part / "data.parquet"))} (FORMAT PARQUET)')
        self.base = [
            '--db', str(self.db), '--source-root', str(self.facts),
            '--attestations-dir', str(self.certs), '--cycle', CYCLE,
            '--memory-limit', '512MB', '--lock-timeout', '10',
        ]
        initial = self._certify()
        self.assertEqual(initial.returncode, 0, initial.stdout + initial.stderr)

    def _certify(self):
        return subprocess.run(
            [sys.executable, str(ROOT / 'scripts/graph_freshness.py'),
             *self.base, 'certify'], cwd=ROOT, capture_output=True, text=True,
            timeout=120)

    def _guarded_rows(self):
        return guarded.read_graph(
            db=self.db, source_root=self.facts, attestations_dir=self.certs,
            cycle=CYCLE, relation='edges', kind='VOTED_FOR_CANDIDATE',
            limit=3, lock_timeout=3, memory_limit='512MB')['rows']

    def _injected_command(self, stage: str, marker: Path):
        return [sys.executable, '-c', INJECTED_CERTIFIER,
                stage, str(marker), *self.base, 'certify']

    def _die_at(self, stage: str) -> None:
        marker = self.root / f'{stage}.marker'
        result = subprocess.run(self._injected_command(stage, marker),
                                cwd=ROOT, capture_output=True, text=True,
                                timeout=120)
        self.assertEqual(result.returncode, -signal.SIGKILL,
                         result.stdout + result.stderr)
        self.assertEqual(marker.read_text(encoding='utf8'), stage)

    def test_death_after_revocation_denies_old_authority_then_restart(self):
        baseline = self._guarded_rows()
        self._die_at('after_revoke')
        self.assertFalse(self.active.exists(), 'old proof resurrected after revoke')
        self.assertTrue(list((self.certs / 'revoked').glob('*.json')))
        with self.assertRaisesRegex(guarded.GuardError, 'Missing active graph certificate'):
            self._guarded_rows()
        restart = self._certify()
        self.assertEqual(restart.returncode, 0, restart.stdout + restart.stderr)
        self.assertEqual(self._guarded_rows(), baseline)

    def test_death_before_hard_link_never_exposes_partial_certificate(self):
        self._die_at('before_link')
        self.assertFalse(self.active.exists())
        self.assertTrue(list(self.certs.glob('*.tmp')),
                        'Falsifier did not reach temporary certificate boundary')
        with self.assertRaisesRegex(guarded.GuardError, 'Missing active graph certificate'):
            self._guarded_rows()
        restart = self._certify()
        self.assertEqual(restart.returncode, 0, restart.stdout + restart.stderr)
        self.assertTrue(self._guarded_rows())

    def test_death_after_hard_link_preserves_committed_authority(self):
        baseline = self._guarded_rows()
        self._die_at('after_link')
        self.assertTrue(self.active.is_file(), 'linked proof disappeared on SIGKILL')
        self.assertEqual(self._guarded_rows(), baseline)
        with self.assertRaisesRegex(freshness.GraphFreshnessError,
                                    'Active certificate exists'):
            freshness.issue_certificate(self.certs, self.db, self.facts,
                                        CYCLE, {'invalid': 'overwrite'})

    def test_publisher_waits_for_certifier_and_recovery_denies_stale_proof(self):
        baseline = self._guarded_rows()
        marker = self.root / 'issuer-holds-locks'
        issuer = subprocess.Popen(self._injected_command('hold_before_issue', marker),
                                  cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.monotonic() + 120
            while not marker.exists() and issuer.poll() is None and time.monotonic() < deadline:
                time.sleep(.025)
            self.assertTrue(marker.is_file(), 'issuer failed before publication gate')
            self.assertIsNone(issuer.poll(), 'issuer exited before concurrent publisher')
            # An independent REAL physical publisher may not publish before the
            # existing graph EX warehouse + partition lock is released.
            blocked = subprocess.run([
                sys.executable, str(ROOT / 'scripts/write_candidate_fact_partition.py'),
                '--db', str(self.db), '--root', str(self.facts),
                '--year', '2024', '--election-type', 'municipal',
                '--warehouse-lock-timeout', '0', '--lock-timeout', '0',
            ], cwd=ROOT, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn('Timed out acquiring warehouse lock', blocked.stderr)
        finally:
            if issuer.poll() is None:
                issuer.kill()
            output, _ = issuer.communicate(timeout=10)
        self.assertEqual(issuer.returncode, -signal.SIGKILL, output)
        self.assertFalse(self.active.exists(), 'old authority survived interrupted reissue')
        # Lock release after worker death lets the already-implemented physical
        # publisher recover, preserving immutable Parquet publication semantics.
        published = subprocess.run([
            sys.executable, str(ROOT / 'scripts/write_candidate_fact_partition.py'),
            '--db', str(self.db), '--root', str(self.facts),
            '--year', '2024', '--election-type', 'municipal',
            '--warehouse-lock-timeout', '3', '--lock-timeout', '3',
        ], cwd=ROOT, capture_output=True, text=True, timeout=40)
        self.assertEqual(published.returncode, 0, published.stdout + published.stderr)
        self.assertTrue((self.part / 'data.parquet').is_symlink())
        with self.assertRaisesRegex(guarded.GuardError, 'Missing active graph certificate'):
            self._guarded_rows()
        restart = self._certify()
        self.assertEqual(restart.returncode, 0, restart.stdout + restart.stderr)
        self.assertEqual(self._guarded_rows(), baseline,
                         'Recovered graph result diverged from continuous control')


if __name__ == '__main__':
    unittest.main()
