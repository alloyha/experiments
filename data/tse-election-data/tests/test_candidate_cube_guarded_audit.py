"""GG-009: real POSIX process boundaries for guarded cube freshness audit.

All DuckDB files, published generations and certificates belong to temp dirs.
No canonical TSE data is accessed, even by the publisher subprocess.
"""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import candidate_cube_freshness as freshness
from scripts import certify_candidate_cube as cube

ROOT = Path(__file__).resolve().parents[1]

try:
    import duckdb
except ImportError:
    duckdb = None

EXCLUSIVE_PROBE = '''import fcntl,sys
with open(sys.argv[1], 'a+b') as lock:
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  sys.exit(7)
'''


def exclusive_probe(path: Path) -> int:
    result = subprocess.run([sys.executable, '-c', EXCLUSIVE_PROBE, str(path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=6)
    return result.returncode


class CubeAuditProcessLockTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='gg009-cube-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db = self.root / 'warehouse.duckdb'
        self.db.write_bytes(b'synthetic for lock-only tests')
        self.sources = self.root / 'facts'
        self.part = self.sources / 'election_type=municipal/year=2024'
        self.part.mkdir(parents=True)
        (self.part / 'data.parquet').write_bytes(b'fixture source')
        self.db_lock = self.root / 'warehouse.duckdb.build.lock'
        self.part_lock = self.part / '.publish.lock'

    def test_both_lock_inodes_block_independent_exclusive_writers(self):
        with cube.guarded_cube_audit_locks(self.db, self.sources, timeout=0) as cycles:
            self.assertEqual(cycles, {'2024:municipal'})
            self.assertEqual(exclusive_probe(self.db_lock), 7)
            self.assertEqual(exclusive_probe(self.part_lock), 7)
        self.assertEqual(exclusive_probe(self.db_lock), 0)
        self.assertEqual(exclusive_probe(self.part_lock), 0)

    def test_timeout_when_independent_writer_holds_partition_lock(self):
        self.part_lock.touch()
        holder = '''import fcntl,sys
with open(sys.argv[1],'a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 print('HELD',flush=True)
 sys.stdin.readline()
'''
        p = subprocess.Popen([sys.executable, '-c', holder, str(self.part_lock)],
                             cwd=ROOT, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(p.stdout.readline().strip(), 'HELD')
            with self.assertRaisesRegex(TimeoutError, 'candidate audit lock'):
                with cube.guarded_cube_audit_locks(self.db, self.sources, timeout=0):
                    self.fail('reader acquired partition held by exclusive writer')
            # On mid-acquisition timeout, the earlier warehouse lock is released.
            self.assertEqual(exclusive_probe(self.db_lock), 0)
        finally:
            if p.poll() is None:
                p.communicate(input='\n', timeout=5)
            else:
                p.communicate(timeout=5)

    def test_two_cycles_hold_both_partition_inodes(self):
        extra = self.sources / 'election_type=general/year=2018'
        extra.mkdir(parents=True)
        (extra / 'data.parquet').write_bytes(b'another synthetic generation')
        with cube.guarded_cube_audit_locks(self.db, self.sources, timeout=0) as cycles:
            self.assertEqual(cycles, {'2018:general', '2024:municipal'})
            self.assertEqual(exclusive_probe(extra / '.publish.lock'), 7)
            self.assertEqual(exclusive_probe(self.part_lock), 7)
        self.assertEqual(exclusive_probe(extra / '.publish.lock'), 0)
        self.assertEqual(exclusive_probe(self.part_lock), 0)

    def test_negative_timeout_denied_without_touching_lock(self):
        with self.assertRaisesRegex(cube.CertificationError, 'nonnegative'):
            with cube.guarded_cube_audit_locks(self.db, self.sources, timeout=-1):
                self.fail('negative timeout allowed')
        self.assertFalse(self.db_lock.exists())

    def test_warehouse_lock_is_scoped_not_global(self):
        other = self.root / 'independent.duckdb'
        other.write_bytes(b'another canary warehouse')
        other_lock = other.with_name(other.name + '.build.lock')
        with cube.guarded_cube_audit_locks(self.db, self.sources, timeout=0):
            self.assertEqual(exclusive_probe(other_lock), 0)
            with cube.guarded_cube_audit_locks(other, self.sources, timeout=0):
                self.assertEqual(exclusive_probe(self.db_lock), 7)
                self.assertEqual(exclusive_probe(other_lock), 7)

    def test_missing_publication_never_creates_partitions(self):
        empty = self.root / 'empty-facts'
        empty.mkdir()
        absent = empty / 'election_type=general/year=2026'
        with self.assertRaisesRegex(cube.CertificationError, 'No published partitions'):
            with cube.guarded_cube_audit_locks(self.db, empty, timeout=0):
                pass
        self.assertFalse(absent.exists())
        self.assertFalse((empty / 'election_type=general').exists())

    def test_mid_audit_failure_releases_all_flocks(self):
        with patch.object(cube, 'audit', side_effect=cube.CertificationError('injected failure')):
            with self.assertRaisesRegex(cube.CertificationError, 'injected failure'):
                cube.guarded_freshness_audit(self.db, self.sources,
                                             self.root / 'certs', timeout=0)
        self.assertEqual(exclusive_probe(self.db_lock), 0)
        self.assertEqual(exclusive_probe(self.part_lock), 0)

    def test_freshness_failure_releases_all_locks(self):
        coverage = {'cycles': {'2024:municipal': (1, 17)}}
        with patch.object(cube, 'audit', return_value=coverage), \
             patch.object(cube, 'audit_freshness',
                          side_effect=freshness.FreshnessError('missing attestation')):
            with self.assertRaisesRegex(freshness.FreshnessError, 'missing attestation'):
                cube.guarded_freshness_audit(self.db, self.sources,
                                             self.root / 'certs', timeout=0)
        self.assertEqual(exclusive_probe(self.db_lock), 0)
        self.assertEqual(exclusive_probe(self.part_lock), 0)

    def test_sigkill_reader_releases_both_locks(self):
        marker = self.root / 'acquired'
        script = '''import sys,time
from pathlib import Path
from scripts.certify_candidate_cube import guarded_cube_audit_locks
with guarded_cube_audit_locks(Path(sys.argv[1]),Path(sys.argv[2]),timeout=5):
 Path(sys.argv[3]).write_text('locked')
 time.sleep(30)
'''
        reader = subprocess.Popen([sys.executable, '-c', script, str(self.db),
                                   str(self.sources), str(marker)], cwd=ROOT,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 6
            while not marker.is_file() and reader.poll() is None and time.monotonic() < deadline:
                time.sleep(.025)
            if not marker.is_file():
                out, err = reader.communicate(timeout=5)
                self.fail(f'guarded process did not acquire locks: {out} {err}')
            self.assertEqual(exclusive_probe(self.db_lock), 7)
            self.assertEqual(exclusive_probe(self.part_lock), 7)
        finally:
            if reader.poll() is None:
                reader.kill()
            reader.communicate(timeout=5)
        self.assertEqual(reader.returncode, -9)
        self.assertEqual(exclusive_probe(self.db_lock), 0)
        self.assertEqual(exclusive_probe(self.part_lock), 0)


@unittest.skipIf(duckdb is None, 'Requires the project DuckDB dependency')
class RealCubeAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='gg009-cube-real-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db = self.root / 'canary.duckdb'
        self.sources = self.root / 'facts'
        self.part = self.sources / 'election_type=municipal/year=2024'
        self.part.mkdir(parents=True)
        self.generation = self.part / 'generation-initial.parquet'
        self.generation.write_bytes(b'nonempty synthetic published generation')
        self.pointer = self.part / 'data.parquet'
        self.pointer.symlink_to(self.generation.name)
        self.certs = self.root / 'certs'
        self.cycle = '2024:municipal'
        with duckdb.connect(str(self.db)) as con:
            con.execute("CREATE TABLE dim_candidate AS SELECT '2024:municipal'::VARCHAR cycle_id, 1::INTEGER candidate_id")
            con.execute("CREATE TABLE dim_geography AS SELECT '2024:municipal'::VARCHAR cycle_id, 5208707::INTEGER municipality_code")
            con.execute("CREATE TABLE cube_candidate_performance_municipality AS SELECT '2024:municipal'::VARCHAR cycle_id, 17::INTEGER nominal_valid_votes")
            con.execute("CREATE TABLE int_candidate_fact_partition AS SELECT 2024::INTEGER election_year, 'municipal'::VARCHAR election_type, 17::INTEGER nominal_valid_votes")
        inputs = freshness.input_signature(self.db, self.sources, self.cycle)
        output = freshness.output_signature(self.db, self.cycle)
        freshness.issue_certificate(self.certs, self.db, self.sources, self.cycle,
                                    inputs, output)

    def publisher(self, lock_timeout: str) -> subprocess.CompletedProcess:
        return subprocess.run([
            sys.executable, str(ROOT / 'scripts/write_candidate_fact_partition.py'),
            '--db', str(self.db), '--root', str(self.sources),
            '--year', '2024', '--election-type', 'municipal',
            '--warehouse-lock-timeout', lock_timeout,
            '--lock-timeout', '0',
        ], cwd=ROOT, capture_output=True, text=True, timeout=40)

    def test_real_fingerprint_audit_rejects_publication_during_held_locks(self):
        original = cube.audit_freshness
        def writer_attempt(*args, **kwargs):
            attempted = self.publisher('0')
            self.assertNotEqual(attempted.returncode, 0, attempted.stdout)
            self.assertIn('Timed out acquiring warehouse lock', attempted.stderr)
            self.assertEqual(os.readlink(self.pointer), self.generation.name)
            return original(*args, **kwargs)
        with patch.object(cube, 'audit_freshness', side_effect=writer_attempt):
            result = cube.guarded_freshness_audit(self.db, self.sources,
                                                  self.certs, timeout=2)
        self.assertEqual(result['fresh_cycles'], [self.cycle])
        self.assertEqual(result['freshness'], 'certified')
        # The independent physical writer CAN publish once the audit finishes.
        published = self.publisher('2')
        self.assertEqual(published.returncode, 0, published.stdout + published.stderr)
        self.assertNotEqual(os.readlink(self.pointer), self.generation.name)
        with self.assertRaisesRegex(freshness.FreshnessError, 'Parquet or dimensions changed'):
            cube.guarded_freshness_audit(self.db, self.sources, self.certs, timeout=2)

    def test_real_unattested_cycle_fails_closed(self):
        path = freshness.certificate_path(self.certs, self.cycle)
        path.unlink()
        with self.assertRaisesRegex(freshness.FreshnessError, 'missing'):
            cube.guarded_freshness_audit(self.db, self.sources, self.certs, timeout=0)

    def test_symlink_certificate_is_not_authority(self):
        path = freshness.certificate_path(self.certs, self.cycle)
        archive = self.certs / 'saved.json'
        path.rename(archive)
        path.symlink_to(archive.name)
        with self.assertRaisesRegex(freshness.FreshnessError, 'invalid certificate path'):
            cube.guarded_freshness_audit(self.db, self.sources, self.certs, timeout=0)


if __name__ == '__main__':
    unittest.main()
