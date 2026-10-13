"""Subprocess-level network interdiction and path-isolation gates for PR42."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import certify_cache_roundtrip as c


class OfflineGuardTests(unittest.TestCase):
    def test_child_python_cannot_connect_network(self):
        with tempfile.TemporaryDirectory(prefix='pr42-offline-') as d:
            policy = c.offline_policy(Path(d) / 'policy')
            script = Path(d) / 'attempt.py'
            script.write_text('import socket; socket.socket().connect(("127.0.0.1", 443))')
            with self.assertRaisesRegex(c.CertificationError, 'network disabled'):
                c.run_command([sys.executable, str(script)], policy=policy, project=Path(d))

    def test_safe_root_rejects_symlinked_ancestor(self):
        with tempfile.TemporaryDirectory(prefix='pr42-safety-') as d:
            root = Path(d) / 'raw'
            root.mkdir()
            secret = Path(d) / 'secret'
            secret.mkdir()
            (root / 'other').symlink_to(secret, target_is_directory=True)
            with self.assertRaisesRegex(c.CertificationError, 'escaped root'):
                c.require_path(root, 'other/foo', kind='source')

    def test_csv_reader_rejects_ragged_records(self):
        with tempfile.TemporaryDirectory(prefix='pr42-csv-') as d:
            f = Path(d) / 'bad.csv'
            f.write_bytes(b'A;B\n1;2\n3;4;5\n')
            with self.assertRaisesRegex(c.CertificationError, 'irregular CSV'):
                c.csv_fingerprint(f)


if __name__ == '__main__':
    unittest.main()
