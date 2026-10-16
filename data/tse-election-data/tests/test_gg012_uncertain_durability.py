"""GG-012: an I/O failure after linking a proof must not leave authority active.

Runs with the existing project test suite and uses only disposable local files.
No real TSE data, DuckDB scans, or modification of canonical certificates.

The test was RED against main@671ad42 and became GREEN after PR #64.
It bounds transient post-link fsync failure, not permanent device failure.
"""
from __future__ import annotations

import errno
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import graph_freshness as freshness


class UncertainGraphCertificateDurabilityTests(unittest.TestCase):
    def test_failed_post_link_fsync_must_rollback_active_authority(self):
        """An EIO after os.link must fail closed and permit a clean retry."""
        with tempfile.TemporaryDirectory(prefix="gg012-uncertain-fsync-") as temp:
            root = Path(temp)
            db = root / "warehouse.duckdb"
            db.write_bytes(b"disposable warehouse identity")
            sources = root / "facts"
            sources.mkdir()
            certs = root / "certificates"
            cycle = "2024:municipal"
            proof = freshness.certificate_path(certs, cycle)
            evidence = {"synthetic": "source and graph were checked"}

            actual_fsync = freshness.fsync_directory
            calls = []

            def fail_once(directory: Path) -> None:
                calls.append(Path(directory))
                if len(calls) == 1:
                    raise OSError(errno.EIO, "injected post-link directory fsync failure")
                actual_fsync(directory)

            with patch.object(freshness, "fsync_directory", side_effect=fail_once):
                with self.assertRaisesRegex(OSError, "injected post-link"):
                    freshness.issue_certificate(certs, db, sources, cycle, evidence)

            # If this fails, the certifier reported a durability failure but
            # nevertheless left an active certificate visible to guarded reads.
            self.assertFalse(
                proof.exists() or proof.is_symlink(),
                "An uncertain, failed certificate issuance left active authority",
            )
            self.assertGreaterEqual(
                len(calls), 2,
                "Rollback must durably fsync the directory after unlinking proof",
            )
            self.assertFalse(list(certs.glob(".*.tmp")),
                             "Failed issuance must not leave temporary evidence")

            # Reissue with the existing create-only API; no compatibility path.
            committed = freshness.issue_certificate(certs, db, sources, cycle, evidence)
            self.assertEqual(committed, proof)
            self.assertTrue(proof.is_file())


if __name__ == "__main__":
    unittest.main()
