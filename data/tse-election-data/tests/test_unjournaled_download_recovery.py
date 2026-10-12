"""PR39: crash after physical download materialization, before journal commit.

All inputs are synthetic CSV bytes under TemporaryDirectory. External network
calls are patched; child processes only SIGKILL themselves, not their parents.
This is a single-process local-filesystem certification, not a power-loss test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tse_ingest as ingest
from test_worker_lifecycle import PACKAGE, RESOURCE

PAYLOAD = b"a;b\n1;2\n"


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        root=root, year=[2026], from_year=None, to_year=None,
        election_type=None, datasets=None, mode="incremental",
        profile="mirror", granularity="all", uf=None,
        resource_regex=None, list_only=False, force=False,
        workers=1, extract_workers=1, checkpoint_every=1,
        extractor="python", heartbeat_seconds=0, no_profile=True,
    )


def _fake_download(session, url, destination, **kwargs):
    destination.write_bytes(PAYLOAD)
    return hashlib.sha256(PAYLOAD).hexdigest(), len(PAYLOAD), 0.0, 1, 0.0, 0, 0


def _run(root: Path, download):
    package = {**PACKAGE, "resources": [RESOURCE]}
    with (
        patch.object(ingest, "parse_args", return_value=_args(root)),
        patch.object(ingest, "discover_packages", return_value=[package]),
        patch.object(ingest, "download_resilient", side_effect=download) as mocked,
    ):
        result = ingest.main()
    return result, mocked.call_count


_CHILD = r'''
import os
import signal
import sys
from pathlib import Path
from unittest.mock import patch

root, project_dir, tests_dir = sys.argv[1:]
sys.path.insert(0, tests_dir)
sys.path.insert(0, project_dir)
import tse_ingest as ingest
from test_unjournaled_download_recovery import _args, _fake_download, PACKAGE, RESOURCE

root = Path(root)
package = {**PACKAGE, "resources": [RESOURCE]}

def kill_before_journal(*args, **kwargs):
    # Called in the coordinator AFTER download_resource_worker returned.
    physical = list(root.glob("raw/**/sha256=*/source/consulta.csv"))
    assert len(physical) == 1 and physical[0].is_file(), physical
    os.kill(os.getpid(), signal.SIGKILL)
    raise AssertionError("SIGKILL unexpectedly returned")

with (
    patch.object(ingest, "parse_args", return_value=_args(root)),
    patch.object(ingest, "discover_packages", return_value=[package]),
    patch.object(ingest, "download_resilient", side_effect=_fake_download),
    patch.object(ingest, "remember_pending_preparation", side_effect=kill_before_journal),
):
    ingest.main()
raise AssertionError("coordinator unexpectedly survived SIGKILL")
'''


def _manifest(root: Path):
    p = ingest.manifest_path(root)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


@unittest.skipUnless(os.name == "posix" and hasattr(signal, "SIGKILL"), "requires Linux/WSL SIGKILL")
class DownloadJournalGapTests(unittest.TestCase):
    def _crash_before_journal(self, root: Path) -> Path:
        project = Path(ingest.__file__).resolve().parent
        tests = Path(__file__).resolve().parent
        child = subprocess.run(
            [sys.executable, "-c", _CHILD, str(root), str(project), str(tests)],
            cwd=project, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(
            child.returncode, -signal.SIGKILL,
            f"child exit={child.returncode}\nstdout={child.stdout}\nstderr={child.stderr}",
        )
        self.assertFalse(ingest.load_state(root), "physical materialization is not publication")
        self.assertFalse(ingest.load_pending(root), "journal was not committed")
        self.assertEqual(_manifest(root), [])
        candidates = list(root.glob("raw/**/sha256=*/source/consulta.csv"))
        self.assertEqual(len(candidates), 1)
        return candidates[0]

    def test_unjournaled_orphan_recovers_by_controlled_redownload(self):
        with tempfile.TemporaryDirectory(prefix="pr39-journal-gap-") as td:
            root = Path(td)
            physical = self._crash_before_journal(root)
            self.assertEqual(physical.read_bytes(), PAYLOAD)

            # No authoritative fingerprint->source association survived the kill.
            # Re-download is safe; silently adopting an arbitrary orphan is not.
            result, downloads = _run(root, _fake_download)
            self.assertEqual(result, 0)
            self.assertEqual(downloads, 1)
            self.assertEqual(physical.read_bytes(), PAYLOAD)
            state = ingest.load_state(root)
            self.assertEqual(len(state), 1)
            self.assertEqual(len(_manifest(root)), 1)
            self.assertFalse(ingest.load_pending(root))
            self.assertEqual(len(list(root.glob("raw/**/sha256=*/source/consulta.csv"))), 1)

            def no_network(*_a, **_kw):
                raise AssertionError("unchanged published resource must not download")

            result, downloads = _run(root, no_network)
            self.assertEqual(result, 0)
            self.assertEqual(downloads, 0)
            self.assertEqual(len(_manifest(root)), 1)

    def test_corrupt_orphan_is_replaced_with_verified_download(self):
        with tempfile.TemporaryDirectory(prefix="pr39-journal-gap-") as td:
            root = Path(td)
            physical = self._crash_before_journal(root)
            physical.write_bytes(b"corrupted-orphan\n")

            result, downloads = _run(root, _fake_download)
            self.assertEqual(result, 0)
            self.assertEqual(downloads, 1)
            self.assertEqual(
                physical.read_bytes(), PAYLOAD,
                "existing physical orphan was adopted without hash verification",
            )
            state = next(iter(ingest.load_state(root).values()))
            self.assertEqual(state["source_sha256"], hashlib.sha256(physical.read_bytes()).hexdigest())
            self.assertEqual(state["source_size_bytes"], physical.stat().st_size)
            self.assertEqual(len(_manifest(root)), 1)
            self.assertFalse(ingest.load_pending(root))


if __name__ == "__main__":
    unittest.main()
