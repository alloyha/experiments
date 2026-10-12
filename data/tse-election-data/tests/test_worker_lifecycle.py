"""Worker-level tests: the lifecycle machine drives the real download decisions."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tse_ingest

PACKAGE = {"name": "candidatos_2026", "title": "Candidatos 2026"}
RESOURCE = {
    "id": "r1",
    "name": "BRASIL - Consulta",
    "url": "https://example.invalid/consulta.csv",
    "format": "CSV",
}


def _complete_previous(root: Path, content: bytes, fingerprint: str) -> dict:
    sha = hashlib.sha256(content).hexdigest()
    domain = tse_ingest.infer_domain(PACKAGE, RESOURCE)
    version_dir = (
        tse_ingest.resource_output_dir(root, 2026, "general", PACKAGE, RESOURCE, domain)
        / f"sha256={sha}"
    )
    source = version_dir / "source" / "consulta.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    return {
        "year": 2026,
        "election_type": "general",
        "dataset_id": PACKAGE["name"],
        "resource_id": "r1",
        "resource_name": RESOURCE["name"],
        "resource_fingerprint": fingerprint,
        "domain": domain,
        "source_url": RESOURCE["url"],
        "source_sha256": sha,
        "source_size_bytes": len(content),
        "source_object": str(source.relative_to(root)),
        "extracted_objects": [str(source.relative_to(root))],
        "selected_members": ["consulta.csv"],
        "selection_complete": True,
        "granularity": "all",
        "uf": "",
        "checked_at": "2026-01-01T00:00:00+00:00",
        "partition": "GLOBAL",
    }


class WorkerLifecycleTests(unittest.TestCase):
    def test_complete_unchanged_resource_skips_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fingerprint = tse_ingest.resource_fingerprint(RESOURCE)
            previous = _complete_previous(root, b"a;b\n1;2\n", fingerprint)

            with patch.object(
                tse_ingest, "download_resilient",
                side_effect=AssertionError("network must not be touched"),
            ):
                outcome = tse_ingest.download_resource_worker(
                    root=root, year=2026, election_type="general", mode="incremental",
                    package=PACKAGE, resource=RESOURCE, granularity="all", uf=None,
                    previous=previous, force=False,
                )

            self.assertEqual(outcome["kind"], "metadata-skip")

    def test_same_bytes_after_download_is_content_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = b"a;b\n1;2\n"
            previous = _complete_previous(root, content, "stale-fingerprint")

            def fake_download(session, url, destination, **_):
                destination.write_bytes(content)
                sha = hashlib.sha256(content).hexdigest()
                return sha, len(content), 0.0, 1, 0.0, 0, 0

            with patch.object(tse_ingest, "download_resilient", side_effect=fake_download):
                outcome = tse_ingest.download_resource_worker(
                    root=root, year=2026, election_type="general", mode="incremental",
                    package=PACKAGE, resource=RESOURCE, granularity="all", uf=None,
                    previous=previous, force=False,
                )

            self.assertEqual(outcome["kind"], "content-skip")

    def test_forced_run_never_content_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = b"a;b\n1;2\n"
            previous = _complete_previous(root, content, "stale-fingerprint")

            def fake_download(session, url, destination, **_):
                destination.write_bytes(content)
                sha = hashlib.sha256(content).hexdigest()
                return sha, len(content), 0.0, 1, 0.0, 0, 0

            with patch.object(tse_ingest, "download_resilient", side_effect=fake_download):
                outcome = tse_ingest.download_resource_worker(
                    root=root, year=2026, election_type="general", mode="incremental",
                    package=PACKAGE, resource=RESOURCE, granularity="all", uf=None,
                    previous=previous, force=True,
                )

            self.assertEqual(outcome["kind"], "pending")


if __name__ == "__main__":
    unittest.main()
