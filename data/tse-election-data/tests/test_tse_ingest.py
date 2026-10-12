"""Regression tests for the PR31 hardening of tse_ingest."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import tse_ingest


class SaveStateExceptionalYearTests(unittest.TestCase):
    def test_save_state_preserves_explicit_type_for_exceptional_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obj = root / "raw" / "exception.csv"
            obj.parent.mkdir(parents=True)
            obj.write_text("x\n1\n", encoding="utf-8")

            state = {
                "exception": {
                    "year": 2028,
                    "election_type": "general",
                    "election_scope": "",
                    "dataset_id": "exceptional",
                    "dataset_title": "Exceptional",
                    "resource_id": "resource",
                    "resource_name": "Resource",
                    "resource_format": "CSV",
                    "domain": "candidate",
                    "partition": "BRASIL",
                    "source_sha256": "abc",
                    "selection_complete": True,
                    "selected_members": ["exception.csv"],
                    "source_object": str(obj.relative_to(root)),
                    "source_url": "https://example.invalid/data.csv",
                    "extracted_objects": [str(obj.relative_to(root))],
                    "checked_at": "2028-01-01T00:00:00+00:00",
                }
            }

            tse_ingest.save_state(root, state)
            row = json.loads(
                tse_ingest.current_objects_path(root).read_text(encoding="utf-8").strip()
            )
            self.assertEqual(row["election_type"], "general")
            self.assertEqual(row["election_scope"], "federal_state")


class NativeExtractorGlobTests(unittest.TestCase):
    def test_auto_extractor_falls_back_for_glob_like_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "test.zip"
            destination = root / "out"
            member = "resultado_[BRASIL].csv"

            with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(member, b"x\n1\n")

            with patch("tse_ingest.shutil.which", return_value="/usr/bin/unzip"):
                outputs, backend, names = tse_ingest.extract_selected_members(
                    source, destination, "all", None, "auto",
                )

            self.assertEqual(backend, "python")
            self.assertEqual(names, [member])
            self.assertEqual(outputs[0].read_bytes(), b"x\n1\n")

    def test_native_extractor_refuses_glob_like_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "test.zip"
            with zipfile.ZipFile(source, "w") as zf:
                zf.writestr("resultado_[BRASIL].csv", b"x\n1\n")

            with patch("tse_ingest.shutil.which", return_value="/usr/bin/unzip"):
                with self.assertRaises(RuntimeError):
                    tse_ingest.extract_selected_members(
                        source, root / "out", "all", None, "native",
                    )


class CrashRecoveryTests(unittest.TestCase):
    def test_restart_after_download_rehydrates_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = {"name": "candidatos_2026", "title": "Candidatos 2026"}
            resource = {
                "id": "r1",
                "name": "BRASIL - Consulta",
                "url": "https://example.invalid/consulta.csv",
                "format": "CSV",
            }
            fingerprint = tse_ingest.resource_fingerprint(resource)
            sha = "a" * 64
            domain = tse_ingest.infer_domain(package, resource)
            version_dir = (
                tse_ingest.resource_output_dir(root, 2026, "general", package, resource, domain)
                / f"sha256={sha}"
            )
            source = version_dir / "source" / "consulta.csv"
            source.parent.mkdir(parents=True)
            source.write_text("a;b\n1;2\n", encoding="utf-8")

            pending_row = {
                "resource_fingerprint": fingerprint,
                "source_object": str(source.relative_to(root)),
                "source_sha256": sha,
                "source_size_bytes": source.stat().st_size,
                "source_url": resource["url"],
                "domain": domain,
            }

            with patch.object(
                tse_ingest, "download_resilient",
                side_effect=AssertionError("network must not be touched"),
            ):
                outcome = tse_ingest.download_resource_worker(
                    root=root, year=2026, election_type="general", mode="incremental",
                    package=package, resource=resource, granularity="all", uf=None,
                    previous=None, force=False, pending_row=pending_row,
                )

            self.assertEqual(outcome["kind"], "pending")
            item = outcome["pending"]
            self.assertEqual(item.attempts, 0)

            state_row, record, _profile = tse_ingest.prepare_resource_worker(
                pending=item, extractor="python",
            )
            self.assertEqual(record.source_sha256, sha)
            self.assertEqual(len(record.extracted_objects), 1)
            self.assertTrue(state_row["selection_complete"])


class PendingJournalTests(unittest.TestCase):
    def test_pending_journal_is_separate_from_active_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tse_ingest.save_pending(root, {"k": {"source_sha256": "b" * 64}})

            self.assertTrue(tse_ingest.pending_path(root).exists())
            self.assertFalse(tse_ingest.state_path(root).exists())
            self.assertEqual(tse_ingest.load_state(root), {})
            self.assertEqual(
                tse_ingest.load_pending(root), {"k": {"source_sha256": "b" * 64}}
            )


if __name__ == "__main__":
    unittest.main()
