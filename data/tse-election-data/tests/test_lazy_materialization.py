from __future__ import annotations

import hashlib

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import tse_ingest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts"),
)
import ensure_materialized


class LazyMaterializationTests(unittest.TestCase):
    def make_evicted_state(self, root: Path) -> tuple[dict, Path]:
        version = (
            root
            / "raw"
            / "election_type=general"
            / "year=2026"
            / "domain=vote_result"
            / "dataset=resultados_2026"
            / "resource=r1"
            / "sha256=abc"
        )
        source = version / "source" / "resource.zip"
        source.parent.mkdir(parents=True)
        member = "resultado_2026_BRASIL.csv"
        with zipfile.ZipFile(
            source,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as zf:
            zf.writestr(
                member,
                '"ANO_ELEICAO";"QT_VOTOS"\n"2026";"1"\n',
            )

        extracted = version / "extracted" / member
        row = {
            "year": 2026,
            "election_type": "general",
            "election_scope": "federal_state",
            "mode": "incremental",
            "dataset_id": "resultados-2026",
            "dataset_title": "Resultados - 2026",
            "resource_id": "r1",
            "resource_name": "Votação",
            "resource_format": "ZIP",
            "resource_fingerprint": "fingerprint",
            "domain": "vote_result",
            "partition": "BRASIL",
            "source_url": "https://example.invalid/resource.zip",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_size_bytes": source.stat().st_size,
            "source_object": str(source.relative_to(root)),
            "extracted_objects": [str(extracted.relative_to(root))],
            "granularity": "brasil",
            "uf": "",
            "checked_at": "2026-10-10T00:00:00+00:00",
            "selection_complete": True,
            "extractor": "python",
            "selected_members": [member],
            "materialization_state": "evicted",
        }
        return row, extracted

    def test_evicted_cache_rehydrates_without_network_and_reindexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, extracted = self.make_evicted_state(root)

            updated, action = (
                ensure_materialized.materialize_state_row(
                    root,
                    row,
                    extractor="python",
                )
            )

            self.assertEqual(action, "rehydrated")
            self.assertTrue(extracted.is_file())
            self.assertEqual(
                updated["materialization_state"],
                "present",
            )
            self.assertEqual(
                updated["extracted_objects"],
                [str(extracted.relative_to(root))],
            )

            tse_ingest.save_state(root, {"key": updated})
            rows = [
                json.loads(line)
                for line in tse_ingest.current_objects_path(root)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["object"],
                str(extracted.relative_to(root)),
            )

            again, second_action = (
                ensure_materialized.materialize_state_row(
                    root,
                    updated,
                    extractor="python",
                )
            )
            self.assertEqual(second_action, "already-present")
            self.assertEqual(
                again["source_sha256"],
                updated["source_sha256"],
            )

    def test_missing_immutable_source_is_not_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, extracted = self.make_evicted_state(root)
            (root / row["source_object"]).unlink()

            with self.assertRaisesRegex(
                RuntimeError,
                "active version is incomplete",
            ):
                ensure_materialized.materialize_state_row(
                    root,
                    row,
                    extractor="python",
                )

            self.assertFalse(extracted.exists())


if __name__ == "__main__":
    unittest.main()
