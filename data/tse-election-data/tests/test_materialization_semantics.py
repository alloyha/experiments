from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tse_ingest


class MaterializationSemanticsTests(unittest.TestCase):
    def test_active_version_survives_missing_regenerable_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = (
                root
                / "raw"
                / "election_type=general"
                / "year=2026"
                / "domain=seats"
                / "dataset=candidatos_2026"
                / "resource=r1"
                / "sha256=abc"
            )
            source = version / "source" / "resource.zip"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"immutable")

            missing = version / "extracted" / "resource_BRASIL.csv"
            row = {
                "year": 2026,
                "election_type": "general",
                "election_scope": "federal_state",
                "mode": "incremental",
                "dataset_id": "candidatos-2026",
                "dataset_title": "Candidatos - 2026",
                "resource_id": "r1",
                "resource_name": "Vagas",
                "resource_format": "ZIP",
                "resource_fingerprint": "fingerprint",
                "domain": "seats",
                "partition": "BRASIL",
                "source_url": "https://example.invalid/resource.zip",
                "source_sha256": "abc",
                "source_size_bytes": source.stat().st_size,
                "source_object": str(source.relative_to(root)),
                "extracted_objects": [str(missing.relative_to(root))],
                "granularity": "brasil",
                "uf": "",
                "checked_at": "2026-10-10T00:00:00+00:00",
                "selection_complete": True,
                "extractor": "python",
                "selected_members": ["resource_BRASIL.csv"],
                "materialization_state": "evicted",
            }

            self.assertTrue(
                tse_ingest.active_version_complete(root, row)
            )
            self.assertFalse(
                tse_ingest.materialization_complete(root, row)
            )

            # PR32 preserves the old ingestion behavior until PR34 wires
            # explicit consumer-side lazy materialization.
            self.assertFalse(
                tse_ingest.local_state_complete(root, row)
            )

            tse_ingest.save_state(root, {"key": row})

            persisted = json.loads(
                tse_ingest.state_path(root).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                persisted["key"]["source_sha256"],
                "abc",
            )

            # current_objects remains a physical index: a missing cache
            # representation must not be advertised to DuckDB/dbt.
            self.assertEqual(
                tse_ingest.current_objects_path(root).read_text(
                    encoding="utf-8"
                ),
                "",
            )


if __name__ == "__main__":
    unittest.main()
