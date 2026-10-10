from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts"),
)
import prune_extracted_raw


class CacheEvictionTests(unittest.TestCase):
    def test_eviction_preserves_active_version_and_removes_physical_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data" / "tse"
            metadata = data_root / "_metadata"
            version = (
                data_root
                / "raw"
                / "election_type=general"
                / "year=2026"
                / "domain=vote_result"
                / "dataset=resultados_2026"
                / "resource=r1"
                / "sha256=abc"
            )
            source = version / "source" / "resource.zip"
            extracted = version / "extracted" / "result_BRASIL.csv"
            source.parent.mkdir(parents=True)
            extracted.parent.mkdir(parents=True)
            metadata.mkdir(parents=True)
            source.write_bytes(b"immutable")
            extracted.write_text("x\n1\n", encoding="utf-8")

            source_rel = str(source.relative_to(data_root))
            extracted_rel = str(extracted.relative_to(data_root))

            state = {
                "key": {
                    "year": 2026,
                    "election_type": "general",
                    "resource_id": "r1",
                    "source_sha256": "abc",
                    "source_object": source_rel,
                    "extracted_objects": [extracted_rel],
                    "selected_members": ["result_BRASIL.csv"],
                    "selection_complete": True,
                    "materialization_state": "present",
                }
            }
            (metadata / "resource_state.json").write_text(
                json.dumps(state),
                encoding="utf-8",
            )
            (metadata / "current_objects.jsonl").write_text(
                json.dumps(
                    {
                        "year": 2026,
                        "election_type": "general",
                        "resource_id": "r1",
                        "domain": "vote_result",
                        "source_object": source_rel,
                        "object": extracted_rel,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            deleted = {extracted_rel}
            self.assertEqual(
                prune_extracted_raw.rewrite_current_objects(
                    data_root,
                    deleted,
                ),
                1,
            )
            self.assertEqual(
                prune_extracted_raw.mark_resource_state_evicted(
                    data_root,
                    deleted,
                ),
                1,
            )
            extracted.unlink()

            persisted = json.loads(
                (metadata / "resource_state.json").read_text(
                    encoding="utf-8"
                )
            )["key"]

            self.assertTrue(source.is_file())
            self.assertEqual(persisted["source_sha256"], "abc")
            self.assertEqual(
                persisted["extracted_objects"],
                [extracted_rel],
            )
            self.assertEqual(
                persisted["materialization_state"],
                "evicted",
            )
            self.assertEqual(
                (metadata / "current_objects.jsonl").read_text(
                    encoding="utf-8"
                ),
                "",
            )

    def test_domain_is_read_from_hive_path(self):
        path = Path(
            "raw/election_type=general/year=2026/"
            "domain=electorate/dataset=x/resource=y/"
            "sha256=z/extracted/a.csv"
        )
        self.assertEqual(
            prune_extracted_raw.resource_domain(path),
            "electorate",
        )


if __name__ == "__main__":
    unittest.main()
