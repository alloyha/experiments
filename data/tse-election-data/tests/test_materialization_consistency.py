import tempfile
import unittest
from pathlib import Path

import tse_ingest


class MaterializationConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.source = "raw/source/archive.zip"
        self.write(self.source)

        self.row = {
            "source_object": self.source,
            "selection_complete": True,
            "selected_members": ["a.csv", "b.csv"],
            "extracted_objects": [],
        }

    def write(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    def test_partial_member_list_is_incomplete(self):
        self.write("raw/extracted/a.csv")
        self.row["extracted_objects"] = [
            "raw/extracted/a.csv",
        ]

        self.assertTrue(
            tse_ingest.active_version_complete(
                self.root, self.row
            )
        )
        self.assertFalse(
            tse_ingest.materialization_complete(
                self.root, self.row
            )
        )

    def test_duplicate_physical_reference_is_incomplete(self):
        self.write("raw/extracted/a.csv")
        self.row["extracted_objects"] = [
            "raw/extracted/a.csv",
            "raw/extracted/a.csv",
        ]

        self.assertFalse(
            tse_ingest.materialization_complete(
                self.root, self.row
            )
        )

    def test_zero_selected_members_need_no_cache(self):
        self.row["selected_members"] = []
        self.row["extracted_objects"] = []

        self.assertTrue(
            tse_ingest.materialization_complete(
                self.root, self.row
            )
        )

    def test_legacy_record_without_member_list(self):
        self.write("raw/extracted/a.csv")
        self.row.pop("selected_members")
        self.row["extracted_objects"] = [
            "raw/extracted/a.csv",
        ]

        self.assertTrue(
            tse_ingest.materialization_complete(
                self.root, self.row
            )
        )

    def test_direct_csv_materialization(self):
        path = "raw/source/plain.csv"
        self.write(path)

        self.row.update({
            "source_object": path,
            "selected_members": ["plain.csv"],
            "extracted_objects": [path],
        })

        self.assertTrue(
            tse_ingest.materialization_complete(
                self.root, self.row
            )
        )


if __name__ == "__main__":
    unittest.main()
