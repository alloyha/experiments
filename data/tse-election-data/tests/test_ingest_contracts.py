from __future__ import annotations

import sys

import zipfile
from pathlib import Path
from unittest.mock import patch


import json
import tempfile
import unittest
from pathlib import Path

import tse_ingest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prune_extracted_raw

class CandidateDomainContractTests(unittest.TestCase):
    def test_candidate_history_is_distinct_domain(self):
        package = {"name": "candidatos-2026"}
        resource = {
            "name": "Histórico de candidaturas",
            "url": (
                "https://cdn.tse.jus.br/estatistica/sead/odsele/"
                "historico_candidatura/historico_candidatura_2026.zip"
            ),
            "format": "CSV",
        }
        self.assertEqual(
            tse_ingest.infer_domain(package, resource),
            "candidate_history",
        )

    def test_candidate_snapshot_remains_candidate(self):
        package = {"name": "candidatos-2026"}
        resource = {
            "name": "Candidatos",
            "url": (
                "https://cdn.tse.jus.br/estatistica/sead/odsele/"
                "consulta_cand/consulta_cand_2026.zip"
            ),
            "format": "CSV",
        }
        self.assertEqual(tse_ingest.infer_domain(package, resource), "candidate")

    def test_analytics_excludes_history_but_extended_keeps_it(self):
        package = {"name": "candidatos-2026"}
        resource = {
            "name": "Histórico de candidaturas",
            "url": (
                "https://cdn.tse.jus.br/estatistica/sead/odsele/"
                "historico_candidatura/historico_candidatura_2026.zip"
            ),
            "format": "ZIP",
        }
        self.assertFalse(
            tse_ingest.resource_allowed(package, resource, "analytics", None)
        )
        self.assertTrue(
            tse_ingest.resource_allowed(package, resource, "extended", None)
        )

    def test_load_state_migrates_legacy_history_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "_metadata"
            metadata.mkdir()
            state = {
                "legacy": {
                    "year": 2026,
                    "election_type": "general",
                    "election_scope": "federal_state",
                    "resource_id": "history-resource",
                    "resource_name": "Histórico de candidaturas",
                    "source_url": (
                        "https://cdn.tse.jus.br/estatistica/sead/odsele/"
                        "historico_candidatura/historico_candidatura_2026.zip"
                    ),
                    "domain": "candidate",
                    "granularity": "brasil",
                    "uf": "",
                }
            }
            (metadata / "resource_state.json").write_text(
                json.dumps(state),
                encoding="utf-8",
            )

            loaded = tse_ingest.load_state(root)
            row = next(iter(loaded.values()))

            self.assertEqual(row["domain"], "candidate_history")

    def test_missing_extracted_object_rehydrates_from_local_source_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            package = {
                "name": "candidatos-2026",
                "title": "Candidatos - 2026",
            }
            resource = {
                "id": "test-vagas-resource",
                "name": "Vagas",
                "url": (
                    "https://cdn.tse.jus.br/estatistica/sead/odsele/"
                    "consulta_vagas/consulta_vagas_2026.zip"
                ),
                "format": "ZIP",
            }

            election_type = "general"
            year = 2026
            granularity = "brasil"

            domain = tse_ingest.infer_domain(package, resource)
            self.assertEqual(domain, "seats")

            version_dir = (
                tse_ingest.resource_output_dir(
                    root,
                    year,
                    election_type,
                    package,
                    resource,
                    domain,
                )
                / "sha256=test-sha"
            )

            source_dir = version_dir / "source"
            source_dir.mkdir(parents=True)

            source = source_dir / "consulta_vagas_2026.zip"

            csv_name = "consulta_vagas_2026_BRASIL.csv"
            csv_contents = (
                '"DT_GERACAO";"HH_GERACAO";"ANO_ELEICAO";"CD_TIPO_ELEICAO"\n'
                '"01/01/2026";"00:00:00";"2026";"2"\n'
            )

            with zipfile.ZipFile(
                source,
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as zf:
                zf.writestr(csv_name, csv_contents.encode("latin-1"))

            source_sha256, source_size, _ = tse_ingest.hash_file(source)

            extracted = version_dir / "extracted" / csv_name

            # Contract precondition:
            # immutable source exists, regenerable extracted object does not.
            self.assertTrue(source.is_file())
            self.assertFalse(extracted.exists())

            fingerprint = tse_ingest.resource_fingerprint(resource)

            previous = {
                "year": year,
                "election_type": election_type,
                "election_scope": "federal_state",
                "mode": "incremental",
                "dataset_id": package["name"],
                "dataset_title": package["title"],
                "resource_id": resource["id"],
                "resource_name": resource["name"],
                "resource_format": resource["format"],
                "resource_fingerprint": fingerprint,
                "domain": domain,
                "partition": "BRASIL",
                "source_url": resource["url"],
                "source_sha256": source_sha256,
                "source_size_bytes": source_size,
                "source_object": str(source.relative_to(root)),
                "extracted_objects": [str(extracted.relative_to(root))],
                "granularity": granularity,
                "uf": "",
                "checked_at": "2026-10-08T00:00:00+00:00",
                "selection_complete": True,
                "extractor": "python",
                "selected_members": [csv_name],
            }

            # The previous state is intentionally incomplete because the
            # extracted cache object was removed.
            self.assertFalse(
                tse_ingest.local_state_complete(root, previous)
            )

            # Any attempt to open a network session means the local-rehydrate
            # contract was violated.
            with patch(
                "tse_ingest.make_session",
                side_effect=AssertionError(
                    "network must not be used during local rehydration"
                ),
            ):
                outcome = tse_ingest.download_resource_worker(
                    root=root,
                    year=year,
                    election_type=election_type,
                    mode="incremental",
                    package=package,
                    resource=resource,
                    granularity=granularity,
                    uf=None,
                    previous=previous,
                    force=False,
                )

            self.assertEqual(outcome["kind"], "pending")

            pending = outcome["pending"]

            # No network transfer occurred.
            self.assertEqual(pending.download_seconds, 0.0)
            self.assertEqual(pending.attempts, 0)
            self.assertEqual(pending.retry_seconds, 0.0)
            self.assertEqual(pending.resumed_bytes, 0)
            self.assertEqual(pending.network_errors, 0)

            # The immutable local ZIP is reused.
            self.assertEqual(pending.source_target, source)
            self.assertEqual(pending.source_sha256, source_sha256)
            self.assertEqual(
                pending.selected_member_names,
                [csv_name],
            )

            state_row, record, profile = (
                tse_ingest.prepare_resource_worker(
                    pending=pending,
                    extractor="python",
                )
            )

            # The missing cache representation was recreated.
            self.assertTrue(extracted.is_file())

            self.assertEqual(
                extracted.read_text(encoding="latin-1"),
                csv_contents,
            )

            # Control-plane state now points to the recreated object.
            self.assertEqual(
                record.extracted_objects,
                [str(extracted.relative_to(root))],
            )
            self.assertEqual(
                state_row["extracted_objects"],
                [str(extracted.relative_to(root))],
            )

            # The state is complete again.
            self.assertTrue(
                tse_ingest.local_state_complete(root, state_row)
            )

            # Observability must correctly describe a local rehydrate.
            self.assertEqual(profile.download_seconds, 0.0)
            self.assertEqual(profile.attempts, 0)
            self.assertEqual(profile.retry_seconds, 0.0)
            self.assertEqual(profile.extractor, "python")

    def test_prune_removes_deleted_object_from_current_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data" / "tse"
            raw_root = data_root / "raw"
            metadata_root = data_root / "_metadata"

            metadata_root.mkdir(parents=True)

            sha_root = (
                raw_root
                / "election_type=general"
                / "year=2026"
                / "domain=seats"
                / "dataset=candidatos_2026"
                / "resource=test-resource"
                / "sha256=test-sha"
            )

            source_dir = sha_root / "source"
            extracted_dir = sha_root / "extracted"

            source_dir.mkdir(parents=True)
            extracted_dir.mkdir(parents=True)

            source = source_dir / "consulta_vagas_2026.zip"
            extracted = extracted_dir / "consulta_vagas_2026_BRASIL.csv"

            source.write_bytes(b"immutable-source")
            extracted.write_text(
                '"ANO_ELEICAO";"QT_VAGAS"\n'
                '"2026";"1"\n',
                encoding="latin-1",
            )

            source_rel = str(source.relative_to(data_root))
            extracted_rel = str(extracted.relative_to(data_root))

            untouched_object = (
                raw_root
                / "election_type=general"
                / "year=2026"
                / "domain=candidate"
                / "dataset=candidatos_2026"
                / "resource=other-resource"
                / "sha256=other-sha"
                / "extracted"
                / "consulta_cand_2026_BRASIL.csv"
            )
            untouched_object.parent.mkdir(parents=True)
            untouched_object.write_text(
                '"ANO_ELEICAO"\n"2026"\n',
                encoding="latin-1",
            )

            untouched_rel = str(untouched_object.relative_to(data_root))

            current_objects = metadata_root / "current_objects.jsonl"

            rows = [
                {
                    "year": 2026,
                    "election_type": "general",
                    "resource_id": "test-resource",
                    "resource_name": "Vagas",
                    "domain": "seats",
                    "source_object": source_rel,
                    "object": extracted_rel,
                },
                {
                    "year": 2026,
                    "election_type": "general",
                    "resource_id": "other-resource",
                    "resource_name": "Candidatos",
                    "domain": "candidate",
                    "source_object": "raw/other/source.zip",
                    "object": untouched_rel,
                },
            ]

            with current_objects.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")

            # Preconditions.
            self.assertTrue(source.is_file())
            self.assertTrue(extracted.is_file())
            self.assertTrue(untouched_object.is_file())

            # Simulate the deletion performed by prune_extracted_raw.py.
            deleted = {extracted_rel}
            extracted.unlink()

            removed = prune_extracted_raw.rewrite_current_objects(
                data_root,
                deleted,
            )

            self.assertEqual(removed, 1)

            # Durable source must remain intact.
            self.assertTrue(source.is_file())

            # Deleted cache object is really gone.
            self.assertFalse(extracted.exists())

            # Unrelated objects must remain untouched.
            self.assertTrue(untouched_object.is_file())

            remaining = [
                json.loads(line)
                for line in current_objects.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            self.assertEqual(len(remaining), 1)
            self.assertEqual(
                remaining[0]["resource_id"],
                "other-resource",
            )
            self.assertEqual(
                remaining[0]["object"],
                untouched_rel,
            )

            # The deleted object must no longer be referenced.
            referenced_objects = {
                row["object"]
                for row in remaining
            }

            self.assertNotIn(
                extracted_rel,
                referenced_objects,
            )

if __name__ == "__main__":
    unittest.main()
