from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tse_ingest


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


if __name__ == "__main__":
    unittest.main()
