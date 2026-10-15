"""Pure, deterministic safety regression tests for the canary certifier."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/certify_candidate_cube.py"
spec = importlib.util.spec_from_file_location("certify_candidate_cube", SOURCE)
certifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(certifier)


class CandidateCubeCertifierTests(unittest.TestCase):
    def test_physical_partition_discovery(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for kind, year in (("general", 2018), ("municipal", 2024)):
                p = root / f"election_type={kind}/year={year}/data.parquet"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"fixture")
            self.assertEqual(certifier.available_cycles(root), {
                "2018:general", "2024:municipal",
            })
            with self.assertRaisesRegex(certifier.CertificationError, "missing published"):
                certifier.require_cycle(root, "2026:general")

    def test_incomplete_or_unexpected_history_must_fail(self):
        with self.assertRaisesRegex(certifier.CertificationError, "missing.*2018"):
            certifier.validate_coverage({"2018:general", "2024:municipal"},
                                        {"2024:municipal"})
        with self.assertRaisesRegex(certifier.CertificationError, "unexpected"):
            certifier.validate_coverage({"2018:general"},
                                        {"2018:general", "2026:general"})
        certifier.validate_coverage({"2018:general"}, {"2018:general"})

    def test_failed_run_never_starts_tests(self):
        with patch.object(certifier, "require_cycle"), \
             patch.object(certifier, "snapshot_other_cycles", return_value=[]), \
             patch.object(certifier, "input_signature", return_value={}), \
             patch.object(certifier, "revoke_certificate"), \
             patch.object(certifier, "dbt_call", side_effect=certifier.CertificationError("run failed")) as call:
            class Args:
                year = 2026
                election_type = "general"
                dbt = "dbt"
                memory_limit = "4GB"
                run_timeout = 600
                test_timeout = 600
                attestations_dir = Path("/tmp/test-attestations")
            with self.assertRaisesRegex(certifier.CertificationError, "run failed"):
                certifier.certify(Args(), Path("/tmp/test.duckdb"), Path("/tmp/facts"))
            self.assertEqual(call.call_count, 1)
            self.assertEqual(call.call_args.args[1], "run")

    def test_snapshot_mismatch_prevents_tests(self):
        with patch.object(certifier, "require_cycle"), \
             patch.object(certifier, "snapshot_other_cycles", side_effect=[[("2024:municipal", "GO", 1)], [("2024:municipal", "GO", 2)]]), \
             patch.object(certifier, "input_signature", return_value={}), \
             patch.object(certifier, "revoke_certificate"), \
             patch.object(certifier, "dbt_call", return_value=1.0) as call:
            class Args:
                year = 2026
                election_type = "general"
                dbt = "dbt"
                memory_limit = "4GB"
                run_timeout = 600
                test_timeout = 600
                attestations_dir = Path("/tmp/test-attestations")
            with self.assertRaisesRegex(certifier.CertificationError, "Unselected cycles changed"):
                certifier.certify(Args(), Path("/tmp/test.duckdb"), Path("/tmp/facts"))
            self.assertEqual(call.call_count, 1)

    def test_refuse_to_overwrite_evidence(self):
        with tempfile.TemporaryDirectory() as t:
            evidence = Path(t) / "cycle.json"
            certifier.report_json(evidence, {"cycle": "2018:general"})
            with self.assertRaises(FileExistsError):
                certifier.report_json(evidence, {"cycle": "2022:general"})

    def test_mutating_certification_requires_attestation_directory(self):
        # Without this gate, a previous certificate could survive a recertify
        # that mutates the DuckDB but silently skips certificate revocation.
        class Args:
            year = 2024
            election_type = "municipal"
            attestations_dir = None

        with patch.object(certifier, "require_cycle") as source, \
             patch.object(certifier, "dbt_call") as run:
            with self.assertRaisesRegex(certifier.CertificationError,
                                        "--attestations-dir is required"):
                certifier.certify(Args(), Path("/tmp/test.duckdb"), Path("/tmp/facts"))
            source.assert_not_called()
            run.assert_not_called()


class CertifierFreshnessIntegrationTests(unittest.TestCase):
    def test_freshness_denied_when_existing_attestation_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            class Args:
                year = 2024
                election_type = 'municipal'
                dbt = 'dbt'
                memory_limit = '4GB'
                run_timeout = 5
                test_timeout = 5
                attestations_dir = path
            with patch.object(certifier, 'require_cycle'), \
                 patch.object(certifier, 'snapshot_other_cycles', return_value=[]), \
                 patch.object(certifier, 'input_signature', return_value={'source': 1}), \
                 patch.object(certifier, 'revoke_certificate') as revoke, \
                 patch.object(certifier, 'dbt_call', side_effect=certifier.CertificationError('run failed')):
                with self.assertRaisesRegex(certifier.CertificationError, 'run failed'):
                    certifier.certify(Args(), Path('/tmp/canary.duckdb'), Path('/tmp/facts'))
                revoke.assert_called_once_with(path, '2024:municipal')

    def test_attestation_is_not_issued_if_inputs_change_during_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            class Args:
                year = 2024
                election_type = 'municipal'
                dbt = 'dbt'
                memory_limit = '4GB'
                run_timeout = 5
                test_timeout = 5
                attestations_dir = path
            with patch.object(certifier, 'require_cycle'), \
                 patch.object(certifier, 'snapshot_other_cycles', return_value=[]), \
                 patch.object(certifier, 'materialized_cycles', return_value={'2024:municipal': (1, 50)}), \
                 patch.object(certifier, 'input_signature', side_effect=[{'a': 1}, {'a': 2}]), \
                 patch.object(certifier, 'output_signature', return_value={'out': 1}), \
                 patch.object(certifier, 'revoke_certificate'), \
                 patch.object(certifier, 'dbt_call', return_value=0.1), \
                 patch.object(certifier, 'issue_certificate') as issue:
                with self.assertRaisesRegex(certifier.CertificationError, 'changed during certification'):
                    certifier.certify(Args(), Path('/tmp/canary.duckdb'), Path('/tmp/facts'))
                issue.assert_not_called()

if __name__ == "__main__":
    unittest.main()
