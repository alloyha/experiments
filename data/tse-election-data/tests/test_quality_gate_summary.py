"""The summary must stay quiet on success, loud on failure, and preserve gates."""
from __future__ import annotations

import io
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.run_quality_gate import STAGES, Stage, run_checks

ROOT = Path(__file__).resolve().parents[1]


class QualityGatePresentationTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.output = io.StringIO()
        self.scratch = tempfile.TemporaryDirectory(prefix="tse-quality-")
        self.addCleanup(self.scratch.cleanup)
        self.summary = Path(self.scratch.name) / "summary.md"

    def run_gate(self, responses, *, verbose=False):
        results = iter(responses)

        def invoke(command, **kwargs):
            self.calls.append((command, kwargs))
            return next(results)

        return run_checks(
            stages=STAGES,
            verbose=verbose,
            output=self.output,
            runner=invoke,
            clock=lambda: 1.0,
            summary_path=self.summary,
            workdir=ROOT,
        )

    @staticmethod
    def result(code=0, log="PASS: some check\n"):
        return subprocess.CompletedProcess(args=[], returncode=code, stdout=log)

    def test_all_four_ci_stages_remain_gated(self):
        self.assertEqual(self.run_gate([self.result()] * len(STAGES)), 0)
        self.assertEqual([call[0][-1] for call in self.calls], [s.make_target for s in STAGES])
        self.assertEqual(len(self.calls), 4)
        self.assertIn("PASS · 4/4", self.output.getvalue())
        self.assertIn("Aprovado: 4/4", self.summary.read_text())
        self.assertNotIn("PASS: some check", self.output.getvalue())

    def test_expected_fail_strings_in_successful_negative_tests_are_not_failures(self):
        responses = [self.result(log="FAIL: deliberately rejected\nRan 706 tests in 80.0s\nOK\n")]
        responses.extend(self.result() for _ in range(3))
        self.assertEqual(self.run_gate(responses), 0)
        self.assertIn("706 testes", self.output.getvalue())
        self.assertNotIn("deliberately rejected", self.output.getvalue())

    def test_real_failure_stops_following_stages_and_shows_tail(self):
        responses = [self.result(), self.result(2, "line one\nERROR: invalid graph\n")]
        self.assertEqual(self.run_gate(responses), 1)
        self.assertEqual(len(self.calls), 2)
        self.assertIn("ERROR: invalid graph", self.output.getvalue())
        self.assertIn("VERBOSE=1", self.output.getvalue())
        report = self.summary.read_text()
        self.assertIn("Reprovado: 1/4", report)
        self.assertIn("Não executado", report)

    def test_warning_on_successful_odcs_check_is_visible(self):
        responses = [self.result(), self.result(),
                     self.result(log="WARNING: consumer change requires review\n"),
                     self.result(), self.result()]
        self.assertEqual(self.run_gate(responses), 0)
        self.assertIn("1 aviso(s)", self.output.getvalue())
        self.assertNotIn("consumer change requires review", self.output.getvalue())

    def test_verbose_mode_displays_full_log(self):
        self.assertEqual(self.run_gate([self.result(log="full diagnostics\n")] * 4, verbose=True), 0)
        self.assertEqual(self.output.getvalue().count("full diagnostics"), 4)

    def test_launch_failure_is_reported_not_masked(self):
        def missing(command, **kwargs):
            raise FileNotFoundError("make unavailable")
        result = run_checks(
            stages=(Stage("Unit", "unit"),),
            runner=missing,
            output=self.output,
            summary_path=self.summary,
        )
        self.assertEqual(result, 1)
        self.assertIn("make unavailable", self.output.getvalue())

    def test_legacy_gates_still_in_makefile_and_ci_uses_compact_entrypoint(self):
        makefile = (ROOT / "Makefile").read_text()
        workflow = (ROOT.parents[1] / ".github/workflows/tse-medallion.yml").read_text()
        self.assertIn("check: setup", makefile)
        self.assertIn("make check", workflow)
        self.assertNotIn("- run: make unit", workflow)
        self.assertEqual(
            [s.make_target for s in STAGES],
            ["unit", "medallion-contracts", "data-contracts-check", "compile"],
        )


if __name__ == "__main__":
    unittest.main()
