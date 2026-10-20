"""Exercise timeout cleanup without a real dbt process or warehouse."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import test_graph_dbt_writer_handoff as handoff


class NestedDbtCleanupTests(unittest.TestCase):
    def test_nested_session_is_killed_when_gateway_is_aborted(self):
        # The gateway launches the dbt child in its OWN session, which is the
        # precise condition that made writer.kill()+communicate(5) hang.
        with tempfile.TemporaryDirectory(prefix="gg009-cleanup-test-") as root:
            marker = Path(root) / "child.pid"
            log_path = Path(root) / "writer.log"
            gateway_script = (
                "import pathlib,subprocess,sys\n"
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(60)'],start_new_session=True)\n"
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n"
                "child.wait()\n"
            )
            with log_path.open("w") as log:
                writer = subprocess.Popen(
                    [sys.executable, "-c", gateway_script, str(marker)],
                    stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            try:
                deadline = time.monotonic() + 8
                while (not marker.is_file() and writer.poll() is None
                       and time.monotonic() < deadline):
                    time.sleep(.02)
                self.assertTrue(marker.is_file(), log_path.read_text())
                pid = int(marker.read_text())
                start = time.monotonic()
                handoff.RealDbtGraphHandoffTests._stop_writer(writer, marker)
                self.assertLess(time.monotonic() - start, 5)
                self.assertIsNotNone(writer.poll())
                # Zombies are already terminated but may await init reaping.
                stat = Path(f"/proc/{pid}/stat")
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    try:
                        state = stat.read_text().split(") ", 1)[1].split()[0]
                    except FileNotFoundError:
                        break  # Already reaped.
                    if state == "Z":
                        break  # Killed, awaiting init reaping.
                    time.sleep(.02)
                else:
                    self.fail("Orphaned dbt child still executing after cleanup")
            finally:
                if writer.poll() is None:
                    handoff.RealDbtGraphHandoffTests._stop_writer(writer, marker)

    def test_child_cleanup_when_gateway_already_exited(self):
        # The earlier finally only called _stop_writer when the gateway was
        # still running. A crashed gateway can leave its child lock-holder.
        with tempfile.TemporaryDirectory(prefix="gg009-dead-gateway-") as root:
            marker = Path(root) / "child.pid"
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                start_new_session=True,
            )
            try:
                marker.write_text(str(child.pid))
                gateway = subprocess.Popen(
                    [sys.executable, "-c", "pass"], start_new_session=True,
                )
                gateway.wait(timeout=5)
                self.assertIsNotNone(gateway.poll())
                handoff.RealDbtGraphHandoffTests._stop_writer(gateway, marker)
                child.wait(timeout=5)
                self.assertLess(child.returncode, 0)
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
