"""Behavioral checks for interruption-safe Batch entrypoint shutdown."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / "infra" / "container" / "batch_entrypoint.sh"


class EntrypointTests(unittest.TestCase):
    def test_term_waits_for_child_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            ready = temporary / "ready"
            environment = os.environ.copy()
            environment.update(
                {
                    "APP_ROOT": str(ROOT),
                    "RUN_ROOT": str(temporary / "run"),
                    "DATASET_DIR": str(temporary / "data"),
                    "TERMINATION_GRACE_SECONDS": "5",
                }
            )
            child_code = (
                "import pathlib,signal,sys,time; "
                f"pathlib.Path({str(ready)!r}).write_text('ready'); "
                "signal.signal(signal.SIGTERM, "
                "lambda *_: (time.sleep(1), sys.exit(143))); "
                "time.sleep(30)"
            )
            process = subprocess.Popen(
                ["bash", str(ENTRYPOINT), sys.executable, "-c", child_code],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(ready.exists(), "child process never became ready")
            started = time.monotonic()
            process.send_signal(signal.SIGTERM)
            output, _ = process.communicate(timeout=8)
            elapsed = time.monotonic() - started
            self.assertGreaterEqual(elapsed, 0.8, output)
            self.assertEqual(143, process.returncode, output)
            status = json.loads(
                (temporary / "run" / "batch_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(143, status["exit_code"])


if __name__ == "__main__":
    unittest.main()
