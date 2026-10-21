"""Runs the frontend's JavaScript tests (tests/js) with Node.js, when it is installed."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_frontend_drops_stale_responses():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    tests = Path(__file__).parent / "js" / "ordering.test.mjs"
    result = subprocess.run(
        [node, "--test", "--test-timeout=20000", str(tests)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
