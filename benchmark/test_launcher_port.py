#!/usr/bin/env python3
"""The launcher's port question, its default, and what it does with a bad one.

The port is the one thing the launcher asks about the *server* rather than the
model, so it has three ways in — the interactive question, `--port`, and
`EMONE_PORT` — and exactly one default, `EMONE_DEFAULT_PORT` in
`tools/emone_models.sh`. These tests pin all four, the validation that runs
before anything is started, and that an unattended run neither hangs nor asks.

Run from this directory, like the other benchmark tests:

    cd benchmark && python3 -m unittest test_launcher_port -v
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/server_launcher.sh"
MODELS_SH = ROOT / "tools/emone_models.sh"
SERVER = ROOT / ".build/release/EmoneServer"
MODELS = ROOT / "models"

INSTALLED = ROOT / "models/qwen3.8-flash-next_125B_A6B_4Bit"
# Flags that answer every other question, so the port is the only prompt left.
# `--concurrency` is one of them: it is asked before the port, so leaving it
# unanswered would consume the newline these tests mean for the port.
QUIET_ANSWERS = (
    "--answers",
    "default",
    "--thinking",
    "off",
    "--ram",
    "9",
    "--engine",
    "gpu",
    "--concurrency",
    "1",
)


def shared_default() -> int:
    """The one default, read from where the scripts read it."""
    match = re.search(r"^EMONE_DEFAULT_PORT=(\d+)$", MODELS_SH.read_text(), re.MULTILINE)
    if match is None:
        raise AssertionError("emone_models.sh no longer declares EMONE_DEFAULT_PORT")
    return int(match.group(1))


def served_model() -> str | None:
    """A model key the launcher accepts, or None when nothing is installed."""
    if not SERVER.is_file():
        return None
    try:
        listing = subprocess.run(
            [str(SERVER), "--catalog", "--models-dir", str(MODELS)],
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        ).stdout
        models = json.loads(listing)["models"]
    except Exception:
        return None
    return models[0]["id"] if models else None


def run_launcher(
    *args: str, env: dict | None = None, stdin: str = "", interactive: bool = False
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["EMONE_LAUNCHER_DRY_RUN"] = "1"
    if interactive:
        environment["EMONE_LAUNCHER_ASSUME_TTY"] = "1"
    else:
        environment.pop("EMONE_LAUNCHER_ASSUME_TTY", None)
    environment.pop("EMONE_PORT", None)
    if env:
        environment.update(env)
    return subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", *args],
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
        timeout=120,
    )


def reported_port(run: subprocess.CompletedProcess[str]) -> str:
    match = re.search(r"\| Port: (\d+) \|", run.stdout)
    if match is None:
        raise AssertionError(f"no port in the summary:\n{run.stdout}\n{run.stderr}")
    return match.group(1)


class PortChoiceTests(unittest.TestCase):
    def setUp(self) -> None:
        model = served_model()
        if model is None:
            self.skipTest("no install under models/ and no built server to list one")
        self.model = model
        self.base = ("--client", "server", "--model", model)

    def test_the_default_is_the_shared_constant(self) -> None:
        run = run_launcher(*self.base)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(reported_port(run), str(shared_default()))
        self.assertIn(f"http://127.0.0.1:{shared_default()}/v1", run.stdout)

    def test_a_flagged_port_is_used_and_the_harness_is_told(self) -> None:
        run = run_launcher(*self.base, "--port", "9123")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(reported_port(run), "9123")
        self.assertIn("http://127.0.0.1:9123/v1", run.stdout)
        # Anything that assumes the default has to be told; the harness route is
        # the one the launcher does not write itself.
        self.assertIn("EMONE_PORT=9123 tools/dsh_route.sh --write", run.stdout)

    def test_the_environment_port_is_used(self) -> None:
        run = run_launcher(*self.base, env={"EMONE_PORT": "9321"})
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(reported_port(run), "9321")
        self.assertIn("EMONE_PORT=9321 tools/dsh_route.sh --write", run.stdout)

    def test_a_flag_beats_the_environment(self) -> None:
        run = run_launcher(*self.base, "--port", "9123", env={"EMONE_PORT": "9321"})
        self.assertEqual(reported_port(run), "9123")

    def test_a_port_the_default_does_not_need_no_hint(self) -> None:
        run = run_launcher(*self.base)
        self.assertNotIn("dsh_route.sh --write", run.stdout)

    def test_bad_ports_are_refused_before_anything_starts(self) -> None:
        for bad in ("abc", "0", "99999", "-1", "80x"):
            with self.subTest(port=bad):
                run = run_launcher(*self.base, "--port", bad)
                self.assertEqual(run.returncode, 2, run.stdout)
                self.assertIn("unknown port", run.stderr)


class PortQuestionTests(unittest.TestCase):
    """The interactive question itself, through the launcher's test seam."""

    def setUp(self) -> None:
        model = served_model()
        if model is None:
            self.skipTest("no install under models/ and no built server to list one")
        self.base = ("--client", "server", "--model", model, *QUIET_ANSWERS)

    def test_an_unattended_run_takes_the_default_without_asking(self) -> None:
        run = run_launcher(*self.base)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("Port for the server?", run.stdout)
        self.assertEqual(reported_port(run), str(shared_default()))

    def test_enter_keeps_the_default(self) -> None:
        run = run_launcher(*self.base, stdin="\n", interactive=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("Port for the server?", run.stdout)
        self.assertEqual(reported_port(run), str(shared_default()))

    def test_a_typed_port_is_taken(self) -> None:
        run = run_launcher(*self.base, stdin="9123\n", interactive=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(reported_port(run), "9123")
        self.assertIn("EMONE_PORT=9123 tools/dsh_route.sh --write", run.stdout)

    def test_a_typed_port_that_is_not_a_number_is_refused(self) -> None:
        run = run_launcher(*self.base, stdin="eighty\n", interactive=True)
        self.assertEqual(run.returncode, 2, run.stdout)
        self.assertIn("unknown port", run.stderr)

    def test_a_flagged_port_skips_the_question(self) -> None:
        run = run_launcher(*self.base, "--port", "9123", stdin="\n", interactive=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("Port for the server?", run.stdout)
        self.assertEqual(reported_port(run), "9123")


if __name__ == "__main__":
    unittest.main()
