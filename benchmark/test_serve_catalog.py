#!/usr/bin/env python3
"""`tools/serve.sh` with no install named serves a folder of installs.

The launcher's point is a one-word start: with no argument it serves
`TINYTITAN_MODELS_DIR` (default `<checkout>/models`) as the catalog, a folder
of installs may be named in place of one install, and a request without a
model name gets `TINYTITAN_DEFAULT_MODEL` or, when that is not installed, the
folder's first completed install. These tests drive `--dry-run` against
synthetic installs (an empty `verified-install.json` is all serve.sh reads), so
nothing is built, loaded or started.

    cd benchmark && python3 -m unittest test_serve_catalog -v
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVE = ROOT / "tools/serve.sh"
INSTALLS = ("ornith-1.5_35B_A3B_4Bit", "qwen3.6_35B_A3B_4Bit", "qwen38flash.gturbo")


class ServeCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory()
        self.models = pathlib.Path(self.scratch.name) / "models"
        for name in INSTALLS:
            (self.models / name).mkdir(parents=True)
            (self.models / name / "verified-install.json").write_text("{}")
        # An installer lock file beside the installs is not an install.
        (self.models / "qwen3.6_35B_A3B_4Bit.install.lock").write_text("")

    def tearDown(self) -> None:
        self.scratch.cleanup()

    def serve(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        full_env = {k: v for k, v in os.environ.items() if not k.startswith("TINYTITAN_")}
        full_env.update(env)
        return subprocess.run(
            ["/bin/bash", str(SERVE), *args, "--dry-run", "--no-live-trace"],
            capture_output=True, text=True, env=full_env, timeout=30)

    def served(self, result: subprocess.CompletedProcess[str]) -> tuple[str, str]:
        self.assertEqual(result.returncode, 0, result.stderr)
        words = result.stdout.split()
        return words[words.index("--model") + 1], words[words.index("--models-dir") + 1]

    def test_no_argument_serves_the_models_dir(self) -> None:
        model, catalog = self.served(self.serve(TINYTITAN_MODELS_DIR=str(self.models)))
        self.assertEqual(catalog, str(self.models))
        self.assertEqual(model, str(self.models / "qwen3.6_35B_A3B_4Bit"))

    def test_a_folder_of_installs_is_a_catalog(self) -> None:
        model, catalog = self.served(self.serve(str(self.models)))
        self.assertEqual(catalog, str(self.models))
        self.assertEqual(model, str(self.models / "qwen3.6_35B_A3B_4Bit"))

    def test_default_model_by_name(self) -> None:
        model, _ = self.served(self.serve(
            str(self.models), TINYTITAN_DEFAULT_MODEL="qwen38flash.gturbo"))
        self.assertEqual(model, str(self.models / "qwen38flash.gturbo"))

    def test_missing_default_falls_back_to_first_install(self) -> None:
        model, _ = self.served(self.serve(
            str(self.models), TINYTITAN_DEFAULT_MODEL="qwen3.5_9B_4Bit"))
        self.assertEqual(model, str(self.models / "ornith-1.5_35B_A3B_4Bit"))

    def test_one_install_keeps_its_directory_as_the_catalog(self) -> None:
        model, catalog = self.served(self.serve(str(self.models / "qwen38flash.gturbo")))
        self.assertEqual(model, str(self.models / "qwen38flash.gturbo"))
        self.assertEqual(catalog, str(self.models))

    def test_folder_without_installs_is_refused(self) -> None:
        empty = pathlib.Path(self.scratch.name) / "empty"
        empty.mkdir()
        result = self.serve(str(empty))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("holds no completed install", result.stderr)

    def test_missing_folder_is_refused(self) -> None:
        result = self.serve(str(pathlib.Path(self.scratch.name) / "unmounted"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is not a directory", result.stderr)


if __name__ == "__main__":
    unittest.main()
