"""The launcher's expert-cache rule: 30% of physical memory, warned not capped.

The rule belongs to the launcher, and it is a recommendation rather than a
limit: a larger `--ram` is warned about in red and passed on, because the
machine is the operator's. These tests pin the arithmetic (three tenths,
floored to whole GB, which is what `--ram` takes) and both sides of the
boundary.

`EMONE_PHYSICAL_RAM_BYTES` is the launcher's seam for exactly this: the mapping
has to be checkable on a machine of any size.

Run from this directory, like the other benchmark tests:

    cd benchmark && python3 -m unittest test_launcher_ram -v
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/server_launcher.sh"
SERVER = ROOT / ".build/release/EmoneServer"
MODELS = ROOT / "models"

# Installed memory in bytes, and the launcher's rule for it: 30%, floored.
MACHINES = {
    8 * 2**30: 2,
    16 * 2**30: 4,
    24 * 2**30: 7,
    32 * 2**30: 9,
    64 * 2**30: 19,
}


def installed_model() -> str | None:
    """The first served id under models/, or None when nothing is installed.

    The launcher's dry run needs a model that is really there — its catalog is
    the only place a served id comes from — and this test has to be runnable on
    a checkout whose models/ is empty.
    """
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


def dry_run(*args: str, physical_bytes: int) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["EMONE_PHYSICAL_RAM_BYTES"] = str(physical_bytes)
    return subprocess.run(
        ["bash", str(LAUNCHER), "--client", "server", *args, "--dry-run"],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )


class RamRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        model = installed_model()
        if model is None:
            self.skipTest("no install under models/ and no built server to list one")
        self.model = model

    def test_rule_is_thirty_percent_rounded_down(self) -> None:
        for memory, expected in MACHINES.items():
            with self.subTest(memory_gb=memory // 2**30):
                run = dry_run("--model", self.model, physical_bytes=memory)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn(
                    f"RAM: model default (measured; 30% of this Mac is {expected} GB)",
                    run.stdout,
                )

    def test_at_the_rule_is_silent_and_above_it_warns(self) -> None:
        memory = 24 * 2**30
        at_rule = dry_run("--model", self.model, "--ram", "7", physical_bytes=memory)
        self.assertEqual(at_rule.returncode, 0, at_rule.stderr)
        self.assertNotIn("WARNING", at_rule.stderr)
        self.assertIn("--ram-budget 7G", at_rule.stdout)

        above = dry_run("--model", self.model, "--ram", "8", physical_bytes=memory)
        self.assertEqual(above.returncode, 0, above.stderr)
        self.assertIn("WARNING: the server would hold about 8 GB", above.stderr)
        self.assertIn("30% of this Mac's 24 GB", above.stderr)
        self.assertIn("Starting anyway with 8 GB", above.stderr)
        # Warned, not capped: the requested size is what the server is given.
        self.assertIn("--ram-budget 8G", above.stdout)
        self.assertIn("over 30% of this Mac's RAM", above.stdout)

    def test_default_path_warns_about_nothing(self) -> None:
        run = dry_run("--model", self.model, physical_bytes=24 * 2**30)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("WARNING", run.stderr)
        self.assertNotIn("--ram-budget", run.stdout)

    def test_unreadable_memory_means_no_rule(self) -> None:
        # No rule, so nothing to warn about even for a large explicit size…
        run = dry_run("--model", self.model, "--ram", "32", physical_bytes=0)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("WARNING", run.stderr)
        self.assertNotIn("30% of this Mac", run.stdout)
        self.assertIn("--ram-budget 32G", run.stdout)
        # …and the default's note claims no percentage either.
        default = dry_run("--model", self.model, physical_bytes=0)
        self.assertIn("RAM: model default (measured) |", default.stdout)

    def test_boundary_scales_with_the_machine(self) -> None:
        # The rule is 2 GB on an 8 GB Mac, so the smallest accepted target (4)
        # is above it there and below it on a 24 GB Mac (rule 7).
        small = dry_run("--model", self.model, "--ram", "4", physical_bytes=8 * 2**30)
        self.assertEqual(small.returncode, 0, small.stderr)
        self.assertIn("WARNING: the server would hold about 4 GB", small.stderr)
        self.assertIn("--ram-budget 4G", small.stdout)

        large = dry_run("--model", self.model, "--ram", "4", physical_bytes=24 * 2**30)
        self.assertEqual(large.returncode, 0, large.stderr)
        self.assertNotIn("WARNING", large.stderr)
        self.assertIn("--ram-budget 4G", large.stdout)

    def test_targets_below_the_floor_are_refused(self) -> None:
        # A streaming install cannot stay under 4 GB: the weights plus the
        # 8-slot minimum cache are ~4.7 GB. The launcher refuses rather than
        # accepting a number the runtime would silently overshoot, and a bare
        # number is refused too instead of being ignored as an unknown word.
        for ram in ("1", "2", "3", "2G"):
            with self.subTest(placement="flag", ram=ram):
                run = dry_run("--model", self.model, "--ram", ram, physical_bytes=24 * 2**30)
                self.assertEqual(run.returncode, 2, run.stdout)
                self.assertIn("unknown RAM target", run.stderr)
                self.assertNotIn("--ram-budget", run.stdout)
        # A bare number is refused too rather than ignored as an unknown word.
        # 1 and 0 are not tested positionally: they are thinking levels ("on",
        # "off") and the optional-value loop reads those first.
        for ram in ("2", "3", "2G"):
            with self.subTest(placement="positional", ram=ram):
                run = dry_run("--model", self.model, ram, physical_bytes=24 * 2**30)
                self.assertEqual(run.returncode, 2, run.stdout)
                self.assertIn("unknown RAM target", run.stderr)
                self.assertNotIn("--ram-budget", run.stdout)


if __name__ == "__main__":
    unittest.main()
