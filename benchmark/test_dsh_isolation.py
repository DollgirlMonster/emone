"""The isolation contract of Emone's private DeepSeek Harness.

`tools/dsh_local.sh` and `tools/install_emone.sh` are what let a factory-new
Mac run the bundle without touching the user's environment — no PATH edit, no
shell-rc edit, no writes into `~/.dsh`, `~/.npm`, `~/Library/pnpm`, `~/.cache` or
`~/.local/state`. That property is not visible in any single line, so it is easy
to lose in a refactor: `--prefix` reads like isolation but moves only where
packages are unpacked, and pnpm creates `~/Library/pnpm` even with `--store-dir`.
Both were real (measured 2026-09-24), so this file pins the redirections that
close them.

The real end-to-end proof was run by hand in a simulated factory-new HOME (engine,
tools, DSH, pnpm, plugin and route under one root; the user's npm/pnpm/XDG caches
untouched) and is recorded in the wiki. What is pinned here is the mechanism, plus
one dry run that must write nothing at all.

    cd benchmark && python3 -m unittest test_dsh_isolation -v
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DSH_LOCAL = ROOT / "tools/dsh_local.sh"
INSTALLER = ROOT / "tools/install_emone.sh"


def logical_lines(text: str) -> list[str]:
    """Join backslash continuations, so one command is one string.

    The redirects sit on the `env` lines and the tool being run on the last line
    of the same command; a per-line search would miss every one of them.
    """
    joined: list[str] = []
    current = ""
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            current += stripped[:-1] + " "
            continue
        joined.append(current + stripped)
        current = ""
    if current:
        joined.append(current)
    return joined


class PrivateHarnessIsolationTests(unittest.TestCase):
    """Every npm and pnpm invocation keeps its writes inside the private root."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = DSH_LOCAL.read_text()
        cls.commands = logical_lines(cls.script)

    def test_every_npm_install_redirects_its_cache_and_user_config(self) -> None:
        # `--prefix` does not move npm's cache or log directory, and npm reads
        # the user's ~/.npmrc unless told not to. DSH, pnpm and Playwright each
        # install with npm, so all three must carry both variables.
        installs = [
            command
            for command in self.commands
            if re.search(r'"\$\(?npm(?:_bin)?\)?"\s+install', command)
        ]
        self.assertEqual(len(installs), 3, "expected the DSH, pnpm and Playwright installs")
        for command in installs:
            head = command.strip()[:70]
            self.assertIn("npm_config_cache=", command, head)
            self.assertIn("npm_config_userconfig=", command, head)

    def test_the_plugin_install_redirects_pnpm_home_and_xdg(self) -> None:
        # `--store-dir` moves the store; PNPM_HOME is what stops pnpm creating
        # ~/Library/pnpm, and the XDG pair catches its cache and state.
        plugin = [c for c in self.commands if "plugin --profile web add" in c]
        self.assertEqual(len(plugin), 1)
        for variable in ("--store-dir", "PNPM_HOME=", "XDG_CACHE_HOME=", "XDG_STATE_HOME="):
            self.assertIn(variable, plugin[0])

    def test_the_web_launch_redirects_caches_but_keeps_config_readable(self) -> None:
        # Caches private; HOME, XDG_CONFIG_HOME and XDG_DATA_HOME left alone so the
        # agent can read the user's git, gh and registry configuration.
        web = [c for c in self.commands if "web --port" in c and "exec env" in c]
        self.assertEqual(len(web), 1)
        for variable in ("XDG_CACHE_HOME=", "XDG_STATE_HOME=", "PNPM_HOME=", "npm_config_cache="):
            self.assertIn(variable, web[0])
        for absent in ("npm_config_userconfig=", "XDG_CONFIG_HOME=", "XDG_DATA_HOME="):
            self.assertNotIn(absent, web[0])

    def test_the_private_paths_live_under_one_root(self) -> None:
        for name in (
            "DSH_NPM_CACHE",
            "DSH_NPMRC",
            "DSH_XDG_CACHE",
            "DSH_XDG_STATE",
            "DSH_PNPM_HOME",
        ):
            with self.subTest(name=name):
                pattern = re.compile(rf'^{name}="\$DSH_ROOT/', re.M)
                self.assertRegex(self.script, pattern, name)

    def test_dsh_is_never_put_on_the_users_path(self) -> None:
        for pattern in (
            r"^\s*export PATH=",
            r"\.zshrc",
            r"\.bash_profile",
            r"\.profile",
            r"npm install -g",
            r"/usr/local/bin",
        ):
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, self.script, re.M), pattern)

    def test_a_dry_run_writes_nothing_at_all(self) -> None:
        home = pathlib.Path(tempfile.mkdtemp(prefix="dsh-isolation-home-"))
        try:
            marker = home / "marker"
            marker.write_text("")
            env = dict(
                os.environ,
                HOME=str(home),
                EMONE_DSH_ROOT=str(home / ".emone/dsh"),
                EMONE_DSH_DRY_RUN="1",
            )
            result = subprocess.run(
                ["bash", str(DSH_LOCAL), "ensure"],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-400:])
            self.assertIn("would", result.stdout)
            written = [
                str(p.relative_to(home)) for p in home.rglob("*") if p.is_file() and p != marker
            ]
            self.assertEqual(written, [], "the dry run wrote into the home")
        finally:
            shutil.rmtree(home, ignore_errors=True)


class ModelsDirectoryTests(unittest.TestCase):
    """The models directory the harness is handed follows the layout.

    `dsh_route.sh` resolves the installed layout when `EMONE_MODELS_DIR` is
    unset, but `dsh_local.sh` always exports it, so a hard-coded `<repo>/models`
    wins over that fallback — and on an installed copy `<repo>` is `<root>/src`,
    where no models are. Measured 2026-09-25 in a simulated install: the plugin's
    boot-time route refresh then dies with "no catalog (it lists no installed
    models)" while a correct `<root>/models` writes the route.
    """

    def install_layout(self) -> pathlib.Path:
        root = pathlib.Path(tempfile.mkdtemp(prefix="tt-models-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        tools = root / "src" / "tools"
        tools.mkdir(parents=True)
        shutil.copy2(DSH_LOCAL, tools / "dsh_local.sh")
        (root / "bin").mkdir()
        (root / "models").mkdir()
        return root

    def paths(
        self, script: pathlib.Path, home: pathlib.Path, environment: dict[str, str] | None = None
    ) -> dict[str, str]:
        env = {"PATH": os.environ["PATH"], "HOME": str(home)}
        if environment:
            env.update(environment)
        result = subprocess.run(
            ["bash", str(script), "paths"], env=env, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stderr[-400:])
        return dict(line.split(None, 1) for line in result.stdout.splitlines() if line.strip())

    def test_an_installed_copy_reads_the_models_beside_bin(self) -> None:
        root = self.install_layout()
        paths = self.paths(root / "src" / "tools" / "dsh_local.sh", root)
        self.assertEqual(paths["models"], str(root / "models"))

    def test_a_checkout_still_reads_its_own_models(self) -> None:
        paths = self.paths(DSH_LOCAL, ROOT)
        self.assertEqual(paths["models"], str(ROOT / "models"))

    def test_an_explicit_directory_still_wins(self) -> None:
        root = self.install_layout()
        paths = self.paths(
            root / "src" / "tools" / "dsh_local.sh",
            root,
            {"EMONE_MODELS_DIR": "/tmp/chosen-models"},
        )
        self.assertEqual(paths["models"], "/tmp/chosen-models")

    def test_the_harness_launch_exports_the_resolved_directory(self) -> None:
        # The regression shape: the export must be the resolved value, never
        # `$REPO_ROOT/models`, which is what an installed copy does not have.
        script = DSH_LOCAL.read_text()
        self.assertNotIn('EMONE_MODELS_DIR="$REPO_ROOT/models"', script)
        launches = [
            c
            for c in logical_lines(script)
            if re.search(r"\bweb --", c) and "EMONE_MODELS_DIR=" in c
        ]
        self.assertEqual(len(launches), 2, "expected the web and smoke launches")
        for launch in launches:
            self.assertIn('EMONE_MODELS_DIR="$MODELS_DIR"', launch)


class InstallerIsolationTests(unittest.TestCase):
    """The installer owns one root and two launcher scripts — nothing else."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = INSTALLER.read_text()

    def test_it_never_edits_a_shell_rc_file(self) -> None:
        # It *prints* the line to add when ~/.local/bin is not on PATH; that is
        # advice, not an edit.
        self.assertIsNone(
            re.search(r'>>?\s*"?\$HOME/\.(?:zshrc|bash_profile|profile)', self.script)
        )
        self.assertIn("Add it to your PATH", self.script)

    def test_its_only_writes_outside_the_root_are_the_two_launchers(self) -> None:
        writes = set(
            re.findall(
                r'(?:cat\s*>\s*|mkdir\s+-p\s+|chmod\s+\+x\s+)"?(\$HOME[^"\\ ]*)', self.script
            )
        )
        allowed = {
            "$HOME/.local/bin",
            "$HOME/.local/bin/emone",
            "$HOME/.local/bin/emone-web",
        }
        self.assertEqual(writes - allowed, set())

    def test_the_install_root_is_overridable_for_a_simulated_machine(self) -> None:
        self.assertIn('INSTALL_ROOT="${EMONE_ROOT:-$HOME/.emone}"', self.script)
        for directory in ("~/.emone", "~/.local/bin"):
            self.assertIn(directory, self.script)


if __name__ == "__main__":
    unittest.main()
