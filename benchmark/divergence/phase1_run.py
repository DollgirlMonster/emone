#!/usr/bin/env python3
"""Phase 1 of the divergence overview (TinyTitan-8d0): teacher-forced surprisal
for every arithmetic switch, one arm per (model, text, switch).

Each arm runs `TinyTitanCLI --score`: all but the last SCORE tokens of a fixed
text are prefilled, the last SCORE are teacher-forced through decode, and one
negative log-likelihood per token is written. Every arm on a (model, text)
reads identical tokens, so any two arms compare token by token. Arms flip one
switch from the profile row's default; `base` is the row as shipped and
`base32` the same with 32-token prefill chunks, the smallest the CLI
accepts (the short-chunk paths: router small tiles, resident rows and the flash
key split all engage at 32 rows; small QMM covers 3-31 rows and cannot be
reached here).

    python3 benchmark/divergence/phase1_run.py [--models qwen36,ornith] [--texts prose,swift,python]

Resumable: an arm whose .nll exists is skipped. Nothing here downloads or
installs a model and nothing is killed: a busy machine waits, then stops.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "phase1-20261119"
SCORE = 512
GUARD = "TinyTitanServer|TinyTitanCLI|TinyTitanPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm"

MODELS = {
    "qwen36": ROOT / "models/qwen3.6_35B_A3B_4Bit",
    "ornith": ROOT / "models/ornith-1.5_35B_A3B_4Bit",
}

# The shipped prefill/decode numerics switches on these rows, all on.
SHIPPED_OFF = {
    "TINYTITAN_ROUTER_TOPK_SIMD": "0",
    "TINYTITAN_PREFILL_ROUTED_MPP": "0",
    "TINYTITAN_PREFILL_ROUTED_QMM": "0",
    "TINYTITAN_PREFILL_GDN_CHUNK": "0",
    "TINYTITAN_PREFILL_SG_QMM": "0",
    "TINYTITAN_PREFILL_DENSE_FLASH": "0",
    "TINYTITAN_PREFILL_ROUTER_LOGITS": "0",
    "TINYTITAN_ATTN_DECODE_LONG": "0",
}

# name: (env, chunk, extra CLI args, kind). kind says what the arm is compared
# with: "base"/"base32" for one flipped switch, "yard" for a yardstick.
ARMS = {
    "base": ({}, 8192, [], "baseline"),
    "base_rep": ({}, 8192, [], "base"),
    # The CLI was rebuilt mid-phase (scoring now honours SEQUENTIAL): the
    # rebuilt binary must reproduce `base` byte for byte.
    "base_rebuilt": ({}, 8192, [], "base"),
    "base32": ({}, 32, [], "baseline"),
    # Yardsticks.
    "seq": ({"TINYTITAN_PREFILL_SEQUENTIAL": "1"}, 8192, [], "base"),
    "kv16": ({}, 8192, ["--kv-bits", "16"], "base"),
    "kv4": ({}, 8192, ["--kv-bits", "4"], "base"),
    "all_shipped_off": (SHIPPED_OFF, 8192, [], "base"),
    "cand_8192": (
        {"TINYTITAN_PREFILL_MPP_WIDE": "1", "TINYTITAN_PREFILL_Q_QMM": "1"},
        8192, [], "base"),
    "cand_32": ({"TINYTITAN_PREFILL_FLASH_SPLIT": "1"}, 32, [], "base32"),
    # Shipped on, flipped off one at a time.
    "off_topk_simd": ({"TINYTITAN_ROUTER_TOPK_SIMD": "0"}, 8192, [], "base"),
    "off_routed_qmm": ({"TINYTITAN_PREFILL_ROUTED_QMM": "0"}, 8192, [], "base"),
    "off_routed_mpp": (
        {"TINYTITAN_PREFILL_ROUTED_MPP": "0", "TINYTITAN_PREFILL_ROUTED_QMM": "0"},
        8192, [], "base"),
    "off_gdn_chunk": ({"TINYTITAN_PREFILL_GDN_CHUNK": "0"}, 8192, [], "base"),
    "off_sg_qmm": ({"TINYTITAN_PREFILL_SG_QMM": "0"}, 8192, [], "base"),
    "off_dense_flash": ({"TINYTITAN_PREFILL_DENSE_FLASH": "0"}, 8192, [], "base"),
    "off_router_logits": ({"TINYTITAN_PREFILL_ROUTER_LOGITS": "0"}, 8192, [], "base"),
    "off_decode_long": ({"TINYTITAN_ATTN_DECODE_LONG": "0"}, 8192, [], "base"),
    "off_router_small_tiles": ({"TINYTITAN_ROUTER_SMALL_TILES": "0"}, 32, [], "base32"),
    "off_resident_rows": ({"TINYTITAN_PREFILL_RESIDENT_ROWS": "0"}, 32, [], "base32"),
    # Candidates, off by default, flipped on one at a time.
    "on_mpp_wide": ({"TINYTITAN_PREFILL_MPP_WIDE": "1"}, 8192, [], "base"),
    "on_q_qmm": ({"TINYTITAN_PREFILL_Q_QMM": "1"}, 8192, [], "base"),
    "on_flash_split": ({"TINYTITAN_PREFILL_FLASH_SPLIT": "1"}, 32, [], "base32"),
}


def busy() -> str:
    r = subprocess.run(["pgrep", "-fl", GUARD], capture_output=True, text=True)
    return r.stdout.strip()


def wait_idle(limit_s: int = 1800) -> None:
    start = time.time()
    while (b := busy()):
        if time.time() - start > limit_s:
            sys.exit(f"model processes still running after {limit_s}s, stopping:\n{b}")
        print(f"  waiting for: {b.splitlines()[0]}", flush=True)
        time.sleep(30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen36,ornith")
    ap.add_argument("--texts", default="prose,swift,python")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    cli = Path(subprocess.run(
        ["swift", "build", "-c", "release", "--show-bin-path"], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout.strip()) / "TinyTitanCLI"
    if not cli.exists():
        sys.exit(f"no release CLI at {cli}")
    for m in args.models.split(","):
        model = MODELS[m]
        if not (model / "verified-install.json").exists():
            print(f"{m}: not installed, not checked", flush=True)
            continue
        for t in args.texts.split(","):
            text = (OUT / "texts" / f"{t}.txt").read_text()
            d = OUT / m / t
            d.mkdir(parents=True, exist_ok=True)
            for name in args.arms.split(","):
                env_over, chunk, extra, _ = ARMS[name]
                nll, log = d / f"{name}.nll", d / f"{name}.log"
                if nll.exists():
                    continue
                wait_idle()
                env = dict(os.environ)
                env.update(env_over)
                cmd = [str(cli), "--model", str(model), "--prompt", text,
                       "--score", str(SCORE), "--score-out", str(nll),
                       "--prefill-chunk", str(chunk), "--max-context", "65536", *extra]
                t0 = time.time()
                with open(log, "w") as fh:
                    code = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                          timeout=3600).returncode
                dt = time.time() - t0
                with open(log, "a") as fh:
                    fh.write(f"\n[arm {name} exit={code} wall_s={dt:.1f} "
                             f"env={env_over} chunk={chunk} extra={extra}]\n")
                if code != 0 and nll.exists():
                    nll.unlink()
                print(f"{time.strftime('%H:%M:%S')} {m}/{t}/{name} exit={code} {dt:.0f}s",
                      flush=True)


if __name__ == "__main__":
    main()
