#!/usr/bin/env python3
"""Phases 2 and 3 of the divergence overview (TinyTitan-8d0), one server per arm.

Each arm starts its own TinyTitanServer (tools/serve.sh) with one switch set,
an empty prompt cache of its own (so no arm restores another arm's state) and
TINYTITAN_LOGIT_TRACE=1, warms the expert cache until every expert is resident,
then runs, greedy at temperature 0:

  phase 2  24 three-turn conversations (phase23_sets.conversations), each turn
           extending the arm's own history: the per-step top-2 trace shows where
           and how close each token flip is;
  phase 3  45 checkable tasks (30 generated arithmetic problems, 15 Python
           functions run against tests) for outcome agreement.

    python3 benchmark/divergence/phase23_run.py [--models qwen36,ornith] [--arms base,...]

Resumable per (model, arm): an arm with a finished marker is skipped. Nothing
here downloads or installs a model; it stops only the server it started.
"""

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase23_sets as sets  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SMOKE = os.environ.get("PHASE23_SMOKE") == "1"
OUT = Path(__file__).resolve().parent / ("phase23-smoke" if SMOKE else "phase23-20261120")
PORT = 8089
GUARD = "TinyTitanServer|TinyTitanCLI|TinyTitanPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm"

MODELS = {
    "qwen36": ("models/qwen3.6_35B_A3B_4Bit", "qwen3.6-35b-a3b_4-Bit"),
    "ornith": ("models/ornith-1.5_35B_A3B_4Bit", "ornith-1.5-35b-a3b_4-Bit"),
}

SHIPPED_OFF = {
    "TINYTITAN_ROUTER_TOPK_SIMD": "0",
    "TINYTITAN_PREFILL_ROUTED_MPP": "0",
    "TINYTITAN_PREFILL_ROUTED_QMM": "0",
    "TINYTITAN_PREFILL_GDN_CHUNK": "0",
    "TINYTITAN_PREFILL_SG_QMM": "0",
    "TINYTITAN_PREFILL_DENSE_FLASH": "0",
    "TINYTITAN_PREFILL_ROUTER_LOGITS": "0",
    "TINYTITAN_ATTN_DECODE_LONG": "0",
    "TINYTITAN_PREFILL_RESIDENT_ROWS": "0",
    "TINYTITAN_ROUTER_SMALL_TILES": "0",
}

# name: (environment, extra server args)
ARMS = {
    "base": ({}, []),
    "base_rep": ({}, []),
    "all_shipped_off": (SHIPPED_OFF, []),
    "off_resident_rows": ({"TINYTITAN_PREFILL_RESIDENT_ROWS": "0"}, []),
    "on_small_qmm": ({"TINYTITAN_PREFILL_SMALL_QMM": "1"}, []),
    "on_flash_split": ({"TINYTITAN_PREFILL_FLASH_SPLIT": "1"}, []),
    "cand": ({"TINYTITAN_PREFILL_SMALL_QMM": "1", "TINYTITAN_PREFILL_FLASH_SPLIT": "1"}, []),
    "kv16": ({}, ["--kv-bits", "16"]),
    "kv4": ({}, ["--kv-bits", "4"]),
}

TURN_TOKENS = 160
MATH_TOKENS = 700
CODE_TOKENS = 500


def busy() -> str:
    return subprocess.run(["pgrep", "-fl", GUARD], capture_output=True, text=True).stdout.strip()


def post(model_id: str, messages: list, max_tokens: int, timeout: int = 1800) -> dict:
    body = json.dumps({"model": model_id, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "seed": 0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    out["_wall_s"] = time.monotonic() - t0
    return out


def wait_health(limit: int = 300) -> None:
    deadline = time.time() + limit
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("server did not come up")


@contextlib.contextmanager
def arm_server(out: Path, model_key: str, arm: str):
    """Start this arm's server with an empty prompt cache of its own, warm it
    until every expert is resident, yield the model id; stop it on the way out."""
    install, model_id = MODELS[model_key]
    env_over, extra = ARMS[arm]
    d = out / model_key / arm
    if (b := busy()):
        sys.exit(f"model process already running, stopping:\n{b}")
    d.mkdir(parents=True, exist_ok=True)
    cache = out / "_cache" / f"{model_key}-{arm}"
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True)
    env = dict(os.environ)
    env.update(env_over)
    env.update({"TINYTITAN_LOGIT_TRACE": "1", "TINYTITAN_PROMPT_CACHE_ROOT": str(cache)})
    cmd = ["script", "-q", "/dev/null", "tools/serve.sh", install, "--reasoning", "off",
           "--no-live-trace", "--no-catalog", "--idle", "7200", "--port", str(PORT)]
    if extra:
        cmd += ["--", *extra]
    log = open(d / "server.log", "wb")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_health()
        # Warm-up: load the model, touch the expert set, then give the
        # background fill time to make every expert resident, so the short-chunk
        # resident paths take the same branch in every measured request.
        warm = sets.doc_slices(1, 30_000)[0]
        post(model_id, [{"role": "user", "content": warm + "\n\nReply OK."}], 1)
        time.sleep(45)
        print(f"{time.strftime('%H:%M:%S')} {model_key}/{arm}: warm", flush=True)
        yield model_id
    finally:
        pids = subprocess.run(["pgrep", "-f", "Release/TinyTitanServer"], capture_output=True,
                              text=True).stdout.split()
        for pid in pids:
            os.kill(int(pid), signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
        log.close()
        for _ in range(30):
            if not busy():
                break
            time.sleep(2)
        shutil.rmtree(cache, ignore_errors=True)


def run_arm(model_key: str, arm: str) -> None:
    d = OUT / model_key / arm
    if (d / "done").exists():
        return
    with arm_server(OUT, model_key, arm) as model_id, open(d / "results.jsonl", "w") as results:
        convs = sets.conversations()
        tasks = sets.tasks()
        if SMOKE:
            convs, tasks = [convs[0], convs[12]], [tasks[0], tasks[30]]
        for conv in convs:
            msgs = []
            for ti, turn in enumerate(conv["turns"]):
                msgs.append({"role": "user", "content": turn})
                r = post(model_id, msgs, TURN_TOKENS)
                text = r["choices"][0]["message"].get("content") or ""
                msgs.append({"role": "assistant", "content": text})
                results.write(json.dumps({"phase": 2, "id": conv["id"], "turn": ti, "text": text,
                                          "usage": r.get("usage"), "wall_s": r["_wall_s"]}) + "\n")
                results.flush()
        print(f"{time.strftime('%H:%M:%S')} {model_key}/{arm}: phase 2 done", flush=True)
        for task in tasks:
            budget = MATH_TOKENS if task["kind"] == "math" else CODE_TOKENS
            r = post(model_id, [{"role": "user", "content": task["prompt"]}], budget)
            text = r["choices"][0]["message"].get("content") or ""
            results.write(json.dumps({"phase": 3, "id": task["id"], "turn": 0, "text": text,
                                      "usage": r.get("usage"), "wall_s": r["_wall_s"]}) + "\n")
            results.flush()
        (d / "done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        print(f"{time.strftime('%H:%M:%S')} {model_key}/{arm}: phase 3 done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen36,ornith")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    for m in args.models.split(","):
        if not (ROOT / MODELS[m][0] / "verified-install.json").exists():
            print(f"{m}: not installed, not checked", flush=True)
            continue
        for arm in args.arms.split(","):
            try:
                run_arm(m, arm)
            except Exception as e:  # one arm's failure must not stop the batch
                print(f"{time.strftime('%H:%M:%S')} {m}/{arm}: FAILED {e!r}", flush=True)


if __name__ == "__main__":
    main()
