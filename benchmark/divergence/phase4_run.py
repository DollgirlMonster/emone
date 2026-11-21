#!/usr/bin/env python3
"""Phase 4 of the divergence overview (Emone-8d0): task outcomes where the
short-chunk candidates actually engage.

Phase 3 asked each task cold (about 50 tokens, no context), so small QMM
(3-31-row chunks) and the flash key split (few rows against a deep cache) never
ran. Here every task is a follow-up: turn 1 is a ~16K-char documentation slice
followed by the task and "reply Ready" (4 tokens), turn 2 is "Answer the task
now." -- about 20 uncached rows prefilled against a ~5K-token cache, then the
answer is decoded. 60 arithmetic problems (the phase-3 30 plus 30 more from the
same seed) and the 15 code tasks, greedy at temperature 0.

    python3 benchmark/divergence/phase4_run.py [--models qwen36,ornith] [--arms base,...]

Resumable per (model, arm). Same server handling as phases 2 and 3.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase23_sets as sets  # noqa: E402
from phase23_run import (CODE_TOKENS, MATH_TOKENS, MODELS, ROOT, arm_server,  # noqa: E402
                         post)

OUT = Path(__file__).resolve().parent / "phase4-20261120"
ARMS = ["base", "on_small_qmm", "on_flash_split", "cand"]
SETUP = "Below is a task. Do not answer it yet: reply with the single word Ready."
ASK = "Answer the task now."


def tasks() -> list[dict]:
    docs = sets.doc_slices(12)
    out = []
    for i, t in enumerate(sets.math_tasks(60) + sets.code_tasks()):
        out.append({**t, "doc": i % len(docs), "setup": f"{docs[i % len(docs)]}\n\n---\n\n{SETUP}"
                    f"\n\n{t['prompt']}"})
    # Same document back to back, so the prompt cache can reuse its prefix.
    return sorted(out, key=lambda t: t["doc"])


def run_arm(model_key: str, arm: str) -> None:
    d = OUT / model_key / arm
    if (d / "done").exists():
        return
    with arm_server(OUT, model_key, arm) as model_id, open(d / "results.jsonl", "w") as results:
        for t in tasks():
            msgs = [{"role": "user", "content": t["setup"]}]
            r1 = post(model_id, msgs, 4)
            ready = r1["choices"][0]["message"].get("content") or ""
            msgs += [{"role": "assistant", "content": ready}, {"role": "user", "content": ASK}]
            r2 = post(model_id, msgs, MATH_TOKENS if t["kind"] == "math" else CODE_TOKENS)
            text = r2["choices"][0]["message"].get("content") or ""
            results.write(json.dumps({"id": t["id"], "ready": ready, "text": text,
                                      "usage1": r1.get("usage"), "usage2": r2.get("usage"),
                                      "wall_s": r1["_wall_s"] + r2["_wall_s"]}) + "\n")
            results.flush()
        (d / "done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        print(f"{time.strftime('%H:%M:%S')} {model_key}/{arm}: done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen36,ornith")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N tasks only")
    args = ap.parse_args()
    if args.limit:
        global tasks
        full = tasks
        tasks = lambda: full()[:args.limit]  # noqa: E731
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
