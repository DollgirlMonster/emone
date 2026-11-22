from pathlib import Path
"""Greedy GSM8K slice through EmoneCLI, one process per problem.

usage: run_gsm8k.py <out.jsonl> <start> <count> <max_new> [extra CLI args...]
Writes one JSON row per problem: id, gold, pred, correct, tokens, capped, text.
Greedy unless TT_SAMPLE=1 (T 0.6, top-p 0.95, top-k 20, presence penalty 1.5,
seed = problem id).
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time

REPO = str(Path(__file__).resolve().parents[2])
CLI = os.environ.get("TT_CLI", f"{REPO}/.build/release/EmoneCLI")
MODEL = f"{REPO}/models/qwen3.5_0.8B_8Bit"
DATA = os.environ.get("GSM8K_JSONL", os.path.join(os.path.dirname(__file__), "data/gsm8k_test.jsonl"))
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def number(s):
    s = s.replace(",", "").replace("$", "").strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def extract(text):
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxed:
        return number(boxed[-1])
    tail = text.split("</think>")[-1]
    return number(tail)


def decoding(problem_id):
    """Greedy by default; TT_SAMPLE=1 uses the sampling of the MIT CSAIL Exp 8
    evaluation (Qwen's thinking-mode settings), seeded by the problem id so an
    original and an edited run draw from the same stream."""
    if os.environ.get("TT_SAMPLE") != "1":
        return ["--temperature", "0"]
    return ["--temperature", "0.6", "--top-p", "0.95", "--top-k", "20",
            "--presence-penalty", "1.5", "--seed", str(problem_id)]


def main():
    out, start, count, max_new = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    extra = sys.argv[5:]
    rows = [json.loads(l) for l in open(DATA)][start : start + count]
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out)}
    with open(out, "a") as sink:
        for i, row in enumerate(rows, start):
            if i in done:
                continue
            gold = number(row["answer"].split("####")[-1])
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump([{"role": "user", "content": row["question"] + SUFFIX}], f)
                msgs = f.name
            t0 = time.time()
            env = dict(os.environ)
            if "TT_CAPTURE_DIR" in env:
                env["EMONE_MLP_CAPTURE"] = f"{env['TT_CAPTURE_DIR']}/cap_{i}.bin"
            p = subprocess.run(
                [CLI, "--model", MODEL, "--messages-file", msgs, "--thinking", "on",
                 *decoding(i), "--max-new", max_new, *extra],
                capture_output=True, text=True, env=env)
            os.unlink(msgs)
            text = p.stdout
            m = re.search(r"stop=(\w+) .*new=(\d+)tok", p.stderr)
            stop, tokens = (m.group(1), int(m.group(2))) if m else (None, None)
            pred = extract(text)
            rec = {
                "id": i, "gold": gold, "pred": pred,
                "correct": pred is not None and gold is not None and abs(pred - gold) < 1e-6,
                "tokens": tokens, "stop": stop, "capped": stop == "maxTokens",
                "seconds": round(time.time() - t0, 1), "rc": p.returncode,
                "stderr_tail": p.stderr[-400:], "text": text,
            }
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            print(i, rec["correct"], rec["tokens"], rec["capped"], rec["seconds"], flush=True)


if __name__ == "__main__":
    main()
