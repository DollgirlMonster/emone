#!/usr/bin/env python3
"""Tabulate phase 4 of the divergence overview (Emone-8d0): task outcomes
asked as follow-ups after a long document, every arm paired against base.

    python3 benchmark/divergence/phase4_report.py > benchmark/divergence/phase4-20261120/report.md
"""

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase23_sets as sets  # noqa: E402
from phase23_report import code_ok, math_ok, sign_p, traces  # noqa: E402
from phase4_run import ARMS, OUT  # noqa: E402

BY_ID = {t["id"]: t for t in sets.math_tasks(60) + sets.code_tasks()}


def load(d: Path):
    res = [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines()]
    tr = traces(d / "server.log")
    out = {}
    for r in res:
        t = BY_ID[r["id"]]
        out[r["id"]] = (math_ok(r["text"], t["answer"]) if t["kind"] == "math"
                        else code_ok(r["text"], t["tests"]))
    return res, tr, out


def main() -> None:
    for m in sorted(p.name for p in OUT.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if not (OUT / m / "base" / "done").exists():
            continue
        bres, btr, bout = load(OUT / m / "base")
        rows = [r["usage2"]["prompt_tokens"] - r["usage2"]["prompt_tokens_details"]["cached_tokens"]
                for r in bres]
        ctx = [r["usage2"]["prompt_tokens"] for r in bres]
        print(f"\n## {m}\n")
        print(f"Turn 2 prefilled {min(rows)}-{max(rows)} uncached rows against {min(ctx)}-{max(ctx)} "
              f"tokens of context; turn 1 replies other than Ready: "
              f"{sum(r['ready'].strip(' .') != 'Ready' for r in bres)}/{len(bres)}.\n")
        print("| arm | math pass | code pass | answer tokens identical | first flip step (median) | "
              "outcome same | same outcome where text moved | base-only / arm-only | sign-test p |")
        print("|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            d = OUT / m / arm
            if not (d / "done").exists():
                continue
            res, tr, out = load(d)
            if len(tr) != 2 * len(res) or len(res) != len(bres):
                print(f"| {arm} | trace/result count mismatch ({len(tr)} / {len(res)}) ||||||||")
                continue
            ids = [r["id"] for r in res]
            same, steps = 0, []
            for i in range(len(res)):
                a, b = [x[0] for x in btr[2 * i + 1]], [x[0] for x in tr[2 * i + 1]]
                if a == b:
                    same += 1
                else:
                    steps.append(next((k for k, (x, y) in enumerate(zip(a, b)) if x != y),
                                      min(len(a), len(b))))
            moved = [k for i, k in enumerate(ids) if res[i]["text"] != bres[i]["text"]]
            maths = sum(v for k, v in out.items() if k.startswith("math"))
            codes = sum(v for k, v in out.items() if k.startswith("code"))
            agree = sum(out[k] == bout[k] for k in ids)
            bo = sum(bout[k] and not out[k] for k in ids)
            ao = sum(out[k] and not bout[k] for k in ids)
            print(f"| {arm} | {maths}/60 | {codes}/15 | {same}/{len(res)} | "
                  f"{st.median(steps) if steps else '-'} | {agree}/{len(ids)} | "
                  f"{sum(out[k] == bout[k] for k in moved)}/{len(moved)} | {bo} / {ao} | "
                  f"{sign_p(bo, ao):.2f} |")


if __name__ == "__main__":
    main()
