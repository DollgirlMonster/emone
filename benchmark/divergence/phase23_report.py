#!/usr/bin/env python3
"""Tabulate phases 2 and 3 of the divergence overview (TinyTitan-8d0).

Phase 2, token flips, every arm against `base` on the same model: per turn,
whether the greedy tokens are identical, the step of the first different token,
and base's top-1 minus top-2 logit gap at that step (a flip at a gap under 0.5
is a near-tie: either token was nearly as likely). Turns after a flip carry a
different history, so "conversations still identical" is the trajectory view.

Phase 3, outcomes: math answers checked against the computed value, code run
against its tests (python3 -I in an empty temporary directory, 10 s limit).
Paired against base: same outcome, and the discordant pairs (base-only pass /
arm-only pass) with an exact two-sided sign-test p.

    python3 benchmark/divergence/phase23_report.py > benchmark/divergence/phase23-20261120/report.md
"""

import json
import math
import re
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase23_sets as sets  # noqa: E402
from phase23_run import ARMS, OUT  # noqa: E402

LOGIT = re.compile(r"\[logit\] pos=(\d+) chosen=(\d+) top1=\d+:(-?[\d.]+) top2=\d+:(-?[\d.]+) "
                   r"margin=(-?[\d.]+)")
ACCEPT = re.compile(r"request chatcmpl-\S+ accepted")


def traces(log: Path) -> list[list[tuple[int, float]]]:
    """Per request (warm-up dropped): [(chosen token, margin)] by step."""
    reqs: list[list[tuple[int, float]]] = []
    for line in log.read_bytes().decode("utf-8", "replace").replace("\r", "").splitlines():
        if ACCEPT.search(line):
            reqs.append([])
        elif (m := LOGIT.search(line)) and reqs:
            reqs[-1].append((int(m.group(2)), float(m.group(5))))
    return reqs[1:]


def results(d: Path) -> list[dict]:
    return [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines()]


def math_ok(text: str, answer: int) -> bool:
    m = re.findall(r"ANSWER:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        return False
    try:
        return round(float(m[-1].replace(",", ""))) == answer
    except ValueError:
        return False


def code_ok(text: str, tests: list[str]) -> bool:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    if not m:
        return False
    src = m.group(1) + "\n" + "\n".join(f"assert {t}" for t in tests) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.py"
        p.write_text(src)
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=tmp, capture_output=True,
                               timeout=10, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0


def sign_p(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def phase2_row(base_tr, arm_tr, base_res, arm_res) -> str:
    idx = [i for i, r in enumerate(base_res) if r["phase"] == 2]
    same_turns, flips, near, steps = 0, 0, 0, []
    convs: dict[str, bool] = {}
    margins = []
    for i in idx:
        a, b = base_tr[i], arm_tr[i]
        k = next((s for s, (x, y) in enumerate(zip(a, b)) if x[0] != y[0]), None)
        if k is None and len(a) != len(b):
            k = min(len(a), len(b))
        cid = base_res[i]["id"]
        if k is None:
            same_turns += 1
            convs.setdefault(cid, True)
            continue
        convs[cid] = False
        flips += 1
        steps.append(k)
        if k < len(a):
            margins.append(a[k][1])
            near += a[k][1] < 0.5
    same_convs = sum(1 for v in convs.values() if v)
    med_step = f"{st.median(steps):.0f}" if steps else "-"
    med_m = f"{st.median(margins):.2f}" if margins else "-"
    return (f"{same_turns}/{len(idx)} | {same_convs}/{len(convs)} | {med_step} | "
            f"{near}/{flips} | {med_m}")


def outcomes(res: list[dict]) -> dict[str, bool]:
    by_id = {t["id"]: t for t in sets.tasks()}
    out = {}
    for r in res:
        if r["phase"] != 3:
            continue
        t = by_id[r["id"]]
        out[r["id"]] = (math_ok(r["text"], t["answer"]) if t["kind"] == "math"
                        else code_ok(r["text"], t["tests"]))
    return out


def main() -> None:
    for m in sorted(p.name for p in OUT.iterdir() if p.is_dir() and not p.name.startswith("_")):
        base_d = OUT / m / "base"
        if not (base_d / "done").exists():
            continue
        base_res, base_tr = results(base_d), traces(base_d / "server.log")
        base_out = outcomes(base_res)
        base_text = {r["id"]: r["text"] for r in base_res if r["phase"] == 3}
        print(f"\n## {m}\n")
        print("Phase 2, token flips against base (72 turns in 24 three-turn conversations).\n")
        print("| arm | turns identical | conversations identical | median first-flip step | "
              "flips at base gap < 0.5 | median base gap at flip |")
        print("|---|---|---|---|---|---|")
        rows3 = []
        for arm in ARMS:
            d = OUT / m / arm
            if not (d / "done").exists():
                continue
            res, tr = results(d), traces(d / "server.log")
            if len(tr) != len(res) or len(res) != len(base_res):
                print(f"| {arm} | trace/result count mismatch ({len(tr)} / {len(res)}) |||||")
                continue
            if arm != "base":
                print(f"| {arm} | {phase2_row(base_tr, tr, base_res, res)} |")
            out = outcomes(res)
            maths = sum(v for k, v in out.items() if k.startswith("math"))
            codes = sum(v for k, v in out.items() if k.startswith("code"))
            texts = {r["id"]: r["text"] for r in res if r["phase"] == 3}
            same_text = sum(texts[k] == base_text[k] for k in texts)
            agree = sum(out[k] == base_out[k] for k in out)
            moved_agree = sum(out[k] == base_out[k] for k in out if texts[k] != base_text[k])
            moved = len(out) - same_text
            bo = sum(base_out[k] and not out[k] for k in out)
            ao = sum(out[k] and not base_out[k] for k in out)
            rows3.append(f"| {arm} | {maths}/30 | {codes}/15 | {same_text}/{len(out)} | "
                         f"{agree}/{len(out)} | {moved_agree}/{moved} | {bo} / {ao} | "
                         f"{sign_p(bo, ao):.2f} |")
        print("\nPhase 3, task outcomes (30 arithmetic, 15 code).\n")
        print("| arm | math pass | code pass | text identical to base | outcome same as base | "
              "same outcome where text moved | base-only / arm-only pass | sign-test p |")
        print("|---|---|---|---|---|---|---|---|")
        print("\n".join(rows3))


if __name__ == "__main__":
    main()
