#!/usr/bin/env python3
"""Tabulate phase 1 of the divergence overview (Emone-8d0).

For every arm, against its reference arm on the same (model, text):
  bias   = mean dNLL in nats, its standard error and t (does it make the
           model worse on average?)
  spread = median and 99th-percentile |dNLL| per token, and the largest
           (how hard it moves the distribution; what predicts token flips)
  speed  = the CLI's own score seconds (prefill + scoring) for the arm.
A switch row reads "on minus off": an off_* arm (shipped switch turned off)
has its sign reversed so every switch row is the cost of having it on.
Pooled columns average the three texts' paired differences.

    python3 benchmark/divergence/phase1_report.py > benchmark/divergence/phase1-20261119/report.md
"""

import math
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_run import ARMS, OUT  # noqa: E402


def load(p: Path):
    try:
        return [float(x) for x in p.read_text().split()]
    except FileNotFoundError:
        return None


def seconds(p: Path):
    try:
        m = re.search(r"\[score .*?seconds=([0-9.]+)\]", p.read_text())
        return float(m.group(1)) if m else None
    except FileNotFoundError:
        return None


def ref_of(name: str) -> str:
    kind = ARMS[name][3]
    return "base32" if kind == "base32" else "base"


def stats(d):
    n = len(d)
    mean = sum(d) / n
    sd = st.pstdev(d) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    a = sorted(abs(x) for x in d)
    p99 = a[min(n - 1, int(round(0.99 * (n - 1))))]
    return mean, se, (mean / se if se > 0 else 0.0), st.median(a), p99, a[-1]


def main() -> None:
    models = sorted(p.name for p in OUT.iterdir() if p.is_dir() and p.name != "texts")
    texts = ["prose", "swift", "python"]
    for m in models:
        print(f"\n## {m}\n")
        print(
            "| arm | vs | pooled bias nats (t) | pooled median / p99 / max abs | "
            + " | ".join(f"{t} bias (t)" for t in texts)
            + " | sec base -> arm (prose) |"
        )
        print("|---|---|---|---|" + "---|" * len(texts) + "---|")
        for name in ARMS:
            if ARMS[name][3] == "baseline":
                continue
            ref = ref_of(name)
            sign = -1.0 if name.startswith("off_") else 1.0
            pooled, cells = [], []
            for t in texts:
                a = load(OUT / m / t / f"{ref}.nll")
                b = load(OUT / m / t / f"{name}.nll")
                if not a or not b or len(a) != len(b):
                    cells.append("-")
                    continue
                d = [sign * (y - x) for x, y in zip(a, b, strict=True)]
                pooled += d
                mean, se, tt, *_ = stats(d)
                cells.append(f"{mean:+.4f} ({tt:+.1f})")
            if not pooled:
                continue
            mean, se, tt, med, p99, mx = stats(pooled)
            s_ref = seconds(OUT / m / "prose" / f"{ref}.log")
            s_arm = seconds(OUT / m / "prose" / f"{name}.log")
            sec = f"{s_ref:.0f} -> {s_arm:.0f}" if s_ref and s_arm else "-"
            print(
                f"| {name} | {ref} | {mean:+.4f} ({tt:+.1f}) | {med:.4f} / {p99:.3f} / {mx:.2f} | "
                + " | ".join(cells)
                + f" | {sec} |"
            )


if __name__ == "__main__":
    main()
