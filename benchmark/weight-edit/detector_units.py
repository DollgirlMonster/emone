"""Per-unit loop detectors in the dense-MLP intermediate activations.

usage: detector_units.py <out.npz> <capture.bin>...

Streams captures layer by layer (memmap), labels positions as in
find_direction.py, and keeps per-unit running sums for looping and normal
positions. A good detector fires on looping positions and stays near zero
otherwise, so each unit is scored by d' and by its normal-position mean |act|
relative to its looping mean (the "leak").
"""

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from find_direction import HIDDEN, INTER, REC, labels  # noqa: E402


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    stats = {}
    for path in paths:
        raw = np.memmap(path, dtype=np.uint8, mode="r")
        n = raw.size // REC
        raw = raw[: n * REC].reshape(n, REC)
        head = np.asarray(raw[:, :12]).copy().view(np.int32).reshape(n, 3)
        layers = np.unique(head[:, 1])
        ref = head[:, 1] == layers[0]
        lab = labels(head[ref, 0], head[ref, 2])
        y = np.array([lab[p] for p in head[:, 0]])
        for L in layers:
            for cls in (0, 1):
                m = np.nonzero((head[:, 1] == L) & (y == cls))[0]
                a = np.asarray(raw[m, 12 + 2 * HIDDEN :]).copy().view(np.float16).astype(np.float64)
                s = stats.setdefault((int(L), cls), [0, np.zeros(INTER), np.zeros(INTER), np.zeros(INTER)])
                s[0] += len(m)
                s[1] += a.sum(0)
                s[2] += (a * a).sum(0)
                s[3] += np.abs(a).sum(0)
        print(path, "done", flush=True)
    result = {}
    for L in sorted({k[0] for k in stats}):
        n0, s0, q0, b0 = stats[(L, 0)]
        n1, s1, q1, b1 = stats[(L, 1)]
        m0, m1 = s0 / n0, s1 / n1
        v0, v1 = q0 / n0 - m0**2, q1 / n1 - m1**2
        d = (m1 - m0) / np.sqrt((v0 + v1) / 2 + 1e-12)
        leak = (b0 / n0) / (b1 / n1 + 1e-12)
        result[f"d{L}"], result[f"m1_{L}"], result[f"m0_{L}"], result[f"leak{L}"] = d, m1, m0, leak
        order = np.argsort(-np.abs(d))[:6]
        print(f"L{L:2d} best |d'| " + "  ".join(
            f"u{j}:{d[j]:+.2f} (loop {m1[j]:+.3f}, normal {m0[j]:+.3f})" for j in order[:3]))
    np.savez(out, **result)


if __name__ == "__main__":
    main()
