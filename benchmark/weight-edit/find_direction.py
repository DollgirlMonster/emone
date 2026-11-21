"""Find a looping direction in dense-MLP outputs from TINYTITAN_MLP_CAPTURE files.

usage: find_direction.py <out.npz> <capture.bin>...

Each capture is one generation. A decode position is labelled *looping* when
the WINDOW tokens ending at it already occurred earlier in the same
generation (an exact n-gram repeat), and *normal* when it precedes the first
such repeat by at least MARGIN tokens. Per layer we take the difference of
class means of the MLP output, unit-normalise it, and score how well the
projection separates the classes (Cohen's d). The npz holds, per layer, the
direction u, the score, and the mean |act| per intermediate unit on looping
positions (for column choice).
"""

import sys

import numpy as np

HIDDEN, INTER = 1024, 3584
REC = 12 + 2 * HIDDEN + 2 * INTER
WINDOW, MARGIN = 24, 64


def load(path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = raw.size // REC
    raw = raw[: n * REC].reshape(n, REC)
    head = raw[:, :12].copy().view(np.int32).reshape(n, 3)
    out = raw[:, 12 : 12 + 2 * HIDDEN].copy().view(np.float16).astype(np.float32)
    act = raw[:, 12 + 2 * HIDDEN :].copy().view(np.float16).astype(np.float32)
    return head, out, act


def labels(positions, tokens):
    """+1 looping, 0 normal, -1 unused, per unique position in order."""
    order = np.argsort(positions)
    pos, tok = positions[order], tokens[order]
    seen, lab, first = set(), np.full(pos.size, -1), None
    for i in range(pos.size):
        if i + 1 < WINDOW:
            continue
        key = tuple(tok[i + 1 - WINDOW : i + 1])
        if key in seen:
            lab[i] = 1
            if first is None:
                first = i
        seen.add(key)
    stop = (first if first is not None else pos.size) - MARGIN
    lab[: max(stop, 0)] = np.where(lab[: max(stop, 0)] == -1, 0, lab[: max(stop, 0)])
    return dict(zip(pos.tolist(), lab.tolist()))


def main():
    out_path, paths = sys.argv[1], sys.argv[2:]
    per_layer = {}
    for path in paths:
        head, out, act = load(path)
        layers = np.unique(head[:, 1])
        ref = head[:, 1] == layers[0]
        lab = labels(head[ref, 0], head[ref, 2])
        y = np.array([lab[p] for p in head[:, 0]])
        print(f"{path}: {ref.sum()} positions, {int((y[ref] == 1).sum())} looping, "
              f"{int((y[ref] == 0).sum())} normal")
        for L in layers:
            m = head[:, 1] == L
            d = per_layer.setdefault(int(L), {"o1": [], "o0": [], "a1": [], "p0": []})
            d["o1"].append(out[m & (y == 1)])
            d["o0"].append(out[m & (y == 0)])
            d["a1"].append(act[m & (y == 1)])
            d["p0"].append(head[m & (y == 0), 0])
    result = {}
    for L, d in sorted(per_layer.items()):
        o1, o0 = np.concatenate(d["o1"]), np.concatenate(d["o0"])
        if len(o1) < 50 or len(o0) < 50:
            continue
        u = o1.mean(0) - o0.mean(0)
        gap = float(np.linalg.norm(u))
        u /= gap
        p1, p0 = o1 @ u, o0 @ u
        pooled = np.sqrt((p1.var() + p0.var()) / 2)
        score = float((p1.mean() - p0.mean()) / pooled)
        # Position confound: the direction along which normal-region outputs
        # drift with position (least-squares slope), and its cosine with u.
        pos = np.concatenate(d["p0"]).astype(np.float32)
        pc = pos - pos.mean()
        drift = (pc[:, None] * (o0 - o0.mean(0))).sum(0) / (pc @ pc)
        cos_pos = float(drift @ u / (np.linalg.norm(drift) + 1e-12))
        result[f"c{L}"] = cos_pos
        dh = drift / (np.linalg.norm(drift) + 1e-12)
        up = u - (u @ dh) * dh
        result[f"q{L}"] = up / np.linalg.norm(up)  # u with the position drift removed
        result[f"u{L}"] = u
        result[f"d{L}"] = score
        result[f"a{L}"] = np.abs(np.concatenate(d["a1"])).mean(0)
        print(f"L{L:2d}  d'={score:6.2f}  |mean gap|={gap:7.3f}  "
              f"loop proj {p1.mean():7.3f}  normal proj {p0.mean():7.3f}  cos(u,pos drift)={cos_pos:5.2f}")
    np.savez(out_path, **result)


if __name__ == "__main__":
    main()
