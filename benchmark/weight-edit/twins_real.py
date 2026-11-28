"""Router twins on real hidden states (Emone-bzm). Reads a `<capture>.moe` file
(EMONE_MLP_CAPTURE on a routed model: position, layer, top-k ids, expert input).

usage: twins_real.py <model_dir> <capture.moe> [pairs_per_layer]

For each captured layer: the closest router pairs (as in expert_twins.py) and
random pairs, compared on the captured inputs -- the expert outputs' mean cosine
and relative gap -- and how often each pair is actually chosen together:
P(both in the top-k | one is), against the k/experts base rate.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from gturbo_reader import GTurboWeights, PackedExperts  # noqa: E402

model, cap = Path(sys.argv[1]), sys.argv[2]
per_layer = int(sys.argv[3]) if len(sys.argv) > 3 else 12
arch = json.loads((model / "manifest.json").read_text())["arch"]
NE, K, D = arch["numExperts"], arch["topKExperts"], arch["hiddenSize"]
w, packed = GTurboWeights(model), PackedExperts(model)
P = next(
    p
    for p in ("model.language_model.", "language_model.model.")
    if f"{p}layers.0.mlp.gate.weight" in w.entries
)

rec = 4 * (2 + K) + 2 * D
raw = np.fromfile(cap, dtype=np.uint8)
raw = raw[: raw.size // rec * rec].reshape(-1, rec)
head = raw[:, : 4 * (2 + K)].copy().view(np.int32).reshape(-1, 2 + K)
X_all = raw[:, 4 * (2 + K) :].copy().view(np.float16).astype(np.float32)


def run(L, e, X):
    gate, up, down = (packed.tensor(L, e, n) for n in ("gate", "up", "down"))
    h = X @ gate.T
    return ((h / (1 + np.exp(-h))) * (X @ up.T)) @ down.T


def compare(L, pairs, X, chosen):
    cs, gaps, joint = [], [], []
    for a, b in pairs:
        ya, yb = run(L, a, X), run(L, b, X)
        cs.append(
            np.mean(
                (ya * yb).sum(1) / (np.linalg.norm(ya, axis=1) * np.linalg.norm(yb, axis=1) + 1e-9)
            )
        )
        gaps.append(np.linalg.norm(ya - yb) / (np.linalg.norm(ya) + 1e-9))
        either = chosen[:, a] | chosen[:, b]
        joint.append((chosen[:, a] & chosen[:, b]).sum() / max(1, either.sum()))
    return np.mean(cs), np.mean(gaps), np.mean(joint), int(sum(1 for j in joint if j > 0))


rng = np.random.default_rng(0)
print(f"{len(X_all)} captured rows; base rate for a pair in the top-{K}: {K / NE:.3f}")
for L in sorted(set(head[:, 1].tolist())):
    m = head[:, 1] == L
    X = X_all[m]
    chosen = np.zeros((m.sum(), NE), bool)
    for i, ids in enumerate(head[m, 2:]):
        chosen[i, ids] = True
    R = w.get(f"{P}layers.{L}.mlp.gate.weight")
    Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
    C = Rn @ Rn.T
    np.fill_diagonal(C, -1)
    iu = np.triu_indices(NE, 1)
    twins = [(iu[0][k], iu[1][k]) for k in np.argsort(-C[iu])[:per_layer]]
    rand = [tuple(rng.choice(NE, 2, replace=False)) for _ in range(per_layer)]
    # Pairs that really co-fire most in this capture, whatever their router cosine.
    co = chosen.T.astype(np.int32) @ chosen.astype(np.int32)
    np.fill_diagonal(co, 0)
    cofire = [(iu[0][k], iu[1][k]) for k in np.argsort(-co[iu])[:per_layer]]
    for label, pairs in (("router twins", twins), ("co-firing", cofire), ("random", rand)):
        c, gap, joint, seen = compare(L, pairs, X, chosen)
        rc = np.mean([C[a, b] for a, b in pairs])
        print(
            f"L{L:2d} {label:12s} router cos {rc:.2f} | output cos {c:+.2f} gap {gap:.2f} | "
            f"chosen together {joint:.2f} of the time ({seen}/{len(pairs)} pairs ever)",
            flush=True,
        )
