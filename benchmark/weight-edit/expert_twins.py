"""Router twins and expert twins in a streamed Qwen-family MoE (TinyTitan-bzm). CPU only.

usage: expert_twins.py <model_dir> [pairs_per_layer] [layers]

Router twins: experts whose router rows (through the layer's MLP input norm,
when the checkpoint has one) point the same way -- the router can hardly
choose one without also scoring the other. Per layer: pairs above cos 0.8 and
0.9, and the number of routing clusters left when cos > 0.9 pairs are merged.

Expert twins: for the most similar router pairs in the chosen layers, run both
experts on the same inputs (rms-normalised token embeddings, through the same
norm) and compare outputs: mean cosine and relative L2 gap of y_a vs y_b.
Random expert pairs give the baseline. A router twin that is also an expert
twin can be routed to its partner with little change in what the layer writes,
which is what consolidation for a streamed model would rely on.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gturbo_reader import GTurboWeights, PackedExperts  # noqa: E402

model = Path(sys.argv[1])
per_layer = int(sys.argv[2]) if len(sys.argv) > 2 else 6
w = GTurboWeights(model)
packed = PackedExperts(model)
arch = json.loads((model / "manifest.json").read_text())["arch"]
NL, NE = arch["numLayers"], arch["numExperts"]
chosen = ([int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3
          else [NL // 4, NL // 2, NL - 6, NL - 3, NL - 1])
P = next(p for p in ("model.language_model.", "language_model.model.")
         if f"{p}layers.0.mlp.gate.weight" in w.entries)


def gain(L):
    name = f"{P}layers.{L}.post_attention_layernorm.weight"
    if name not in w.entries:
        return 1.0
    v = w.get(name)
    return 1.0 + v if abs(float(v.mean())) < 0.5 else v


def components(adj):
    n, seen, count = adj.shape[0], np.zeros(adj.shape[0], bool), 0
    for s in range(n):
        if seen[s]:
            continue
        count += 1
        stack = [s]
        seen[s] = True
        while stack:
            i = stack.pop()
            for j in np.nonzero(adj[i] & ~seen)[0]:
                seen[j] = True
                stack.append(j)
    return count


def expert(L, e, X):
    gate, up, down = (packed.tensor(L, e, n) for n in ("gate", "up", "down"))
    h = X @ gate.T
    return ((h / (1 + np.exp(-h))) * (X @ up.T)) @ down.T


E = w.get(f"{P}embed_tokens.weight")
rng = np.random.default_rng(0)
X0 = E[rng.choice(np.arange(1000, 150000), 512, replace=False)]
X0 = X0 / np.sqrt((X0 ** 2).mean(1, keepdims=True))

print(f"{NL} layers, {NE} experts, top-{arch['topKExperts']}, prefix {P}")
print("layer  pairs>0.8  pairs>0.9  clusters@0.9  max-cos")
rows = {}
for L in range(NL):
    R = w.get(f"{P}layers.{L}.mlp.gate.weight") * gain(L)
    Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
    C = Rn @ Rn.T
    np.fill_diagonal(C, -1)
    rows[L] = C
    print(f"L{L:2d}   {int((C > 0.8).sum() // 2):7d}   {int((C > 0.9).sum() // 2):7d}   "
          f"{components(C > 0.9):8d}      {C.max():.3f}", flush=True)

print("\nexpert twins: output similarity on shared inputs (cos, relative gap)")
for L in chosen:
    C = rows[L]
    X = X0 * gain(L)
    iu = np.triu_indices(NE, 1)
    order = np.argsort(-C[iu])[:per_layer]
    twins = [(iu[0][k], iu[1][k]) for k in order]
    rand = [tuple(rng.choice(NE, 2, replace=False)) for _ in range(per_layer)]
    for label, pairs in (("router twins", twins), ("random pairs", rand)):
        cs, gaps = [], []
        for a, b in pairs:
            ya, yb = expert(L, a, X), expert(L, b, X)
            cs.append(float(np.mean((ya * yb).sum(1) / (np.linalg.norm(ya, axis=1) * np.linalg.norm(yb, axis=1) + 1e-9))))
            gaps.append(float(np.linalg.norm(ya - yb) / (np.linalg.norm(ya) + 1e-9)))
        desc = ", ".join(f"{a}/{b} r{C[a, b]:.2f} y{c:+.2f}" for (a, b), c in zip(pairs, cs))
        print(f"L{L:2d} {label:12s} output cos mean {np.mean(cs):+.2f}, gap {np.mean(gaps):.2f}   [{desc}]",
              flush=True)
