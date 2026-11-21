"""A weights-only map of a Qwen-family MoE's experts (Emone-bzm). CPU only.

usage: moe_map.py <model_dir> [layer,...]   (default layers 0,1,2,20,37,38,39)

Routers, every layer (resident; read through the layer's post-attention norm):
  dup      pairs of experts whose router rows point the same way (cos > 0.8),
           i.e. experts the router can barely tell apart
  load     route a sample of token embeddings straight through the router
           (direct path, so meaningful only in the first layers): how evenly
           the top-k choices spread, and how many experts are never chosen
Experts, chosen layers (streamed weights, dequantized):
  writes   the tokens each expert's top output direction (first left singular
           vector of down) pushes through the final norm and tied/untied head
  attract  the tokens whose embeddings most raise that expert's router logit
  focus    how lexically specific an expert looks: the z-score of its best
           token in the write lens; the most specific experts are listed
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gturbo_reader import GTurboWeights, PackedExperts  # noqa: E402

model = Path(sys.argv[1])
chosen = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 20, 37, 38, 39]
w = GTurboWeights(model)
packed = PackedExperts(model)
P = "language_model.model."
arch = json.loads((model / "manifest.json").read_text())["arch"]
NL, NE, K = arch["numLayers"], arch["numExperts"], arch["topKExperts"]
vocab = {i: s for s, i in json.load(open(model / "tokenizer/tokenizer.json"))["model"]["vocab"].items()}


def tok(i):
    return vocab.get(int(i), f"<{i}>").replace("Ġ", "·").replace("Ċ", "\\n")[:10]


def g(name):
    v = w.get(name)
    return 1.0 + v if abs(float(v.mean())) < 0.5 else v


E = w.get(P + "embed_tokens.weight")
head = w.get("language_model.lm_head.weight") if "language_model.lm_head.weight" in w.entries else E
U = head * g(P + "norm.weight")
rng = np.random.default_rng(0)
sample = rng.choice(np.arange(1000, min(E.shape[0], 150000)), 20000, replace=False)
X0 = E[sample] / np.sqrt((E[sample] ** 2).mean(1, keepdims=True))

print(f"{NL} layers, {NE} experts, top-{K}")
print("\n== routers ==")
print("layer  dup-pairs  max-cos  | direct-path load: never-chosen  top-expert-share  entropy/max")
for L in range(NL):
    R = w.get(f"{P}layers.{L}.mlp.gate.weight") * g(f"{P}layers.{L}.post_attention_layernorm.weight")
    Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
    C = Rn @ Rn.T
    np.fill_diagonal(C, -1)
    dup = int((C > 0.8).sum() // 2)
    top = np.argpartition(-(X0 @ R.T), K, axis=1)[:, :K]
    counts = np.bincount(top.ravel(), minlength=NE)
    p = counts / counts.sum()
    ent = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(NE)
    print(f"L{L:2d}   {dup:6d}   {C.max():6.2f}   | {int((counts == 0).sum()):4d}   "
          f"{p.max():6.3f}   {ent:5.3f}")

for L in chosen:
    print(f"\n== layer {L} experts ==")
    R = w.get(f"{P}layers.{L}.mlp.gate.weight") * g(f"{P}layers.{L}.post_attention_layernorm.weight")
    vecs, norms = [], []
    for e in range(NE):
        D = packed.tensor(L, e, "down")  # [hidden x inter]
        u, s, _ = np.linalg.svd(D, full_matrices=False)
        vecs.append(u[:, 0] * np.sign(u[:, 0] @ (U.mean(0)) + 1e-12))
        norms.append(s[0])
    V = np.stack(vecs, 1)  # [hidden x experts]
    lens = U @ V  # [vocab x experts]
    z = (lens.max(0) - lens.mean(0)) / lens.std(0)
    attract = E[sample] @ R.T  # [tokens x experts]
    order = np.argsort(-z)
    print(f"  write-lens focus z: median {np.median(z):.1f}, max {z.max():.1f}; "
          f"top singular value median {np.median(norms):.2f}")
    for e in order[:10]:
        hi = np.argsort(-lens[:, e])[:6]
        lo = np.argsort(lens[:, e])[:4]
        at = sample[np.argsort(-attract[:, e])[:5]]
        print(f"  e{e:3d} z{z[e]:5.1f}  writes +[{' '.join(tok(i) for i in hi)}] "
              f"-[{' '.join(tok(i) for i in lo)}]  attracted by [{' '.join(tok(i) for i in at)}]")
