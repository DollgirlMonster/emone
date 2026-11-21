"""Trace the repetition circuit of the dense 0.8B through its weights (CPU only).

usage: trace_repetition.py <dirs.npz>   (recipe.py direction output: u15, u19)

Anchors: the repetition directions u15 / u19 (residual after layers 15 / 19,
repetitive minus productive matched branches) and layer 19 MLP neuron 11, the
strongest writer of u19. Every score is relative to a random direction, so 1.0
means "no more than chance" and the ranking is comparable across component
types:

  writers   how strongly a component's output space points along u:
            ||W_out^T u|| / (||W_out||_F / sqrt(hidden)) for heads, |cos| of
            the down_proj column for MLP units
  readers   how strongly a later component's inputs pick up u, through that
            layer's own input norm gain (q, k, v, gate of full attention;
            q, k, v, beta, decay, z of DeltaNet; gate/up rows of MLP units)
  neuron    L19 n11: what writes into its gate/up rows (upstream heads' OV
            output spaces), what tokens its own write promotes, and which
            tokens' embeddings switch it on directly
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gturbo_reader import GTurboWeights  # noqa: E402

MODEL = os.environ.get("TT_MODEL", str(ROOT / "models/qwen3.5_0.8B_8Bit"))
w = GTurboWeights(MODEL)
P = "language_model.model."
D, HEADS, KV, HD = 1024, 8, 2, 256
FULL = [3, 7, 11, 15, 19, 23]
VH, VD, KH, KD = 16, 128, 16, 128
vocab = {i: s for s, i in json.load(open(f"{MODEL}/tokenizer/tokenizer.json"))["model"]["vocab"].items()}


def g(name):
    v = w.get(name)
    return 1.0 + v if abs(float(v.mean())) < 0.5 else v


def tok(i):
    return vocab.get(int(i), f"<{i}>").replace("Ġ", "·").replace("Ċ", "\\n")[:12]


def out_score(Wout, u):
    """How much an output block [D x k] can write along u, vs a random direction."""
    return float(np.linalg.norm(Wout.T @ u) / (np.linalg.norm(Wout) / np.sqrt(D)))


def in_score(Win, u):
    """How much an input block [k x D] (gain already folded in) reads u, vs random."""
    return float(np.linalg.norm(Win @ u) / (np.linalg.norm(Win) / np.sqrt(D)))


dirs = np.load(sys.argv[1])
u15, u19 = dirs["u15"], dirs["u19"]
print(f"cos(u15, u19) = {u15 @ u19:+.2f}")
E = w.get(P + "embed_tokens.weight")
gf = g(P + "norm.weight")
lens = E @ (gf * u19)
print("u19 logit lens  +", [tok(i) for i in np.argsort(-lens)[:10]])
print("                -", [tok(i) for i in np.argsort(lens)[:10]])

# ---------------------------------------------------------------- writers
print("\n== writers of u19 (layers 0-19), x chance ==")
rows = []
for L in range(20):
    a = f"{P}layers.{L}."
    if L in FULL:
        Wo = w.get(a + "self_attn.o_proj.weight")
        for h in range(HEADS):
            rows.append((out_score(Wo[:, h * HD:(h + 1) * HD], u19), f"L{L} attn h{h}"))
    else:
        Wo = w.get(a + "linear_attn.out_proj.weight")
        for h in range(VH):
            rows.append((out_score(Wo[:, h * VD:(h + 1) * VD], u19), f"L{L} dnet h{h}"))
    Wd = w.get(a + "mlp.down_proj.weight")
    cos = (u19 @ Wd) / np.linalg.norm(Wd, axis=0)
    for j in np.argsort(-np.abs(cos))[:3]:
        rows.append((float(abs(cos[j]) * np.sqrt(D)), f"L{L} mlp n{j} ({'+' if cos[j] > 0 else '-'})"))
rows.sort(reverse=True)
for s, name in rows[:24]:
    print(f"  {s:6.2f}  {name}")

# ---------------------------------------------------------------- readers
print("\n== readers of u19 (layers 20-23), x chance ==")
rows = []
for L in range(20, 24):
    a = f"{P}layers.{L}."
    gin, gpost = g(a + "input_layernorm.weight"), g(a + "post_attention_layernorm.weight")
    if L in FULL:
        Wq = w.get(a + "self_attn.q_proj.weight").reshape(HEADS, 2, HD, D)
        Wk = w.get(a + "self_attn.k_proj.weight").reshape(KV, HD, D)
        Wv = w.get(a + "self_attn.v_proj.weight").reshape(KV, HD, D)
        for h in range(HEADS):
            rows.append((in_score(Wq[h, 0] * gin, u19), f"L{L} attn h{h} query"))
            rows.append((in_score(Wq[h, 1] * gin, u19), f"L{L} attn h{h} output-gate"))
        for k in range(KV):
            rows.append((in_score(Wk[k] * gin, u19), f"L{L} attn kv{k} key"))
            rows.append((in_score(Wv[k] * gin, u19), f"L{L} attn kv{k} value"))
    else:
        qkv = w.get(a + "linear_attn.in_proj_qkv.weight") * gin
        z = w.get(a + "linear_attn.in_proj_z.weight") * gin
        b = w.get(a + "linear_attn.in_proj_b.weight") * gin
        da = w.get(a + "linear_attn.in_proj_a.weight") * gin
        rows.append((in_score(qkv[:KH * KD], u19), f"L{L} dnet query"))
        rows.append((in_score(qkv[KH * KD:2 * KH * KD], u19), f"L{L} dnet key"))
        rows.append((in_score(qkv[2 * KH * KD:], u19), f"L{L} dnet value"))
        rows.append((in_score(b, u19), f"L{L} dnet beta (write strength)"))
        rows.append((in_score(da, u19), f"L{L} dnet decay"))
        rows.append((in_score(z, u19), f"L{L} dnet output-gate z"))
    Wg = w.get(a + "mlp.gate_proj.weight") * gpost
    cos = (Wg @ u19) / np.linalg.norm(Wg, axis=1)
    Wdn = w.get(a + "mlp.down_proj.weight")
    for j in np.argsort(-np.abs(cos))[:3]:
        lj = E @ (gf * Wdn[:, j])
        rows.append((float(abs(cos[j]) * np.sqrt(D)),
                     f"L{L} mlp n{j} gate ({'+' if cos[j] > 0 else '-'}) writes +{[tok(i) for i in np.argsort(-lj)[:4]]}"))
rows.sort(reverse=True)
for s, name in rows[:22]:
    print(f"  {s:6.2f}  {name}")

# ---------------------------------------------------------------- neuron 11
print("\n== L19 mlp neuron 11 ==")
a = f"{P}layers.19."
gpost = g(a + "post_attention_layernorm.weight")
gate = w.get(a + "mlp.gate_proj.weight")[11] * gpost
up = w.get(a + "mlp.up_proj.weight")[11] * gpost
col = w.get(a + "mlp.down_proj.weight")[:, 11]
print(f"  down column: |w| {np.linalg.norm(col):.3f}, cos with u19 {col @ u19 / np.linalg.norm(col):+.2f}, "
      f"with u15 {col @ u15 / np.linalg.norm(col):+.2f}")
lc = E @ (gf * col)
print("  its write promotes +", [tok(i) for i in np.argsort(-lc)[:10]])
print("                      -", [tok(i) for i in np.argsort(lc)[:10]])
print(f"  gate row reads u19 {gate @ u19 / np.linalg.norm(gate):+.2f}, u15 {gate @ u15 / np.linalg.norm(gate):+.2f}; "
      f"up row reads u19 {up @ u19 / np.linalg.norm(up):+.2f}")
X = E[np.random.default_rng(0).choice(np.arange(1000, 60000), 20000, replace=False)]
X = X / np.sqrt((X * X).mean(1, keepdims=True))
pre = (X @ gate) / (1 + np.exp(-(X @ gate))) * (X @ up)
ids = np.random.default_rng(0).choice(np.arange(1000, 60000), 20000, replace=False)
print("  embeddings that drive it most (direct path):", [tok(ids[i]) for i in np.argsort(-np.abs(pre))[:12]])
print("  upstream feeders of its gate row (OV output space . gate, x chance):")
feed = []
gu = gate / np.linalg.norm(gate)
for L in range(19):
    b = f"{P}layers.{L}."
    if L in FULL:
        Wo = w.get(b + "self_attn.o_proj.weight")
        for h in range(HEADS):
            feed.append((out_score(Wo[:, h * HD:(h + 1) * HD], gu), f"L{L} attn h{h}"))
    else:
        Wo = w.get(b + "linear_attn.out_proj.weight")
        for h in range(VH):
            feed.append((out_score(Wo[:, h * VD:(h + 1) * VD], gu), f"L{L} dnet h{h}"))
    Wd = w.get(b + "mlp.down_proj.weight")
    cos = (gu @ Wd) / np.linalg.norm(Wd, axis=0)
    j = int(np.argmax(np.abs(cos)))
    feed.append((float(abs(cos[j]) * np.sqrt(D)), f"L{L} mlp n{j} ({'+' if cos[j] > 0 else '-'})"))
feed.sort(reverse=True)
for s, name in feed[:14]:
    print(f"    {s:6.2f}  {name}")
