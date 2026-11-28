"""Read the attention machinery of the dense 0.8B straight from its weights.

No embeddings, no probes. Per full-attention head:
  qk  the attention score is sum_d (gq_d q_d)(gk_d k_d) after q_norm/k_norm,
      so gq*gk per dim says what the score is made of: rotary dims (position;
      NeoX pairs (i, i+32), frequency theta^(-2i/64), low i = fast = local)
      vs. the 192 non-rotary dims (content). Also the share of the QK
      bilinear form's energy in each part.
  ov  M = W_O[h] W_V[kv] diag(g_in), the head's map from residual to residual.
      Its eigenvalues say what it does to what it reads: real positive mass =
      copies it forward, negative = writes it back inverted (suppression).
Per Gated DeltaNet layer: per-head memory half-life from A_log and dt_bias
(decay = exp(-exp(A_log) * softplus(dt_bias)), at a zero input to dt), and
the conv1d taps: channels whose 4-tap kernel is dominated by the previous
position are a previous-token shift.
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gturbo_reader import GTurboWeights  # noqa: E402

w = GTurboWeights(os.environ.get("TT_MODEL", str(ROOT / "models/qwen3.5_0.8B_8Bit")))
P = "language_model.model.layers."
HEADS, KV, HD, ROT = 8, 2, 256, 64
FULL = [3, 7, 11, 15, 19, 23]


def g(name):
    v = w.get(name)
    return 1.0 + v if abs(float(v.mean())) < 0.5 else v


print("== full attention ==")
print(
    "head  rot-share  fast-rot-share  qk-rot-energy  |  ov: eig+ eig-  copy(sum re/sum |l|)  top eig"
)
for L in FULL:
    a = f"{P}{L}."
    gin = g(a + "input_layernorm.weight")
    Wq = w.get(a + "self_attn.q_proj.weight").reshape(HEADS, 2, HD, -1)[:, 0]
    Wk = w.get(a + "self_attn.k_proj.weight").reshape(KV, HD, -1)
    Wv = w.get(a + "self_attn.v_proj.weight").reshape(KV, HD, -1)
    Wo = w.get(a + "self_attn.o_proj.weight").reshape(-1, HEADS, HD)
    gqk = np.abs(g(a + "self_attn.q_norm.weight") * g(a + "self_attn.k_norm.weight"))
    rot_share = gqk[:ROT].sum() / gqk.sum()
    fast = np.r_[0:8, 32:40]  # the fastest four... eight rotary pairs
    fast_share = gqk[fast].sum() / gqk[:ROT].sum()
    for h in range(HEADS):
        kv = h // (HEADS // KV)
        # QK bilinear form in residual space, split by head dims.
        qd = Wq[h] * gin  # [HD x hidden]
        kd = Wk[kv] * gin
        e_rot = np.linalg.norm((qd[:ROT].T * gqk[:ROT]) @ kd[:ROT]) ** 2
        e_all = np.linalg.norm((qd.T * gqk) @ kd) ** 2
        M = (Wo[:, h, :] @ Wv[kv]) * gin  # [hidden x hidden], rank <= 256
        lam = np.linalg.eigvals(M)
        pos = lam.real[lam.real > 0].sum()
        neg = -lam.real[lam.real < 0].sum()
        copy = lam.real.sum() / np.abs(lam).sum()
        top = lam[np.argmax(np.abs(lam))]
        print(
            f"L{L:2d}h{h}  {rot_share:9.2f}  {fast_share:14.2f}  {e_rot / e_all:13.2f}  |  "
            f"{pos:5.2f} {neg:5.2f}  {copy:+18.2f}  {top.real:+.2f}{top.imag:+.2f}i",
            flush=True,
        )

print("\n== gated deltanet ==")
print("layer  half-life (tokens) per head, sorted                     prev-token conv channels")
for L in range(24):
    if L in FULL:
        continue
    a = f"{P}{L}.linear_attn."
    A = np.exp(w.get(a + "A_log"))
    dt = np.log1p(np.exp(w.get(a + "dt_bias")))
    decay = np.exp(-A * dt)
    half = np.log(0.5) / np.log(np.clip(decay, 1e-9, 1 - 1e-9))
    conv = w.get(a + "conv1d.weight")[:, 0, :]  # [channels x 4], last tap = current position
    prev = np.abs(conv[:, 2]) > 2 * np.abs(conv[:, [0, 1, 3]]).max(1)
    print(
        f"L{L:2d}   {' '.join(f'{x:6.1f}' for x in np.sort(half))}   {int(prev.sum()):4d}/{len(prev)}"
    )

print("\n== gated deltanet OV: out_proj . diag(norm) . W_v, per value head ==")
print(
    "layer  (half-life, copy) for heads with half-life > 100 tokens; then the strongest copier and suppressor"
)
VH, VD, KH, KD = 16, 128, 16, 128
for L in range(24):
    if L in FULL:
        continue
    a = f"{P}{L}.linear_attn."
    gin = g(f"{P}{L}.input_layernorm.weight")
    qkv = w.get(a + "in_proj_qkv.weight")  # [q (KH*KD) ; k (KH*KD) ; v (VH*VD)] x hidden
    Wv = qkv[2 * KH * KD :].reshape(VH, VD, -1)
    Wo = w.get(a + "out_proj.weight").reshape(-1, VH, VD)
    nw = g(a + "norm.weight")
    A = np.exp(w.get(a + "A_log"))
    dt = np.log1p(np.exp(w.get(a + "dt_bias")))
    half = np.log(0.5) / np.log(np.clip(np.exp(-A * dt), 1e-9, 1 - 1e-9))
    cs = []
    for h in range(VH):
        M = (Wo[:, h, :] * nw) @ Wv[h] * gin
        lam = np.linalg.eigvals(M)
        cs.append(lam.real.sum() / np.abs(lam).sum())
    cs = np.array(cs)
    longs = " ".join(f"({half[h]:.0f},{cs[h]:+.2f})" for h in np.argsort(-half) if half[h] > 100)
    print(
        f"L{L:2d}  {longs}   | max {cs.max():+.2f} h{int(cs.argmax())} (hl {half[cs.argmax()]:.0f})  "
        f"min {cs.min():+.2f} h{int(cs.argmin())} (hl {half[cs.argmin()]:.0f})"
    )
