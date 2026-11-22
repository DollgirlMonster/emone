"""Weight-only scores for every full-attention head of the dense 0.8B.

Direct path from token embeddings (ignores what earlier layers add, so it is
a proxy that is crudest in the middle of the stack), on a sample of ordinary
tokens:

  same-token  the query of token t against the keys of all sampled tokens,
              content (non-rotary) dims only, after q_norm / k_norm: the
              z-score of the diagonal against its row. High = the head looks
              back for earlier copies of the current token, position-free.
  copy        the OV circuit through o_proj, the final-norm gain and the
              tied embedding: the z-score of logit(t) when attending to t.
              High = whatever the head attends to, it pushes as the next token.

A head high on both is the classic repetition engine. CPU only.
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gturbo_reader import GTurboWeights  # noqa: E402

MODEL = os.environ.get("TT_MODEL", str(ROOT / "models/qwen3.5_0.8B_8Bit"))
HEADS, KV, HD, ROT = 8, 2, 256, 64
FULL = [3, 7, 11, 15, 19, 23]
w = GTurboWeights(MODEL)
P = "language_model.model."


def gain(name):
    g = w.get(name)
    # Zero-centred RMSNorm weights (scale = 1 + w) sit near 0; plain ones near 1+.
    return 1.0 + g if abs(float(g.mean())) < 0.5 else g


def rms(x):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)


def zdiag(m):
    d = np.diag(m)
    off = (m.sum(1) - d) / (m.shape[1] - 1)
    sd = np.sqrt(((m - off[:, None]) ** 2).sum(1) / (m.shape[1] - 1))
    return float(np.median((d - off) / (sd + 1e-9)))


E = w.get(P + "embed_tokens.weight")
rng = np.random.default_rng(0)
toks = rng.choice(np.arange(1000, 60000), 1500, replace=False)
X0 = rms(E[toks])
gf = gain(P + "norm.weight")
U = E[toks] * gf  # unembedding rows for the sampled tokens
print("norm convention: input_layernorm mean",
      round(float(w.get(P + "layers.3.input_layernorm.weight").mean()), 3))
rows = []
for L in FULL:
    a = f"{P}layers.{L}."
    X = X0 * gain(a + "input_layernorm.weight")
    Wq = w.get(a + "self_attn.q_proj.weight").reshape(HEADS, 2, HD, -1)[:, 0]  # query half
    Wk = w.get(a + "self_attn.k_proj.weight").reshape(KV, HD, -1)
    Wv = w.get(a + "self_attn.v_proj.weight").reshape(KV, HD, -1)
    Wo = w.get(a + "self_attn.o_proj.weight").reshape(-1, HEADS, HD)
    gq, gk = gain(a + "self_attn.q_norm.weight"), gain(a + "self_attn.k_norm.weight")
    for h in range(HEADS):
        kv = h // (HEADS // KV)
        q = rms(X @ Wq[h].T) * gq
        k = rms(X @ Wk[kv].T) * gk
        same = zdiag(q[:, ROT:] @ k[:, ROT:].T)
        ov = (X @ Wv[kv].T) @ Wo[:, h, :].T  # [tokens x hidden]
        copy = zdiag(ov @ U.T)
        rows.append((L, h, same, copy))
        print(f"L{L:2d} h{h}  same-token z={same:6.2f}   copy z={copy:6.2f}", flush=True)
rows.sort(key=lambda r: -(r[2] + r[3]))
print("\nranked by same-token + copy:")
for L, h, s, c in rows[:8]:
    print(f"  L{L} h{h}: {s:5.2f} + {c:5.2f}")
