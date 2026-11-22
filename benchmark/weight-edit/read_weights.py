"""Weight-only survey of every layer's mlp.down_proj (CPU, no forward passes).

Per layer: top singular values (is there one dominant write direction?), the
spread of column norms (outlier units), and a logit lens on the top singular
directions -- the tokens they push up and down through the tied embedding,
after the final norm's weight.
"""

import json
import os
import sys

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from gturbo_reader import GTurboWeights  # noqa: E402

MODEL = os.environ.get("TT_MODEL", str(Path(__file__).resolve().parents[2] / "models/qwen3.5_0.8B_8Bit"))
w = GTurboWeights(MODEL)
tok = json.load(open(f"{MODEL}/tokenizer/tokenizer.json"))
vocab = {i: s for s, i in tok["model"]["vocab"].items()}
for a in tok.get("added_tokens", []):
    vocab[a["id"]] = a["content"]


def show(i):
    s = vocab.get(int(i), f"<{i}>").replace("Ġ", "·").replace("Ċ", "\\n")
    return s[:14]


E = w.get("language_model.model.embed_tokens.weight")  # [vocab x hidden]
norm_names = [n for n in w.names("norm") if "layers" not in n]
fnorm = w.get(norm_names[0]) if norm_names else np.ones(E.shape[1], np.float32)
print("final norm:", norm_names, "mean", float(fnorm.mean()))
# Qwen3.5 RMSNorm weights are stored zero-centred (scale = 1 + w) when their mean sits near 0.
gain = 1 + fnorm if abs(float(fnorm.mean())) < 0.5 else fnorm

rows = []
for L in range(24):
    W = w.get(f"language_model.model.layers.{L}.mlp.down_proj.weight")  # [hidden x inter]
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    cn = np.linalg.norm(W, axis=0)
    rows.append((L, S[0], S[0] / S[1], S[0] / np.median(S), cn.max() / np.median(cn),
                 int(np.argmax(cn))))
    lens = []
    for k in range(2):
        logits = E @ (gain * U[:, k])
        hi, lo = np.argsort(-logits)[:6], np.argsort(logits)[:6]
        lens.append(f"  sv{k} +[{' '.join(show(i) for i in hi)}]  -[{' '.join(show(i) for i in lo)}]")
    print(f"L{L:2d} s1={S[0]:6.2f} s1/s2={S[0]/S[1]:5.2f} s1/med={S[0]/np.median(S):6.2f} "
          f"colmax/med={cn.max()/np.median(cn):6.2f} (unit {int(np.argmax(cn))})")
    print("\n".join(lens))
