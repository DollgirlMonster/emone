"""Targeted weight read: units whose down_proj column pushes loop-restart tokens.

The restart set is the opener of the lines that repeat in the baseline loops
(bullet, 'Let', 'Wait', 'The', 'prompt', 'check', 'assume'). For every layer
and unit, score = mean logit of the restart set minus the mean over 2,000
random tokens, through the final-norm gain and the tied embedding, in units
of that column's own spread over the random tokens (a z-score).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from gturbo_reader import GTurboWeights

M = os.environ.get(
    "TT_MODEL", str(Path(__file__).resolve().parents[2] / "models/qwen3.5_0.8B_8Bit")
)
w = GTurboWeights(M)
tok = json.load(open(f"{M}/tokenizer/tokenizer.json"))["model"]["vocab"]
words = [
    "*",
    "Ġ*",
    "ĠĠĠ*",
    "Let",
    "ĠLet",
    "Wait",
    "ĠWait",
    "The",
    "ĠThe",
    "Ġprompt",
    "Ġcheck",
    "Ġassume",
    "ĠCould",
    "ĠIf",
]
ids = [tok[x] for x in words if x in tok]
print("restart tokens:", [x for x in words if x in tok])
E = w.get("language_model.model.embed_tokens.weight")
g = w.get("language_model.model.norm.weight")
rng = np.random.default_rng(0)
rand = rng.choice(min(E.shape[0], 150000), 2000, replace=False)
Es, Er = E[ids] * g, E[rand] * g
for L in range(24):
    W = w.get(f"language_model.model.layers.{L}.mlp.down_proj.weight")
    s, r = Es @ W, Er @ W  # [k x inter], [2000 x inter]
    z = (s.mean(0) - r.mean(0)) / r.std(0)
    top = np.argsort(-z)[:5]
    print(
        f"L{L:2d} max z={z.max():5.2f}  units>3={int((z > 3).sum()):3d}  top {[(int(j), round(float(z[j]), 2)) for j in top[:3]]}"
    )
