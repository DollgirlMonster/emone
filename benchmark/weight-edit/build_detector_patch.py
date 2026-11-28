"""Constructive patch: loop-detector units -> a write toward an exit token.

usage: build_detector_patch.py <units.npz> <out_dir> <layer> <k> <alpha> <token_id> [check.bin]

Rank-1 down_proj delta at <layer>: A reads the k best low-leak detector units
(|d'| ranked, normal mean |act| under 15% of the looping mean), each scaled so
the summed score is about 1 on an average looping position; B = alpha * e,
where e is the unit residual direction that raises <token_id>'s logit through
the final norm gain and the tied embedding. If a capture is given, the score
is evaluated on it per class so leakage is visible before any forward pass.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from build_patch import FORMAT, MODEL, safetensors  # noqa: E402
from find_direction import HIDDEN, REC, labels  # noqa: E402
from gturbo_reader import GTurboWeights  # noqa: E402


def main():
    units = np.load(sys.argv[1])
    out = Path(sys.argv[2])
    L, k, alpha, token = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
    d, m1, m0 = units[f"d{L}"], units[f"m1_{L}"], units[f"m0_{L}"]
    ok = np.abs(m0) < 0.15 * np.abs(m1)
    pick = [int(j) for j in np.argsort(-np.abs(d)) if ok[j]][:k]
    a = np.zeros_like(d, dtype=np.float32)
    a[pick] = (1.0 / k) / m1[pick]  # sign included: each picked unit contributes ~+1/k in a loop
    w = GTurboWeights(MODEL)
    g = w.get("language_model.model.norm.weight")
    e_tok = w.get("language_model.model.embed_tokens.weight")[token]
    e = (g * e_tok).astype(np.float32)
    e /= np.linalg.norm(e)
    tensors = {
        f"layers.{L}.mlp.down_proj.lora_A": a[None, :],
        f"layers.{L}.mlp.down_proj.lora_B": (alpha * e)[:, None],
    }
    report = {"units": pick, "d_prime": [round(float(d[j]), 2) for j in pick]}
    if len(sys.argv) > 7:
        raw = np.memmap(sys.argv[7], dtype=np.uint8, mode="r")
        n = raw.size // REC
        raw = raw[: n * REC].reshape(n, REC)
        head = np.asarray(raw[:, :12]).copy().view(np.int32).reshape(n, 3)
        m = np.nonzero(head[:, 1] == L)[0]
        lab = labels(head[m, 0], head[m, 2])
        y = np.array([lab[p] for p in head[m, 0]])
        act = np.asarray(raw[m, 12 + 2 * HIDDEN :]).copy().view(np.float16).astype(np.float32)
        s = act @ a
        for cls, name in ((0, "normal"), (1, "looping")):
            v = s[y == cls]
            report[f"score_{name}"] = {
                "mean": round(float(v.mean()), 3),
                "p95": round(float(np.percentile(v, 95)), 3),
                "p05": round(float(np.percentile(v, 5)), 3),
            }
    out.mkdir(parents=True, exist_ok=True)
    blob = safetensors(tensors)
    (out / "weight_patch.safetensors").write_bytes(blob)
    manifest = {
        "format": FORMAT,
        "base": {"model_id": "qwen3.5-0.8b"},
        "rank": 1,
        "description": f"Loop detector at layer {L} ({k} MLP units) writes toward token "
        f"{token} with strength {alpha} (Emone-bbm.3).",
        "provenance": {
            "method": "per-unit d' on looping vs. normal decode positions from "
            "8 greedy GSM8K captures; exit direction = final-norm gain * "
            "tied embedding row, unit length",
            "layer": L,
            "alpha": alpha,
            "token": token,
            **report,
        },
        "files": {"weight_patch.safetensors": hashlib.sha256(blob).hexdigest()},
    }
    (out / "weight_patch.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["provenance"]))


if __name__ == "__main__":
    main()
