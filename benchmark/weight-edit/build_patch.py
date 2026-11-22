"""Write a emone-patch-v1 package that projects u out of chosen down_proj columns.

usage: build_patch.py <directions.npz> <out_dir> <columns|all> <raw|noposition> <layer>...

For layer L with unit direction u and the chosen column set S, the edit is
W'[:, S] = (I - u u^T) W[:, S], i.e. dW = -u v^T with v_j = u . W[:, j] for
j in S and 0 elsewhere: one rank-1 pair, A = v^T [1 x inter], B = -u [hidden x 1].
Columns are ranked by |v_j| * mean|act_j| on looping positions.
"""

import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from gturbo_reader import GTurboWeights  # noqa: E402

MODEL = os.environ.get("TT_MODEL", str(Path(__file__).resolve().parents[2] / "models/qwen3.5_0.8B_8Bit"))
FORMAT = "emone-patch-v1"


def safetensors(tensors):
    header, body = {"__metadata__": {"format": FORMAT}}, b""
    for name, arr in tensors.items():
        data = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [len(body), len(body) + len(data)]}
        body += data
    js = json.dumps(header, sort_keys=True).encode()
    js += b" " * (-len(js) % 8)
    return struct.pack("<Q", len(js)) + js + body


def main():
    dirs = np.load(sys.argv[1])
    out = Path(sys.argv[2])
    k = sys.argv[3]
    key = "q" if sys.argv[4] == "noposition" else "u"
    layers = [int(x) for x in sys.argv[5:]]
    w = GTurboWeights(MODEL)
    tensors, chosen = {}, {}
    for L in layers:
        u = dirs[f"{key}{L}"]
        W = w.get(f"language_model.model.layers.{L}.mlp.down_proj.weight")
        v = u @ W
        if k == "all":
            mask = np.ones_like(v, dtype=bool)
        else:
            rank = np.argsort(-np.abs(v) * dirs[f"a{L}"])[: int(k)]
            mask = np.zeros_like(v, dtype=bool)
            mask[rank] = True
        a = np.where(mask, v, 0.0)[None, :]
        b = -u[:, None]
        tensors[f"layers.{L}.mlp.down_proj.lora_A"] = a
        tensors[f"layers.{L}.mlp.down_proj.lora_B"] = b
        chosen[str(L)] = {"columns": int(mask.sum()), "d_prime": float(dirs[f"d{L}"]),
                          "weights_changed": int(mask.sum()) * W.shape[0]}
    out.mkdir(parents=True, exist_ok=True)
    blob = safetensors(tensors)
    (out / "weight_patch.safetensors").write_bytes(blob)
    manifest = {
        "format": FORMAT,
        "base": {"model_id": "qwen3.5-0.8b"},
        "rank": 1,
        "description": "Projects a looping direction out of mlp.down_proj columns "
                       f"in layers {layers} (Emone-a37 spike).",
        "provenance": {
            "method": "difference of class means of dense-MLP outputs on decode positions "
                      "inside vs. before exact 24-gram repeats, greedy GSM8K generations; "
                      "columns ranked by |u.W_j| * mean|act_j| on looping positions",
            "columns": k, "direction": key, "layers": chosen,
        },
        "files": {"weight_patch.safetensors": hashlib.sha256(blob).hexdigest()},
    }
    (out / "weight_patch.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["provenance"], indent=1))


if __name__ == "__main__":
    main()
