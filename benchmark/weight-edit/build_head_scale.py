"""Write a emone-patch-v1 package that scales whole attention heads.

usage: build_head_scale.py <out_dir> "<description>" <layer>:<head>=<factor>...

Each factor multiplies that head's output before o_proj (the same as scaling
its o_proj columns); unnamed heads stay at 1. On a Gated DeltaNet layer (all
but 3, 7, 11, 15, 19, 23) <head> is a value head, 0-15, scaled after the gated
norm and before out_proj. Heads are chosen by reading the
weights (read_heads.py: copiers and suppressors from the sign of W_O W_V).
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_patch import FORMAT, safetensors  # noqa: E402

HEADS = 8  # full-attention query heads
VHEADS = 16  # Gated DeltaNet value heads
FULL = {3, 7, 11, 15, 19, 23}


def main():
    out, description = Path(sys.argv[1]), sys.argv[2]
    scales = {}
    for spec in sys.argv[3:]:
        where, factor = spec.split("=")
        layer, head = (int(x) for x in where.split(":"))
        n = HEADS if layer in FULL else VHEADS
        scales.setdefault(layer, np.ones(n, np.float32))[head] = float(factor)
    tensors = {f"layers.{L}.{'self_attn' if L in FULL else 'linear_attn'}.head_scale": v
               for L, v in sorted(scales.items())}
    blob = safetensors(tensors)
    out.mkdir(parents=True, exist_ok=True)
    (out / "weight_patch.safetensors").write_bytes(blob)
    manifest = {
        "format": FORMAT,
        "base": {"model_id": "qwen3.5-0.8b"},
        "rank": 1,
        "description": description,
        "provenance": {"method": "per-head output scale chosen from W_O W_V eigenvalue sign "
                                 "(benchmark/weight-edit/read_heads.py)",
                       "scales": {str(L): v.tolist() for L, v in sorted(scales.items())}},
        "files": {"weight_patch.safetensors": hashlib.sha256(blob).hexdigest()},
    }
    (out / "weight_patch.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["provenance"]["scales"]))


if __name__ == "__main__":
    main()
