"""Add inhibitor units to a detector patch's A row.

usage: add_inhibitor.py <in_dir> <out_dir> <rep_means.npy> <layer> <unit>...
Each inhibitor unit j gets A_j = 1/|mean_rep_j| with the sign that makes its
contribution -1 on the repeat positions it was found on (TinyTitan-bbm.3).
"""
import hashlib, json, struct, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from build_patch import safetensors

src, dst, means, L = Path(sys.argv[1]), Path(sys.argv[2]), np.load(sys.argv[3]), int(sys.argv[4])
units = [int(x) for x in sys.argv[5:]]
b = (src / "weight_patch.safetensors").read_bytes()
hl = struct.unpack("<Q", b[:8])[0]
h = json.loads(b[8:8 + hl])
def get(name):
    o, shape = h[name]["data_offsets"], h[name]["shape"]
    return np.frombuffer(b[8 + hl + o[0]:8 + hl + o[1]], dtype=np.float32).reshape(shape).copy()
a, bb = get(f"layers.{L}.mlp.down_proj.lora_A"), get(f"layers.{L}.mlp.down_proj.lora_B")
for j in units:
    a[0, j] = -1.0 / means[j]   # mean_rep * A_j = -1
blob = safetensors({f"layers.{L}.mlp.down_proj.lora_A": a, f"layers.{L}.mlp.down_proj.lora_B": bb})
dst.mkdir(parents=True, exist_ok=True)
(dst / "weight_patch.safetensors").write_bytes(blob)
m = json.loads((src / "weight_patch.json").read_text())
m["description"] += f" Inhibitor units {units} cancel it on repeated exit tokens."
m["provenance"]["inhibitors"] = {str(j): round(float(means[j]), 3) for j in units}
m["files"] = {"weight_patch.safetensors": hashlib.sha256(blob).hexdigest()}
(dst / "weight_patch.json").write_text(json.dumps(m, indent=1) + "\n")
print(m["description"])
