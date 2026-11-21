"""benchmark/kev/kev_quant_sim.py for Rev-0.8B: how much of the engine's distance from the fp32 reference is the
install's 8-bit weight quantization. Quantize -> dequantize the merged fp32 backbone by tools/prepare_qwen35.py's rules
(group-64 affine, bf16 scale/bias, embedding at 8-bit, in_proj_a/b and non-matrix tensors at bf16), score the fixture's
rows on CPU with fp32 activations, compare with the reference.

usage: python -I rev_quant_sim.py <kev-src> <base-dir> <rev-checkpoint-dir> <out-dir>
"""

import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.argv = sys.argv[:5]
sys.path.insert(0, str(Path(__file__).parent))
import rev_reference as ref  # noqa: E402 (after the sys.path insert)

G = 64


def qdq(w, bits):
    g = w.float().reshape(*w.shape[:-1], w.shape[-1] // G, G)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    levels = (1 << bits) - 1
    scale = torch.where(hi == lo, torch.ones_like(hi), (hi - lo) / levels).bfloat16().float()
    bias = lo.bfloat16().float()
    q = torch.round((g - bias) / scale).clamp(0, levels)
    return (q * scale + bias).reshape(w.shape)


def bits_for(name):
    if not name.endswith(".weight"):
        return None
    if name.endswith("conv1d.weight") or name.endswith("norm.weight"):
        return None
    if name.endswith((".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight")):
        return None
    return 8


@torch.no_grad()
def main():
    fixture = json.loads((ref.OUT / "rev08b_fixture.json").read_text())
    _, m, _ = ref.load(merge=True)
    pristine = {k: v.clone() for k, v in m.lm.state_dict().items()}
    report = {}
    for width in ("bf16", 8):
        m.lm.load_state_dict(
            {
                k: (qdq(v, bits_for(k)) if width == 8 and bits_for(k) else v.bfloat16().float())
                for k, v in pristine.items()
            }
        )
        worst, flips = 0.0, 0
        for q in fixture["questions"]:
            h = (
                m.lm(input_ids=torch.tensor([q["ids"]]), position_ids=torch.tensor([q["pos"]]))
                .last_hidden_state[0]
                .float()
            )
            p = F.softmax(m.head(h[q["decide"]], h[torch.tensor(q["opts"])]), -1).tolist()
            worst = max(worst, max(abs(a - b) for a, b in zip(p, q["probs_kev"], strict=True)))
            flips += max(range(len(p)), key=p.__getitem__) != max(
                range(len(p)), key=q["probs_kev"].__getitem__
            )
            print(f"{width!s:5s} {q['key']:18s} {[round(x, 4) for x in p]}")
        report[str(width)] = {
            "max_abs_dp": worst,
            "argmax_flips": flips,
            "questions": len(fixture["questions"]),
        }
    print(json.dumps(report, indent=1))
    (ref.OUT / "rev08b_quant_sim.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
