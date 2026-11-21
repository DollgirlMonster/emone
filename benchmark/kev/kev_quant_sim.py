"""Predict how Emone's install quantization moves Kev-0.8B's probabilities, before any Swift work.

Applies tools/prepare_qwen35.py's rules to the merged fp32 backbone as quantize -> dequantize (group-64 affine, bf16
scale/bias, embedding at 8-bit, linear_attn.in_proj_a/b and every non-matrix tensor at bf16), then scores the fixture's
rows on CPU with fp32 activations. Isolates weight-quantization error from the engine's own bf16/fp16 arithmetic.

usage: python -I kev_quant_sim.py <kev-src> <base-dir> <kev-dir> <out-dir>
"""

import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.argv = sys.argv[:5]
here = Path(__file__).parent
sys.path.insert(0, str(here))
import kev_reference as ref  # noqa: E402 (after the sys.path insert)

G = 64


def qdq(w, bits):
    g = w.float().reshape(*w.shape[:-1], w.shape[-1] // G, G)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    levels = (1 << bits) - 1
    scale = torch.where(hi == lo, torch.ones_like(hi), (hi - lo) / levels).bfloat16().float()
    bias = lo.bfloat16().float()
    q = torch.round((g - bias) / scale).clamp(0, levels)
    return (q * scale + bias).reshape(w.shape)


def bits_for(name, width):
    if not name.endswith(".weight"):
        return None
    if name.endswith("conv1d.weight") or name.endswith("norm.weight"):
        return None
    if name.endswith((".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight")):
        return None
    if name.endswith("embed_tokens.weight"):
        return 8
    return width


@torch.no_grad()
def main():
    fixture = json.loads((ref.OUT / "kev08b_fixture.json").read_text())
    tok, m, _ = ref.load(merge=True)
    pristine = {k: v.clone() for k, v in m.lm.state_dict().items()}
    report = {}
    for width in ("bf16", 8, 4):
        sd = {}
        for k, v in pristine.items():
            b = None if width == "bf16" else bits_for(k, width)
            sd[k] = qdq(v, b) if b else v.bfloat16().float()
        m.lm.load_state_dict(sd)
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
            print(f"{width!s:5s} {q['key']:22s} {[round(x, 4) for x in p]}")
        report[str(width)] = {
            "max_abs_dp": worst,
            "argmax_flips": flips,
            "questions": len(fixture["questions"]),
        }
    print(json.dumps(report, indent=1))
    (ref.OUT / "kev08b_quant_sim.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
