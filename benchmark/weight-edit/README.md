# Weight-space edits (TinyTitan-bbm)

Hand-written weight patches for the dense Qwen 3.5 0.8B, applied at load with
`TinyTitanCLI --weight-patch <dir>` (`WeightPatch`, format `tinytitan-patch-v1`:
rank-r `mlp.down_proj` deltas, prefill and decode). No training, no gradients:
read weights and activations, write an edit, test it on practice problems.

The plain 0.8B has no profile row, so its prefill is per-token and has no seam;
patched runs need `TINYTITAN_PREFILL_SG_QMM=1`, which on its own changes greedy
output. Always compare a patch against a control run with the same variable.

## Workflow

| Step | Script | Notes |
|---|---|---|
| Generate / score | `run_gsm8k.py <out.jsonl> <start> <count> <max_new> [CLI args]` | greedy, thinking on, one CLI per problem; `GSM8K_JSONL` names the data, `TT_CLI` the binary, `TT_CAPTURE_DIR` turns on the MLP capture |
| Capture | env `TINYTITAN_MLP_CAPTURE=<file>`, `TINYTITAN_MLP_CAPTURE_LAYERS=15,19` | per decode position and layer: position, layer, token, MLP output, down_proj input (fp16); a diagnostic, it waits per layer |
| Probe a direction | `find_direction.py <out.npz> <capture>...` | diff of means, looping vs. pre-loop positions; reports d' and the cosine with position drift |
| Ablate it | `build_patch.py <dirs.npz> <out> <k\|all> <raw\|noposition> <layer>...` | rank-1 `W' = (I - uu^T) W` on chosen columns |
| Read weights | `read_weights.py`, `restart_units.py` | spectra, outlier units, logit lens; CPU only |
| Find detector units | `detector_units.py <out.npz> <capture>...` | per-unit d' and leak, looping vs. normal |
| Constructive edit | `build_detector_patch.py <units.npz> <out> <layer> <k> <alpha> <token> [capture]` | detector units -> write toward an exit token |
| Inhibit a failure | `add_inhibitor.py <in> <out> <rep_means.npy> <layer> <unit>...` | units that fire on the failure cancel the detector |
| Read heads | `attention_heads.py`, `read_heads.py` | per-head QK make-up, OV eigenvalue sign (copy / suppress), DeltaNet half-lives and conv shifts |
| Scale heads | `build_head_scale.py <out> "<why>" <layer>:<head>=<factor>...` | `head_scale` patch; needs no batched FFN or QMM prefill |

## Results so far (M1 Max, greedy, 4,096-token cap)

Baseline GSM8K test 0-39: 4/40 correct, 40/40 capped. Practice problems
1008-1013, six each, against a QMM-prefill control (0/6 correct, 6/6 capped):

| Edit | Correct | Capped |
|---|---|---|
| Ablate looping direction, L15+19 (top-64, all, position-orthogonal), L13+14, all 24 layers | 0-2/6 | 6/6 every variant |
| Detector (L22, 8 units) -> `</think>`, alpha 2 | 2/6 | 4/6 (self-loop on `</think>`) |
| + inhibitors 182, 1190 on repeated `</think>` | 3/6 | 4/6 |

Upstream, chosen from the weights alone (`read_heads.py`: copiers and
suppressors by the sign of `W_O W_V`), applied as `head_scale`, plain prefill,
against a plain control (1/6 correct, 6/6 capped, loop onset word 347):

| Edit | Correct | Capped | Loop onset |
|---|---|---|---|
| Suppressors L7h0, L11h6 x1.5 | 0/6 | 6/6 | 230 |
| Copiers L7h2, L11h4, L11h7 x0.7 | 1/6 | 6/6 | 279 |
| Copiers x0.3 | 0/6 | 6/6 | 152 |

Weakening retrieval makes loops start earlier, not later: the loops look like
re-verification under uncertainty rather than runaway copying.

Held-out results and the running log live on the beads (`bd show TinyTitan-a37`,
`TinyTitan-bbm.3`).
