# Next models: Gemma 4 26B-A4B, GLM-5.3, and speculative decoding

Research only; nothing here is wired. Written 2026-09-27 from the published
configs, `transformers` `models/gemma4/modeling_gemma4.py` (`main`), and this
repository's history and verdicts. Target machine: M1 Max, 32-core GPU, 64 GB,
models on an external Thunderbolt NVMe (~2.3 GB/s measured in the prefill
spikes). Every speed below is an estimate from bytes and FLOPs, not a
measurement.

## Decision, 2026-09-27: Qwen3.8-27B instead of a Gemma port

The assistant model is **Qwen3.8-27B** (`Qwen/Qwen3.8-27B` @ `1d4bf0f`), wired
in `55e7508`. It is published as `qwen3_5_text`, the dense family this engine
already serves, so it is a converter and profile entry rather than a port. Its
chat template is byte-identical to Flash-Next's. Vendor-reported scores; the
Gemma column is Google's own card, and "--" means not reported:

| benchmark | Qwen3.8-27B | Gemma 4 26B-A4B | Qwen3.8-Flash-Next | GLM-5.3-Flash |
| --- | ---: | ---: | ---: | ---: |
| GPQA Diamond | 89.2 | 82.3 | 91.7 | 90.15 |
| LiveCodeBench v6 | 90.3 | 77.1 | 91.9 | -- |
| HLE (no tools) | 30.8 | 8.7 | 35.9 | -- |
| IFBench | 79.5 | -- | 81.3 | -- |
| Toolathlon Verified | 67.1 | -- | 73.5 | 78.4 |
| CoWorkBench | 70.7 | -- | 73.9 | -- |
| SWE-bench Pro | 61.7 | -- | 62.5 | -- |
| DeepSWE 1.1 | 42.2 | -- | 58.7 | 63.4 |
| Tau2 | -- | 68.2 | -- | -- |

Qwen's Qwen 3.6 card also measures Gemma 4 26B-A4B directly, and there it
trails Qwen 3.6 35B-A3B on every agent benchmark listed:
- TAU3 59.0 against 67.2;
- MCP-Atlas 50.0 against 62.8;
- Tool Decathlon 12.0 against 26.9.

Trade-offs:
- **Speed.** A dense 27B reads all of its 16.5 GB (4-bit) or 28.6 GB (8-bit)
  per token. That gives a decode ceiling of roughly 18 or 10 tok/s here,
  against several times that for an A4B MoE held in RAM.
- **Tone.** Benchmarks say nothing about tone or verbosity; that needs the
  user's own evals.
- **Not yet in this step.** Qwen3.8's reasoning-effort levels (low, medium,
  xhigh) and its instruct-mode sampling row (0.7 / 0.80, presence 1.5). Both
  are keyed by family, and this model runs as the Qwen 3.5 dense family. Until
  then thinking is on or off, and the template's default effort (xhigh)
  applies when it is on.

The Gemma 4 26B-A4B research below stays as the record, in case its tone
wins the user's evals.

## Gemma 4 26B-A4B (the MoE)

`google/gemma-4-26B-A4B-it` at `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, not
gated, Apache 2.0, two bf16 shards (51.6 GB). The config matches the baseline
the removed support recorded (`docs/gemma4-31b-port.md`):

| field | value |
| --- | --- |
| hidden / layers | 2816 / 30; full attention at 5, 11, 17, 23, 29 |
| attention | 16 query heads; sliding 8 KV x 256, full 2 KV x 512, K = V on full layers |
| sliding window, rope | 1024; sliding theta 10k, full proportional theta 1M, partial 0.25 |
| dense MLP | 2112 wide, `gelu_pytorch_tanh`, every layer |
| routed experts | 128, top-8, 704 wide, fused `experts.gate_up_proj` [E, 2I, D] |
| router | no-scale RMSNorm, x `router.scale` x D^-0.5, softmax, top-8, renormalise, x `per_expert_scale` |
| PLE / KV sharing / double-wide MLP | off (0 / 0 / false) |
| vocab, tied, softcap | 262,144, tied, 30.0 |
| sampling | temperature 1.0, top-k 64, top-p 0.95; EOS 1, 106, 50 |

### The layer, from `Gemma4TextDecoderLayer.forward`

```
r = x
x = post_attention_layernorm(attn(input_layernorm(x)));  x = r + x
r = x
a = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(r)))      # dense branch
w, idx = router(r)                                                       # reads r, not a norm of it
b = post_feedforward_layernorm_2(experts(pre_feedforward_layernorm_2(r), idx, w))
x = r + post_feedforward_layernorm(a + b)
x = x * layer_scalar
```

The dense branch and the routed branch read the same residual and do not depend
on each other. This matters for streaming: the router can run first, and the
expert reads can be in flight while the dense MLP computes. That is the overlap
Qwen's shared expert already gets (`7b15168`).

### Sizes

| part | params | 4-bit (g64 affine, 4.5 b/w) | 8-bit (8.5 b/w) |
| --- | ---: | ---: | ---: |
| routed experts (30 x 128 x 5.95M) | 22.8B | 12.8 GB | 24.3 GB |
| everything else (attention, dense MLP, embedding, norms) | 2.4B | 1.35 GB | 2.5 GB |
| **total** | **25.2B** | **~14 GB** | **~27 GB** |
| routed bytes per token (240 experts) | 1.43B | 0.80 GB | 1.52 GB |

Either width fits entirely in 64 GB. So "streaming" here means the expert cache
can hold the whole corpus once it is warm, and the drive is only paid on the
first touch of each expert. With idle unload (`tools/serve.sh`, 15 minutes) it
does not sit in RAM between sessions. A smaller `--ram` would stream the rest,
which is the design this engine has for its 35B-A3B models.

Rough decode ceilings with everything in RAM, from the bytes read per token
(~2.1 GB at 4-bit and ~4 GB at 8-bit) at ~300 GB/s effective: ~140 tok/s at
4-bit and ~75 tok/s at 8-bit. The real number will be well below that; the
35B-A3B class runs 21 tok/s on a base M3, which has a quarter of this machine's
bandwidth.

### What exists, and what has to be built

The engine still carries most of the attention side:
- K = V on full layers, sliding-window layers and their KV ring;
- per-layer head dim and KV-head count, Gemma's partial proportional RoPE;
- softcap, sqrt(hidden) embedding scale, GELU kernels;
- a router-scale fold (`effectiveScaleBuffers`).

The removed code, 1,088 commits back (`19aafd8^`), is the math reference but
not a revert: the tree has been renamed and restructured since.

1. **Family, schema, manifest.** A `gemma4` family; a `TensorSchema` for
   `model.language_model.layers.N.*`; accept `gelu_pytorch_tanh`; carry
   `layer_scalar`, `router.scale`, `per_expert_scale`, and the five FFN norms.
2. **Converter.** Split the fused `experts.gate_up_proj` [128, 1408, 2816] into
   per-expert gate and up, then quantize and pack like the Qwen MoE installs.
   The vision tower is never repacked (text only). Needs the 51.6 GB bf16 source
   once, streamed.
3. **Decode.** The two-branch FFN above in the runner:
   - the dense MLP in the slot the shared expert uses now, with Gemma's norms
     and no sigmoid gate;
   - the router's no-scale norm and per-expert weight scale;
   - `layer_scalar`;
   - check against `Gemma4TextAttention` rather than assume: Q/K norms, V
     norm without scale, attention scale 1.0.
4. **Prefill.** The same, batched: grouped routed tiles (now MPP GEMMs), the
   dense MLP as GEMMs, and sliding-window attention. No QSA and no GDN, so the
   M1 prefill work on those does not apply here, but the MPP GEMM paths do.
5. **Server.** Gemma's turn markers (`<|turn>`, `<turn|>`), thinking on
   `<|channel>` ... `<channel|>` mapped to `reasoning_content`, and the tool-call
   dialect `<|tool_call>call:name{...}<tool_call|>`: restore
   `GemmaToolCallParser` / `GemmaToolSchema` from `19aafd8^` and diff against
   this template (18.7 KB, revised for "tool-calling loops, turn closures, and
   thinking content-ordering").
6. **Verification.** Per `docs/adding-a-model.md`:
   - a Python reference on CPU for a short prompt, compared per layer;
   - a golden baseline;
   - the tool-call round trip through the server.

### QAT or not, and 4-bit or 8-bit

`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` is bf16 weights trained to
survive Q4_0: blocks of 32, one scale, symmetric. This engine quantizes in
affine groups of 64 (scale and bias).
- A Q4_0 block *is* an affine group of 32 (bias = -8 x scale), so the QAT
  weights would load exactly if the 4-bit kernels took group 32. That is kernel
  work, and 5.0 bits per weight instead of 4.5.
- Rounded into groups of 64 instead, the QAT benefit is partial and unmeasured.

Recommendation: **8-bit from the standard checkpoint first.**
- It fits in RAM whole.
- It is near-lossless, which makes it the parity reference for everything
  after it.
- It needs no kernel change.

A 4-bit build is a speed option afterwards, from the QAT weights at group 32
if the kernels gain it.

### Google's MTP drafter

`google/gemma-4-26B-A4B-it-assistant` is a speculative-decoding drafter:
- 4 layers, hidden 1024, 8192-wide dense MLP;
- `num_kv_shared_layers: 4`, so every layer reads the target's KV cache instead
  of keeping its own;
- a 2,048-centroid output head (`num_centroids`, top-32).

It is small enough to be nearly free per draft token. It needs KV sharing and
the centroid head, neither of which this engine has. See the speculative
decoding section.

## GLM-5.3 and GLM-5.3-Flash

| | GLM-5.3 | GLM-5.3-Flash |
| --- | --- | --- |
| repo, sha | `zai-org/GLM-5.3` `aca966e` | `zai-org/GLM-5.3-Flash` `eb9eb20` |
| total / active | 753B / ~40B | 321B / ~18B |
| layers | 78, `glm_moe_dsa` | 45, `glm5_next_text` |
| attention | MLA + DeepSeek sparse attention, every layer | 34 KDA linear layers + 11 MLA sparse layers (3:1), NoPE |
| experts | 256, top-8, 2048 wide, 1 shared; first 3 layers dense | 288, top-8, 2048 wide, 1 shared; first 3 dense |
| router | sigmoid, correction bias (`noaux_tc`), routed scale 2.5 | same |
| extras | MTP 1 layer | hyper-connections x4 (Sinkhorn), MTP 1 layer, SwiGLU clamp 10, vision |
| released weights | FP8 e4m3, block 128 (BF16 repo too) | FP8 e4m3, block 128 (BF16 repo too) |
| sampling | | temperature 1.0, top-p 0.95; `reasoning_effort` low / high / max |

### What it would cost to run here

Flash, 4-bit:
- **Disk and reads.** About 180 GB on disk, of which 171 GB is the routed
  corpus (42 x 288 x 25.2M params). A token routes 336 experts: 8.5B params,
  4.8 GB at 4-bit. Qwen3.8 125B-A6B routes 1.3 GB per token from a 68 GB
  corpus, so Flash reads ~3.6x the bytes per token from a ~2.5x larger corpus.
- **Decode.** Qwen3.8 decodes at ~4 tok/s here. Scaled by bytes, and with a
  cache that covers a smaller share of the corpus, Flash would decode at
  **~1 tok/s**.
- **Prefill.** Each chunk sweeps the whole corpus: 171 GB at ~2.3 GB/s is ~75 s
  per chunk of reads alone, and the compute is ~3x Qwen3.8's. A 17K-token
  prompt would take **~4-6 minutes**, against 156 s for Qwen3.8.

GLM-5.3 (the 753B): ~420 GB at 4-bit and ~40B active, so **~0.3 tok/s**. Not
practical on this machine.

### How much of the engine it reuses

Flash is architecturally closer to Qwen3.8 than anything else supported:
- **Hyper-connections:** four streams, as in Qwen3.8.
- **Sparse-attention indexer:** pooled blocks of 4 with the tail always kept
  and a top-2,048 budget. That is Qwen3.8's QSA scheme with 32 indexer heads
  instead of 4, so the M1 indexer and selection work would carry over.
- **Linear attention:** a delta-rule family, but KDA gates per channel where
  Qwen's GDN gates per head, so the kernel needs a new gate path.
- **New work:**
  - MLA: latent KV (rank 512), with q through a 1,536 LoRA;
  - the sigmoid router with correction bias;
  - dense first layers;
  - the SwiGLU clamp;
  - an FP8 converter.

It is the biggest port of the three and the slowest result. Verdict: **defer**.
Revisit only if a model in this class is needed and the ~1 tok/s decode is
acceptable, or if a faster drive changes the arithmetic.

## Speculative decoding

The engine already has it: `--mtp-model` attaches a native draft head
(`StreamingMTP`), and Qwen3.8 has an MTP sidecar (`tools/prepare_qwen38_mtp.py`).
It measured badly (`docs/qwen38-mtp-and-overlap-verdict.md`, 2026-09-21, base M3
24 GB, `--ram 8`):
- acceptance 85.7% and 1.86 tokens per pass, but **-43%** tok/s. Output was
  identical.
- Why: the verify runs through the prefill kernels at 1.6-1.7x a decode step,
  plus ~200 ms of host and commit time per pass. In a drive-bound decode,
  verifying two tokens also reads the union of both tokens' experts.

Two further limits apply to how these models are actually used:
- **Greedy only.** The draft is used only under pure greedy decoding
  (temperature 0, no penalties); `RawCompletion.isPureGreedy` gates it. Qwen3.8
  and Gemma both sample at temperature 1.0, so the draft never runs by default.
  Using it there needs speculative *sampling*: accept or reject each draft token
  against the target distribution. That is standard, but not implemented.
- **The verify path.** A win needs a decode-native verify, a 2-4 token GEMV
  batch, instead of the prefill kernels.

Where it could pay:
- **Gemma 26B-A4B at 8-bit, fully in RAM.** There is no drive on the critical
  path. The dense 2.4B is read once per verify pass instead of once per token,
  and Google's drafter is tiny. This is the best candidate by far.
- **Qwen3.8 on this machine.** Re-run the gate before believing -43%: that was
  an M3 at `--ram 8`, and this M1 Max holds a far larger expert cache. It is
  one command (`benchmark/tinytitan_mtp_b3_qualification.py --target <install>
  --sidecar <mtp> --presence-penalty 0`).

A plan, in order:
1. Re-measure the Qwen3.8 gate on the M1.
2. Speculative sampling, so a draft is usable at temperature 1.0.
3. A decode-native verify for 2-4 tokens.
4. The Gemma drafter, which needs KV sharing and the centroid head.
