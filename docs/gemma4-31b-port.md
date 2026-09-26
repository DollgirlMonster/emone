# Gemma 4 31B (dense): port research

Status: **research only; nothing is wired.** Written 2026-09-26 from the
`transformers` source (`src/transformers/models/gemma4/`, `main`) and this
repository's own history. The 31B's `config.json` and `generation_config.json`
were read afterwards; see "The 31B's config".

## Why this is not a greenfield port

This tree served **Gemma 4 26B-A4B** before commit 19aafd8 ("Production
hardening: audit fixes, Qwen-only") removed the family: the arch config, the
Gemma tool-call parser and schema, the repack sources and the FFN sandwich.
The 26B baseline it carried, checked then against an installed manifest:

| field | 26B-A4B |
| --- | --- |
| hidden / layers | 2816 / 30 (every 6th full attention) |
| heads / KV heads (sliding / full) | 16 / 8 / 2 |
| head dim (sliding / full) | 256 / 512 |
| sliding window | 1024 |
| rope theta (sliding / full), partial factor | 10k / 1M, 0.25 |
| experts / top-k / expert width | 128 / 8 / 704 |
| vocab, tied, softcap | 262,144, tied, 30.0 |
| K = V on full layers, GELU-tanh, sqrt(hidden) embed scale | yes |

The 31B shares the attention, norm and embedding design and replaces the
two-branch FFN (dense MLP + routed experts) with the dense MLP alone.

## What the current runtime still has

- K = V on full-attention layers (`attentionKEqV`), decode and prefill.
- Embedding scaled by sqrt(hidden), final logit softcap.
- Gemma's RoPE convention (`ropeNeoxSubdim: false`): full layers rotate
  `fullHeadDim * partialRotaryFactor / 2` pairs, sliding layers all of them;
  the prefill epilogue carries it, and the V-norm epilogue with it.
- Sliding-window layers and their KV ring; per-layer head dim and KV heads.
- GELU kernels (the shared-expert and prefill paths); `silu` is a flag.
- A GPU dense-FFN prefill path (`encodeDenseFFNPrefill`, used by Qwen 3.5).

## What has to be (re)built

1. **Family and schema.** A `gemma4` family (manifest `arch.family`) and a
   `TensorSchema` for `model.language_model.layers.N.*`; the vision and audio
   towers are never repacked (text-only, as for Qwen3.8).
2. **The FFN sandwich, dense form.** Gemma's `post_attention_layernorm`
   normalises the *attention output* before the residual add (Qwen's same-named
   tensor is the pre-FFN norm), then `pre_feedforward_layernorm` -> MLP ->
   `post_feedforward_layernorm` -> residual, then `hidden *= layer_scalar`. The
   removed runner implemented the two-branch version with all of these
   (`git show 19aafd8^:sources/NVMAI/Runtime/Inference/RealForwardRunner.swift`,
   the `ffnSandwichNorms` sites); the dense form drops branch 2.
3. **GELU-tanh dense MLP** (`act(gate(x)) * up(x)` then down) in decode and
   prefill, and `ManifestReader` accepting `gelu_pytorch_tanh` for this family
   (it refuses anything but `silu` today).
4. **V norm without scale** on non-shared layers, and Q/K norms per head:
   check the surviving epilogue against `Gemma4TextAttention` rather than
   assume.
5. **Attention scale 1.0** (`self.scaling = 1.0`; the Q/K norms carry it).
6. **Converter.** A dense Gemma converter modelled on `tools/prepare_qwen35.py`
   / `tools/repack_dense.sh`, mapping the `model.language_model.*` namespace.
7. **Tool calls.** Gemma 4 emits
   `<|tool_call>call:name{arg:<|"|>value<|"|>}<tool_call|>`. Restore
   `GemmaToolCallParser` and `GemmaToolSchema` from 19aafd8^ and select the
   dialect by family; diff the checkpoint's `chat_template.jinja` against what
   that parser expects before trusting it.
8. The eight wiring points and the §4 verification bar of
   `docs/adding-a-model.md`, and the scope line in `AGENTS.md`, which says
   Qwen-family only.

## The 31B's config, read from the source

`google/gemma-4-31B-it` at `842da3794eaa0b77d5f08bae87a17459d91ff475`, not gated,
two safetensors shards, `chat_template.jinja` present.

| field | value |
| --- | --- |
| hidden / layers / MLP width | 5376 / 60 / 21504, `gelu_pytorch_tanh` |
| layer pattern | 5 sliding : 1 full, 10 full layers (5, 11, ... 59) |
| heads | 32 query; sliding 16 KV x 256, full 4 KV x 512 |
| `attention_k_eq_v` | true (full layers have no `v_proj`) |
| sliding window | 1024 |
| rope | sliding default theta 10k; full proportional, theta 1M, partial 0.25 |
| PLE / KV sharing / double-wide MLP | off (`0` / `0` / false) |
| vocab, tied, softcap | 262,144, tied, 30.0 |
| sampling (`generation_config.json`) | temperature 1.0, top-k 64, top-p 0.95; EOS ids 1, 106, 50 |

Every feature it turns on is one the removed 26B support already handled; the
two that would have been new work (PLE, KV sharing) are off. Parameters: about
29.3B in the layers (~480M per sliding layer, ~534M per full one) plus 1.41B
of tied embedding.

## Why the 31B does not stream

The engine's SSD streaming works because a routed-expert model touches a
small, repeating fraction of its weights per token. A dense model touches all
of them: ~17.5 GB per decoded token at 4-bit. From the external drive
(~2.3 GB/s measured in the M1 spikes) that is ~7.6 s per token; from an
internal M1 Max SSD, ~3 s. Keeping only part resident does not change the
shape -- every streamed byte is paid on every token. Prefill is the exception
(a 16K chunk reads each weight once for 16K tokens), but decode sets the
experience. The low-RAM Gemma for this engine is the 26B-A4B, the MoE the
tree served before: 128 experts top-8, ~3B dense parameters resident and the
~23B of experts streamed and cached.

## Features the config decides (as defined by `transformers`)

`transformers` defines these for every Gemma 4; which ones the 31B turns on is
*unverified*:

- `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
  `num_key_value_heads`, `num_global_key_value_heads`, `head_dim`,
  `global_head_dim`, `sliding_window`, `layer_types`, `rope_parameters`.
- `attention_k_eq_v` (the 26B had it).
- `num_kv_shared_layers`: the last N layers reuse the K/V of the last
  non-shared layer of the same type -- a KV-cache aliasing feature this runtime
  does not have. Expected 0 on the large models; if not, it is new work.
- `hidden_size_per_layer_input` (per-layer embeddings, PLE): the blog calls it
  a feature of the *smaller* models; if the 31B sets it, it is new work (the
  Qwen3.8 PLE block is a different design).
- `use_double_wide_mlp` (only on KV-shared layers), `final_logit_softcapping`,
  `tie_word_embeddings`, `vocab_size`.
- `generation_config.json`: sampling and EOS ids; `chat_template.jinja`:
  thinking markers and the tool-call dialect.

## What to expect on an M1 Max 64 GB (estimates, not measurements)

- 4-bit (g64 affine, ~4.5 bits/weight): ~17-18 GB of weights, 8-bit ~33 GB.
  Both fit in RAM: nothing streams from the SSD, so the external drive that
  bounds Qwen3.8's prefill does not matter here.
- Decode reads every weight per token: at ~300-380 GB/s effective, a ceiling of
  roughly 17-22 tok/s at 4-bit and 9-11 at 8-bit.
- Prefill is compute: ~62 GFLOP per token against ~10 TFLOPS fp16 peak; at the
  40-60% the MPP path reaches, roughly 65-100 tok/s before attention cost.
