#!/bin/bash
# Files the three LeapQuant / multiplexing beads. Run from the TinyTitan
# checkout on the Mac that holds the beads database, then delete this file.
set -euo pipefail

id_of() { python3 -c 'import json,sys; d=json.load(sys.stdin); print((d[0] if isinstance(d,list) else d)["id"])'; }

D1='Today effectivePromptCacheMode turns the prompt cache off when slots > 1, and frontier checkpoints are off with it (docs/plan-continuous-batching.md Phase 4; docs/frontier-prefix-cache.md guards). So multiplexing (--max-concurrent-sequences > 1) and prompt reuse cannot be used together. Goal: a per-slot prompt/frontier cache. Restoring a snapshot into slot s must land in that slot KV and GDN regions, and stores must share the RAM/SSD budget with BatchedMemoryBudget. Gate: B concurrent == B sequential with the cache on, plus golden at slots == 1. Prerequisite for the snapshot-compression beads to help multiplexing.'

D2='Store frontier and prompt-cache snapshots with the GDN delta-rule state as BF16 instead of FP32, and widen back to FP32 on restore. Live decode state stays FP32, and no Metal kernels change. Saving: 61 MiB -> ~30 MiB of GDN state per snapshot on a 35B-A3B (Qwen 3.6 / Ornith). Per-snapshot totals with int8 KV at ~10.6 KB/token: 4K ~103 -> ~72 MiB, 32K ~400 -> ~370 MiB, so more checkpoints fit before the store budget evicts. Touches InferenceStateSnapshot and the store byte accounting. Gate: restore-then-continue vs fresh prefill of the same prompt, compared with the perplexity A/B (benchmark/quant_perplexity_ab.py) on qwen3.5_4B or 9B. The golden baseline never restores a snapshot, so it cannot catch this.'

D3='Follow-up to the BF16 bead. Store snapshot GDN state in LeapQuant format (arXiv 2609.38166, github.com/leapquant/leapquant): a rank-1 compensator fitted by power iteration (r=4 FP16 tokens at 8-bit), per-key-row smoothing scale sqrt(mean|row|), and an 8-bit residual. About 1.19 bytes/element vs 4, so 61 -> ~18 MiB per snapshot (~42% smaller at 4K, ~11% at 32K). Training-free, with no checkpoint change. Prototype it first in the CPU reference (TinyTitanValidation GDNReference) and measure with the same restore-vs-fresh perplexity A/B. Out of scope: quantizing live decode state. The paper speedups come from batch 256-512 on NVIDIA. At batch 1 we are SSD-bound (~200 ms/token), so state traffic is under 1%. Revisit live-state quantization only if multiplexed short-context width becomes the limit.'

A=$(bd create "Prompt and frontier cache for slots > 1" -t feature -p 2 -d "$D1" --json | id_of)
B=$(bd create "BF16 GDN state in prompt-cache snapshots" -t feature -p 2 -d "$D2" --json | id_of)
C=$(bd create "LeapQuant 8-bit GDN state in snapshots" -t feature -p 3 -d "$D3" --json | id_of)
bd dep add "$C" "$B"
echo "filed: $A $B $C ($C depends on $B)"
