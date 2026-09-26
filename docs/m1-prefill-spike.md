# Prefill on an M1 Max: what the spikes measured

Qwen3.8-Flash-Next 125B-A6B 4-bit on an M1 Max (32-core GPU, 64 GB), macOS 27.0,
model on an external Thunderbolt NVMe (`Device Location: External`, PCI-Express).
Every Qwen3.8 verdict elsewhere in `docs/` was measured on a 24 GiB base M3 with an
internal SSD; this is the first record from an M1. Driver: `tools/m1_spike.sh`
(7,879-token prompt, greedy, seed 1, chunk 4096, arms interleaved).

## Spike 1 (commit 85e104a, two rounds)

| arm | prefill s (r1, r2) | vs base | note |
| --- | --- | --- | --- |
| base | 199.23, 202.44 | -- | **39 tok/s** |
| s128 (128 expert slots) | 209.42, 194.46 | +0.6% | noise |
| s256 (256 slots, ~34 GiB) | 211.58, 197.34 | +1.8% | swapped 8 GB, 2x sys time |
| nobound (`TINYTITAN_BOUNDED_IO=0`) | 199.91, 199.01 | -0.7% | noise |
| s256nobound | 217.60, 214.86 | +7.7% | decode collapsed (0.1 tok/s) |
| c2048 (half chunk) | 221.93, 234.36 | +13.6% | ~13 s per extra chunk |

Base, GPU time by role (r1 / r2):

| role | ms | share of GPU |
| --- | ---: | ---: |
| `prefill_attn_router` (12 full-attention layers x 2 chunks) | 59,953 / 59,821 | 42% |
| `prefill_gdn_router` (36 GDN layers x 2) | 31,665 / 31,717 | 22% |
| `prefill_routed_tile` | 29,845 / 31,181 | 21% |
| `prefill_shared_expert` | 12,938 / 12,747 | 9% |
| `prefill_qsa_index` | 6,212 / 6,208 | 4% |
| GPU busy of span | 141,484 of 184,539 / 142,292 of 186,680 | 77% / 76% |

Read: expert cache and page cache do nothing for prefill here (each chunk touches
nearly every expert), and prefill is mostly GPU compute. The chunk-count cost is
real: going 4096 -> 2048 added ~8 s of GPU and ~19 s of non-GPU time.

## Found in the code, then measured in spike 2

- **One compute encoder per token.** The shared-expert scalar gate (and its
  sigmoid) runs one GEMV plus one elementwise dispatch per token, each in its own
  encoder: 8,192 encoders per layer-chunk. The per-token GEMV fallback for a
  q-family projection (attention `q_proj`, GDN `in_proj_qkv`) and for
  bf16-promoted projections does the same. `TINYTITAN_PREFILL_COALESCE=1` keeps
  every kernel, argument and grid and puts a loop's dispatches in one encoder
  (`GEMVRows`); `GEMVRowsTests` requires the result to be byte-identical.
- **Which projection path an M1 takes was never logged.** The Metal 4 tensor-op
  QMM (`MPPPrefillInt4QMM`) is optional; when it does not compile, `.q` falls to
  one GEMV per token. With `TINYTITAN_KERNEL_STATS` the runner now prints a
  `prefill paths:` line naming the path per family and the MPP state.
- **`TINYTITAN_PREFILL_Q_QMM=1`** serves `.q` with the batched QMM instead. It sums
  in a different order, so it can change the output and needs a quality check
  (`benchmark/quant_perplexity_ab.py`) before it could become a default.
- **`TINYTITAN_PREFILL_SPLIT=1`** (diagnostic) commits and times each stage of a
  prefill layer as `prefill_split_*` roles: `hc_in`, `attn_qkv_proj`,
  `attn_rope_kv`, `attn_core`, `attn_o_proj`, `gdn_in_proj`, `gdn_scan`,
  `gdn_out_proj`, `hc_out`. The waits serialize stages; read the roles, not
  the wall clock.
- Two existing switches, off by default pending proof of identical output, are
  spike arms: `TINYTITAN_HC_FUSED=1` and `TINYTITAN_QSA_GPU_SELECT=1` (QSA key
  selection otherwise runs on the CPU between the indexer and attention).

## Not pursued, with the reason

- **Neural Engine attention.** Measured on the M3 for this model
  (`benchmark/ane-prefill/ane-correctness-v5.6-ane-38-4bit.json`): 439 s against
  184 s on the GPU. On an M1 Max the GPU is larger and the ANE older, so it would
  lose by more.
- **Bigger expert cache for prefill.** Spike 1 above.

## Spike 2 (commit 83b019b, two rounds)

Every arm produced the same output as base, `coalesce` included (it is
bit-identical by construction and `GEMVRowsTests` says so), and so did
`TINYTITAN_HC_FUSED=1` and `TINYTITAN_QSA_GPU_SELECT=1`, which were off only until
that was shown. The M1 compiles the Metal 4 tensor-op QMM: every attention and
GDN projection already takes it (`prefill paths: mpp_int4=available attn_q=mpp/4b
gdn_in=mpp/4b kv=mpp/4b o=mpp/4b`), so the per-token q GEMV never runs here and
`qqmm` is a second base.

| arm | prefill s (r1, r2) | GPU busy s (r1, r2) |
| --- | --- | --- |
| base | 197.7, 205.7 | 141.8, 146.3 |
| coalesce | 242.6, 206.3 | 183.9, 145.6 |
| qqmm (= base) | 222.7, 199.5 | 170.7, 145.5 |
| hcfused | 213.2, 209.2 | 158.6, 148.1 |
| qsagpu | 199.3, 210.8 | 145.6, 142.6 |
| combo | 195.0, 194.9 | 141.1, 141.8 |

The spread is the clock, not the arms: an unchanged binary ran 12.6% apart, and
in the slow runs every role -- including ones the flag never touches -- slowed
together. `powermetrics` during prefill showed 93-98% GPU residency at
1.10-1.22 GHz average, 23-57% of the time on the 972 MHz step. Nothing here is
separable from that; `m1_spike.sh --gpu-clock` now reports busy x clock
(Gcycles) so the next comparison is.

Where the GPU time goes (`TINYTITAN_PREFILL_SPLIT=1`, mean of two rounds, s):

| stage | s | share | note |
| --- | ---: | ---: | --- |
| `attn_core` (QSA sparse attention, 12 layers x 2 chunks) | 54.2 | 38% | one kernel, ~2.3 s per layer-chunk |
| `routed_tile` | 30.6 | 21% | one-output-per-thread expert GEMV |
| `hc_in` + `hc_out` (hyper-connections) | 17.5 | 12% | projections on the scalar `prefillQMM`, not MPP |
| `shared_expert` | 12.9 | 9% | unchanged by coalescing: the MLP, not dispatch |
| routers (`attn_router` + `gdn_router` remainders) | 7.2 | 5% | |
| `gdn_in_proj` + `gdn_out_proj` | 8.4 | 6% | already MPP |
| `gdn_scan` (conv, norms, delta rule, gated norm) | 5.1 | 4% | the sequential recurrence is *not* a hotspot |
| `qsa_index` | 4.3 | 3% | |
| `attn_qkv_proj` + `attn_o_proj` + `attn_rope_kv` | 2.9 | 2% | already MPP |

Outside the GPU: ~55-60 s of each ~200 s prefill, of which ~15 s precedes the
first kernel. `TURBO_FIELDFARE_PHASES` (now on stderr, and on in every spike
arm) splits the host side per chunk.

## Spike 3: the scalar GEMMs onto MPP, and the attention kernel's shape

Two findings from reading the code behind spike 2's split:

- **The prefill shared expert is the decode path, once per token.**
  `PrefillSharedExpert.encodeBlock` runs gate, up, activation and down for each
  of a chunk's 4,096 tokens through a one-row scratch, so every token's four
  dispatches wait on the previous token's. Coalescing never touched this loop,
  which is why it did nothing for the 12.9 s. `TINYTITAN_PREFILL_MPP_WIDE=1` runs
  it as three MPP GEMMs over the chunk plus one elementwise activation, and
  routes the other projections still on the scalar `prefillQMM`
  (hyper-connection gates, QSA indexer, PLE) to MPP. It sums in a different
  order: `PrefillSharedExpertBatchedTests` holds it within 2% of the per-token
  path, and a speed win still needs `benchmark/quant_perplexity_ab.py`.
- **The QSA attention kernel repeats its key/value reads 12 times.** It runs one
  threadgroup per (token, query head), but Qwen3.8's 24 query heads share 2
  key/value heads and the QSA selection is per token, shared by every head: the
  12 query heads of a KV head each re-read and re-dequantize the same ~2,048
  selected K and V rows. One threadgroup per (token, KV head) covering its 12
  query heads reads each row once, and its scores (12 x 256 by 256 x keys) and
  output (12 x keys by keys x 256) are matrix-unit shapes. That is the next
  kernel; the Apple10 tensor-ops attention path cannot help here, because a
  selection is present for every 4,096-token chunk and forces the tiled kernel.

### The grouped QSA kernel (`TINYTITAN_PREFILL_QSA_GQA=1`)

`attention_prefill_causal_qsa_gqa` runs one threadgroup per (token, KV head) for
all of its query heads (12 for Qwen3.8, at most 16, head dim at most 256). It
loads the G query rows once, computes each (head, key) score with the same
`prefill_qsa_dot` in phase A (G x 128 dots per tile over 256 threads, where the
per-head kernel left half its threads idle), keeps each head's running max and
sum in threadgroup memory updated by one thread per head in key order, writes
each weight once, and in phase D loads every V element once and feeds it to the
G accumulators a thread owns. Per head, every value goes through the same
expression in the same order as `attention_prefill_causal_qsa_tiled`, so the
output is meant to be byte-identical: `PrefillAttentionQSAGroupedTests` checks
that on Qwen3.8's shape (24/2 heads, dim 256, int8 KV) for the compacted and the
mask selection. K/V traffic per selected key falls 12x; the matrix units are not
used yet -- the next step if this kernel is still the top role.

First test run on the M1: both byte-equality tests failed by exactly one fp16
ULP (max |diff| 0.000244 at values up to 0.503). The grouped kernel used 64-key
tiles against the per-head kernel's 128, and the online softmax rescales at each
tile boundary: the same sum, rounded at different points. The tile is now tied
to `kPrefillQSATile`.

## Spike 3 results (commit 195d317, three rounds, `--gpu-clock`)

The grouped-kernel byte-equality tests pass on the M1 with the tile fix.

| arm | prefill s (r1, r2, r3) | Gcycles (r1, r2, r3) | GPU work vs base | output |
| --- | --- | --- | ---: | --- |
| base | 228.1, 202.6, 202.0 | 164.5, 182.4, 176.2 | -- | reference (`0.1.0.`) |
| mppwide | 184.3, 177.5, 174.8 | 140.7, 143.7, 150.9 | -16.8% | differs (empty reply) |
| gqa | 179.9, 173.8, 172.9 | 139.3, 135.5, 150.6 | -18.7% | **same as base** |
| wide (both) | 157.1, 151.6, 150.2 | 110.5, 108.5, 118.1 | -35.6% | differs (empty reply) |

`gqa` is byte-identical and the largest single cut so far: `attn_core` went from
54.2 s to 28.7 s. `mppwide` and `wide` differ in the same way every round: the
prompt ends mid-sentence in repository prose ("... at tag"), base continues with
`0.1.0.` and the wide arms end the turn at once. That is one close first-token
decision flipping, the same way three times -- rounding, not noise -- and it is
exactly what a quality loss would also look like, so `mppwide` stays off until
`benchmark/quant_perplexity_ab.py` clears it. The kernel itself was re-read for
a real defect: N edges are masked, accumulation is float, and it is built for
the same weight width as the `prefillQMM` it replaces; the one numeric
difference is each dequantized weight rounded to fp16 before the multiply.

Where the GPU time goes now (`splitwide`, one round, s):

| stage | s | spike 2 | note |
| --- | ---: | ---: | --- |
| `routed_tile` | 30.1 | 30.6 | now the top role |
| `attn_core` | 28.7 | 54.2 | grouped kernel |
| `gdn_in_proj` + `gdn_out_proj` | 8.6 | 8.4 | |
| `gdn_scan` | 5.6 | 5.1 | |
| routers (remainders) | 7.4 | 7.2 | |
| `shared_expert` | 4.7 | 12.9 | batched MPP |
| `qsa_index` | 3.9 | 4.3 | |
| `hc_in` + `hc_out` | 3.4 | 17.5 | MPP |
| `attn_qkv_proj` + `attn_o_proj` + `attn_rope_kv` | 2.9 | 2.9 | |
| GPU busy of span | 95.6 of 134.2 | 141-146 of ~185 | 71% occupied |

The host side (`wide`, round 2, `TURBO_FIELDFARE_PHASES`): per chunk,
"route readback + GPU" 31.7 s / 38.4 s and "expert fetch + tiles" 25.2 s /
24.0 s, with 432.5 / 411.5 distinct experts per layer. That is 57.5 + 54.7 GB
of routed experts (active x 48 layers x 2,768,896 B) streamed in 49.2 s, about
2.3 GB/s, against 30.6 s of routed-tile GPU work in the same phase. Read: the
routed half of prefill is now bound by the external drive, not the GPU, and
each extra chunk is one more near-full sweep of the expert corpus. The levers,
in order: a chunk above 4,096 (one sweep for this prompt instead of two), a
faster drive, and only then the routed-tile kernel. About 17 s of each prefill
still precedes the first GPU command buffer.

## After spike 3: what changed, and spike 4

- **The grouped QSA kernel is the default** (`TINYTITAN_PREFILL_QSA_GQA=0` restores
  the per-head one). Its byte-equality test now also covers int4 and fp16 KV.
- **The chunk ceiling is 16,384** (`--prefill-chunk 8192|16384`; the default stays
  4,096 until spike 4 measures it). Everything a chunk sizes was already sized
  from the configured chunk; the one new limit is the key selection, indexed
  `row * visibleKeys + key` in 32 bits, so chunk x context may not pass 2^32 --
  every native context takes 16K, the 512K/1M YaRN contexts cap it at 8K/4K,
  and `RuntimeConfiguration.validate(maxContext:)` refuses the rest with the
  largest chunk that fits. Qwen3.8's chunk scratch is 0.81 GiB at 4K and
  3.22 GiB at 16K.
- **The shared expert no longer blocks the expert reads.** Its command buffer
  is committed and awaited only before the tail that reads it, so the host
  starts the routed tiles' reads while the GPU runs it. Same kernels, same
  queue order.
- **Installer-made models now find their profile row.** The installer writes
  `qwen3.8-flash-next-4bit`; the table is keyed `qwen3.8-flash-next`, so every
  installed model ran on its family fallback (`family-default` in the log). On
  Qwen3.8 that left the expert cache unwired through prefill, which the row
  measured at 1.6-4.7 s per request on re-wire.

Spike 4 (`tools/m1_spike.sh` defaults: ~16K-token prompt, arms `base c8192
c16384`): 4, 2 and 1 chunks, so 4, 2 and 1 sweeps of the routed experts. The
summary now splits each run into its host phases, where the saving should
show: "expert fetch + tiles" is the half the chunk count multiplies.

## Speedups so far, 7,879-token prompt

| step | prefill s | tok/s | vs spike-1 base |
| --- | ---: | ---: | ---: |
| spike 1 base (commit 85e104a) | ~201 | 39 | -- |
| + grouped QSA kernel (now default, same output) | ~173 | 45 | 1.16x |
| + MPP for the remaining GEMMs (opt-in, output differs) | ~151 | 52 | 1.33x |

Rounds 2-3 of spike 3; round 1 of base read the model cold.

## Spike 4 results (commit bd484b5, two rounds, 16,931-token prompt)

| arm | chunks | prefill s (r1, r2) | tok/s | vs base | output |
| --- | ---: | --- | ---: | ---: | --- |
| base (4,096) | 5 | 451.5, 419.3 | 38.9 | -- | reference |
| c8192 | 3 | 395.7, 391.9 | 43.0 | -9.5% | same as base |
| c16384 | 2 | 401.8, 394.4 | 42.5 | -8.6% | same as base |

GPU work is flat across the arms (+0.8% / +2.7% Gcycles), and the install
now resolves its profile row (`tabled`, `keep_wired=true`). Decode, now that
the runs generate 64 tokens: 3.8-4.1 tok/s on every arm.

**Correction to the spike 3 reading.** "Expert fetch + tiles" scales with
*tokens*, not with sweeps of the corpus: ~29.5 s per 4,096 tokens at base,
~23 s at 8K and 16K. Its bulk is GPU -- routed tiles 66-69 s and the per-token
shared expert 28-30 s over the prompt -- and the drive stall is what is left:
~33 s at base, ~11 s at 16K. So the routed half was mostly GPU-bound; the
drive cost ~2-3 s per extra chunk sweep, not the ~25 s spike 3 inferred from
bandwidth alone. The bigger chunk is still a clean win (same output, less
stall), just a smaller one.

**The ~69 s nobody had named.** Every arm spent ~69 s of prefill outside the
per-layer phases -- the same at 5, 3 and 2 chunks, and ~4.1 ms per token
(spike 3's 7.9K prompt: ~31 s, the same rate). That is the PLE n-gram gather:
16 uncached 320-byte `pread`s per token, issued one at a time, each a full
round trip to the external drive. They are now issued together
(`NgramTableReader.gatherConcurrently`); the bytes and where they land are
unchanged, so the output is too.

Where a 16.9K-token prefill goes at c16384 (round 2, 394 s):

| part | s | note |
| --- | ---: | --- |
| dense layers on the GPU (attention + GDN + routers) | ~150 | |
| host time inside the dense phase | ~68 | grows with context: QSA selection runs on the CPU |
| routed tiles (GPU) | ~66 | |
| shared expert, per-token path | ~28 | `TINYTITAN_PREFILL_MPP_WIDE=1` makes it ~10 |
| drive stall in the routed phase | ~11 | |
| PLE gather | ~69 | now concurrent |

## Spike 5 (commit f761aae, two rounds, 16,931-token prompt)

| arm | prefill s (r1, r2) | tok/s | vs base | output |
| --- | --- | ---: | ---: | --- |
| base | 388.6, 352.9 | 45.7 | -- | reference |
| c8192 | 346.5, 333.4 | 49.8 | -8.3% | same as base |
| c8192qsa | 332.0, 346.3 | 50.0 | -8.5% | same as base |
| c8192wide | 280.4, 289.6 | 59.4 | -23.1% | differs (MPP) |

**The PLE gather fix measured:** base 419-451 s in spike 4 -> 353-389 s, and the
time outside the layer phases 69 s -> ~10 s, as predicted.

**`TINYTITAN_QSA_GPU_SELECT` is decode-only** (`RealForwardRunner+Residual`
reads it in the decode QSA path alone), so `c8192qsa` was a second `c8192` --
which is what it measured. The spike-4 table's "~68 s of QSA selection" was also
wrong in its size: it left out `prefill_qsa_index` (20-23 s of GPU).

What is left in the dense phase: 213.6 s (c8192, r2) against 173.9 s of GPU
(attention 79.7, GDN 71.4, QSA indexer 22.8), so **~40 s of host time**, the
same at every chunk size and with or without MPP. Its size and growth match the
prefill key selection: every query row sorts all of its block scores on one
core, 12 layers x every token, O(rows x context log context). The rows are
independent and now run in parallel (`QSAIndexer.selectPrefillRows`,
byte-identical to row order by `QSAPrefillSelectionTests`); the next run says
how much of the 40 s that was.

Best measured so far on this prompt: `c8192wide`, 59.4 tok/s (output differs,
MPP arithmetic); `c8192`, 49.8 tok/s with base's output.

## Spike 6 (commit cbd6004, two rounds, 16,931-token prompt)

| arm | prefill s (r1, r2) | tok/s | vs spike 5 | output |
| --- | --- | ---: | ---: | --- |
| base | 338.8, 314.8 | 51.9 | -10% | reference |
| c8192 | 300.9, 297.4 | 56.6 | -12% | same as base |
| c8192wide | 251.0, 246.0 | 68.2 | -13% | differs (MPP) |

**The parallel prefill key selection measured:** the dense phase's host time
fell from ~40 s to ~7 s (c8192, r2: 177.7 s of phase against 171.0 s of GPU),
and GPU occupancy rose to 82-87%. Output unchanged.

Against spike 4's base on this prompt (419-451 s, which already had the grouped
QSA kernel), `c8192` is now 1.45x and `c8192wide` 1.75x. Prefill is now GPU-bound:
of c8192's 297 s, 269 s is kernels -- attention layers 78, GDN layers 70,
routed tiles 67, shared expert 28 (11 with MPP), QSA indexer 22 -- so the next
wins are kernel work, not scheduling.

One scheduling residue: the 547-token tail chunk costs ~20 s (13.7 s routed,
6.9 s dense), about twice its share per token, because it still reads ~286
experts per layer for few tokens. Balanced chunks (16,931 as 2 x 8,466 under a
16K ceiling) would remove it, but frontier checkpoints and split prefills
assume chunk boundaries at whole multiples of the chunk size from the resume
point, so it is not a planner-only change.

## Spike 7 (commit d61a8b5, two rounds, 16,931-token prompt)

| arm | prefill s (r1, r2) | tok/s | Gcycles |
| --- | --- | ---: | ---: |
| c8192 | 307.3, 302.9 | 55.5 | 333 |
| c8192wide (MPP) | 245.1, 249.6 | 68.5 | 270 |
| c8192sg (simdgroup QMM) | 262.8, 267.7 | 63.8 | 293 |

The simdgroup-matrix QMM compiles and matches the scalar kernel, but it is
slower than the MPP QMM here (GDN role 59.8 s against 47.3 s): it stays opt-in.

**Correction to the efficiency reading behind it.** Dividing whole roles by
their FLOPs put every GEMM at ~1 TFLOPS. Spike 2's per-stage split says
otherwise for the dense projections: `gdn_in_proj` + `gdn_out_proj` did
~33 TFLOP in 8.4 s at 7.9K tokens, ~3.9 TFLOPS on MPP. What is genuinely slow:

- **the routed tiles**, ~80 TFLOP in ~70 s, ~1.2 TFLOPS: one output element
  per thread, a full dot product each. `TINYTITAN_PREFILL_ROUTED_MPP=1`
  (`PrefillRoutedExpertGEMM`) runs each tile as grouped MPP GEMMs instead --
  gather the tile's token rows, gate and up per expert, one activation over
  the tile, down per expert, scatter to the same route-partial slots.
- **the shared expert's scalar gate**: ~2 GFLOP of work, but 2 x 8,192 one-token
  encoders per layer-chunk, most of `shared_expert`'s ~11 s under MPP.
  `TINYTITAN_PREFILL_COALESCE` (bit-identical, spike 2) is now on by default.

## Spike 8 (commit 8ad4c17, two rounds, 16,931-token prompt, on battery)

| arm | prefill s (r1, r2) | tok/s | Gcycles | routed GPU s |
| --- | --- | ---: | ---: | ---: |
| c8192 (coalesced rows now default) | 292.2, 289.3 | 58.2 | 324 | 66.1 |
| c8192wide | 239.4, 235.9 | 71.3 | 256 | 66.5 |
| c8192widerouted | 227.4, 229.2 | 74.2 | 261 | 65.7 |

Coalescing measured: `shared_expert` 29.3 -> 26.0 s, and 11.3 -> 8.6 s under
MPP (spike 7 vs 8). The grouped routed GEMMs changed the output but not the
routed GPU time: each expert's GEMM was its own encoder of ~60 threadgroups,
51 of them in sequence per tile, under two threadgroups per core. They now
go into one concurrent encoder per phase (commit 92481b7), and grouped tiles
are timed as `prefill_routed_gemm`.

## Surprisal check: MPP wide + routed MPP (commit 8ad4c17)

`tools/prefill_surprisal_ab.sh`, 18,652 tokens of context prefilled at chunk
8,192, then 512 tokens teacher-forced (token hashes equal):

| arm | mean NLL | perplexity |
| --- | ---: | ---: |
| A: no switches | 2.21386 | 9.1510 |
| B: `TINYTITAN_PREFILL_MPP_WIDE=1 TINYTITAN_PREFILL_ROUTED_MPP=1` | 2.21293 | 9.1425 |

B - A: mean dNLL -0.00093, se 0.01676, t -0.06 (perplexity -0.09%): no
measurable change; at this length the test resolves about +/-3.4% perplexity
(2 se). Individual tokens do move -- mean |dNLL| 0.21, 66 of 512 by more than
0.5 nats -- in both directions, as expected when the prefill arithmetic shifts
which keys the QSA selection keeps. Concurrent encoding (92481b7) leaves every
value the same, so the verdict covers it.

## Spike 9 (commit 4450911, two rounds, 16,931-token prompt)

| arm | prefill s (r1, r2) | tok/s | Gcycles | routed GPU s |
| --- | --- | ---: | ---: | ---: |
| c8192wide | 243.1, 236.5 | 70.6 | 256 | 66.6 (tile kernels) |
| c8192widerouted | 214.6, 213.8 | 79.1 | 218 | 34.9 (grouped GEMM, all 7,307 tiles) |

With each tile's expert GEMMs in concurrent encoders the routed half runs at
~2.3 TFLOPS, half its old time, -15% GPU work overall. The GPU is now idle
~20% of the span again (80% occupied): "expert fetch + tiles" is 64.5 s
against 43.4 s of routed and shared GPU, so the drive is back on the path.

Against spike 4's base on this prompt (419-451 s, already with the grouped QSA
kernel): 2.03x. Against the original engine's estimated ~485 s: ~2.3x.

## Spike 10 plan: the last compute on the GPU

Spike 9's 214 s splits roughly: QSA attention ~71 s, GDN ~45 s, routed ~35 s,
QSA indexer ~17 s, shared expert ~8.5 s, off-GPU ~45 s. Three switches, each
arm adding one:

- `c8192fast`: `TINYTITAN_QSA_SCORE_MMA=1`, the indexer's block scores
  (`qsa_block_scores_rows_mma`) on the matrix units; previously ~26 GFLOPS.
  Float sums in another order only.
- `c16384fast`: the same at chunk 16,384. The ~16K prompt then sweeps the
  routed experts twice instead of three times; this tells whether the
  drive-bound tail chunk is worth the larger scratch.
- `c8192attn` / `c16384attn`: plus `TINYTITAN_PREFILL_QSA_MMA=1`, QSA attention
  (`attention_prefill_causal_qsa_gqa_mma`) on the matrix units. Q is padded
  to 16 rows per KV head; for each 16 selected keys, QK^T and P.V run as 8x8
  float matrix products over four simdgroups. The online softmax rescales every
  16 keys rather than every 128, so the output moves by rounding; that needs
  the surprisal A/B before it could become a default.

The weights stay on the external drive: the drive is only worth moving if it
remains the bottleneck after these.

## Spike 10 (commit 773381b, two rounds, 16,931-token prompt)

| arm | prefill s (r1, r2) | tok/s | Gcycles | chunks | dense / routed phase s (r2) |
| --- | --- | ---: | ---: | ---: | --- |
| c8192widerouted | 213.8, 212.9 | 79.4 | 217 | 3 | 139.7 / 63.7 |
| c8192fast | 219.5, 214.4 | 78.1 | 198-201 | 3 | 125.5 / 78.9 |
| c16384fast | 189.5, 183.6 | 90.8 | 192 | 2 | 125.8 / 48.4 |
| c8192attn | 180.4, 170.5 | 96.6 | 163 | 3 | 97.5 / 63.5 |
| c16384attn | 156.6, 156.1 | 108.4 | 160 | 2 | 98.7 / 48.0 |

- **Indexer on the matrix units** (`TINYTITAN_QSA_SCORE_MMA`):
  `prefill_qsa_index` fell from 17.0 s to 1.3 s. The time did not go away at
  8,192: the dense phase dropped ~14 s, but the routed phase rose by about as
  much, because the expert prefetch no longer overlaps as much dense work and
  the drive now shows. The output hash matches `c8192widerouted`.
- **Chunk 16,384**: one routed sweep fewer, so the routed phase drops from
  ~64-79 s to ~48 s, and GPU occupancy rises to 86%. It costs 0.6 GiB more RSS
  and no swap.
- **QSA attention on the matrix units** (`TINYTITAN_PREFILL_QSA_MMA`):
  `prefill_attn_router` fell from 72 s to 45.5 s, a 27% cut in GPU work
  overall. The output changes, as expected from rounding, and stays coherent.
  Decode tok/s in these arms (4.2-4.4 against 3.7) follows from generating
  different text, not from the switch: decode does not run the prefill kernel.

`c16384attn` against spike 4's base (419-451 s): 2.8x. Against the original
engine's estimated ~485 s: ~3.1x. What is left: GDN 45.7 s; attention layers
45.6 s (projections, hyper-connections and attention); routed 30.3 s; shared
expert 8.3 s; indexer 1.4 s. Of the routed phase's 48 s, about 10 s is the
drive waiting beyond the routed and shared GPU work. The drive is not yet the
last bottleneck, so the weights stay external.

Surprisal judges the whole launch configuration against the no-switch
reference (`tools/prefill_surprisal_ab.sh --b-chunk 16384`).

## Surprisal: the spike 10 launch configuration (commit efad983)

19,087 tokens of context, then 512 tokens teacher-forced (the token hashes
match):

| arm | chunk | mean NLL | perplexity |
| --- | ---: | ---: | ---: |
| A: no switches | 8,192 | 2.13440 | 8.4520 |
| B: `MPP_WIDE ROUTED_MPP QSA_SCORE_MMA PREFILL_QSA_MMA` | 16,384 | 2.14580 | 8.5489 |

B - A: mean dNLL +0.0114, se 0.0124, t +0.92 (perplexity +1.15%). No
measurable change; the test resolves about +/-2.5% perplexity (2 se). The
per-token movement is smaller than the MPP-only check: mean |dNLL| 0.160
against 0.214, 38 tokens beyond 0.5 nats against 66, none beyond 2. 276 tokens
went down and 236 went up.

The two surprisal runs used different text: the default prose includes this
document, which grew between them, so their A arms are not comparable (2.214
against 2.134). `--text <earlier run>/text.txt` now pins the text.

Launch configuration (108 tok/s prefill on the ~17K prompt):

    TINYTITAN_PREFILL_MPP_WIDE=1 TINYTITAN_PREFILL_ROUTED_MPP=1 \
    TINYTITAN_QSA_SCORE_MMA=1 TINYTITAN_PREFILL_QSA_MMA=1 ... --prefill-chunk 16384
