# Verdict: speculation, placement and compression for Qwen3.8 4-bit on this M3

> **Note on `--ram` provenance, 2026-09-22.** `--ram-budget` (the launcher's
> `--ram`) now names a target for the *whole process*, not the expert cache: the
> cache gets the target minus the resident weights and a measured runtime reserve.
> Every `--ram 8` measurement in this document was taken under the old meaning,
> where 8 GB was 64 slots of cache and ~11.6 GiB of real use; today `--ram 8` is
> a ~32-slot / ~7.8 GiB configuration and `--ram 12` is the flag that reproduces
> the old one. See Lever 10 of
> `benchmark/internal-speeds/v2-qwen38-4bit-telemetry.txt`.

Written 2026-09-21 after testing the three concepts against the *installed*
configuration. The proposal they came from was written for an FP8 install on an
8-16 GB device; this machine runs the 4-bit affine install, and several of its
premises do not hold here. Everything below is measured on this build unless
labelled otherwise.

## Premise corrections

| proposal assumes | this install |
| --- | --- |
| FP8, ~6B activated FP8 parameters | **4-bit affine**, group 64, BF16 scales/biases |
| ~2.36 GB routed experts | **63 GB** packed experts (512 experts x 48 layers x 2.77 MB) |
| 51 GB n-gram table | **102.4 GB** (`ngram_table.bin`) |
| 8-16 GB device, cache absorbs 15-35% of expert reads | 24 GiB device; measured **75.7%** hit rate at the shipped 64 slots |
| pin the dense backbone in RAM | **already the design**: `model_weights.bin` is mapped resident; only `packed_experts` stream |

## Concept 1 — MTP speculation: unreachable by default, and it loses when forced

**It cannot engage on the shipped sampling configuration.** The decode loop picks
the draft producer only when the request is pure greedy
(`RawCompletion.swift`: `temperature == 0 && presencePenalty == 0 &&
repetitionPenalty == 1`). Since the 2026-09-21 sampling change, every
thinking-off Qwen3.8 request resolves to the instruct row with **presence penalty
1.5**, so `isPureGreedy` is false and the draft path is skipped. Evidence: with
the sidecar attached (`mtp=on:384MiB`) and shipped sampling, both arms of a
256-token A/B ran **256 scalar passes** (`sample count=256`,
`head_logits count=255`) — the head was loaded and never used.

Forced reachable with `presence_penalty: 0` on **both** arms, `--ram-budget 8G`,
256 tokens, temperature 0, two paired rounds:

| round | scalar | MTP | delta |
| ---: | ---: | ---: | ---: |
| 1 | 4.611 | 3.121 | -32.3% |
| 2 | 4.612 | 2.448 | -46.9% |

The verify path was genuinely running: `verify_routed_pair` 7,872 dispatches,
`verify_head` 164 passes, i.e. ~1.56 emitted tokens per pass.

**And the output was not identical.** Two pure-greedy arms (same target profile
echo, same sampling echo, only the draft attached) diverged at character 553:
scalar "*...safety, and modern UI/UX patterns*" against MTP "*...safety, and ease
of use within the Apple ecosystem*", 1,254 against 1,258 characters. The target's
profile line is identical in both arms; the only extra profile line is the
sidecar's own. So this is the verify path, not a configuration difference.

The repo's own Track B3 gate requires **byte-identical greedy output and >=10%
median gain** (`benchmark/emone_mtp_b3_qualification.py`). This measurement
fails both halves. I did not run the B3 harness itself; the A/B above is the
evidence. Prior art agrees on the loss and explains it: the 2026-09-18
measurement put a pair pass at 2.238x a scalar token for 1.869 emitted tokens
(86.9-92.6% acceptance), concluding "the lever is the verify path, not
acceptance" — and the scalar arm has since gained prefetch depth 1, so the same
per-pass cost now reads worse.

The n-gram/prompt-lookup half is **not implemented**: the 102 GB `ngram_table.bin`
is the model's own PLE gather (deterministic lookups from token ids), not a draft
source, and adding a draft source does not change the verify-side cost that is
the recorded cause.

**Cross-feature finding:** the presence-penalty change silently disabled the
speculative path for every default Qwen3.8 chat request, and nothing caught it
because the two features were never exercised together. Whoever revisits MTP
should start with the B3 harness at `presence_penalty: 0`, and must explain the
output mismatch above before quoting any speed number.

## Concept 2 — placement rebuild: half of it is already done, and the rest is low-value here

The dense backbone, embeddings and head are already **resident** (mapped, not
streamed); only the 63 GB expert corpus is streamed. The remaining ideas (hot/cold
expert pin file, co-activation-sorted disk layout, draft-driven prefetch) face a
machine where the SSD is **not saturated** — 109 GiB of device traffic per
256-token run, ~1.4-2 GB/s average against ~3.6 GB/s — and where the cache is at
its measured optimum (64 slots; 96 and 128 are 12% and 50% slower, telemetry
Lever 3 and 7). The binding constraint here is the wait, not bandwidth, so a
layout permutation has little to win. It would need an offline routing-frequency
profile per workload plus a 63 GB repack; worth doing only if the wait analysis
ever shows bandwidth-bound behaviour.

## Concept 3 — lossless compression: ~10% ceiling, and 0% with a codec fast enough for the hot path

Measured on 128 MiB samples of the real files (ratio = compressed / original):

| sample | lz4 -1 | zstd -1 | zstd -3 | zstd -12 | xz -6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `packed_experts/layer_00.bin` | 100.00% | 90.87% | 89.76% | **89.56%** | 89.16% |
| same, even/odd byte planes | 95.96% | 89.19% | 88.83% | 89.08% | 89.09% |
| `ngram_table.bin` | 100.00% | 76.91% | 76.91% | 76.26% | **71.12%** |

The expert bytes are 4-bit affine — already dense — so ~10% is the ceiling, and
the de-interleaved plane split only improves it by ~1 point. **lz4, the only
codec fast enough to sit in a ~3.6 GB/s read path, achieves 0%.** zstd -12 costs
single-digit MB/s per core, orders of magnitude below the device. The n-gram
table does compress 23-29%, but it carries ~5 KB per token, so that is disk
occupancy rather than throughput. Roughly 6 GB of disk saved on 63 GB, paid for
with a decompression core per unit of read bandwidth, is a bad trade for a decode
path that is wait-bound rather than bandwidth-bound.

### Why "fewer bytes" does not become "more bandwidth" here

Compression does cut bytes on the wire; the question is whether the decompressor
can keep up, and on this data the answer is no by a wide margin. Measured on the
real expert bytes: `zstd -12` gives 134,217,728 -> 120,206,303 B (89.56%) and
decompresses at **0.96 GB/s per core** (120.2 MB in 0.12-0.13 s).

Per decode token, at the shipped 64 slots (`--ram 8`):

| quantity | value |
| --- | ---: |
| expert demand reads | 307 MiB (116 misses x 2.638 MiB) |
| compressed to 89.56% | 275 MiB |
| I/O time saved at the saturated ~3.1 GB/s | **~10.5 ms** |
| decompression of 275 MiB at 0.96 GB/s/core | **~300 ms of core time** |

So the byte saving is ~10 ms/token and the CPU cost is ~300 ms/token on one core
— about 30x the prize, and even spread over eight cores (~37 ms) it exceeds the
saving while competing for the same memory controller and package power. The repo
has measured that competition directly: real CPU compute during decode raised
GPU-busy 44.9% and cut throughput 22.6% (CPU co-execution), so "spread it over
spare cores" is not free either. Compression would also add a staging copy, since
today the bounded reader `pread`s straight into the slot buffer.

**Why the ratio is so small.** The checkpoint is already a lossy compression: 4-bit
affine, group 64, with BF16 scale and bias per group. A group's record is 32 B of
4-bit codes plus 2 B scale plus 2 B bias, so only ~11% of the bytes are even
candidate material for an entropy coder, and the 4-bit codes are near-uniform.
Perfect coding of the scale/bias streams would therefore cap out near 11%, which
is what zstd's 89.6% is already approaching. The ratios in the ZipNN paper come
from BF16/FP8 checkpoints whose exponent bytes are highly redundant; those bytes
do not exist in this install.

Where compression *would* pay on this machine: the 102 GB `ngram_table.bin`
(71-77% with xz / zstd -12, i.e. ~25-30 GB of disk) whose per-token traffic is
~5 KB, and any future checkpoint that ships BF16 or FP8 rather than 4-bit.

### Could the decompression move to another engine stage?

"Compress the stream and unpack it inside the engine" changes *who* pays, not
whether it pays. The decisive comparison is the SSD wait removed against the cost
of undoing the compression, per decode token (307 MiB of demand reads at
`--ram 8`):

| option | cost per token | versus 10.5 ms saved |
| --- | ---: | --- |
| CPU `zstd -d` at 0.96 GB/s/core | ~300 ms (1 core), ~37 ms (8 cores) | loses 3-30x |
| GPU unpack: read 275 + write 307 MiB at ~60 GB/s | **~10 ms** | a wash before other costs |
| staging copy alone (307 MiB memcpy, measured) | ~9 ms | already eats the saving |

The GPU path is the only one that is even close, and it is close because the
decompression *is itself* a memory-bandwidth operation: it saves 32 MiB on the
3.1 GB/s link and spends 582 MiB on the ~60 GB/s one, which at a 1.1x ratio and a
~20x link-to-memory gap is designed to cancel. It would also land in series with
the layer's compute (the MoE kernel cannot start until the record is unpacked),
needing the same overlap that the exposed SSD wait needs, and would require a
repacked 63 GB checkpoint plus a new unpack kernel for zero expected net gain.

The engine already uses compression where the arithmetic works, and it is lossy
on purpose: 4-bit affine weights (4x fewer bytes than FP16) and 8-bit KV (with a
4-bit option). Those pay because the reduction is 2-4x, not 1.1x — the general
rule being that compression inside a stage wins when the slow link is far slower
than the memory doing the decoding *and* the ratio is large. A network hop between
shards (10-100x slower than memory) is the case that qualifies; an NVMe at
3 GB/s feeding a GPU with 60 GB/s and a 1.1x ratio is not.

### Tested, 2026-09-21: the GPU-unpack variant

The estimate above (~10 ms for a GPU pass) was measured rather than left standing.
A pure GPU blit of one token's records — 307 MiB read plus 307 MiB write, i.e.
**the floor for any unpack kernel**, since a decoder can only be slower than a
copy — costs:

| quantity | measured |
| --- | ---: |
| GPU unpack pass, lower bound (blit) | **7.87 ms** (81.8 GB/s of traffic) |
| SSD saved, zstd -12 ratio (89.56%) | 10.5 ms |
| SSD saved, order-0 codec bound (byte entropy 7.444/8 = 93.0%) | 6.9 ms |
| CPU `zstd -d` | ~300 ms per core |

The numbers that decide it are the entropy ones, because a GPU can implement an
**order-0** coder (rANS/Huffman) and cannot practically implement LZ match
finding. Measured on the real bytes: raw byte entropy **7.444/8 bits**, 4-bit code
entropy **3.773/4** — so an order-0 coder caps out near 93%, saving **6.9 ms**,
which is *less than the 7.87 ms floor* its own unpack pass costs. Only a
match-finding codec (zstd -12 at 89.56%) beats the pass, and then only by
2.6 ms/token (~1%), while being impractical on the GPU and costing 300 ms/core on
the CPU. A plane split is a permutation and saves 0% on its own — the 88.83% it
reaches still needs the entropy coder.

So the variant is not worth building: under a codec a GPU can actually run it is a
net loss, and under the one codec that would win by ~1% it cannot run there.

## Summary

| concept | measured outcome |
| --- | --- |
| MTP speculation | not reachable on shipped sampling (presence 1.5); forced reachable it is 32-47% slower **and** changes greedy output |
| n-gram drafting | not implemented; does not address the recorded verify-path cost |
| placement rebuild | dense backbone already resident; SSD unsaturated; cache at optimum -> low value here |
| lossless compression | ~10% limit on experts, 0% with a fast codec; n-gram table 23-29% for negligible traffic |

## Item 3 — the I/O path: split reads and separated pools cannot add bandwidth

Code facts (`sources/EmoneKernelsC/expert_io.c`): each miss is one `pread` of
the whole 2,768,896 B expert; a pool of 4 workers with one fd each claims indices
under a mutex; `submit_batch` is **single-batch-at-a-time and synchronous**, so a
demand batch waits on `batch_idle` behind any in-flight speculative batch; and
speculative batches already run on a throttled disk tier (`setiopolicy_np`).

Device measurements on the real install, `F_NOCACHE`, random experts across the
whole 63 GB corpus:

| outstanding 2.64 MiB reads | per-read latency | aggregate |
| ---: | ---: | ---: |
| 1 | 1.20 ms | 2.15 GB/s |
| 4 | 3.42 ms | 3.02 GB/s |
| 8 | 6.49 ms | 3.18 GB/s |

The device saturates at **~3.0-3.2 GB/s**. Past roughly four outstanding reads
extra concurrency buys no bandwidth and only lengthens every read.

- **Splitting a miss is refuted.** Four sub-reads of one expert measure p50
  1.04 ms against 1.10 unsplit — a tail improvement only (p99 7.2 -> 1.45 ms) —
  and applied to a 2-3-miss layer it would raise outstanding reads from ~3 to
  ~12, which the table prices at 6-15 ms per read with flat aggregate.
- **Separating the pools is refuted.** With the prefetch ring off, in-situ
  `expert_load_p50_ms` is still **4 ms** (`io_ms` 116.7/114.9, `wait_ms`
  91.4/90.3, 96,399 MiB read) against **4 ms** with the ring on (95.7/95.5,
  83.1/82.8, 80,647 MiB) — the 4 ms is the device at its 4-outstanding knee
  (3.42 ms measured), not speculative interference. The ring in fact *lowers*
  bytes and wait and decodes 17% faster (3.945/4.051 against 4.659/4.666 tok/s).
- Throttling speculative reads is the wrong direction for queue occupancy: at 16
  outstanding, utility-tier reads take 24.7 ms against 15.5 ms at the default
  tier, so a throttled read occupies the queue longer.

**Why there is almost no room.** 116 misses/token x 2.638 MiB = **307 MiB/token**,
and at the saturated 3.0-3.2 GB/s that is **94-100 ms/token of device time** —
which is what `io_ms` (96-117) and `wait_ms` (83-91) measure. Decode is running
at the device's saturated throughput. The only levers that can move it are
reading fewer bytes (the cache is at its measured optimum, and precision cuts and
expert dropping are excluded by the quality and RAM rules) or overlapping those
307 MiB with compute — the architectural item the repo measured as backwards in
the pre-prefetch era and which the per-layer dependency makes hard. Re-arranging
reads cannot add bandwidth.
