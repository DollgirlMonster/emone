# Expert-prefetch predictor: what is achievable, and what actually limits it

> **Note on `--ram` provenance, 2026-09-22.** `--ram-budget` (the launcher's
> `--ram`) now names a target for the *whole process*, not the expert cache: the
> cache gets the target minus the resident weights and a measured runtime reserve.
> Every `--ram 8` measurement in this document was taken under the old meaning,
> where 8 GB was 64 slots of cache and ~11.6 GiB of real use; today `--ram 8` is
> a ~32-slot / ~7.8 GiB configuration and `--ram 12` is the flag that reproduces
> the old one. See Lever 10 of
> `benchmark/internal-speeds/v2-qwen38-4bit-telemetry.txt`.

Round-1 study for the goal "develop a better expert predictor so we get closer to
10 tok/s", 2026-09-22, all runs at `--ram-budget 8G` on the 24 GiB M3. Every
number here is measured; the method is reproducible from the trace facility.
**Round 2**, at the end of this document, runs the one scheduling idea round 1 left
open and closes it.

## Method: ground truth instead of inference

`EMONE_PREFETCH_TRACE=<path>` writes one JSON line per (layer, position) with
`experts` (the router's exact selection), `misses`, `resident` (both captured
before that layer's plan), `next_layer_prediction` and `next_weights` (the probe
for the next layer), and `next2_layer_prediction`. Two traces were captured: a
255-token prose decode and a 52-token code decode.

One subtlety had to be resolved before any scoring was meaningful: the recorded
`misses` are **post-adoption** (an expert the ring staged and the plan adopted is
resident by then, so it is not a miss). Scoring a prediction against `misses`
therefore reports ~1% for a predictor that is actually 55% right — the first
offline pass made exactly that mistake. The correct target is the *selection*
set, filtered by the residency the ring sees at issue time. Validated against the
engine's own footer: the simulation reproduces `prefetch_issued_per_token=43.0`
against the measured 42.74 and `prefetch_adopted_per_token=23.7` against 23.04.

## The probe is not the weak link

Selection precision by rank, after dropping experts already resident:

| ordering | P@1 | P@2 | P@3 | P@4 | adopted/token @1 / @2 / @3 |
| --- | ---: | ---: | ---: | ---: | --- |
| **probe (shipped)** | **0.552** | **0.450** | 0.379 | 0.328 | **23.7 / 16.3 / 10.8** |
| probe re-ranked by live per-layer frequency | 0.522 | 0.452 | 0.401 | 0.352 | 22.4 / 16.4 / 11.4 |
| per-layer frequency alone | 0.024 | 0.025 | 0.026 | 0.025 | 1.0 / 1.1 / 1.1 |
| previous token's selection at that layer | 0.000 | 0.000 | 0.000 | 0.000 | 0 / 0 / 0 |
| previous token's misses at that layer | 0.000 | 0.000 | 0.000 | 0.000 | 0 / 0 / 0 |
| probe then previous token (and the reverse) | 0.552 | 0.450 | 0.379 | 0.328 | 23.7 / 16.3 / 10.8 |

The code-shaped prompt gives the same picture (probe 0.554 / 0.428 / 0.364 /
0.305). Two findings:

- **The shipped probe is the best ordering available from the information on
  hand.** Re-ranking by frequency, by the previous token's routing, by an EMA of
  probe scores, or by unions of those, is neutral or worse. The previous token's
  experts are always already resident, which is why a temporal predictor stages
  nothing at all: the cache has already absorbed that reuse.
- **The probe's ranking is underused.** Ranks 2 and 3 are still 45% and 38%
  precise. Staging two reads per layer would adopt ~40/token instead of 23.7, and
  three would adopt ~51 — the largest measured headroom in the whole decode path.

## Why more staged reads did not pay

The ring staged only one read per layer because it submitted **one
`ExpertLoadOperation` for the whole batch** and assigned that single operation to
every slot it filled; `readyBuffers` adopts a slot only when its operation is
`.completed`, so a slow sibling gates a fast one. That is why the earlier
`PREFETCH_TOP_M=2` arm adopted just 15.8/token while issuing 76.1.

Fix (in this commit, opt-in `EMONE_PREFETCH_PER_EXPERT=1`): submit one
operation per staged expert so readiness is per slot. Two interleaved rounds per
arm, 256-token prose prompt:

| arm | decode (r1 / r2) | vs shipped | issued/token | adopted/token | precision | expert read MiB | wait_ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shipped (1 slot, batch op) | 4.626 / 4.472 | — | 42.8 | 23.3 | 0.54 | 80,639 | 83.0 |
| 1 slot, per-expert | 4.428 / 4.484 | -2.0% | 42.7 | 22.8 | 0.53 | 80,996 | 86.8 |
| 2 slots, per-expert | 4.498 / 4.300 | -3.3% | 76.2 | **26.0** | 0.34 | 78,812 | 85.4 |
| 3 slots, per-expert | 4.055 / 4.068 | -10.7% | 102.9 | **26.8** | 0.26 | 78,273 | 91.1 |
| *(earlier, for contrast)* 2 slots, batch op | 4.236 / 4.245 | -6.6% | 76.1 | 15.8 | 0.21 | 85,709 | — |

The fix does what it was designed to do: two-slot adoption rises 15.8 -> 26.0
(+65%), device reads fall to *below* the shipped arm, and every arm stays
byte-identical. It still loses throughput, for two reasons that the counters name:

1. **The device is saturated, so speculation taxes demand.** `wait_ms` climbs
   83.0 -> 85.4 -> 91.1 as slots rise, and the ~2.7 extra adopted reads/token
   (≈7 MiB ≈ 2 ms of device time) are outweighed by longer demand queues.
2. **One layer of lookahead is about one loaded read long.** The ring issues at
   layer L for L+1, giving roughly 4.5 ms of headroom; measured cold-device
   per-read latency is 1.2 ms idle but 3.4-6.5 ms with four or more reads
   outstanding, so a good share of staged reads are still in flight when the
   target plan runs. That is also why precision per issued read falls
   (0.54 -> 0.34 -> 0.26) even though the ranking's intrinsic precision does not.

## Conclusion and next step

At one-layer lookahead the shipped configuration is the optimum, and a *better
predictor by itself cannot reach 10 tok/s*: the ranking is already the best of
every candidate testable from the available information, and its unused tail
cannot be cashed in while the device is saturated.

The one predictor-side avenue left is **lookahead length**, not ranking quality:
two layers ahead gives about 9 ms of completion headroom instead of 4.5, which is
the measured difference between a staged read arriving and not. A trace captured
with `EMONE_PROBE2_TRACE=1` scores the engine's two-layer-ahead prediction
against the same ground truth:

| prediction | issued/token | would-adopt/token | P@1 |
| --- | ---: | ---: | ---: |
| L+1 probe (shipped) | 43.0 | 23.7 | **0.552** |
| L+2 probe | 42.9 | 19.8 | **0.462** |

So the far-horizon prediction is nearly as good (46.2%), and the earlier
`EMONE_PREFETCH_AHEAD=2` A/B adopted **19.0**/token — essentially all of what
the L+2 ranking offers, i.e. unlike the two-slot case the reads *did* arrive in
time with double the headroom. It still lost 3.7%.

That closes the question the way the byte accounting predicts. The ring's whole
benefit is *byte removal*: its 23.7 adopted reads/token are exactly the 62 MiB
per token by which device reads fall (measured 377 -> 315 MiB/token, 23.7 x
2.64 MiB = 62.6). Per-expert readiness shows the same identity in the other
direction: +2.7 adopted reads/token is -7 MiB/token of reads, measured. But every
extra speculative read is issued *into the same saturated device*, so it costs
demand-read queue time (wait_ms 83.0 -> 85.4 -> 91.1) that cancels — and at three
slots, exceeds — the bytes it saves.

**Where that leaves the goal.** A better predictor cannot reach 10 tok/s on this
machine. The ranking is already the best available at both horizons (55%/46%), its
unused tail is real, and the read that would cash it is issued into a queue the
device cannot drain faster than 3.0-3.3 GB/s. The remaining levers are the ones
already established elsewhere: fewer bytes per token (the cache is at its measured
optimum; precision cuts and expert dropping are excluded by the quality and RAM
rules), a faster device, or scheduling work that removes bytes *without* adding
queue pressure — the last of which is the only untested idea this study leaves,
and it is a scheduling question rather than a predictor one.

Round 2 below runs that last idea and closes it: there is no such schedule.

---

# Round 2 — the last idea: spend the second slot on a second horizon

Measured 2026-09-22, same machine, same conditions (`--ram 8`, 256-token prose
prompt, temperature 0, two interleaved rounds, binary `d9e1a084…`). The idea is an
allocation question: a two-slot ring can buy its second read from the next
non-resident expert of L+1's probe or from the first of L+2's. The two are almost
equally precise (0.450 against 0.462 on the round-1 trace) but the far read gets a
whole extra layer of compute to land in — and readiness is where the two-slot arm's
adoption went (26.0 adopted of the 40 its ranking offered, with the missing 14
re-read by the demand path). The arm was built as an experiment patch (one read per
horizon, each the first non-resident expert in its own rank order), measured, and
then **reverted**: the tree carries no losing code, only the measurement. The
patch was two `begin` calls where the shipped path makes one, plus enabling the
L+2 probe behind a `prefetchMixedHorizon` flag — neither is in the tree. The arms
below are reproducible from this description and from
`benchmark/emone_knob_sweep.py`'s `prefetch_per_expert`.

| arm | r1 | r2 | mean | delta | issued/token | adopted/token | adopt/issued | demand MiB/token | io_ms | wait_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shipped (near, 1 read) | 4.570 | 4.548 | **4.559** | — | 42.7 | 23.3 | 0.55 | 315.0 | 95.6 | 85.9 |
| mixed horizon (near + far) | 4.509 | 4.477 | 4.493 | **-1.4%** | 58.7 | 24.8 | 0.42 | 311.0 | 96.8 | 87.0 |
| two ranks of L+1, per-expert | 4.481 | 4.514 | 4.498 | **-1.3%** | 76.2 | 26.2 | 0.34 | 307.3 | 99.3 | 86.1 |

All three arms returned the same response (`2087738fce…`), and all three ran at the
same hit rate to within 0.6 points (0.7577 / 0.7607 / 0.7635).

- **The second slot substitutes a read rather than adding one.** The arm issued
  16.0 more reads a token and adopted 1.5 more. The ring counters say why: a far
  read holds its slot for two whole layers, so `held` is 0.58 slots at a `begin`
  (against 0.00 shipped) and the ring is nearly full (`free` 1.03 of 2). The slot
  the far read occupies is the slot the near read would have had, and the near
  read is the better one (0.55 against 0.43), so the trade is roughly a wash on
  adoption and a loss on device time.
- **More bytes removed, slower anyway.** Demand bytes fall in both wider arms
  (315.0 → 311.0 → 307.3 MiB/token) and `io_ms` *rises* (95.6 → 96.8 → 99.3 ms) —
  demand throughput falls 3.30 → 3.09 GB/s. This is the direct measurement of the
  thing round 1 only inferred: the extra speculative read is not hidden in idle
  device time, it enters the same saturated device and takes service from the
  demand reads.
- **Every speculative read gets more expensive as the ring widens.** `spec_load_ms`
  1.16 (42.7 reads) → 1.59 (58.7) → 1.74 ms (76.2): the device's latency-vs-depth
  curve, measured from the ring's own reclaimed operations.
- **An adopted read is worth 0.80 ms.** 2.64 MiB at the measured 3.29 GB/s. A
  speculative read costs 1.16 ms of device time at the shipped width. The ring still
  pays because part of that 1.16 ms lands in the window the compute leaves idle
  while the 0.80 ms it saves is exposed wait — that window is what makes the ring
  worth +13.6% — but it is finite, and **42.7 reads a token is where it is full.**

**The exchange rate closes the question.** With the near probe at 0.55 precision and
the alternatives at 0.43 (far horizon) and 0.34 (second rank), the first
non-resident L+1 prediction is the highest-value read available, and the ring
already issues one per layer. There is no leftover device-time budget to spend on a
second read of any shape, so there is no schedule that removes bytes without adding
queue pressure: the bytes can only come from a speculative read, and every
speculative read past the idle window is charged in full against the demand reads
it was meant to help.

## What the failed first implementation measured

The first version of the arm staged `prediction.prefix(1)` at each horizon and
issued **14.6** reads a token (adopted 8.6, demand 353.8 MiB, 4.101 against 4.543 =
**-9.7%**). That is the useful half of the result: the probe's rank-1 prediction is
usually *already cached*, so a rank-1-only ring stages nothing on most layers. The
shipped 42.7 reads/token are the **first non-resident** expert in rank order — the
ring walks down the ranked list and the residency filter decides where it stops.
Any future work on prefetch staging has to keep passing the whole list.

## Round-2 verdict

Do not carry a mixed-horizon ring. It was measured negative and its patch is not in
the tree — the project commits improvements, not losing knobs, so the result lives
here as a record rather than as an opt-in. The decode line for this model on this
machine is closed: the shipped 42.7 speculative reads a token fill the device's
idle window exactly, adoption is precision-limited at 55%, and every route to more
adoption costs more device time than the bytes it removes. What is left is a faster
device or a fundamentally better next-token router — not a scheduling change.
