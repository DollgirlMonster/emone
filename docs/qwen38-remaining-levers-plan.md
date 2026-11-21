# Qwen3.8-Flash-Next 4-bit on the 24 GiB M3: what is left, and in what order

Written 2026-09-21, after the tuning round that closed the knob space
(`benchmark/internal-speeds/v2-qwen38-4bit-telemetry.txt`, Levers 1-6) and after
the wait matrix below was run. **Status: closed. No code is proposed by this
document.** Every claim is a measurement from this session, a measurement from
another doc it cites, or a labelled prediction.

The short version: kernel tuning is closed; CPU co-execution and ANE are measured
dead; more cache is closed on measurement and would break the RAM rule; the
page-cache trade and expert-dropping are closed by decision; and the wait
measurement that motivated this plan has been explained far enough to show that
nothing follows from it. Decode on this machine is at its quality-preserving,
RAM-respecting limit with the current architecture.

## 1. Closed — do not re-open without a new mechanism

| avenue | how it was closed |
| --- | --- |
| Attention kernel tuning | `docs/qwen38-decode-20tps-concept.md`: `attn_norm_qkv` runs at 26.2 GB/s against 100 GB/s peak because 48 dependent small GEMVs cannot be fused without changing the dependency chain; three attempts measured and reverted ("do not re-open it as a kernel-tuning problem") |
| CPU expert co-execution | `docs/cpu-coexecution-plan.md`: 8 threads of real dequant work raised GPU-busy 44.9% and cut throughput 22.6%; no split ratio wins |
| ANE offload | Same doc: unreachable (Core ML only, no forced placement, per-token routing not expressible) |
| More expert cache / more RAM | This session, 3 paired rounds: 96 slots gives hit 0.757 -> 0.827 and 29% fewer expert bytes, and is **15% slower** (4.96 -> 4.07 tok/s) with exposed wait rising 81 -> 135 ms. Also excluded by the standing rule that the RAM budget must stay enforced |
| Reading fewer experts (top-8, top-6) | Rejected by decision 2026-09-21: output quality is not negotiable, and every version changes the output (section 3) |
| The page-cache trade (`BOUNDED_IO=0`) | Rejected by decision: it is worth +4-5% decode but makes the footprint undeclared |
| Knob families (prefetch quality, retention, I/O sync/submission/advice, parallel I/O) | Levers 1-6 of the telemetry file: shipped defaults are the measured optimum or inside noise |
| MTP, chunk size, QSA trio, fused head, compression, weight reshaping, command-buffer consolidation | Recorded closed in the respective docs |

## 2. The one number that does not fit, and it points at the wait

The 2026-09-05 concept doc models decode as `bytes / 3.6 GB/s`, giving a 12.3
tok/s ceiling at the shipped geometry, and lists as a falsifier "exposed expert
I/O measuring well under 100 ms/token at 96 slots". Neither holds here:

| shipped 64 slots, prefetch 1 | 96 slots, prefetch 1 |
| ---: | ---: |
| 4.52-5.00 tok/s | 4.07-4.14 tok/s |
| hit 0.757, 315 MiB/token | hit **0.827**, **225 MiB/token** |
| `wait_ms` 79-85 | `wait_ms` **134-139** |
| `io_ms` 86-99 | `io_ms` **68-69** |
| `io_hidden_pct` 13.8-15.5 | `io_hidden_pct` **32-34** |
| `io_host_waits` 9,064 | **7,425** |
| `cache_plan_ms` 2.91 | 4.49 |

Fewer bytes, fewer host waits, a third of the wait hidden — and the token gets
half again as slow, because each wait is longer. `bytes / bandwidth` cannot
explain that. The measured decode split says where the time sits: the largest
decode gap is `moe_phase1_hit -> moe_phase1_miss_fixup_phase2` at **110.2 ms per
token** (against ~130 ms of total decode GPU busy), so essentially all of the
exposed wait is that one hand-off inside the routed-MoE phase.

Three hypotheses, each with the run that separates it (all ~85 s per arm, paired
against the shipped arm):

1. **Resident/ring budget competition.** The prefetch ring's slot count is
   `top-M x ahead` inside the same budget; a larger resident set changes what
   the ring can hold or how often an adopted read is still in flight
   (`io_hidden_pct` tripling while `wait` rises is the signature). Checks:
   96 slots with `PREDICTIVE_PREFETCH=0`; 80 slots; `EMONE_PREFETCH_TRACE=1`
   on both sizes.
2. **Slot-search and lease cost scaling with resident slots.** `cache_plan_ms`
   rose 2.91 -> 4.49, but that is 1.6 ms of a 54 ms regression, so this is at
   most a contributor. Check: 80/128 slots to fit the scaling.
3. **Wired-memory pressure.** 12 GiB wired against 8 GiB. Weakened already: swap
   held 3.3-3.4 GB in both arms and did not move. Check: the same A/B with the
   cache wired off (`KEEP_WIRED=0`), which changes only the wiring.

Order matters: run (1) first, because if the ring is the cause the fix is the
adaptive resident/ring split already proposed as tracker Items 11-12, and if it
is not, the only remaining route is section 3.

### Measured 2026-09-21: the ring is exonerated, wiring explains about half

The matrix ran (2 + 3 paired rounds; every arm output-identical; `s80` is not a
legal slot count — the allowed list has no 80 — so H2 was probed through the
counters instead). Pooled over every sample this session has taken on this
shape:

| arm | n | mean tok/s | vs shipped | wait_ms | read MiB | cache_plan_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shipped (64, wired) | 11 | 4.796 | — | 79-84 | 80,608 | 2.91 |
| 96, wired | 11 | 4.277 | **-10.8%** | 109-138 | 57,449 | 4.62 |
| 96, unwired | 5 | 4.524 | -5.7% | 105-107 | 57,460 | 4.33 |
| 96, prefetch off | 2 | 4.344 | -9.4% | 106-112 | 69,863 | 1.52 |

- **H1 (resident/ring competition) is falsified.** Turning prefetch off at 96
  slots does not recover the regression (-9.4%), and it *raises* read bytes
  57.4 -> 69.9 GiB while cutting `cache_plan_ms` 4.62 -> 1.52 — so the ring is a
  net positive even at the larger cache, and the plan-time growth is ring-driven
  rather than slot-driven.
- **H3 (wiring pressure) is partial.** Not wiring the cache recovers about half
  of the regression (-10.8% -> -5.7%) and leaves the hit rate at 0.827 with 29%
  fewer read bytes.
- **H2 (slot-search scaling) is minor**: 1.7 ms of a ~50 ms regression.

The remaining ~6% is unattributed, and it does not change any decision: the
96-slot variants are slower than shipped *and* break the standing rule that the
declared RAM budget stays enforced, so there is nothing to adopt. **Item 11 (the
adaptive resident/ring split) is therefore not justified by this measurement and
should not be built.** Combined with section 3 being rejected, there is no
remaining quality-preserving, RAM-respecting change that improves decode on this
machine with the current architecture; the shipped configuration is the limit,
and the honest statement of that is the result of this plan.

## 3. Reading fewer experts — REJECTED by decision (2026-09-21)

At the shipped hit rate and 3.6 GB/s, the concept doc's arithmetic puts 20 tok/s
behind a 86.5% hit rate (route A, needs RAM we cannot spend) or a 7.2 GB/s
device (route B, not this machine). The third route is arithmetic that works here
today: dropping the two lowest-weighted of the ten experts cuts expert traffic
20%, six of ten cuts it 40%.

**It is rejected and must not be implemented or re-proposed: output quality is
not negotiable on this project, and every version of this route changes the
output by construction.** That closes the last route with real headroom, so from
here the only remaining work is making the existing byte flow faster — i.e. the
wait items in section 2 — and whatever the kernel record already closed stays
closed. If the wait work does not land, the honest answer is that decode on this
machine is at its quality-preserving limit with the current architecture.

For the record, what it would have needed (so a future decision is informed
rather than re-derived): a routing hook (the expert count comes from
`config.topKExperts`, with no override today) plus weight renormalisation behind
an env flag, and a quality gate using the golden text plus
`benchmark/quant_perplexity_ab.py`. Predicted gain was ~+1.3-1.8 tok/s, to be
falsified by the same measurement that section 2 describes.

## 4. Method (non-negotiable, learned the hard way)

- Interleave arms, pair them within a round, and carry a shipped arm in every
  batch: this machine moved 4.37-5.00 tok/s on the *same* configuration within
  one session, which is wider than most of the effects above.
- Quote counters (`expert_read_mib`, `wait_ms`, `io_hidden_pct`, `cache_plan_ms`,
  gap lines) next to wall clock. Wall clock alone has a ±15% spread here.
- Two paired rounds minimum for a direction, more for a magnitude, and state the
  round-to-round values rather than a mean alone.
- Any arm that changes bytes gets a quality gate before it is called a win.

## 5. Status: exhausted

1. **Done** — the existing logs were mined; the counters are in section 2's table.
2. **Done** — the matrix ran: H1 falsified, H3 partial, H2 minor (section 2).
3. **Not done, and not justified** — the adaptive resident/ring split, because
   the ring is not the cause of the wait.

Section 3 is rejected by decision (quality is not negotiable) and section 1's
avenues are closed by measurement or by the RAM rule. **The plan ends here:
decode on this machine is at its quality-preserving, RAM-respecting limit with
the current architecture.** Any further gain has to come from outside it — a
different model, more RAM, or a faster device, the three routes the concept doc
already names. A future session that finds itself re-deriving any of this should
read section 1 first: the work has been done, and the answers are measurements,
not opinions.
