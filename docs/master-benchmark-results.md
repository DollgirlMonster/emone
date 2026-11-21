# Master-prompt benchmark results

The ten scenarios in `docs/benchmark-master-prompts.md`, run end to end through
two arms by `benchmark/memory_master.py`: **summary** — memory off, the client
carrying its own 200-word note, which is this benchmark's "no method" baseline —
and **memory auto** — memory on, tools off, the engine writing by consolidation.
`qwen36` 4-bit (Qwen 3.6 35B-A3B) on this machine. **The build is not uniform,
and that is a defect in this table.** The first six worlds' r1 runs used the 5.9
release build (`.build/release` built 05:13). `vantage`, `kitchen`, `cohort`,
`filing` and every repeat then ran after **another session rebuilt the tree at
15:07 with an uncommitted, behaviour-affecting change** to the expert-cache
budget clamp (`min(wanted, physicalMemory / 2)` became `/ 3`, cutting the budget
from 12 GiB to 8 GiB on this 24 GiB machine). The harness's staleness guard
compares the binary against source mtimes, so it cannot see that the binary is no
longer the committed tree. A re-run on one committed build is required before
these numbers are treated as final.

**This supersedes the earlier four-scenario write-up.** That pass measured with
an instrument that could lose the quiz entirely: a session cut off at the token
ceiling scored as a full set of memory misses, and a skipped consolidation was
waited out for ten minutes and charged to memory. Its numbers are not comparable,
and its headline claim — memory 20/20 against the summary's 14/20 on `pigeon` —
did not survive. The summary's "collapse" in that pass was one truncated
session; with the instrument fixed, `pigeon` is a consistent memory loss.

## Method

- **Session shape.** Every session receives the scenario's brief or work
  instruction plus its injected change, and is asked to **open its reply with the
  continuity quiz as a JSON block**, then do the work in at most 1,200 words. The
  quiz is first so the instrument cannot be lost to a long or truncated reply,
  and so the answer comes from retention rather than from work re-derived in the
  same reply.
- **Ceiling.** 6,000 completion tokens (5,200 for `pong`, whose stages emit a
  file of code), overridable with `EMONE_MEMVAL_MAX_TOKENS`.
- **Scoring.** *Foundation* — keys that never change, a no-regression check.
  *Carryable* — keys that change at least once, the signal, as carried/total.
  *Stale* — an answer that is an older value of a key, given after it changed;
  the memory-specific failure. All three are recomputed from the stored answers.
- **Invalid sessions.** A reply with no parseable quiz is excluded from the
  denominators and reported with its reason. Across the whole suite there were
  **zero**: 232 stored sessions, every one scored.
- **Cost.** Model time only: session generation, plus the memory arm's real
  consolidation generations, plus the summary arm's own summary requests. The
  harness's own waits are never counted.
- **Runs.** One run per arm for seven worlds; `photograph`, `pigeon` and
  `contract` were each run three times per arm, so their figures aggregate all
  runs and their spread is shown separately.
- **Conditions.** Each arm logs host load and `dasd` CPU at its start. The first
  six worlds (their r1) ran with `dasd` saturated at 95–105% of a core, which cut
  generation to ~5–14 tokens/s; from `vantage` onward the machine was quiet
  (`dasd` 0.0%, load ~2). Cross-world wall clocks are therefore indicative only.

## Results

Per world — memory / summary. `carry` and `fnd` are percentages; `stale` and
`cost` are totals over that world's runs (three runs for the starred worlds).

| world | carry (mem / sum) | fnd (mem / sum) | stale (mem / sum) | cost min (mem / sum) |
|---|---:|---:|---:|---:|
| photograph * | 51.1 / **85.9** | 67.1 / **80.2** | 36 / 12 | 80.7 / 45.7 |
| pong | 100 / 100 | 100 / 100 | 0 / 0 | 10.6 / 8.3 |
| ledger | 52.4 / **61.9** | **95.2** / 66.7 | 4 / 0 | 41.7 / 15.4 |
| pigeon * | 63.3 / **75.0** | **40.0** / 26.7 | 14 / 1 | 61.7 / 36.1 |
| contract * | 61.1 / **69.4** | 50.0 / **63.9** | 15 / 16 | 71.8 / 47.4 |
| compound_k | **65.0** / 50.0 | **100** / 20.0 | 3 / 1 | 18.8 / 12.6 |
| vantage | 53.6 / **75.0** | n/a | 4 / 0 | 45.5 / 23.8 |
| kitchen | **76.7** / 43.3 | **83.3** / 16.7 | 5 / 6 | 22.2 / 13.8 |
| cohort | **62.9** / 37.1 | **42.9** / 14.3 | 2 / 2 | 25.1 / 15.4 |
| filing | **62.5** / 29.2 | **100** / 66.7 | 4 / 6 | 21.6 / 16.6 |

`*` three runs per arm; the other worlds are one run per arm.

**Pooled over every scored key-instance, all runs:**

| arm | carryable | foundation | stale | model cost |
|---|---:|---:|---:|---:|
| summary | **294/431 = 68.2%** | **251/347 = 72.3%** | **44** | **235.1 min** |
| memory | 256/431 = 59.4% | 234/347 = 67.4% | 87 | 399.6 min |

**Unweighted mean of the per-world percentages** (each world counts once), one
standard deviation across worlds:

| metric | memory | summary | delta |
|---|---:|---:|---:|
| carryable | 64.9% (sd 14.4) | 62.7% (sd 22.6) | **+2.2 pp** |
| foundation | 75.4% (sd 25.7) | 50.6% (sd 31.6) | **+24.8 pp** |

**Repeat spread** (carryable, then foundation, per run):

| world | arm | r1 | r2 | r3 | mean carry | mean fnd |
|---|---|---:|---:|---:|---:|---:|
| photograph | summary | 89 / 93 | 80 / 48 | 89 / 100 | 86.7% | 80.2% |
| photograph | memory | 49 / 73 | 49 / 35 | 56 / 94 | 51.3% | 67.3% |
| pigeon | summary | 70 / 40 | 75 / 20 | 80 / 20 | 75.0% | 26.7% |
| pigeon | memory | 65 / 40 | 60 / 60 | 65 / 20 | 63.3% | 40.0% |
| contract | summary | 75 / 67 | 54 / 58 | 79 / 67 | 69.3% | 64.2% |
| contract | memory | 67 / 33 | 71 / 33 | 46 / 83 | 61.3% | 49.7% |

## What the benchmark says

1. **On changing facts there is no reliable difference.** Pooled, memory is
   8.8 pp behind (59.4% vs 68.2%); unweighted, it is 2.2 pp ahead (64.9% vs
   62.7%). The two averages disagree because `photograph` alone supplies 135 of
   the 431 carryable checks and memory is far worse there (51.1% vs 85.9%). The
   defensible claim is a wash on carryable, not a win in either direction.
2. **Memory's advantage is invariant facts.** Unweighted foundation is 75.4%
   against the summary's 50.6%, +24.8 pp — and the pooled figure only looks level
   because `photograph` supplies 243 of the 347 foundation checks and memory
   loses it there. Memory wins foundation in `compound_k` (100 vs 20), `kitchen`
   (83 vs 17), `filing` (100 vs 67), `ledger` (95 vs 67) and `pigeon` (40 vs
   27), and loses it in `photograph` (67 vs 80) and `contract` (50 vs 64). It is
   a real advantage, not a uniform one: on prose it also drops invariant facts.
3. **The failure it is meant to avoid is worse, not better.** Stale answers are
   87 against the summary's 44 — 2.0× — and memory is the staler arm in five of
   the ten worlds (`photograph`, `ledger`, `pigeon`, `compound_k`, `vantage`),
   ties in two and is less stale in three.
4. **It costs 1.7× the model time** (399.6 vs 235.1 min). The memory arm pays a
   consolidation generation per session; the summary pays one summary request
   per session.
5. **Per world:** memory wins `compound_k`, `kitchen`, `cohort`, `filing`; ties
   `pong`; loses `photograph`, `ledger` (carryable), `pigeon`, `contract`,
   `vantage`. Every win is a world where the summary drops invariant facts and
   never recovers them.
6. **The repeats separate the stable signals from the noisy ones.** Carryable is
   directionally stable where it matters: memory loses `pigeon` in all three
   runs and `photograph` in all three. Foundation is the noisy metric —
   `photograph`'s summary ranges 48–100% and `contract`'s memory 33–83% — so the
   aggregate foundation advantage rests on the repeats, not on one sample.

**Verdict.** Memory is not a general improvement over a client's own summary:
on changing facts it is a wash, and it is 1.7× the cost with twice the stale
answers. What it does reliably is hold *invariant* facts that the baseline
summary drops outright — worth enabling where that failure mode matters, not as
a default. Nothing here justifies removing the code; it justifies leaving it
opt-in. TT-035's structural defect is fixed and pinned (below); the score
figures above still come from the runs that had it, so a claim about the size
of the improvement awaits a re-run.

## Defects this suite exposed

Each was found because the numbers looked wrong, and each is fixed and pinned:

1. **A skipped consolidation did not end the wait** (`ea63d6f`). The harness
   counted only `memory consolidated session=` lines, so a session the engine
   deliberately skipped — "nothing to distil" — looked like work in flight and
   burned the whole 600 s limit. On the discarded first pass that was 1,998 of
   `photograph`/auto's 2,580 s and 2,419 of `ledger`/auto's 3,619 s: harness
   overhead charged to memory. The same bug is fixed in `memory_book.py`,
   `memory_value.py`, `memory_correct.py`, `memory_projects.py` and
   `memory_volume.py`.
2. **A truncated session scored as a full set of misses** (`8af3205`, `94c8f4d`).
   The ceiling sat below what a verbose session needs, so the model was cut off
   before the quiz, and the truncated session was still journalled into memory
   and polluted it. The ceiling is 6,000 now and a session with no parseable quiz
   is invalid rather than zero.
3. **`finish_reason` was read from the wrong object** (`94c8f4d`). It is a
   sibling of `message` inside each choice, not a field of the message, so every
   reply looked as if it had none and truncation could not be told from a missing
   quiz.
4. **TT-035 — consolidation keeps an amended fact beside the original**
   (`d6d2cc3`). On `contract`, sessions distilled the same facts under
   different prefixes (`mga/*`, then `msa/calder/*`; in the fresh run `msa/*`,
   then `agreement/*`), so no supersession could fire and both values stayed
   live. The logs show why, and it was not the router: the two distillations of
   a run overlapped. Session 2's extraction was requested at 04:44:57 and wrote
   at 04:46:54; session 3's was requested at the same minute, built its prompt
   from a store that was still empty, and wrote `agreement/*` at 04:52:42 —
   which is why `consolidation routed …` appears nowhere in any of the three
   runs. `MemoryBackend` now chains distillations per scope, so a later one
   reads memory only after every earlier one in that scope has written. The
   decision the row asked for: do **not** loosen `ServerMemory.reconcile` to
   resolve prefixes by shared segments — a rule that matches on the last
   segment alone routes `characters/tomas/location` onto `setting/location` and
   `characters/ines/knows_photo_content` onto a different character's fact, the
   false positive the router was narrowed to prevent. Where the model renames a
   path's *shape* (`state/msa/notice_days` against
   `msa/commercial/termination_notice_days`) no mechanical rule is safe; the
   side-engine duplicate check is the guard there, and the extraction's
   instruction already requires reuse. Pinned by
   `MemoryV3Tests/aLaterConsolidationReadsMemoryOnlyAfterTheEarlierOneWrote`
   (which reproduces two live addresses without the chain) and
   `.../anAmendedFactUnderANewPrefixLandsOnTheExistingKey`. Its score impact
   stays noisy and unmeasured — `contract`'s memory foundation was 33%, 33%,
   83% across the three runs above, all of them with the defect.

The suite also gained `benchmark/memory_master.py stats` (`81f1219`), which
computes both aggregates from the same scorer the reports use, so every number
above is reproducible rather than hand-derived.

## What this does and does not say

- **Does:** show that the arms separate, that the separation follows the
  invariant-versus-changing fact split rather than the domain, and that the
  memory arm's cost and stale count are consistently higher.
- **Does not:** support a general "memory is better". Three worlds have three
  runs and seven have one; one model, one engine, one machine; `foundation` is
  noisy enough that any single world's figure should be read with its repeat
  spread in mind.
- **Caveats.** *Carryable* conflates a memory failure with a model reasoning
  error (a hallucinated answer counts as a miss); *stale* is the clean
  memory-specific signal. Wall clocks from the first six worlds were measured
  under heavy background load and from `vantage` onward under a quiet machine, so
  cross-world cost is indicative only. The instrument change above is the larger
  caveat: two builds are mixed in one table, so the cost column in particular is
  not a single-engine comparison. The quiz-first instrument is harder than
  the discarded one for both arms, and all of these numbers are its output, not
  the earlier instrument's.

## Reproduce

```bash
EMONE_MEMVAL_MODEL=qwen36 EMONE_MEMVAL_QUANT=4 EMONE_MEMVAL_RUNS=1 \
  benchmark/memval_master.sh
EMONE_MEMVAL_MODEL=qwen36 EMONE_MEMVAL_QUANT=4 \
EMONE_MEMVAL_RUNS=2 EMONE_MEMVAL_FIRST_RUN=2 \
  benchmark/memval_master.sh pigeon contract photograph
python3 benchmark/memory_master.py stats
```

Results live under `.build/benchmark-logs/memory-<scenario>-qwen36-4bit/` and are
not committed; this file is the durable record.
