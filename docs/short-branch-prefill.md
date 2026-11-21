# Short-branch prefill (Rev2 System One, DFlash verify)

Status (2026-11-21): **written in a cloud session, unbuilt and untested.** No
Swift or Metal was compiled; the M1 Max is the only machine that can. The
lint gates that run without a toolchain (force-cast, func-length, sendable,
arch-path) pass. Everything below marked *to measure* is a prediction.

## The problem

Rev2's lmw.15 extraction (M1 Max, Qwen 3.6 35B-A3B 4-bit, ANE off, small QMM
on, `EmoneCLI features`, 825 decision-v7 test questions) fitted wall time
per prefill call:

| call | fixed per call | per token | typical size |
|---|---|---|---|
| question branch | ~390 ms | 1.33 ms | 72 tokens |
| state prefill | ~320 ms | 2.12 ms | 51 tokens |

The target is a 2-option decision at 1k held context under ~150 ms, with
System Two's probe logits byte-identical after rewind.

## What the code says (read, not profiled)

The chunk path already has a no-wait routed stage for short chunks
(`ce71d43`, `RealForwardRunner+ResidentPrefill.swift`): when every routed
expert of every layer is resident, routing is resolved on the GPU by
`moe_resolve_resident_experts`, the row-batched decode kernels run in the
layer's own command buffer, and the only host wait is after the last layer
(21 rows: 310 -> 184 ms when it landed). Otherwise each layer takes the
host-planned tiles: commit and wait for the router, read the picks back, group
them, wait for the shared branch, wait per tile, wait for the tail. That is
four or more host round trips in each of 40 layers.

Two things would keep the lmw.15 calls off the no-wait stage, and either
explains a fixed cost of this size:

1. **The row limit is 64** (`MoE.maxRoutedRows`, the kernels' 512-pointer
   argument struct). A typical question branch is 72 rows, so it took tiles.
2. **Residency is never filled on the CLI path.** The per-row stage needs
   `prepareResidentDecode` to see every layer full; prefill asks with
   `warm: false`, the idle warm (`warmResidentExpertsWhileIdle`) is started
   only by the server, and a `features` run never decodes, which is where the
   inline warm lives. The tile path caches what it touches, but an expert no
   question routes to stays missing, and then every call takes tiles. That
   would also explain why the 51-token state prefill, under the 64-row limit,
   still costs ~320 ms fixed.

Neither is confirmed. The trace below says which one it was in one run.

## What this change adds

`RealForwardRunner.prefillShortBranch(tokens:startPosition:slot:outputMode:config:into:options:)`
(new file `RealForwardRunner+ShortBranchPrefill.swift`), and
`runShortBranchPrefill` for a caller already holding the forward gate. For a
call of at most 128 tokens that fits one chunk it:

- **warms the routed-expert cache to full residency** first, when the cache
  has a slot for every expert and the per-row stage is refused only for
  residency (`ShortBranchPrefillOptions.warmToResidency`, default on). On
  Qwen 3.6 4-bit the first call after a load pays ~14 GB of reads; later
  calls pay nothing.
- **raises the per-row row limit to 128**: the routed rows run as blocks of
  64, each with its own argument, activation and route-log range
  (`ResidentRowBlocks`; `MoE.encodeRoutedRows` and
  `encodeResolveResidentExperts` take byte offsets now). Ordinary chunks keep
  the 64-row limit, so their path and numerics do not change.
- **leaves the last layer unwaited**, so the head's wait is the call's only
  host wait (`residentPrefillDeferLastWait`, checked afterwards by
  `finishResidentPrefillBuffers`).
- **reports what ran**: `ShortBranchPrefillReport.routedStage` is
  `.residentRows` or `.hostPlannedTiles(reason:)`, with warm time, wall time,
  and the layers' GPU busy and elapsed time.

Switches (renamed `EMONE_*` after the rename):

- `EMONE_SHORT_BRANCH=1` sends every `prefillChunked` call of at most 128
  tokens through `runShortBranchPrefill`. Off by default. It exists so the
  uncommitted fork harness (`fastpath`, `features`) can A/B the path on one
  binary without code changes.
- `EMONE_SHORT_BRANCH_TRACE=1` prints one line per call:
  `[short-branch] rows= start= stage= warm_ms= warmed= wall_ms= gpu_busy_ms= gpu_elapsed_ms=`.
- `EMONE_PREFILL_RESIDENT_ROWS=0` still turns the per-row stage off
  everywhere, short branches included; a smaller value caps them too.

Not changed: attention, GDN, the shared expert, the dense projections, the
head, KV / GDN writes, scratch sizing. The per-row routed stage still commits
one command buffer per layer (no wait between them); folding the whole branch
into one buffer would also need the hidden-state readout's per-layer blits to
move into that buffer, which touches the uncommitted readout code, so it is
left until the profile says commits matter.

## What the entry point assumes of its caller

The fork / branch-point code (`makeDecisionBranchPoint` /
`restoreDecisionBranchPoint`) is uncommitted on the Mac, so this was written
against `prefillChunked`'s contract and nothing else:

- The KV cursor of `slot` is at `startPosition`. The call writes KV rows
  `[startPosition, startPosition + count)`, updates GDN state and conv tails in
  place, and advances the cursor, exactly like `prefillChunked`. Saving and
  rewinding are the caller's.
- `prefillShortBranch` takes the forward gate; a caller under the gate uses
  `runShortBranchPrefill`.
- The model's cache can hold every routed expert. For the 125B, whose experts
  stream from SSD, routing has to reach the host to plan the reads, so the
  call takes tiles and says so. A short branch on the 125B needs a different
  design (overlapping the next layer's reads with this layer's compute), not
  this one.
- DFlash verify (~16 rows) fits as is. It reads one row's logits or greedy
  token, as `PrefillOutputMode` allows; a per-row head for verify is not
  added here.

## Numerics

The per-row routed arithmetic is decode's, not the tiles'. A branch that used
to fall back to tiles now produces slightly different rows (the existing
resident-prefill note measured 7 of 24 conversations picking a different
token at top-2 margins <= 0.30, neither path closer to the sequential
reference). Consequences:

- **lmw.15's features came from whichever path ran then, likely tiles.** A
  head fitted on them reads rows from this path at serving time. Check its
  accuracy on rows read this way before trusting the old numbers, or
  re-extract.
- System Two's probe logits after rewind should still match byte for byte:
  the branch only writes rows past the held point and GDN state the rewind
  restores. *To measure.*
- Goldens are unaffected with the switch off: nothing on the ordinary path
  changed. *To measure* (`golden-baseline.sh --check qwen36-4`).

## Runbook on the M1 Max

Not before the local session reports "lmw.15 done", and not before the
rename lands. Then:

1. Bring the branch across the rename: on this branch run
   `python3 rename_to_emone.py`, commit, merge the renamed `main`. Every name
   below becomes `EMONE_*` / `Emone*`.
2. Build release (`swift build -c release`), then
   `swift test --no-parallel --filter ShortBranchPrefillTests`.
3. AGENTS.md preconditions, then
   `pgrep -fl 'EmoneServer|EmoneCLI|EmonePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`
   must be empty.
4. Golden, switch off: `tools/golden-baseline.sh --check qwen36-4` must be
   identical (exit 0).
5. Diagnose the old path first, on the fork harness with the switch off and
   `EMONE_KERNEL_STATS=1`, one 72-token branch at 1k held context: if the
   summed GPU time of the prefill roles is far below wall time, the host waits
   are confirmed. (`EMONE_PREFILL_SPLIT=1` splits the GPU time further by
   stage, but it waits at every split itself, so never compare wall time
   under it.)
6. Same harness with `EMONE_SHORT_BRANCH=1 EMONE_SHORT_BRANCH_TRACE=1`.
   Every call should print `stage=resident_rows`; the first one shows
   `warmed=` > 0. Any `tiles (...)` line names why.
7. Re-run the lmw.13 grid (N = 1k/4k/16k/64k, M = 0/512/2k/8k, 2 options,
   median of 3 after a warmup) with the switch on and off, and the byte-for-byte
   probe check after rewind.
8. Report wall against `gpu_elapsed_ms`. If wall is close to GPU elapsed and
   still above 150 ms at 1k, the remaining cost is kernel time at small t
   (dense projections, GDN chunk scan, the routed rows re-reading each
   expert per row), and the next step is per-stage GPU time with
   `EMONE_PREFILL_SPLIT=1` (GPU numbers only), not more host work.

## If the target is still missed

Estimates, not measurements: the per-token slope (1.33 ms) puts a 72-row
branch at ~100 ms of token cost alone, so 150 ms leaves ~50 ms for everything
fixed. Candidates, in the order the profile would point to them:

- the routed rows read every picked expert once per row; at 72-128 rows a
  GPU-side grouping (sort the picks by expert, one tile per expert, indirect
  dispatch) reads each expert once, which is the tile path without the
  readback;
- one command buffer per branch instead of one per layer;
- the small-chunk QMM (uncommitted on the Mac) for the dense projections.
