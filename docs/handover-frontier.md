# Handover: short-chunk prefill, the divergence overview, and the DFlash drafter

**Paste this into the next session:**

> Continue the emone frontier work from branch `claude/practical-hypatia-o9a22c`.
> Read `AGENTS.md`, then `docs/handover-frontier.md`, then `bd show Emone-9ul.11`.
> The short-chunk prefill path is done and measured: the dense-flash key split is
> committed and on by default for Qwen 3.6 and Ornith 1.5 4-bit (`111963ba`), the
> divergence overview found no switch that moves quality (`benchmark/divergence`),
> and the M1 Max golden baselines for `qwen36-4` and `ornith-4` are captured
> (`0d27242a`). The only target machine is the user's M1 Max (MacBookPro18,2, 64 GB):
> every build check, model run, timing and golden happens there, never anywhere else.
> Next is the DFlash block drafter (Emone-9ul.11); start with the items in
> "Open work" that do not need the Mac, and hand anything that needs a GPU or a
> model back to a session on the M1 Max.

## Where things stand (2026-11-20)

Commits on `claude/practical-hypatia-o9a22c` (not yet pushed when this was written):

- `111963ba` **Prefill: split dense-flash keys for short chunks; divergence overview.**
  A chunk with fewer than 32 (8-row tile, KV head) groups splits its keys into
  1,024-key ranges, one threadgroup each, merged by `qsa_flash_split_merge`
  (`sources/Emone/Metal/Prefill/prefill.metal`,
  `sources/Emone/Kernels/Attention/PrefillAttention.swift`). The switch is
  `ModelProfile.prefillFlashKeySplit`, on for Qwen 3.6 and Ornith 1.5 4-bit;
  `EMONE_PREFILL_FLASH_SPLIT=0/1` overrides it. The same commit makes
  `--score` honour `EMONE_PREFILL_SEQUENTIAL` (it ignored it before), and adds
  the four-phase divergence benchmark.
- `0d27242a` **golden: capture qwen36-4 and ornith-4 on the M1 Max.** The old
  baselines were the upstream developer's M3 captures and never matched here
  (Emone-h7l, closed).

Measurements on the M1 Max, through the server, 17-token continuation after an
~8.5K-token first turn (steady-state prefill per turn):

| model (4-bit) | both off | flash split | split + small QMM |
|---|---|---|---|
| Qwen 3.6 | 277 ms | 189 ms | 167 ms |
| Ornith 1.5 | 222 ms | — | 150 ms |

What the divergence overview established (Emone-8d0, closed; reports in
`benchmark/divergence/phase*/report.md`):

- Greedy output is deterministic run to run (72/72 turns, 45/45 tasks identical).
- No shipped or candidate arithmetic switch moves teacher-forced surprisal
  measurably; only the int4 KV cache does.
- Every arithmetic change flips greedy tokens early, almost always at near-ties
  (median top-1/top-2 gap 0.09–0.43). The *more* precise fp16 KV cache flips as
  many tokens as turning every shipped switch off, so token identity measures
  path sensitivity, not quality.
- Task outcomes don't move: 60 arithmetic + 15 code tasks asked as follow-ups after
  a long document (where small QMM and the split actually run) stay within one
  task of base, sign-test p = 1.00 everywhere.

## Open work

Marked **[Mac]** when it needs the M1 Max (any Swift build or test, since this is a
Metal package; any model run; any timing). Marked **[anywhere]** when it is code,
scripts or text that a session without the Mac can write. A session without the
Mac must say its Swift is unbuilt and untested, and hand it back.

1. **Emone-9ul.11: DFlash block drafter with exact verification.** The bead
   has the literature numbers. The verify it needs, a ~16-row prefill at 8.5K
   context, now costs ~150–167 ms. Break-even at that cost plus a ~10 ms draft is
   about 5.2 committed tokens per verification; the paper reports 7.7 (GSM8K) and
   13 (HumanEval), which would give roughly 1.5x and 2.5x over autoregressive
   decode, if acceptance holds against a 4-bit target (open risk).
   - **[anywhere]** Read the drafter's config (`z-lab/Qwen3.6-35B-A3B-DFlash`;
     sources disagree on 4, 6 or 8 layers) and write the drafter forward pass
     against the existing pieces: hidden-state capture
     (`RealForwardRunner+HiddenCapture`), exact GDN rollback by replay (9ul.3),
     the MTP verify path (9ul.6).
   - **[Mac, operator]** Installing the drafter weights (~0.9 GiB) is an
     operator-requested job; ask before fetching anything.
   - **[Mac]** The two costs left in the verify: routed-expert rows (~97 ms of GPU)
     and the per-token `in_a`/`in_b` GEMVs (~13 ms).
2. **Small QMM on by default: committed, or parked?** Small QMM is on for Qwen 3.6
   and Ornith 1.5 4-bit only in the M1 Max checkout's working tree. The switch
   itself (`prefillSmallChunkQMM`) is part of older uncommitted work, about 900
   lines of kernels plus the System One / LoRA / decision code, and no session is
   working on that. **It is not on the branch,** so a cloud session won't see it.
   The user decides whether to commit that work.
3. **[anywhere] The `qwen38-4` golden target points at the wrong directory.**
   `tools/golden-baseline.sh` expects `models/qwen3.8-flash-next_125B_A6B_4Bit`;
   the install is `models/qwen38flash.gturbo`, so the big Qwen never gets a golden
   check. Fix the script, not the install: renaming the directory breaks its
   receipt. Then **[Mac]** capture the M1 Max baseline.
4. **[Mac] Re-time Emone-lmw.13's 35B arms.** They ran with both switches off:
   the binary predates the default-on change. That's the Rev v2 session's bead.

Other open beads in the decode epic (Emone-9ul): 9ul.1 (re-measure early hits on
the M1 Max), 9ul.2 and 9ul.6 (MTP), 9ul.10 (QSA key sort on Qwen 3.8), 9ul.12
(cache-aware routing on Qwen 3.8).

## Traps that already cost time

- **Goldens are M1 Max only.** A baseline is valid for one (machine, build, model)
  triple. Besides these two and `qwen3.5-0.8b-8bit`, the files in
  `benchmark/golden/` are still M3 captures, and every one of them will mismatch on
  the M1 Max. That's expected, not a regression, and
  re-capturing one needs a deliberate numerics change.
- **Coordinate model runs with the other sessions.** The Rev v2 session (System One,
  Emone-lmw) shares the M1 Max and announces its runs. One model process at a
  time. Run the AGENTS.md `pgrep` guard before every run, and don't rebuild the
  release binary while another session is timing on it.
- **`tools/serve.sh` builds `EmoneServer` first.** After a full `swift build`,
  the product build still runs, about 90 s. A 60-second health wait gives up, and
  `pgrep -f EmoneServer` then matches the `swift-build --product
  EmoneServer` process. Build the product first, or wait longer, and match
  `Release/EmoneServer`.
- **Server stdout is buffered** unless it runs under a pty
  (`script -q /dev/null tools/serve.sh …`). Without one, the `prefill_s` lines
  never reach the log.
- **The CLI's `--prefill-chunk` accepts only 32–32768,** so the 3–31-row small-QMM
  path can't be reached from `--score`. Measure it through the server, as a short
  follow-up turn after a long one.
- **The repo ignores `*.log`.** The divergence phases' server logs (46 MB, the
  per-token logit traces phase 2's tables are built from) weren't committed and
  live only in the M1 Max checkout.
- **Ornith ignores "reply Ready" in phase 4.** It says "Ready." and keeps going
  until the 4-token cap cuts it off. That's the model's own behaviour, the same in
  every arm.
- The user calls Qwen3.8-Flash-Next 125B "the big Qwen", and uses Qwen for most
  things. Ornith is a fallback. Model weights stay on the external drive.

## Beads

All of these are in the M1 Max checkout's beads database. A cloud session sees them
only after `bd dolt push` from the Mac and `bd dolt pull` in the cloud, and neither
has been run.

- Open: Emone-9ul.11 (drafter), the Emone-9ul epic.
- Closed in this stretch: Emone-9ul.11.1 (key split), Emone-8d0
  (divergence), Emone-h7l (golden mismatch).
- Memory: `target-hardware` (`bd memories target-hardware`).
