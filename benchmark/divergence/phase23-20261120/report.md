
## ornith

Phase 2, token flips against base (72 turns in 24 three-turn conversations).

| arm | turns identical | conversations identical | median first-flip step | flips at base gap < 0.5 | median base gap at flip |
|---|---|---|---|---|---|
| base_rep | 72/72 | 24/24 | - | 0/0 | - |
| all_shipped_off | 0/72 | 0/24 | 6 | 42/72 | 0.43 |
| off_resident_rows | 11/72 | 0/24 | 8 | 43/61 | 0.31 |
| on_small_qmm | 12/72 | 0/24 | 16 | 39/60 | 0.25 |
| on_flash_split | 47/72 | 12/24 | 5 | 16/25 | 0.25 |
| cand | 12/72 | 0/24 | 10 | 39/60 | 0.27 |
| kv16 | 4/72 | 0/24 | 6 | 43/68 | 0.35 |
| kv4 | 0/72 | 0/24 | 5 | 28/72 | 0.59 |

Phase 3, task outcomes (30 arithmetic, 15 code).

| arm | math pass | code pass | text identical to base | outcome same as base | same outcome where text moved | base-only / arm-only pass | sign-test p |
|---|---|---|---|---|---|---|---|
| base | 27/30 | 13/15 | 45/45 | 45/45 | 0/0 | 0 / 0 | 1.00 |
| base_rep | 27/30 | 13/15 | 45/45 | 45/45 | 0/0 | 0 / 0 | 1.00 |
| all_shipped_off | 25/30 | 12/15 | 14/45 | 42/45 | 28/31 | 3 / 0 | 0.25 |
| off_resident_rows | 27/30 | 11/15 | 34/45 | 43/45 | 9/11 | 2 / 0 | 0.50 |
| on_small_qmm | 27/30 | 13/15 | 45/45 | 45/45 | 0/0 | 0 / 0 | 1.00 |
| on_flash_split | 27/30 | 13/15 | 45/45 | 45/45 | 0/0 | 0 / 0 | 1.00 |
| cand | 27/30 | 13/15 | 45/45 | 45/45 | 0/0 | 0 / 0 | 1.00 |
| kv16 | 27/30 | 11/15 | 19/45 | 41/45 | 22/26 | 3 / 1 | 0.62 |
| kv4 | 27/30 | 13/15 | 7/45 | 41/45 | 34/38 | 2 / 2 | 1.00 |

## qwen36

Phase 2, token flips against base (72 turns in 24 three-turn conversations).

| arm | turns identical | conversations identical | median first-flip step | flips at base gap < 0.5 | median base gap at flip |
|---|---|---|---|---|---|
| base_rep | 72/72 | 24/24 | - | 0/0 | - |
| all_shipped_off | 7/72 | 0/24 | 16 | 47/65 | 0.19 |
| off_resident_rows | 19/72 | 0/24 | 21 | 45/53 | 0.12 |
| on_small_qmm | 30/72 | 3/24 | 22 | 38/42 | 0.09 |
| on_flash_split | 47/72 | 12/24 | 26 | 20/25 | 0.11 |
| cand | 22/72 | 1/24 | 20 | 43/50 | 0.09 |
| kv16 | 11/72 | 1/24 | 16 | 44/61 | 0.17 |
| kv4 | 1/72 | 0/24 | 9 | 47/71 | 0.28 |

Phase 3, task outcomes (30 arithmetic, 15 code).

| arm | math pass | code pass | text identical to base | outcome same as base | same outcome where text moved | base-only / arm-only pass | sign-test p |
|---|---|---|---|---|---|---|---|
| base | 24/30 | 13/15 | 45/45 | 45/45 | 0/0 | 0 / 0 | 1.00 |
| base_rep | 24/30 | 13/15 | 45/45 | 45/45 | 0/0 | 0 / 0 | 1.00 |
| all_shipped_off | 24/30 | 13/15 | 25/45 | 43/45 | 18/20 | 1 / 1 | 1.00 |
| off_resident_rows | 24/30 | 13/15 | 42/45 | 45/45 | 3/3 | 0 / 0 | 1.00 |
| on_small_qmm | 24/30 | 13/15 | 45/45 | 45/45 | 0/0 | 0 / 0 | 1.00 |
| on_flash_split | 24/30 | 13/15 | 45/45 | 45/45 | 0/0 | 0 / 0 | 1.00 |
| cand | 24/30 | 13/15 | 45/45 | 45/45 | 0/0 | 0 / 0 | 1.00 |
| kv16 | 23/30 | 13/15 | 26/45 | 44/45 | 18/19 | 1 / 0 | 1.00 |
| kv4 | 27/30 | 13/15 | 10/45 | 42/45 | 32/35 | 0 / 3 | 0.25 |

## Reading

- Determinism holds: `base_rep` reproduces `base` token for token on both models (72/72 turns, 45/45 tasks).
- Every arm that changes arithmetic on a path the request takes flips tokens early (median first flip at step 5-26),
  and most flips land where base's top two logits were nearly tied (median gap 0.09-0.43). `kv16`, the *more*
  precise KV cache, diverges as much as `all_shipped_off`: token divergence measures path sensitivity, not quality.
- Outcomes: no arm changes task results measurably (every sign-test p >= 0.25). Where the text moved, the outcome
  held in 18/20 to 34/38 tasks per arm, and the discordant tasks go both ways (`kv4` on Qwen 3.6 is 0 / 3).
  The most one-sided result is Ornith `all_shipped_off`, 3 / 0 in favour of the shipped switches (p = 0.25).
- Gap: phase 3 prompts are about 50 tokens with no prior context, so small QMM (3-31-row follow-ups) and the
  flash key split (deep context) never engage there: their phase-3 text is identical to base by construction. Their
  only evidence here is phase 2 (flash split flips only in the long-document conversations; small QMM in both).
  An outcome test for them needs checkable tasks asked as follow-ups after a long document.
