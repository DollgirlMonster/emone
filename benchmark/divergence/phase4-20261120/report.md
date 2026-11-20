
## ornith

Turn 2 prefilled 19-20 uncached rows against 4205-6387 tokens of context; turn 1 replies other than Ready: 40/75.

| arm | math pass | code pass | answer tokens identical | first flip step (median) | outcome same | same outcome where text moved | base-only / arm-only | sign-test p |
|---|---|---|---|---|---|---|---|---|
| base | 57/60 | 11/15 | 75/75 | - | 75/75 | 0/0 | 0 / 0 | 1.00 |
| on_small_qmm | 56/60 | 11/15 | 30/75 | 28 | 70/75 | 40/45 | 3 / 2 | 1.00 |
| on_flash_split | 56/60 | 11/15 | 23/75 | 30.0 | 72/75 | 49/52 | 2 / 1 | 1.00 |
| cand | 56/60 | 11/15 | 17/75 | 23.5 | 70/75 | 53/58 | 3 / 2 | 1.00 |

## qwen36

Turn 2 prefilled 19-20 uncached rows against 4205-6385 tokens of context; turn 1 replies other than Ready: 6/75.

| arm | math pass | code pass | answer tokens identical | first flip step (median) | outcome same | same outcome where text moved | base-only / arm-only | sign-test p |
|---|---|---|---|---|---|---|---|---|
| base | 51/60 | 15/15 | 75/75 | - | 75/75 | 0/0 | 0 / 0 | 1.00 |
| on_small_qmm | 51/60 | 15/15 | 41/75 | 111.5 | 71/75 | 30/34 | 2 / 2 | 1.00 |
| on_flash_split | 51/60 | 15/15 | 31/75 | 107.5 | 73/75 | 42/44 | 1 / 1 | 1.00 |
| cand | 52/60 | 15/15 | 39/75 | 90.5 | 72/75 | 33/36 | 1 / 2 | 1.00 |

## Reading

- The candidates engage here: answer tokens move in 17-41 of 75 tasks per arm (phase 3: none).
- Outcomes do not move: every arm is within one task of base on math and identical on code; discordant tasks
  are 1-3 each way, all sign-test p = 1.00. Where the answer text moved, the outcome held in 30/34 to 53/58.
- Ornith usually says "Ready." and keeps going; the 4-token cap cuts it, so turn 1 ends in a fragment ("Ready.\n\nLet").
  That fragment is the same for base and small QMM; the flash split, which also runs on turn 1's ~100-row
  prefill, changes it in 9/75 tasks. Both are the arm's own effect, not a harness artefact.
