# Frontier prefix checkpoints, RAM TTL and prefill progress

Three server additions, ported from a pre-rename (NVMAI-era, base `722af22`)
local branch onto the current tree. None of them changes numerics or a default
that was already there; **none has been built or run on a Mac yet** — see
[Verification](#verification).

## 1. Token-prefix frontier checkpoints

**The miss it fixes.** The multi-prefix prompt cache matches on whole-message
equality. An edit anywhere in an earlier message — a system prompt that carries
a clock, a memory log, a changing tool list — makes every later turn a full
re-prefill, even when the render still shares tens of thousands of leading
tokens with the last one.

**How.** `FrontierTracker` keeps the leading token run recent renders agree on
(the *frontier*). While a prefill runs, `runRawCompletion` splits it at
chunk-aligned positions on that frontier and snapshots the KV+GDN state at each
split (`captureBoundaries` / `onCapture`). After the turn the server stores each
snapshot in `ServerPromptStateStore` as a *frontier entry*: a pure token prefix,
no message shape. On a later request the message-shaped lookup runs first; if a
frontier entry is an exact token prefix of the new render and deeper than what
that lookup found, it is restored and prefill resumes there.

**Why the split is free.** `PrefillChunkPlanner` cuts spans from `startPosition`
in whole chunks, and a split is only taken a whole number of chunks past the
current position. The split calls therefore run exactly the spans one call
would, in order (`RawCompletionCaptureTests` pins this). The extra cost is one
output head per split plus the snapshot copy.

**Where checkpoints go.** A halving ladder, not every chunk: the deepest chunk
boundary on the frontier, then half as many chunks in, a quarter, and so on
down to one chunk. When a later render diverges at M, the frontier shrinks to M
and the deepest boundary under M is the first rung of that request's ladder, so
the right checkpoint exists from the next turn on.

**SSD write volume.** A snapshot is the whole prefix state, not a delta. On a
35B-A3B (Qwen 3.6 / Ornith geometry) with int8 KV that is about 61 MiB of GDN
state plus about 10.6 KB per token:

| prompt | one snapshot | every chunk (4096) | halving ladder |
|---|---|---|---|
| 32K | 359 MiB | 1.7 GB (7) | 0.68 GB (3) |
| 64K | 699 MiB | 6.3 GB (15) | 1.4 GB (4) |
| 128K | 1.35 GiB | 24 GB (31) | 2.9 GB (5) |

The ladder writes this burst only when the frontier is new: the first request,
a reseed, or a divergence. An append-only conversation is served by the
message-shaped cache and adds no frontier writes. Positions already held for the
same prefix are never rewritten.

**Bounds and guards.**

- Active only in `multi-prefix` mode **with** `--prompt-cache-disk`. At the
  default 256 MiB RAM budget, one long prompt's checkpoints would evict the
  message-shaped entries on arrival.
- At most 16 frontier entries, least recently used dropped first. They share
  the store's RAM/SSD byte budgets and eviction with the message-shaped entries.
- One snapshot above 2 GiB (or the store's own ceiling) is skipped.
- Off on the MTP path, and off for `slots > 1`: the cache is already off there.
- Captures only when the runner prefills the render verbatim, never the spliced
  array a message-shaped continuation hands it.
- A render sharing less than one chunk with the frontier (another client, a
  title request) reseeds the frontier rather than pinning it below a chunk. The
  original version shrank monotonically, so one such request switched capture
  off until restart.
- `TINYTITAN_FRONTIER_CACHE=off` disables capture and restore.

**Log lines.** `prompt_cache hit tier=frontier-ram|frontier-ssd cached_tokens=N`,
`prompt_cache frontier_stored tokens=N state_bytes=B`, and
`frontier_restore_failed` / `frontier disk_write_failed` on stderr.

## 2. `--prompt-cache-memory-ttl-seconds <n>`

Releases a RAM snapshot after `n` idle seconds when it also has an SSD copy
(0…86400, default 0 = never). A background timer on the store's disk queue
drives it, so an idle server gives the RAM back without waiting for a request.
A later hit restores from SSD and promotes the entry back to RAM. RAM-only
entries never expire, since dropping them would lose them. The ready banner
shows `prompt_cache_memory_ttl=<n>s` when it can act.

## 3. `GET /v1/prefill-progress`

Returns `{"phase":"idle|prefill|decode","done":N,"total":N,"generation":G}` so
a client can draw a bar through a long prompt that produces no tokens yet.
Process-wide. With more than one sequence in flight it follows the most recently
started one: updates from an older generation are ignored. Any method other
than GET is a 405.

## Verification

Not yet done — the port was made on Linux, where this package cannot build.
On an Apple Silicon Mac:

```bash
swift build -c release
swift test --no-parallel --filter 'FrontierTracker|PrefillProgress|RawCompletionCapture|ServerPromptStateStore|ServerArgument|HTTPServer'
swift test --no-parallel
tools/lint.sh
tools/golden-baseline.sh --check <installed target>
```

The golden check covers the unchanged single-call path. The split path only
runs in the server. To exercise it, start the server with `--prompt-cache-disk`
and send the same long request twice with one character changed near the end
of the system prompt: the second turn should log `tier=frontier-…`.
