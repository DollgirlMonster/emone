# One-token-ahead decode (draft design)

Status: design draft, not built. Gate: `EMONE_DECODE_AHEAD=1`, off by default.

## Why

On the M1 Max, warm short-context decode of Qwen 3.6 35B-A3B 4-bit measured
54 tok/s in one probe (18.5 ms/token) with the GPU 79% busy, so about 4 ms
of each token is GPU idle. mlx-lm (0.32.2, `generate_step`) never lets the
GPU wait on the host between tokens: it builds and `async_eval`s token n+1
before it reads token n, because its sampler and its embedding lookup are
both on the GPU.

emone's decode step has two host waits that bracket every token:

1. `produceToken` starts with `runSync` around the embedding lookup
   (`RealForwardRunner+Decode.swift`, "Embed lookup + sqrt(H) fused"),
   because `embed_lookup_int4` takes the token id as a `constant uint&`
   from the host.
2. It ends with `runSync` around the head. On the greedy path
   (`useFusedGreedyHead`, `outputMode == .greedyIfAvailable`) the fused head
   already writes the argmax into `greedyTokenBuf` on the GPU, and the host
   loads it into `lastGreedyToken`.

Between them the GPU sits idle while the host wakes, reads the token, returns
to the caller (detokenize, stop check, stream), re-enters `produce`, runs the
preamble (pin check, `kv.reserve`, `requireQSAExact`), commits and waits for the
embedding, gathers the PLE rows, and encodes layer 0.

**Go/no-go:** the Mac's per-token idle split (inside a token vs between tokens)
decides whether this is worth building. If most of the ~4 ms sits inside a token
(between the per-layer attn/softmax/tail command buffers), this design recovers
little and buffer fusion comes first.

## Design

Scope: the greedy fused-head path, a resident token (`prepareResidentDecode`
true), slot 0, with no diagnostics active (the same guard as the resident path:
`residentPathSettingsRefusal == nil`). Every other case runs today's code.

1. **Embedding from a device buffer.** Add `embed_lookup_int4_dev` (and the
   affine variant) that read `token_id` from `device const uint*`, i.e. from
   `greedyTokenBuf`. Same arithmetic, so the output is byte-identical.
2. **Split the step into encode and finish.**
   - `encodeAheadToken(position:)`: the preamble, embed (committed, not
     waited), the layer loop as today, and the fused head committed without a
     wait. It returns a handle holding the head's command buffer.
   - `finishAheadToken(handle)`: waits on the head, then reads
     `greedyTokenBuf` into `lastGreedyToken`, runs `finishResidentToken`, and
     reaps warm reads.
   - The caller loop becomes: encode n+1, finish n, hand token n to the
     caller, repeat. The host-side bookkeeping for n+1 (`kv.reserve`,
     position) is known in advance: position + 1.
3. **`greedyTokenBuf` double-buffered.** Token n's head writes buffer
   `n % 2`, and token n+1's embedding reads it, while its own head writes
   `(n + 1) % 2`. That way the host read of token n never races the write of
   n+1.
4. **Stopping after a speculative token.** When token n is a stop token (or
   the caller stops), token n+1 is already in flight. It has written the KV
   at position n+1 (harmless: the cursor is not advanced, so the entry is
   overwritten later) and advanced the GDN recurrent state and conv tails
   (not harmless: a later continuation from the cache would start from the
   wrong state). Two options:
   - **(a) Ping-pong GDN state.** Run `gdn_delta_step_decode` and
     `gdn_conv_tail_update` out of place, reading state A and writing state B,
     and alternate per token. Rollback is then flipping the index back. This
     reads and writes the same bytes as today, so it costs nothing, but it
     touches two kernels' bindings and doubles GDN state memory (about
     60 MB on this model).
   - **(b) Branch-point restore.** Snapshot the GDN state before each
     speculative token with the existing decision-branch-point copy. That
     costs a state copy every token, which is the bandwidth (a) avoids.

   Recommended: (a). For the CLI one-shot case neither matters. For the
   server's prompt cache it is required.
5. **Numerics.** The same kernels in the same order on the same data, so
   output must be byte-identical to today's greedy path. The bar:
   golden qwen36-4 identical with the switch on, and probe logits identical.

## Not in scope

Sampling with temperature (the sampler is on the host), MTP/speculative
decode, multi-slot batches, the 125B's streamed path (routing must reach the
host). Each would need its own GPU-side step.

## Measure

Interleaved rotated rounds with cool-downs, warm cache, short/8k/32k, 256
tokens: on against off, plus GPU busy % and the measured inter-token gap.
