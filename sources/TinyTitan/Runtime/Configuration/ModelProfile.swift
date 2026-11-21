import Foundation

/// One tuning profile per (model, routed-expert width): everything the
/// runtime chooses for a model that is not architecture -- the expert-cache
/// budget, the prefetch depth, the prefill chunk, the sampling defaults and
/// the kernel switches -- so each install is tuned on its own and a change
/// to one never touches another.
///
/// Resolution order, last wins:
///   1. the family default (`RuntimeConfiguration.decodeTuning`,
///      `GenerationDefaults.forFamily`, the family's chunk);
///   2. the entry for this (modelID, width) in `ModelProfile.table`;
///   3. the environment switches, so an experiment can still override a
///      shipped value without editing the table.
///
/// The table spells every entry out in full, even where it equals the family
/// default, precisely so that editing one row cannot change another. The
/// resolved profile is logged once at load under TINYTITAN_RUNNER_STATS.
public struct ModelProfile: Sendable, Equatable {
    public struct Key: Hashable, Sendable {
        public let modelID: String
        public let weightBits: Int
        public init(_ modelID: String, _ weightBits: Int) {
            self.modelID = modelID
            self.weightBits = weightBits
        }
    }

    public let key: Key
    public let family: ModelFamily
    /// Target bytes for the routed-expert slot cache (before the RAM clamp).
    public var expertCacheBudgetBytes: Int
    /// Speculative expert reads in flight; 0 disables prefetch.
    public var prefetchDepth: Int
    /// Prefill chunk in tokens; nil takes the front end's fallback.
    public var prefillChunkTokens: Int?
    public var sampling: GenerationDefaults.Sampling
    /// One-simdgroup router top-k (both k == 8 and k != 8).
    public var routerTopKSimd: Bool
    /// Simdgroup-per-key sparse decode attention (keep-mask layers only).
    public var attentionSimdPartial: Bool
    /// Fused hyper-connection gates (hyper-connection families only).
    public var hcFused: Bool
    /// GPU-side QSA key selection (sparse-attention families only).
    public var qsaGPUSelect: Bool
    /// Keep the routed-expert cache wired through prefill instead of
    /// unpinning it at prefill start and re-wiring it on the first decode
    /// token. Measured on Qwen3.8 4-bit: the re-wire faults a swapped-out
    /// 12 GiB cache back in, 1.6-4.7 s per request; holding it is a wash on
    /// decode throughput and prefill time. The row carries the measured default
    /// and is the whole decision: the env override that used to force it either
    /// way measured a wash on decode (-0.37%) and is gone, so a row that wires
    /// the cache keeps it wired.
    public var keepExpertCacheWired: Bool
    /// Prefill's remaining scalar GEMMs (hyper-connection gates, QSA indexer
    /// projections) on the MPP tensor-op QMM, and the shared expert as GEMMs
    /// over the chunk. Changes the output by rounding, so it ships only on a
    /// row whose surprisal A/B passed. `TINYTITAN_PREFILL_MPP_WIDE` overrides.
    public var prefillWideMPP: Bool
    /// Streamed routed-expert tiles as grouped MPP GEMMs
    /// (`PrefillRoutedExpertGEMM`). Same rule; `TINYTITAN_PREFILL_ROUTED_MPP`
    /// overrides.
    public var prefillRoutedMPP: Bool
    /// The Gated-DeltaNet prefill recurrence in 8-token WY chunks on the
    /// matrix units (`gdn_delta_chunk_prefill_dk*`). Same rule;
    /// `TINYTITAN_PREFILL_GDN_CHUNK` overrides.
    public var prefillGDNChunked: Bool
    /// QSA prefill attention as masked dense flash tiles
    /// (`attention_prefill_qsa_masked_flash`) instead of the gathered grouped
    /// kernels. `TINYTITAN_PREFILL_QSA_FLASH` overrides.
    public var prefillQSAFlash: Bool
    /// Routed-expert tiles as two tiled QMM launches on the simdgroup matrix
    /// units (`prefill_routed_qmm_gate_up`, `prefill_routed_qmm`) instead of
    /// one MPP GEMM per expert and projection; needs `prefillRoutedMPP`.
    /// `TINYTITAN_PREFILL_ROUTED_QMM` overrides.
    public var prefillRoutedQMM: Bool
    /// Batched dense prefill projections (attention, GDN, the shared expert)
    /// on the tiled simdgroup QMM (`prefill_dense_qmm`) instead of the MPP or
    /// scalar QMM. `TINYTITAN_PREFILL_SG_QMM` overrides.
    public var prefillDenseQMM: Bool
    /// Causal prefill attention with no selection on the flash kernel
    /// (`attention_prefill_qsa_masked_flash`, causal mode) instead of
    /// `attention_prefill_causal_tiled`. `TINYTITAN_PREFILL_DENSE_FLASH`
    /// overrides.
    public var prefillDenseFlash: Bool
    /// QSA flash attention over each row tile's selected 4-key blocks, packed
    /// four to a tile, instead of whole 16-key tiles.
    /// `TINYTITAN_PREFILL_QSA_PACKED` overrides.
    public var prefillQSAPacked: Bool
    /// The prefill router as float logits on the tiled simdgroup QMM, then a
    /// top-k pass (`prefill_router_topk_logits`), instead of the fused scalar
    /// router. `TINYTITAN_PREFILL_ROUTER_LOGITS` overrides.
    public var prefillRouterLogits: Bool
    /// Decode attention past 4,096 keys with no selection on the simdgroup
    /// pass 1, one chunk per 1,024 keys (`Attention.longDecodeEnabled`),
    /// instead of the serial pass 1 at a fixed 16 chunks, whose cost grows
    /// with context. Same rule as the prefill switches;
    /// `TINYTITAN_ATTN_DECODE_LONG` overrides.
    public var decodeLong: Bool

    /// The shipped entries. Measured values, each on its own install.
    public static let table:
        [Key: (
            budget: Int, prefetch: Int, chunk: Int?,
            sampling: GenerationDefaults.Sampling,
            topKSimd: Bool, attnSimd: Bool,
            hcFused: Bool, qsaSelect: Bool, keepWired: Bool,
            wideMPP: Bool, routedMPP: Bool, gdnChunked: Bool, qsaFlash: Bool,
            routedQMM: Bool, denseQMM: Bool, denseFlash: Bool, qsaPacked: Bool,
            routerLogits: Bool, decodeLong: Bool
        )] = [
            // The 35B rows take prefetch depth 1 and hold the expert cache wired
            // through prefill (2026-09-05, on the repaired ring). Measured per
            // install, five interleaved pairs on Qwen 3.6 and three on the other
            // two, 512-token generations:
            //
            //   prefetch depth 1   4-bit            8-bit
            //     Qwen 3.6         +1.8%            +11.3% (+9.8% on a rerun)
            //     Ornith 1.5       +1.8%            +12.6%
            //     AgentWorld       +1.4%            +11.4%
            //
            // Every interval excludes zero. Depth 2 gives only +2.7% where depth 1
            // gives +9.8%, so the ring stays one read deep, as it was at 4.7.
            // The 8-bit gain is the one the clogged ring had been hiding since
            // 2026-09-04: those rows asked for depth 1 and got nothing.
            //
            // Keeping the cache wired is worth +0.9% / +1.6% on 512-token
            // generations and +7.1% / +5.1% on 48-token ones (Qwen 3.6 4/8-bit),
            // where the first decode token no longer faults a swapped-out 10-12
            // GiB cache back in. Measured on Qwen 3.6; the other two installs
            // have identical geometry and cache sizes and take it by inference.
            // Qwen 3.6 35B-A3B, slot A/B 2026-09-05 (essay, interleaved, swap
            // sampled): 4-bit 128 slots 19.50 / 20.45, 160 (10 GiB) 20.55 / 21.04
            // with swap flat, 192 (12 GiB) 21.61 / 21.60 but 1.5 GB pushed to swap
            // on first contact -- 160 is the 24 GB default, 192 is one budget
            // setting away. 8-bit 64 slots 9.85 / 9.72, 96 (12 GiB) 11.13 / 11.19,
            // swap flat: the 8-bit hit rate was 79% at 64 (37% of the token in
            // exposed expert reads) and the trace simulation halves the misses
            // at 96. Prefetch one deep still pays at 8-bit; utility-tier depth 2
            // measured a wash at both widths.
            //
            // Sampling: the Qwen 3.5/3.6 series runs at temperature 0.6 / top-p
            // 0.95. The rows state it rather than borrow `house`, which happens
            // to hold the same numbers today: a later house change must not move
            // a model off its series' settings.
            // 2026-10-03, M1 Max: the prefill kernels measured on Qwen3.8 --
            // chunked GDN, routed experts on the tiled QMM (routed_mpp is its
            // host path), dense projections on the tiled QMM -- plus causal
            // attention on the flash kernel, which this family needs most: its
            // full-attention layers ran on the scalar tiled kernel at ~0.08
            // TFLOPS. 8,529-token prompt: attention 78.1 -> 3.2 s of GPU, prefill
            // 111.3 -> 15.6 s (7.2x). Surprisal at 19,087 tokens of context:
            // Qwen 3.6 +0.003 nats (t +0.85), Ornith 1.5 -0.010 (t -1.40), no
            // measurable change. 4-bit rows only; AgentWorld and the 8-bit
            // rows keep the old kernels until checked.
            // Router logits on the tiled QMM (2026-10-03): 8,529-token prompt,
            // chunk 4096, prefill 14.7 / 12.6 -> 11.9 s, decode unchanged.
            // Surprisal, 512 tokens after 20,197 of context: Qwen 3.6 -0.002
            // nats (t -0.65), Ornith 1.5 -0.005 (t -0.74), no measurable change.
            // 4-bit budget 17 GiB (2026-10-04, 64 GB M1 Max): the whole expert
            // set (256 slots x 40 layers x 1.77 MB = 16.9 GiB), under a third of
            // RAM there. 512 tokens after 2,092: 20.3 / 21.7 tok/s at 160 slots
            // (96.8% hit) -> 23.1 / 24.3 (99.8%). A 24 GB machine is still cut
            // to 8 GiB by `affordableExpertCacheBudget`.
            // Long-context decode attention (2026-10-09, M1 Max): decode no
            // longer falls with context -- live on Qwen 3.6, 33K 24.8 tok/s and
            // 57K 21.3, against ~6.5 and ~4.5 before. Surprisal, 512 tokens
            // after 47,587 of context: Qwen 3.6 +0.002 nats (t +0.57), Ornith
            // 1.5 -0.001 (t -0.10), no measurable change. 4-bit rows only, as
            // above.
            // Prefill chunk 16,384 (2026-10-09, M1 Max, ~16.9K-token prompt, two
            // rounds, output byte-identical): Qwen 3.6 26.4 / 28.0 -> 20.3 / 20.3 s
            // (-25%; 8,192 -21%, 32,768 -22% and noisier); Ornith 1.5 29.0 / 28.9
            // -> 23.4 / 23.3 s (-19%). Decode and RSS unchanged. With the whole
            // expert set resident the saving is per-chunk overhead, not expert
            // reads (expert fetch + tiles 12.0 -> 6.7 s).
            Key("qwen3.6-35b-a3b", 4): (
                17 << 30, 1, 16_384,
                GenerationDefaults.Sampling(
                    temperature: 0.6, topK: GenerationDefaults.topK, topP: 0.95),
                true, true, false, false, true, false, true, true, false, true, true, true, false,
                true, true
            ),
            Key("qwen3.6-35b-a3b", 8): (
                12 << 30, 1, 4_096,
                GenerationDefaults.Sampling(
                    temperature: 0.6, topK: GenerationDefaults.topK, topP: 0.95),
                true, true, false, false, true, false, false, false, false, false, false, false,
                false, false, false
            ),
            // Ornith 1.5, same geometry, measured on its own 2026-09-05: 4-bit
            // 128 slots 19.91 / 20.41 vs 160 20.84 / 21.02; 8-bit 64 slots
            // 8.69 / 9.12 vs 96 10.83 / 10.86, swap flat on every arm.
            // 4-bit: 17 GiB, the whole expert set, as Qwen 3.6 (same geometry).
            Key("ornith-1.5-35b-a3b", 4): (
                17 << 30, 1, 16_384, GenerationDefaults.house,
                true, true, false, false, true, false, true, true, false, true, true, true, false,
                true, true
            ),
            Key("ornith-1.5-35b-a3b", 8): (
                12 << 30, 1, 4_096, GenerationDefaults.house,
                true, true, false, false, true, false, false, false, false, false, false, false,
                false, false, false
            ),
            // AgentWorld, measured on its own 2026-09-05: 4-bit 128 slots 20.52 /
            // 20.50 vs 160 21.11 / 20.92; 8-bit 64 slots 9.31 / 9.25 vs 96
            // 11.15 / 11.21, swap flat. Residency and barrier execution both
            // lose on it. AgentWorld is a Qwen 3.6 fine-tune (manifest family
            // qwen36, same geometry; its template is Qwen 3.6's plus an audio
            // branch), so it takes the Qwen 3.6 sampling.
            Key("qwen-agentworld", 4): (
                10 << 30, 1, 4_096,
                GenerationDefaults.Sampling(
                    temperature: 0.6, topK: GenerationDefaults.topK, topP: 0.95),
                true, true, false, false, true, false, false, false, false, false, false, false,
                false, false, false
            ),
            Key("qwen-agentworld", 8): (
                12 << 30, 1, 4_096,
                GenerationDefaults.Sampling(
                    temperature: 0.6, topK: GenerationDefaults.topK, topP: 0.95),
                true, true, false, false, true, false, false, false, false, false, false, false,
                false, false, false
            ),
            // KAT-Coder-V2.5-Dev: a Qwen3.6-35B-A3B fine-tune with the same
            // geometry, so the cache budget, prefetch depth and wired cache are
            // taken from the rows above **by inference, not measured on this
            // install** -- the same inheritance the AgentWorld comment records.
            // What is not inherited is the sampling: the checkpoint's own
            // `generation_config.json` specifies temperature 1.0 with top-k 20 and
            // top-p 0.95, so stating 0.6 here (the Qwen 3.6 series setting) would
            // quietly run a coding model at a temperature its authors did not ask
            // for.
            Key("kat-coder-v2.5", 4): (
                10 << 30, 1, 4_096,
                GenerationDefaults.Sampling(
                    temperature: 1.0, topK: GenerationDefaults.topK, topP: 0.95),
                true, true, false, false, true, false, false, false, false, false, false, false,
                false, false, false
            ),
            Key("kat-coder-v2.5", 8): (
                12 << 30, 1, 4_096,
                GenerationDefaults.Sampling(
                    temperature: 1.0, topK: GenerationDefaults.topK, topP: 0.95),
                true, true, false, false, true, false, false, false, false, false, false, false,
                false, false, false
            ),
            // Qwen3.8-Flash-Next: 96 slots (12 GiB) still climbing; its card
            // specifies temperature 1.0 / top-p 0.95. The fused hyper-connection
            // gates and the GPU key select are measured washes and stay off.
            // 4-bit, 2026-09-21: prefetch depth 1, ON. The 2026-09-05 decision to
            // leave the ring off came from 512-token story runs where it lost. On
            // this engine it now wins at both prompt lengths measured: the 7-token
            // prompt 3.993 -> 4.621 tok/s decode (+15.7%), the ~500-token prompt
            // 3.627 -> 4.158 (+14.6%), with the expert hit rate up 4.5-4.7 points,
            // misses down 11-17% and io_hidden up. The ring turns speculative reads
            // into hits *before* demand rather than merely warming pages, and the
            // response is byte-identical (golden re-checked). The cache stays wired
            // through prefill (see keepExpertCacheWired).
            // Sampling is the *thinking* row here; the request's thinking mode
            // selects between it and the instruct row at validation time
            // (`GenerationDefaults.forFamily(_:thinking:)`).
            // Prefill, 2026-09-26 on an M1 Max 64 GB (docs/m1-prefill-spike.md,
            // spikes 5-10): chunk 16,384 and both MPP switches take a
            // 16,931-token prefill from 353-389 s (4,096, switches off) to 156 s.
            // The expert corpus is swept once per chunk, so the chunk count is
            // the cost. Surprisal against the switch-free 8,192 reference over
            // 512 teacher-forced tokens: +0.011 nats, t +0.92 (no measurable
            // change). The front ends cap the chunk to what the context allows
            // (`RuntimeConfiguration.largestPrefillChunk(forContext:)`).
            // 2026-09-30, same machine: chunk 32,768 puts that prompt in one
            // chunk instead of 16,384 + a 547-token tail whose expert sweep
            // cost ~13 s. Three interleaved rounds: prefill 154.7-159.9 s ->
            // 142.4-151.2 s (-7.0%), output byte-identical, decode unchanged
            // (median 4.22 -> 4.26 tok/s), RSS +0.2 GiB, no swap. It fits a
            // context up to 131,072; a larger one is capped back to 16,384.
            // 2026-10-03, same machine: the GDN recurrence in 8-token WY chunks
            // on the matrix units (a port of MLX's fused-chunk kernel). Kernel
            // alone at 16,384 rows: 342 -> 24.8 ms per layer. Two interleaved
            // rounds of the 16,931-token prompt: GDN layers 49.9 -> 39.2 s of
            // GPU, prefill 153.5 / 151.9 -> 148.6 / 141.3 s. Surprisal over 512
            // teacher-forced tokens after 19,087 of context: +0.007 nats, t
            // +0.52 (no measurable change). Measured on this row only; the other
            // GDN families keep the sequential kernel until checked.
            // 2026-10-03, same machine: QSA attention as masked dense flash
            // tiles (8 tokens x a KV head's 12 query heads per threadgroup, key
            // tiles no token kept skipped) instead of the gathered grouped
            // kernel. 16,931-token prompt: attention 44.7 -> 20.5 s of GPU,
            // prefill 132.0 -> 108.4 s. Surprisal, same text and context:
            // -0.007 nats, t -0.55 (no measurable change). This row only.
            // 2026-10-03, same machine: routed-expert tiles as two tiled QMM
            // launches on the simdgroup matrix units (gate+up+activation, then
            // down; ~6.3 / 5.6 TFLOPS) instead of one MPP GEMM per expert and
            // projection. Routed GEMM 25.2 -> 15.5 s of GPU, prefill 103.7 ->
            // 99.6 s (before the layer readahead). Surprisal: +0.005 nats,
            // t +0.42 (no measurable change). This row only.
            // 2026-10-03, same machine: the batched dense projections on the
            // tiled simdgroup QMM (5.55 TFLOPS against the MPP QMM's 4.0 at the
            // GDN shapes). GDN layers 35.5 -> 29.2 s of GPU, prefill 91.2 ->
            // 82.5 s. Surprisal: -0.029 nats, t -2.03 (if anything, less
            // surprised). This row only.
            // Causal attention with no selection (chunks inside the 2,048-key
            // dense window) on the same flash kernel: the masked path with every
            // key kept.
            // QSA attention over each 8-row tile's selected 4-key blocks packed
            // four to a 16-key tile, instead of whole 16-key tiles: ~0.42 of the
            // dense work against ~0.65. Attention tiles -22% / -31% in two
            // interleaved pairs on a noisy machine; surprisal +0.015 nats,
            // t +1.06 (no measurable change).
            // 2026-10-03, same machine: the prefill router as float logits on
            // the tiled QMM plus a top-k pass, instead of the fused scalar
            // router (~0.5 TFLOPS, each lane reading its own expert's row a
            // byte at a time). Router ~82 -> ~5 ms per layer, prefill 50.6 ->
            // 46.9-47.1 s, first 8 tokens unchanged. Surprisal: +0.022 nats,
            // t +1.33 (half logits: +0.009, t +0.69); no measurable change.
            // This row only.
            // 2026-10-01, same machine: 16 GiB (128 slots) against 12 GiB (96).
            // Three interleaved rounds of 256 decoded tokens (--ignore-eos):
            // decode 4.56 / 5.48 / 5.15 -> 5.73 / 5.83 / 5.29 tok/s (mean +11%),
            // expert-read wait 19-23 s -> 18-19 s, output byte-identical, prefill
            // unchanged, peak RSS 17.5 -> 21.5 GiB, no swap. The 24 GiB M3 found
            // a bigger cache slower; that was memory pressure, and
            // `affordableExpertCacheBudget` still caps the cache at a third of
            // RAM, so only a machine of 48 GB or more gets the full 128 slots.
            Key("qwen3.8-flash-next", 4): (
                16 << 30, 1, 32_768,
                GenerationDefaults.qwen38Thinking,
                true, true, false, false, true, true, true, true, true, true, true, true, true,
                true, false
            ),
            // 8-bit: 32 slots (8 GiB) 2.05 / 2.06 tok/s; 40 slots (9.5 GiB) 2.18 /
            // 2.27 with swap falling; 48 (13 GiB) 2.24-2.33 but ~1 GB of swap
            // growth per run on this 24 GB machine. 40 is the no-paging middle.
            // Prefetch depth 1 here is inferred from the 4-bit A/B, not measured:
            // the ring is family-level and width-independent, and this install is
            // not present to A/B. The depth-0 this replaces was inferred the same
            // way from the 2026-09-05 measurement. The 4-bit row's prefill
            // switches are not carried over: the routed GEMMs would run on 8-bit
            // weights no surprisal check has seen.
            Key("qwen3.8-flash-next", 8): (
                Int(9.5 * Double(1 << 30)), 1, 4_096,
                GenerationDefaults.qwen38Thinking,
                true, true, false, false, true, false, false, false, false, false, false, false,
                false, false, false
            ),
            // Qwen3.8-27B: the Qwen3.8 generation's dense model, served by the
            // Qwen 3.5 dense family (same `qwen3_5_text` architecture). Dense,
            // so there is no expert cache or prefetch ring: the budget and depth
            // are the family's fallback values and nothing reads them. Chunk
            // 4,096, the size the dense ANE sidecar is exported at. Sampling is
            // the card's thinking-mode row (1.0 / top-k 20 / 0.95), the same as
            // Flash-Next's; its instruct row (0.7 / 0.80, presence 1.5) is not
            // applied yet, because this family has one row for both modes
            // (docs/next-models-research.md). Nothing here is measured.
            // 2026-10-03, M1 Max: chunked GDN, the dense projections and FFN on
            // the tiled QMM, causal attention on the flash kernel. 8,529-token
            // prompt: dense FFN 374.4 -> 42.4 s of GPU, prefill 644.0 -> 91.9 s
            // (7.0x), GPU 99% busy. Surprisal, 512 tokens after 8,904 of
            // context: +0.002 nats (t +1.14), no measurable change. 4-bit only.
            Key("qwen3.8-27b", 4): (
                RuntimeConfiguration.defaultExpertCacheBudgetBytes, 0, 4_096,
                GenerationDefaults.qwen38Thinking,
                true, true, false, false, false, false, false, true, false, false, true, true,
                false, false, false
            ),
            Key("qwen3.8-27b", 8): (
                RuntimeConfiguration.defaultExpertCacheBudgetBytes, 0, 4_096,
                GenerationDefaults.qwen38Thinking,
                true, true, false, false, false, false, false, false, false, false, false, false,
                false, false, false
            ),
        ]

    /// Environment switches applied last. Read once per process.
    public static let environment = ProcessInfo.processInfo.environment

    public static func resolve(
        modelID: String, family: ModelFamily, weightBits: Int,
        environment env: [String: String] = environment
    ) -> ModelProfile {
        let familyTuning = RuntimeConfiguration.decodeTuning(family: family, weightBits: weightBits)
        var profile = ModelProfile(
            key: Key(tableModelID(modelID), weightBits), family: family,
            expertCacheBudgetBytes: familyTuning.expertCacheBudgetBytes,
            prefetchDepth: familyTuning.prefetchDepth,
            prefillChunkTokens: nil,
            sampling: GenerationDefaults.forFamily(family),
            routerTopKSimd: true, attentionSimdPartial: true,
            hcFused: false, qsaGPUSelect: false,
            keepExpertCacheWired: false,
            prefillWideMPP: false, prefillRoutedMPP: false,
            prefillGDNChunked: false, prefillQSAFlash: false,
            prefillRoutedQMM: false, prefillDenseQMM: false, prefillDenseFlash: false,
            prefillQSAPacked: false, prefillRouterLogits: false, decodeLong: false)
        if let row = table[profile.key] {
            profile.expertCacheBudgetBytes = row.budget
            profile.prefetchDepth = row.prefetch
            profile.prefillChunkTokens = row.chunk
            profile.sampling = row.sampling
            profile.routerTopKSimd = row.topKSimd
            profile.attentionSimdPartial = row.attnSimd
            profile.hcFused = row.hcFused
            profile.qsaGPUSelect = row.qsaSelect
            profile.keepExpertCacheWired = row.keepWired
            profile.prefillWideMPP = row.wideMPP
            profile.prefillRoutedMPP = row.routedMPP
            profile.prefillGDNChunked = row.gdnChunked
            profile.prefillQSAFlash = row.qsaFlash
            profile.prefillRoutedQMM = row.routedQMM
            profile.prefillDenseQMM = row.denseQMM
            profile.prefillDenseFlash = row.denseFlash
            profile.prefillQSAPacked = row.qsaPacked
            profile.prefillRouterLogits = row.routerLogits
            profile.decodeLong = row.decodeLong
        }
        if let v = env["TINYTITAN_ROUTER_TOPK_SIMD"] { profile.routerTopKSimd = v != "0" }
        if let v = env["TINYTITAN_ATTN_SIMD_PARTIAL"] { profile.attentionSimdPartial = v != "0" }
        if let v = env["TINYTITAN_HC_FUSED"] { profile.hcFused = v == "1" }
        if let v = env["TINYTITAN_QSA_GPU_SELECT"] {
            profile.qsaGPUSelect = v == "1" || v == "verify"
        }
        if let v = env["TINYTITAN_PREFILL_MPP_WIDE"] { profile.prefillWideMPP = v == "1" }
        if let v = env["TINYTITAN_PREFILL_ROUTED_MPP"] { profile.prefillRoutedMPP = v == "1" }
        if let v = env["TINYTITAN_PREFILL_GDN_CHUNK"] { profile.prefillGDNChunked = v == "1" }
        if let v = env["TINYTITAN_PREFILL_QSA_FLASH"] { profile.prefillQSAFlash = v == "1" }
        if let v = env["TINYTITAN_PREFILL_ROUTED_QMM"] { profile.prefillRoutedQMM = v == "1" }
        if let v = env["TINYTITAN_PREFILL_SG_QMM"] { profile.prefillDenseQMM = v == "1" }
        if let v = env["TINYTITAN_PREFILL_DENSE_FLASH"] { profile.prefillDenseFlash = v == "1" }
        if let v = env["TINYTITAN_PREFILL_QSA_PACKED"] { profile.prefillQSAPacked = v == "1" }
        if let v = env["TINYTITAN_PREFILL_ROUTER_LOGITS"] { profile.prefillRouterLogits = v == "1" }
        if let v = env["TINYTITAN_ATTN_DECODE_LONG"] { profile.decodeLong = v == "1" }
        if let v = env["TINYTITAN_PREDICTIVE_PREFETCH"] {
            profile.prefetchDepth = v == "1" ? max(1, profile.prefetchDepth) : 0
        }
        // The ring depth itself has no override: every value but 1 measured a
        // loss (docs/qwen38-prefetch-predictor-study.md, Lever 5/8), so the
        // profile row's depth is the only source.
        return profile
    }

    /// The table's name for an install: the manifest id without the width
    /// suffix the installer writes (`qwen3.8-flash-next-4bit`), because the
    /// width is already the key's other half. Without this every installer-made
    /// model missed its row and ran on its family's fallback -- on Qwen3.8 that
    /// dropped the wired expert cache, 1.6-4.7 s per request.
    public static func tableModelID(_ manifestModelID: String) -> String {
        for suffix in ["-4bit", "-8bit", "-6bit"] where manifestModelID.hasSuffix(suffix) {
            return String(manifestModelID.dropLast(suffix.count))
        }
        return manifestModelID
    }

    public static func resolve(identity: ManifestIdentity) -> ModelProfile {
        resolve(modelID: identity.modelID, family: identity.family, weightBits: identity.weightBits)
    }

    /// Whether the table names this install, as opposed to falling back to
    /// its family. Draft-head sidecars and unknown ids fall back.
    public var isTabled: Bool { Self.table[key] != nil }

    /// One line for the log, so a benchmark records what ran.
    public var summary: String {
        "profile model=\(key.modelID) bits=\(key.weightBits) family=\(family.rawValue) "
            + (isTabled ? "tabled" : "family-default") + " "
            + "budget=\(expertCacheBudgetBytes >> 20)MiB prefetch=\(prefetchDepth) "
            + "chunk=\(prefillChunkTokens.map(String.init) ?? "fallback") "
            + "sampling=\(sampling.temperature)/\(sampling.topK)/\(sampling.topP) "
            + "topk_simd=\(routerTopKSimd) attn_simd=\(attentionSimdPartial) "
            + "hc_fused=\(hcFused) qsa_select=\(qsaGPUSelect) keep_wired=\(keepExpertCacheWired) "
            + "mpp_wide=\(prefillWideMPP) routed_mpp=\(prefillRoutedMPP) "
            + "gdn_chunk=\(prefillGDNChunked) qsa_flash=\(prefillQSAFlash) "
            + "routed_qmm=\(prefillRoutedQMM) dense_qmm=\(prefillDenseQMM) "
            + "dense_flash=\(prefillDenseFlash) qsa_packed=\(prefillQSAPacked) "
            + "router_logits=\(prefillRouterLogits) decode_long=\(decodeLong)"
    }
}
