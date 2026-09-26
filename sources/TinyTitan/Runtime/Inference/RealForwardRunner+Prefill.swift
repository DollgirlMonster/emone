import Foundation
import Metal

/// Chunked prefill: the chunk executor, both attention encoders, the KV cache writes, the routed-MoE tile stage, and the final head.
///
/// Split from RealForwardRunner.swift in the modularity refactor
/// (docs/modularity-refactor.md) as pure code motion: one concern
/// per file, no signature or behavior changes.
extension RealForwardRunner {
    /// Prefill-to-decode transition: forget prefill's LFU counts so decode's
    /// working set can take the slots. Opt-in (TINYTITAN_EXPERT_USE_RESET=1):
    /// measured decode-only on a 3.7k-token prompt, the first 64 decode
    /// tokens hit 68% / 82% / 87% (tokens 0-16 / 16-32 / 32-64) with and
    /// without it, identical to three decimals. The cache is not cold after
    /// prefill -- the "52%" that suggested it was the runner's cumulative
    /// statistic with prefill's tile misses mixed in -- and the leftovers
    /// decode reuses outweigh the ones it has to evict.
    var expertUseResetEnabled: Bool {
        ProcessInfo.processInfo.environment["TINYTITAN_EXPERT_USE_RESET"] == "1"
    }
    func resetExpertUseCountsAfterPrefill() {
        guard expertUseResetEnabled else { return }
        model.resetRoutedExpertUseCounts()
    }

    /// The one-token-at-a-time prefill a hyper-connection family started
    /// on, kept as the oracle the batched path is checked against.
    ///
    /// It runs the verified decode path per token, so it produces the KV
    /// state and logits the batched path must reproduce. It is also
    /// unusably slow -- every token pays a full pass over the routed
    /// experts, where a chunk amortizes them -- so it is not the default.
    private func prefillSequentialHyperConnection(
        tokens: ArraySlice<Int32>, startPosition: Int, slot: Int,
        outputMode: PrefillOutputMode, into logits: MTLBuffer,
        onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        var position = startPosition
        for (offset, token) in tokens.enumerated() {
            try Task.checkCancellation()
            try await produceToken(
                token: token,
                position: position,
                slot: slot,
                into: logits,
                emitHead: offset == tokens.count - 1,
                outputMode: outputMode)
            position += 1
            onProgress(offset + 1)
        }
        return PrefillResult(newPosition: position, seed: .logitsWritten)
    }

    /// Intended to release the slot-cache wiring for prefill, which streams
    /// experts in bulk and, on the ANE path, has to leave Core ML room for
    /// its arenas.
    ///
    /// In practice this is a no-op and has always been: the cache is not
    /// wired until the first decode token, so `slotsPinned` is already
    /// false when prefill asks. Measured with `TINYTITAN_WIRE_TRACE=1` over a
    /// full ANE-prefill request: 40 `mlock` calls at the handover, zero
    /// `munlock` calls anywhere. The shipped behaviour is "wire once, at
    /// the handover", not the release/re-apply cycle `703f35a`'s message
    /// describes.
    ///
    /// Kept because it is correct for any future path that does wire
    /// earlier, and because removing it would silently change that path's
    /// behaviour. It is not load-bearing today.
    ///
    /// Skipped when the profile keeps the cache wired: pinning at allocation
    /// and then releasing here is self-defeating, and cost one wrong
    /// conclusion already. For a row that does not wire it, this call is what
    /// makes the cache pageable through prefill (TT-008).
    private func releasePrefillCacheWiring() {
        if !profile.keepExpertCacheWired {
            model.setExpertCachePinned(false)
        }
    }

    private func validateChunkedPrefill(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        config: PrefillRuntimeConfig,
        slot: Int = 0
    ) throws {
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0 else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill startPosition must be non-negative")
        }
        let kvPosition = kv?.position(slot: slot) ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition) for slot \(slot)"
            )
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)"
            )
        }
    }

    public func prefillChunked(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig,
        into logits: MTLBuffer,
        onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        try await prefillChunked(
            tokens: tokens, startPosition: startPosition, slot: 0,
            outputMode: outputMode, config: config, into: logits,
            onProgress: onProgress)
    }

    /// Slot-aware chunked prefill: the chunk lands in `slot`'s KV and GDN
    /// regions, so every sequence takes the same (golden) prefill path instead
    /// of the numerically different decode-as-prefill fallback.
    public func prefillChunked(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        slot: Int,
        outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig,
        into logits: MTLBuffer,
        onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        // Prefill shares the runner's scratch with decode, so a batched slot
        // must not run a chunk while another slot is decoding. One gate covers
        // both; a prefill holds it for its whole burst.
        try await forwardStepGate.acquire()
        do {
            let result = try await runPrefillChunked(
                tokens: tokens, startPosition: startPosition, slot: slot,
                outputMode: outputMode, config: config, into: logits,
                onProgress: onProgress)
            await forwardStepGate.release()
            return result
        } catch {
            await forwardStepGate.release()
            throw error
        }
    }

    func runPrefillChunked(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        slot: Int = 0,
        outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig,
        into logits: MTLBuffer,
        onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        defer { resetExpertUseCountsAfterPrefill() }
        // The chunked path does not go through `produceToken`, so it needs
        // the sparse-attention gate of its own. The chunk's last query sees
        // the most keys and decides the whole chunk.
        try requireQSADensePrefill(visibleKeys: startPosition + tokens.count)
        // The one-token-at-a-time prefill a hyper-connection family started
        // on, kept as the oracle the batched path is checked against.
        //
        // It runs the verified decode path per token, so it produces the KV
        // state and logits the batched path must reproduce. It is also
        // unusably slow -- every token pays a full pass over the routed
        // experts, where a chunk amortizes them -- so it is not the default.
        if cfg.hyperConnections.enabled && Self.sequentialHyperConnectionPrefill {
            return try await prefillSequentialHyperConnection(
                tokens: tokens, startPosition: startPosition, slot: slot,
                outputMode: outputMode, into: logits, onProgress: onProgress)
        }
        releasePrefillCacheWiring()
        try validateChunkedPrefill(
            tokens: tokens, startPosition: startPosition,
            config: config, slot: slot)
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }

        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(
            tokenCount: tokens.count,
            startPosition: startPosition,
            config: config)
        do {
            for (spanIndex, span) in spans.enumerated() {
                try Task.checkCancellation()
                let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
                let upper = tokens.index(lower, offsetBy: span.tokenCount)
                try await executePrefillChunk(
                    tokens: tokens[lower..<upper],
                    startPosition: span.startPosition,
                    slot: slot,
                    outputMode: outputMode,
                    logits: logits,
                    scratch: scratch,
                    config: config,
                    writeFinalHead: spanIndex == spans.count - 1)
                try Task.checkCancellation()
                onProgress(span.completedCount)
            }
        } catch {
            // Any failure — cancellation, a GPU command-buffer error, an I/O
            // error mid-routed-fetch — may have written partial KV rows and
            // left the chunk state dirty. Reset so the next request does not
            // trip `chunkedRunnerDirty` on a stale in-flight chunk.
            reset()
            throw error
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(
                newPosition: startPosition + tokens.count,
                seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(
            newPosition: startPosition + tokens.count,
            seed: .logitsWritten)
    }

    func ensurePrefillScratch(config: PrefillRuntimeConfig) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg, runtime: config)
        if let scratch = prefillScratch, scratch.layout == layout {
            return scratch
        }
        let scratch = try PrefillChunkScratchBuffers.allocate(
            device: ctx.device, layout: layout, batchedSharedExpert: prefillBatchedSharedExpert)
        prefillScratch = scratch
        return scratch
    }

    /// lint:allow-long the orchestrator for one prefill chunk: scratch setup,
    /// the per-layer dispatch, and the head. Each stage it calls is its own
    /// method; what remains is the sequence, and inlining less of it would
    /// only hide the order the stages must run in.
    func executePrefillChunk(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        slot: Int = 0,
        outputMode: PrefillOutputMode,
        logits: MTLBuffer,
        scratch: PrefillChunkScratchBuffers,
        config: PrefillRuntimeConfig,
        writeFinalHead: Bool,
        preparedHidden: MTLBuffer? = nil,
        snapshotGDNAfterFirstToken: Bool = false,
        useTwoRowProjection: Bool = false,
        pairRoutedMoE: Bool = false
    ) async throws {
        // Layer-major prefill (one band of chunks walked layer by layer, with
        // a residual per chunk) is gone: every call runs the whole stack in
        // order, so the prologue and epilogue are unconditional.
        let runPrologue = true
        let runEpilogue = true
        guard !tokens.isEmpty else { return }
        guard kv != nil else {
            throw PrefillError.chunkedUnsupported("chunked prefill attention requires a KV cache")
        }
        let kvPosition = kv?.position(slot: slot) ?? 0
        // Chunk-major advances the cursor per chunk, so it always equals this
        // chunk's start. Layer-major writes a whole band ahead of the cursor
        // and advances once at the end, so the cursor is at or behind the
        // start. Writes themselves are position-addressed and validateRange
        // only bounds against maxContext, so running ahead is safe; this guard
        // is an invariant, not a mechanism.
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        // KV grows on demand rather than reserving maxContext, so make room for
        // this chunk before anything writes into it.
        try kv?.reserve(tokens: startPosition + tokens.count, slot: slot)
        guard startPosition >= 0, startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)"
            )
        }
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)"
            )
        }
        guard !snapshotGDNAfterFirstToken || tokens.count == 2 else {
            throw PrefillError.chunkedUnsupported(
                "Gated-DeltaNet speculative checkpoint requires two rows")
        }
        if let kv, kv.fp16RingEnabled,
            let ringLayer = (0..<cfg.numLayers).first(where: {
                kv.ringCapacity(layer: $0) > 0
            })
        {
            let requiredCapacity = min(maxContext, cfg.slidingWindow + config.chunkTokens)
            let ringCapacity = kv.ringCapacity(layer: ringLayer)
            guard requiredCapacity <= ringCapacity else {
                throw PrefillError.chunkedUnsupported(
                    "KV ring capacity \(ringCapacity) cannot hold required capacity \(requiredCapacity) for maxContext \(maxContext), slidingWindow \(cfg.slidingWindow), and prefillChunkTokens \(config.chunkTokens)"
                )
            }
        }

        let layerViews = try makeLayerPrefillViews()

        // Reused UInt32 token-ID buffer, sized to the largest chunk seen so
        // far and grown on demand (R23); the prefill hot path never allocates
        // a Metal buffer per chunk.
        let tokenBytes = tokens.count * MemoryLayout<UInt32>.stride
        let tokenBuffer: MTLBuffer
        if let existing = prefillTokenBuffer, existing.length >= tokenBytes {
            tokenBuffer = existing
        } else {
            guard
                let made = ctx.device.makeBuffer(
                    length: tokenBytes,
                    options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            made.label = "prefill.tokenIDs"
            prefillTokenBuffer = made
            tokenBuffer = made
        }
        let tokenPtr = tokenBuffer.contents().assumingMemoryBound(to: UInt32.self)
        for (i, token) in tokens.enumerated() {
            tokenPtr[i] = UInt32(bitPattern: token)
        }
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let embedOutScale =
            cfg.embeddingScaledBySqrtHidden
            ? Float(D).squareRoot()
            : 1.0
        let t = tokens.count
        let emb = try model.embedding()

        if runPrologue {
            prefillChunkState.markDirty(
                startPosition: startPosition,
                tokenCount: tokens.count)
        }
        // The n-gram rows depend only on token ids, so the whole chunk's
        // gather runs before any layer needs it.
        try gatherPLERowsPrefill(tokens: tokens)

        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        // The prologue seeds the residual once per chunk.
        if runPrologue {
            if let preparedHidden {
                // The caller hands over `[t, D]` rows -- an MTP draft's fused
                // hidden. A hyper-connection stack starts every stream from that
                // same vector, so it lands in the narrow staging buffer and is
                // widened exactly the way an embedding would be.
                let target = hyperConnection == nil ? scratch.hidden : scratch.normed
                guard let blit = cb.makeBlitCommandEncoder() else {
                    throw ModelError.residentBufferWrapFailed
                }
                blit.copy(
                    from: preparedHidden,
                    sourceOffset: 0,
                    to: target,
                    destinationOffset: 0,
                    size: t * D * MemoryLayout<Float16>.stride)
                blit.endEncoding()
                if hyperConnection != nil {
                    try requireElementwise().encodeHCExpand(
                        commandBuffer: cb,
                        source: scratch.normed, destination: scratch.hidden,
                        dim: D, streamCount: residualStreamCount, tokens: t)
                }
            } else {
                // A hyper-connection stack starts every stream from the token
                // embedding, so the lookup lands in a one-stream staging buffer
                // and is widened from there. `normed` is free until the first
                // layer's read gate writes it.
                let embedTarget = hyperConnection == nil ? scratch.hidden : scratch.normed
                try prefillEmbed.encode(
                    commandBuffer: cb,
                    table: emb.buffer,
                    tableOffset: Int(emb.offset),
                    scales: emb.buffer,
                    scalesOffset: Int(emb.scaleOffset),
                    biases: emb.buffer,
                    biasesOffset: Int(emb.biasOffset),
                    tokens: tokenBuffer,
                    out: embedTarget,
                    t: UInt32(t),
                    d: UInt32(D),
                    outScale: embedOutScale,
                    vocab: UInt32(cfg.vocabSize))
                if hyperConnection != nil {
                    try requireElementwise().encodeHCExpand(
                        commandBuffer: cb,
                        source: scratch.normed, destination: scratch.hidden,
                        dim: D, streamCount: residualStreamCount, tokens: t)
                }
            }
        }

        // Track A: whether this chunk's full-attention layers run on the ANE.
        // The MTP verify (two-row projection / GDN snapshot), MTP adapter
        // chunks (preparedHidden), non-4096 chunk configs, and prompts beyond
        // the sidecar's history variants all stay on the GPU; continuity is
        // enforced inside eligibleChunk so a fallback mid-prompt sticks for
        // the rest of the request.
        let aneChunk: ANEPrefillAttention? = {
            guard let ane = anePrefill,
                slot == 0,
                !snapshotGDNAfterFirstToken,
                !useTwoRowProjection,
                !pairRoutedMoE,
                preparedHidden == nil,
                ane.eligibleChunk(
                    startPosition: startPosition,
                    tokenCount: tokens.count,
                    configChunkTokens: config.chunkTokens)
            else { return nil }
            return ane
        }()

        let prefillProfile = ProcessInfo.processInfo.environment["TURBO_FIELDFARE_PHASES"] != nil
        var prefillRouteNanos: UInt64 = 0
        var prefillTileNanos: UInt64 = 0
        var prefillTailNanos: UInt64 = 0
        var prefillActiveExperts: UInt64 = 0

        for L in 0..<cfg.numLayers {
            try await runPrefillLayer(
                L, cb: &cb, scratch: scratch, layerViews: layerViews,
                tokens: tokens, startPosition: startPosition, t: t, D: D,
                eps: eps, useTwoRowProjection: useTwoRowProjection,
                snapshotGDNAfterFirstToken: snapshotGDNAfterFirstToken,
                aneChunk: aneChunk, pairRoutedMoE: pairRoutedMoE,
                prefillRouteNanos: &prefillRouteNanos,
                prefillTileNanos: &prefillTileNanos,
                prefillTailNanos: &prefillTailNanos,
                prefillActiveExperts: &prefillActiveExperts,
                slot: slot)
        }

        if prefillProfile {
            // stderr, not stdout: stdout is the generated text, and a benchmark
            // that hashes it to compare arms must not see timings in it.
            let prefillTotal = prefillRouteNanos + prefillTileNanos + prefillTailNanos
            let ms = { (n: UInt64) in String(format: "%.1f", Double(n) / 1e6) }
            let perLayer = Double(prefillActiveExperts) / Double(max(1, cfg.numLayers))
            let lines =
                "[prefill phases over \(t) tokens, \(prefillTotal / 1_000_000) ms total]\n"
                + "  route readback + GPU: \(ms(prefillRouteNanos)) ms\n"
                + "  expert fetch + tiles: \(ms(prefillTileNanos)) ms\n"
                + "  tail + residual:      \(ms(prefillTailNanos)) ms\n"
                + "  active experts/layer: \(String(format: "%.2f", perLayer))"
                + " (topK=\(cfg.topKExperts), max possible \(t * cfg.topKExperts))\n"
            FileHandle.standardError.write(Data(lines.utf8))
        }

        if writeFinalHead, runEpilogue {
            try encodeFinalHead(
                logits: logits, scratch: scratch,
                tokenCount: t, hiddenSize: D, rmsEps: eps,
                outputMode: outputMode)
        }

        if runEpilogue {
            aneChunk?.finishChunk(
                startPosition: startPosition,
                tokenCount: tokens.count)
            kv?.advance(slot: slot, by: tokens.count)
            prefillChunkState.markCommitted()
        }
    }

    /// Per-layer tensor views resolved once before the chunk loop.
    struct LayerPrefillQKVViews {
        let inputNorm: TensorView
        let postAttention: TensorView
        /// The router, or nil for a family with no routed mixture (dense).
        let router: TensorView?
        // Softmax-attention layers only (nil on linear-attention layers).
        let q: TensorView?
        let k: TensorView?
        let v: TensorView?
        let o: TensorView?
        let qNorm: TensorView?
        let kNorm: TensorView?
        // Gated-DeltaNet linear-attention layers only.
        let linQKV: TensorView?
        let linZ: TensorView?
        let linA: TensorView?
        let linB: TensorView?
        let linOut: TensorView?
        let linConv: TensorView?
        let linALog: TensorView?
        let linDtBias: TensorView?
        let linNorm: TensorView?

        /// The view for a weight this path requires, or a thrown error naming it.
        ///
        /// A softmax-attention layer must carry q/k/v/o and the two norms; the
        /// call sites used to force-unwrap them, so a profile or family
        /// mismatch crashed the process instead of naming the missing weight.
        func require(_ view: TensorView?, _ name: String) throws -> TensorView {
            guard let view else {
                throw ModelError.internalInconsistency(
                    detail: "full-attention prefill requires the \(name) weights, "
                        + "which this layer does not carry")
            }
            return view
        }
    }

    /// The simdgroup-matrix QMM, when switched on, takes every projection
    /// family and both widths ahead of the MPP path (see
    /// `prefillSimdgroupQMM`). False when it is off or cannot take the shape.
    func encodeSimdgroupProjection(
        commandBuffer: MTLCommandBuffer, weightBits: Int, weights: TensorView,
        x: MTLBuffer, y: MTLBuffer, rows: Int, columns: Int, tokenCount: Int
    ) throws -> Bool {
        guard let simdgroup = prefillSimdgroupQMMKernel,
            simdgroup.accepts(bits: weightBits, k: columns)
        else { return false }
        try simdgroup.encode(
            commandBuffer: commandBuffer,
            weights: weights.buffer, weightsOffset: Int(weights.offset),
            scales: weights.buffer, scalesOffset: Int(weights.scaleOffset),
            biases: weights.buffer, biasesOffset: Int(weights.biasOffset),
            x: x, y: y, t: tokenCount, n: rows, k: columns, bits: weightBits)
        return true
    }

    /// `weightBits` is the *role's* width, not the attention slot's: the dense
    /// Qwen 3.5 installs keep k/v at 8 bits with q/o at 4, and the int4-only
    /// batched paths below would read an 8-bit tensor as packed nibbles.
    func encodeAffineProjection(
        commandBuffer: MTLCommandBuffer,
        family: PrefillProjectionFamily,
        weightBits: Int,
        weights: TensorView,
        x: MTLBuffer,
        y: MTLBuffer,
        rows: Int,
        columns: Int,
        tokenCount: Int,
        xStrideElements: Int,
        yStrideElements: Int,
        useTwoRowProjection: Bool
    ) throws {
        // A promoted tensor carries no scales or biases, so none of the
        // batched paths below can read it -- they would take the companions
        // from offset zero, which is the file header. Fall straight to the
        // per-row GEMV, which dispatches on dtype.
        //
        // The batching lost here is cheap: every promoted family is among the
        // smallest tensors in the model, which is why they were chosen.
        if weights.dtype == 1 {
            try encodeRoleGEMVPerToken(
                commandBuffer: commandBuffer, projection: weights, weightBits: weightBits,
                x: x, y: y, tokenCount: tokenCount,
                xStrideElements: xStrideElements, yStrideElements: yStrideElements,
                m: UInt32(rows), n: UInt32(columns))
            return
        }
        if tokenCount >= 32, xStrideElements == columns, yStrideElements == rows,
            try encodeSimdgroupProjection(
                commandBuffer: commandBuffer, weightBits: weightBits, weights: weights,
                x: x, y: y, rows: rows, columns: columns, tokenCount: tokenCount)
        {
            return
        }
        if tokenCount >= 32, weightBits == 4,
            family == .q || family == .kv || family == .o,
            let candidate = prefillMPPAffineInt4
        {
            let path = try candidate.encode(
                commandBuffer: commandBuffer,
                weights: weights.buffer,
                weightsOffset: Int(weights.offset),
                scales: weights.buffer,
                scalesOffset: Int(weights.scaleOffset),
                biases: weights.buffer,
                biasesOffset: Int(weights.biasOffset),
                x: x,
                y: y,
                m: tokenCount,
                n: rows,
                k: columns)
            if path == .affineThreadgroupF16 {
                return
            }
        }
        if useTwoRowProjection && tokenCount == 2
            && xStrideElements == columns && yStrideElements == rows
        {
            if weightBits == 4 {
                try int4.encodeTwoRows(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    m: UInt32(rows),
                    n: UInt32(columns))
            } else {
                try requireAffine().encodeTwoRows(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    m: UInt32(rows),
                    n: UInt32(columns))
            }
            return
        }
        if weightBits == model.attentionWeightBits,
            PrefillProjectionDispatchPolicy.selectedDispatch(
                for: family,
                chunkTokens: tokenCount) == .qmm
        {
            try prefillQMM.encode(
                commandBuffer: commandBuffer,
                weights: weights.buffer,
                weightsOffset: Int(weights.offset),
                scales: weights.buffer,
                scalesOffset: Int(weights.scaleOffset),
                biases: weights.buffer,
                biasesOffset: Int(weights.biasOffset),
                x: x,
                y: y,
                t: tokenCount,
                n: rows,
                k: columns)
            return
        }
        try encodeRoleGEMVPerToken(
            commandBuffer: commandBuffer, projection: weights, weightBits: weightBits,
            x: x, y: y, tokenCount: tokenCount,
            xStrideElements: xStrideElements, yStrideElements: yStrideElements,
            m: UInt32(rows), n: UInt32(columns))
    }

    /// The shared-expert block for a chunk: three GEMMs over every token when
    /// `prefillWideMPP` and the GPU allow it, else the per-token decode path.
    func encodePrefillSharedExpertBlock(
        commandBuffer: MTLCommandBuffer,
        projections sharedProj: LayerSharedExpertProjections,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int, hiddenSize D: Int
    ) throws {
        if prefillBatchedSharedExpert,
            try prefillSharedExpert.encodeBlockBatched(
                commandBuffer: commandBuffer,
                x: scratch.routedX, y: scratch.h1,
                gate: sharedProj.gate, up: sharedProj.up, down: sharedProj.down,
                scratchGate: scratch.sharedGateScratch,
                scratchUp: scratch.sharedUpScratch,
                scratchAct: scratch.sharedActScratch,
                queryCount: t, d: D, intermediate: cfg.intermediateSize,
                xStrideElements: D, yStrideElements: D)
        {
            return
        }
        try prefillSharedExpert.encodeBlock(
            commandBuffer: commandBuffer,
            x: scratch.routedX,
            y: scratch.h1,
            gate: sharedProj.gate,
            up: sharedProj.up,
            down: sharedProj.down,
            scratchGate: scratch.sharedGateScratch,
            scratchUp: scratch.sharedUpScratch,
            scratchAct: scratch.sharedActScratch,
            queryCount: t,
            d: D,
            intermediate: cfg.intermediateSize,
            xStrideElements: D,
            yStrideElements: D)
    }

    /// With `prefillSplitTiming`, end the command buffer here and time what it
    /// holds under `role`; the layer continues in a fresh one. A no-op
    /// otherwise, so the default schedule is untouched.
    func prefillSplitPoint(_ cb: inout MTLCommandBuffer, role: String) throws {
        guard Self.prefillSplitTiming else { return }
        cb.commit()
        try waitForCompletion(cb)
        recordKernelGPU(role: role, cb)
        guard let next = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        cb = next
    }

    /// One GEMV per token of the chunk: the fallback for a projection no
    /// batched kernel serves. With `prefillCoalescedRows` the tokens share one
    /// compute encoder instead of opening one each; the kernel, arguments and
    /// grid per token are unchanged, so the result is too.
    func encodeRoleGEMVPerToken(
        commandBuffer: MTLCommandBuffer,
        projection: TensorView,
        weightBits: Int,
        x: MTLBuffer, y: MTLBuffer,
        tokenCount: Int,
        xStrideElements: Int, yStrideElements: Int,
        m: UInt32, n: UInt32
    ) throws {
        let halfBytes = MemoryLayout<Float16>.stride
        if Self.prefillCoalescedRows {
            try encodeRoleGEMVRows(
                commandBuffer: commandBuffer, projection: projection, weightBits: weightBits,
                x: x, y: y,
                run: GEMVRows(
                    xOffset: 0, xRowStride: xStrideElements * halfBytes,
                    yOffset: 0, yRowStride: yStrideElements * halfBytes,
                    count: tokenCount),
                m: m, n: n)
            return
        }
        for row in 0..<tokenCount {
            try encodeRoleGEMV(
                commandBuffer: commandBuffer,
                projection: projection,
                weightBits: weightBits,
                x: x,
                xOffset: row * xStrideElements * halfBytes,
                y: y,
                yOffset: row * yStrideElements * halfBytes,
                m: m,
                n: n)
        }
    }

    func copyPrefillKV(
        commandBuffer: MTLCommandBuffer,
        source: MTLBuffer,
        destination: (buffer: MTLBuffer, offset: Int, stride: Int),
        sourceTokenOffset: Int,
        tokenCount: Int,
        bytesPerToken: Int
    ) throws {
        guard tokenCount > 0 else { return }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.copy(
            from: source,
            sourceOffset: sourceTokenOffset * bytesPerToken,
            to: destination.buffer,
            destinationOffset: destination.offset,
            size: tokenCount * bytesPerToken)
        blit.endEncoding()
    }

    func copyPrefillKVToCache(
        commandBuffer: MTLCommandBuffer,
        kv: KVCacheManager,
        layer: Int,
        startPosition: Int,
        tokenCount: Int,
        slot: Int = 0,
        keySource: MTLBuffer,
        valueSource: MTLBuffer,
        bytesPerToken: Int
    ) throws {
        if kv.precision.isQuantized {
            guard let kvQuantizer else {
                throw ModelError.internalInconsistency(
                    detail: "quantized KV cache has no quantizer")
            }
            let elements = bytesPerToken / MemoryLayout<Float16>.stride
            let capacity = kv.capacity(layer: layer)
            let physicalStart = startPosition % capacity
            let firstSpan = min(tokenCount, capacity - physicalStart)
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: keySource,
                sourceTokenStrideElements: elements,
                destination: kv.keyRangeView(
                    layer: layer, start: startPosition,
                    count: firstSpan, slot: slot),
                tokenCount: firstSpan,
                elementCount: elements)
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: valueSource,
                sourceTokenStrideElements: elements,
                destination: kv.valueRangeView(
                    layer: layer, start: startPosition,
                    count: firstSpan, slot: slot),
                tokenCount: firstSpan,
                elementCount: elements)
            guard firstSpan < tokenCount else { return }
            let secondCount = tokenCount - firstSpan
            let secondStart = startPosition + firstSpan
            let sourceOffset = firstSpan * bytesPerToken
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: keySource,
                sourceOffset: sourceOffset,
                sourceTokenStrideElements: elements,
                destination: kv.keyRangeView(
                    layer: layer, start: secondStart,
                    count: secondCount, slot: slot),
                tokenCount: secondCount,
                elementCount: elements)
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: valueSource,
                sourceOffset: sourceOffset,
                sourceTokenStrideElements: elements,
                destination: kv.valueRangeView(
                    layer: layer, start: secondStart,
                    count: secondCount, slot: slot),
                tokenCount: secondCount,
                elementCount: elements)
            return
        }
        let capacity = kv.capacity(layer: layer)
        let physicalStart = startPosition % capacity
        let firstSpan = min(tokenCount, capacity - physicalStart)
        let keyFirst = kv.kRange(
            layer: layer, start: startPosition, count: firstSpan,
            slot: slot)
        let valueFirst = kv.vRange(
            layer: layer, start: startPosition, count: firstSpan,
            slot: slot)
        try copyPrefillKV(
            commandBuffer: commandBuffer,
            source: keySource,
            destination: keyFirst,
            sourceTokenOffset: 0,
            tokenCount: firstSpan,
            bytesPerToken: bytesPerToken)
        try copyPrefillKV(
            commandBuffer: commandBuffer,
            source: valueSource,
            destination: valueFirst,
            sourceTokenOffset: 0,
            tokenCount: firstSpan,
            bytesPerToken: bytesPerToken)
        guard firstSpan < tokenCount else { return }

        let secondCount = tokenCount - firstSpan
        let secondStart = startPosition + firstSpan
        let keySecond = kv.kRange(
            layer: layer, start: secondStart, count: secondCount,
            slot: slot)
        let valueSecond = kv.vRange(
            layer: layer, start: secondStart, count: secondCount,
            slot: slot)
        try copyPrefillKV(
            commandBuffer: commandBuffer,
            source: keySource,
            destination: keySecond,
            sourceTokenOffset: firstSpan,
            tokenCount: secondCount,
            bytesPerToken: bytesPerToken)
        try copyPrefillKV(
            commandBuffer: commandBuffer,
            source: valueSource,
            destination: valueSecond,
            sourceTokenOffset: firstSpan,
            tokenCount: secondCount,
            bytesPerToken: bytesPerToken)
    }

    func encodeQuantizedKV(
        commandBuffer: MTLCommandBuffer,
        kv: KVCacheManager,
        layer: Int,
        position: Int,
        slot: Int = 0,
        keySource: MTLBuffer,
        valueSource: MTLBuffer,
        elementCount: Int
    ) throws {
        guard let kvQuantizer else {
            throw ModelError.internalInconsistency(
                detail: "quantized KV cache has no quantizer")
        }
        try kvQuantizer.encode(
            commandBuffer: commandBuffer,
            source: keySource,
            sourceTokenStrideElements: elementCount,
            destination: kv.keyRangeView(
                layer: layer, start: position, count: 1,
                slot: slot),
            tokenCount: 1,
            elementCount: elementCount)
        try kvQuantizer.encode(
            commandBuffer: commandBuffer,
            source: valueSource,
            sourceTokenStrideElements: elementCount,
            destination: kv.valueRangeView(
                layer: layer, start: position, count: 1,
                slot: slot),
            tokenCount: 1,
            elementCount: elementCount)
    }

    /// Resolve every layer's tensor views once, before the chunk loop.
    func makeLayerPrefillViews() throws -> [LayerPrefillQKVViews] {
        try (0..<cfg.numLayers).map { L in
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let isLinear = cfg.layerIsLinear(L)
            return LayerPrefillQKVViews(
                inputNorm: try model.inputNorm(layer: L),
                postAttention: try model.postAttnNorm(layer: L),
                router: cfg.numExperts == 0 ? nil : try model.router(layer: L),
                q: isLinear ? nil : try model.qProj(layer: L),
                k: isLinear ? nil : try model.kProj(layer: L),
                v: isLinear
                    ? nil
                    : ((isFull && cfg.attentionKEqV)
                        ? (try model.kProj(layer: L))
                        : (try model.vProj(layer: L))),
                o: isLinear ? nil : try model.oProj(layer: L),
                qNorm: isLinear ? nil : try model.qNorm(layer: L),
                kNorm: isLinear ? nil : try model.kNorm(layer: L),
                linQKV: isLinear ? try model.linearInProjQKV(layer: L) : nil,
                linZ: isLinear ? try model.linearInProjZ(layer: L) : nil,
                linA: isLinear ? try model.linearInProjA(layer: L) : nil,
                linB: isLinear ? try model.linearInProjB(layer: L) : nil,
                linOut: isLinear ? try model.linearOutProj(layer: L) : nil,
                linConv: isLinear ? try model.linearConv1d(layer: L) : nil,
                linALog: isLinear ? try model.linearALog(layer: L) : nil,
                linDtBias: isLinear ? try model.linearDtBias(layer: L) : nil,
                linNorm: isLinear ? try model.linearNorm(layer: L) : nil)
        }
    }

    /// Final norm and lm_head for the last chunk, writing logits or a fused
    /// greedy token depending on the output mode.
    func encodeFinalHead(
        logits: MTLBuffer,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        hiddenSize D: Int,
        rmsEps eps: Float,
        outputMode: PrefillOutputMode
    ) throws {
        let finalNorm = try model.finalNorm()
        let lm = try model.lmHead()
        guard let finalCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        if let hc = hyperConnection {
            // The stack ends by collapsing the streams through the
            // model-level mixer, and only the last row feeds the head, so the
            // one-row decode gate serves here. `normed` is free once the last
            // layer has run.
            let rowBytes = D * residualStreamCount * MemoryLayout<Float16>.stride
            try hc.encodeRead(
                commandBuffer: finalCB,
                streamsBuffer: scratch.hidden,
                streamsOffset: (t - 1) * rowBytes,
                hcNorm: finalNorm.buffer,
                hcNormOffset: Int(finalNorm.offset),
                down: gateWeightsPublic(try model.hcMixerDown()),
                up: gateWeightsPublic(try model.hcMixerUp()),
                blockInput: scratch.normed, eps: eps)
            try encodeHeadGEMV(
                commandBuffer: finalCB,
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                x: scratch.normed, y: logits,
                m: UInt32(cfg.vocabSize), n: UInt32(D))
            finalCB.commit()
            try waitForCompletion(finalCB)
            recordKernelGPU(role: "prefill_head", finalCB)
            if activationDumpDirectory != nil {
                dumpActivation(
                    "prefill_logits", logits, count: cfg.vocabSize,
                    position: 0)
            }
            return
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            try fusionHead.encodeGreedyDecode(
                commandBuffer: finalCB,
                hidden: scratch.hidden,
                hiddenOffset: (t - 1) * D * MemoryLayout<Float16>.stride,
                normWeight: finalNorm.buffer,
                normOffset: Int(finalNorm.offset),
                weights: lm.buffer,
                weightsOffset: Int(lm.offset),
                scales: lm.buffer,
                scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer,
                biasesOffset: Int(lm.biasOffset),
                outToken: greedyTokenBuf,
                d: UInt32(D),
                vocab: UInt32(cfg.vocabSize),
                rmsEps: eps)
        } else {
            try prefillFinalRowHead.encodeLogits(
                commandBuffer: finalCB,
                hiddenBlock: scratch.hidden,
                row: t - 1,
                rowStrideElements: D,
                normWeight: finalNorm.buffer,
                normWeightOffset: Int(finalNorm.offset),
                weights: lm.buffer,
                weightsOffset: Int(lm.offset),
                scales: lm.buffer,
                scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer,
                biasesOffset: Int(lm.biasOffset),
                logits: logits,
                d: UInt32(D),
                vocab: UInt32(cfg.vocabSize),
                rmsEps: eps)
        }
        finalCB.commit()
        try waitForCompletion(finalCB)
        if activationDumpDirectory != nil {
            // The head has run and is complete, so these are the numbers the
            // reference's top-k printout compares against.
            dumpActivation(
                "prefill_logits", logits, count: cfg.vocabSize,
                position: 0)
            dumpActivationPrivate(
                "final_normed", scratch.normed,
                count: D, position: 0)
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
        }
    }

    /// The dense family's prefill FFN.
    ///
    /// The shared-expert block *is* this family's feed-forward network, so this
    /// is that block plus the residual add the MoE tail performs for the shared
    /// branch -- the same arithmetic, without a mixture to route, fetch or
    /// reduce. `sharedExpertGated` is false for the family and its schema has no
    /// gate tensor, so the scalar-gate step the MoE path applies has nothing to
    /// read here.
    func encodeDenseFFNPrefill(
        cb: inout MTLCommandBuffer,
        layer L: Int,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        hiddenSize D: Int
    ) throws {
        // Commit and drain the incoming buffer first: it carries this layer's
        // attention and its MLP entry norm. The MoE stage does the same at the
        // top of its pipeline (it needs the routing readback), and without it
        // the layer's attention work is never submitted -- the buffers below
        // stay zero and the stack produces a zero hidden state, which reads as
        // confident nonsense rather than as an error.
        cb.commit()
        try waitForCompletion(cb)
        guard let sharedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        let sharedProj = sharedExpertProjections[L]
        try encodePrefillSharedExpertBlock(
            commandBuffer: sharedCB, projections: sharedProj,
            scratch: scratch, tokenCount: t, hiddenSize: D)
        sharedCB.commit()
        try waitForCompletion(sharedCB)
        recordKernelGPU(role: "prefill_shared_expert", sharedCB)

        guard let tailCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try requireElementwise().encodeResidualAdd(
            commandBuffer: tailCB,
            hidden: scratch.hidden,
            delta: scratch.h1,
            count: t * D)
        tailCB.commit()
        try waitForCompletion(tailCB)
        recordKernelGPU(role: "prefill_dense_ffn_tail", tailCB)
        if L + 1 < cfg.numLayers {
            guard let nextCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            cb = nextCB
        }
    }

    /// Router, routed-expert fetch and the MoE tail for one prefill layer.
    ///
    /// lint:allow-long one layer's MoE stage is a single ordered pipeline:
    /// route readback, expert streaming, tiled phase-1/phase-2, then the
    /// residual tail. It rebinds the command buffer partway through (the
    /// resident buffer wraps between layers), so the stages share mutable
    /// encoding state and cannot be separated without threading it back out.
    func encodeRoutedMoEPrefill(
        cb: inout MTLCommandBuffer,
        layer L: Int,
        views: LayerPrefillQKVViews,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        startPosition: Int,
        hiddenSize D: Int,
        layerStart prefillLayerStart: UInt64,
        routeNanos prefillRouteNanos: inout UInt64,
        tileNanos prefillTileNanos: inout UInt64,
        tailNanos prefillTailNanos: inout UInt64,
        activeExperts prefillActiveExperts: inout UInt64
    ) async throws {
        // A dense model has no routed mixture: its FFN is the shared-expert
        // block, so the router, the readback and the routed tiles do not exist
        // for it. Taken here, at the top, so none of them is encoded.
        if cfg.numExperts == 0 {
            try encodeDenseFFNPrefill(
                cb: &cb, layer: L, scratch: scratch,
                tokenCount: t, hiddenSize: D)
            return
        }
        guard let routerView = views.router else {
            throw ModelError.internalInconsistency(
                detail: "routed prefill stage reached for a model with no router")
        }
        var prefillRouteEnd = prefillLayerStart
        var prefillTileEnd = prefillLayerStart
        let perExpertScale: (buffer: any MTLBuffer, offset: Int) =
            (try requireOnesPerExpertScale(), 0)
        try prefillRouter.encodeBlock(
            commandBuffer: cb,
            weights: routerView.buffer,
            weightsOffset: Int(routerView.offset),
            scales: routerView.buffer,
            scalesOffset: Int(routerView.scaleOffset),
            biases: routerView.buffer,
            biasesOffset: Int(routerView.biasOffset),
            hidden: scratch.routedX,
            effectiveScale: effectiveScaleBuffers[L],
            perExpertScale: perExpertScale.buffer,
            perExpertScaleOffset: perExpertScale.offset,
            outIndices: scratch.routeIDs,
            outWeights: scratch.routeWeights,
            queryCount: UInt32(t),
            numExperts: UInt32(cfg.numExperts),
            d: UInt32(D),
            topK: UInt32(cfg.topKExperts),
            hiddenStrideElements: UInt32(D))

        cb.commit()
        try waitForCompletion(cb)
        // Prefill had no occupancy instrumentation at all: these buffers
        // never reached recordKernelGPU, so TINYTITAN_KERNEL_STATS reported
        // only the decode tokens of a request and prefill looked idle.
        // Split by layer kind: the Track A go/no-go needs to know how
        // the attention-block time divides between full-attention
        // layers (whole block is ANE-expressible) and Gated-DeltaNet
        // layers (only the dense projections are; the recurrent scan
        // is not representable in a static Core ML graph).
        recordKernelGPU(
            role: cfg.layerIsLinear(L)
                ? "prefill_gdn_router"
                : "prefill_attn_router", cb)

        let routeCount = t * cfg.topKExperts
        let idPtr = scratch.routeIDs.contents()
            .bindMemory(to: UInt32.self, capacity: routeCount)
        let weightPtr = scratch.routeWeights.contents()
            .bindMemory(to: Float16.self, capacity: routeCount)
        // Reused per-chunk host scratch (R38): cleared in place so
        // the routed-tile planner never allocates per chunk.
        routeIDScratch.removeAll(keepingCapacity: true)
        routeWeightScratch.removeAll(keepingCapacity: true)
        routeIDScratch.reserveCapacity(routeCount)
        routeWeightScratch.reserveCapacity(routeCount)
        for i in 0..<routeCount {
            routeIDScratch.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
            routeWeightScratch.append(weightPtr[i])
        }
        if routeTraceFD >= 0 {
            // Same line format as the decode trace, prefixed with
            // the token's absolute position, so tail-window routing
            // can be compared with the response's.
            let k = cfg.topKExperts
            for row in 0..<t {
                recordRouteTrace(
                    layer: L, position: startPosition + row,
                    experts: routeIDScratch[row * k..<(row + 1) * k].map { Int($0) })
            }
        }
        let pairs = PrefillRouter.makeTokenExpertPairs(
            indices: routeIDScratch,
            weights: routeWeightScratch,
            queryCount: t,
            topK: cfg.topKExperts)
        let schedulerConfig: PrefillRoutedTileSchedulerConfig
        let routeTileExpertCount: Int
        if let slotCount = model.routedExpertCacheSlotCount() {
            guard
                let fitted = Self.prefillRoutedTileSchedulerConfig.fitting(
                    slotCount: slotCount)
            else {
                throw PrefillError.chunkedUnsupported(
                    "prefill routed tiles cannot fit the \(slotCount)-slot expert cache")
            }
            schedulerConfig = fitted
            routeTileExpertCount = fitted.tileExperts
        } else {
            schedulerConfig = Self.prefillRoutedTileSchedulerConfig
            routeTileExpertCount = schedulerConfig.tileExperts
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: t,
            topK: cfg.topKExperts,
            numExperts: cfg.numExperts,
            tileExpertCount: routeTileExpertCount,
            expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))
        prefillRouteEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        prefillRouteNanos &+= prefillRouteEnd - prefillLayerStart
        // One group per *distinct* expert this chunk touches. For a
        // 1-token chunk this is topK; for a speculative 2-token verify
        // it is the union of the two tokens' routes, which is what
        // decides whether the extra row rides along on weights the
        // first row already pulled in or pays for its own.
        prefillActiveExperts &+= UInt64(routes.groups.count)

        guard let sharedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        let sharedProj = sharedExpertProjections[L]
        try encodePrefillSharedExpertBlock(
            commandBuffer: sharedCB, projections: sharedProj,
            scratch: scratch, tokenCount: t, hiddenSize: D)
        if cfg.sharedExpertGated {
            // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX),
            // per chunk row.
            let gateView = try requireTensorView(sharedProj.scalarGate, "shared-expert scalar gate")
            let halfBytes = MemoryLayout<Float16>.stride
            if Self.prefillCoalescedRows {
                // The same per-row gate GEMV and per-element sigmoid, in two
                // encoders for the whole chunk instead of two per token.
                try encodeScalarGateRows(
                    commandBuffer: sharedCB,
                    view: gateView,
                    x: scratch.routedX,
                    y: scratch.sharedScalarGate,
                    run: GEMVRows(
                        xOffset: 0, xRowStride: D * halfBytes,
                        yOffset: 0, yRowStride: halfBytes,
                        count: t),
                    n: UInt32(D))
                try requireElementwise().encodeSigmoidRowsMul(
                    commandBuffer: sharedCB,
                    y: scratch.h1,
                    gate: scratch.sharedScalarGate,
                    width: D, rows: t)
            } else {
                for row in 0..<t {
                    try encodeScalarGate(
                        commandBuffer: sharedCB,
                        view: gateView,
                        x: scratch.routedX,
                        xOffset: row * D * halfBytes,
                        y: scratch.sharedScalarGate,
                        yOffset: row * halfBytes,
                        n: UInt32(D))
                }
                for row in 0..<t {
                    try requireElementwise().encodeSigmoidScalarMul(
                        commandBuffer: sharedCB,
                        y: scratch.h1,
                        yOffset: row * D * halfBytes,
                        gate: scratch.sharedScalarGate,
                        gateOffset: row * halfBytes,
                        count: D)
                }
            }
        }
        // Committed, not awaited: the routed tiles' expert reads can start
        // while the GPU runs the shared expert. Command buffers on one queue
        // run in commit order -- the tiles below already rely on that for
        // their own shared scratch -- and nothing before the tail reads h1, so
        // the wait moves to just before the tail. Same kernels, same order.
        sharedCB.commit()

        let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
            device: ctx.device,
            routes: routes)
        let routedOffsets = try model.routedExpertOffsets(layer: L)
        struct PendingPrefillTile {
            let tileIndex: Int
            let commandBuffer: MTLCommandBuffer
            let fetch: PrefillStreamedTileFetchResult
            let argumentBuffer: PrefillStreamedTileArgumentBuffer
            /// Ran as grouped MPP GEMMs rather than the tile kernels; timed
            /// under its own role so a spike shows how many tiles took it.
            let groupedGEMM: Bool
        }
        var pendingTiles: [PendingPrefillTile] = []
        var tileLifetime = PrefillStreamedTileSlotLifetime()
        // `withExtendedLifetime` below takes a non-throwing closure,
        // so the wait error is captured here and rethrown after the
        // fetched blobs are released.
        var pendingTileError: Error?
        var tailError: Error?
        func drainOldestPendingTile() throws {
            guard !pendingTiles.isEmpty else { return }
            let pending = pendingTiles.removeFirst()
            withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                do {
                    try waitForCompletion(pending.commandBuffer)
                    recordKernelGPU(
                        role: pending.groupedGEMM ? "prefill_routed_gemm" : "prefill_routed_tile",
                        pending.commandBuffer)
                } catch {
                    // Rethrown after the fetched blobs are released.
                    pendingTileError = error
                }
            }
            if let error = pendingTileError {
                pendingTileError = nil
                throw error
            }
            if !pending.fetch.plannedMissSlots.isEmpty {
                try tileLifetime.complete(tileIndex: pending.tileIndex)
            }
        }

        let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
        for (tileIndex, tile) in routes.tiles.enumerated() {
            let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                forTile: tileIndex,
                routes: routes)
            var plannedFetch: RoutedExpertFetchPlan?
            if !pendingTiles.isEmpty {
                let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                if !pendingAssignedSlots.isEmpty {
                    let pendingSlots = Set(pendingAssignedSlots)
                    let plan = try model.planRoutedExpertsIfPossible(
                        layer: L,
                        experts: expertIDs,
                        avoidingSlots: pendingSlots)
                    let decision = routedTileScheduler.decide(
                        PrefillRoutedTileSchedulerInput(
                            hasPendingTile: true,
                            pendingDepth: pendingTiles.count,
                            pendingAssignedSlots: pendingAssignedSlots,
                            avoidingSlotPlanAvailable: plan != nil))
                    switch decision {
                    case .prefetchNext:
                        guard let plan else {
                            throw ModelError.indexCorrupt(
                                detail: "routed tile scheduler requested missing plan")
                        }
                        plannedFetch = plan
                    case .drainBeforeIssue:
                        try drainOldestPendingTile()
                    case .issueWithoutPending:
                        throw ModelError.indexCorrupt(
                            detail: "routed tile scheduler ignored pending tile")
                    }
                } else {
                    let decision = routedTileScheduler.decide(
                        PrefillRoutedTileSchedulerInput(
                            hasPendingTile: true,
                            pendingDepth: pendingTiles.count,
                            pendingAssignedSlots: [],
                            avoidingSlotPlanAvailable: false))
                    switch decision {
                    case .drainBeforeIssue:
                        try drainOldestPendingTile()
                    case .issueWithoutPending, .prefetchNext:
                        throw ModelError.indexCorrupt(
                            detail: "routed tile scheduler failed to drain empty-slot pending tile")
                    }
                }
            } else {
                let decision = routedTileScheduler.decide(
                    PrefillRoutedTileSchedulerInput(
                        hasPendingTile: false,
                        pendingAssignedSlots: [],
                        avoidingSlotPlanAvailable: false))
                switch decision {
                case .issueWithoutPending:
                    break
                case .prefetchNext, .drainBeforeIssue:
                    throw ModelError.indexCorrupt(
                        detail:
                            "routed tile scheduler requested pending action without pending tile")
                }
            }
            let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                model: model,
                layer: L,
                tileIndex: tileIndex,
                routes: routes,
                plannedFetch: plannedFetch,
                avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
            try fetch.binding.validateCoversPairs(
                routes.sortedPairs,
                pairStart: Int(tile.pairStart),
                pairCount: Int(tile.pairCount))
            if !fetch.plannedMissSlots.isEmpty {
                try tileLifetime.begin(
                    tileIndex: tileIndex,
                    plannedSlots: fetch.plannedMissSlots)
            }
            let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                device: ctx.device,
                binding: fetch.binding)
            let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tile.pairStart,
                pairCount: tile.pairCount,
                d: UInt32(D),
                routedIntermediate: UInt32(cfg.moeIntermediateSize),
                topK: UInt32(cfg.topKExperts),
                hiddenStrideElements: UInt32(D),
                binding: fetch.binding,
                offsets: routedOffsets)
            guard let tileCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            let groupStart = Int(tile.groupStart)
            let tookGroupedGEMM =
                try prefillRoutedGEMM?.encodeTile(
                    commandBuffer: tileCB,
                    hidden: scratch.routedX, hiddenStrideElements: D,
                    sortedPairs: metadata.sortedPairs, routePartials: scratch.routePartials,
                    tile: tile,
                    groups: routes.groups[groupStart..<(groupStart + Int(tile.groupCount))],
                    binding: fetch.binding, offsets: routedOffsets,
                    d: D, f: cfg.moeIntermediateSize, topK: cfg.topKExperts) ?? false
            if !tookGroupedGEMM {
                _ = try prefillGroupedMoE.encodeStreamedBatched(
                    commandBuffer: tileCB,
                    hidden: scratch.routedX,
                    sortedPairs: metadata.sortedPairs,
                    routePartials: scratch.routePartials,
                    gateUpActScratch: scratch.routedGateUpActScratch,
                    downScratch: scratch.routedDownScratch,
                    argumentBuffer: argumentBuffer,
                    binding: fetch.binding,
                    params: streamedParams,
                    pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
            }
            tileCB.commit()
            pendingTiles.append(
                PendingPrefillTile(
                    tileIndex: tileIndex,
                    commandBuffer: tileCB,
                    fetch: fetch,
                    argumentBuffer: argumentBuffer,
                    groupedGEMM: tookGroupedGEMM))
            while pendingTiles.count > schedulerConfig.maxPendingDepth {
                try drainOldestPendingTile()
            }
        }
        while !pendingTiles.isEmpty {
            try drainOldestPendingTile()
        }
        try waitForCompletion(sharedCB)
        recordKernelGPU(role: "prefill_shared_expert", sharedCB)
        prefillTileEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        prefillTileNanos &+= prefillTileEnd - prefillRouteEnd
        guard let tailCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try prefillMoE.encodeReduceTokenMajor(
            commandBuffer: tailCB,
            routePartials: scratch.routePartials,
            routeWeights: scratch.routeWeights,
            h2: scratch.h2,
            queryCount: UInt32(t),
            topK: UInt32(cfg.topKExperts),
            d: UInt32(D))
        if hyperConnection != nil {
            // The gated write injects one block output per stream, so
            // the two MLP branches are summed first and written once.
            // Adding them separately would apply the inject gate
            // twice.
            try requireElementwise().encodeResidualAdd(
                commandBuffer: tailCB,
                hidden: scratch.h2,
                delta: scratch.h1,
                count: t * D)
            try encodeResidualExitPrefill(
                commandBuffer: tailCB,
                hidden: scratch.hidden,
                delta: scratch.h2,
                sublayer: .mlp, layer: L,
                tokens: t)
        } else {
            // Plain pre-norm tail: hidden += gated shared branch
            // + routed branch.
            try requireElementwise().encodeResidualAdd(
                commandBuffer: tailCB,
                hidden: scratch.hidden,
                delta: scratch.h1,
                count: t * D)
            try requireElementwise().encodeResidualAdd(
                commandBuffer: tailCB,
                hidden: scratch.hidden,
                delta: scratch.h2,
                count: t * D)
        }
        tailCB.commit()
        withExtendedLifetime(metadata) {
            do {
                try waitForCompletion(tailCB)
                recordKernelGPU(role: "prefill_moe_reduce", tailCB)
            } catch {
                // Rethrown after `metadata` is released.
                tailError = error
            }
        }
        if let error = tailError {
            tailError = nil
            throw error
        }
        if L + 1 < cfg.numLayers {
            guard let nextCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            cb = nextCB
        }
        prefillTailNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - prefillTileEnd
    }

    /// One layer's prefill pass over one chunk.
    ///
    /// Extracted verbatim from the layer loop so the loops can be inverted:
    /// layer-major prefill runs this for every chunk of a band before moving
    /// to the next layer, which is what makes each layer's experts stream once
    /// instead of once per chunk. Everything here is per (layer, chunk) except
    /// `scratch.hidden`, which is the residual and therefore the one buffer the
    /// caller must supply per chunk rather than per pass.
    private func runPrefillLayer(
        _ L: Int,
        cb: inout MTLCommandBuffer,
        scratch: PrefillChunkScratchBuffers,
        layerViews: [LayerPrefillQKVViews],
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        t: Int,
        D: Int,
        eps: Float,
        useTwoRowProjection: Bool,
        snapshotGDNAfterFirstToken: Bool,
        aneChunk: ANEPrefillAttention?,
        pairRoutedMoE: Bool,
        prefillRouteNanos: inout UInt64,
        prefillTileNanos: inout UInt64,
        prefillTailNanos: inout UInt64,
        prefillActiveExperts: inout UInt64,
        slot: Int = 0
    ) async throws {
        try Task.checkCancellation()
        let prefillLayerStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        model.beginOpeningRoutedExpertStreamer(layer: L)
        let views = layerViews[L]
        let isLinear = cfg.layerIsLinear(L)
        let isFull = cfg.fullAttentionLayerMask[L] == 1
        let headDim = isFull ? cfg.fullHeadDim : cfg.headDim
        let numKVHeads = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
        let qDim = cfg.numHeads * headDim
        let kvDim = numKVHeads * headDim

        if cfg.ple.layerIndices.contains(L) {
            try encodePLEPrefill(
                commandBuffer: cb,
                hidden: scratch.hidden,
                layer: L, tokens: t, eps: eps)
        }
        try encodeResidualEntryPrefill(
            commandBuffer: cb,
            hidden: scratch.hidden,
            norm: views.inputNorm,
            out: scratch.normed,
            sublayer: .attention, layer: L,
            tokens: t, eps: eps)
        try prefillSplitPoint(&cb, role: "prefill_split_hc_in")
        // The indexer caches a key for every prefilled token, in or out
        // of the dense-exact window: decode crossing the boundary later
        // must not find holes behind it.
        let qsaSelection = try encodeQSAPrefill(
            cb: &cb, blockInput: scratch.normed,
            layer: L, startPosition: startPosition,
            tokens: t, eps: eps)
        if isLinear {
            try encodeLinearAttentionPrefill(
                cb: &cb, layer: L, views: views, scratch: scratch,
                tokenCount: t, hiddenSize: D,
                snapshotGDNAfterFirstToken: snapshotGDNAfterFirstToken,
                useTwoRowProjection: useTwoRowProjection,
                slot: slot)
        } else if let ane = aneChunk, ane.coveredLayers.contains(L) {
            // The indexer's selection is computed above for this layer whether
            // it runs here or on the GPU; the ANE has to be fed the same
            // choice, or it attends to keys the model drops.
            try await runANEFullAttentionPrefill(
                ane: ane, cb: &cb, layer: L, scratch: scratch,
                tokenCount: t, hiddenSize: D,
                startPosition: startPosition, kvDim: kvDim,
                selection: qsaSelection)
        } else {
            try encodeFullAttentionPrefill(
                cb: &cb, layer: L, views: views, scratch: scratch,
                tokenCount: t, hiddenSize: D, startPosition: startPosition,
                isFull: isFull, headDim: headDim, numKVHeads: numKVHeads,
                qDim: qDim, kvDim: kvDim, rmsEps: eps,
                useTwoRowProjection: useTwoRowProjection,
                keepMask: qsaSelection,
                slot: slot)
        }
        // Plain pre-norm residual block: hidden += attention branch,
        // then one post-attention norm feeds router, shared expert,
        // and routed phase 1 (routedX doubles as moeX).
        try encodeResidualExitPrefill(
            commandBuffer: cb,
            hidden: scratch.hidden,
            delta: scratch.h1,
            sublayer: .attention, layer: L,
            tokens: t)
        try encodeResidualEntryPrefill(
            commandBuffer: cb,
            hidden: scratch.hidden,
            norm: views.postAttention,
            out: scratch.routedX,
            sublayer: .mlp, layer: L,
            tokens: t, eps: eps)
        try prefillSplitPoint(&cb, role: "prefill_split_hc_out")
        if pairRoutedMoE, t == 2 {
            try await encodeRoutedMoEVerifyPair(
                cb: &cb, layer: L, views: views, scratch: scratch,
                hiddenSize: D)
        } else {
            try await encodeRoutedMoEPrefill(
                cb: &cb, layer: L, views: views, scratch: scratch,
                tokenCount: t, startPosition: startPosition, hiddenSize: D,
                layerStart: prefillLayerStart,
                routeNanos: &prefillRouteNanos,
                tileNanos: &prefillTileNanos,
                tailNanos: &prefillTailNanos,
                activeExperts: &prefillActiveExperts)
        }
        // The stage above awaits its own command buffers, so the layer's output
        // hidden is complete here. This is the number a reference dump compares
        // against (`layerN`), which is what localizes a wrong stage to a layer.
        if activationDumpActive(position: startPosition), L <= dumpLayerLimit {
            dumpActivationPrivate(
                "L\(L)_after", scratch.hidden,
                count: t * D, position: startPosition)
            flushDeferredDumps()
        }
    }
}
