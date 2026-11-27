import Foundation
import Metal

/// Runtime LoRA (Emone-lmw.11): a decision adapter laid over this
/// runner's prefill projections, so a System One model runs on the resident
/// base weights instead of a merged copy. Every adapted projection gets
/// `y += (alpha / r) * (x A^T) B^T` after its base GEMM: q/k/v/o, the
/// Gated-DeltaNet in_proj_qkv/z/a/b and out_proj in
/// `encodeAffineProjection`, and -- dense family only -- the MLP's gate/up
/// (before the activation) and down inside the batched dense FFN. Decode is
/// never adapted: a decision only prefills.
///
/// Two ways to lay one on (Emone-lmw.14):
/// - `setLoRAAdapter`: for every prefill this runner does, until removed. Only
///   for a runner that is nothing but a decision model (Rev on dense Qwen 3.5,
///   `DecisionModel`), because it adapts the context too.
/// - `prefillShortBranch(adapter:)`: for that one branch and nothing else, on
///   any family. This is System One on System Two's runner: the held context
///   was computed without the adapter and stays that way; only the branch
///   rows (and the KV / GDN rows they write, which the caller rewinds) see it.
extension RealForwardRunner {
    /// Lays `adapter` over this runner's prefills, or removes it with nil.
    /// Refused for anything but the dense Qwen 3.5 family -- on any other the
    /// adapter belongs on a branch (`prefillShortBranch(adapter:)`) -- and when
    /// the adapter does not fit this model.
    public func setLoRAAdapter(_ adapter: LoRAAdapter?) throws {
        guard let adapter else {
            loraAdapter = nil
            return
        }
        guard cfg.family == .qwen35Dense else {
            throw ModelError.internalInconsistency(
                detail:
                    "a standing runtime LoRA is for a dense Qwen 3.5 decision runner; on "
                    + "\(cfg.family.rawValue) lay it on a branch with prefillShortBranch(adapter:)"
            )
        }
        if let refusal = loraAdapterRefusal(adapter) {
            throw ModelError.internalInconsistency(detail: refusal)
        }
        try ensureLoRADeltaKernel()
        loraAdapter = adapter
    }

    /// Why `adapter` cannot run over this runner's prefills, or nil. Checked
    /// before any GPU work: the base, every target's shape and, given the
    /// largest chunk a call will encode, the adapter's scratch.
    public func loraAdapterRefusal(_ adapter: LoRAAdapter, chunkRows: Int? = nil) -> String? {
        guard adapter.baseModelID == model.modelID else {
            return "adapter \(adapter.modelID) is for \(adapter.baseModelID), not \(model.modelID)"
        }
        if let refusal = LoRAAdapterFit.refusal(entries: adapter.declared, config: cfg) {
            return refusal
        }
        if let rows = chunkRows, adapter.maxTokens < rows {
            return "adapter scratch holds \(adapter.maxTokens) tokens; this call's chunk is \(rows)"
        }
        let ffn: [LoRATarget] = [.gate, .up, .down]
        if !prefillBatchedSharedExpert,
            adapter.declared.contains(where: { ffn.contains($0.target) })
        {
            return "an MLP adapter needs the batched dense FFN (the profile's dense QMM switch)"
        }
        return nil
    }

    func ensureLoRADeltaKernel() throws {
        if loraDeltaKernel == nil { loraDeltaKernel = try LoRADelta(context: ctx) }
    }

    /// True when any low-rank delta (adapter or patch) is laid over prefill.
    /// Paths with no delta seams (the ANE attention sidecar, the sequential
    /// hyper-connection oracle) must not run while this holds.
    var hasLowRankDeltas: Bool { loraAdapter != nil || weightPatch != nil }

    /// One adapted projection's delta: where it reads x and accumulates y.
    struct LoRADeltaItem {
        let target: LoRATarget
        let x: MTLBuffer
        let xStrideElements: Int
        let y: MTLBuffer
        let yStrideElements: Int
        let rows: Int
        let columns: Int
    }

    func encodeLoRADelta(
        commandBuffer: MTLCommandBuffer, layer: Int, target: LoRATarget,
        x: MTLBuffer, xStrideElements: Int, y: MTLBuffer, yStrideElements: Int,
        rows: Int, columns: Int, tokenCount: Int
    ) throws {
        try encodeLoRADeltas(
            commandBuffer: commandBuffer, layer: layer,
            items: [
                LoRADeltaItem(
                    target: target, x: x, xStrideElements: xStrideElements, y: y,
                    yStrideElements: yStrideElements, rows: rows, columns: columns)
            ],
            tokenCount: tokenCount)
    }

    /// One set of low-rank deltas the runner lays over its projections: the
    /// decision adapter or a weight patch (Emone-a37). Both keep A and B^T
    /// in one fp32 buffer and share the kernel and the seams; they differ only
    /// in where they come from and how they are scaled.
    struct LowRankDeltaSource {
        let label: String
        let buffer: MTLBuffer
        let scratch: MTLBuffer
        let rank: Int
        let scale: Float
        let maxTokens: Int
        let entry: (Int, LoRATarget) -> (aOffset: Int, bOffset: Int, rows: Int, columns: Int)?
    }

    /// The delta sets in effect, adapter first. Empty when neither is set.
    var lowRankDeltaSources: [LowRankDeltaSource] {
        var sources: [LowRankDeltaSource] = []
        if let adapter = loraAdapter {
            sources.append(
                LowRankDeltaSource(
                    label: "LoRA", buffer: adapter.buffer, scratch: adapter.scratch,
                    rank: adapter.rank, scale: adapter.scale, maxTokens: adapter.maxTokens,
                    entry: { layer, target in
                        adapter.entry(layer: layer, target: target).map {
                            ($0.aOffset, $0.bOffset, $0.rows, $0.columns)
                        }
                    }))
        }
        if let patch = weightPatch {
            sources.append(
                LowRankDeltaSource(
                    label: "weight patch", buffer: patch.buffer, scratch: patch.scratch,
                    rank: patch.rank, scale: 1, maxTokens: patch.maxTokens,
                    entry: { layer, target in
                        patch.entry(layer: layer, target: target).map {
                            ($0.aOffset, $0.bOffset, $0.rows, $0.columns)
                        }
                    }))
        }
        return sources
    }

    /// The deltas of several projections of layer `L` in one encoder -- the
    /// ones that read the same input run after the last of their base GEMMs,
    /// which cuts an encoder per target (Emone-lmw.12). Targets no delta
    /// set carries are skipped; nothing is encoded without one.
    func encodeLoRADeltas(
        commandBuffer: MTLCommandBuffer, layer: Int, items: [LoRADeltaItem], tokenCount: Int
    ) throws {
        guard hasLowRankDeltas else { return }
        for source in lowRankDeltaSources {
            try encodeLowRankDeltas(
                source, commandBuffer: commandBuffer, layer: layer, items: items,
                tokenCount: tokenCount)
        }
    }

    private func encodeLowRankDeltas(
        _ source: LowRankDeltaSource,
        commandBuffer: MTLCommandBuffer, layer: Int, items: [LoRADeltaItem], tokenCount: Int
    ) throws {
        let present = items.compactMap { item in
            source.entry(layer, item.target).map { (item, $0) }
        }
        guard !present.isEmpty, tokenCount > 0 else { return }
        guard let kernel = loraDeltaKernel else {
            throw ModelError.internalInconsistency(
                detail: "\(source.label) set without its kernel")
        }
        guard tokenCount <= source.maxTokens else {
            throw ModelError.internalInconsistency(
                detail:
                    "\(source.label) scratch holds \(source.maxTokens) tokens; "
                    + "this chunk has \(tokenCount)")
        }
        for (item, entry) in present where entry.rows != item.rows || entry.columns != item.columns
        {
            throw ModelError.internalInconsistency(
                detail:
                    "\(source.label) \(item.target.rawValue) of layer \(layer) is "
                    + "\(entry.rows)x\(entry.columns); the projection is \(item.rows)x\(item.columns)"
            )
        }
        guard let encoder = commandBuffer.timedComputeEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        defer { encoder.releaseEncoder() }
        let first = present[0].0
        if present.count > 1, present.count <= LoRADelta.maximumGroup,
            present.allSatisfy({
                $0.0.x === first.x && $0.0.xStrideElements == first.xStrideElements
                    && $0.0.columns == first.columns
            })
        {
            try kernel.encodeGroup(
                encoder: encoder, x: first.x, xStrideElements: first.xStrideElements,
                columns: first.columns,
                members: present.map { item, entry in
                    LoRADelta.Member(
                        y: item.y, yStrideElements: item.yStrideElements, rows: item.rows,
                        aOffset: entry.aOffset, bOffset: entry.bOffset)
                },
                adapter: source.buffer, scratch: source.scratch,
                rank: source.rank, tokens: tokenCount, scale: source.scale)
            return
        }
        for (item, entry) in present {
            try kernel.encode(
                encoder: encoder,
                x: item.x, xStrideElements: item.xStrideElements,
                y: item.y, yStrideElements: item.yStrideElements,
                a: source.buffer, aOffset: entry.aOffset, b: source.buffer,
                bOffset: entry.bOffset,
                scratch: source.scratch,
                rank: source.rank, rows: item.rows, columns: item.columns, tokens: tokenCount,
                scale: source.scale)
        }
    }

    /// The batched dense FFN's two seams for layer `L`, or nil when no
    /// adapter or patch carries an FFN delta for it (the FFN then runs exactly
    /// as before).
    func loraFFNHooks(
        layer L: Int, scratch: PrefillChunkScratchBuffers, tokenCount t: Int, hiddenSize D: Int
    ) -> (gateUp: (MTLCommandBuffer) throws -> Void, down: (MTLCommandBuffer) throws -> Void)? {
        // Only a delta set that carries an FFN delta for this layer needs the
        // seams; a head-scale-only patch leaves the FFN exactly as it was.
        let ffn: [LoRATarget] = [.gate, .up, .down]
        guard
            lowRankDeltaSources.contains(where: { source in
                ffn.contains { source.entry(L, $0) != nil }
            })
        else { return nil }
        let intermediate = cfg.intermediateSize
        let gateUp: (MTLCommandBuffer) throws -> Void = { cb in
            try self.encodeLoRADeltas(
                commandBuffer: cb, layer: L,
                items: [
                    LoRADeltaItem(
                        target: .gate, x: scratch.routedX, xStrideElements: D,
                        y: scratch.sharedGateScratch, yStrideElements: intermediate,
                        rows: intermediate, columns: D),
                    LoRADeltaItem(
                        target: .up, x: scratch.routedX, xStrideElements: D,
                        y: scratch.sharedUpScratch, yStrideElements: intermediate,
                        rows: intermediate, columns: D),
                ],
                tokenCount: t)
        }
        let down: (MTLCommandBuffer) throws -> Void = { cb in
            try self.encodeLoRADelta(
                commandBuffer: cb, layer: L, target: .down,
                x: scratch.sharedActScratch, xStrideElements: intermediate,
                y: scratch.h1, yStrideElements: D,
                rows: D, columns: intermediate, tokenCount: t)
        }
        return (gateUp, down)
    }
}
