import Foundation
import Metal

/// Short prefill chunks on the decode routed kernels.
///
/// A chunk's routed stage normally reads the router's picks back, groups the
/// (token, expert) pairs by expert and runs one tile per group: right for a
/// long chunk, where each expert serves many rows, and mostly overhead for a
/// short one, where a 21-row chunk touches ~62 experts a layer for 2.7 rows
/// each (196 of a 320 ms pass on Qwen 3.6 4-bit). When every expert is
/// resident a row needs no plan at all: the resolve kernel turns every row's
/// picks into device pointers and the row-batched decode kernels run on
/// them in the layer's own command buffer, with no readback (21 rows 310 ->
/// 184 ms, 32 rows 330 -> 204).
///
/// The arithmetic is decode's, not the tile path's -- per row, the shared
/// branch is the reduce's residual -- so the two paths round differently.
/// Measured on Qwen 3.6 4-bit over 40 greedy requests: 7 of 24 conversations
/// pick a different token at a top-2 margin <= 0.30, and against a prompt run
/// through the decode step token by token (`TINYTITAN_PREFILL_SEQUENTIAL=1`)
/// neither path is closer (median |logit - reference| 0.047 here, 0.062 on
/// tiles; closer at 738 and 780 of 1,736 positions). What it does cost is
/// that a prompt prefilled on a cold cache and one prefilled on a warm cache
/// no longer take the same path. `TINYTITAN_PREFILL_RESIDENT_ROWS=0` turns it
/// off; a smaller value lowers the row limit.
extension RealForwardRunner {
    /// TINYTITAN_PREFILL_RESIDENT_ROWS when set: it caps a short branch too,
    /// so `=0` turns the per-row stage off everywhere.
    static let residentPrefillRowLimitOverride: Int? =
        ProcessInfo.processInfo.environment["TINYTITAN_PREFILL_RESIDENT_ROWS"]
        .flatMap { Int($0) }.map { min(max(0, $0), MoE.maxRoutedRows) }
    static let residentPrefillRowLimit: Int =
        residentPrefillRowLimitOverride ?? MoE.maxRoutedRows

    /// Rows the per-row routed stage can take in one chunk: blocks of
    /// `MoE.maxRoutedRows`, one argument / activation range each.
    /// `prefillShortBranch` asks for up to this; ordinary chunks keep
    /// `residentPrefillRowLimit`, so their path does not change.
    static let residentPrefillMaxBlockedRows = 2 * MoE.maxRoutedRows

    /// Whether a chunk of `tokenCount` rows takes the per-row routed stage:
    /// short enough, a family the row-batched kernels cover, no diagnostic
    /// that needs the picks on the host, and every layer fully resident.
    /// `rowLimit` defaults to `residentPrefillRowLimit`; a short-branch
    /// prefill passes its own, at most `residentPrefillMaxBlockedRows`.
    func prepareResidentPrefill(tokenCount: Int, rowLimit: Int? = nil) throws -> Bool {
        guard residentPrefillRefusal(tokenCount: tokenCount, rowLimit: rowLimit) == nil
        else { return false }
        return try prepareResidentDecode(warm: false)
    }

    /// Why a chunk of `tokenCount` rows cannot take the per-row routed stage
    /// for a reason other than the cache's contents; nil when only residency
    /// is left to check.
    func residentPrefillRefusal(tokenCount: Int, rowLimit: Int? = nil) -> String? {
        let limit = min(
            rowLimit ?? Self.residentPrefillRowLimit,
            Self.residentPrefillRowLimitOverride ?? Int.max,
            Self.residentPrefillMaxBlockedRows)
        if tokenCount <= 0 { return "empty chunk" }
        if tokenCount > limit { return "\(tokenCount) rows exceed the row limit \(limit)" }
        if routeTraceFD >= 0 { return "TINYTITAN_ROUTE_TRACE" }
        if Self.prefillLayerTrace { return "TINYTITAN_PREFILL_LAYER_TRACE" }
        if !moe.canEncodeRoutedRows(
            d: UInt32(cfg.hiddenSize), f: UInt32(cfg.moeIntermediateSize))
        {
            return "the row-batched routed kernels do not cover this family"
        }
        return residentPathSettingsRefusal
    }

    /// The shared branch, every row's routed mixture and the residual exit of
    /// one layer, encoded behind the router already in `cb`. The buffer is
    /// committed and not waited on, except the last layer's (and not that one
    /// either under `residentPrefillDeferLastWait`). A chunk past
    /// `MoE.maxRoutedRows` runs the routed rows as blocks of that many.
    func encodeResidentRoutedRowsPrefill(
        cb: inout MTLCommandBuffer, layer L: Int,
        scratch: PrefillChunkScratchBuffers, tokenCount t: Int, hiddenSize D: Int
    ) throws {
        guard L < residentLayers.count, let addresses = residentLayers[L].addresses
        else { throw ModelError.residentBufferWrapFailed }
        try prefillSplitPoint(&cb, role: "prefill_split_router")
        try encodePrefillSharedBranch(
            commandBuffer: cb, layer: L, scratch: scratch, tokenCount: t, hiddenSize: D)
        try prefillSplitPoint(&cb, role: "prefill_split_shared")
        let k = cfg.topKExperts
        let d = UInt32(D)
        let f = UInt32(cfg.moeIntermediateSize)
        let half = MemoryLayout<Float16>.stride
        let slots = residentLayers[L].buffers
        let routedOffsets = try model.routedExpertOffsets(layer: L)
        // One argument / activation / log range per block of rows, so blocks
        // of a chunk past `MoE.maxRoutedRows` never share scratch (and never
        // serialize on it). A chunk within one block uses only the first.
        let blockPairs = MoE.maxRoutedRows * 8
        let blockCount = Self.residentPrefillMaxBlockedRows / MoE.maxRoutedRows
        let argBlockBytes = blockPairs * MoE.routedArgumentPointerStride
        let actsBlockBytes = blockPairs * cfg.moeIntermediateSize * half
        let logBlockBytes = blockPairs * MemoryLayout<UInt32>.stride
        if (residentRowsArgBuffer?.length ?? 0) < blockCount * argBlockBytes {
            residentRowsArgBuffer = ctx.device.makeBuffer(
                length: blockCount * argBlockBytes, options: .storageModeShared)
            residentRowsActs = ctx.device.makeBuffer(
                length: blockCount * actsBlockBytes, options: .storageModeShared)
            residentRowsRouteLog = ctx.device.makeBuffer(
                length: blockCount * logBlockBytes, options: .storageModeShared)
        }
        guard let rowArgs = residentRowsArgBuffer, let rowActs = residentRowsActs,
            let rowLog = residentRowsRouteLog
        else { throw MetalError.bufferAllocationFailed("prefill.residentRows") }
        for block in ResidentRowBlocks.blocks(rows: t, blockRows: MoE.maxRoutedRows) {
            let rowBytes = D * half
            try moe.encodeResolveResidentExperts(
                commandBuffer: cb,
                topKIndices: scratch.routeIDs,
                topKIndicesOffset: block.start * k * MemoryLayout<UInt32>.stride,
                expertAddresses: addresses,
                routedArgBuffer: rowArgs,
                routedArgBufferOffset: block.index * argBlockBytes,
                routeLog: rowLog, routeLogOffset: block.index * logBlockBytes,
                topK: UInt32(block.count * k), numExperts: UInt32(cfg.numExperts))
            try moe.encodeRoutedRows(
                commandBuffer: cb,
                routedArgBuffer: rowArgs, routedArgOffset: block.index * argBlockBytes,
                routedOffsets: routedOffsets,
                x: scratch.routedX, xOffset: block.start * rowBytes,
                acts: rowActs, actsOffset: block.index * actsBlockBytes,
                routingWeights: scratch.routeWeights,
                routingWeightsOffset: block.start * k * half,
                residual: scratch.h1, residualOffset: block.start * rowBytes,
                y: scratch.h2, yOffset: block.start * rowBytes,
                rows: block.count, d: d, f: f, allResident: slots)
        }
        try prefillSplitPoint(&cb, role: "prefill_split_routed_rows")
        // h2 already carries the shared branch, as decode's does.
        if hyperConnection != nil {
            try encodeResidualExitPrefill(
                commandBuffer: cb, hidden: scratch.hidden, delta: scratch.h2,
                sublayer: .mlp, layer: L, tokens: t)
        } else {
            try requireElementwise().encodeResidualAdd(
                commandBuffer: cb, hidden: scratch.hidden, delta: scratch.h2, count: t * D)
        }
        cb.commit()
        residentTokenBuffers.append(("prefill_resident_rows", cb))
        guard L + 1 < cfg.numLayers else {
            // A short branch leaves the last layer unwaited: its head buffer
            // follows on the same queue, and `finishResidentPrefillBuffers`
            // checks these once that has landed. One host wait per branch.
            if !residentPrefillDeferLastWait { try finishResidentPrefillBuffers() }
            guard let next = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            cb = next
            return
        }
        guard let next = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        cb = next
    }

    /// Waits for the last per-row chunk buffer (a no-op once a later buffer on
    /// the queue has been awaited), then surfaces any command error and
    /// records the GPU spans. Returns those spans summed, for the short-branch
    /// trace.
    @discardableResult
    func finishResidentPrefillBuffers() throws -> ResidentPrefillGPUSpan {
        defer { residentTokenBuffers.removeAll(keepingCapacity: true) }
        guard let last = residentTokenBuffers.last else { return ResidentPrefillGPUSpan() }
        try waitForCompletion(last.cb)
        var span = ResidentPrefillGPUSpan()
        for (role, buffer) in residentTokenBuffers {
            if let error = buffer.error {
                throw ModelError.commandBufferFailed(
                    detail: "\(role) command buffer: \(error)")
            }
            recordKernelGPU(role: role, buffer)
            span.add(start: buffer.gpuStartTime, end: buffer.gpuEndTime)
        }
        return span
    }
}

/// Splits a chunk's rows into blocks the row-batched routed kernels take in
/// one call each. Pure, so the arithmetic is tested without a GPU.
enum ResidentRowBlocks {
    struct Block: Equatable {
        /// Block number, which picks its argument / activation range.
        let index: Int
        let start: Int
        let count: Int
    }

    static func blocks(rows: Int, blockRows: Int) -> [Block] {
        guard rows > 0, blockRows > 0 else { return [] }
        return stride(from: 0, to: rows, by: blockRows).enumerated().map { index, start in
            Block(index: index, start: start, count: min(blockRows, rows - start))
        }
    }
}

/// The GPU time of a chunk's command buffers: busy (the buffers' own spans
/// summed) against elapsed (first start to last end). Times are seconds on
/// the host's mach clock, as `MTLCommandBuffer.gpuStartTime` reports them.
struct ResidentPrefillGPUSpan: Equatable {
    var buffers = 0
    var busySeconds: Double = 0
    var firstStart: Double = 0
    var lastEnd: Double = 0

    mutating func add(start: Double, end: Double) {
        guard end >= start, start > 0 else { return }
        if buffers == 0 || start < firstStart { firstStart = start }
        lastEnd = max(lastEnd, end)
        busySeconds += end - start
        buffers += 1
    }

    /// First buffer start to last buffer end; 0 for none.
    var elapsedSeconds: Double { buffers == 0 ? 0 : lastEnd - firstStart }
}
