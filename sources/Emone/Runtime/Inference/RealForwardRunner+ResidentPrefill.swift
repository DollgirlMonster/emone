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
/// through the decode step token by token (`EMONE_PREFILL_SEQUENTIAL=1`)
/// neither path is closer (median |logit - reference| 0.047 here, 0.062 on
/// tiles; closer at 738 and 780 of 1,736 positions). What it does cost is
/// that a prompt prefilled on a cold cache and one prefilled on a warm cache
/// no longer take the same path. `EMONE_PREFILL_RESIDENT_ROWS=0` turns it
/// off; a smaller value lowers the row limit.
extension RealForwardRunner {
    static let residentPrefillRowLimit: Int =
        ProcessInfo.processInfo.environment["EMONE_PREFILL_RESIDENT_ROWS"]
        .flatMap { Int($0) }.map { min(max(0, $0), MoE.maxRoutedRows) } ?? MoE.maxRoutedRows

    /// Whether a chunk of `tokenCount` rows takes the per-row routed stage:
    /// short enough, a family the row-batched kernels cover, no diagnostic
    /// that needs the picks on the host, and every layer fully resident.
    func prepareResidentPrefill(tokenCount: Int) throws -> Bool {
        guard tokenCount > 0, tokenCount <= Self.residentPrefillRowLimit,
            routeTraceFD < 0, !Self.prefillLayerTrace,
            moe.canEncodeRoutedRows(
                d: UInt32(cfg.hiddenSize), f: UInt32(cfg.moeIntermediateSize))
        else { return false }
        return try prepareResidentDecode(warm: false)
    }

    /// The shared branch, every row's routed mixture and the residual exit of
    /// one layer, encoded behind the router already in `cb`. The buffer is
    /// committed and not waited on, except the last layer's.
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
        let pairs = MoE.maxRoutedRows * 8
        if residentRowsArgBuffer == nil {
            residentRowsArgBuffer = ctx.device.makeBuffer(
                length: pairs * MoE.routedArgumentPointerStride, options: .storageModeShared)
            residentRowsActs = ctx.device.makeBuffer(
                length: pairs * cfg.moeIntermediateSize * half, options: .storageModeShared)
            residentRowsRouteLog = ctx.device.makeBuffer(
                length: pairs * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        }
        guard let rowArgs = residentRowsArgBuffer, let rowActs = residentRowsActs,
            let rowLog = residentRowsRouteLog
        else { throw MetalError.bufferAllocationFailed("prefill.residentRows") }
        try moe.encodeResolveResidentExperts(
            commandBuffer: cb,
            topKIndices: scratch.routeIDs, expertAddresses: addresses,
            routedArgBuffer: rowArgs, routeLog: rowLog, routeLogOffset: 0,
            topK: UInt32(t * k), numExperts: UInt32(cfg.numExperts))
        try moe.encodeRoutedRows(
            commandBuffer: cb, routedArgBuffer: rowArgs, routedOffsets: routedOffsets,
            x: scratch.routedX, acts: rowActs, routingWeights: scratch.routeWeights,
            residual: scratch.h1, y: scratch.h2,
            rows: t, d: d, f: f, allResident: slots)
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
            try waitForCompletion(cb)
            defer { residentTokenBuffers.removeAll(keepingCapacity: true) }
            for (role, buffer) in residentTokenBuffers {
                if let error = buffer.error {
                    throw ModelError.commandBufferFailed(
                        detail: "\(role) command buffer: \(error)")
                }
                recordKernelGPU(role: role, buffer)
            }
            return
        }
        guard let next = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        cb = next
    }
}
