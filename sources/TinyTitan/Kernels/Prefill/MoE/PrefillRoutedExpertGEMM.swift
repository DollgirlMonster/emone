import Foundation
import Metal

/// A streamed routed-expert tile as grouped GEMMs on the MPP tensor-op QMM
/// (`TINYTITAN_PREFILL_ROUTED_MPP`).
///
/// The default tile kernels compute one output element per thread with a full
/// dot product each, ~1.2 TFLOPS on an M1 Max, while the MPP QMM runs the dense
/// prefill projections at ~4. A tile's pairs are sorted by expert, so this
/// gathers their token rows into one block, runs gate and up per expert as MPP
/// GEMMs over that expert's contiguous rows, applies the activation to the whole
/// tile at once, runs down per expert, and scatters the rows to the (token,
/// rank) slots of the route partials -- the same slots, unweighted, that
/// `prefill_grouped_routed_moe_batched_down` writes, so the reduce is unchanged.
///
/// The MPP QMM rounds each dequantized weight to half and sums in its own order,
/// so the output is close to the tile kernels', not identical.
final class PrefillRoutedExpertGEMM {
    private let mpp: MPPPrefillInt4QMM
    /// `prefill_routed_qmm_gate_up` / `prefill_routed_qmm`: the whole tile in
    /// two launches on the simdgroup matrix units instead of one MPP GEMM per
    /// expert and projection. Nil keeps the MPP GEMMs.
    private let tiledGateUp: MTLComputePipelineState?
    private let tiledDown: MTLComputePipelineState?
    static let gateUpEdge = 32
    static let downEdge = 64
    private let gather: MTLComputePipelineState
    private let scatter: MTLComputePipelineState
    private let activation: MTLComputePipelineState
    private let device: MTLDevice
    private var rowsIn: MTLBuffer?
    private var gateOut: MTLBuffer?
    private var upOut: MTLBuffer?
    private var activated: MTLBuffer?
    private var rowsOut: MTLBuffer?
    private var capacityPairs = 0

    /// Nil when this GPU has no MPP QMM or the width is not one it serves.
    init?(
        context: MetalContext, weightBits: Int, siluActivation: Bool, tiledQMM: Bool = false
    ) throws {
        guard [4, 8].contains(weightBits) else { return nil }
        let mpp = MPPPrefillInt4QMM(context: context, weightBits: weightBits)
        guard mpp.isAvailable || tiledQMM else { return nil }
        self.mpp = mpp
        if tiledQMM {
            let constants = [
                MetalFunctionConstant(index: 77, value: .bool(siluActivation)),
                MetalFunctionConstant(index: 78, value: .uint32(UInt32(weightBits))),
            ]
            // Fused gate+up holds two accumulator sets, so it keeps the 32 x 32
            // tile (6.3 TFLOPS; 64 x 64: 5.9); down takes 64 x 64 (6.5; 32 x 32:
            // 5.6). Microbenchmark at Qwen3.8's routed shape, M1 Max.
            self.tiledGateUp = try context.pipeline(
                "prefill_routed_qmm_gate_up_\(Self.gateUpEdge)x\(Self.gateUpEdge)",
                constants: constants)
            self.tiledDown = try context.pipeline(
                "prefill_routed_qmm_\(Self.downEdge)x\(Self.downEdge)", constants: constants)
        } else {
            self.tiledGateUp = nil
            self.tiledDown = nil
        }
        self.gather = try context.pipeline("prefill_routed_gather_rows")
        self.scatter = try context.pipeline("prefill_routed_scatter_rows")
        self.activation = try context.pipeline(siluActivation ? "silu_mul_fp16" : "gelu_mul_fp16")
        self.device = context.device
    }

    private func reserve(pairs: Int, d: Int, f: Int) throws {
        guard pairs > capacityPairs else { return }
        let half = MemoryLayout<Float16>.stride
        func make(_ elements: Int, _ label: String) throws -> MTLBuffer {
            guard let made = device.makeBuffer(length: max(1, elements * half), options: .storageModePrivate)
            else { throw MetalError.bufferAllocationFailed(label) }
            made.label = label
            return made
        }
        rowsIn = try make(pairs * d, "prefill.routedGEMM.rowsIn")
        gateOut = try make(pairs * f, "prefill.routedGEMM.gate")
        upOut = try make(pairs * f, "prefill.routedGEMM.up")
        activated = try make(pairs * f, "prefill.routedGEMM.act")
        rowsOut = try make(pairs * d, "prefill.routedGEMM.rowsOut")
        capacityPairs = pairs
    }

    /// Whether every expert of the tile has a shape and layout the MPP QMM
    /// takes. Checked before anything is encoded, so a refusal leaves the tile
    /// to the default kernels whole.
    static func accepts(
        d: Int, f: Int, binding: PrefillStreamedTileBinding, offsets: MoEExpertOffsets
    ) -> Bool {
        let group = MPPPrefillInt4QMM.tileK
        guard d > 0, f > 0, d % group == 0, f % group == 0 else { return false }
        let halves = [
            offsets.gateSOff, offsets.gateBOff, offsets.upSOff, offsets.upBOff,
            offsets.downSOff, offsets.downBOff,
        ]
        return binding.views.allSatisfy { view in
            halves.allSatisfy { (Int(view.offset) + Int($0)) % MemoryLayout<UInt16>.stride == 0 }
        }
    }

    /// Encodes the tile into `commandBuffer`. Returns false, having encoded
    /// nothing, when `accepts` refuses it.
    func encodeTile(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenStrideElements: Int,
        sortedPairs: MTLBuffer, routePartials: MTLBuffer,
        tile: PrefillMoETile, groups: ArraySlice<PrefillMoEGroup>,
        binding: PrefillStreamedTileBinding, offsets: MoEExpertOffsets,
        d: Int, f: Int, topK: Int
    ) throws -> Bool {
        let pairs = Int(tile.pairCount)
        guard pairs > 0, Self.accepts(d: d, f: f, binding: binding, offsets: offsets) else {
            return false
        }
        if let tiledGateUp, let tiledDown,
            Self.acceptsTiled(d: d, f: f, binding: binding, offsets: offsets)
        {
            try reserve(pairs: pairs, d: d, f: f)
            guard let rowsIn, let activated, let rowsOut else {
                throw MetalError.bufferAllocationFailed("prefill.routedGEMM")
            }
            try encodeRows(
                commandBuffer, gather, source: hidden, destination: rowsIn,
                sortedPairs: sortedPairs, tile: tile, d: d, last: UInt32(hiddenStrideElements))
            try encodeTiled(
                commandBuffer, gateUp: tiledGateUp, down: tiledDown,
                rowsIn: rowsIn, activated: activated, rowsOut: rowsOut,
                tile: tile, groups: groups, binding: binding, offsets: offsets, d: d, f: f)
            try encodeRows(
                commandBuffer, scatter, source: rowsOut, destination: routePartials,
                sortedPairs: sortedPairs, tile: tile, d: d, last: UInt32(topK))
            return true
        }
        try reserve(pairs: pairs, d: d, f: f)
        guard let rowsIn, let gateOut, let upOut, let activated, let rowsOut else {
            throw MetalError.bufferAllocationFailed("prefill.routedGEMM")
        }
        try encodeRows(
            commandBuffer, gather, source: hidden, destination: rowsIn,
            sortedPairs: sortedPairs, tile: tile, d: d, last: UInt32(hiddenStrideElements))

        let half = MemoryLayout<Float16>.stride
        // Every expert's gate and up GEMMs are independent -- disjoint rows in,
        // disjoint rows out -- and each alone is ~60 threadgroups, too few to
        // fill the GPU. One concurrent encoder lets them all run side by side;
        // spike 8 measured them serialized, one encoder each, no faster than the
        // tile kernels.
        guard let gateUp = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent)
        else { throw MetalError.commandEncoderFailed }
        // Scoped so the encoder ends here on every path: one left open when a
        // projection throws traps when the command buffer is released, and the
        // next encoder cannot be made while it is open.
        do {
            defer { gateUp.endEncoding() }
            for group in groups where group.pairCount > 0 {
                let view = try Self.view(for: group, in: binding)
                let base = Int(view.offset)
                let row = Int(group.pairStart - tile.pairStart)
                let count = Int(group.pairCount)
                try project(
                    commandBuffer, gateUp, view.buffer, base,
                    offsets.gateWOff, offsets.gateSOff, offsets.gateBOff,
                    x: rowsIn, xOffset: row * d * half, y: gateOut, yOffset: row * f * half,
                    m: count, n: f, k: d)
                try project(
                    commandBuffer, gateUp, view.buffer, base,
                    offsets.upWOff, offsets.upSOff, offsets.upBOff,
                    x: rowsIn, xOffset: row * d * half, y: upOut, yOffset: row * f * half,
                    m: count, n: f, k: d)
            }
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        // Elementwise, so the whole tile's rows in one dispatch.
        encoder.setComputePipelineState(activation)
        encoder.setBuffer(gateOut, offset: 0, index: 0)
        encoder.setBuffer(upOut, offset: 0, index: 1)
        encoder.setBuffer(activated, offset: 0, index: 2)
        var count = UInt32(pairs * f)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: pairs * f, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(activation.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        encoder.endEncoding()

        guard let down = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent)
        else { throw MetalError.commandEncoderFailed }
        do {
            defer { down.endEncoding() }
            for group in groups where group.pairCount > 0 {
                let view = try Self.view(for: group, in: binding)
                let row = Int(group.pairStart - tile.pairStart)
                try project(
                    commandBuffer, down, view.buffer, Int(view.offset),
                    offsets.downWOff, offsets.downSOff, offsets.downBOff,
                    x: activated, xOffset: row * f * half, y: rowsOut, yOffset: row * d * half,
                    m: Int(group.pairCount), n: d, k: f)
            }
        }
        try encodeRows(
            commandBuffer, scatter, source: rowsOut, destination: routePartials,
            sortedPairs: sortedPairs, tile: tile, d: d, last: UInt32(topK))
        return true
    }

    /// The tiled kernels' extra terms: 32-column tiles in both projections,
    /// group-64 scales, and weight rows the kernel reads a word (4-bit) or two
    /// (8-bit) at a time.
    static func acceptsTiled(
        d: Int, f: Int, binding: PrefillStreamedTileBinding, offsets: MoEExpertOffsets
    ) -> Bool {
        guard d % 64 == 0, f % 64 == 0 else { return false }
        let words = [offsets.gateWOff, offsets.upWOff, offsets.downWOff]
        return binding.views.allSatisfy { view in
            words.allSatisfy { (Int(view.offset) + Int($0)) % 8 == 0 }
        }
    }

    /// The tile's rows as work items of at most one tile edge, each inside one
    /// expert (32 rows for gate+up, 64 for down); then
    /// gate+up+activation in one launch and down in another, every expert of
    /// the tile at once. The experts are reached through a table of device
    /// addresses (`PrefillStreamedRoutedBlobsMSL`'s layout), indexed by the
    /// expert's position in `binding`.
    private func encodeTiled(
        _ cb: MTLCommandBuffer,
        gateUp: MTLComputePipelineState, down: MTLComputePipelineState,
        rowsIn: MTLBuffer, activated: MTLBuffer, rowsOut: MTLBuffer,
        tile: PrefillMoETile, groups: ArraySlice<PrefillMoEGroup>,
        binding: PrefillStreamedTileBinding, offsets: MoEExpertOffsets,
        d: Int, f: Int
    ) throws {
        var addresses = [UInt64](repeating: 0, count: 16)
        for (slot, view) in binding.views.enumerated() {
            addresses[slot] = view.buffer.gpuAddress + UInt64(view.offset)
        }
        func workItems(edge: Int) throws -> (buffer: MTLBuffer, count: Int) {
            var work: [UInt32] = []
            for group in groups where group.pairCount > 0 {
                guard let slot = binding.expertIDs.firstIndex(of: Int(group.expert)) else {
                    throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                        "expert \(group.expert) of the tile has no binding")
                }
                let first = Int(group.pairStart - tile.pairStart)
                let count = Int(group.pairCount)
                for start in stride(from: 0, to: count, by: edge) {
                    work += [
                        UInt32(slot), UInt32(first + start), UInt32(min(edge, count - start)), 0,
                    ]
                }
            }
            guard !work.isEmpty,
                let buffer = device.makeBuffer(
                    bytes: work, length: work.count * 4, options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed("prefill.routedQMM.work") }
            return (buffer, work.count / 4)
        }
        let gateUpWork = try workItems(edge: Self.gateUpEdge)
        let downWork = try workItems(edge: Self.downEdge)
        let gateUpParams: [UInt32] = [
            UInt32(f), UInt32(d),
            offsets.gateWOff, offsets.gateSOff, offsets.gateBOff,
            offsets.upWOff, offsets.upSOff, offsets.upBOff, 64,
        ]
        let downParams: [UInt32] = [
            UInt32(d), UInt32(f),
            offsets.downWOff, offsets.downSOff, offsets.downBOff, 0, 0, 0, 64,
        ]
        for (pso, x, y, params, n, work, edge) in [
            (gateUp, rowsIn, activated, gateUpParams, f, gateUpWork, Self.gateUpEdge),
            (down, activated, rowsOut, downParams, d, downWork, Self.downEdge),
        ] {
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw MetalError.commandEncoderFailed
            }
            enc.setComputePipelineState(pso)
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(y, offset: 0, index: 1)
            enc.setBytes(&addresses, length: addresses.count * 8, index: 2)
            enc.setBuffer(work.buffer, offset: 0, index: 3)
            var p = params
            enc.setBytes(&p, length: p.count * 4, index: 4)
            for view in binding.views { enc.useResource(view.buffer, usage: .read) }
            enc.dispatchThreadgroups(
                MTLSize(width: n / edge, height: work.count, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.endEncoding()
        }
    }

    private static func view(
        for group: PrefillMoEGroup, in binding: PrefillStreamedTileBinding
    ) throws -> TensorView {
        guard let index = binding.expertIDs.firstIndex(of: Int(group.expert)) else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "expert \(group.expert) of the tile has no binding")
        }
        return binding.views[index]
    }

    /// One expert projection into `encoder`. `accepts` has already checked what
    /// the QMM checks, so a refusal here is an inconsistency, not a fallback.
    private func project(
        _ cb: MTLCommandBuffer, _ encoder: MTLComputeCommandEncoder,
        _ blob: MTLBuffer, _ base: Int,
        _ weightsOff: UInt32, _ scalesOff: UInt32, _ biasesOff: UInt32,
        x: MTLBuffer, xOffset: Int, y: MTLBuffer, yOffset: Int,
        m: Int, n: Int, k: Int
    ) throws {
        let path = try mpp.encode(
            commandBuffer: cb,
            weights: blob, weightsOffset: base + Int(weightsOff),
            scales: blob, scalesOffset: base + Int(scalesOff),
            biases: blob, biasesOffset: base + Int(biasesOff),
            x: x, xOffset: xOffset, y: y, yOffset: yOffset,
            m: m, n: n, k: k, required: true, into: encoder)
        guard path == .affineThreadgroupF16 else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "the MPP QMM refused an expert projection it had accepted")
        }
    }

    /// The gather or the scatter over the tile's pairs: both take the pairs,
    /// the row width and one trailing constant (the hidden stride, or top-k).
    private func encodeRows(
        _ cb: MTLCommandBuffer, _ pipeline: MTLComputePipelineState,
        source: MTLBuffer, destination: MTLBuffer, sortedPairs: MTLBuffer,
        tile: PrefillMoETile, d: Int, last: UInt32
    ) throws {
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(sortedPairs, offset: 0, index: 1)
        encoder.setBuffer(destination, offset: 0, index: 2)
        var start = tile.pairStart
        var count = tile.pairCount
        var width = UInt32(d)
        var trailing = last
        encoder.setBytes(&start, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&width, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&trailing, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: d, height: Int(tile.pairCount), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 4, depth: 1))
        encoder.endEncoding()
    }
}
