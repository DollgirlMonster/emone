import Foundation
import Metal

/// The short-branch routed stage grouped by expert on the GPU
/// (`EMONE_SHORT_BRANCH_GROUPED=1`, layered on `EMONE_SHORT_BRANCH=1`).
///
/// The per-row stage reads every picked expert once per row; the tile path
/// reads each once per tile but needs the picks on the host to plan. This is
/// the tile path's arithmetic with the plan built on the GPU: the router's
/// picks are sorted by expert into the tile path's pair layout
/// (`short_branch_group_pairs`), the rows gathered, gate+up+activation and down
/// run as the tile path's tiled QMM (`prefill_routed_qmm_body`, through
/// wrappers that reach the expert by the resident address table and skip
/// padding), and the rows scattered into the route partials for the tile
/// path's token-major reduce. The work list is padded to its worst case, so
/// nothing about the grid depends on the routing and nothing is read back.
final class ShortBranchGroupedMoE {
    /// Work-item edges (rows per QMM tile) with kernels: the tile path's 64,
    /// and 32 / 16 for branches whose experts hold few rows each.
    static let edges = [16, 32, 64]
    static let gateUpColumns = 32
    static let downColumns = 64

    private let group: MTLComputePipelineState
    /// Per edge: gate+up (edge x 32 tiles) and down (edge x 64).
    private let gateUp: [Int: MTLComputePipelineState]
    private let down: [Int: MTLComputePipelineState]
    private let gather: MTLComputePipelineState
    private let scatter: MTLComputePipelineState
    private let device: MTLDevice
    private var sortedPairs: MTLBuffer?
    private var work: MTLBuffer?
    private var rowsIn: MTLBuffer?
    private var activated: MTLBuffer?
    private var rowsOut: MTLBuffer?
    private var capacityPairs = 0

    init(context: MetalContext, weightBits: Int, siluActivation: Bool) throws {
        let constants = [
            MetalFunctionConstant(index: 77, value: .bool(siluActivation)),
            MetalFunctionConstant(index: 78, value: .uint32(UInt32(weightBits))),
        ]
        self.group = try context.pipeline("short_branch_group_pairs")
        var gateUp: [Int: MTLComputePipelineState] = [:]
        var down: [Int: MTLComputePipelineState] = [:]
        for edge in Self.edges {
            gateUp[edge] = try context.pipeline(
                "short_branch_qmm_gate_up_\(edge)x\(Self.gateUpColumns)", constants: constants)
            down[edge] = try context.pipeline(
                "short_branch_qmm_\(edge)x\(Self.downColumns)", constants: constants)
        }
        self.gateUp = gateUp
        self.down = down
        self.gather = try context.pipeline("prefill_routed_gather_rows")
        self.scatter = try context.pipeline("prefill_routed_scatter_rows")
        self.device = context.device
    }

    /// Work items the grid is sized for: at most one partial item per live
    /// expert, plus one per full `edge` rows.
    static func workCapacity(pairs: Int, numExperts: Int, edge: Int = 64) -> Int {
        min(numExperts, pairs) + pairs / edge + 1
    }

    /// Whether the tiled kernels take this shape: 64-wide projections and
    /// weight rows a kernel reads a word at a time.
    static func accepts(d: Int, f: Int, numExperts: Int, offsets: MoEExpertOffsets) -> Bool {
        guard d % 64 == 0, f % 64 == 0, numExperts > 0, numExperts <= 1024 else { return false }
        return [offsets.gateWOff, offsets.upWOff, offsets.downWOff].allSatisfy { $0 % 8 == 0 }
    }

    private func reserve(pairs: Int, d: Int, f: Int, numExperts: Int) throws {
        guard pairs > capacityPairs else { return }
        let half = MemoryLayout<Float16>.stride
        func make(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard let made = device.makeBuffer(length: max(16, bytes), options: .storageModePrivate)
            else { throw MetalError.bufferAllocationFailed(label) }
            made.label = label
            return made
        }
        sortedPairs = try make(pairs * 16, "shortBranch.grouped.pairs")
        work = try make(
            Self.workCapacity(pairs: pairs, numExperts: numExperts, edge: Self.edges.min() ?? 16)
                * 16, "shortBranch.grouped.work"
        )
        rowsIn = try make(pairs * d * half, "shortBranch.grouped.rowsIn")
        activated = try make(pairs * f * half, "shortBranch.grouped.act")
        rowsOut = try make(pairs * d * half, "shortBranch.grouped.rowsOut")
        capacityPairs = pairs
    }

    /// Encodes the routed stage of `tokens` rows into `cb`: route partials
    /// `[tokens * topK, d]` unweighted, as the tile path leaves them.
    func encode(
        commandBuffer cb: inout MTLCommandBuffer,
        routeIDs: MTLBuffer, routeWeights: MTLBuffer,
        hidden: MTLBuffer, routePartials: MTLBuffer,
        expertAddresses: MTLBuffer, residentBuffers: [MTLBuffer],
        offsets: MoEExpertOffsets,
        tokens: Int, topK: Int, d: Int, f: Int, numExperts: Int, edge: Int = 64,
        split: (inout MTLCommandBuffer, String) throws -> Void = { _, _ in }
    ) throws {
        let pairs = tokens * topK
        precondition(pairs > 0)
        try reserve(pairs: pairs, d: d, f: f, numExperts: numExperts)
        guard let sortedPairs, let work, let rowsIn, let activated, let rowsOut else {
            throw MetalError.bufferAllocationFailed("shortBranch.grouped")
        }
        guard let gateUpPSO = gateUp[edge], let downPSO = down[edge] else {
            throw MetalError.missingFunction("short_branch_qmm edge \(edge)")
        }
        let capacity = Self.workCapacity(pairs: pairs, numExperts: numExperts, edge: edge)

        guard let g = cb.timedComputeEncoder() else { throw MetalError.commandEncoderFailed }
        g.setComputePipelineState(group)
        g.setBuffer(routeIDs, offset: 0, index: 0)
        g.setBuffer(routeWeights, offset: 0, index: 1)
        g.setBuffer(sortedPairs, offset: 0, index: 2)
        g.setBuffer(work, offset: 0, index: 3)
        var pairCount = UInt32(pairs)
        var k = UInt32(topK)
        var experts = UInt32(numExperts)
        var cap = UInt32(capacity)
        var edgeRows = UInt32(edge)
        g.setBytes(&pairCount, length: 4, index: 4)
        g.setBytes(&k, length: 4, index: 5)
        g.setBytes(&experts, length: 4, index: 6)
        g.setBytes(&cap, length: 4, index: 7)
        g.setBytes(&edgeRows, length: 4, index: 8)
        g.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(1024, group.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        g.releaseEncoder()
        try split(&cb, "prefill_split_grouped_plan")

        try encodeRows(
            cb, gather, source: hidden, destination: rowsIn, sortedPairs: sortedPairs,
            pairs: pairs, d: d, last: UInt32(d))
        try split(&cb, "prefill_split_grouped_gather")

        let gateUpParams: [UInt32] = [
            UInt32(f), UInt32(d),
            offsets.gateWOff, offsets.gateSOff, offsets.gateBOff,
            offsets.upWOff, offsets.upSOff, offsets.upBOff, 64,
        ]
        let downParams: [UInt32] = [
            UInt32(d), UInt32(f),
            offsets.downWOff, offsets.downSOff, offsets.downBOff, 0, 0, 0, 64,
        ]
        for (pso, x, y, params, n, columns, role) in [
            (gateUpPSO, rowsIn, activated, gateUpParams, f, Self.gateUpColumns, "gate_up"),
            (downPSO, activated, rowsOut, downParams, d, Self.downColumns, "down"),
        ] {
            guard let enc = cb.timedComputeEncoder() else {
                throw MetalError.commandEncoderFailed
            }
            enc.setComputePipelineState(pso)
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(y, offset: 0, index: 1)
            enc.setBuffer(expertAddresses, offset: 0, index: 2)
            enc.setBuffer(work, offset: 0, index: 3)
            var p = params
            enc.setBytes(&p, length: p.count * 4, index: 4)
            enc.useResources(residentBuffers, usage: .read)
            enc.dispatchThreadgroups(
                MTLSize(width: n / columns, height: capacity, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.releaseEncoder()
            try split(&cb, "prefill_split_grouped_\(role)")
        }

        try encodeRows(
            cb, scatter, source: rowsOut, destination: routePartials, sortedPairs: sortedPairs,
            pairs: pairs, d: d, last: UInt32(topK))
        try split(&cb, "prefill_split_grouped_scatter")
    }

    private func encodeRows(
        _ cb: MTLCommandBuffer, _ pipeline: MTLComputePipelineState,
        source: MTLBuffer, destination: MTLBuffer, sortedPairs: MTLBuffer,
        pairs: Int, d: Int, last: UInt32
    ) throws {
        guard let encoder = cb.timedComputeEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(sortedPairs, offset: 0, index: 1)
        encoder.setBuffer(destination, offset: 0, index: 2)
        var start = UInt32(0)
        var count = UInt32(pairs)
        var width = UInt32(d)
        var trailing = last
        encoder.setBytes(&start, length: 4, index: 3)
        encoder.setBytes(&count, length: 4, index: 4)
        encoder.setBytes(&width, length: 4, index: 5)
        encoder.setBytes(&trailing, length: 4, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: d, height: pairs, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 4, depth: 1))
        encoder.releaseEncoder()
    }
}
