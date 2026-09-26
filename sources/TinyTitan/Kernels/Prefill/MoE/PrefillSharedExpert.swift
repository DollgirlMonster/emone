import Foundation
import Metal

final class PrefillSharedExpert {
    private let shared: SharedExpertRuntime
    /// The whole-chunk path: gate and up as two MPP GEMMs over every token, one
    /// elementwise activation, down as a third GEMM. Without it the block runs
    /// the decode path once per token through a one-row scratch, so every
    /// token's four dispatches wait on the previous token's.
    private let batched: (gemm: BatchedGEMM, activation: MTLComputePipelineState)?

    /// Which QMM serves the three GEMMs.
    private enum BatchedGEMM {
        case mpp(MPPPrefillInt4QMM)
        case simdgroup(PrefillAffineSimdgroupQMM, bits: Int)
    }

    /// Whether `encodeBlockBatched` can take a chunk at all on this GPU.
    var batchedAvailable: Bool { batched != nil }

    init(
        context: MetalContext, weightBits: Int = 8, siluActivation: Bool = false,
        batchedMPP: Bool = false, batchedSimdgroup: Bool = false
    ) throws {
        self.shared = try SharedExpertRuntime(
            context: context,
            weightBits: weightBits,
            siluActivation: siluActivation)
        // MPP serves the 4- and 8-bit affine layouts (its init traps on any
        // other width), and is optional on older GPUs.
        let activation = siluActivation ? "silu_mul_fp16" : "gelu_mul_fp16"
        if batchedSimdgroup, [4, 8].contains(weightBits) {
            self.batched = (
                BatchedGEMM.simdgroup(try PrefillAffineSimdgroupQMM(context: context), bits: weightBits),
                try context.pipeline(activation)
            )
        } else if batchedMPP, [4, 8].contains(weightBits) {
            let mpp = MPPPrefillInt4QMM(context: context, weightBits: weightBits)
            self.batched =
                mpp.isAvailable ? (BatchedGEMM.mpp(mpp), try context.pipeline(activation)) : nil
        } else {
            self.batched = nil
        }
    }

    /// The block as three GEMMs over the chunk. Returns false, having written
    /// nothing the caller depends on, when this chunk cannot take the batched
    /// path; the caller then runs `encodeBlock`, which recomputes `y` whole.
    /// The GEMMs sum in a different order from the per-token GEMVs, so the
    /// output is close, not bit-identical.
    func encodeBlockBatched(
        commandBuffer cb: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        gate: SharedExpertInt8Proj, up: SharedExpertInt8Proj, down: SharedExpertInt8Proj,
        scratchGate: MTLBuffer, scratchUp: MTLBuffer, scratchAct: MTLBuffer,
        queryCount: Int, d: Int, intermediate: Int,
        xStrideElements: Int, yStrideElements: Int
    ) throws -> Bool {
        let halfBytes = MemoryLayout<Float16>.stride
        let chunkBytes = queryCount * intermediate * halfBytes
        guard let batched, queryCount > 0,
            xStrideElements == d, yStrideElements == d,
            gate.rows == UInt32(intermediate), gate.cols == UInt32(d),
            up.rows == UInt32(intermediate), up.cols == UInt32(d),
            down.rows == UInt32(d), down.cols == UInt32(intermediate),
            scratchGate.length >= chunkBytes, scratchUp.length >= chunkBytes,
            scratchAct.length >= chunkBytes
        else { return false }
        func project(
            _ p: SharedExpertInt8Proj, _ input: MTLBuffer, _ inputOffset: Int,
            _ output: MTLBuffer, _ outputOffset: Int, rows: Int, columns: Int
        ) throws -> Bool {
            switch batched.gemm {
            case .mpp(let mpp):
                return try mpp.encode(
                    commandBuffer: cb,
                    weights: p.weights, weightsOffset: p.weightsOffset,
                    scales: p.scales, scalesOffset: p.scalesOffset,
                    biases: p.biases, biasesOffset: p.biasesOffset,
                    x: input, xOffset: inputOffset,
                    y: output, yOffset: outputOffset,
                    m: queryCount, n: rows, k: columns) == .affineThreadgroupF16
            case .simdgroup(let qmm, let bits):
                guard qmm.accepts(bits: bits, k: columns) else { return false }
                try qmm.encode(
                    commandBuffer: cb,
                    weights: p.weights, weightsOffset: p.weightsOffset,
                    scales: p.scales, scalesOffset: p.scalesOffset,
                    biases: p.biases, biasesOffset: p.biasesOffset,
                    x: input, xOffset: inputOffset,
                    y: output, yOffset: outputOffset,
                    t: queryCount, n: rows, k: columns, bits: bits)
                return true
            }
        }
        guard try project(gate, x, xOffset, scratchGate, 0, rows: intermediate, columns: d),
            try project(up, x, xOffset, scratchUp, 0, rows: intermediate, columns: d)
        else { return false }
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        // The per-token activation kernel, over every token's row at once: it is
        // elementwise, so one dispatch of tokens x intermediate is the same math.
        encoder.setComputePipelineState(batched.activation)
        encoder.setBuffer(scratchGate, offset: 0, index: 0)
        encoder.setBuffer(scratchUp, offset: 0, index: 1)
        encoder.setBuffer(scratchAct, offset: 0, index: 2)
        var count = UInt32(queryCount * intermediate)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(batched.activation.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: queryCount * intermediate, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        return try project(down, scratchAct, 0, y, yOffset, rows: d, columns: intermediate)
    }

    func encodeBlock(
        commandBuffer cb: MTLCommandBuffer,
        x: MTLBuffer,
        xOffset: Int = 0,
        y: MTLBuffer,
        yOffset: Int = 0,
        gate: SharedExpertInt8Proj,
        up: SharedExpertInt8Proj,
        down: SharedExpertInt8Proj,
        scratchGate: MTLBuffer,
        scratchGateOffset: Int = 0,
        scratchUp: MTLBuffer,
        scratchUpOffset: Int = 0,
        scratchAct: MTLBuffer,
        scratchActOffset: Int = 0,
        queryCount: Int,
        d: Int,
        intermediate: Int,
        xStrideElements: Int,
        yStrideElements: Int
    ) throws {
        precondition(queryCount >= 0, "queryCount must be non-negative")
        precondition(d > 0, "d must be positive")
        precondition(intermediate > 0, "intermediate must be positive")
        precondition(xStrideElements >= d, "x stride is too small")
        precondition(yStrideElements >= d, "y stride is too small")
        guard gate.rows == UInt32(intermediate), gate.cols == UInt32(d),
            up.rows == UInt32(intermediate), up.cols == UInt32(d),
            down.rows == UInt32(d), down.cols == UInt32(intermediate)
        else {
            throw SharedExpertInt8Error.dimensionMismatch(
                "expected gate/up=(\(intermediate),\(d)) down=(\(d),\(intermediate))")
        }

        let halfBytes = MemoryLayout<Float16>.stride
        for row in 0..<queryCount {
            try shared.encode(
                commandBuffer: cb,
                x: x,
                xOffset: xOffset + row * xStrideElements * halfBytes,
                gate: gate,
                up: up,
                down: down,
                y: y,
                yOffset: yOffset + row * yStrideElements * halfBytes,
                scratchGate: scratchGate,
                scratchGateOffset: scratchGateOffset,
                scratchUp: scratchUp,
                scratchUpOffset: scratchUpOffset,
                scratchAct: scratchAct,
                scratchActOffset: scratchActOffset)
        }
    }
}
