import Foundation
import Metal

@frozen
public struct PrefillTokenExpertPair: Equatable, Sendable {
    public var token: UInt32
    public var expert: UInt32
    public var rank: UInt32
    public var weightBitsAndReserved: UInt32

    public init(token: UInt32, expert: UInt32, rank: UInt32, weight: Float16) {
        self.token = token
        self.expert = expert
        self.rank = rank
        self.weightBitsAndReserved = UInt32(weight.bitPattern)
    }

    public init(token: UInt32, expert: UInt32, rank: UInt32, weightBitsAndReserved: UInt32) {
        self.token = token
        self.expert = expert
        self.rank = rank
        self.weightBitsAndReserved = weightBitsAndReserved
    }

    public var weight: Float16 {
        Float16(bitPattern: UInt16(truncatingIfNeeded: weightBitsAndReserved))
    }
}

final class PrefillRouter {
    private let pso: MTLComputePipelineState
    /// `prefill_router_rows`: 8 tokens per threadgroup, the same scores and
    /// selection as `prefill_router_block` (one token per threadgroup) at a
    /// fraction of the cache traffic. On unless a test asks for the old one.
    private let rowsPSO: MTLComputePipelineState
    var useRowsKernel = true
    /// `prefill_router_topk_logits`: selection over logits the tiled QMM
    /// wrote (see `encodeFromLogits`).
    private let topKLogitsPSO: MTLComputePipelineState
    /// `prefill_dense_qmm_64x64_f32out` at the router's width.
    private let logitsPSO: MTLComputePipelineState?
    /// `[tokens, experts]` float logits, grown to the largest chunk seen.
    private var logits: MTLBuffer?
    static let rowsPerThreadgroup = 8
    private let device: MTLDevice
    private let weightBits: Int

    init(context: MetalContext, weightBits: Int = 8) throws {
        // 16 means the router is unquantized bf16; the shader branches on it,
        // and must stay in step with moe.metal's decode-side router.
        precondition([4, 8, 16].contains(weightBits))
        self.pso = try context.pipeline(
            "prefill_router_block",
            constants: [
                MetalFunctionConstant(
                    index: 79,
                    value: .uint32(UInt32(weightBits)))
            ])
        self.rowsPSO = try context.pipeline(
            "prefill_router_rows",
            constants: [MetalFunctionConstant(index: 79, value: .uint32(UInt32(weightBits)))])
        self.topKLogitsPSO = try context.pipeline("prefill_router_topk_logits")
        self.logitsPSO =
            weightBits == 16
            ? nil
            : try context.pipeline(
                "prefill_dense_qmm_64x64_f32out",
                constants: [MetalFunctionConstant(index: 78, value: .uint32(UInt32(weightBits)))])
        self.device = context.device
        self.weightBits = weightBits
    }

    func encodeBlock(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer,
        weightsOffset: Int = 0,
        scales: MTLBuffer,
        scalesOffset: Int = 0,
        biases: MTLBuffer,
        biasesOffset: Int = 0,
        hidden: MTLBuffer,
        hiddenOffset: Int = 0,
        effectiveScale: MTLBuffer,
        effectiveScaleOffset: Int = 0,
        perExpertScale: MTLBuffer,
        perExpertScaleOffset: Int = 0,
        outIndices: MTLBuffer,
        outIndicesOffset: Int = 0,
        outWeights: MTLBuffer,
        outWeightsOffset: Int = 0,
        queryCount: UInt32,
        numExperts: UInt32,
        d: UInt32,
        topK: UInt32,
        hiddenStrideElements: UInt32
    ) throws {
        precondition(queryCount > 0, "queryCount must be positive")
        // Must track kPrefillRouterMaxExperts in prefill.metal: past it the
        // kernel clamps rather than failing, which routes as if the surplus
        // experts did not exist.
        precondition(numExperts <= 512, "numExperts > 512 is not supported")
        precondition(topK > 0 && topK <= 64, "topK must be in 1...64")
        precondition(
            d % UInt32(Quantization.groupSize) == 0,
            "D must be a multiple of \(Quantization.groupSize)")
        precondition(hiddenStrideElements >= d, "hidden stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(useRowsKernel ? rowsPSO : pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(hidden, offset: hiddenOffset, index: 3)
        enc.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 4)
        enc.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 5)
        enc.setBuffer(outIndices, offset: outIndicesOffset, index: 6)
        enc.setBuffer(outWeights, offset: outWeightsOffset, index: 7)
        var tVar = queryCount
        var neVar = numExperts
        var dVar = d
        var topKVar = topK
        var strideVar = hiddenStrideElements
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&neVar, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&topKVar, length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&strideVar, length: MemoryLayout<UInt32>.size, index: 12)
        if useRowsKernel {
            let tiles = (Int(queryCount) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
            enc.dispatchThreadgroups(
                MTLSize(width: tiles, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            let tgWidth = min(max(Int(numExperts), 32), pso.maxTotalThreadsPerThreadgroup)
            enc.dispatchThreadgroups(
                MTLSize(width: Int(queryCount), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        }
        enc.endEncoding()
    }

    /// The router as two passes: float logits through the tiled simdgroup
    /// QMM (`prefill_dense_qmm_64x64_f32out`, ~7 TFLOPS), then
    /// `prefill_router_topk_logits`. Returns false, encoding nothing, when the
    /// shape does not fit 64 x 64 tiles or the router is bf16; the caller then
    /// runs `encodeBlock`. The effective scale must be ones for the logits to
    /// need no input scaling -- the runner's is -- and the caller says so with
    /// `unitInputScale`.
    func encodeFromLogits(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int,
        scales: MTLBuffer, scalesOffset: Int,
        biases: MTLBuffer, biasesOffset: Int,
        hidden: MTLBuffer,
        unitInputScale: Bool,
        perExpertScale: MTLBuffer, perExpertScaleOffset: Int,
        outIndices: MTLBuffer, outWeights: MTLBuffer,
        queryCount: Int, numExperts: Int, d: Int, topK: Int,
        hiddenStrideElements: Int
    ) throws -> Bool {
        guard unitInputScale, let logitsPSO, hiddenStrideElements == d,
            numExperts <= 512, numExperts % 64 == 0, topK > 0, topK <= 64,
            d % Quantization.groupSize == 0, queryCount > 0
        else { return false }
        let bytes = queryCount * numExperts * MemoryLayout<Float>.stride
        if logits == nil || logits!.length < bytes {
            guard let made = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
                throw MetalError.commandEncoderFailed
            }
            made.label = "prefill.routerLogits"
            logits = made
        }
        guard let logits else { return false }
        guard let gemm = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        gemm.setComputePipelineState(logitsPSO)
        gemm.setBuffer(hidden, offset: 0, index: 0)
        gemm.setBuffer(logits, offset: 0, index: 1)
        gemm.setBuffer(weights, offset: weightsOffset, index: 2)
        var m = UInt32(queryCount)
        gemm.setBytes(&m, length: 4, index: 3)
        var params: [UInt32] = [
            UInt32(numExperts), UInt32(d), 0, 0, 0, 0, 0, 0, UInt32(Quantization.groupSize),
        ]
        gemm.setBytes(&params, length: params.count * 4, index: 4)
        gemm.setBuffer(scales, offset: scalesOffset, index: 5)
        gemm.setBuffer(biases, offset: biasesOffset, index: 6)
        gemm.dispatchThreadgroups(
            MTLSize(width: numExperts / 64, height: (queryCount + 63) / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        gemm.endEncoding()
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(topKLogitsPSO)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 1)
        enc.setBuffer(outIndices, offset: 0, index: 2)
        enc.setBuffer(outWeights, offset: 0, index: 3)
        var topKParams = (UInt32(queryCount), UInt32(numExperts), UInt32(topK))
        enc.setBytes(&topKParams.0, length: 4, index: 4)
        enc.setBytes(&topKParams.1, length: 4, index: 5)
        enc.setBytes(&topKParams.2, length: 4, index: 6)
        let width = 64
        enc.dispatchThreadgroups(
            MTLSize(width: (queryCount + width - 1) / width, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
        return true
    }

    static func makeTokenExpertPairs(
        indices: [UInt32],
        weights: [Float16],
        queryCount: Int,
        topK: Int
    ) -> [PrefillTokenExpertPair] {
        precondition(queryCount >= 0, "queryCount must be non-negative")
        precondition(topK >= 0, "topK must be non-negative")
        precondition(indices.count == queryCount * topK, "indices count mismatch")
        precondition(weights.count == queryCount * topK, "weights count mismatch")
        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(indices.count)
        for token in 0..<queryCount {
            for rank in 0..<topK {
                let i = token * topK + rank
                pairs.append(
                    PrefillTokenExpertPair(
                        token: UInt32(token),
                        expert: indices[i],
                        rank: UInt32(rank),
                        weight: weights[i]))
            }
        }
        return pairs
    }
}
