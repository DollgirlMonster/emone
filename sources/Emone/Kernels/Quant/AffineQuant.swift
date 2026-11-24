import Metal

/// K27: shared two-row GEMV encode used by the MTP adapter in both
/// `AffineQuantGEMV` and `DequantInt4GEMV` (previously copy-pasted). The
/// two-row kernels share the buffer layout and dispatch; only the pipeline
/// selection and the alignment preconditions differ, and those stay at the
/// call sites.
func encodeTwoRowsShared(
    commandBuffer: MTLCommandBuffer,
    pipeline: MTLComputePipelineState,
    weights: MTLBuffer, weightsOffset: Int,
    scales: MTLBuffer, scalesOffset: Int,
    biases: MTLBuffer, biasesOffset: Int,
    x: MTLBuffer, xOffset: Int,
    y: MTLBuffer, yOffset: Int,
    m: UInt32, n: UInt32
) throws {
    guard let encoder = commandBuffer.timedComputeEncoder() else {
        throw MetalError.commandEncoderFailed
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(weights, offset: weightsOffset, index: 0)
    encoder.setBuffer(scales, offset: scalesOffset, index: 1)
    encoder.setBuffer(biases, offset: biasesOffset, index: 2)
    encoder.setBuffer(x, offset: xOffset, index: 3)
    encoder.setBuffer(y, offset: yOffset, index: 4)
    var rows = m
    var columns = n
    encoder.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 5)
    encoder.setBytes(&columns, length: MemoryLayout<UInt32>.size, index: 6)
    let rowsPerThreadgroup = 8
    encoder.dispatchThreadgroups(
        MTLSize(
            width: (Int(m) + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(
            width: 32 * rowsPerThreadgroup,
            height: 1, depth: 1))
    encoder.releaseEncoder()
}

final class AffineQuantGEMV {
    let weightBits: Int
    private let pipeline: MTLComputePipelineState
    private let twoRowPipeline: MTLComputePipelineState
    /// Many inputs in one dispatch, 4 per simdgroup pass (see `encodeRows`).
    private let manyXPipeline: MTLComputePipelineState
    static let manyXBlock = 4

    init(context: MetalContext, weightBits: Int) throws {
        precondition([4, 8].contains(weightBits))
        self.weightBits = weightBits
        self.pipeline = try context.pipeline(
            "affine_quant_gemv_simd",
            constants: [
                MetalFunctionConstant(
                    index: 100,
                    value: .uint32(UInt32(weightBits)))
            ],
            maxTotalThreadsPerThreadgroup: 256)
        self.manyXPipeline = try context.pipeline(
            "affine_quant_gemv_simd_many_x",
            constants: [
                MetalFunctionConstant(
                    index: 100,
                    value: .uint32(UInt32(weightBits)))
            ],
            maxTotalThreadsPerThreadgroup: 256)
        self.twoRowPipeline = try context.pipeline(
            "affine_quant_gemv2_simd",
            constants: [
                MetalFunctionConstant(
                    index: 100,
                    value: .uint32(UInt32(weightBits)))
            ],
            maxTotalThreadsPerThreadgroup: 256)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: UInt32, n: UInt32
    ) throws {
        try encodeRows(
            commandBuffer: commandBuffer,
            weights: weights, weightsOffset: weightsOffset,
            scales: scales, scalesOffset: scalesOffset,
            biases: biases, biasesOffset: biasesOffset,
            x: x, y: y, rows: .single(xOffset: xOffset, yOffset: yOffset),
            m: m, n: n)
    }

    /// `encode` for several independent rows in one encoder (see `GEMVRows`).
    func encodeRows(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, y: MTLBuffer, rows: GEMVRows,
        m: UInt32, n: UInt32
    ) throws {
        precondition(n.isMultiple(of: UInt32(Quantization.groupSize)))
        precondition(weightsOffset.isMultiple(of: MemoryLayout<UInt32>.alignment))
        guard let encoder = commandBuffer.timedComputeEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        let half = MemoryLayout<Float16>.stride
        if rows.count > 1, rows.xRowStride % half == 0, rows.yRowStride % half == 0 {
            // Every input in one dispatch, each weight pair unpacked once per
            // 4 inputs; the same per-(row, input) arithmetic, so the same bits.
            encoder.setComputePipelineState(manyXPipeline)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(scales, offset: scalesOffset, index: 1)
            encoder.setBuffer(biases, offset: biasesOffset, index: 2)
            encoder.setBuffer(x, offset: rows.xOffset, index: 3)
            encoder.setBuffer(y, offset: rows.yOffset, index: 4)
            var params = (
                m, n, UInt32(rows.count), UInt32(rows.xRowStride / half),
                UInt32(rows.yRowStride / half)
            )
            encoder.setBytes(&params.0, length: 4, index: 5)
            encoder.setBytes(&params.1, length: 4, index: 6)
            encoder.setBytes(&params.2, length: 4, index: 7)
            encoder.setBytes(&params.3, length: 4, index: 8)
            encoder.setBytes(&params.4, length: 4, index: 9)
            encoder.dispatchThreadgroups(
                MTLSize(
                    width: (Int(m) + 7) / 8,
                    height: (rows.count + Self.manyXBlock - 1) / Self.manyXBlock, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.releaseEncoder()
            return
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: rows.xOffset, index: 3)
        encoder.setBuffer(y, offset: rows.yOffset, index: 4)
        var outputRows = m
        var columns = n
        encoder.setBytes(&outputRows, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&columns, length: MemoryLayout<UInt32>.size, index: 6)
        let rowsPerThreadgroup = 8
        rows.dispatch(
            encoder, xIndex: 3, yIndex: 4,
            threadgroups: MTLSize(
                width: (Int(m) + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 32 * rowsPerThreadgroup,
                height: 1, depth: 1))
        encoder.releaseEncoder()
    }

    func encodeTwoRows(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: UInt32, n: UInt32
    ) throws {
        precondition(n.isMultiple(of: UInt32(Quantization.groupSize)))
        precondition(weightsOffset.isMultiple(of: MemoryLayout<UInt32>.alignment))
        try encodeTwoRowsShared(
            commandBuffer: commandBuffer,
            pipeline: twoRowPipeline,
            weights: weights, weightsOffset: weightsOffset,
            scales: scales, scalesOffset: scalesOffset,
            biases: biases, biasesOffset: biasesOffset,
            x: x, xOffset: xOffset,
            y: y, yOffset: yOffset,
            m: m, n: n)
    }
}

final class AffineQuantEmbeddingLookup {
    let weightBits: Int
    private let pipeline: MTLComputePipelineState

    init(context: MetalContext, weightBits: Int) throws {
        precondition([4, 8].contains(weightBits))
        self.weightBits = weightBits
        self.pipeline = try context.pipeline(
            "affine_quant_embedding_lookup",
            constants: [
                MetalFunctionConstant(
                    index: 100,
                    value: .uint32(UInt32(weightBits)))
            ])
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        table: MTLBuffer, tableOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        tokenId: UInt32, d: UInt32, outScale: Float,
        vocab: UInt32
    ) throws {
        precondition(d.isMultiple(of: UInt32(Quantization.groupSize)))
        precondition(tableOffset.isMultiple(of: MemoryLayout<UInt32>.alignment))
        guard let encoder = commandBuffer.timedComputeEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(table, offset: tableOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var token = tokenId
        var dimension = d
        var scale = outScale
        var vocabCount = vocab
        encoder.setBytes(&token, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 6)
        encoder.setBytes(&vocabCount, length: MemoryLayout<UInt32>.size, index: 7)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: width,
                height: 1, depth: 1))
        encoder.releaseEncoder()
    }
}
