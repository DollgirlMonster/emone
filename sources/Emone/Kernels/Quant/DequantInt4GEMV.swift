import Metal

/// MLX-affine INT4 matrix-vector multiplication.
/// Eight SIMD groups process eight output rows per threadgroup.
final class DequantInt4GEMV {
    private struct Shape: Hashable {
        var m: UInt32
        var n: UInt32
    }

    private static let rowsPerThreadgroup = 8
    private static let realDecodeShapes: [Shape] = [
        Shape(m: 4096, n: 2816),
        Shape(m: 2048, n: 2816),
        Shape(m: 2816, n: 4096),
        Shape(m: 8192, n: 2816),
        Shape(m: 1024, n: 2816),
        Shape(m: 2816, n: 8192),
    ]

    private let pipeline: MTLComputePipelineState
    private let twoRowPipeline: MTLComputePipelineState
    private let specializedPipelines: [Shape: MTLComputePipelineState]
    private let specializedTwoRowPipelines: [Shape: MTLComputePipelineState]
    /// `encodeRows` over many inputs in one dispatch, generic and specialized
    /// on the same shapes as `pipeline`, so each input's arithmetic is compiled
    /// exactly as the per-input dispatch's is.
    private let manyXPipeline: MTLComputePipelineState
    private let specializedManyXPipelines: [Shape: MTLComputePipelineState]
    static let manyXBlock = 4

    /// `additionalShapes` compiles extra constant-folded variants for
    /// Qwen 3.6's decode shapes. Measured: an unspecialized
    /// 4096x2048 GEMV runs at 102 GB/s, the specialized one at 141 GB/s.
    init(
        context: MetalContext,
        additionalShapes: [(m: Int, n: Int)] = []
    ) throws {
        self.pipeline = try context.pipeline(
            "dequant_int4_gemv_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)
        self.twoRowPipeline = try context.pipeline(
            "dequant_int4_gemv2_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)
        self.manyXPipeline = try context.pipeline(
            "dequant_int4_gemv_simd_many_x",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)

        let shapes =
            Self.realDecodeShapes
            + additionalShapes.map { Shape(m: UInt32($0.m), n: UInt32($0.n)) }
        var specializedPipelines: [Shape: MTLComputePipelineState] = [:]
        var specializedTwoRowPipelines: [Shape: MTLComputePipelineState] = [:]
        var specializedManyXPipelines: [Shape: MTLComputePipelineState] = [:]
        for shape in shapes {
            specializedManyXPipelines[shape] = try context.pipeline(
                "dequant_int4_gemv_simd_many_x",
                constants: [
                    MetalFunctionConstant(index: 20, value: .uint32(shape.m)),
                    MetalFunctionConstant(index: 21, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 22, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 512)
            specializedPipelines[shape] = try context.pipeline(
                "dequant_int4_gemv_simd",
                constants: [
                    MetalFunctionConstant(index: 20, value: .uint32(shape.m)),
                    MetalFunctionConstant(index: 21, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 22, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 512)
            specializedTwoRowPipelines[shape] = try context.pipeline(
                "dequant_int4_gemv2_simd",
                constants: [
                    MetalFunctionConstant(index: 20, value: .uint32(shape.m)),
                    MetalFunctionConstant(index: 21, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 22, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPipelines = specializedPipelines
        self.specializedTwoRowPipelines = specializedTwoRowPipelines
        self.specializedManyXPipelines = specializedManyXPipelines
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer,
        weightsOffset: Int = 0,
        scales: MTLBuffer,
        scalesOffset: Int = 0,
        biases: MTLBuffer,
        biasesOffset: Int = 0,
        x: MTLBuffer,
        xOffset: Int = 0,
        y: MTLBuffer,
        yOffset: Int = 0,
        m: UInt32,
        n: UInt32
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
        precondition(
            n % UInt32(Quantization.groupSize) == 0,
            "N must be a multiple of \(Quantization.groupSize)")
        // The kernel reads packed weights through a `ushort*`; the repacker
        // guarantees two-byte sub-tensor alignment but not four-byte alignment.
        precondition(
            weightsOffset % 2 == 0,
            "dequant_int4_gemv_simd needs a 2-aligned weightsOffset, got \(weightsOffset)")
        guard let encoder = commandBuffer.timedComputeEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        let half = MemoryLayout<Float16>.stride
        // The body reads x as half4, so each input row must stay 8-byte aligned.
        if rows.count > 1, rows.xRowStride % 8 == 0, rows.xOffset % 8 == 0,
            rows.yRowStride % half == 0
        {
            encoder.setComputePipelineState(
                specializedManyXPipelines[Shape(m: m, n: n)] ?? manyXPipeline)
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
                    width: (Int(m) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
                    height: (rows.count + Self.manyXBlock - 1) / Self.manyXBlock, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: 32 * Self.rowsPerThreadgroup, height: 1, depth: 1))
            encoder.endEncoding()
            return
        }
        encoder.setComputePipelineState(
            specializedPipelines[Shape(m: m, n: n)] ?? pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: rows.xOffset, index: 3)
        encoder.setBuffer(y, offset: rows.yOffset, index: 4)
        var mValue = m
        var nValue = n
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)

        let threadgroupSize = MTLSize(
            width: 32 * Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        let threadgroupCount = MTLSize(
            width: (Int(m) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        rows.dispatch(
            encoder, xIndex: 3, yIndex: 4,
            threadgroups: threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
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
        precondition(n % UInt32(Quantization.groupSize) == 0)
        precondition(weightsOffset % 2 == 0)
        try encodeTwoRowsShared(
            commandBuffer: commandBuffer,
            pipeline: specializedTwoRowPipelines[Shape(m: m, n: n)] ?? twoRowPipeline,
            weights: weights, weightsOffset: weightsOffset,
            scales: scales, scalesOffset: scalesOffset,
            biases: biases, biasesOffset: biasesOffset,
            x: x, xOffset: xOffset,
            y: y, yOffset: yOffset,
            m: m, n: n)
    }
}
