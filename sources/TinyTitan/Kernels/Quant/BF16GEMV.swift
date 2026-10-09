import Metal

/// GEMV over a weight matrix kept at the checkpoint's own bf16.
///
/// Its reason to exist is the 8-bit build: promoting a family to bf16 removes
/// the last of its quantization error, and for the small families that is
/// nearly free -- the seven promoted ones are 2% of the active parameters per
/// token and cost ~108 MB resident. Decode reads resident weights from RAM,
/// not from SSD, so the cost lands where there is bandwidth to spare.
///
/// Shares the affine kernels' launch geometry so `SlotGEMV` can pick between
/// them per tensor without the caller knowing which it got.
final class BF16GEMV {
    private let pipeline: MTLComputePipelineState
    /// One-row weight over many inputs, one simdgroup per input.
    private let oneRowManyX: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pipeline = try context.pipeline(
            "bf16_gemv_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 256)
        self.oneRowManyX = try context.pipeline(
            "bf16_gemv_simd_one_row_many_x",
            constants: [],
            maxTotalThreadsPerThreadgroup: 256)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: UInt32, n: UInt32
    ) throws {
        try encodeRows(
            commandBuffer: commandBuffer,
            weights: weights, weightsOffset: weightsOffset,
            x: x, y: y, rows: .single(xOffset: xOffset, yOffset: yOffset),
            m: m, n: n)
    }

    /// `encode` for several independent rows in one encoder (see `GEMVRows`).
    func encodeRows(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        x: MTLBuffer, y: MTLBuffer, rows: GEMVRows,
        m: UInt32, n: UInt32
    ) throws {
        precondition(
            n.isMultiple(of: 64),
            "bf16 GEMV expects a column count that is a multiple of 64")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        let half = MemoryLayout<Float16>.stride
        if m == 1, rows.count > 1, rows.xRowStride % half == 0, rows.yRowStride % half == 0 {
            // A one-row weight over many inputs (the shared-expert scalar gate
            // across a prefill chunk): one dispatch, not one per input.
            encoder.setComputePipelineState(oneRowManyX)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(x, offset: rows.xOffset, index: 1)
            encoder.setBuffer(y, offset: rows.yOffset, index: 2)
            var params = (
                UInt32(rows.count), n,
                UInt32(rows.xRowStride / half), UInt32(rows.yRowStride / half))
            encoder.setBytes(&params.0, length: 4, index: 3)
            encoder.setBytes(&params.1, length: 4, index: 4)
            encoder.setBytes(&params.2, length: 4, index: 5)
            encoder.setBytes(&params.3, length: 4, index: 6)
            encoder.dispatchThreadgroups(
                MTLSize(width: (rows.count + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            return
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(x, offset: rows.xOffset, index: 1)
        encoder.setBuffer(y, offset: rows.yOffset, index: 2)
        var outputRows = m
        var columns = n
        encoder.setBytes(&outputRows, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&columns, length: MemoryLayout<UInt32>.size, index: 4)
        let rowsPerThreadgroup = 8
        let groups = (Int(m) + rowsPerThreadgroup - 1) / rowsPerThreadgroup
        rows.dispatch(
            encoder, xIndex: 1, yIndex: 2,
            threadgroups: MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 32 * rowsPerThreadgroup,
                height: 1, depth: 1))
        encoder.endEncoding()
    }
}
