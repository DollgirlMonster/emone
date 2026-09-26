import Metal

/// Where a run of independent GEMVs reads and writes: row `r` reads x at
/// `xOffset + r * xRowStride` and writes y at `yOffset + r * yRowStride`
/// (bytes). One row is an ordinary single GEMV.
struct GEMVRows {
    let xOffset: Int
    let xRowStride: Int
    let yOffset: Int
    let yRowStride: Int
    let count: Int

    static func single(xOffset: Int, yOffset: Int) -> GEMVRows {
        GEMVRows(xOffset: xOffset, xRowStride: 0, yOffset: yOffset, yRowStride: 0, count: 1)
    }

    /// Dispatch the same grid once per row in the encoder the caller already
    /// set up, moving only the x and y bindings between dispatches.
    ///
    /// A prefill chunk used to issue one GEMV per token, each in its own
    /// compute encoder -- 4,096 encoder boundaries per projection per layer.
    /// Here the pipeline, weights, constants and grid are exactly what each
    /// separate encoder had, so every row computes the same bits; only the
    /// encoder boundaries between them are gone.
    func dispatch(
        _ encoder: MTLComputeCommandEncoder,
        xIndex: Int, yIndex: Int,
        threadgroups: MTLSize, threadsPerThreadgroup: MTLSize
    ) {
        for row in 0..<count {
            if row > 0 {
                encoder.setBufferOffset(xOffset + row * xRowStride, index: xIndex)
                encoder.setBufferOffset(yOffset + row * yRowStride, index: yIndex)
            }
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        }
    }
}
