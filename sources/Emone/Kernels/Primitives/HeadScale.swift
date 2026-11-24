import Foundation
import Metal

/// Per-head output scale (Emone-bbm.3): `x[t, h, :] *= scale[h]` on an
/// attention output before `o_proj`, in place. A weight patch uses it to
/// strengthen or weaken whole heads -- the same as scaling the head's o_proj
/// columns, which is rank `headDim` and so out of reach of a low-rank delta.
final class HeadScale {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        pso = try context.pipeline("head_scale_fp16")
    }

    /// `x` holds `tokens` contiguous rows of `heads * dim` fp16 values;
    /// `scale` holds `heads` floats at `scaleOffset`.
    func encode(
        commandBuffer: MTLCommandBuffer, x: MTLBuffer, scale: MTLBuffer, scaleOffset: Int,
        heads: Int, dim: Int, tokens: Int
    ) throws {
        let total = tokens * heads * dim
        guard total > 0 else { return }
        guard let encoder = commandBuffer.timedComputeEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        var h = UInt32(heads)
        var d = UInt32(dim)
        var n = UInt32(total)
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(scale, offset: scaleOffset, index: 1)
        encoder.setBytes(&h, length: 4, index: 2)
        encoder.setBytes(&d, length: 4, index: 3)
        encoder.setBytes(&n, length: 4, index: 4)
        let width = min(pso.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
