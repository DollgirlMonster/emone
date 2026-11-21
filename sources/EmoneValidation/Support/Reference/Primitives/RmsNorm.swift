import Accelerate
import Foundation

/// FP32 RMSNorm reference computed with Accelerate vectorized primitives.
///
/// Deliberately reaches the same answer via a *different* sequence of
/// operations than the Metal kernel: `vDSP_svesq` for the sum-of-squares,
/// `vDSP_vmul` + `vDSP_vsmul` for the per-element scale. The kernel uses a
/// per-thread block reduction with `simd_sum`; this reference uses
/// Accelerate's pipeline. Different summation tree, different rounding chain,
/// same math — which is what makes this a useful comparator.
public enum RmsNormRef {
    /// y[i] = x[i] * weight[i] * 1 / sqrt(mean(x^2) + eps)
    public static func apply(x: [Float], weight: [Float], eps: Float) -> [Float] {
        precondition(x.count == weight.count, "x and weight must match length")
        let d = x.count
        // Nothing to normalise, and no base address to read: the empty case
        // used to reach the force unwrap below.
        guard d > 0 else { return [] }

        var sumSq: Float = 0
        x.withUnsafeBufferPointer { pBuffer in
            guard let p = pBuffer.baseAddress else { return }
            vDSP_svesq(p, 1, &sumSq, vDSP_Length(d))
        }
        let invRms = 1.0 / (sumSq / Float(d) + eps).squareRoot()

        var y = [Float](repeating: 0, count: d)
        x.withUnsafeBufferPointer { pxBuffer in
            weight.withUnsafeBufferPointer { pwBuffer in
                y.withUnsafeMutableBufferPointer { pyBuffer in
                    guard let px = pxBuffer.baseAddress,
                        let pw = pwBuffer.baseAddress,
                        let py = pyBuffer.baseAddress
                    else { return }
                    vDSP_vmul(px, 1, pw, 1, py, 1, vDSP_Length(d))
                    var s = invRms
                    vDSP_vsmul(py, 1, &s, py, 1, vDSP_Length(d))
                }
            }
        }
        return y
    }
}
