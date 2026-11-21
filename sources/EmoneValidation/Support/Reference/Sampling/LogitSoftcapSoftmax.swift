import Accelerate
import Foundation

/// FP32 reference for `softmax(softcap * tanh(x / softcap))`.
///
/// The kernel runs a single-pass online safe softmax with the softcap
/// fused into the running-max update. This reference is deliberately
/// *two-pass*: apply the softcap to a temp buffer, find the max, subtract,
/// exp via `vForce.exp`, sum via `vDSP_sve`, divide. Different staging,
/// different summation order — any bug that depends on the kernel's online
/// (m, d) merge logic won't replicate here.
///
/// Softcap is `c * tanh(x / c)` with c=30.0 by default.
public enum LogitSoftcapSoftmaxRef {
    public static func apply(x: [Float], softcap: Float) -> [Float] {
        let v = x.count
        // An empty array has no vDSP base address, and there is nothing to
        // softmax: the old force unwrap made that case a crash rather than an
        // empty distribution.
        guard v > 0 else { return [] }
        let invC = 1.0 / softcap

        // 1. Apply softcap: y = softcap * tanh(x / softcap)
        //    Two-step: y = x * invC; y = tanh(y); y = y * softcap.
        var y = [Float](repeating: 0, count: v)
        var s = invC
        x.withUnsafeBufferPointer { pxBuffer in
            y.withUnsafeMutableBufferPointer { pyBuffer in
                guard let px = pxBuffer.baseAddress, let py = pyBuffer.baseAddress else { return }
                vDSP_vsmul(px, 1, &s, py, 1, vDSP_Length(v))
            }
        }
        y = vForce.tanh(y)
        var c = softcap
        y.withUnsafeMutableBufferPointer { pyBuffer in
            guard let py = pyBuffer.baseAddress else { return }
            vDSP_vsmul(py, 1, &c, py, 1, vDSP_Length(v))
        }

        // 2. Numerically stable softmax: subtract max, exp, divide by sum.
        var mx: Float = -.infinity
        y.withUnsafeBufferPointer { pyBuffer in
            guard let py = pyBuffer.baseAddress else { return }
            vDSP_maxv(py, 1, &mx, vDSP_Length(v))
        }
        var negMax = -mx
        y.withUnsafeMutableBufferPointer { pyBuffer in
            guard let py = pyBuffer.baseAddress else { return }
            vDSP_vsadd(py, 1, &negMax, py, 1, vDSP_Length(v))
        }
        y = vForce.exp(y)

        var sum: Float = 0
        y.withUnsafeBufferPointer { pyBuffer in
            guard let py = pyBuffer.baseAddress else { return }
            vDSP_sve(py, 1, &sum, vDSP_Length(v))
        }
        // An all-zero exponent vector (sum == 0) has no valid distribution;
        // return a zero distribution instead of dividing by zero.
        guard sum > 0 else {
            return [Float](repeating: 0, count: v)
        }
        var invSum = 1.0 / sum
        y.withUnsafeMutableBufferPointer { pyBuffer in
            guard let py = pyBuffer.baseAddress else { return }
            vDSP_vsmul(py, 1, &invSum, py, 1, vDSP_Length(v))
        }
        return y
    }
}
