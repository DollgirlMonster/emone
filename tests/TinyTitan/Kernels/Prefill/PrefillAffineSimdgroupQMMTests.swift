import Foundation
import Metal
import Testing

@testable import TinyTitan

/// The simdgroup-matrix QMM computes what the scalar prefill QMM computes, in
/// a different order: both dequantize the weights in float and accumulate in
/// float, so the outputs may differ only by the rounding of those sums -- a
/// half-precision ulp or two, not the 2% the MPP path is allowed. Shapes cover
/// the partial edge tiles in both T and N, both widths, and a production K.
@Suite struct PrefillAffineSimdgroupQMMTests {
    private struct LCG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 33
        }
        mutating func unit() -> Float { Float(next() % 20_001) / 10_000 - 1 }
    }

    private static func bf16(_ value: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: value.bitPattern >> 16)
    }

    private static func buffer<T>(_ ctx: MetalContext, _ values: [T]) throws -> MTLBuffer {
        let made = values.withUnsafeBytes { raw -> MTLBuffer? in
            guard let base = raw.baseAddress else { return nil }
            return ctx.device.makeBuffer(bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let made else { throw MetalError.commandEncoderFailed }
        return made
    }

    struct Case: Sendable, CustomTestStringConvertible {
        let t: Int, n: Int, k: Int, bits: Int
        var testDescription: String { "T\(t) N\(n) K\(k) \(bits)-bit" }
    }

    @Test(arguments: [
        Case(t: 37, n: 45, k: 128, bits: 4),
        Case(t: 64, n: 64, k: 256, bits: 4),
        Case(t: 5, n: 33, k: 64, bits: 8),
        Case(t: 70, n: 96, k: 2_560, bits: 4),
        Case(t: 33, n: 40, k: 640, bits: 8),
    ])
    func matchesTheScalarKernel(_ shape: Case) throws {
        let ctx = try MetalContext()
        let scalar = try PrefillInt4QMM(context: ctx, weightBits: shape.bits)
        let simdgroup = try PrefillAffineSimdgroupQMM(context: ctx)
        #expect(simdgroup.accepts(bits: shape.bits, k: shape.k))

        var rng = LCG(state: UInt64(shape.t * 131 + shape.n * 7 + shape.k + shape.bits))
        let groups = shape.k / Quantization.groupSize
        let rowBytes = shape.k * shape.bits / 8
        let weights = try Self.buffer(
            ctx, (0..<(shape.n * rowBytes)).map { _ in UInt8(rng.next() & 0xFF) })
        let scaleSize: Float = shape.bits == 4 ? 0.02 : 0.002
        let scales = try Self.buffer(
            ctx, (0..<(shape.n * groups)).map { _ in Self.bf16(scaleSize * (0.5 + abs(rng.unit()))) })
        let biases = try Self.buffer(
            ctx, (0..<(shape.n * groups)).map { _ in Self.bf16(rng.unit() * 0.1) })
        let x = try Self.buffer(
            ctx, (0..<(shape.t * shape.k)).map { _ in Float16(rng.unit()) })

        let outBytes = shape.t * shape.n * MemoryLayout<Float16>.stride
        guard let reference = ctx.device.makeBuffer(length: outBytes, options: .storageModeShared),
            let candidate = ctx.device.makeBuffer(length: outBytes, options: .storageModeShared),
            let cb = ctx.queue.makeCommandBuffer()
        else { throw MetalError.commandEncoderFailed }
        memset(candidate.contents(), 0x7C, outBytes)
        try scalar.encode(
            commandBuffer: cb, weights: weights, scales: scales, biases: biases,
            x: x, y: reference, t: shape.t, n: shape.n, k: shape.k)
        try simdgroup.encode(
            commandBuffer: cb, weights: weights, scales: scales, biases: biases,
            x: x, y: candidate, t: shape.t, n: shape.n, k: shape.k, bits: shape.bits)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let count = shape.t * shape.n
        let a = UnsafeBufferPointer(
            start: reference.contents().bindMemory(to: Float16.self, capacity: count), count: count)
        let b = UnsafeBufferPointer(
            start: candidate.contents().bindMemory(to: Float16.self, capacity: count), count: count)
        var worst: Float = 0
        var maxRef: Float = 0
        for i in 0..<count {
            let ref = Float(a[i])
            maxRef = max(maxRef, abs(ref))
            // Two half ulps at the reference's magnitude, with a floor near zero.
            let allowed = max(Float(1e-3), 2 * Float(Float16(abs(ref)).ulp))
            worst = max(worst, abs(Float(b[i]) - ref) / allowed)
        }
        #expect(maxRef > 0.1, "the reference output is too small to prove anything")
        #expect(worst <= 1, "worst element is \(worst)x its allowance (max |ref| \(maxRef))")
    }
}
