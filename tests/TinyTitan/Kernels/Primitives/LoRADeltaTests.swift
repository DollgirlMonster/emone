import Foundation
import Metal
import Testing

@testable import TinyTitan

/// `LoRADelta` against a CPU reference: `y += scale * (x A^T) B^T` over fp16
/// rows at a stride wider than the row, accumulated into whatever y already
/// held. The kernel sums in fp32 and rounds y once, so the reference does the
/// same and the bound is one fp16 rounding of the result.
@Suite struct LoRADeltaTests {
    private struct LCG {
        var state: UInt64
        mutating func unit() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float((state >> 33) % 20_001) / 10_000 - 1
        }
    }

    private static func buffer<T>(_ ctx: MetalContext, _ values: [T]) throws -> MTLBuffer {
        let made = values.withUnsafeBytes { raw -> MTLBuffer? in
            guard let base = raw.baseAddress else { return nil }
            return ctx.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let made else { throw MetalError.commandEncoderFailed }
        return made
    }

    @Test(arguments: [(1, 16, 64, 96), (7, 16, 1024, 48), (33, 8, 200, 6144)])
    func matchesTheCPUReference(tokens: Int, rank: Int, columns: Int, rows: Int) throws {
        let ctx = try MetalContext()
        let delta = try LoRADelta(context: ctx)
        var rng = LCG(state: UInt64(tokens * 31 + rows))
        let xStride = columns + 3
        let yStride = rows + 5
        let x = (0..<(tokens * xStride)).map { _ in Float16(rng.unit()) }
        let y0 = (0..<(tokens * yStride)).map { _ in Float16(rng.unit()) }
        let a = (0..<(rank * columns)).map { _ in rng.unit() * 0.05 }
        let b = (0..<(rows * rank)).map { _ in rng.unit() * 0.05 }
        let scale: Float = 2

        let xBuf = try Self.buffer(ctx, x)
        let yBuf = try Self.buffer(ctx, y0)
        let aBuf = try Self.buffer(ctx, a)
        // The kernel reads B transposed, [rank x rows], as `LoRAAdapter` lays it out.
        let bT = (0..<(rank * rows)).map { k in b[(k % rows) * rank + k / rows] }
        let bBuf = try Self.buffer(ctx, bT)
        guard
            let scratch = ctx.device.makeBuffer(
                length: tokens * rank * 4, options: .storageModePrivate),
            let cb = ctx.queue.makeCommandBuffer()
        else { throw MetalError.commandEncoderFailed }
        try delta.encode(
            commandBuffer: cb, x: xBuf, xStrideElements: xStride, y: yBuf, yStrideElements: yStride,
            a: aBuf, aOffset: 0, b: bBuf, bOffset: 0, scratch: scratch,
            rank: rank, rows: rows, columns: columns, tokens: tokens, scale: scale)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let got = Array(
            UnsafeBufferPointer(
                start: yBuf.contents().bindMemory(to: Float16.self, capacity: y0.count),
                count: y0.count))
        var worst: Float = 0
        for t in 0..<tokens {
            let u = (0..<rank).map { j in
                (0..<columns).reduce(Float(0)) {
                    $0 + Float(x[t * xStride + $1]) * a[j * columns + $1]
                }
            }
            for i in 0..<rows {
                let d = (0..<rank).reduce(Float(0)) { $0 + u[$1] * b[i * rank + $1] }
                let want = Float(Float16(Float(y0[t * yStride + i]) + scale * d))
                worst = max(worst, abs(Float(got[t * yStride + i]) - want))
            }
            // Columns past the row width are padding: untouched.
            for i in rows..<yStride {
                #expect(got[t * yStride + i] == y0[t * yStride + i])
            }
        }
        // One fp16 ulp near 2 is 2^-9; the fp32 sums differ only in order.
        #expect(worst <= 0.002, "max |dy| \(worst)")
    }

    /// The grouped form (one down and one up dispatch for up to four
    /// projections reading the same x) against the per-target form: the same
    /// arithmetic per element, so the outputs match bit for bit.
    @Test func groupedDeltasMatchThePerTargetOnes() throws {
        let ctx = try MetalContext()
        let delta = try LoRADelta(context: ctx)
        var rng = LCG(state: 99)
        let tokens = 13, rank = 16, columns = 256
        let rows = [96, 16, 512, 40]
        let x = (0..<(tokens * columns)).map { _ in Float16(rng.unit()) }
        // One adapter-like buffer: each member's A then its B^T, packed.
        var packed: [Float] = []
        var offsets: [(a: Int, b: Int)] = []
        for n in rows {
            let a = packed.count * 4
            packed += (0..<(rank * columns)).map { _ in rng.unit() * 0.05 }
            let b = packed.count * 4
            packed += (0..<(rank * n)).map { _ in rng.unit() * 0.05 }
            offsets.append((a, b))
        }
        let y0 = rows.map { n in (0..<(tokens * n)).map { _ in Float16(rng.unit()) } }
        let xBuf = try Self.buffer(ctx, x)
        let adapter = try Self.buffer(ctx, packed)
        let single = try y0.map { try Self.buffer(ctx, $0) }
        let grouped = try y0.map { try Self.buffer(ctx, $0) }
        guard
            let scratch = ctx.device.makeBuffer(
                length: 4 * tokens * rank * 4, options: .storageModePrivate),
            let cb = ctx.queue.makeCommandBuffer()
        else { throw MetalError.commandEncoderFailed }
        for (m, n) in rows.enumerated() {
            try delta.encode(
                commandBuffer: cb, x: xBuf, xStrideElements: columns, y: single[m],
                yStrideElements: n, a: adapter, aOffset: offsets[m].a, b: adapter,
                bOffset: offsets[m].b, scratch: scratch, rank: rank, rows: n, columns: columns,
                tokens: tokens, scale: 2)
        }
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        try delta.encodeGroup(
            encoder: encoder, x: xBuf, xStrideElements: columns, columns: columns,
            members: rows.enumerated().map { m, n in
                LoRADelta.Member(
                    y: grouped[m], yStrideElements: n, rows: n, aOffset: offsets[m].a,
                    bOffset: offsets[m].b)
            },
            adapter: adapter, scratch: scratch, rank: rank, tokens: tokens, scale: 2)
        encoder.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        for (m, n) in rows.enumerated() {
            #expect(memcmp(single[m].contents(), grouped[m].contents(), tokens * n * 2) == 0)
        }
    }
}
