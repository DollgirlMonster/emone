import Foundation
import Metal
import Testing

@testable import TinyTitan

/// `qsa_block_scores_rows_mma` computes what `qsa_block_scores_rows` computes --
/// the sum over heads of ReLU(query . pooled block) -- with each head's products
/// on the matrix units. Only the order of the float sums changes, so the scores
/// must agree to float rounding, including the partial query and block tiles at
/// the edges and blocks whose score is zero after the ReLU.
@Suite struct QSABlockScoresMMATests {
    private struct LCG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 33
        }
        mutating func unit() -> Float { Float(next() % 20_001) / 10_000 - 1 }
    }

    private static func buffer<T>(_ ctx: MetalContext, _ values: [T]) throws -> MTLBuffer {
        let made = values.withUnsafeBytes { raw -> MTLBuffer? in
            guard let base = raw.baseAddress else { return nil }
            return ctx.device.makeBuffer(bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let made else { throw MetalError.commandEncoderFailed }
        return made
    }

    struct Shape: Sendable, CustomTestStringConvertible {
        let tokens: Int, blocks: Int
        var testDescription: String { "T\(tokens) blocks \(blocks)" }
    }

    @Test(arguments: [Shape(tokens: 37, blocks: 70), Shape(tokens: 64, blocks: 64), Shape(tokens: 5, blocks: 33)])
    func matrixUnitScoresMatchTheScalarKernel(_ shape: Shape) throws {
        let ctx = try MetalContext()
        let (heads, dim) = (4, 128)
        var rng = LCG(state: UInt64(shape.tokens * 31 + shape.blocks))
        let query = try Self.buffer(
            ctx, (0..<(shape.tokens * heads * dim)).map { _ in Float16(rng.unit() * 0.5) })
        let pooled = try Self.buffer(
            ctx, (0..<(shape.blocks * dim)).map { _ in Float16(rng.unit() * 0.5) })
        let count = shape.tokens * shape.blocks
        guard let reference = ctx.device.makeBuffer(length: count * 4, options: .storageModeShared),
            let candidate = ctx.device.makeBuffer(length: count * 4, options: .storageModeShared),
            let cb = ctx.queue.makeCommandBuffer()
        else { throw MetalError.commandEncoderFailed }
        memset(candidate.contents(), 0xFF, count * 4)

        var d = UInt32(dim)
        var h = UInt32(heads)
        var b = UInt32(shape.blocks)
        var t = UInt32(shape.tokens)
        for (name, out) in [("qsa_block_scores_rows", reference), ("qsa_block_scores_rows_mma", candidate)] {
            let pipeline = try ctx.pipeline(name)
            guard let enc = cb.makeComputeCommandEncoder() else { throw MetalError.commandEncoderFailed }
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(query, offset: 0, index: 0)
            enc.setBuffer(pooled, offset: 0, index: 1)
            enc.setBuffer(out, offset: 0, index: 2)
            enc.setBytes(&d, length: 4, index: 3)
            enc.setBytes(&h, length: 4, index: 4)
            enc.setBytes(&b, length: 4, index: 5)
            enc.setBytes(&t, length: 4, index: 6)
            if name.hasSuffix("_mma") {
                enc.dispatchThreadgroups(
                    MTLSize(width: (shape.blocks + 31) / 32, height: (shape.tokens + 31) / 32, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            } else {
                enc.dispatchThreads(
                    MTLSize(width: count, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
            enc.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let a = UnsafeBufferPointer(
            start: reference.contents().bindMemory(to: Float.self, capacity: count), count: count)
        let c = UnsafeBufferPointer(
            start: candidate.contents().bindMemory(to: Float.self, capacity: count), count: count)
        var worst: Float = 0
        var maxRef: Float = 0
        var zeros = 0
        for i in 0..<count {
            maxRef = max(maxRef, a[i])
            if a[i] == 0 { zeros += 1 }
            worst = max(worst, abs(a[i] - c[i]))
        }
        #expect(maxRef > 1, "the reference scores are too small to prove anything")
        #expect(worst <= 1e-4 * maxRef, "max |diff| \(worst) against max score \(maxRef)")
        #expect(zeros < count, "every score is zero")
    }
}
