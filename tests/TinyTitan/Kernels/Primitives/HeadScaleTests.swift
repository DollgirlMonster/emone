import Foundation
import Metal
import Testing

@testable import TinyTitan

/// `HeadScale` against a CPU reference: each head's slice of every row is
/// multiplied by that head's factor, read at a byte offset into the buffer,
/// and nothing else moves. The kernel rounds once, as the reference does.
@Suite struct HeadScaleTests {
    @Test(arguments: [(1, 8, 256), (5, 8, 256), (3, 4, 16)])
    func scalesEachHeadSlice(tokens: Int, heads: Int, dim: Int) throws {
        let ctx = try MetalContext()
        let kernel = try HeadScale(context: ctx)
        let x0 = (0..<(tokens * heads * dim)).map { Float16(Float($0 % 97) / 50 - 1) }
        // Two leading floats, so the offset is exercised, then one factor per head.
        let scales: [Float] = [9, 9] + (0..<heads).map { 0.5 + Float($0) * 0.25 }
        let x = try #require(
            x0.withUnsafeBytes {
                ctx.device.makeBuffer(
                    bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            })
        let s = try #require(
            scales.withUnsafeBytes {
                ctx.device.makeBuffer(
                    bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            })
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try kernel.encode(
            commandBuffer: cb, x: x, scale: s, scaleOffset: 8, heads: heads, dim: dim,
            tokens: tokens)
        cb.commit()
        cb.waitUntilCompleted()
        let got = x.contents().bindMemory(to: Float16.self, capacity: x0.count)
        for i in 0..<x0.count {
            let h = (i / dim) % heads
            #expect(got[i] == Float16(Float(x0[i]) * scales[2 + h]))
        }
    }
}
