import Foundation
import Metal
import Testing

@testable import TinyTitan

/// GPU time of each step of a prefill hyper-connection read and write at
/// Qwen3.8's shape (4 streams x 2560, low rank 320, 4-bit gates), each in
/// its own command buffer. Prints; asserts nothing.
/// TINYTITAN_HC_BENCH=1 to run.
@Suite struct HyperConnectionPrefillBenchTests {
    /// The eight-wide inject writes the same bytes as the one-wide kernel.
    @Test func injectX8MatchesScalarExactly() throws {
        let ctx = try MetalContext()
        let (d, streams, tokens) = (2_560, 4, 7)
        var state: UInt64 = 99
        func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float((state >> 33) % 20_001) / 10_000 - 1
        }
        let base = (0..<(tokens * streams * d)).map { _ in Float16(next() * 3) }
        let blockOut = (0..<(tokens * d)).map { _ in Float16(next()) }
        let inject = (0..<(tokens * streams)).map { _ in Float16(next() * 4) }
        func buffer<T>(_ v: [T]) throws -> MTLBuffer {
            guard let b = v.withUnsafeBytes({ ctx.device.makeBuffer(
                bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
            else { throw MetalError.commandEncoderFailed }
            return b
        }
        var outputs: [[UInt8]] = []
        for (name, perThread) in [("hc_stream_inject_fp16", 1), ("hc_stream_inject_fp16_x8", 8)] {
            let pso = try ctx.pipeline(name)
            let st = try buffer(base)
            guard let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
            else { throw MetalError.commandEncoderFailed }
            enc.setComputePipelineState(pso)
            enc.setBuffer(st, offset: 0, index: 0)
            enc.setBuffer(try buffer(blockOut), offset: 0, index: 1)
            enc.setBuffer(try buffer(inject), offset: 0, index: 2)
            var (ud, us, ut, scale) = (UInt32(d), UInt32(streams), UInt32(tokens), Float(0.25))
            enc.setBytes(&ud, length: 4, index: 3)
            enc.setBytes(&us, length: 4, index: 4)
            enc.setBytes(&ut, length: 4, index: 5)
            enc.setBytes(&scale, length: 4, index: 6)
            enc.dispatchThreads(
                MTLSize(width: tokens * streams * d / perThread, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            outputs.append(Array(UnsafeRawBufferPointer(start: st.contents(), count: st.length)))
        }
        #expect(outputs[0] == outputs[1])
        #expect(outputs[0] != base.withUnsafeBytes { Array($0) })
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_HC_BENCH"] == "1"))
    func steps() throws {
        let ctx = try MetalContext()
        let (dim, streams, lowRank) = (2_560, 4, 320)
        let t = Int(ProcessInfo.processInfo.environment["TINYTITAN_HC_BENCH_ROWS"] ?? "") ?? 16_931
        let wide = dim * streams
        let rms = try RMSNorm(context: ctx)
        let elementwise = try Elementwise(context: ctx)
        let dense = try PrefillAffineSimdgroupQMM(context: ctx)
        let scalarQMM = try PrefillInt4QMM(context: ctx, weightBits: 4)
        func buffer(_ bytes: Int, fill: UInt8 = 0x3C) throws -> MTLBuffer {
            guard let b = ctx.device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw MetalError.commandEncoderFailed
            }
            memset(b.contents(), Int32(fill), bytes)
            return b
        }
        // 4-bit weights, bf16 scales and biases (0x3C3C is a small positive bf16).
        func gate(rows: Int, cols: Int) throws -> (MTLBuffer, MTLBuffer, MTLBuffer) {
            let groups = rows * cols / 64
            return (try buffer(rows * cols / 2, fill: 0x5A), try buffer(groups * 2, fill: 0x3B),
                try buffer(groups * 2, fill: 0x00))
        }
        let streamsBuf = try buffer(t * wide * 2, fill: 0x2C)
        let normWeight = try buffer(wide * 2, fill: 0x3F)
        let normed = try buffer(t * wide * 2)
        let low = try buffer(t * lowRank * 2)
        let mix = try buffer(t * wide * 2)
        let blockInput = try buffer(t * dim * 2)
        let injectOut = try buffer(t * streams * 2)
        let down = try gate(rows: lowRank, cols: wide)
        let up = try gate(rows: wide, cols: lowRank)
        let inject = try gate(rows: streams, cols: wide)

        func time(_ label: String, bytes: Double, flop: Double = 0, _ body: (MTLCommandBuffer) throws -> Void) throws {
            var best = Double.infinity
            for _ in 0..<4 {
                guard let cb = ctx.queue.makeCommandBuffer() else { throw MetalError.commandEncoderFailed }
                try body(cb)
                cb.commit()
                cb.waitUntilCompleted()
                best = min(best, cb.gpuEndTime - cb.gpuStartTime)
            }
            print(String(format: "[hc-bench] %-14@ %7.2f ms  %6.0f GB/s  %5.2f TFLOP/s",
                label as NSString, best * 1000, bytes / best / 1e9, flop / best / 1e12))
        }
        let h = Double(t) * 2
        try time("rms", bytes: h * Double(wide) * 2) { cb in
            try rms.encodeBF16WGrouped(
                commandBuffer: cb, x: streamsBuf, weight: normWeight, out: normed,
                groupDim: UInt32(dim), numGroups: streams, eps: 1e-6, tokens: t)
        }
        try time("down", bytes: h * Double(wide + lowRank), flop: 2 * Double(t * wide * lowRank)) { cb in
            try dense.encode(
                commandBuffer: cb, weights: down.0, scales: down.1, biases: down.2,
                x: normed, y: low, t: t, n: lowRank, k: wide, bits: 4)
        }
        try time("silu", bytes: h * Double(lowRank) * 2) { cb in
            try elementwise.encodeSilu(commandBuffer: cb, x: low, out: low, count: lowRank * t, inScale: 1)
        }
        try time("up", bytes: h * Double(wide + lowRank), flop: 2 * Double(t * wide * lowRank)) { cb in
            try dense.encode(
                commandBuffer: cb, weights: up.0, scales: up.1, biases: up.2,
                x: low, y: mix, t: t, n: wide, k: lowRank, bits: 4)
        }
        try time("mix_reduce", bytes: h * Double(2 * wide + dim)) { cb in
            try elementwise.encodeHCMixReduce(
                commandBuffer: cb, mix: mix, normed: normed, out: blockInput,
                dim: dim, streams: streams, tokens: t, inScale: 1)
        }
        try time("inject_proj", bytes: h * Double(wide), flop: 2 * Double(t * wide * streams)) { cb in
            try scalarQMM.encode(
                commandBuffer: cb, weights: inject.0, scales: inject.1, biases: inject.2,
                x: normed, y: injectOut, t: t, n: streams, k: wide)
        }
        try time("inject", bytes: h * Double(2 * wide + dim)) { cb in
            try elementwise.encodeHCInject(
                commandBuffer: cb, streams: streamsBuf, blockOut: blockInput, inject: injectOut,
                dim: dim, streamCount: streams, tokens: t, inScale: 1)
        }
    }
}
