import Foundation
import Metal
import Testing

@testable import Emone

/// `encodeRows` runs a prefill chunk's per-token GEMVs in one compute encoder
/// instead of one each (`EMONE_PREFILL_COALESCE`). What makes it safe to
/// switch on is that nothing else changes -- the same pipeline, bindings and
/// grid per row -- so its output has to match the per-row `encode` loop bit
/// for bit, not within a tolerance. Same for the batched sigmoid gate.
@Suite struct GEMVRowsTests {
    /// Deterministic bytes; the kernels accept any nibble or int8 code.
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
            return ctx.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let made else { throw MetalError.commandEncoderFailed }
        return made
    }

    private static func zeroed(_ ctx: MetalContext, bytes: Int) throws -> MTLBuffer {
        guard let made = ctx.device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw MetalError.commandEncoderFailed
        }
        memset(made.contents(), 0, bytes)
        return made
    }

    private static func run(_ ctx: MetalContext, _ body: (MTLCommandBuffer) throws -> Void) throws {
        guard let cb = ctx.queue.makeCommandBuffer() else { throw MetalError.commandEncoderFailed }
        try body(cb)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
    }

    private static func same(_ a: MTLBuffer, _ b: MTLBuffer, bytes: Int) -> Bool {
        memcmp(a.contents(), b.contents(), bytes) == 0
    }

    /// Operands for `tokens` rows through an m x n group-64 affine projection.
    private struct Operands {
        let weights: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer, x: MTLBuffer
    }

    private static func operands(
        _ ctx: MetalContext, m: Int, n: Int, tokens: Int, weightBytesPerRow: Int, seed: UInt64
    ) throws -> Operands {
        var rng = LCG(state: seed)
        let groups = m * n / Quantization.groupSize
        return Operands(
            weights: try buffer(
                ctx, (0..<(m * weightBytesPerRow)).map { _ in UInt8(rng.next() & 0xFF) }),
            scales: try buffer(ctx, (0..<groups).map { _ in bf16(0.01 + abs(rng.unit()) * 0.05) }),
            biases: try buffer(ctx, (0..<groups).map { _ in bf16(rng.unit() * 0.1) }),
            x: try buffer(ctx, (0..<(tokens * n)).map { _ in Float16(rng.unit()) }))
    }

    @Test func int4RowsMatchThePerRowLoopExactly() throws {
        let ctx = try MetalContext()
        let gemv = try DequantInt4GEMV(context: ctx)
        try int4RowsMatch(ctx, gemv, m: 48, n: 256, tokens: 37)
        // A shape the class constant-folds (Qwen 3.6's 1024 x 2816), so the
        // specialized many-input pipeline is checked against the specialized
        // per-input one.
        try int4RowsMatch(ctx, gemv, m: 1_024, n: 2_816, tokens: 6)
    }

    private func int4RowsMatch(
        _ ctx: MetalContext, _ gemv: DequantInt4GEMV, m: Int, n: Int, tokens: Int
    ) throws {
        let ops = try Self.operands(
            ctx, m: m, n: n, tokens: tokens, weightBytesPerRow: n / 2, seed: 4)
        let half = MemoryLayout<Float16>.stride
        let perRow = try Self.zeroed(ctx, bytes: tokens * m * half)
        let coalesced = try Self.zeroed(ctx, bytes: tokens * m * half)
        try Self.run(ctx) { cb in
            for row in 0..<tokens {
                try gemv.encode(
                    commandBuffer: cb, weights: ops.weights, scales: ops.scales, biases: ops.biases,
                    x: ops.x, xOffset: row * n * half, y: perRow, yOffset: row * m * half,
                    m: UInt32(m), n: UInt32(n))
            }
            try gemv.encodeRows(
                commandBuffer: cb, weights: ops.weights, scales: ops.scales, biases: ops.biases,
                x: ops.x, y: coalesced,
                rows: GEMVRows(
                    xOffset: 0, xRowStride: n * half, yOffset: 0, yRowStride: m * half,
                    count: tokens),
                m: UInt32(m), n: UInt32(n))
        }
        #expect(Self.same(perRow, coalesced, bytes: tokens * m * half))
    }

    /// The shared-expert scalar gate's shape: one output per token (m = 1),
    /// written to consecutive halves.
    @Test func int8ScalarGateRowsMatchThePerRowLoopExactly() throws {
        let ctx = try MetalContext()
        let gemv = try DequantInt8GEMV(context: ctx)
        let (n, tokens) = (512, 29)
        let ops = try Self.operands(ctx, m: 1, n: n, tokens: tokens, weightBytesPerRow: n, seed: 8)
        let half = MemoryLayout<Float16>.stride
        let perRow = try Self.zeroed(ctx, bytes: tokens * half)
        let coalesced = try Self.zeroed(ctx, bytes: tokens * half)
        try Self.run(ctx) { cb in
            for row in 0..<tokens {
                try gemv.encode(
                    commandBuffer: cb, weights: ops.weights, scales: ops.scales, biases: ops.biases,
                    x: ops.x, xOffset: row * n * half, y: perRow, yOffset: row * half,
                    m: 1, n: UInt32(n))
            }
            try gemv.encodeRows(
                commandBuffer: cb, weights: ops.weights, scales: ops.scales, biases: ops.biases,
                x: ops.x, y: coalesced,
                rows: GEMVRows(
                    xOffset: 0, xRowStride: n * half, yOffset: 0, yRowStride: half, count: tokens),
                m: 1, n: UInt32(n))
        }
        #expect(Self.same(perRow, coalesced, bytes: tokens * half))
    }

    /// The same scalar-gate shape when the gate was promoted to bf16.
    @Test func bf16ScalarGateRowsMatchThePerRowLoopExactly() throws {
        let ctx = try MetalContext()
        let gemv = try BF16GEMV(context: ctx)
        let (n, tokens) = (2_560, 37)
        var rng = LCG(state: 21)
        let weights = try Self.buffer(ctx, (0..<n).map { _ in Self.bf16(rng.unit() * 0.2) })
        let x = try Self.buffer(ctx, (0..<(tokens * n)).map { _ in Float16(rng.unit()) })
        let half = MemoryLayout<Float16>.stride
        let perRow = try Self.zeroed(ctx, bytes: tokens * half)
        let coalesced = try Self.zeroed(ctx, bytes: tokens * half)
        try Self.run(ctx) { cb in
            for row in 0..<tokens {
                try gemv.encode(
                    commandBuffer: cb, weights: weights,
                    x: x, xOffset: row * n * half, y: perRow, yOffset: row * half,
                    m: 1, n: UInt32(n))
            }
            try gemv.encodeRows(
                commandBuffer: cb, weights: weights, x: x, y: coalesced,
                rows: GEMVRows(
                    xOffset: 0, xRowStride: n * half, yOffset: 0, yRowStride: half, count: tokens),
                m: 1, n: UInt32(n))
        }
        #expect(Self.same(perRow, coalesced, bytes: tokens * half))
        let untouched = try Self.zeroed(ctx, bytes: tokens * half)
        #expect(!Self.same(untouched, coalesced, bytes: tokens * half))
    }

    /// A many-row bf16 weight over many inputs -- the Gated-DeltaNet
    /// `in_proj_a` / `in_proj_b` shape (16 rows) and a row count that leaves a
    /// threadgroup partly empty (20) -- now one dispatch instead of one per
    /// input. Same bits as the per-input loop.
    @Test func bf16ManyRowsMatchThePerRowLoopExactly() throws {
        let ctx = try MetalContext()
        let gemv = try BF16GEMV(context: ctx)
        for (m, n, tokens) in [(16, 1_024, 37), (20, 256, 5)] {
            var rng = LCG(state: UInt64(m * 31 + tokens))
            let weights = try Self.buffer(
                ctx, (0..<(m * n)).map { _ in Self.bf16(rng.unit() * 0.2) })
            let x = try Self.buffer(ctx, (0..<(tokens * n)).map { _ in Float16(rng.unit()) })
            let half = MemoryLayout<Float16>.stride
            let perRow = try Self.zeroed(ctx, bytes: tokens * m * half)
            let coalesced = try Self.zeroed(ctx, bytes: tokens * m * half)
            try Self.run(ctx) { cb in
                for row in 0..<tokens {
                    try gemv.encode(
                        commandBuffer: cb, weights: weights,
                        x: x, xOffset: row * n * half, y: perRow, yOffset: row * m * half,
                        m: UInt32(m), n: UInt32(n))
                }
                try gemv.encodeRows(
                    commandBuffer: cb, weights: weights, x: x, y: coalesced,
                    rows: GEMVRows(
                        xOffset: 0, xRowStride: n * half, yOffset: 0, yRowStride: m * half,
                        count: tokens),
                    m: UInt32(m), n: UInt32(n))
            }
            #expect(Self.same(perRow, coalesced, bytes: tokens * m * half), "m=\(m)")
            let untouched = try Self.zeroed(ctx, bytes: tokens * m * half)
            #expect(!Self.same(untouched, coalesced, bytes: tokens * m * half), "m=\(m)")
        }
    }

    @Test func sigmoidRowsMulMatchesThePerRowScalarMulExactly() throws {
        let ctx = try MetalContext()
        let elementwise = try Elementwise(context: ctx)
        let (width, rows) = (320, 23)
        var rng = LCG(state: 15)
        let y0 = (0..<(width * rows)).map { _ in Float16(rng.unit() * 4) }
        let gate = try Self.buffer(ctx, (0..<rows).map { _ in Float16(rng.unit() * 6) })
        let perRow = try Self.buffer(ctx, y0)
        let batched = try Self.buffer(ctx, y0)
        let half = MemoryLayout<Float16>.stride
        try Self.run(ctx) { cb in
            for row in 0..<rows {
                try elementwise.encodeSigmoidScalarMul(
                    commandBuffer: cb, y: perRow, yOffset: row * width * half,
                    gate: gate, gateOffset: row * half, count: width)
            }
            try elementwise.encodeSigmoidRowsMul(
                commandBuffer: cb, y: batched, gate: gate, width: width, rows: rows)
        }
        #expect(Self.same(perRow, batched, bytes: width * rows * half))
        // And it did something: the rows are not the untouched input.
        let untouched = try Self.buffer(ctx, y0)
        #expect(!Self.same(untouched, batched, bytes: width * rows * half))
    }

    @Test func aSingleRowIsTheOrdinaryEncode() {
        let single = GEMVRows.single(xOffset: 64, yOffset: 128)
        #expect(single.count == 1)
        #expect(single.xOffset == 64 && single.yOffset == 128)
    }
}

/// The tiled simdgroup QMM picks a 32-row tile for small chunks
/// (`PrefillAffineSimdgroupQMM.tile(n:t:)`). That is only a scheduling change if
/// every output element is the same bits at either edge.
@Suite struct SimdgroupQMMTileTests {
    @Test(arguments: [4, 8], [5, 40, 64])
    func thirtyTwoAndSixtyFourRowTilesAgreeBitForBit(bits: Int, t: Int) throws {
        let ctx = try MetalContext()
        let qmm = try PrefillAffineSimdgroupQMM(context: ctx)
        let (n, k) = (192, 256)
        var state: UInt64 = UInt64(bits * 1_000 + t)
        func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 33
        }
        func unit() -> Float { Float(next() % 20_001) / 10_000 - 1 }
        func bf16(_ v: Float) -> UInt16 { UInt16(truncatingIfNeeded: v.bitPattern >> 16) }
        func buffer<T>(_ values: [T]) throws -> MTLBuffer {
            try #require(
                values.withUnsafeBytes { raw in
                    raw.baseAddress.flatMap {
                        ctx.device.makeBuffer(
                            bytes: $0, length: raw.count, options: .storageModeShared)
                    }
                })
        }
        let groups = n * k / Quantization.groupSize
        let weights = try buffer((0..<(n * k * bits / 8)).map { _ in UInt8(next() & 0xFF) })
        let scales = try buffer((0..<groups).map { _ in bf16(0.01 + abs(unit()) * 0.05) })
        let biases = try buffer((0..<groups).map { _ in bf16(unit() * 0.1) })
        let x = try buffer((0..<(t * k)).map { _ in Float16(unit()) })
        let bytes = t * n * MemoryLayout<Float16>.stride
        let small = try #require(ctx.device.makeBuffer(length: bytes, options: .storageModeShared))
        let large = try #require(ctx.device.makeBuffer(length: bytes, options: .storageModeShared))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        for (edge, y) in [(32, small), (64, large)] {
            try qmm.encode(
                commandBuffer: cb, weights: weights, scales: scales, biases: biases,
                x: x, y: y, t: t, n: n, k: k, bits: bits, forcedEdge: edge)
        }
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        #expect(memcmp(small.contents(), large.contents(), bytes) == 0)
    }
}

/// The 8-bit many-row path: one dispatch over 16-input blocks must match the
/// per-input loop bit for bit, including a partial last block.
@Suite struct Int8ManyRowsTests {
    @Test(arguments: [(48, 256, 37), (16, 1_024, 16), (20, 128, 3)])
    func int8ManyRowsMatchThePerRowLoopExactly(shape: (Int, Int, Int)) throws {
        let (m, n, tokens) = shape
        let ctx = try MetalContext()
        let gemv = try DequantInt8GEMV(context: ctx)
        var state = UInt64(m * 7 + tokens)
        func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 33
        }
        func unit() -> Float { Float(next() % 20_001) / 10_000 - 1 }
        func bf16(_ v: Float) -> UInt16 { UInt16(truncatingIfNeeded: v.bitPattern >> 16) }
        func buffer<T>(_ values: [T]) throws -> MTLBuffer {
            try #require(
                values.withUnsafeBytes { raw in
                    raw.baseAddress.flatMap {
                        ctx.device.makeBuffer(
                            bytes: $0, length: raw.count, options: .storageModeShared)
                    }
                })
        }
        let groups = m * n / Quantization.groupSize
        let weights = try buffer((0..<(m * n)).map { _ in UInt8(next() & 0xFF) })
        let scales = try buffer((0..<groups).map { _ in bf16(0.01 + abs(unit()) * 0.05) })
        let biases = try buffer((0..<groups).map { _ in bf16(unit() * 0.1) })
        let x = try buffer((0..<(tokens * n)).map { _ in Float16(unit()) })
        let half = MemoryLayout<Float16>.stride
        let bytes = tokens * m * half
        let perRow = try #require(ctx.device.makeBuffer(length: bytes, options: .storageModeShared))
        let coalesced = try #require(
            ctx.device.makeBuffer(length: bytes, options: .storageModeShared))
        memset(coalesced.contents(), 0, bytes)
        let cb = try #require(ctx.queue.makeCommandBuffer())
        for row in 0..<tokens {
            try gemv.encode(
                commandBuffer: cb, weights: weights, scales: scales, biases: biases,
                x: x, xOffset: row * n * half, y: perRow, yOffset: row * m * half,
                m: UInt32(m), n: UInt32(n))
        }
        try gemv.encodeRows(
            commandBuffer: cb, weights: weights, scales: scales, biases: biases,
            x: x, y: coalesced,
            rows: GEMVRows(
                xOffset: 0, xRowStride: n * half, yOffset: 0, yRowStride: m * half,
                count: tokens),
            m: UInt32(m), n: UInt32(n))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        #expect(memcmp(perRow.contents(), coalesced.contents(), bytes) == 0)
    }
}

/// The generic affine GEMV's many-row path, at both widths it is built for.
@Suite struct AffineManyRowsTests {
    @Test(arguments: [4, 8], [(48, 256, 37), (16, 1_024, 16), (20, 128, 3)])
    func affineManyRowsMatchThePerRowLoopExactly(bits: Int, shape: (Int, Int, Int)) throws {
        let (m, n, tokens) = shape
        let ctx = try MetalContext()
        let gemv = try AffineQuantGEMV(context: ctx, weightBits: bits)
        var state = UInt64(bits * 1_000 + m * 7 + tokens)
        func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 33
        }
        func unit() -> Float { Float(next() % 20_001) / 10_000 - 1 }
        func bf16(_ v: Float) -> UInt16 { UInt16(truncatingIfNeeded: v.bitPattern >> 16) }
        func buffer<T>(_ values: [T]) throws -> MTLBuffer {
            try #require(
                values.withUnsafeBytes { raw in
                    raw.baseAddress.flatMap {
                        ctx.device.makeBuffer(
                            bytes: $0, length: raw.count, options: .storageModeShared)
                    }
                })
        }
        let groups = m * n / Quantization.groupSize
        let weights = try buffer((0..<(m * n * bits / 8)).map { _ in UInt8(next() & 0xFF) })
        let scales = try buffer((0..<groups).map { _ in bf16(0.01 + abs(unit()) * 0.05) })
        let biases = try buffer((0..<groups).map { _ in bf16(unit() * 0.1) })
        let x = try buffer((0..<(tokens * n)).map { _ in Float16(unit()) })
        let half = MemoryLayout<Float16>.stride
        let bytes = tokens * m * half
        let perRow = try #require(ctx.device.makeBuffer(length: bytes, options: .storageModeShared))
        let coalesced = try #require(
            ctx.device.makeBuffer(length: bytes, options: .storageModeShared))
        memset(coalesced.contents(), 0, bytes)
        let cb = try #require(ctx.queue.makeCommandBuffer())
        for row in 0..<tokens {
            try gemv.encode(
                commandBuffer: cb, weights: weights, scales: scales, biases: biases,
                x: x, xOffset: row * n * half, y: perRow, yOffset: row * m * half,
                m: UInt32(m), n: UInt32(n))
        }
        try gemv.encodeRows(
            commandBuffer: cb, weights: weights, scales: scales, biases: biases,
            x: x, y: coalesced,
            rows: GEMVRows(
                xOffset: 0, xRowStride: n * half, yOffset: 0, yRowStride: m * half,
                count: tokens),
            m: UInt32(m), n: UInt32(n))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        #expect(memcmp(perRow.contents(), coalesced.contents(), bytes) == 0)
    }
}
