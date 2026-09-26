import Foundation
import Metal
import Testing

@testable import TinyTitan

private let mppAvailableForSharedExpert: Bool = {
    guard let context = try? MetalContext() else { return false }
    return MPPPrefillInt4QMM(context: context).isAvailable
}()

/// The batched shared expert (`TINYTITAN_PREFILL_MPP_WIDE`) replaces the decode
/// path run once per token with three GEMMs over the chunk. The GEMMs sum in a
/// different order, so the check is closeness to the per-token path -- the one
/// production uses today -- not bit equality, and a layout mistake (a
/// transposed projection, a wrong stride) would miss by far more than rounding.
@Suite struct PrefillSharedExpertBatchedTests {
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

    private static func zeroed(_ ctx: MetalContext, bytes: Int) throws -> MTLBuffer {
        guard let made = ctx.device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw MetalError.commandEncoderFailed
        }
        memset(made.contents(), 0, bytes)
        return made
    }

    /// A 4-bit group-64 affine projection of `rows` x `cols`, small enough that
    /// the silu-gated products stay well inside fp16.
    private static func projection(
        _ ctx: MetalContext, rows: Int, cols: Int, rng: inout LCG
    ) throws -> SharedExpertProjection {
        let groups = rows * cols / Quantization.groupSize
        return SharedExpertProjection(
            weights: try buffer(ctx, (0..<(rows * cols / 2)).map { _ in UInt8(rng.next() & 0xFF) }),
            scales: try buffer(ctx, (0..<groups).map { _ in bf16(0.002 + abs(rng.unit()) * 0.004) }),
            biases: try buffer(ctx, (0..<groups).map { _ in bf16(-0.015 + rng.unit() * 0.002) }),
            rows: UInt32(rows), cols: UInt32(cols))
    }

    @Test(.enabled(if: mppAvailableForSharedExpert, "Requires runtime MPP TensorOps support"))
    func batchedBlockTracksThePerTokenPath() throws {
        let ctx = try MetalContext()
        try Self.tracksThePerTokenPath(
            ctx,
            try PrefillSharedExpert(
                context: ctx, weightBits: 4, siluActivation: true, batchedMPP: true))
    }

    /// The same block on the simdgroup-matrix QMM (`TINYTITAN_PREFILL_SG_QMM`),
    /// which every Apple GPU has, so no capability gate.
    @Test func simdgroupBatchedBlockTracksThePerTokenPath() throws {
        let ctx = try MetalContext()
        try Self.tracksThePerTokenPath(
            ctx,
            try PrefillSharedExpert(
                context: ctx, weightBits: 4, siluActivation: true, batchedSimdgroup: true))
    }

    private static func tracksThePerTokenPath(
        _ ctx: MetalContext, _ block: PrefillSharedExpert
    ) throws {
        #expect(block.batchedAvailable)
        let (d, intermediate, tokens) = (256, 128, 40)
        var rng = LCG(state: 23)
        let gate = try Self.projection(ctx, rows: intermediate, cols: d, rng: &rng)
        let up = try Self.projection(ctx, rows: intermediate, cols: d, rng: &rng)
        let down = try Self.projection(ctx, rows: d, cols: intermediate, rng: &rng)
        let x = try Self.buffer(ctx, (0..<(tokens * d)).map { _ in Float16(rng.unit()) })
        let half = MemoryLayout<Float16>.stride
        let scratch = (0..<3).map { _ in try? Self.zeroed(ctx, bytes: tokens * intermediate * half) }
        guard let sg = scratch[0], let su = scratch[1], let sa = scratch[2] else {
            Issue.record("scratch allocation failed")
            return
        }
        let perToken = try Self.zeroed(ctx, bytes: tokens * d * half)
        let batched = try Self.zeroed(ctx, bytes: tokens * d * half)

        guard let cb = ctx.queue.makeCommandBuffer() else { throw MetalError.commandEncoderFailed }
        try block.encodeBlock(
            commandBuffer: cb, x: x, y: perToken, gate: gate, up: up, down: down,
            scratchGate: sg, scratchUp: su, scratchAct: sa,
            queryCount: tokens, d: d, intermediate: intermediate,
            xStrideElements: d, yStrideElements: d)
        let took = try block.encodeBlockBatched(
            commandBuffer: cb, x: x, y: batched, gate: gate, up: up, down: down,
            scratchGate: sg, scratchUp: su, scratchAct: sa,
            queryCount: tokens, d: d, intermediate: intermediate,
            xStrideElements: d, yStrideElements: d)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        #expect(took)

        let a = perToken.contents().bindMemory(to: Float16.self, capacity: tokens * d)
        let b = batched.contents().bindMemory(to: Float16.self, capacity: tokens * d)
        var maxRef: Float = 0
        var maxDiff: Float = 0
        for i in 0..<(tokens * d) {
            maxRef = max(maxRef, abs(Float(a[i])))
            maxDiff = max(maxDiff, abs(Float(a[i]) - Float(b[i])))
        }
        #expect(maxRef > 0, "the reference output is all zeros; the fixture proves nothing")
        #expect(
            maxDiff <= 0.02 * maxRef,
            "batched vs per-token: max |diff| \(maxDiff) against max |ref| \(maxRef)")
    }

    @Test func aChunkTheBatchedPathCannotTakeFallsBackUntouched() throws {
        let ctx = try MetalContext()
        // Without the switch there is no batched path at all.
        let block = try PrefillSharedExpert(context: ctx, weightBits: 4, siluActivation: true)
        #expect(!block.batchedAvailable)
        var rng = LCG(state: 5)
        let p = try Self.projection(ctx, rows: 64, cols: 64, rng: &rng)
        let buf = try Self.zeroed(ctx, bytes: 64 * 64 * 2)
        guard let cb = ctx.queue.makeCommandBuffer() else { throw MetalError.commandEncoderFailed }
        let took = try block.encodeBlockBatched(
            commandBuffer: cb, x: buf, y: buf, gate: p, up: p, down: p,
            scratchGate: buf, scratchUp: buf, scratchAct: buf,
            queryCount: 1, d: 64, intermediate: 64,
            xStrideElements: 64, yStrideElements: 64)
        #expect(!took)
    }
}
