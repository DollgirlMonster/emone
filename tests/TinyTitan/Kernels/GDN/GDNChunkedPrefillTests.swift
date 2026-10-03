import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The chunked prefill recurrence (`gdn_delta_chunk_prefill_dk*`) against the
/// sequential one (`gdn_delta_step_prefill`) on the same inputs: y and the
/// carried state must agree by rounding, across row counts that land on, off
/// and below the 8-token chunk, from a zero and a non-zero starting state.
@Suite struct GDNChunkedPrefillTests {

    /// Deterministic uniform values in [lo, hi).
    private struct LCG {
        var s: UInt64
        mutating func next(_ lo: Float, _ hi: Float) -> Float {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return lo + (hi - lo) * Float(s >> 40) / Float(1 << 24)
        }
    }

    private static func bf16Buffer(_ device: MTLDevice, _ values: [Float]) -> MTLBuffer? {
        let bits = values.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
        return device.makeBuffer(bytes: bits, length: bits.count * 2, options: .storageModeShared)
    }

    private struct Outputs {
        let y: [Float]
        let state: [Float]
    }

    /// Runs q/k norm then one prefill recurrence over `rows`, starting from
    /// `initialState`, and returns y and the final state.
    private static func run(
        ctx: MetalContext, cfg: LinearAttentionConfig, chunked: Bool,
        convOut: [Float16], a: [Float16], b: [Float16],
        aLog: [Float], dtBias: [Float], initialState: [Float], rows: Int
    ) throws -> Outputs {
        let gdn = try GDN(context: ctx, config: cfg, chunkedPrefill: chunked)
        let stateCount = cfg.numVHeads * cfg.valueHeadDim * cfg.keyHeadDim
        guard
            let conv = Fp16Buffer.make(ctx.device, halves: convOut),
            let aBuf = Fp16Buffer.make(ctx.device, halves: a),
            let bBuf = Fp16Buffer.make(ctx.device, halves: b),
            let aLogBuf = bf16Buffer(ctx.device, aLog),
            let dtBuf = bf16Buffer(ctx.device, dtBias),
            let state = ctx.device.makeBuffer(
                bytes: initialState, length: stateCount * 4, options: .storageModeShared),
            let y = Fp16Buffer.make(ctx.device, count: rows * cfg.valueDim),
            let cb = ctx.queue.makeCommandBuffer()
        else {
            throw MetalError.noDevice
        }
        try gdn.encodeQKNorm(commandBuffer: cb, convOut: conv, rows: rows)
        try gdn.encodeDeltaStepPrefill(
            commandBuffer: cb, convOut: conv,
            aProj: aBuf, bProj: bBuf,
            aLog: aLogBuf, aLogOffset: 0,
            dtBias: dtBuf, dtBiasOffset: 0,
            state: state, y: y, rows: rows)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw error }
        let yPtr = y.contents().bindMemory(to: Float16.self, capacity: rows * cfg.valueDim)
        let sPtr = state.contents().bindMemory(to: Float.self, capacity: stateCount)
        return Outputs(
            y: (0..<(rows * cfg.valueDim)).map { Float(yPtr[$0]) },
            state: (0..<stateCount).map { sPtr[$0] })
    }

    private static func compare(cfg: LinearAttentionConfig, rows: Int, warmState: Bool, seed: UInt64)
        throws
    {
        var rng = LCG(s: seed)
        let ctx = try MetalContext()
        let convOut = (0..<(rows * cfg.qkvDim)).map { _ in Float16(rng.next(-1, 1)) }
        let a = (0..<(rows * cfg.numVHeads)).map { _ in Float16(rng.next(-2, 2)) }
        let b = (0..<(rows * cfg.numVHeads)).map { _ in Float16(rng.next(-2, 2)) }
        let aLog = (0..<cfg.numVHeads).map { _ in rng.next(-1, 1.5) }
        let dtBias = (0..<cfg.numVHeads).map { _ in rng.next(-0.5, 0.5) }
        let stateCount = cfg.numVHeads * cfg.valueHeadDim * cfg.keyHeadDim
        let initial = (0..<stateCount).map { _ in warmState ? rng.next(-0.5, 0.5) : 0 }

        let seq = try run(
            ctx: ctx, cfg: cfg, chunked: false, convOut: convOut, a: a, b: b,
            aLog: aLog, dtBias: dtBias, initialState: initial, rows: rows)
        let chunk = try run(
            ctx: ctx, cfg: cfg, chunked: true, convOut: convOut, a: a, b: b,
            aLog: aLog, dtBias: dtBias, initialState: initial, rows: rows)

        var yErr: Float = 0
        var yScale: Float = 0
        for i in 0..<seq.y.count {
            #expect(chunk.y[i].isFinite, "y[\(i)] not finite")
            yErr = max(yErr, abs(seq.y[i] - chunk.y[i]))
            yScale = max(yScale, abs(seq.y[i]))
        }
        var sErr: Float = 0
        for i in 0..<stateCount { sErr = max(sErr, abs(seq.state[i] - chunk.state[i])) }
        // y is FP16 (one ULP at |y| ~ 1 is ~1e-3); the state is FP32.
        #expect(
            yErr <= max(2e-3, yScale * 2e-3),
            "rows \(rows) warm \(warmState): y max err \(yErr) at scale \(yScale)")
        #expect(sErr <= 1e-3, "rows \(rows) warm \(warmState): state max err \(sErr)")
    }

    private static let small = LinearAttentionConfig(
        numKHeads: 2, numVHeads: 4, keyHeadDim: 32, valueHeadDim: 32,
        convKernelSize: 4)

    // Qwen3.8-Flash-Next's linear-attention geometry.
    private static let qwen38 = LinearAttentionConfig(
        numKHeads: 16, numVHeads: 48, keyHeadDim: 128, valueHeadDim: 128,
        convKernelSize: 4)

    @Test(arguments: [1, 7, 8, 9, 37, 64])
    func smallShapeMatchesSequential(rows: Int) throws {
        try Self.compare(cfg: Self.small, rows: rows, warmState: false, seed: 0x5EED + UInt64(rows))
        try Self.compare(cfg: Self.small, rows: rows, warmState: true, seed: 0xFACE + UInt64(rows))
    }

    @Test func qwen38ShapeMatchesSequential() throws {
        try Self.compare(cfg: Self.qwen38, rows: 203, warmState: true, seed: 0x38)
    }
}
