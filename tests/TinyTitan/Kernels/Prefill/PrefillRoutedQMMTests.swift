import Foundation
import Metal
import Testing

@testable import TinyTitan

/// `prefill_routed_qmm_gate_up` and `prefill_routed_qmm` against a host
/// reference: a tile of experts with uneven row counts (one below a 32-row work
/// item, one spanning several), 4- and 8-bit weights in TinyTitan's affine
/// layout, SiLU. The kernels round each dequantized weight and the gate/up
/// activation to half and sum in another order, so the check is relative.
@Suite struct PrefillRoutedQMMTests {
    private struct LCG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 33
        }
        mutating func unit() -> Float { Float(next() % 20_001) / 10_000 - 1 }
    }

    /// One expert blob: gate [F][D], up [F][D], down [D][F], each packed
    /// weights then bf16 scales then bf16 biases, group 64.
    struct Layout {
        let d: Int, f: Int, bits: Int
        var projBytes: Int { 0 }
        func packed(_ n: Int, _ k: Int) -> Int { n * k * bits / 8 }
        func groups(_ n: Int, _ k: Int) -> Int { n * (k / 64) * 2 }
        func size(_ n: Int, _ k: Int) -> Int { packed(n, k) + 2 * groups(n, k) }
        var gate: Int { 0 }
        var up: Int { size(f, d) }
        var down: Int { 2 * size(f, d) }
        var total: Int { 2 * size(f, d) + size(d, f) }
    }

    private static func bf16(_ v: Float) -> UInt16 { UInt16(v.bitPattern >> 16) }
    private static func fromBF16(_ b: UInt16) -> Float { Float(bitPattern: UInt32(b) << 16) }

    /// Writes a random projection at `base` and returns its dequantized
    /// weights [n][k] as the kernel sees them (rounded to half).
    private static func writeProjection(
        _ bytes: UnsafeMutableRawPointer, base: Int, n: Int, k: Int, bits: Int, rng: inout LCG
    ) -> [Float] {
        let packedBytes = n * k * bits / 8
        let groups = k / 64
        var w = [Float](repeating: 0, count: n * k)
        let p = bytes + base
        for i in 0..<packedBytes { p.storeBytes(of: UInt8(rng.next() & 0xFF), toByteOffset: i, as: UInt8.self) }
        let maxQ: Float = bits == 4 ? 15 : 255
        for row in 0..<n {
            for g in 0..<groups {
                let scale = bf16((0.02 + abs(rng.unit()) * 0.02) * 15 / maxQ)
                let bias = bf16(-0.3 + rng.unit() * 0.05)
                p.storeBytes(of: scale, toByteOffset: packedBytes + (row * groups + g) * 2, as: UInt16.self)
                p.storeBytes(
                    of: bias, toByteOffset: packedBytes + n * groups * 2 + (row * groups + g) * 2,
                    as: UInt16.self)
                for c in (g * 64)..<(g * 64 + 64) {
                    let bit = (row * k + c) * bits
                    let byte = p.load(fromByteOffset: bit / 8, as: UInt8.self)
                    let q = bits == 4 ? (bit % 8 == 0 ? byte & 0xF : byte >> 4) : byte
                    w[row * k + c] = Float(Float16(Float(q) * fromBF16(scale) + fromBF16(bias)))
                }
            }
        }
        return w
    }

    struct Fixture {
        let ctx: MetalContext
        let blob: MTLBuffer
        let args: MTLBuffer
        let work: MTLBuffer
        let workCount: Int
        let rows: Int
        let layout: Layout
        let rowExpert: [Int]
        let gate: [[Float]], up: [[Float]], down: [[Float]]
    }

    static func fixture(
        bits: Int, d: Int, f: Int, rowsPerExpert: [Int], reference: Bool, edge: Int = 32
    ) throws -> Fixture
    {
        let ctx = try MetalContext()
        let layout = Layout(d: d, f: f, bits: bits)
        let experts = rowsPerExpert.count
        guard let blob = ctx.device.makeBuffer(length: experts * layout.total, options: .storageModeShared),
            let args = ctx.device.makeBuffer(length: 16 * 8, options: .storageModeShared)
        else { throw MetalError.commandEncoderFailed }
        var rng = LCG(state: UInt64(bits * 1000 + d))
        var gate: [[Float]] = [], up: [[Float]] = [], down: [[Float]] = []
        for e in 0..<experts {
            let base = e * layout.total
            if reference {
                gate.append(writeProjection(blob.contents(), base: base + layout.gate, n: f, k: d, bits: bits, rng: &rng))
                up.append(writeProjection(blob.contents(), base: base + layout.up, n: f, k: d, bits: bits, rng: &rng))
                down.append(writeProjection(blob.contents(), base: base + layout.down, n: d, k: f, bits: bits, rng: &rng))
            } else {
                memset(blob.contents() + base, 0x5A, layout.total)
            }
            args.contents().storeBytes(of: blob.gpuAddress + UInt64(base), toByteOffset: e * 8, as: UInt64.self)
        }
        var work: [UInt32] = []
        var rowExpert: [Int] = []
        var row = 0
        for (e, count) in rowsPerExpert.enumerated() {
            for start in stride(from: 0, to: count, by: edge) {
                work += [UInt32(e), UInt32(row + start), UInt32(min(edge, count - start)), 0]
            }
            rowExpert += Array(repeating: e, count: count)
            row += count
        }
        guard let workBuf = ctx.device.makeBuffer(bytes: work, length: work.count * 4, options: .storageModeShared)
        else { throw MetalError.commandEncoderFailed }
        return Fixture(
            ctx: ctx, blob: blob, args: args, work: workBuf, workCount: work.count / 4,
            rows: row, layout: layout, rowExpert: rowExpert, gate: gate, up: up, down: down)
    }

    static func params(_ fx: Fixture, down: Bool) -> [UInt32] {
        let l = fx.layout
        if down {
            let o = l.down
            return [UInt32(l.d), UInt32(l.f), UInt32(o), UInt32(o + l.packed(l.d, l.f)),
                    UInt32(o + l.packed(l.d, l.f) + l.groups(l.d, l.f)), 0, 0, 0, 64]
        }
        let g = l.gate, u = l.up
        return [UInt32(l.f), UInt32(l.d),
                UInt32(g), UInt32(g + l.packed(l.f, l.d)), UInt32(g + l.packed(l.f, l.d) + l.groups(l.f, l.d)),
                UInt32(u), UInt32(u + l.packed(l.f, l.d)), UInt32(u + l.packed(l.f, l.d) + l.groups(l.f, l.d)),
                64]
    }

    static func encode(
        _ fx: Fixture, _ cb: MTLCommandBuffer, pso: MTLComputePipelineState,
        x: MTLBuffer, y: MTLBuffer, params: [UInt32], n: Int, edge: Int = 32
    ) throws {
        guard let enc = cb.makeComputeCommandEncoder() else { throw MetalError.commandEncoderFailed }
        enc.setComputePipelineState(pso)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(y, offset: 0, index: 1)
        enc.setBuffer(fx.args, offset: 0, index: 2)
        enc.setBuffer(fx.work, offset: 0, index: 3)
        var p = params
        enc.setBytes(&p, length: p.count * 4, index: 4)
        enc.useResource(fx.blob, usage: .read)
        enc.dispatchThreadgroups(
            MTLSize(width: n / edge, height: fx.workCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
    }

    static func pipelines(_ ctx: MetalContext, bits: Int, edge: Int = 32) throws
        -> (MTLComputePipelineState, MTLComputePipelineState)
    {
        let constants = [
            MetalFunctionConstant(index: 77, value: .bool(true)),
            MetalFunctionConstant(index: 78, value: .uint32(UInt32(bits))),
        ]
        return (
            try ctx.pipeline("prefill_routed_qmm_gate_up_\(edge)x\(edge)", constants: constants),
            try ctx.pipeline("prefill_routed_qmm_\(edge)x\(edge)", constants: constants)
        )
    }

    @Test(arguments: [(4, 32), (8, 32), (4, 64), (8, 64)])
    func matchesTheHostReference(bits: Int, edge: Int) throws {
        let (d, f) = (256, 128)
        let fx = try Self.fixture(
            bits: bits, d: d, f: f, rowsPerExpert: [5, 70, 33, 130], reference: true, edge: edge)
        let (gateUp, down) = try Self.pipelines(fx.ctx, bits: bits, edge: edge)
        var rng = LCG(state: 9)
        let xs = (0..<(fx.rows * d)).map { _ in Float16(rng.unit()) }
        guard let x = fx.ctx.device.makeBuffer(bytes: xs, length: xs.count * 2, options: .storageModeShared),
            let act = fx.ctx.device.makeBuffer(length: fx.rows * f * 2, options: .storageModeShared),
            let out = fx.ctx.device.makeBuffer(length: fx.rows * d * 2, options: .storageModeShared),
            let cb = fx.ctx.queue.makeCommandBuffer()
        else { throw MetalError.commandEncoderFailed }
        try Self.encode(
            fx, cb, pso: gateUp, x: x, y: act, params: Self.params(fx, down: false), n: f, edge: edge)
        try Self.encode(
            fx, cb, pso: down, x: act, y: out, params: Self.params(fx, down: true), n: d, edge: edge)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        let actGPU = act.contents().bindMemory(to: Float16.self, capacity: fx.rows * f)
        let outGPU = out.contents().bindMemory(to: Float16.self, capacity: fx.rows * d)
        var worstAct: Float = 0, worstOut: Float = 0, scaleOut: Float = 0
        for r in 0..<fx.rows {
            let e = fx.rowExpert[r]
            var a = [Float](repeating: 0, count: f)
            for c in 0..<f {
                var g: Float = 0, u: Float = 0
                for k in 0..<d {
                    g += Float(xs[r * d + k]) * fx.gate[e][c * d + k]
                    u += Float(xs[r * d + k]) * fx.up[e][c * d + k]
                }
                a[c] = Float(Float16(g / (1 + exp(-g)) * u))
                worstAct = max(worstAct, abs(Float(actGPU[r * f + c]) - a[c]) / max(1, abs(a[c])))
            }
            for c in 0..<d {
                var o: Float = 0
                for k in 0..<f { o += Float(actGPU[r * f + k]) * fx.down[e][c * f + k] }
                scaleOut = max(scaleOut, abs(o))
                worstOut = max(worstOut, abs(Float(outGPU[r * d + c]) - o) / max(1, abs(o)))
            }
        }
        #expect(worstAct < 0.02, "gate/up: worst relative error \(worstAct)")
        #expect(worstOut < 0.02, "down: worst relative error \(worstOut) at scale \(scaleOut)")
    }

    /// Qwen3.8's routed shape: 16 experts x 352 rows, D 2560, F 640, 4-bit.
    /// Prints GPU time and TFLOP/s. TINYTITAN_ROUTED_QMM_BENCH=1 to run.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_ROUTED_QMM_BENCH"] == "1"))
    func benchmark() throws {
        let (d, f) = (2_560, 640)
        for edge in [32, 64] {
        let fx = try Self.fixture(
            bits: 4, d: d, f: f, rowsPerExpert: Array(repeating: 352, count: 16), reference: false,
            edge: edge)
        let (gateUp, down) = try Self.pipelines(fx.ctx, bits: 4, edge: edge)
        guard let x = fx.ctx.device.makeBuffer(length: fx.rows * d * 2, options: .storageModePrivate),
            let act = fx.ctx.device.makeBuffer(length: fx.rows * f * 2, options: .storageModePrivate),
            let out = fx.ctx.device.makeBuffer(length: fx.rows * d * 2, options: .storageModePrivate)
        else { throw MetalError.commandEncoderFailed }
        for _ in 0..<3 {
            for (name, pso, isDown) in [("gate_up", gateUp, false), ("down", down, true)] {
                guard let cb = fx.ctx.queue.makeCommandBuffer() else { throw MetalError.commandEncoderFailed }
                try Self.encode(
                    fx, cb, pso: pso, x: isDown ? act : x, y: isDown ? out : act,
                    params: Self.params(fx, down: isDown), n: isDown ? d : f, edge: edge)
                cb.commit()
                cb.waitUntilCompleted()
                let seconds = cb.gpuEndTime - cb.gpuStartTime
                let flops = Double(fx.rows * d * f * 2) * (isDown ? 1 : 2)
                print(String(format: "[routed-qmm-bench] %dx%d %@ %.2f ms %.2f TFLOP/s", edge, edge, name, seconds * 1000, flops / seconds / 1e12))
            }
        }
        }
    }

    /// Every `prefill_dense_qmm_<BM>x<BN>` against the MPP QMM on the same
    /// weights: agreement, and with TINYTITAN_ROUTED_QMM_BENCH=1 GPU times at
    /// Qwen3.8's GDN input (N 16,384, K 2,560) and output (N 2,560, K 6,144)
    /// projections, each kernel run alone.
    @Test(arguments: [(96, 256, 512), (4_096, 16_384, 2_560), (4_096, 2_560, 6_144)])
    func denseMatchesMPP(m: Int, n: Int, k: Int) throws {
        let bench = ProcessInfo.processInfo.environment["TINYTITAN_ROUTED_QMM_BENCH"] == "1"
        if m > 96 && !bench { return }
        let ctx = try MetalContext()
        let mpp = MPPPrefillInt4QMM(context: ctx, weightBits: 4)
        var rng = LCG(state: UInt64(n))
        let packed = n * k / 2
        let groups = n * (k / 64)
        guard let w = ctx.device.makeBuffer(length: packed + 4 * groups, options: .storageModeShared)
        else { throw MetalError.commandEncoderFailed }
        _ = Self.writeProjection(w.contents(), base: 0, n: n, k: k, bits: 4, rng: &rng)
        let xs = (0..<(m * k)).map { _ in Float16(rng.unit()) }
        guard let x = ctx.device.makeBuffer(bytes: xs, length: xs.count * 2, options: .storageModeShared),
            let y1 = ctx.device.makeBuffer(length: m * n * 2, options: .storageModeShared),
            let y2 = ctx.device.makeBuffer(length: m * n * 2, options: .storageModeShared)
        else { throw MetalError.commandEncoderFailed }
        let flops = Double(2 * m * n * k)
        func timed(_ encode: (MTLCommandBuffer) throws -> Void) throws -> Double {
            var best = Double.infinity
            for _ in 0..<(bench ? 3 : 1) {
                guard let cb = ctx.queue.makeCommandBuffer() else { throw MetalError.commandEncoderFailed }
                try encode(cb)
                cb.commit()
                cb.waitUntilCompleted()
                best = min(best, cb.gpuEndTime - cb.gpuStartTime)
            }
            return best
        }
        let mppTime = try timed { cb in
            _ = try mpp.encode(
                commandBuffer: cb, weights: w, scales: w, scalesOffset: packed,
                biases: w, biasesOffset: packed + 2 * groups, x: x, y: y1, m: m, n: n, k: k,
                required: true)
        }
        if bench {
            print(String(format: "[dense-qmm-bench] m%d n%d k%d mpp %.2f ms %.2f TF", m, n, k, mppTime * 1000, flops / mppTime / 1e12))
        }
        for (bm, bn) in [(32, 32), (64, 32), (32, 64), (64, 64)] {
            let pso = try ctx.pipeline(
                "prefill_dense_qmm_\(bm)x\(bn)",
                constants: [MetalFunctionConstant(index: 78, value: .uint32(4))])
            memset(y2.contents(), 0, m * n * 2)
            let t = try timed { cb in
                guard let enc = cb.makeComputeCommandEncoder() else { throw MetalError.commandEncoderFailed }
                enc.setComputePipelineState(pso)
                enc.setBuffer(x, offset: 0, index: 0)
                enc.setBuffer(y2, offset: 0, index: 1)
                enc.setBuffer(w, offset: 0, index: 2)
                var mm = UInt32(m)
                enc.setBytes(&mm, length: 4, index: 3)
                var p: [UInt32] = [UInt32(n), UInt32(k), 0, 0, 0, 0, 0, 0, 64]
                enc.setBytes(&p, length: p.count * 4, index: 4)
                enc.setBuffer(w, offset: packed, index: 5)
                enc.setBuffer(w, offset: packed + 2 * groups, index: 6)
                enc.dispatchThreadgroups(
                    MTLSize(width: (n + bn - 1) / bn, height: (m + bm - 1) / bm, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                enc.endEncoding()
            }
            if bench {
                print(String(format: "[dense-qmm-bench] m%d n%d k%d tiled %dx%d %.2f ms %.2f TF", m, n, k, bm, bn, t * 1000, flops / t / 1e12))
            }
            let a = y1.contents().bindMemory(to: Float16.self, capacity: m * n)
            let b = y2.contents().bindMemory(to: Float16.self, capacity: m * n)
            var worst: Float = 0
            for i in 0..<(m * n) {
                worst = max(worst, abs(Float(a[i]) - Float(b[i])) / max(1, abs(Float(a[i]))))
            }
            #expect(worst < 0.02, "\(bm)x\(bn) vs MPP: worst relative difference \(worst)")
        }
    }
}
