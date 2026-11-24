import Foundation
import Metal
import Emone

/// `EmoneBench p1 [iterations]`: the routed phase-1 gate+up kernel on real
/// resident experts with real-shaped activations, checked against a CPU
/// reference.
///
/// Why a second harness: the `moe*` modes wrote CPU addresses
/// (`contents()`) into the RoutedBlobs argument buffer. The GPU address of a
/// buffer is `gpuAddress`, a different virtual address, so every expert read
/// went to an unmapped page: the kernels ran on zeros (acts all 0) at a
/// fraction of the real bandwidth. The synthetic blobs were also never made
/// resident (`useResource`).
///
/// What it does: reads `topK` experts from each of the 40 layer files of an
/// installed model (seeded pick) into one shared buffer, so a pass over the
/// 40 layer sets reads ~566 MB from DRAM rather than a cache-resident 14 MB,
/// as warm decode does. `x` is a seeded unit-RMS normal vector (the shape of
/// a post-RMSNorm hidden state). Every launch writes its own layer's slice of
/// `acts`, so the whole output is checked against an fp32 CPU reference that
/// factors the dot product the way the shader does.
///
/// Env:
///   EMONE_BENCH_P1_MODEL     install dir (default models/qwen3.6_35B_A3B_4Bit)
///   EMONE_BENCH_P1_VARIANTS  comma list (default: all), see `variants`
///   EMONE_BENCH_P1_REPEATS   timed command buffers per variant (default 7)
/// lint:allow-long EmoneBench is a development harness: one linear measurement
/// script whose setup, dispatch and reporting only make sense read top to bottom.
extension EmoneBench {
    struct P1Variant {
        let name: String
        let kernel: String
        let threads: Int
        let rowsPerSimdgroup: Int
    }

    static let p1Variants: [P1Variant] = [
        P1Variant(
            name: "base", kernel: "moe_phase1_gate_up_act_u16load", threads: 512,
            rowsPerSimdgroup: 1),
        // The existing x-from-device 16-row kernel (dev_vec helper).
        P1Variant(
            name: "r16dev", kernel: "moe_phase1_gate_up_act_u16load_r16", threads: 512,
            rowsPerSimdgroup: 1),
        P1Variant(name: "coopx", kernel: "moe_phase1_p1_coopx", threads: 512, rowsPerSimdgroup: 1),
        P1Variant(
            name: "coopx256", kernel: "moe_phase1_p1_coopx", threads: 256, rowsPerSimdgroup: 1),
        P1Variant(
            name: "w1tg512", kernel: "moe_phase1_p1_wide_r1_tg", threads: 512, rowsPerSimdgroup: 1),
        P1Variant(
            name: "w1tg256", kernel: "moe_phase1_p1_wide_r1_tg", threads: 256, rowsPerSimdgroup: 1),
        P1Variant(
            name: "w2tg512", kernel: "moe_phase1_p1_wide_r2_tg", threads: 512, rowsPerSimdgroup: 2),
        P1Variant(
            name: "w2tg256", kernel: "moe_phase1_p1_wide_r2_tg", threads: 256, rowsPerSimdgroup: 2),
        P1Variant(
            name: "w4tg512", kernel: "moe_phase1_p1_wide_r4_tg", threads: 512, rowsPerSimdgroup: 4),
        P1Variant(
            name: "w4tg256", kernel: "moe_phase1_p1_wide_r4_tg", threads: 256, rowsPerSimdgroup: 4),
        P1Variant(
            name: "w1dev512", kernel: "moe_phase1_p1_wide_r1_dev", threads: 512, rowsPerSimdgroup: 1
        ),
        P1Variant(
            name: "w1dev256", kernel: "moe_phase1_p1_wide_r1_dev", threads: 256, rowsPerSimdgroup: 1
        ),
        P1Variant(
            name: "w2dev512", kernel: "moe_phase1_p1_wide_r2_dev", threads: 512, rowsPerSimdgroup: 2
        ),
        P1Variant(
            name: "w2dev256", kernel: "moe_phase1_p1_wide_r2_dev", threads: 256, rowsPerSimdgroup: 2
        ),
        P1Variant(
            name: "w4dev512", kernel: "moe_phase1_p1_wide_r4_dev", threads: 512, rowsPerSimdgroup: 4
        ),
        P1Variant(
            name: "w4dev256", kernel: "moe_phase1_p1_wide_r4_dev", threads: 256, rowsPerSimdgroup: 4
        ),
    ]

    // lint:allow-long development harness, read top to bottom (see above)
    static func runP1(iterations: Int, context: MetalContext) throws {
        let env = ProcessInfo.processInfo.environment
        let device = context.device
        let modelDir = env["EMONE_BENCH_P1_MODEL"] ?? "models/qwen3.6_35B_A3B_4Bit"
        let repeats = Int(env["EMONE_BENCH_P1_REPEATS"] ?? "") ?? 7
        let D = 2048, F = 512, topK = 8, layers = 40
        let off = CPUExpertFFN.Offsets(d: D, f: F)
        let stride = off.strideBytes

        // --- real experts, one shared buffer, 16-byte aligned slots ---------
        guard
            let blobs = device.makeBuffer(
                length: layers * topK * stride, options: .storageModeShared),
            let argBuf = device.makeBuffer(length: layers * 16 * 8, options: .storageModeShared),
            let ioStatus = device.makeBuffer(length: 16, options: .storageModeShared)
        else { throw BenchHarnessError.metalObjectUnavailable("p1 buffers") }
        memset(ioStatus.contents(), 0, 16)
        var rng: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next() -> UInt64 {
            rng ^= rng << 13
            rng ^= rng >> 7
            rng ^= rng << 17
            return rng
        }
        let args = argBuf.contents().assumingMemoryBound(to: UInt64.self)
        for layer in 0..<layers {
            let path = String(format: "%@/packed_experts/layer_%02d.bin", modelDir, layer)
            let fd = open(path, O_RDONLY)
            guard fd >= 0 else { throw BenchHarnessError.metalObjectUnavailable("open \(path)") }
            defer { close(fd) }
            let experts = Int(lseek(fd, 0, SEEK_END)) / stride
            var picked = Set<Int>()
            while picked.count < topK { picked.insert(Int(next() % UInt64(experts))) }
            for (slot, expert) in picked.sorted().enumerated() {
                let at = (layer * topK + slot) * stride
                let got = pread(
                    fd, blobs.contents().advanced(by: at), stride, off_t(expert * stride))
                guard got == stride else {
                    throw BenchHarnessError.metalObjectUnavailable("short read \(path)")
                }
                args[layer * 16 + slot] = blobs.gpuAddress + UInt64(at)
            }
        }

        // --- x: seeded unit-RMS normal, fp16 ---------------------------------
        guard let x = device.makeBuffer(length: D * 2, options: .storageModeShared),
            let acts = device.makeBuffer(length: layers * topK * F * 2, options: .storageModeShared)
        else { throw BenchHarnessError.metalObjectUnavailable("x/acts") }
        let xh = x.contents().assumingMemoryBound(to: UInt16.self)
        var xf = [Float](repeating: 0, count: D)
        for i in 0..<D {
            let u1 = (Double(next() % 1_000_000) + 0.5) / 1_000_000
            let u2 = Double(next() % 1_000_000) / 1_000_000
            let v = Float(sqrt(-2 * log(u1)) * cos(2 * Double.pi * u2))
            xh[i] = Float16(v).bitPattern
            xf[i] = Float(Float16(v))
        }

        // --- CPU reference (fp32, shader-factored) -------------------------
        func bf16(_ p: UnsafeRawPointer, _ i: Int) -> Float {
            Float(bitPattern: UInt32(p.load(fromByteOffset: i * 2, as: UInt16.self)) << 16)
        }
        var reference = [Float](repeating: 0, count: layers * topK * F)
        let groups = D / 64
        for e in 0..<(layers * topK) {
            let blob = UnsafeRawPointer(blobs.contents().advanced(by: e * stride))
            for f in 0..<F {
                var gu: [Float] = [0, 0]
                for (m, (w, s, b)) in [
                    (off.gate, off.gateScales, off.gateBiases),
                    (off.up, off.upScales, off.upBiases),
                ]
                .enumerated() {
                    var acc: Float = 0
                    for g in 0..<groups {
                        var dot: Float = 0
                        var sum: Float = 0
                        for k in 0..<32 {
                            let byte = blob.load(
                                fromByteOffset: w + f * D / 2 + g * 32 + k, as: UInt8.self)
                            let e0 = xf[g * 64 + 2 * k], e1 = xf[g * 64 + 2 * k + 1]
                            dot += Float(byte & 0x0F) * e0 + Float(byte >> 4) * e1
                            sum += e0 + e1
                        }
                        acc +=
                            bf16(blob + s, f * groups + g) * dot + bf16(blob + b, f * groups + g)
                            * sum
                    }
                    gu[m] = acc
                }
                reference[e * F + f] = gu[0] / (1 + exp(-gu[0])) * gu[1]
            }
        }
        let refMax = reference.map(abs).max() ?? 0

        // Production constants (MoE.swift: moeConstants for 4-bit, SiLU, host-gated I/O).
        let constants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 0, value: .uint32(UInt32(D))),
            MetalFunctionConstant(index: 1, value: .uint32(UInt32(F))),
            MetalFunctionConstant(index: 2, value: .uint32(UInt32(topK))),
            MetalFunctionConstant(index: 3, value: .bool(true)),
            MetalFunctionConstant(index: 4, value: .bool(true)),
            MetalFunctionConstant(index: 6, value: .bool(false)),
        ]
        // Bytes a launch must read: gate+up weights, scales and biases.
        let bytesPerLaunch = topK * (off.down - off.gate)
        let wanted = env["EMONE_BENCH_P1_VARIANTS"].map {
            Set($0.split(separator: ",").map(String.init))
        }
        var baseActs: [UInt16]?
        print(
            "p1: \(modelDir), \(layers) layers x \(topK) experts (\(layers * topK * stride / 1_000_000) MB), "
                + "\(bytesPerLaunch) B/launch, iterations \(iterations), repeats \(repeats), ref max |a| \(refMax)"
        )
        for v in p1Variants where wanted?.contains(v.name) ?? true {
            let pso = try context.pipeline(
                v.kernel, constants: constants, maxTotalThreadsPerThreadgroup: v.threads)
            let rowsPerTG = v.threads / 32 * v.rowsPerSimdgroup
            let groupsPerLaunch = (topK * F + rowsPerTG - 1) / rowsPerTG
            var Dv = UInt32(D), Fv = UInt32(F), K = UInt32(topK)
            var offsets = MoEExpertOffsets(
                gateWOff: UInt32(off.gate), gateSOff: UInt32(off.gateScales),
                gateBOff: UInt32(off.gateBiases),
                upWOff: UInt32(off.up), upSOff: UInt32(off.upScales), upBOff: UInt32(off.upBiases),
                downWOff: UInt32(off.down), downSOff: UInt32(off.downScales),
                downBOff: UInt32(off.downBiases))
            memset(acts.contents(), 0, acts.length)
            var perLaunch: [Double] = []
            for rep in 0...repeats {  // rep 0 is the warm-up
                let cb = try requireCommandBuffer(context.queue)
                guard let enc = cb.makeComputeCommandEncoder() else {
                    throw BenchHarnessError.metalObjectUnavailable("encoder")
                }
                enc.useResource(blobs, usage: .read)
                enc.setComputePipelineState(pso)
                enc.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
                enc.setBuffer(x, offset: 0, index: 2)
                enc.setBytes(&Dv, length: 4, index: 4)
                enc.setBytes(&Fv, length: 4, index: 5)
                enc.setBytes(&K, length: 4, index: 6)
                enc.setBuffer(ioStatus, offset: 0, index: 7)
                for i in 0..<iterations {
                    let layer = i % layers
                    enc.setBuffer(argBuf, offset: layer * 16 * 8, index: 0)
                    enc.setBuffer(acts, offset: layer * topK * F * 2, index: 3)
                    enc.dispatchThreadgroups(
                        MTLSize(width: groupsPerLaunch, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: v.threads, height: 1, depth: 1))
                }
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                if let err = cb.error { throw err }
                if rep > 0 {
                    perLaunch.append((cb.gpuEndTime - cb.gpuStartTime) / Double(iterations))
                }
            }
            let sorted = perLaunch.sorted()
            let median = sorted[sorted.count / 2]
            let got = Array(
                UnsafeBufferPointer(
                    start: acts.contents().assumingMemoryBound(to: UInt16.self),
                    count: layers * topK * F))
            var maxErr: Float = 0
            var zeros = 0
            for i in 0..<got.count {
                let a = Float(Float16(bitPattern: got[i]))
                if got[i] == 0 { zeros += 1 }
                maxErr = max(maxErr, abs(a - reference[i]))
            }
            if v.name == "base" { baseActs = got }
            let diffVsBase = baseActs.map { b in zip(b, got).filter { $0 != $1 }.count }
            print(
                String(
                    format:
                        "%-9@ %6.2f us  %6.1f GB/s  (min %6.2f max %6.2f)  max|err| %.3g (%.2g of max)  zeros %d  "
                        + "differ-vs-base %@",
                    v.name as NSString, median * 1e6, Double(bytesPerLaunch) / median / 1e9,
                    sorted[0] * 1e6, sorted[sorted.count - 1] * 1e6, maxErr,
                    maxErr / max(refMax, 1e-30),
                    zeros, (diffVsBase.map { "\($0)/\(got.count)" } ?? "n/a") as NSString))
        }
    }
}

extension EmoneBench {
    /// `EmoneBench p1rows [iterations]`: the resident-prefill row-batched
    /// phase 1 (`moe_phase1_gate_up_act_u16load_rows`, `encodeRoutedRows`)
    /// against `moe_phase1_p1_rows_dev`, which reads each token's x from
    /// device memory instead of staging it per simdgroup. Real experts: token
    /// r uses layer r's 8 picked experts. EMONE_BENCH_P1_ROWS sets the token
    /// count (default 16, at most 64). Outputs are compared bit for bit.
    // lint:allow-long development harness, read top to bottom
    static func runP1Rows(iterations: Int, context: MetalContext) throws {
        let env = ProcessInfo.processInfo.environment
        let device = context.device
        let modelDir = env["EMONE_BENCH_P1_MODEL"] ?? "models/qwen3.6_35B_A3B_4Bit"
        let repeats = Int(env["EMONE_BENCH_P1_REPEATS"] ?? "") ?? 7
        let rows = min(64, Int(env["EMONE_BENCH_P1_ROWS"] ?? "") ?? 16)
        let D = 2048, F = 512, topK = 8
        let off = CPUExpertFFN.Offsets(d: D, f: F)
        let stride = off.strideBytes
        guard
            let blobs = device.makeBuffer(
                length: rows * topK * stride, options: .storageModeShared),
            let argBuf = device.makeBuffer(length: 512 * 8, options: .storageModeShared),
            let x = device.makeBuffer(length: rows * D * 2, options: .storageModeShared),
            let acts = device.makeBuffer(length: rows * topK * F * 2, options: .storageModeShared)
        else { throw BenchHarnessError.metalObjectUnavailable("p1rows buffers") }
        var rng: UInt64 = 0x2545_F491_4F6C_DD1D
        func next() -> UInt64 {
            rng ^= rng << 13
            rng ^= rng >> 7
            rng ^= rng << 17
            return rng
        }
        let args = argBuf.contents().assumingMemoryBound(to: UInt64.self)
        for r in 0..<rows {
            let path = String(format: "%@/packed_experts/layer_%02d.bin", modelDir, r % 40)
            let fd = open(path, O_RDONLY)
            guard fd >= 0 else { throw BenchHarnessError.metalObjectUnavailable("open \(path)") }
            defer { close(fd) }
            let experts = Int(lseek(fd, 0, SEEK_END)) / stride
            var picked = Set<Int>()
            while picked.count < topK { picked.insert(Int(next() % UInt64(experts))) }
            for (slot, expert) in picked.sorted().enumerated() {
                let at = (r * topK + slot) * stride
                guard
                    pread(fd, blobs.contents().advanced(by: at), stride, off_t(expert * stride))
                        == stride
                else { throw BenchHarnessError.metalObjectUnavailable("short read \(path)") }
                args[r * topK + slot] = blobs.gpuAddress + UInt64(at)
            }
        }
        let xh = x.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<(rows * D) {
            let u1 = (Double(next() % 1_000_000) + 0.5) / 1_000_000
            let u2 = Double(next() % 1_000_000) / 1_000_000
            xh[i] = Float16(Float(sqrt(-2 * log(u1)) * cos(2 * Double.pi * u2))).bitPattern
        }
        let constants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 0, value: .uint32(UInt32(D))),
            MetalFunctionConstant(index: 1, value: .uint32(UInt32(F))),
            MetalFunctionConstant(index: 2, value: .uint32(UInt32(topK))),
            MetalFunctionConstant(index: 3, value: .bool(true)),
            MetalFunctionConstant(index: 4, value: .bool(true)),
            MetalFunctionConstant(index: 6, value: .bool(false)),
        ]
        let bytes = rows * topK * (off.down - off.gate)
        print(
            "p1rows: \(rows) tokens x \(topK) experts, \(bytes) B/launch, iterations \(iterations)")
        var baseline: [UInt16]?
        for (name, kernel) in [
            ("rows", "moe_phase1_gate_up_act_u16load_rows"), ("rows_dev", "moe_phase1_p1_rows_dev"),
        ] {
            let pso = try context.pipeline(
                kernel, constants: constants, maxTotalThreadsPerThreadgroup: 512)
            var Dv = UInt32(D), Fv = UInt32(F), K = UInt32(topK)
            var offsets = MoEExpertOffsets(
                gateWOff: UInt32(off.gate), gateSOff: UInt32(off.gateScales),
                gateBOff: UInt32(off.gateBiases),
                upWOff: UInt32(off.up), upSOff: UInt32(off.upScales), upBOff: UInt32(off.upBiases),
                downWOff: UInt32(off.down), downSOff: UInt32(off.downScales),
                downBOff: UInt32(off.downBiases))
            memset(acts.contents(), 0, acts.length)
            var times: [Double] = []
            for rep in 0...repeats {
                let cb = try requireCommandBuffer(context.queue)
                guard let enc = cb.makeComputeCommandEncoder() else {
                    throw BenchHarnessError.metalObjectUnavailable("encoder")
                }
                enc.useResource(blobs, usage: .read)
                enc.setComputePipelineState(pso)
                enc.setBuffer(argBuf, offset: 0, index: 0)
                enc.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
                enc.setBuffer(x, offset: 0, index: 2)
                enc.setBuffer(acts, offset: 0, index: 3)
                enc.setBytes(&Dv, length: 4, index: 4)
                enc.setBytes(&Fv, length: 4, index: 5)
                enc.setBytes(&K, length: 4, index: 6)
                for _ in 0..<iterations {
                    enc.dispatchThreadgroups(
                        MTLSize(width: (topK * F + 15) / 16, height: rows, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
                }
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                if let err = cb.error { throw err }
                if rep > 0 { times.append((cb.gpuEndTime - cb.gpuStartTime) / Double(iterations)) }
            }
            let sorted = times.sorted()
            let median = sorted[sorted.count / 2]
            let got = Array(
                UnsafeBufferPointer(
                    start: acts.contents().assumingMemoryBound(to: UInt16.self),
                    count: rows * topK * F))
            if baseline == nil { baseline = got }
            let differ = zip(baseline ?? got, got).filter { $0 != $1 }.count
            let zeros = got.filter { $0 == 0 }.count
            print(
                String(
                    format:
                        "%-9@ %8.2f us  %6.1f GB/s  (min %8.2f max %8.2f)  zeros %d  differ-vs-rows %d/%d",
                    name as NSString, median * 1e6, Double(bytes) / median / 1e9, sorted[0] * 1e6,
                    sorted[sorted.count - 1] * 1e6, zeros, differ, got.count))
        }
    }
}
