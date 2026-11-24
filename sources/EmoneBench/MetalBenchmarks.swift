//
//  MetalBenchmarks.swift
//  EmoneBench
//
//  The Metal kernel benchmarks: routed-MoE decode, GDN, and attention-head
//  shapes. Split out of `main.swift` so each file holds one family of
//  measurements; the dispatch that selects between them is in `main.swift`.
//

import Foundation
import Metal
import Emone

/// A Metal object the harness needs but could not create. The benchmark
/// harnesses used to force-unwrap these, so a machine under memory pressure
/// crashed instead of reporting which allocation failed.
enum BenchHarnessError: Error, CustomStringConvertible {
    case metalObjectUnavailable(String)

    var description: String {
        switch self {
        case .metalObjectUnavailable(let what):
            return "could not create the Metal \(what) for this benchmark"
        }
    }
}

/// A command buffer from the queue, or a named error.
func requireCommandBuffer(_ queue: MTLCommandQueue) throws -> MTLCommandBuffer {
    guard let cb = queue.makeCommandBuffer() else {
        throw BenchHarnessError.metalObjectUnavailable("command buffer")
    }
    return cb
}

extension EmoneBench {
    /// Shader-side `ExpertOffsets` mirror: 9 packed UInt32 in the same order.
    private struct MoEBenchOffsets {
        var gateW: UInt32
        var gateS: UInt32
        var gateB: UInt32
        var upW: UInt32
        var upS: UInt32
        var upB: UInt32
        var downW: UInt32
        var downS: UInt32
        var downB: UInt32
    }

    /// Routed-MoE decode kernels at the real 4-bit shapes. Weights are the
    /// metric: phase-1 reads gate+up (8 x 2 x F*D/2 bytes), phase-2 reads
    /// down (8 x D*F/2 bytes). The combined "moe" mode dispatches both in
    /// one command buffer, mirroring the decode routedCB.
    /// lint:allow-long EmoneBench is a development harness, not a shipped
    /// product: each run* is one linear measurement script whose setup,
    /// dispatch and reporting only make sense read top to bottom.
    static func runMoE(
        kernelName: String,
        iterations: Int,
        context: MetalContext
    ) throws {
        let device = context.device
        // Default shape is Qwen 3.6 35B-A3B. EMONE_BENCH_MOE_SHAPE=qwen38
        // selects Qwen3.8-Flash-Next's routed expert (D 2560, F 640, top-10),
        // the shape the decode profile's moe_* numbers come from.
        let qwen38 = ProcessInfo.processInfo.environment["EMONE_BENCH_MOE_SHAPE"] == "qwen38"
        let D: UInt32 = qwen38 ? 2560 : 2048  // hiddenSize
        let F: UInt32 = qwen38 ? 640 : 512  // moeIntermediateSize
        let topK: UInt32 = qwen38 ? 10 : 8
        let groupCount = Int(D) / 64  // kMoEGroupSize = 64 elements

        // Per-blob 4-bit layout: gate_W, gate_s, gate_b, up_W, up_s, up_b,
        // down_W, down_s, down_b.
        let fD2 = Int(F) * Int(D) / 2
        let groups = Int(F) * groupCount * 2
        let gateWOff = 0
        let gateSOff = fD2
        let gateBOff = gateSOff + groups
        let upWOff = gateBOff + groups
        let upSOff = upWOff + fD2
        let upBOff = upSOff + groups
        let downWOff = upBOff + groups
        let downSOff = downWOff + Int(D) * Int(F) / 2
        let downBOff = downSOff + groups
        let blobBytes = downBOff + groups

        func makeBuffer(_ bytes: Int, _ value: UInt8) throws -> MTLBuffer {
            guard
                let buf = device.makeBuffer(
                    length: bytes,
                    options: .storageModeShared)
            else {
                throw BenchHarnessError.metalObjectUnavailable("buffer of \(bytes) bytes")
            }
            memset(buf.contents(), Int32(value), bytes)
            return buf
        }
        var blobs: [MTLBuffer] = []
        for i in 0..<Int(topK) {
            let blob = try makeBuffer(blobBytes, UInt8(0x11 + i))
            // Scale/bias regions as bf16 0x3C3C (~0.011) rather than the
            // byte pattern (~1e-38), so the activations are non-zero and
            // acts_fnv actually compares kernel variants.
            for off in [gateSOff, gateBOff, upSOff, upBOff, downSOff, downBOff] {
                memset(blob.contents().advanced(by: off), 0x3C, groups)
            }
            blobs.append(blob)
        }
        // Non-uniform activation pattern so staging/indexing bugs surface:
        // the uniform fill used before masked wrong-element reads.
        let x = try makeBuffer(Int(D) * 2, 0)
        let xPtr = x.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<Int(D) {
            xPtr[i] = Float16(Float(i + 1) * 0.001).bitPattern
        }
        let acts = try makeBuffer(Int(topK) * Int(F) * 2, 0)
        let routingW = try makeBuffer(Int(topK) * 2, 0x3F)
        let residual = try makeBuffer(Int(D) * 2, 0x33)
        let y = try makeBuffer(Int(D) * 2, 0)

        // RoutedBlobs arg buffer: 8 device pointers. These must be GPU
        // addresses (`gpuAddress`); the CPU address (`contents()`) is a
        // different virtual address, and writing it here made every expert
        // read fault to zeros. The blobs are made resident below.
        guard
            let argBuf = device.makeBuffer(
                length: Int(topK) * 8,
                options: .storageModeShared)
        else {
            fatalError("arg buffer alloc failed")
        }
        let argPtr = argBuf.contents().assumingMemoryBound(to: UInt64.self)
        for i in 0..<Int(topK) {
            argPtr[i] = blobs[i].gpuAddress
        }

        // DIAG (local only): EMONE_BENCH_MOE_REAL=<packed_experts dir> reads 8
        // real experts per layer from each of the 40 layer files and rotates
        // the layer set every iteration, so the reads come from DRAM (566 MB a
        // pass) instead of the 14 MB of fixed blobs the system cache holds.
        var argBufs: [MTLBuffer] = [argBuf]
        var realSets: [MTLBuffer] = []
        if let dir = ProcessInfo.processInfo.environment["EMONE_BENCH_MOE_REAL"] {
            argBufs = []
            var rng = SystemRandomNumberGenerator()
            for layer in 0..<40 {
                let path = String(format: "%@/layer_%02d.bin", dir, layer)
                let fd = open(path, O_RDONLY)
                guard fd >= 0 else { fatalError("cannot open \(path)") }
                let experts = Int(lseek(fd, 0, SEEK_END)) / blobBytes
                let set = try makeBuffer(Int(topK) * blobBytes, 0)
                guard
                    let arg = device.makeBuffer(length: Int(topK) * 8, options: .storageModeShared)
                else { fatalError("arg buffer alloc failed") }
                let ptrs = arg.contents().assumingMemoryBound(to: UInt64.self)
                var picked = Set<Int>()
                while picked.count < Int(topK) {
                    picked.insert(Int.random(in: 0..<experts, using: &rng))
                }
                for (slot, expert) in picked.sorted().enumerated() {
                    let dst = set.contents().advanced(by: slot * blobBytes)
                    let got = pread(fd, dst, blobBytes, off_t(expert * blobBytes))
                    guard got == blobBytes else { fatalError("short read \(path)") }
                    ptrs[slot] = set.gpuAddress + UInt64(slot * blobBytes)
                }
                close(fd)
                realSets.append(set)
                argBufs.append(arg)
            }
            print(
                "real expert blobs: 40 layers x \(topK) experts, \(40 * Int(topK) * blobBytes / 1_000_000) MB"
            )
        }

        var offsets = MoEBenchOffsets(
            gateW: UInt32(gateWOff), gateS: UInt32(gateSOff), gateB: UInt32(gateBOff),
            upW: UInt32(upWOff), upS: UInt32(upSOff), upB: UInt32(upBOff),
            downW: UInt32(downWOff), downS: UInt32(downSOff), downB: UInt32(downBOff))

        // Variant dispatch: r4/r16 change rows-per-threadgroup, xsh8/16 stage
        // the activation in threadgroup memory. The production kernel is the
        // 16-row threadgroup-staged layout, so all modes dispatch 16 rows.
        let phase1Variant: String?
        let phase1RowsPerTG: Int
        switch kernelName {
        case "moe_phase1_r8":
            phase1Variant = "moe_phase1_gate_up_act_u16load_r8"
            phase1RowsPerTG = 8
        case "moe_phase1_r16":
            phase1Variant = "moe_phase1_gate_up_act_u16load_r16"
            phase1RowsPerTG = 16
        case "moe_phase1_xsh8":
            phase1Variant = "moe_phase1_gate_up_act_u16load"
            phase1RowsPerTG = 16
        case "moe_phase1_xsh16":
            phase1Variant = "moe_phase1_gate_up_act_u16load"
            phase1RowsPerTG = 16
        case "moe_phase1_v2":
            // Two rows per simdgroup, 16 simdgroups: 32 rows per 512 threads.
            phase1Variant = "moe_phase1_gate_up_act_u16load_v2"
            phase1RowsPerTG = 32
        default:
            // The production kernel is the 16-row threadgroup-staged layout.
            phase1Variant = nil
            phase1RowsPerTG = 16
        }
        let phase1Threads = kernelName == "moe_phase1_v2" ? 512 : phase1RowsPerTG * 32
        let phase1Kernel = phase1Variant ?? "moe_phase1_gate_up_act_u16load"

        // EMONE_BENCH_MOE_SPECIALIZE=1 builds the phase-1 pipeline with the
        // runtime's function constants (D, F, top-k, silu, host-gated I/O);
        // the unspecialized kernel reads its shape from buffers and measures
        // ~35 GB/s where the specialized one moves the same blobs at ~60.
        let specialize =
            ProcessInfo.processInfo.environment["EMONE_BENCH_MOE_SPECIALIZE"] == "1"
        let phase1Constants: [MetalFunctionConstant] =
            specialize
            ? [
                MetalFunctionConstant(index: 0, value: .uint32(D)),
                MetalFunctionConstant(index: 1, value: .uint32(F)),
                MetalFunctionConstant(index: 2, value: .uint32(topK)),
                MetalFunctionConstant(index: 3, value: .bool(true)),
                MetalFunctionConstant(index: 4, value: .bool(true)),
                MetalFunctionConstant(index: 6, value: .bool(false)),
            ] : []
        let phase1PSO = try context.pipeline(
            phase1Kernel,
            constants: phase1Constants,
            maxTotalThreadsPerThreadgroup: phase1Threads)
        let phase2PSO = try context.pipeline(
            topK == 8 ? "moe_phase2_down_reduce_k8" : "moe_phase2_down_reduce_kn",
            constants: specialize ? phase1Constants : [],
            maxTotalThreadsPerThreadgroup: 256)
        let subsetPSO = try context.pipeline(
            kernelName == "moe_phase1_subset"
                ? "moe_phase1_gate_up_act_subset_u16load"
                : "moe_phase1_gate_up_act_u16load",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)

        let phase1Groups = (Int(topK) * Int(F) + phase1RowsPerTG - 1) / phase1RowsPerTG
        let phase1Bytes = UInt64(Int(topK) * 2 * Int(F) * Int(D) / 2)
        let phase2Bytes = UInt64(Int(topK) * Int(D) * Int(F) / 2)

        var Dv = D
        var Fv = F
        var TK = topK

        let cb = try requireCommandBuffer(context.queue)
        guard let enc = cb.makeComputeCommandEncoder() else {
            fatalError("could not create compute encoder")
        }
        let runPhase1 = kernelName == "moe" || kernelName.hasPrefix("moe_phase1")
        let phase1TG = MTLSize(width: phase1Threads, height: 1, depth: 1)
        let runPhase2 = kernelName == "moe_phase2" || kernelName == "moe"
        let runSubset = kernelName == "moe_phase1_subset"
        // active-slot buffer for the subset mode (all 8 experts active).
        var activeSlots = [UInt32](0..<topK)
        let activeSlotsBuf = try makeBuffer(Int(topK) * MemoryLayout<UInt32>.size, 0)
        activeSlotsBuf.contents().copyMemory(
            from: &activeSlots,
            byteCount: Int(topK) * MemoryLayout<UInt32>.size)
        var activeCount = topK
        for set in realSets { enc.useResource(set, usage: .read) }
        for blob in blobs { enc.useResource(blob, usage: .read) }
        for iteration in 0..<iterations {
            let argBuf = argBufs[iteration % argBufs.count]
            if runSubset {
                enc.setComputePipelineState(subsetPSO)
                enc.setBuffer(argBuf, offset: 0, index: 0)
                enc.setBytes(&offsets, length: MemoryLayout<MoEBenchOffsets>.stride, index: 1)
                enc.setBuffer(x, offset: 0, index: 2)
                enc.setBuffer(acts, offset: 0, index: 3)
                enc.setBytes(&Dv, length: MemoryLayout<UInt32>.size, index: 4)
                enc.setBytes(&Fv, length: MemoryLayout<UInt32>.size, index: 5)
                enc.setBytes(&TK, length: MemoryLayout<UInt32>.size, index: 6)
                enc.setBuffer(activeSlotsBuf, offset: 0, index: 7)
                enc.setBytes(&activeCount, length: MemoryLayout<UInt32>.size, index: 8)
                enc.dispatchThreadgroups(
                    MTLSize(width: (Int(activeCount * F) + 15) / 16, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
            }
            if runPhase1 {
                enc.setComputePipelineState(phase1PSO)
                enc.setBuffer(argBuf, offset: 0, index: 0)
                enc.setBytes(&offsets, length: MemoryLayout<MoEBenchOffsets>.stride, index: 1)
                enc.setBuffer(x, offset: 0, index: 2)
                enc.setBuffer(acts, offset: 0, index: 3)
                enc.setBytes(&Dv, length: MemoryLayout<UInt32>.size, index: 4)
                enc.setBytes(&Fv, length: MemoryLayout<UInt32>.size, index: 5)
                enc.setBytes(&TK, length: MemoryLayout<UInt32>.size, index: 6)
                enc.dispatchThreadgroups(
                    MTLSize(width: phase1Groups, height: 1, depth: 1),
                    threadsPerThreadgroup: phase1TG)
            }
            if runPhase2 {
                enc.setComputePipelineState(phase2PSO)
                enc.setBuffer(argBuf, offset: 0, index: 0)
                enc.setBytes(&offsets, length: MemoryLayout<MoEBenchOffsets>.stride, index: 1)
                enc.setBuffer(acts, offset: 0, index: 2)
                enc.setBuffer(routingW, offset: 0, index: 3)
                enc.setBuffer(residual, offset: 0, index: 4)
                enc.setBuffer(y, offset: 0, index: 5)
                enc.setBytes(&Dv, length: MemoryLayout<UInt32>.size, index: 6)
                enc.setBytes(&Fv, length: MemoryLayout<UInt32>.size, index: 7)
                enc.dispatchThreadgroups(
                    MTLSize(width: Int(D), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        enc.endEncoding()
        cb.commitEncoded()
        cb.waitUntilCompleted()
        if let err = cb.error {
            print("COMMAND BUFFER ERROR: \(err)")
        }

        let totalSeconds = cb.gpuEndTime - cb.gpuStartTime
        let perIteration = totalSeconds / Double(iterations)
        let bytes: UInt64 = (runPhase1 ? phase1Bytes : 0) + (runPhase2 ? phase2Bytes : 0)
        let gbPerSec = Double(bytes) / perIteration / 1_000_000_000
        let theoretical = 100.0
        // Correctness probe: FNV-1a over the acts buffer (the phase-1 output)
        // and the y buffer (the phase-2 output).
        var hash: UInt32 = 0x811c_9dc5
        let actsPtr = acts.contents().assumingMemoryBound(to: UInt8.self)
        for i in 0..<min(acts.length, 8192) {
            hash ^= UInt32(actsPtr[i])
            hash &*= 0x0100_0193
        }
        var yHash: UInt32 = 0x811c_9dc5
        let yPtr = y.contents().assumingMemoryBound(to: UInt8.self)
        for i in 0..<min(y.length, 8192) {
            yHash ^= UInt32(yPtr[i])
            yHash &*= 0x0100_0193
        }
        let actsHalf = acts.contents().assumingMemoryBound(to: UInt16.self)
        let sample = (0..<min(16, acts.length / 2)).map { String(format: "%04x", actsHalf[$0]) }
            .joined(separator: " ")
        print(
            "kernel=\(kernelName) iterations=\(iterations) "
                + "total=\(String(format: "%.4f", totalSeconds))s "
                + "per_launch=\(String(format: "%.2f", perIteration * 1_000_000))us")
        print(
            "bytes/launch=\(bytes) (phase1=\(phase1Bytes) phase2=\(phase2Bytes)) "
                + "achieved=\(String(format: "%.1f", gbPerSec)) GB/s "
                + "efficiency=\(String(format: "%.0f", gbPerSec / theoretical * 100))% of ~100 GB/s peak "
                + "acts_fnv=\(String(format: "%08x", hash)) y_fnv=\(String(format: "%08x", yHash)) "
                + "acts_sample=\(sample)")
    }

    /// GDN fused input-projection GEMV at the real qwen36 shapes
    /// (qkvDim 8192, valueDim 4096, ab 32 each, N=2048). The weight read is
    /// the metric: qkv + z + a + b ~= 12.7 MB/layer. Variants:
    /// gdn_inproj (production 8-row), gdn_inproj_xsh8 (8-row + tgmem x),
    /// gdn_inproj_r16 (16-row device x), gdn_inproj_xsh16 (16-row + tgmem).
    /// lint:allow-long EmoneBench is a development harness, not a shipped
    /// product: each run* is one linear measurement script whose setup,
    /// dispatch and reporting only make sense read top to bottom.
    static func runGDN(
        kernelName: String,
        iterations: Int,
        context: MetalContext
    ) throws {
        let device = context.device
        // Default shape is the Qwen 3.6 GDN layer. EMONE_BENCH_GDN_SHAPE=qwen38
        // selects Qwen3.8-Flash-Next's (16 k-heads x 128 + 48 v-heads x 128
        // for qkv, 48 x 128 for z, 48 for a/b, hidden 2560), the shape the
        // decode profile's gdn.inproj number comes from.
        let qwen38 = ProcessInfo.processInfo.environment["EMONE_BENCH_GDN_SHAPE"] == "qwen38"
        let qkvRows: UInt32 = qwen38 ? 16384 : 8192
        let zRows: UInt32 = qwen38 ? 6144 : 4096
        let abRows: UInt32 = qwen38 ? 48 : 32
        let N: UInt32 = qwen38 ? 2560 : 2048
        let groupCount = Int(N) / 64

        func makeBuffer(_ bytes: Int, _ value: UInt8) throws -> MTLBuffer {
            guard let buf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw BenchHarnessError.metalObjectUnavailable("buffer of \(bytes) bytes")
            }
            memset(buf.contents(), Int32(value), bytes)
            return buf
        }
        let qkvW = try makeBuffer(Int(qkvRows) * Int(N) / 2, 0x12)
        let qkvS = try makeBuffer(Int(qkvRows) * groupCount * 2, 0x01)
        let qkvB = try makeBuffer(Int(qkvRows) * groupCount * 2, 0x00)
        let zW = try makeBuffer(Int(zRows) * Int(N) / 2, 0x34)
        let zS = try makeBuffer(Int(zRows) * groupCount * 2, 0x01)
        let zB = try makeBuffer(Int(zRows) * groupCount * 2, 0x00)
        let aW = try makeBuffer(Int(abRows) * Int(N) / 2, 0x56)
        let aS = try makeBuffer(Int(abRows) * groupCount * 2, 0x01)
        let aB = try makeBuffer(Int(abRows) * groupCount * 2, 0x00)
        let bW = try makeBuffer(Int(abRows) * Int(N) / 2, 0x78)
        let bS = try makeBuffer(Int(abRows) * groupCount * 2, 0x01)
        let bB = try makeBuffer(Int(abRows) * groupCount * 2, 0x00)
        let x = try makeBuffer(Int(N) * 2, 0)
        let xPtr = x.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<Int(N) {
            xPtr[i] = Float16(Float(i + 1) * 0.001).bitPattern
        }
        let qkvY = try makeBuffer(Int(qkvRows) * 2, 0)
        let zY = try makeBuffer(Int(zRows) * 2, 0)
        let aY = try makeBuffer(Int(abRows) * 2, 0)
        let bY = try makeBuffer(Int(abRows) * 2, 0)

        let kernelName2: String
        let rowsPerTG: Int
        switch kernelName {
        case "gdn_inproj_xsh8":
            kernelName2 = "gdn_in_proj_gemv_simd_xsh8"
            rowsPerTG = 8
        case "gdn_inproj_r16":
            kernelName2 = "gdn_in_proj_gemv_simd_r16"
            rowsPerTG = 16
        case "gdn_inproj_xsh16":
            kernelName2 = "gdn_in_proj_gemv_simd_xsh16"
            rowsPerTG = 16
        case "gdn_inproj_u4":
            kernelName2 = "gdn_in_proj_gemv_simd_u4"
            rowsPerTG = 8
        case "gdn_inproj_sk4":
            kernelName2 = "gdn_in_proj_gemv_simd_sk4"
            rowsPerTG = 2
        case "gdn_inproj_bw":
            kernelName2 = "gdn_in_proj_gemv_simd_bw"
            rowsPerTG = 8
        default:
            kernelName2 = "gdn_in_proj_gemv_simd"
            rowsPerTG = 8
        }
        // sk4 runs four simdgroups per row, so its threadgroup is wider than
        // rows * 32.
        let threadsPerTG = kernelName == "gdn_inproj_sk4" ? 256 : rowsPerTG * 32
        let pso = try context.pipeline(
            kernelName2, constants: [],
            maxTotalThreadsPerThreadgroup: threadsPerTG)

        // Full GDN decode chain (gdn_chain): in_proj + conv + qk_norm +
        // delta-step + gated_norm in one command buffer, mirroring the
        // decode's linear-attention block. Measures the extras' cost.
        let chain = kernelName == "gdn_chain"
        let convPSO = try context.pipeline("gdn_conv_mix_decode")
        let qkNormPSO = try context.pipeline(
            "gdn_qk_norm",
            constants: [MetalFunctionConstant(index: 95, value: .uint32(128))])
        let deltaPSO = try context.pipeline("gdn_delta_step_decode")
        let gatedNormPSO = try context.pipeline(
            "gdn_gated_norm",
            constants: [MetalFunctionConstant(index: 95, value: .uint32(128))])
        let convTail = try makeBuffer(3 * Int(qkvRows), 0x44)  // [K-1, qkvDim] halfs
        let convW = try makeBuffer(Int(qkvRows) * 4 * 2, 0x55)  // [qkvDim, K] bfloat
        let convOut = try makeBuffer(Int(qkvRows) * 2, 0)
        let aLog = try makeBuffer(Int(abRows) * 2, 0x60)
        let dtBias = try makeBuffer(Int(abRows) * 2, 0x60)
        let state = try makeBuffer(Int(abRows) * 128 * 128 * 4, 0)  // FP32 [Hv, Dv, Dk]
        let deltaY = try makeBuffer(Int(zRows) * 2, 0)
        let gatedW = try makeBuffer(Int(zRows) * 2, 0x66)
        let gatedOut = try makeBuffer(Int(zRows) * 2, 0)

        let totalRows = Int(qkvRows + zRows + 2 * abRows)
        let bytes = UInt64(
            Int(qkvRows) * Int(N) / 2 + Int(zRows) * Int(N) / 2
                + 2 * Int(abRows) * Int(N) / 2)
        var qkvVar = qkvRows
        var zVar = zRows
        var abVar = abRows
        var nVar = N

        let cb = try requireCommandBuffer(context.queue)
        guard let enc = cb.makeComputeCommandEncoder() else {
            fatalError("could not create compute encoder")
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(qkvW, offset: 0, index: 0)
        enc.setBuffer(qkvS, offset: 0, index: 1)
        enc.setBuffer(qkvB, offset: 0, index: 2)
        enc.setBuffer(zW, offset: 0, index: 3)
        enc.setBuffer(zS, offset: 0, index: 4)
        enc.setBuffer(zB, offset: 0, index: 5)
        enc.setBuffer(aW, offset: 0, index: 6)
        enc.setBuffer(aS, offset: 0, index: 7)
        enc.setBuffer(aB, offset: 0, index: 8)
        enc.setBuffer(bW, offset: 0, index: 9)
        enc.setBuffer(bS, offset: 0, index: 10)
        enc.setBuffer(bB, offset: 0, index: 11)
        enc.setBuffer(x, offset: 0, index: 12)
        enc.setBuffer(qkvY, offset: 0, index: 13)
        enc.setBuffer(zY, offset: 0, index: 14)
        enc.setBuffer(aY, offset: 0, index: 15)
        enc.setBuffer(bY, offset: 0, index: 16)
        enc.setBytes(&qkvVar, length: MemoryLayout<UInt32>.size, index: 17)
        enc.setBytes(&zVar, length: MemoryLayout<UInt32>.size, index: 18)
        enc.setBytes(&abVar, length: MemoryLayout<UInt32>.size, index: 19)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 20)
        for _ in 0..<iterations {
            enc.setComputePipelineState(pso)
            enc.dispatchThreadgroups(
                MTLSize(width: (totalRows + rowsPerTG - 1) / rowsPerTG, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerTG, height: 1, depth: 1))
            if chain {
                // conv: tail(0) qkv(1) convW(2) out(3) channels(4) taps(5)
                enc.setComputePipelineState(convPSO)
                enc.setBuffer(convTail, offset: 0, index: 0)
                enc.setBuffer(qkvY, offset: 0, index: 1)
                enc.setBuffer(convW, offset: 0, index: 2)
                enc.setBuffer(convOut, offset: 0, index: 3)
                var ch = qkvRows
                var taps: UInt32 = 4
                enc.setBytes(&ch, length: MemoryLayout<UInt32>.size, index: 4)
                enc.setBytes(&taps, length: MemoryLayout<UInt32>.size, index: 5)
                enc.dispatchThreads(
                    MTLSize(width: Int(ch), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                // qk_norm: convOut(0) kHeads(1) keyDim(2) rowStride(3)
                enc.setComputePipelineState(qkNormPSO)
                enc.setBuffer(convOut, offset: 0, index: 0)
                var kHeads: UInt32 = 16
                var keyDim: UInt32 = 128
                var rowStride = qkvRows
                enc.setBytes(&kHeads, length: MemoryLayout<UInt32>.size, index: 1)
                enc.setBytes(&keyDim, length: MemoryLayout<UInt32>.size, index: 2)
                enc.setBytes(&rowStride, length: MemoryLayout<UInt32>.size, index: 3)
                enc.dispatchThreadgroups(
                    MTLSize(width: 2 * 16, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                // delta: convOut(0) aProj(1) bProj(2) aLog(3) dtBias(4) state(5) y(6) dims 7-10
                enc.setComputePipelineState(deltaPSO)
                enc.setBuffer(convOut, offset: 0, index: 0)
                enc.setBuffer(aY, offset: 0, index: 1)
                enc.setBuffer(bY, offset: 0, index: 2)
                enc.setBuffer(aLog, offset: 0, index: 3)
                enc.setBuffer(dtBias, offset: 0, index: 4)
                enc.setBuffer(state, offset: 0, index: 5)
                enc.setBuffer(deltaY, offset: 0, index: 6)
                var vHeads: UInt32 = 32
                var vDim: UInt32 = 128
                enc.setBytes(&kHeads, length: MemoryLayout<UInt32>.size, index: 7)
                enc.setBytes(&vHeads, length: MemoryLayout<UInt32>.size, index: 8)
                enc.setBytes(&keyDim, length: MemoryLayout<UInt32>.size, index: 9)
                enc.setBytes(&vDim, length: MemoryLayout<UInt32>.size, index: 10)
                enc.dispatchThreadgroups(
                    MTLSize(width: Int(vHeads), height: Int(vDim) / 4, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
                // gated_norm: y(0) z(1) weight(2) out(3) vHeads(4) valueDim(5)
                enc.setComputePipelineState(gatedNormPSO)
                enc.setBuffer(deltaY, offset: 0, index: 0)
                enc.setBuffer(zY, offset: 0, index: 1)
                enc.setBuffer(gatedW, offset: 0, index: 2)
                enc.setBuffer(gatedOut, offset: 0, index: 3)
                enc.setBytes(&vHeads, length: MemoryLayout<UInt32>.size, index: 4)
                enc.setBytes(&vDim, length: MemoryLayout<UInt32>.size, index: 5)
                enc.dispatchThreadgroups(
                    MTLSize(width: Int(vHeads), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
        }
        enc.endEncoding()
        cb.commitEncoded()
        cb.waitUntilCompleted()
        if let err = cb.error {
            print("COMMAND BUFFER ERROR: \(err)")
        }

        let totalSeconds = cb.gpuEndTime - cb.gpuStartTime
        let perIteration = totalSeconds / Double(iterations)
        let gbPerSec = Double(bytes) / perIteration / 1_000_000_000
        let theoretical = 100.0
        print(
            "kernel=\(kernelName) iterations=\(iterations) "
                + "total=\(String(format: "%.4f", totalSeconds))s "
                + "per_launch=\(String(format: "%.2f", perIteration * 1_000_000))us")
        print(
            "bytes/launch=\(bytes) "
                + "achieved=\(String(format: "%.1f", gbPerSec)) GB/s "
                + "efficiency=\(String(format: "%.0f", gbPerSec / theoretical * 100))% of ~100 GB/s peak"
        )
    }

    /// The vocabulary head GEMV at the 35B family's shape (248,320 x 2048),
    /// the one GEMV every install runs at 8-bit. head_affine8 is the shipped
    /// generic affine kernel at bits=8 (measured 89.9 GB/s, at the ceiling;
    /// an 8-byte-per-lane specialization measured 91.7, noise, and was not
    /// kept), head_affine4 the same kernel at bits=4 (51.8 GB/s), head_int4
    /// the int4 head kernel (91.5). EMONE_BENCH_HEAD_SHAPE=qwen38 takes
    /// 248,320 x 2560.
    static func runHead(
        kernelName: String,
        iterations: Int,
        context: MetalContext
    ) throws {
        let device = context.device
        let qwen38 = ProcessInfo.processInfo.environment["EMONE_BENCH_HEAD_SHAPE"] == "qwen38"
        let rows: UInt32 = 248_320
        let n: UInt32 = qwen38 ? 2560 : 2048
        let groupCount = Int(n) / 64
        let bits: Int
        let kernel: String
        switch kernelName {
        case "head_affine4":
            bits = 4
            kernel = "affine_quant_gemv_simd"
        case "head_int4":
            bits = 4
            kernel = "dequant_int4_gemv_simd"
        default:
            bits = 8
            kernel = "affine_quant_gemv_simd"
        }
        let rowBytes = Int(n) * bits / 8
        func makeBuffer(_ bytes: Int, _ value: UInt8) throws -> MTLBuffer {
            guard let buf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw BenchHarnessError.metalObjectUnavailable("buffer of \(bytes) bytes")
            }
            memset(buf.contents(), Int32(value), bytes)
            return buf
        }
        let w = try makeBuffer(Int(rows) * rowBytes, 0x5A)
        let s = try makeBuffer(Int(rows) * groupCount * 2, 0x3C)
        let bb = try makeBuffer(Int(rows) * groupCount * 2, 0x00)
        let x = try makeBuffer(Int(n) * 2, 0)
        let xPtr = x.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<Int(n) { xPtr[i] = Float16(Float(i % 97 + 1) * 0.001).bitPattern }
        let y = try makeBuffer(Int(rows) * 2, 0)
        let constants =
            kernel.hasPrefix("affine")
            ? [MetalFunctionConstant(index: 100, value: .uint32(UInt32(bits)))] : []
        let pso = try context.pipeline(
            kernel, constants: constants,
            maxTotalThreadsPerThreadgroup: 256)
        var rowsVar = rows
        var nVar = n
        let threadgroups = (Int(rows) + 7) / 8
        let cb = try requireCommandBuffer(context.queue)
        let enc = try {
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw BenchHarnessError.metalObjectUnavailable("compute encoder")
            }
            return enc
        }()
        enc.setComputePipelineState(pso)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(s, offset: 0, index: 1)
        enc.setBuffer(bb, offset: 0, index: 2)
        enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(y, offset: 0, index: 4)
        enc.setBytes(&rowsVar, length: 4, index: 5)
        enc.setBytes(&nVar, length: 4, index: 6)
        for _ in 0..<iterations {
            enc.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        enc.endEncoding()
        cb.commitEncoded()
        cb.waitUntilCompleted()
        let total = cb.gpuEndTime - cb.gpuStartTime
        let per = total / Double(iterations)
        let bytes = UInt64(rows) * UInt64(rowBytes + groupCount * 4)
        let yPtr = y.contents().assumingMemoryBound(to: UInt16.self)
        let y0 = Float(Float16(bitPattern: yPtr[0]))
        let yLast = Float(Float16(bitPattern: yPtr[Int(rows) - 1]))
        print(
            "kernel=\(kernel) bits=\(bits) n=\(n) iterations=\(iterations) "
                + "per_launch=\(String(format: "%.1f", per * 1_000_000))us "
                + "achieved=\(String(format: "%.1f", Double(bytes) / per / 1e9)) GB/s "
                + "y0=\(y0) yLast=\(yLast)")
    }
}
