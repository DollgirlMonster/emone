import Foundation
import Metal
import Testing

@testable import TinyTitan

/// `attention_prefill_causal_qsa_gqa` serves all of a KV head's query heads in
/// one threadgroup so each selected K/V row is read once, not once per query
/// head. It keeps the per-head kernel's arithmetic -- the same dot helper, the
/// same running max, the same weights and in-order sums -- so the claim tested
/// here is byte equality with `attention_prefill_causal_qsa_tiled`, on Qwen3.8's
/// shape (24 query heads over 2 KV heads, head dim 256; int8 KV, then int4 and
/// fp16) with a sparse per-token selection long enough to span several tiles.
@Suite struct PrefillAttentionQSAGroupedTests {
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

    /// One KV cache in the runtime's row layout. Quantized (8 or 4 bits): the
    /// values of every KV head, packed two to a byte at 4 bits, then a half
    /// scale and a half bias per affine group. Unquantized (16): plain halves.
    private static func kvCache(
        _ ctx: MetalContext, bits: Int, rows: Int, elementsPerRow: Int, groupSize: Int,
        rng: inout LCG
    ) throws -> (buffer: MTLBuffer, strideBytes: Int, valueBytes: Int) {
        if bits == 16 {
            let values = (0..<(rows * elementsPerRow)).map { _ in Float16(rng.unit() * 0.8) }
            let stride = elementsPerRow * MemoryLayout<Float16>.stride
            return (try buffer(ctx, values), stride, stride)
        }
        let valueBytes = elementsPerRow * bits / 8
        let groups = (elementsPerRow + groupSize - 1) / groupSize
        let stride = valueBytes + 2 * groups * MemoryLayout<Float16>.stride
        // Scale x the mid code is about 0.6 at either width, so the bias
        // centres the values near zero.
        let (scaleFloor, scaleSpread): (Float, Float) = bits == 8 ? (0.004, 0.006) : (0.06, 0.04)
        var bytes = [UInt8](repeating: 0, count: rows * stride)
        for row in 0..<rows {
            let base = row * stride
            for e in 0..<valueBytes { bytes[base + e] = UInt8(rng.next() & 0xFF) }
            for g in 0..<groups {
                // Little-endian fp16, as the kernel reads them.
                let scale = Float16(scaleFloor + abs(rng.unit()) * scaleSpread).bitPattern
                let bias = Float16(-0.6 + rng.unit() * 0.05).bitPattern
                let scaleAt = base + valueBytes + g * 2
                let biasAt = base + valueBytes + groups * 2 + g * 2
                bytes[scaleAt] = UInt8(scale & 0xFF)
                bytes[scaleAt + 1] = UInt8(scale >> 8)
                bytes[biasAt] = UInt8(bias & 0xFF)
                bytes[biasAt + 1] = UInt8(bias >> 8)
            }
        }
        return (try buffer(ctx, bytes), stride, valueBytes)
    }

    private struct Result {
        let bytes: [UInt8]
        let values: [Float16]
    }

    /// `matrixUnits` runs the grouped side on the simdgroup-matrix kernel.
    private static func run(
        compacted: Bool, kvBits: Int = 8, matrixUnits: Bool = false
    ) throws -> (perHead: Result, grouped: Result) {
        let ctx = try MetalContext()
        let attention = try PrefillAttention(context: ctx)
        let (qHeads, kvHeads, headDim) = (24, 2, 256)
        let (startPosition, queries) = (300, 6)
        let valid = startPosition + queries
        let groupSize = KVCacheManager.quantizationGroupSize
        var rng = LCG(state: 38)

        let kCache = try kvCache(
            ctx, bits: kvBits, rows: valid, elementsPerRow: kvHeads * headDim,
            groupSize: groupSize, rng: &rng)
        let vCache = try kvCache(
            ctx, bits: kvBits, rows: valid, elementsPerRow: kvHeads * headDim,
            groupSize: groupSize, rng: &rng)
        let q = try buffer(ctx, (0..<(queries * qHeads * headDim)).map { _ in Float16(rng.unit() * 0.5) })

        // A sparse selection per query -- about a third of its visible keys,
        // ~100 of them, so each row spans two tiles of the grouped kernel --
        // as both a mask and a compacted index list.
        let keepStride = valid
        var mask = [UInt8](repeating: 0, count: queries * keepStride)
        var indices = [UInt32](repeating: 0, count: queries * valid)
        var counts = [UInt32](repeating: 0, count: queries)
        for t in 0..<queries {
            var n = 0
            for key in 0...(startPosition + t) where rng.next() % 3 == 0 || key == startPosition + t {
                mask[t * keepStride + key] = 1
                indices[t * valid + n] = UInt32(key)
                n += 1
            }
            counts[t] = UInt32(n)
        }
        let maskBuffer = try buffer(ctx, mask)
        let indexBuffer = try buffer(ctx, indices)
        let countBuffer = try buffer(ctx, counts)

        let params = PrefillAttentionParams(
            startPosition: UInt32(startPosition),
            queryCount: UInt32(queries),
            headDim: UInt32(headDim),
            numQHeads: UInt32(qHeads),
            numKVHeads: UInt32(kvHeads),
            kvValidCount: UInt32(valid),
            slidingWindow: 0,
            kvTokenStrideElements: UInt32(kvHeads * headDim),
            qTokenStrideElements: UInt32(qHeads * headDim),
            oTokenStrideElements: UInt32(qHeads * headDim),
            scale: 0.0625,
            kvBits: UInt32(kvBits),
            kvTokenStrideBytes: UInt32(kCache.strideBytes),
            kvValueBytes: UInt32(kCache.valueBytes),
            kvGroupSize: UInt32(groupSize))

        let outBytes = queries * qHeads * headDim * MemoryLayout<Float16>.stride
        func once(grouped: Bool, mma: Bool) throws -> Result {
            guard let out = ctx.device.makeBuffer(length: outBytes, options: .storageModeShared),
                let cb = ctx.queue.makeCommandBuffer()
            else { throw MetalError.commandEncoderFailed }
            memset(out.contents(), 0xAB, outBytes)
            try attention.encodeCausal(
                commandBuffer: cb, q: q, k: kCache.buffer, v: vCache.buffer, out: out,
                params: params,
                keepMask: maskBuffer, keepStride: keepStride,
                keepIndices: compacted ? indexBuffer : nil,
                keepIndexStride: valid,
                keepCounts: compacted ? countBuffer : nil,
                groupedQueryHeads: grouped,
                matrixUnits: mma)
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
            let raw = UnsafeRawBufferPointer(start: out.contents(), count: outBytes)
            let values = out.contents().bindMemory(to: Float16.self, capacity: outBytes / 2)
            return Result(
                bytes: Array(raw),
                values: Array(UnsafeBufferPointer(start: values, count: outBytes / 2)))
        }
        return (try once(grouped: false, mma: false), try once(grouped: true, mma: matrixUnits))
    }

    private static func check(_ pair: (perHead: Result, grouped: Result)) {
        var maxDiff: Float = 0
        var maxRef: Float = 0
        for (a, b) in zip(pair.perHead.values, pair.grouped.values) {
            maxRef = max(maxRef, abs(Float(a)))
            maxDiff = max(maxDiff, abs(Float(a) - Float(b)))
        }
        #expect(maxRef > 0, "the per-head output is all zeros; the fixture proves nothing")
        #expect(
            pair.perHead.bytes == pair.grouped.bytes,
            "grouped differs from per-head: max |diff| \(maxDiff), max |ref| \(maxRef)")
    }

    /// The matrix-unit kernel sums in another order and rescales per 16-key
    /// tile, so it is held to rounding, not bytes: four half ulps at the
    /// reference's magnitude, with a floor near zero.
    private static func checkClose(_ pair: (perHead: Result, grouped: Result)) {
        var worst: Float = 0
        var maxRef: Float = 0
        var changed = 0
        for (a, b) in zip(pair.perHead.values, pair.grouped.values) {
            let ref = Float(a)
            maxRef = max(maxRef, abs(ref))
            if a != b { changed += 1 }
            let allowed = max(Float(2e-3), 4 * Float(Float16(abs(ref)).ulp))
            worst = max(worst, abs(Float(b) - ref) / allowed)
        }
        #expect(maxRef > 0.05, "the per-head output is too small to prove anything")
        #expect(
            worst <= 1,
            "matrix units vs per-head: worst element \(worst)x its allowance, \(changed) differ, max |ref| \(maxRef)")
    }

    /// Compiled strictly here, so a Metal compiler error names itself instead
    /// of the runtime quietly falling back to the scalar kernel.
    @Test func matrixUnitKernelCompiles() throws {
        let ctx = try MetalContext()
        _ = try ctx.pipeline("attention_prefill_causal_qsa_gqa_mma")
        #expect(try PrefillAttention(context: ctx).hasQSAMatrixUnitKernel)
    }

    @Test(arguments: [8, 4, 16])
    func matrixUnitKernelTracksThePerHeadKernel(_ kvBits: Int) throws {
        Self.checkClose(try Self.run(compacted: true, kvBits: kvBits, matrixUnits: true))
    }

    @Test func matrixUnitKernelTakesTheMaskSelection() throws {
        Self.checkClose(try Self.run(compacted: false, matrixUnits: true))
    }

    @Test func compactedSelectionMatchesThePerHeadKernelExactly() throws {
        Self.check(try Self.run(compacted: true))
    }

    @Test func maskSelectionMatchesThePerHeadKernelExactly() throws {
        Self.check(try Self.run(compacted: false))
    }

    /// The grouped kernel is the default for every KV width, and the int4 and
    /// fp16 caches take their own branches of the shared dot and load helpers.
    @Test(arguments: [4, 16])
    func otherKVWidthsMatchThePerHeadKernelExactly(_ kvBits: Int) throws {
        Self.check(try Self.run(compacted: true, kvBits: kvBits))
    }
}
