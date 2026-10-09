import Foundation
import Metal
import Testing

@testable import TinyTitan

/// `attention_prefill_qsa_masked_flash` runs a QSA selection as dense flash
/// tiles under the selection mask, skipping key tiles no row of a 32-row tile
/// kept. Same keys and softmax as the gathered kernels, another sum order, so
/// it is held to rounding against the per-head QSA kernel on Qwen3.8's shape
/// (24 query heads over 2 KV heads, head dim 256), with a partial last row
/// tile, row tiles offset into the chunk's mask, and every KV width.
@Suite struct PrefillAttentionQSAFlashTests {
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

    private static func run(
        kvBits: Int, queries: Int, startPosition: Int, rowTile: Int?, blockSparse: Bool,
        packed: Bool = false
    ) throws -> (reference: [Float16], flash: [Float16]) {
        let ctx = try MetalContext()
        let attention = try PrefillAttention(context: ctx)
        let (qHeads, kvHeads, headDim) = (24, 2, 256)
        let valid = startPosition + queries
        let groupSize = KVCacheManager.quantizationGroupSize
        var rng = LCG(state: 0xF1A5 &+ UInt64(kvBits))

        let kCache = try kvCache(
            ctx, bits: kvBits, rows: valid, elementsPerRow: kvHeads * headDim,
            groupSize: groupSize, rng: &rng)
        let vCache = try kvCache(
            ctx, bits: kvBits, rows: valid, elementsPerRow: kvHeads * headDim,
            groupSize: groupSize, rng: &rng)
        let q = try buffer(
            ctx, (0..<(queries * qHeads * headDim)).map { _ in Float16(rng.unit() * 0.5) })

        // Either QSA's shape -- whole 4-key blocks, a few dozen per row, plus
        // the row's own key -- or a scattered third of the visible keys.
        let keepStride = valid
        var mask = [UInt8](repeating: 0, count: queries * keepStride)
        for t in 0..<queries {
            let visible = startPosition + t + 1
            if blockSparse {
                for block in 0..<(visible / 4) where rng.next() % 8 == 0 {
                    for key in (block * 4)..<(block * 4 + 4) { mask[t * keepStride + key] = 1 }
                }
            } else {
                for key in 0..<visible where rng.next() % 3 == 0 { mask[t * keepStride + key] = 1 }
            }
            mask[t * keepStride + visible - 1] = 1
        }
        let maskBuffer = try buffer(ctx, mask)

        func params(start: Int, count: Int) -> PrefillAttentionParams {
            PrefillAttentionParams(
                startPosition: UInt32(start),
                queryCount: UInt32(count),
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
        }

        let half = MemoryLayout<Float16>.stride
        let rowElements = qHeads * headDim
        let outBytes = queries * rowElements * half
        func once(flash: Bool) throws -> [Float16] {
            attention.maskedFlash = flash
            attention.packedFlash = packed
            guard let out = ctx.device.makeBuffer(length: outBytes, options: .storageModeShared),
                let cb = ctx.queue.makeCommandBuffer()
            else { throw MetalError.commandEncoderFailed }
            memset(out.contents(), 0xAB, outBytes)
            let step = rowTile ?? queries
            var first = 0
            while first < queries {
                let count = min(step, queries - first)
                try attention.encodeCausal(
                    commandBuffer: cb,
                    q: q, qOffset: first * rowElements * half,
                    k: kCache.buffer, v: vCache.buffer,
                    out: out, outOffset: first * rowElements * half,
                    params: params(start: startPosition + first, count: count),
                    keepMask: maskBuffer, keepStride: keepStride,
                    keepRowOffset: first,
                    groupedQueryHeads: false, matrixUnits: false)
                first += count
            }
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
            let values = out.contents().bindMemory(to: Float16.self, capacity: outBytes / half)
            return Array(UnsafeBufferPointer(start: values, count: outBytes / half))
        }
        return (try once(flash: false), try once(flash: true))
    }

    /// Rounding, not bytes: the flash kernel holds K and V in half and sums
    /// in another order. Allowance: 1% of the reference or 3e-3, whichever is
    /// larger.
    private static func check(_ pair: (reference: [Float16], flash: [Float16]), _ label: String) {
        var worst: Float = 0
        var maxRef: Float = 0
        for (a, b) in zip(pair.reference, pair.flash) {
            let ref = Float(a)
            maxRef = max(maxRef, abs(ref))
            #expect(Float(b).isFinite, "\(label): non-finite output")
            let allowed = max(Float(3e-3), abs(ref) * 0.01)
            worst = max(worst, abs(Float(b) - ref) / allowed)
        }
        #expect(maxRef > 0.05, "\(label): reference output too small to prove anything")
        #expect(worst <= 1, "\(label): worst element \(worst)x its allowance, max |ref| \(maxRef)")
    }

    @Test func kernelsCompile() throws {
        let ctx = try MetalContext()
        _ = try ctx.pipeline("attention_prefill_qsa_masked_flash")
        _ = try ctx.pipeline("qsa_flash_tile_flags")
        _ = try ctx.pipeline("qsa_flash_pack_q")
    }

    @Test(arguments: [8, 4, 16])
    func tracksTheQSAKernel(_ kvBits: Int) throws {
        Self.check(
            try Self.run(
                kvBits: kvBits, queries: 77, startPosition: 300, rowTile: nil,
                blockSparse: true),
            "kv\(kvBits) block-sparse")
    }

    /// Causal mode, no selection: against `attention_prefill_causal_tiled`,
    /// Qwen 3.6's shape (16 query heads over 2 KV heads) and Qwen3.8's, a
    /// partial last row tile, and a chunk that starts partway into the cache.
    @Test(arguments: [(16, 8), (24, 8), (16, 16)])
    func denseCausalTracksTheTiledKernel(qHeads: Int, kvBits: Int) throws {
        let ctx = try MetalContext()
        let attention = try PrefillAttention(context: ctx)
        let (kvHeads, headDim) = (2, 256)
        let (startPosition, queries) = (300, 77)
        let valid = startPosition + queries
        let groupSize = KVCacheManager.quantizationGroupSize
        var rng = LCG(state: UInt64(qHeads * 100 + kvBits))
        let kCache = try Self.kvCache(
            ctx, bits: kvBits, rows: valid, elementsPerRow: kvHeads * headDim,
            groupSize: groupSize, rng: &rng)
        let vCache = try Self.kvCache(
            ctx, bits: kvBits, rows: valid, elementsPerRow: kvHeads * headDim,
            groupSize: groupSize, rng: &rng)
        let q = try Self.buffer(
            ctx, (0..<(queries * qHeads * headDim)).map { _ in Float16(rng.unit() * 0.5) })
        let params = PrefillAttentionParams(
            startPosition: UInt32(startPosition), queryCount: UInt32(queries),
            headDim: UInt32(headDim), numQHeads: UInt32(qHeads), numKVHeads: UInt32(kvHeads),
            kvValidCount: UInt32(valid), slidingWindow: UInt32(valid),
            kvTokenStrideElements: UInt32(kvHeads * headDim),
            qTokenStrideElements: UInt32(qHeads * headDim),
            oTokenStrideElements: UInt32(qHeads * headDim),
            scale: 0.0625, kvBits: UInt32(kvBits),
            kvTokenStrideBytes: UInt32(kCache.strideBytes),
            kvValueBytes: UInt32(kCache.valueBytes), kvGroupSize: UInt32(groupSize))
        let count = queries * qHeads * headDim
        func once(flash: Bool) throws -> [Float16] {
            attention.denseFlash = flash
            guard let out = ctx.device.makeBuffer(length: count * 2, options: .storageModeShared),
                let cb = ctx.queue.makeCommandBuffer()
            else { throw MetalError.commandEncoderFailed }
            memset(out.contents(), 0xAB, count * 2)
            try attention.encodeCausal(
                commandBuffer: cb, q: q, k: kCache.buffer, v: vCache.buffer, out: out, params: params)
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
            return Array(UnsafeBufferPointer(
                start: out.contents().bindMemory(to: Float16.self, capacity: count), count: count))
        }
        Self.check((try once(flash: false), try once(flash: true)), "dense q\(qHeads) kv\(kvBits)")
    }

    /// Packed mode walks each row tile's selected 4-key blocks instead of
    /// whole 16-key tiles: same selection, so it tracks the gathered kernel
    /// too, on block-shaped and scattered selections and offset row tiles.
    @Test(arguments: [8, 4, 16])
    func packedTracksTheQSAKernel(_ kvBits: Int) throws {
        Self.check(
            try Self.run(
                kvBits: kvBits, queries: 77, startPosition: 300, rowTile: nil,
                blockSparse: true, packed: true),
            "packed kv\(kvBits)")
    }

    @Test func packedScatteredAndRowTiles() throws {
        Self.check(
            try Self.run(
                kvBits: 8, queries: 45, startPosition: 130, rowTile: nil, blockSparse: false,
                packed: true),
            "packed scattered")
        Self.check(
            try Self.run(
                kvBits: 8, queries: 150, startPosition: 64, rowTile: 64, blockSparse: true,
                packed: true),
            "packed row tiles")
    }

    @Test func scatteredSelection() throws {
        Self.check(
            try Self.run(kvBits: 8, queries: 45, startPosition: 130, rowTile: nil, blockSparse: false),
            "scattered")
    }

    /// Row tiles start partway into the chunk's mask and flag buffer, as
    /// `encodeCausalTiled` dispatches them.
    @Test func rowTilesOffsetIntoTheChunk() throws {
        Self.check(
            try Self.run(kvBits: 8, queries: 150, startPosition: 64, rowTile: 64, blockSparse: true),
            "row tiles")
    }
    /// Packed mode at the Qwen3.8 shape of one 4,096-row tile at the end of a
    /// 16.9K prompt, int8 KV, with QSA's measured row correlation: each row
    /// keeps 512 4-key blocks drawn from its 8-row tile's pool of ~0.42 of the
    /// visible blocks (the union the real prompt showed). Prints GPU ms and
    /// the rate on the tiles walked; asserts nothing.
    /// TINYTITAN_QSA_FLASH_BENCH=1 to run.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_QSA_FLASH_BENCH"] == "1"))
    func packedBenchmark() throws {
        let ctx = try MetalContext()
        let attention = try PrefillAttention(context: ctx)
        let (qHeads, kvHeads, headDim) = (24, 2, 256)
        let (startPosition, queries) = (12_835, 4_096)
        let valid = startPosition + queries
        let groupSize = KVCacheManager.quantizationGroupSize
        var rng = LCG(state: 11)
        let kCache = try Self.kvCache(
            ctx, bits: 8, rows: valid, elementsPerRow: kvHeads * headDim,
            groupSize: groupSize, rng: &rng)
        let vCache = try Self.kvCache(
            ctx, bits: 8, rows: valid, elementsPerRow: kvHeads * headDim,
            groupSize: groupSize, rng: &rng)
        let q = try Self.buffer(
            ctx, (0..<(queries * qHeads * headDim)).map { _ in Float16(rng.unit() * 0.5) })
        var mask = [UInt8](repeating: 0, count: queries * valid)
        var walkedBlocks = 0
        for tile in 0..<(queries / 8) {
            let blocks = (startPosition + tile * 8 + 8) / 4
            var pool = Array(0..<blocks)
            for i in 0..<pool.count {
                let j = i + Int(rng.next() % UInt64(pool.count - i))
                pool.swapAt(i, j)
            }
            pool = Array(pool.prefix(max(512, blocks * 42 / 100)))
            var union = Set<Int>()
            for r in 0..<8 {
                let t = tile * 8 + r
                let visible = startPosition + t + 1
                var picked = 0
                for b in pool where picked < 512 && b * 4 + 4 <= visible && rng.next() % 2 == 0 {
                    for key in (b * 4)..<(b * 4 + 4) { mask[t * valid + key] = 1 }
                    union.insert(b)
                    picked += 1
                }
                mask[t * valid + visible - 1] = 1
                union.insert((visible - 1) / 4)
            }
            walkedBlocks += (union.count + 3) / 4 * 4
        }
        let maskBuffer = try Self.buffer(ctx, mask)
        let params = PrefillAttentionParams(
            startPosition: UInt32(startPosition), queryCount: UInt32(queries),
            headDim: UInt32(headDim), numQHeads: UInt32(qHeads), numKVHeads: UInt32(kvHeads),
            kvValidCount: UInt32(valid), slidingWindow: UInt32(valid),
            kvTokenStrideElements: UInt32(kvHeads * headDim),
            qTokenStrideElements: UInt32(qHeads * headDim),
            oTokenStrideElements: UInt32(qHeads * headDim),
            scale: 0.0625, kvBits: 8,
            kvTokenStrideBytes: UInt32(kCache.strideBytes),
            kvValueBytes: UInt32(kCache.valueBytes), kvGroupSize: UInt32(groupSize))
        guard let out = ctx.device.makeBuffer(
            length: queries * qHeads * headDim * 2, options: .storageModeShared)
        else { throw MetalError.commandEncoderFailed }
        // Two products of 8 rows x 4 keys x 256 per block per query head.
        let flop = Double(walkedBlocks) * 2 * 2 * 8 * 4 * Double(headDim) * Double(qHeads)
        attention.maskedFlash = true
        attention.packedFlash = true
        var best = Double.infinity
        for round in 0..<6 {
            guard let cb = ctx.queue.makeCommandBuffer() else { throw MetalError.commandEncoderFailed }
            try attention.encodeCausal(
                commandBuffer: cb, q: q, k: kCache.buffer, v: vCache.buffer, out: out,
                params: params, keepMask: maskBuffer, keepStride: valid,
                groupedQueryHeads: false, matrixUnits: false)
            cb.commit()
            cb.waitUntilCompleted()
            if round > 0 { best = min(best, cb.gpuEndTime - cb.gpuStartTime) }
        }
        print(String(format: "[qsa-packed-bench] best_gpu_ms=%.1f walked_tflops=%.2f",
            best * 1000, flop / best / 1e12))
    }

    /// GPU time at the Qwen3.8 shape of one 4,096-row attention tile at the
    /// end of a 16.9K prompt, int8 KV, every row keeping 2,048 keys in 4-key
    /// blocks (random, so near-dense per 8-row tile: the worst case). Prints;
    /// asserts nothing. TINYTITAN_QSA_FLASH_BENCH=1 to run.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_QSA_FLASH_BENCH"] == "1"))
    func benchmark() throws {
        let ctx = try MetalContext()
        let attention = try PrefillAttention(context: ctx)
        let (qHeads, kvHeads, headDim) = (24, 2, 256)
        let (startPosition, queries) = (12_835, 4_096)
        let valid = startPosition + queries
        let groupSize = KVCacheManager.quantizationGroupSize
        var rng = LCG(state: 7)
        let kCache = try Self.kvCache(
            ctx, bits: 8, rows: valid, elementsPerRow: kvHeads * headDim,
            groupSize: groupSize, rng: &rng)
        let vCache = try Self.kvCache(
            ctx, bits: 8, rows: valid, elementsPerRow: kvHeads * headDim,
            groupSize: groupSize, rng: &rng)
        let q = try Self.buffer(
            ctx, (0..<(queries * qHeads * headDim)).map { _ in Float16(rng.unit() * 0.5) })
        var mask = [UInt8](repeating: 0, count: queries * valid)
        var indices = [UInt32](repeating: 0, count: queries * 2_048)
        var counts = [UInt32](repeating: 0, count: queries)
        for t in 0..<queries {
            let visible = startPosition + t + 1
            let blocks = visible / 4
            var chosen = Set<Int>()
            while chosen.count < 512 { chosen.insert(Int(rng.next() % UInt64(blocks))) }
            var n = 0
            for block in chosen.sorted() {
                for key in (block * 4)..<(block * 4 + 4) {
                    mask[t * valid + key] = 1
                    indices[t * 2_048 + n] = UInt32(key)
                    n += 1
                }
            }
            counts[t] = UInt32(n)
        }
        let maskBuffer = try Self.buffer(ctx, mask)
        let indexBuffer = try Self.buffer(ctx, indices)
        let countBuffer = try Self.buffer(ctx, counts)
        let params = PrefillAttentionParams(
            startPosition: UInt32(startPosition), queryCount: UInt32(queries),
            headDim: UInt32(headDim), numQHeads: UInt32(qHeads), numKVHeads: UInt32(kvHeads),
            kvValidCount: UInt32(valid), slidingWindow: UInt32(valid),
            kvTokenStrideElements: UInt32(kvHeads * headDim),
            qTokenStrideElements: UInt32(qHeads * headDim),
            oTokenStrideElements: UInt32(qHeads * headDim),
            scale: 0.0625, kvBits: 8,
            kvTokenStrideBytes: UInt32(kCache.strideBytes),
            kvValueBytes: UInt32(kCache.valueBytes), kvGroupSize: UInt32(groupSize))
        guard let out = ctx.device.makeBuffer(
            length: queries * qHeads * headDim * 2, options: .storageModeShared)
        else { throw MetalError.commandEncoderFailed }
        for flash in [false, true, false, true] {
            attention.maskedFlash = flash
            guard let cb = ctx.queue.makeCommandBuffer() else { throw MetalError.commandEncoderFailed }
            try attention.encodeCausal(
                commandBuffer: cb, q: q, k: kCache.buffer, v: vCache.buffer, out: out,
                params: params, keepMask: maskBuffer, keepStride: valid,
                keepIndices: indexBuffer, keepIndexStride: 2_048, keepCounts: countBuffer)
            cb.commit()
            cb.waitUntilCompleted()
            print(String(format: "[qsa-flash-bench] flash=%d gpu_ms=%.1f", flash ? 1 : 0,
                (cb.gpuEndTime - cb.gpuStartTime) * 1000))
        }
    }
}
