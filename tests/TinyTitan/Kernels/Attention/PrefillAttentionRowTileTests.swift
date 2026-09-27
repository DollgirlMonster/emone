import Foundation
import Metal
import Testing

@testable import TinyTitan

/// Prefill attention issued in row tiles, one command buffer each, so a deep
/// chunk cannot hold the GPU long enough for macOS to kill it
/// (kIOGPUCommandBufferCallbackErrorImpactingInteractivity). The claim is
/// byte equality with one dispatch over the whole chunk: every kernel indexes
/// by dispatch-local row and takes the query's position from
/// `startPosition + row`, so a tile is the same computation with offsets.
/// Checked on Qwen3.8's shape (24 query heads over 2 KV heads, head dim 256,
/// int8 KV) with no selection, a mask selection, and a compacted selection on
/// the matrix-unit kernel, at a tile size that does not divide the chunk.
@Suite struct PrefillAttentionRowTileTests {
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
            return ctx.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let made else { throw MetalError.commandEncoderFailed }
        return made
    }

    private enum Selection { case none, mask, compacted }

    private static func run(_ selection: Selection) throws -> (whole: [UInt8], tiled: [UInt8]) {
        let ctx = try MetalContext()
        let attention = try PrefillAttention(context: ctx)
        let (qHeads, kvHeads, headDim) = (24, 2, 256)
        let (startPosition, queries) = (300, 10)
        let valid = startPosition + queries
        let groupSize = KVCacheManager.quantizationGroupSize
        var rng = LCG(state: 77)

        let elements = kvHeads * headDim
        let valueBytes = elements
        let groups = (elements + groupSize - 1) / groupSize
        let stride = valueBytes + 4 * groups
        func cache() throws -> MTLBuffer {
            var bytes = [UInt8](repeating: 0, count: valid * stride)
            for row in 0..<valid {
                let base = row * stride
                for e in 0..<valueBytes { bytes[base + e] = UInt8(rng.next() & 0xFF) }
                for g in 0..<groups {
                    let scale = Float16(0.004 + abs(rng.unit()) * 0.006).bitPattern
                    let bias = Float16(-0.6 + rng.unit() * 0.05).bitPattern
                    bytes[base + valueBytes + g * 2] = UInt8(scale & 0xFF)
                    bytes[base + valueBytes + g * 2 + 1] = UInt8(scale >> 8)
                    bytes[base + valueBytes + groups * 2 + g * 2] = UInt8(bias & 0xFF)
                    bytes[base + valueBytes + groups * 2 + g * 2 + 1] = UInt8(bias >> 8)
                }
            }
            return try buffer(ctx, bytes)
        }
        let k = try cache()
        let v = try cache()
        let q = try buffer(
            ctx, (0..<(queries * qHeads * headDim)).map { _ in Float16(rng.unit() * 0.5) })

        var mask = [UInt8](repeating: 0, count: queries * valid)
        var indices = [UInt32](repeating: 0, count: queries * valid)
        var counts = [UInt32](repeating: 0, count: queries)
        for t in 0..<queries {
            var n = 0
            for key in 0...(startPosition + t)
            where rng.next() % 3 == 0 || key == startPosition + t {
                mask[t * valid + key] = 1
                indices[t * valid + n] = UInt32(key)
                n += 1
            }
            counts[t] = UInt32(n)
        }
        let maskBuffer = try buffer(ctx, mask)
        let indexBuffer = try buffer(ctx, indices)
        let countBuffer = try buffer(ctx, counts)

        let params = PrefillAttentionParams(
            startPosition: UInt32(startPosition), queryCount: UInt32(queries),
            headDim: UInt32(headDim), numQHeads: UInt32(qHeads), numKVHeads: UInt32(kvHeads),
            kvValidCount: UInt32(valid), slidingWindow: 0,
            kvTokenStrideElements: UInt32(elements),
            qTokenStrideElements: UInt32(qHeads * headDim),
            oTokenStrideElements: UInt32(qHeads * headDim),
            scale: 0.0625, kvBits: 8,
            kvTokenStrideBytes: UInt32(stride), kvValueBytes: UInt32(valueBytes),
            kvGroupSize: UInt32(groupSize))
        let keep = selection == .none ? nil : maskBuffer
        let idx = selection == .compacted ? indexBuffer : nil
        let cnt = selection == .compacted ? countBuffer : nil
        let outBytes = queries * qHeads * headDim * MemoryLayout<Float16>.stride

        func output(tileRows: Int?) throws -> [UInt8] {
            guard let out = ctx.device.makeBuffer(length: outBytes, options: .storageModeShared),
                let cb = ctx.queue.makeCommandBuffer()
            else { throw MetalError.commandEncoderFailed }
            memset(out.contents(), 0xAB, outBytes)
            let last = try attention.encodeCausalTiled(
                commandBuffer: cb, queue: ctx.queue, q: q, k: k, v: v, out: out,
                params: params, keepMask: keep, keepStride: valid,
                keepIndices: idx, keepIndexStride: valid, keepCounts: cnt,
                path: .causalTiled, tileRows: tileRows ?? queries)
            last.commit()
            last.waitUntilCompleted()
            return [UInt8](UnsafeRawBufferPointer(start: out.contents(), count: outBytes))
        }
        return (try output(tileRows: nil), try output(tileRows: 3))
    }

    @Test("Row tiles match one dispatch byte for byte, with no selection")
    func dense() throws {
        let r = try Self.run(.none)
        #expect(r.whole == r.tiled)
        #expect(r.whole.contains { $0 != 0xAB })
    }

    @Test("Row tiles match one dispatch byte for byte, with a mask selection")
    func mask() throws {
        let r = try Self.run(.mask)
        #expect(r.whole == r.tiled)
    }

    @Test("Row tiles match one dispatch byte for byte, with a compacted selection")
    func compacted() throws {
        let r = try Self.run(.compacted)
        #expect(r.whole == r.tiled)
    }

    @Test("Tiles only where the chunk is deep, sized to the depth")
    func tileSizing() {
        #expect(PrefillAttention.causalTileRows(queryCount: 16_384, visibleEnd: 16_384) == 4_096)
        #expect(PrefillAttention.causalTileRows(queryCount: 16_384, visibleEnd: 74_347) == 3_584)
        #expect(PrefillAttention.causalTileRows(queryCount: 16_384, visibleEnd: 262_144) == 1_024)
        #expect(PrefillAttention.causalTileRows(queryCount: 2_048, visibleEnd: 30_000) == nil)
    }
}
