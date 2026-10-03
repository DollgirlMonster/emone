import Foundation
import Metal
import Testing

@testable import TinyTitan

/// `qsa_select_prefill` against the host selection it replaces
/// (`QSAIndexer.selectPrefillRows`): the mask must be byte-identical, including
/// on rows inside the dense window, rows whose tail is ragged, and scores with
/// many ties (broken toward the lower block, as the host sort does).
@Suite struct QSAPrefillGPUSelectionTests {
    private struct LCG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 33
        }
    }

    private static func compare(
        startPosition: Int, tokens: Int, width: Int, ratio: Int, distinctScores: Int,
        seed: UInt64
    ) throws {
        let ctx = try MetalContext()
        let pso = try ctx.pipeline("qsa_select_prefill")
        let stride = startPosition + tokens
        let scoredBlocks = (stride + ratio - 1) / ratio
        var rng = LCG(state: seed)
        // Few distinct values, so ties are common; one in 16 is -0.0 vs 0.0.
        let scores: [Float] = (0..<(tokens * scoredBlocks)).map { _ in
            let v = Float(Int(rng.next() % UInt64(distinctScores)) - distinctScores / 2) * 0.25
            return v == 0 && rng.next() % 2 == 0 ? -0.0 : v
        }
        var hostKeep = [UInt8](repeating: 0xEE, count: tokens * stride)
        var indices = [UInt32](repeating: 0, count: tokens * width)
        var counts = [UInt32](repeating: 0, count: tokens)
        scores.withUnsafeBufferPointer { sp in
            hostKeep.withUnsafeMutableBufferPointer { kp in
                indices.withUnsafeMutableBufferPointer { ip in
                    counts.withUnsafeMutableBufferPointer { cp in
                        QSAIndexer.selectPrefillRows(
                            .init(
                                startPosition: startPosition, tokens: tokens, stride: stride,
                                indexWidth: width, selectionWidth: width,
                                compressRatio: ratio, scoredBlocks: scoredBlocks),
                            keep: kp.baseAddress!, indices: ip.baseAddress!,
                            counts: cp.baseAddress!, scores: sp.baseAddress!,
                            concurrent: false)
                    }
                }
            }
        }

        guard
            let scoreBuf = ctx.device.makeBuffer(
                bytes: scores, length: scores.count * 4, options: .storageModeShared),
            let keepBuf = ctx.device.makeBuffer(length: tokens * stride, options: .storageModeShared),
            let cb = ctx.queue.makeCommandBuffer(),
            let enc = cb.makeComputeCommandEncoder()
        else { throw MetalError.commandEncoderFailed }
        memset(keepBuf.contents(), 0xEE, tokens * stride)
        enc.setComputePipelineState(pso)
        enc.setBuffer(scoreBuf, offset: 0, index: 0)
        enc.setBuffer(keepBuf, offset: 0, index: 1)
        var params = [
            UInt32(startPosition), UInt32(ratio), UInt32(width), UInt32(scoredBlocks),
            UInt32(stride),
        ]
        for i in 0..<params.count { enc.setBytes(&params[i], length: 4, index: 2 + i) }
        enc.dispatchThreadgroups(
            MTLSize(width: tokens, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        let gpu = Array(
            UnsafeBufferPointer(
                start: keepBuf.contents().bindMemory(to: UInt8.self, capacity: tokens * stride),
                count: tokens * stride))
        var firstBad: Int?
        for i in 0..<gpu.count where gpu[i] != hostKeep[i] {
            firstBad = i
            break
        }
        #expect(
            firstBad == nil,
            "start \(startPosition) tokens \(tokens): first differing byte row \((firstBad ?? 0) / stride) key \((firstBad ?? 0) % stride)")
        #expect(hostKeep.contains(0) && hostKeep.contains(1), "the fixture selects nothing")
    }

    /// Rows on both sides of the dense window, ragged tails, many ties.
    @Test(arguments: [(0, 300), (150, 97), (1_000, 61)])
    func matchesTheHostSelection(start: Int, tokens: Int) throws {
        try Self.compare(
            startPosition: start, tokens: tokens, width: 131, ratio: 4,
            distinctScores: 9, seed: UInt64(start + tokens))
    }

    /// Qwen3.8's budget (2,048 + ratio - 1) at a long context, distinct-ish scores.
    @Test func qwen38Budget() throws {
        try Self.compare(
            startPosition: 6_000, tokens: 40, width: 2_051, ratio: 4,
            distinctScores: 4_001, seed: 38)
    }
}
