import Foundation
import Testing

@testable import TinyTitan

/// The prefill key selection runs its rows in parallel. Each row reads only its
/// own scores and writes only its own mask row, index row and count, so the
/// result has to be byte-identical to the rows done in order -- and each row
/// still has to follow the rule: its ragged tail, then whole blocks by score
/// (higher first, lower block index on a tie) until the cell budget is spent.
@Suite struct QSAPrefillSelectionTests {
    private struct LCG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 33
        }
    }

    /// Budget 16 at a compress ratio of 4: 19 cells a row. Rows start below the
    /// window (keep everything) and run well past it.
    private static let geometry = QSAIndexer.PrefillSelectionGeometry(
        startPosition: 6, tokens: 90, stride: 96, indexWidth: 19,
        selectionWidth: 19, compressRatio: 4, scoredBlocks: 24)

    private struct Output: Equatable {
        var keep: [UInt8]
        var indices: [UInt32]
        var counts: [UInt32]
    }

    private static func select(scores: [Float], concurrent: Bool) throws -> Output {
        let g = geometry
        // The same stale bytes in both runs, so anything a row fails to write
        // compares equal only if both paths leave it alone.
        var out = Output(
            keep: [UInt8](repeating: 0xEE, count: g.tokens * g.stride),
            indices: [UInt32](repeating: 0xDEAD, count: g.tokens * g.indexWidth),
            counts: [UInt32](repeating: 0xBEEF, count: g.tokens))
        try out.keep.withUnsafeMutableBufferPointer { keep in
            try out.indices.withUnsafeMutableBufferPointer { indices in
                try out.counts.withUnsafeMutableBufferPointer { counts in
                    try scores.withUnsafeBufferPointer { scores in
                        QSAIndexer.selectPrefillRows(
                            g,
                            keep: try #require(keep.baseAddress),
                            indices: try #require(indices.baseAddress),
                            counts: try #require(counts.baseAddress),
                            scores: try #require(scores.baseAddress),
                            concurrent: concurrent)
                    }
                }
            }
        }
        return out
    }

    /// Few distinct values, so ties are common and the tie-break is exercised.
    private static func scores(seed: UInt64) -> [Float] {
        var rng = LCG(state: seed)
        let g = geometry
        return (0..<(g.tokens * g.scoredBlocks)).map { _ in Float(rng.next() % 7) }
    }

    @Test func parallelRowsMatchRowOrderExactly() throws {
        for seed: UInt64 in [1, 2, 3] {
            let s = Self.scores(seed: seed)
            let parallel = try Self.select(scores: s, concurrent: true)
            let inOrder = try Self.select(scores: s, concurrent: false)
            #expect(parallel == inOrder)
        }
    }

    @Test func aRowPastTheWindowKeepsItsTailThenTheBestBlocks() throws {
        let g = Self.geometry
        var s = [Float](repeating: 0, count: g.tokens * g.scoredBlocks)
        // Row 40 sees 47 keys: 11 whole blocks (44 cells) and a 3-cell tail,
        // so 16 cells remain for 4 blocks. Blocks 9, 2, 7 score highest; 4 and
        // 5 tie for fourth, and the lower index wins.
        let row = 40
        for (block, value) in [(9, 5.0), (2, 4.0), (7, 3.0), (4, 2.0), (5, 2.0)] {
            s[row * g.scoredBlocks + block] = Float(value)
        }
        let out = try Self.select(scores: s, concurrent: true)
        let kept = (0..<Int(out.counts[row])).map { Int(out.indices[row * g.indexWidth + $0]) }
        let expected = [2, 4, 7, 9].flatMap { b in (b * 4)..<(b * 4 + 4) } + [44, 45, 46]
        #expect(kept == expected)
        #expect(out.counts[row] == 19)
    }

    @Test func rowsInsideTheWindowKeepEverythingTheySee() throws {
        let g = Self.geometry
        let out = try Self.select(scores: Self.scores(seed: 9), concurrent: true)
        for row in 0..<(g.selectionWidth - g.startPosition) {
            let visible = g.startPosition + row + 1
            #expect(out.counts[row] == UInt32(visible))
            #expect(out.keep[(row * g.stride)..<(row * g.stride + visible)].allSatisfy { $0 == 1 })
            #expect(out.keep[(row * g.stride + visible)..<((row + 1) * g.stride)].allSatisfy { $0 == 0 })
        }
    }
}
