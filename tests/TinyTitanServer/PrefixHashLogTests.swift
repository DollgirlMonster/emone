import Foundation
import Testing

@testable import TinyTitanServerCore

@Suite("Prefix hash log")
struct PrefixHashLogTests {
    private func tokens(_ range: Range<Int>, salt: Int32 = 0) -> [Int32] {
        range.map { Int32($0) &+ salt }
    }

    @Test func onlyWholeBlocksAreHashed() {
        #expect(PrefixHashLog.blockHashes(tokens(0..<3), blockTokens: 4).isEmpty)
        #expect(PrefixHashLog.blockHashes(tokens(0..<11), blockTokens: 4).count == 2)
        #expect(PrefixHashLog.blockHashes(tokens(0..<12), blockTokens: 4).count == 3)
    }

    @Test func aSharedPrefixSharesItsLeadingHashesAndNoMore() {
        let a = tokens(0..<16)
        var b = a
        b[9] = -1  // diverges inside the third block
        let ha = PrefixHashLog.blockHashes(a, blockTokens: 4)
        let hb = PrefixHashLog.blockHashes(b, blockTokens: 4)
        #expect(ha[0] == hb[0])
        #expect(ha[1] == hb[1])
        #expect(ha[2] != hb[2])
        // Chained: once diverged, every later block differs even where the
        // block's own tokens agree.
        #expect(ha[3] != hb[3])
    }

    @Test func aTokenInTheHighBytesChangesTheHash() {
        let a: [Int32] = [1, 2, 3, 4]
        let b: [Int32] = [1, 2, 3, 4 | (1 << 24)]
        #expect(
            PrefixHashLog.blockHashes(a, blockTokens: 4)
                != PrefixHashLog.blockHashes(b, blockTokens: 4))
    }

    @Test func theLineIsOneJSONObjectWithNoTokens() throws {
        let text = PrefixHashLog.line(
            date: Date(timeIntervalSince1970: 1), promptTokens: 600, cachedTokens: 512,
            chunkTokens: 16_384, hashes: [0xabc, 0x1])
        #expect(text.hasSuffix("\n"))
        let object = try #require(
            JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        #expect(object["prompt_tokens"] as? Int == 600)
        #expect(object["cached_tokens"] as? Int == 512)
        #expect(object["chunk_tokens"] as? Int == 16_384)
        #expect(object["block_tokens"] as? Int == PrefixHashLog.blockTokens)
        #expect(object["hashes"] as? [String] == ["abc", "1"])
    }
}
