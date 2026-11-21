import Foundation
import Testing

/// Shared ChatML tokenizer fixture for the EmoneServer test target. The same
/// synthetic byte-level BPE fixture the Emone/Core tokenizer tests use,
/// bundled here so server tests exercise the real ChatML pipeline offline.
enum TokenizerFixture {
    static func folder() throws -> URL {
        try #require(
            Bundle.module.url(
                forResource: "ChatMLTokenizer",
                withExtension: nil,
                subdirectory: "Fixtures"))
    }
}
