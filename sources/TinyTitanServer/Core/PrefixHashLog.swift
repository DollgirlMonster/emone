import Foundation

/// Measure-only record of which prompt prefixes repeat across requests
/// (TINYTITAN_PREFIX_LOG=<path>, off by default).
///
/// The question it answers is whether pinning prefix snapshots would pay: how
/// often two requests share a long leading run and then diverge, where they
/// diverge, and how much of that shared run the prompt cache actually served.
/// It changes nothing about serving -- one JSON line per completed request,
/// appended to the file, read offline by `benchmark/prefix_log_report.py`.
///
/// Hashes only, never tokens or text: each entry is a chained hash of one
/// whole `blockTokens` block, so equal entries at index i mean the two prompts
/// agree on their first (i + 1) x `blockTokens` tokens. A trailing partial
/// block is not hashed; divergence is resolved to a block, which is enough to
/// place it against a 4K-16K prefill chunk.
enum PrefixHashLog {
    static let blockTokens = 256

    /// The log path, or nil when logging is off.
    static let path: String? = {
        guard let raw = ProcessInfo.processInfo.environment["TINYTITAN_PREFIX_LOG"],
            !raw.isEmpty
        else { return nil }
        return raw
    }()

    private static let writeLock = NSLock()

    /// Chained FNV-1a over whole blocks: entry i covers tokens [0, (i+1)*block).
    static func blockHashes(_ tokens: [Int32], blockTokens: Int = blockTokens) -> [UInt64] {
        precondition(blockTokens > 0, "blockTokens must be positive")
        let blocks = tokens.count / blockTokens
        var hashes: [UInt64] = []
        hashes.reserveCapacity(blocks)
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for block in 0..<blocks {
            for token in tokens[(block * blockTokens)..<((block + 1) * blockTokens)] {
                var bits = UInt32(bitPattern: token)
                for _ in 0..<4 {
                    hash ^= UInt64(bits & 0xff)
                    hash = hash &* 0x0000_0100_0000_01b3
                    bits >>= 8
                }
            }
            hashes.append(hash)
        }
        return hashes
    }

    /// One JSON line for a completed request.
    static func line(
        date: Date, promptTokens: Int, cachedTokens: Int, chunkTokens: Int, hashes: [UInt64]
    ) -> String {
        let hex = hashes.map { "\"" + String($0, radix: 16) + "\"" }.joined(separator: ",")
        return "{\"ts\":\(String(format: "%.3f", date.timeIntervalSince1970)),"
            + "\"prompt_tokens\":\(promptTokens),\"cached_tokens\":\(cachedTokens),"
            + "\"chunk_tokens\":\(chunkTokens),\"block_tokens\":\(blockTokens),"
            + "\"hashes\":[\(hex)]}\n"
    }

    /// Append a record when logging is on. A write failure is dropped: this is
    /// a measurement, and must never fail a request.
    static func record(promptIDs: [Int32], cachedTokens: Int, chunkTokens: Int) {
        guard let path else { return }
        let text = line(
            date: Date(), promptTokens: promptIDs.count, cachedTokens: cachedTokens,
            chunkTokens: chunkTokens, hashes: blockHashes(promptIDs))
        writeLock.withLock {
            let manager = FileManager.default
            if !manager.fileExists(atPath: path) {
                _ = manager.createFile(atPath: path, contents: nil)
            }
            guard let handle = FileHandle(forWritingAtPath: path) else { return }
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(text.utf8))
        }
    }
}
