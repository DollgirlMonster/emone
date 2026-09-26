import Testing

@testable import TinyTitan

@Suite struct PrefillRuntimeConfigTests {
    @Test(arguments: [32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384])
    func productionUsesCompleteChunkedPath(_ chunkTokens: Int) throws {
        let config = PrefillRuntimeConfig.production(chunkTokens: chunkTokens)
        #expect(config.mode == .chunked)
        #expect(config.chunkTokens == chunkTokens)
    }

    @Test func offDisablesChunkedPrefill() {
        let config = PrefillRuntimeConfig.off
        #expect(config.mode == .off)
        #expect(!config.enabled)
    }

    @Test func plannerUsesConfiguredChunkSize() {
        let spans = PrefillChunkPlanner.spans(
            tokenCount: 130,
            startPosition: 7,
            config: .production(chunkTokens: 64))
        #expect(spans.map(\.tokenCount) == [64, 64, 2])
        #expect(spans.map(\.startPosition) == [7, 71, 135])
    }

    @Test func plannerSupportsLongQwenChunks() {
        let spans = PrefillChunkPlanner.spans(
            tokenCount: 8_194,
            startPosition: 0,
            config: .production(chunkTokens: 4_096))
        #expect(spans.map(\.tokenCount) == [4_096, 4_096, 2])
        #expect(spans.map(\.startPosition) == [0, 4_096, 8_192])
    }

    /// The point of a bigger chunk: a 7.9K-token prompt is one sweep of the
    /// routed experts at 8K, not two.
    @Test func plannerTakesChunksAbove4096() {
        #expect(
            PrefillChunkPlanner.spans(
                tokenCount: 7_879, startPosition: 0, config: .production(chunkTokens: 8_192)
            ).map(\.tokenCount) == [7_879])
        let long = PrefillChunkPlanner.spans(
            tokenCount: 40_000, startPosition: 0, config: .production(chunkTokens: 16_384))
        #expect(long.map(\.tokenCount) == [16_384, 16_384, 7_232])
        // Nothing above the ceiling: a larger request is cut at it.
        #expect(
            PrefillChunkPlanner.spans(tokenCount: 40_000, startPosition: 0, chunkTokens: 65_536)
                .map(\.tokenCount) == [16_384, 16_384, 7_232])
    }

    @Test func diagnosticsPreserveUnknownValues() {
        let diagnostics = PrefillExecutionDiagnostics(
            config: .production(chunkTokens: 128),
            executedMode: .unsupported,
            kvStorageMode: nil,
            unsupportedReason: "unavailable")
        #expect(diagnostics.kvStorageMode == nil)
        #expect(diagnostics.chunkCompleteness == .unsupported)
        #expect(diagnostics.unsupportedReason == "unavailable")
    }
}
