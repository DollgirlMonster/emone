import Testing

@testable import TinyTitan

/// The frontier checkpoints split one chunked prefill into several adjacent
/// calls. That is only free if the split calls run the same spans, at the same
/// positions, as the single call would -- which is what these pin.
@Suite("Raw completion checkpoint capture")
struct RawCompletionCaptureTests {
    @Test func onlyWholeChunkBoundariesInsideTheUncachedRangeSurvive() {
        #expect(
            capturePositions(
                [12, 4, 8, 8, 3, 0, 16, 20], from: 0, promptCount: 16, chunkTokens: 4)
                == [4, 8, 12])
        // Resumed at 6: boundaries count chunks from there, not from zero.
        #expect(
            capturePositions([8, 10, 14, 18], from: 6, promptCount: 20, chunkTokens: 4)
                == [10, 14, 18])
        #expect(capturePositions([4], from: 0, promptCount: 16, chunkTokens: 0).isEmpty)
    }

    struct SplitCase: Sendable, CustomTestStringConvertible {
        let start: Int
        let promptCount: Int
        let chunk: Int
        var testDescription: String { "start \(start), \(promptCount) tokens, chunk \(chunk)" }
    }

    @Test(arguments: [
        SplitCase(start: 0, promptCount: 37, chunk: 4),
        SplitCase(start: 6, promptCount: 50, chunk: 4),
        SplitCase(start: 0, promptCount: 8_193, chunk: 4_096),
        SplitCase(start: 4_000, promptCount: 30_000, chunk: 4_096),
    ])
    func aSplitPrefillRunsExactlyTheSpansOfOneCall(_ split: SplitCase) {
        let start = split.start
        let promptCount = split.promptCount
        let chunk = split.chunk
        let single = PrefillChunkPlanner.spans(
            tokenCount: promptCount - start, startPosition: start, chunkTokens: chunk)
            .map { [$0.startPosition, $0.tokenCount] }

        let everyBoundary = Array(stride(from: start + chunk, to: promptCount, by: chunk))
        var spans: [[Int]] = []
        var position = start
        for boundary in capturePositions(
            everyBoundary, from: start, promptCount: promptCount, chunkTokens: chunk)
            + [promptCount]
        {
            spans += PrefillChunkPlanner.spans(
                tokenCount: boundary - position, startPosition: position, chunkTokens: chunk)
                .map { [$0.startPosition, $0.tokenCount] }
            position = boundary
        }
        #expect(spans == single)
    }
}
