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

    @Test func anAnchorSurvivesUnalignedAndLaterBoundariesCountFromIt() {
        // 4096 is aligned from 0 but not from the anchor at 3000, where the
        // next call starts; 7096 is.
        #expect(
            capturePositions(
                [3_000, 4_096, 7_096], from: 0, promptCount: 20_000, chunkTokens: 4_096,
                anchor: 3_000) == [3_000, 7_096])
        // Without the anchor, 3000 is dropped as before.
        #expect(
            capturePositions([3_000, 4_096], from: 0, promptCount: 20_000, chunkTokens: 4_096)
                == [4_096])
    }

    @Test func anAnchorAddsExactlyOneSplitToTheSpans() {
        let (chunk, anchor, count) = (4, 7, 30)
        var spans: [[Int]] = []
        var position = 0
        for boundary in capturePositions(
            [anchor, 15], from: 0, promptCount: count, chunkTokens: chunk, anchor: anchor)
            + [count]
        {
            spans += PrefillChunkPlanner.spans(
                tokenCount: boundary - position, startPosition: position, chunkTokens: chunk)
                .map { [$0.startPosition, $0.tokenCount] }
            position = boundary
        }
        // Whole chunks to the anchor, one partial chunk, whole chunks after.
        #expect(spans == [[0, 4], [4, 3], [7, 4], [11, 4], [15, 4], [19, 4], [23, 4], [27, 3]])
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
