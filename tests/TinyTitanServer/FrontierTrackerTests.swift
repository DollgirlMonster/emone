import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanServerCore

@Suite("Frontier checkpoint tracker")
struct FrontierTrackerTests {
    private func tokens(_ range: Range<Int>, salt: Int32 = 0) -> [Int32] {
        range.map { Int32($0) &+ salt }
    }

    @Test func firstRenderSeedsTheFrontierAndLaddersDownFromTheDeepestChunk() {
        var tracker = FrontierTracker(chunkTokens: 4)
        let render = tokens(0..<37)
        tracker.observe(render)
        #expect(tracker.frontier == render)
        // Deepest boundary strictly inside 37 is 36 (9 chunks); then 4, 2, 1
        // chunks: 16, 8, 4.
        #expect(tracker.captureTargets(render: render, resumeFrom: 0, held: []) == [36, 16, 8, 4])
    }

    @Test func aBoundaryEqualToTheRenderLengthIsNeverATarget() {
        var tracker = FrontierTracker(chunkTokens: 4)
        let render = tokens(0..<16)
        tracker.observe(render)
        // 16 would leave nothing for the final prefill call to seed from.
        #expect(tracker.captureTargets(render: render, resumeFrom: 0, held: []) == [12, 4])
    }

    @Test func aDivergentTailShrinksTheFrontierToTheDivergence() {
        var tracker = FrontierTracker(chunkTokens: 4)
        tracker.observe(tokens(0..<40))
        var next = tokens(0..<40)
        next[22] = -1  // the mutated system prompt
        tracker.observe(next)
        #expect(tracker.frontier == tokens(0..<22))
        // The deepest boundary under the divergence is the first rung.
        #expect(tracker.captureTargets(render: next, resumeFrom: 0, held: []) == [20, 8, 4])
    }

    @Test func aForeignRenderReseedsInsteadOfPinningTheFrontier() {
        var tracker = FrontierTracker(chunkTokens: 4)
        tracker.observe(tokens(0..<40))
        // Shares 2 tokens (< one chunk): a title request, another client.
        let foreign = [0, 1] + tokens(0..<10, salt: 1_000)
        tracker.observe(foreign)
        #expect(tracker.frontier == foreign)
        // Coming back must not be stuck at a sub-chunk frontier.
        let back = tokens(0..<40)
        tracker.observe(back)
        #expect(tracker.frontier == back)
        #expect(!tracker.captureTargets(render: back, resumeFrom: 0, held: []).isEmpty)
    }

    @Test func heldPositionsAreSkippedAndTheLadderStartsFromTheResumePoint() {
        var tracker = FrontierTracker(chunkTokens: 4)
        let render = tokens(0..<40)
        tracker.observe(render)
        // Resumed at 6 (a message-shaped hit): every target is 6 + k*4.
        #expect(tracker.captureTargets(render: render, resumeFrom: 6, held: []) == [38, 22, 14, 10])
        #expect(
            tracker.captureTargets(render: render, resumeFrom: 6, held: [38, 14])
                == [22, 10])
        // Nothing past the resume point to capture.
        #expect(tracker.captureTargets(render: render, resumeFrom: 38, held: []).isEmpty)
    }

    @Test func aZeroChunkCapturesNothing() {
        var tracker = FrontierTracker(chunkTokens: 0)
        tracker.observe(tokens(0..<40))
        #expect(tracker.captureTargets(render: tokens(0..<40), resumeFrom: 0, held: []).isEmpty)
    }

    @Test func frontierEntriesAreRecognizedByShapeAndRoundTrip() {
        let domain = ServerPromptCacheDomain(
            modelID: "model", sourceSnapshotHash: "snapshot",
            runtimeProfileHash: "profile", maximumContext: 16_384,
            kvStorage: "int8", fp16RingEnabled: true, templateSHA256: "template")
        let entry = ServerModelSession.makeFrontierEntry(domain: domain, tokens: [5, 6, 7])
        #expect(ServerModelSession.isFrontierEntry(entry))
        #expect(entry.kvPosition == 3)
        // A chat entry always carries a message, so it is never taken for one.
        let chat = ServerPromptCacheEntry(
            id: UUID(), domain: domain,
            inputMessages: [GFTokenizer.Message(role: .user, content: "hi")],
            tools: [],
            assistantTurn: entry.assistantTurn,
            kvBackedTokenIDs: [5, 6, 7], uncommittedBoundaryTokenIDs: [],
            kvPosition: 3)
        #expect(!ServerModelSession.isFrontierEntry(chat))
    }

    @Test func theEnvironmentSwitchTurnsTheFrontierCacheOff() {
        #expect(ServerModelSession.frontierCacheEnabled([:]))
        #expect(ServerModelSession.frontierCacheEnabled(["TINYTITAN_FRONTIER_CACHE": "on"]))
        for off in ["off", "0", "false", "NO"] {
            #expect(!ServerModelSession.frontierCacheEnabled(["TINYTITAN_FRONTIER_CACHE": off]))
        }
    }
}
