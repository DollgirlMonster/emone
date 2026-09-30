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

    @Test func aSystemPromptInsideTheFirstChunkIsCheckpointedAtItsEnd() {
        // Qwen3.8 4-bit prefills in 16K chunks and an agent system prompt is
        // about 9K tokens: no aligned boundary exists under it, so without the
        // anchor a new conversation could never resume past token zero.
        var tracker = FrontierTracker(chunkTokens: 16_384)
        let render = tokens(0..<9_300)
        tracker.observe(render)
        #expect(tracker.captureTargets(render: render, resumeFrom: 0, held: []).isEmpty)
        #expect(
            tracker.captureTargets(render: render, resumeFrom: 0, held: [], anchor: 9_211)
                == [9_211])
    }

    @Test func rungsPastTheAnchorCountChunksFromTheAnchor() {
        var tracker = FrontierTracker(chunkTokens: 4_096)
        let render = tokens(0..<20_000)
        tracker.observe(render)
        // The prefill resumes at the anchor, so 3000 + k*4096, not k*4096.
        #expect(
            tracker.captureTargets(render: render, resumeFrom: 0, held: [], anchor: 3_000)
                == [19_384, 11_192, 7_096, 3_000])
    }

    @Test func anAnchorIsSkippedWhenHeldOrAlreadyCoveredByAChunk() {
        var tracker = FrontierTracker(chunkTokens: 128)
        let render = tokens(0..<9_300)
        tracker.observe(render)
        let plain = tracker.captureTargets(render: render, resumeFrom: 0, held: [])
        // 9211 is 123 past the aligned 9088: not worth an extra pass.
        #expect(tracker.captureTargets(render: render, resumeFrom: 0, held: [], anchor: 9_211) == plain)

        var big = FrontierTracker(chunkTokens: 16_384)
        big.observe(render)
        #expect(
            big.captureTargets(render: render, resumeFrom: 0, held: [9_211], anchor: 9_211)
                .isEmpty)
        // Not past the resume point, or not strictly inside the render.
        #expect(big.captureTargets(render: render, resumeFrom: 9_211, held: [], anchor: 9_211).isEmpty)
        #expect(big.captureTargets(render: render, resumeFrom: 0, held: [], anchor: 9_300).isEmpty)
    }

    @Test func theGuidanceAnchorIsWhereTheSystemBlockStopsMatching() {
        let system = GFTokenizer.Message(role: .system, content: "rules")
        let user = GFTokenizer.Message(role: .user, content: "hi")
        let full: [Int32] = [1, 2, 3, 4, 10, 11, 12]
        // The partial render goes on with the stand-in query (7, 8), which
        // the full render does not share.
        let anchor = ServerModelSession.guidanceAnchor(promptIDs: full, messages: [system, user]) {
            // The system block plus a stand-in query: Qwen's template refuses
            // a chat with no user message.
            #expect($0.count == 2 && $0[0].role == .system && $0[1].role == .user)
            return [1, 2, 3, 4, 7, 8]
        }
        #expect(anchor == 4)
        // No leading system block, or nothing after it: no anchor.
        #expect(ServerModelSession.guidanceAnchor(promptIDs: full, messages: [user]) { _ in [1] } == nil)
        #expect(ServerModelSession.guidanceAnchor(promptIDs: full, messages: [system]) { _ in [1] } == nil)
        // A render that fails only means no checkpoint.
        struct Boom: Error {}
        #expect(
            ServerModelSession.guidanceAnchor(promptIDs: full, messages: [system, user]) { _ in
                throw Boom()
            } == nil)
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

    @Test func aProvenSharedPrefixStopsWhereTheSystemBlockVaries() {
        // Two conversation starts share 4,864 tokens, then a date inside the
        // system block differs; the system block itself ends at 5,400.
        var tracker = FrontierTracker(chunkTokens: 16_384)
        let first = tokens(0..<9_000)
        var second = tokens(0..<9_400)
        second[4_864] = -7
        #expect(tracker.provenSharedPrefix(first) == 0)
        tracker.rememberVerbatimRender(first)
        #expect(tracker.provenSharedPrefix(second) == 4_864)
        // The proven prefix wins over the (varying) system-block end...
        #expect(FrontierTracker.anchor(proven: 4_864, systemBlockEnd: 5_400) == 4_864)
        // ...and the first conversation, with nothing proven, falls back to it.
        #expect(FrontierTracker.anchor(proven: 0, systemBlockEnd: 5_400) == 5_400)
        #expect(FrontierTracker.anchor(proven: 900, systemBlockEnd: nil) == nil)
    }

    @Test func aProvenPrefixCanRunPastTheSystemBlockIntoInlinedFiles() {
        // Same system prompt and the same AGENTS.md inlined in the first user
        // message: the proven prefix covers both, past the system block.
        var tracker = FrontierTracker(chunkTokens: 16_384)
        let shared = tokens(0..<7_000)
        tracker.rememberVerbatimRender(shared + tokens(0..<500, salt: 100_000))
        let next = shared + tokens(0..<800, salt: 200_000)
        #expect(tracker.provenSharedPrefix(next) == 7_000)
        #expect(FrontierTracker.anchor(proven: 7_000, systemBlockEnd: 5_400) == 7_000)
    }

    @Test func sideCallsDoNotEvictTheConversationStartTheyFollow() {
        var tracker = FrontierTracker(chunkTokens: 16_384)
        let start = tokens(0..<6_000)
        tracker.rememberVerbatimRender(start)
        for side in 1...(FrontierTracker.recentVerbatimLimit - 1) {
            tracker.rememberVerbatimRender(tokens(0..<300, salt: Int32(side * 1_000_000)))
        }
        #expect(tracker.provenSharedPrefix(tokens(0..<6_500)) == 6_000)
        tracker.rememberVerbatimRender(tokens(0..<300, salt: 9_000_000))
        #expect(tracker.recentVerbatimRenders.count == FrontierTracker.recentVerbatimLimit)
        #expect(tracker.provenSharedPrefix(tokens(0..<6_500)) == 0)
    }
}
