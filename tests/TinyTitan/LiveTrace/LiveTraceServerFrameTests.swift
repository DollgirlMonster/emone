import Foundation
import Testing
import TinyTitan

@testable import TinyTitanLiveTrace

/// The frame a server draws: header with the model, client, queue and prefill
/// progress, and the idle frame between requests. Pure, like the CLI's.
@Suite struct LiveTraceServerFrameTests {
    static let shape = ExpertTraceShape(
        modelName: "ornith-1.5-35b-a3b", routedExpertBits: 8, layers: 40, numExperts: 256,
        topK: 8, layerKinds: (0..<40).map { $0 % 4 == 3 ? 1 : 2 }, expertBytes: 5_000_000)
    static let full = LiveTraceViewPlan.full(width: 78, height: 30)
    static let compact = LiveTraceViewPlan.compact(width: 78, height: 9)

    static func status(
        client: String? = nil, waiting: Int = 0, running: Int = 0, last: LiveTraceServerStatus.Last? = nil
    ) -> LiveTraceServerStatus {
        var status = LiveTraceServerStatus()
        status.residency = .loaded
        status.title = "ornith-1.5-35b-a3b · 8-bit"
        status.client = client
        status.waiting = waiting
        status.running = running
        status.last = last
        return status
    }

    static func input(
        status: LiveTraceServerStatus, feed: LiveTraceFeedSnapshot = LiveTraceFeedSnapshot(),
        plan: LiveTraceViewPlan = full, model: LiveTraceModel? = nil, log: [String] = []
    ) -> LiveTraceFrameInput {
        LiveTraceFrameInput(
            shape: shape, model: model, feed: feed, log: log, plan: plan, spinner: 3,
            depth: .none, server: status)
    }

    static func plain(_ lines: [String]) -> String { LiveTraceFrameTests.plain(lines) }

    static func header(_ input: LiveTraceFrameInput) -> String {
        let width = input.plan.width ?? 78
        let line = LiveTraceFrame.header(input, width: width)
        #expect(line.width <= width)
        return LiveTraceFrameTests.stripANSI(line.text)
    }

    // MARK: the header

    @Test func theHeaderNamesTheModelTheClientTheQueueAndPrefillProgress() {
        var feed = LiveTraceFeedSnapshot()
        feed.phase = .prefill
        feed.prefillDone = 2048
        feed.prefillTotal = 9000
        let text = Self.header(
            Self.input(
                status: Self.status(client: "gpt-5-codex", waiting: 2, running: 1), feed: feed))
        #expect(text.contains("ornith-1.5-35b-a3b · 8-bit"))
        #expect(text.contains("← gpt-5-codex"))
        #expect(text.contains("queue 2"))
        #expect(text.contains("prefill 2048/9000 tok"))
    }

    @Test func decodeShowsTokensAndRateWithoutAMaximum() {
        var feed = LiveTraceFeedSnapshot()
        feed.phase = .decode
        feed.tokens = 128
        feed.tokensPerSecond = 14.2
        let text = Self.header(
            Self.input(status: Self.status(client: "claude-sonnet", running: 1), feed: feed))
        #expect(text.contains("128 tok"))
        #expect(text.contains("14.2 tok/s"))
        #expect(!text.contains("/0"))
        #expect(!text.contains("queue"))
    }

    @Test func aLongClientIdGivesWayBeforeTheRightSideOrTheModel() {
        var feed = LiveTraceFeedSnapshot()
        feed.phase = .prefill
        feed.prefillDone = 12288
        feed.prefillTotal = 45000
        let client = String(repeating: "very-long-client-model-id-", count: 4)
        let text = Self.header(
            Self.input(status: Self.status(client: client, waiting: 12, running: 1), feed: feed))
        #expect(text.contains("prefill 12288/45000 tok"))
        #expect(text.contains("queue 12"))
        #expect(text.contains("ornith-1.5-35b-a3b"))
        #expect(text.contains("←"))
        #expect(LiveTraceText.width(of: text) == 78)
    }

    @Test func aModelBeingLoadedIsNamedInTheHeader() {
        var status = Self.status(client: "gpt-5", running: 1)
        status.residency = .loading
        status.title = "loading qwen3.5-9b_4-bit…"
        let text = Self.header(Self.input(status: status))
        #expect(text.contains("loading qwen3.5-9b_4-bit…"))
        #expect(text.contains("← gpt-5"))
    }

    @Test func theCliHeaderIsUntouchedByTheServerFields() throws {
        let model = try LiveTraceFrameTests.synthetic()
        let cli = LiveTraceFrameTests.input(model: model, plan: .full(width: 78, height: 30))
        #expect(cli.server == nil)
        let text = LiveTraceFrameTests.stripANSI(LiveTraceFrame.header(cli, width: 78).text)
        #expect(text.contains("12/64 tok"))
    }

    // MARK: between requests

    static func idleModel() throws -> LiveTraceModel {
        var model = try #require(LiveTraceModel(shape: shape, slotsPerLayer: 16))
        var trace = SyntheticTrace()
        for token in 0..<4 {
            for layer in 0..<40 {
                trace.add(
                    position: token, layer: layer, ids: [layer, layer + 1, 200, 201, 202, 3, 4, 5],
                    misses: token == 0 ? [0, 1] : [], atNanos: UInt64(token) * 70_000_000)
            }
        }
        model.apply(trace.batch)
        model.clearLive()
        return model
    }

    @Test func theIdleFrameShowsTheResidentModelTheLastRequestAndNoPicks() throws {
        let last = LiveTraceServerStatus.Last(
            tokensPerSecond: 14.2, newTokens: 812, promptTokens: 9200, cachedTokens: 9016)
        var feed = LiveTraceFeedSnapshot()
        feed.phase = .done
        feed.tokens = 812
        feed.textTail = "…and that is how the linked list is reversed in place."
        let lines = LiveTraceFrame.compose(
            Self.input(
                status: Self.status(last: last), feed: feed, model: try Self.idleModel(),
                log: ["[2026-10-02T13:04:05Z] request chatcmpl-1 completed in 41.2s"]))
        #expect(lines.count == 30)
        let text = Self.plain(lines)
        #expect(Self.plain([lines[0]]).contains("idle  last 14.2 tok/s  cache 98%"))
        #expect(Self.plain([lines[0]]).contains("ornith-1.5-35b-a3b · 8-bit"))
        #expect(!Self.plain([lines[0]]).contains("←"))
        #expect(text.contains("─ output · last reply"))
        #expect(text.contains("…and that is how the linked list is reversed in place."))
        // No picks, no layer in flight: the panel says so instead of showing the
        // last token's.
        #expect(text.contains("layer    idle"))
        #expect(!text.contains("MISS"))
        // The last request's expert hit rate stays on the cache row.
        #expect(text.contains("% hit"))
        #expect(!text.contains("-- hit"))
        // The heat is still on the grid (without colour, as block heights), so
        // the picture is the model's recent past rather than a blank.
        #expect(text.contains("█") || text.contains("▇") || text.contains("▆"))
    }

    @Test func idleWithNoModelAndNothingHappenedYetSaysWhatItIsWaitingFor() {
        var status = LiveTraceServerStatus()
        status.residency = .none
        let lines = LiveTraceFrame.compose(Self.input(status: status, plan: Self.compact))
        #expect(lines.count == 9)
        let text = Self.plain(lines)
        #expect(Self.plain([lines[0]]).contains("no model loaded"))
        #expect(Self.plain([lines[0]]).contains("idle"))
        #expect(text.contains("no model resident: the next request loads one"))
        #expect(text.contains("(nothing logged)"))
    }

    @Test func idleHeatCoolsWithoutAnyRequest() throws {
        var model = try Self.idleModel()
        let before = model.cells(layer: 3).map(\.heat)
        for _ in 0..<60 { model.idleStep() }
        let after = model.cells(layer: 3).map(\.heat)
        #expect(zip(before, after).allSatisfy { $0 >= $1 })
        #expect(zip(before, after).contains { $0 > $1 })
        #expect((after.max() ?? 0) < (before.max() ?? 0))
        // Nothing about the real cache changed: the residents are still believed in.
        #expect(model.cells(layer: 3).filter(\.cached).count == 6)
    }

    @Test func aWaitingRequestIsReportedWhileTheModelLoads() {
        var status = Self.status(waiting: 1)
        status.residency = .loading
        status.title = "loading qwen3.5-9b_4-bit…"
        let lines = LiveTraceFrame.compose(Self.input(status: status, plan: Self.compact))
        let text = Self.plain(lines)
        #expect(Self.plain([lines[0]]).contains("queue 1"))
        #expect(text.contains("loading the model for the next request"))
    }

    // MARK: the compact layout

    @Test func theCompactServerFrameIsNineLinesAtEverySize() {
        for width in [40, 60, 78] {
            let plan = LiveTraceViewPlan.compact(width: width, height: 9)
            var feed = LiveTraceFeedSnapshot()
            feed.phase = .decode
            feed.tokens = 7
            feed.textTail = "hello there"
            let lines = LiveTraceFrame.compose(
                Self.input(status: Self.status(client: "gpt-5", running: 1), feed: feed, plan: plan))
            #expect(lines.count == 9)
            #expect(lines.allSatisfy { LiveTraceText.width(of: LiveTraceFrameTests.stripANSI($0)) <= width })
        }
    }
}
