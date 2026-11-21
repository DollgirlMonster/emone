import Foundation
import Synchronization
import Testing
import TinyTitan

@testable import TinyTitanLiveTrace

/// A runner that records nothing and can be asked for a ring, so the view's
/// following of a catalog's model switches is exercised without a model.
final class FakeTraceHost: ExpertTraceHosting {
    let shape: ExpertTraceShape
    private(set) var enableCalls = 0
    private(set) var ring: ExpertTraceRing?

    init(shape: ExpertTraceShape) { self.shape = shape }

    func expertTraceShape() -> ExpertTraceShape { shape }

    @discardableResult
    func enableExpertTrace(capacity: Int) -> ExpertTraceRing? {
        enableCalls += 1
        guard shape.isRouted else { return nil }
        let made = ExpertTraceRing(shape: shape, capacity: capacity)
        ring = made
        return made
    }

    func disableExpertTrace() { ring = nil }
}

/// Collects the notes a view says, from whatever thread says them.
final class NoteCollector: Sendable {
    private let lines = Mutex<[String]>([])
    func add(_ line: String) { lines.withLock { $0.append(line) } }
    var all: [String] { lines.withLock { $0 } }
}

/// Everything one view test needs: a pseudo-terminal standing for the server's
/// terminal, a pipe standing for a stderr the operator redirected, and a log
/// file, with the descriptors the view captures being private duplicates so the
/// test process's own stdout and stderr are never touched.
final class ViewHarness {
    let pty: PseudoTerminal
    let stdoutTarget: Int32
    let stderrTarget: Int32
    let stderrRead: Int32
    let logPath: String
    let notes = NoteCollector()
    var view: LiveTraceServerView?

    static let tty = TerminalProbe(
        stdoutIsTTY: true, term: "xterm-256color", colorTerm: nil, noColor: false, cols: 100,
        rows: 40)

    init?(probe: TerminalProbe = ViewHarness.tty, nameOverride: String? = nil) {
        guard let pty = PseudoTerminal(cols: UInt16(probe.cols), rows: UInt16(probe.rows)) else {
            return nil
        }
        self.pty = pty
        stdoutTarget = dup(pty.slave)
        var fds: [Int32] = [0, 0]
        guard pipe(&fds) == 0 else { return nil }
        stderrRead = fds[0]
        stderrTarget = fds[1]
        logPath = NSTemporaryDirectory() + "live-trace-\(UUID().uuidString)/server.log"
        let collector = notes
        view = LiveTraceServerView.prepare(
            options: .init(logPath: logPath, nameOverride: nameOverride), probe: probe,
            stdoutFD: stdoutTarget, stderrFD: stderrTarget,
            note: { line in collector.add(line) })
    }

    deinit {
        view?.close()
        close(stderrRead)
        close(stderrTarget)
        close(stdoutTarget)
        pty.close()
        try? FileManager.default.removeItem(
            atPath: (logPath as NSString).deletingLastPathComponent)
    }

    var noteLines: [String] { notes.all }

    /// What the view has written to its terminal so far.
    func terminalOutput(wait: TimeInterval = 0.2) -> String {
        _ = fcntl(pty.master, F_SETFL, O_NONBLOCK)
        var out = [UInt8]()
        var buffer = [UInt8](repeating: 0, count: 65536)
        let deadline = Date().addingTimeInterval(wait)
        while Date() < deadline {
            let count = read(pty.master, &buffer, buffer.count)
            if count > 0 {
                out.append(contentsOf: buffer[0..<count])
            } else {
                Thread.sleep(forTimeInterval: 0.01)
            }
        }
        return String(decoding: out, as: UTF8.self)
    }

    func logFileText() -> String { (try? String(contentsOfFile: logPath, encoding: .utf8)) ?? "" }

    /// Compose frames until `needle` shows (the capture threads are asynchronous).
    func waitForFrame(containing needle: String, timeout: TimeInterval = 3) -> [String] {
        let deadline = Date().addingTimeInterval(timeout)
        var lines: [String] = []
        while Date() < deadline {
            lines = view?.composeFrameForTesting() ?? []
            if lines.contains(where: { $0.contains(needle) }) { return lines }
            Thread.sleep(forTimeInterval: 0.02)
        }
        return lines
    }
}

@Suite(.serialized) struct LiveTraceServerViewTests {
    static let routed = ExpertTraceShape(
        modelName: "ornith-1.5-35b-a3b", routedExpertBits: 8, layers: 40, numExperts: 256,
        topK: 8, layerKinds: (0..<40).map { $0 % 4 == 3 ? 1 : 2 }, expertBytes: 5_000_000)
    static let routed512 = ExpertTraceShape(
        modelName: "qwen3.8-flash-next", routedExpertBits: 4, layers: 48, numExperts: 512,
        topK: 10, layerKinds: (0..<48).map { $0 % 4 == 3 ? 1 : 2 }, expertBytes: 7_000_000)
    static let dense = ExpertTraceShape(
        modelName: "qwen3.5-4b", routedExpertBits: nil, layers: 32, numExperts: 0, topK: 0,
        layerKinds: [], expertBytes: 0)

    static func info(
        _ name: String, shape: ExpertTraceShape?, engine: String = "gpu",
        unavailable: String? = nil
    ) -> LiveTraceModelInfo {
        LiveTraceModelInfo(
            name: name, engine: engine, shape: shape, slotsPerLayer: 16,
            gridUnavailable: unavailable)
    }

    static func plain(_ lines: [String]) -> String { LiveTraceFrameTests.plain(lines) }

    static func header(_ lines: [String]) -> String {
        LiveTraceFrameTests.stripANSI(lines.first ?? "")
    }

    // MARK: gating

    @Test func offByDefaultAndDrawsNothingWithoutAnOpen() throws {
        let harness = try #require(ViewHarness())
        #expect(harness.view != nil)
        #expect(harness.terminalOutput(wait: 0.05).isEmpty)
        #expect(harness.noteLines.isEmpty)
    }

    @Test func aNonTerminalGetsOneLineAndNoView() throws {
        var probe = ViewHarness.tty
        probe.stdoutIsTTY = false
        let harness = try #require(ViewHarness(probe: probe))
        #expect(harness.view == nil)
        #expect(harness.noteLines == ["note: --live-trace is off: stdout is not a terminal"])
    }

    @Test func aDumbTerminalAndATinyOneAreRefusedWithTheirReasons() throws {
        var dumb = ViewHarness.tty
        dumb.term = "dumb"
        let first = try #require(ViewHarness(probe: dumb))
        #expect(first.view == nil)
        #expect(first.noteLines.first?.contains("TERM") == true)

        var narrow = ViewHarness.tty
        narrow.cols = 30
        let second = try #require(ViewHarness(probe: narrow))
        #expect(second.view == nil)
        #expect(second.noteLines.first?.contains("30 columns") == true)

        var short = ViewHarness.tty
        short.rows = 6
        let third = try #require(ViewHarness(probe: short))
        #expect(third.view == nil)
        #expect(third.noteLines.first?.contains("6 rows") == true)
    }

    @Test func theSmallestTerminalThatCanShowAnythingIsFortyByTen() throws {
        var probe = ViewHarness.tty
        probe.cols = 40
        probe.rows = 10
        let harness = try #require(ViewHarness(probe: probe))
        #expect(harness.view != nil)
        #expect(harness.noteLines.isEmpty)
    }

    // MARK: the layout follows the model

    @Test func aRoutedModelGetsTheGridWhereTheTerminalHasRoomForIt() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        view.modelLoaded(token: UUID(), info: Self.info("ornith", shape: Self.routed))
        #expect(view.currentPlanForTesting == .full(width: 78, height: 30))
        view.modelLoaded(token: UUID(), info: Self.info("qwen3.8", shape: Self.routed512))
        #expect(view.currentPlanForTesting == .full(width: 78, height: 30))
        #expect(harness.noteLines.isEmpty)
    }

    @Test func aSmallTerminalGetsTheCompactViewAndTheSameNoteAsTheCli() throws {
        var probe = ViewHarness.tty
        probe.cols = 60
        probe.rows = 20
        let harness = try #require(ViewHarness(probe: probe))
        let view = try #require(harness.view)
        view.modelLoaded(token: UUID(), info: Self.info("ornith", shape: Self.routed))
        #expect(view.currentPlanForTesting == .compact(width: 60, height: 9))
        #expect(
            harness.noteLines == [
                "note: --live-trace grid hidden: the terminal is 60x20; the grid needs at least 78x31"
            ])
    }

    @Test func aDenseOrCpuModelGetsTheCompactViewWithoutANote() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        view.modelLoaded(token: UUID(), info: Self.info("qwen3.5-4b", shape: Self.dense))
        #expect(view.currentPlanForTesting == .compact(width: 78, height: 9))
        view.modelLoaded(
            token: UUID(), info: Self.info("qwen3.5-2b", shape: nil, engine: "cpu"))
        #expect(view.currentPlanForTesting == .compact(width: 78, height: 9))
        #expect(harness.noteLines.isEmpty)
    }

    @Test func batchedOrMtpServingCannotFeedAGridAndSaysWhy() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        view.modelLoaded(
            token: UUID(),
            info: Self.info(
                "ornith", shape: Self.routed,
                unavailable: "batched serving (--max-concurrent-sequences above 1) does not record routing"
            ))
        #expect(view.currentPlanForTesting == .compact(width: 78, height: 9))
        #expect(harness.noteLines.count == 1)
        #expect(harness.noteLines[0].contains("does not record routing"))
    }

    @Test func theRingFollowsTheRunnerThatServesTheRequest() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        let first = FakeTraceHost(shape: Self.routed)
        let firstToken = UUID()
        view.modelLoaded(token: firstToken, info: Self.info("ornith", shape: Self.routed))
        // Nothing records until a request asks, once.
        #expect(first.enableCalls == 0)
        view.attach(token: firstToken, runner: first)
        view.attach(token: firstToken, runner: first)
        #expect(first.enableCalls == 1)
        #expect(view.currentRingForTesting === first.ring)

        // A catalog switch: a new model of another shape, a new runner.
        let second = FakeTraceHost(shape: Self.routed512)
        let secondToken = UUID()
        view.modelLoaded(token: secondToken, info: Self.info("qwen3.8", shape: Self.routed512))
        #expect(view.currentRingForTesting == nil)
        // The old model's late unload must not clear the new one.
        view.modelUnloaded(token: firstToken)
        view.attach(token: secondToken, runner: second)
        #expect(second.enableCalls == 1)
        #expect(view.currentRingForTesting === second.ring)
        #expect(view.currentRingForTesting !== first.ring)
        #expect(view.currentRingForTesting?.shape.numExperts == 512)

        // Unloading the resident model drops the ring (its runner is gone).
        view.modelUnloaded(token: secondToken)
        #expect(view.currentRingForTesting == nil)
        // A request for a runner the view was not told about records nothing.
        let stray = FakeTraceHost(shape: Self.routed)
        view.attach(token: UUID(), runner: stray)
        #expect(stray.enableCalls == 0)
    }

    @Test func aCompactModelNeverEnablesTheHook() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        let dense = FakeTraceHost(shape: Self.dense)
        let token = UUID()
        view.modelLoaded(token: token, info: Self.info("qwen3.5-4b", shape: Self.dense))
        view.attach(token: token, runner: dense)
        #expect(dense.enableCalls == 0)

        let batched = FakeTraceHost(shape: Self.routed)
        let batchedToken = UUID()
        view.modelLoaded(
            token: batchedToken,
            info: Self.info("ornith", shape: Self.routed, unavailable: "batched"))
        view.attach(token: batchedToken, runner: batched)
        #expect(batched.enableCalls == 0)
    }

    @Test func aSwitchRelaysTheRegionOutAndBackWithoutLeavingAStaleFrame() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        view.modelLoaded(token: UUID(), info: Self.info("ornith", shape: Self.routed))
        #expect(view.open(startRenderThread: false))
        view.drawFrameForTesting()
        #expect(view.composeFrameForTesting().count == 30)
        _ = harness.terminalOutput(wait: 0.05)

        view.modelLoaded(token: UUID(), info: Self.info("qwen3.5-4b", shape: Self.dense))
        view.drawFrameForTesting()
        let compact = harness.terminalOutput(wait: 0.1)
        // Cleared from the region's first line to the end of the screen, nine
        // lines reserved, the cursor back on the first of them, then the frame.
        #expect(compact.contains("\r\u{1B}[J" + String(repeating: "\n", count: 9)))
        #expect(compact.contains("\u{1B}[9A"))
        #expect(compact.contains("qwen3.5-4b"))
        #expect(!compact.contains("\u{1B}[30A"))

        view.modelLoaded(token: UUID(), info: Self.info("qwen3.8", shape: Self.routed512))
        view.drawFrameForTesting()
        let full = harness.terminalOutput(wait: 0.1)
        #expect(full.contains("\r\u{1B}[J" + String(repeating: "\n", count: 30)))
        #expect(full.contains("\u{1B}[30A"))
        #expect(full.contains("qwen3.8"))
    }

    // MARK: requests

    @Test func theHeaderFollowsARequestFromQueueToPrefillToDecode() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        view.modelLoaded(token: UUID(), info: Self.info("ornith-1.5-35b-a3b", shape: Self.routed))
        #expect(view.open(startRenderThread: false))

        view.requestAccepted(id: "req-1", client: "gpt-5-codex")
        view.requestAccepted(id: "req-2", client: "claude-sonnet")
        var header = Self.header(view.composeFrameForTesting())
        #expect(header.contains("queue 2"))
        #expect(header.contains("idle"))

        view.requestGenerating(id: "req-1")
        let generation = view.beginGeneration()
        generation.handle(.prefill(done: 2048, total: 9000))
        header = Self.header(view.composeFrameForTesting())
        #expect(header.contains("queue 1"))
        #expect(header.contains("← gpt-5-codex"))
        #expect(header.contains("prefill 2048/9000 tok"))
        #expect(header.contains("ornith-1.5-35b-a3b · 8-bit"))

        generation.handle(.prefill(done: 9000, total: 9000))
        generation.handle(.token(index: 0, id: 1, delta: "Hello"))
        generation.handle(.token(index: 1, id: 2, delta: " world"))
        let lines = view.composeFrameForTesting()
        header = Self.header(lines)
        #expect(header.contains("2 tok"))
        #expect(header.contains("queue 1"))
        #expect(Self.plain(lines).contains("Hello world"))

        generation.complete(promptTokens: 9000, cachedTokens: 4500, newTokens: 2)
        generation.close()
        view.requestClosed(id: "req-1")
        // The second request is next, still waiting.
        header = Self.header(view.composeFrameForTesting())
        #expect(header.contains("queue 1"))
        #expect(header.contains("idle"))
        #expect(header.contains("cache 50%"))
    }

    @Test func aCancelledRequestLeavesTheQueue() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        #expect(view.open(startRenderThread: false))
        view.requestAccepted(id: "gone", client: "x")
        #expect(Self.header(view.composeFrameForTesting()).contains("queue 1"))
        view.requestClosed(id: "gone")
        #expect(!Self.header(view.composeFrameForTesting()).contains("queue"))
    }

    @Test func aGenerationClaimsTheOldestRunningRequestAndAnInternalOneHasNone() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        #expect(view.open(startRenderThread: false))
        view.requestAccepted(id: "a", client: "client-a")
        view.requestAccepted(id: "b", client: "client-b")
        view.requestGenerating(id: "a")
        view.requestGenerating(id: "b")
        let first = view.beginGeneration()
        #expect(first.client == "client-a")
        let second = view.beginGeneration()
        #expect(second.client == "client-b")
        // Batched serving: two running, the header says so and follows the newer.
        let header = Self.header(view.composeFrameForTesting())
        #expect(header.contains("2 running"))
        #expect(header.contains("client-b"))
        first.close()
        second.close()
        view.requestClosed(id: "a")
        view.requestClosed(id: "b")
        let internalGeneration = view.beginGeneration()
        #expect(internalGeneration.client == "(internal)")
        internalGeneration.close()
        internalGeneration.close()
    }

    @Test func theIdleFrameKeepsTheLastRequestsSpeedAndCacheHit() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        view.modelLoaded(token: UUID(), info: Self.info("ornith-1.5-35b-a3b", shape: Self.routed))
        #expect(view.open(startRenderThread: false))
        view.requestAccepted(id: "r", client: "gpt-5")
        view.requestGenerating(id: "r")
        let generation = view.beginGeneration()
        for index in 0..<6 {
            generation.handle(.token(index: index, id: 1, delta: "tok\(index) "))
            Thread.sleep(forTimeInterval: 0.02)
        }
        generation.complete(promptTokens: 200, cachedTokens: 150, newTokens: 6)
        generation.close()
        view.requestClosed(id: "r")
        let lines = view.composeFrameForTesting()
        let header = Self.header(lines)
        #expect(header.contains("idle"))
        #expect(header.contains("tok/s"))
        #expect(header.contains("cache 75%"))
        #expect(!header.contains("gpt-5"))
        #expect(Self.plain(lines).contains("tok4 tok5"))
        #expect(Self.plain(lines).contains("output · last reply"))
    }

    @Test func theOutputAreaShowsTheTailOfTheReplyInTwoLines() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        view.modelLoaded(token: UUID(), info: Self.info("qwen3.5-4b", shape: Self.dense))
        #expect(view.open(startRenderThread: false))
        view.requestAccepted(id: "r", client: "c")
        view.requestGenerating(id: "r")
        let generation = view.beginGeneration()
        generation.handle(.token(index: 0, id: 1, delta: "first line\nsecond line\nthird line"))
        let text = Self.plain(view.composeFrameForTesting())
        #expect(text.contains("second line\nthird line"))
        #expect(!text.contains("first line"))
        generation.close()
    }

    @Test func aLongReplyIsNotKeptWhole() throws {
        let generation = LiveTraceFeed(maxNew: 0, retainAll: false)
        for _ in 0..<200 { generation.noteToken(String(repeating: "x", count: 1000)) }
        generation.finish()
        #expect(generation.fullText().utf8.count < 200_000)
        #expect(generation.snapshot().textTail.count == LiveTraceFeed.tailCharacters)
        let cli = LiveTraceFeed(maxNew: 0)
        for _ in 0..<200 { cli.noteToken(String(repeating: "x", count: 1000)) }
        cli.finish()
        #expect(cli.fullText().utf8.count == 200_000)
    }

    @Test func aModelBeingLoadedAndAFailedLoadAreShownAndForgotten() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        #expect(view.open(startRenderThread: false))
        #expect(Self.header(view.composeFrameForTesting()).contains("no model loaded"))
        view.modelLoading("qwen3.5-9b_4-bit")
        #expect(Self.header(view.composeFrameForTesting()).contains("loading qwen3.5-9b_4-bit…"))
        view.modelLoadFailed()
        #expect(Self.header(view.composeFrameForTesting()).contains("no model loaded"))
        view.modelLoading("ornith")
        let token = UUID()
        view.modelLoaded(token: token, info: Self.info("ornith", shape: Self.routed))
        #expect(Self.header(view.composeFrameForTesting()).contains("ornith · 8-bit"))
        view.modelUnloaded(token: token)
        #expect(Self.header(view.composeFrameForTesting()).contains("no model loaded"))
    }

    @Test func anOperatorsModelIdNamesTheModelInTheHeader() throws {
        let harness = try #require(ViewHarness(nameOverride: "my-local-model"))
        let view = try #require(harness.view)
        view.modelLoaded(token: UUID(), info: Self.info("ornith", shape: Self.routed))
        #expect(view.open(startRenderThread: false))
        #expect(Self.header(view.composeFrameForTesting()).contains("my-local-model · 8-bit"))
    }

    // MARK: log routing

    @Test func logLinesReachTheTailInRedAndEveryWhereTheyWentBefore() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        #expect(view.open(startRenderThread: false))
        #expect(
            harness.noteLines.contains {
                $0.contains("the full server log is in \(harness.logPath)")
            })

        // stderr was already redirected (a pipe): it is passed through verbatim.
        let errLine = "[2026-10-02T13:04:05Z] request r failed phase=generating error=boom\n"
        _ = errLine.withCString { write(harness.stderrTarget, $0, strlen($0)) }
        // stdout is the terminal: it can only go to the file.
        let outLine = "TinyTitan [decode expert io] hits 400\n"
        _ = outLine.withCString { write(harness.stdoutTarget, $0, strlen($0)) }

        let lines = harness.waitForFrame(containing: "[decode expert io]")
        let joined = lines.joined(separator: "\n")
        #expect(joined.contains("error=boom") || joined.contains("error=b"))
        // The error line is red, the ordinary one is not.
        let red = LiveTraceTheme.foreground(.miss, depth: .ansi256)
        let errorRow = try #require(lines.first { $0.contains("failed phase") })
        #expect(errorRow.contains(red))
        let infoRow = try #require(lines.first { $0.contains("[decode expert io]") })
        #expect(!infoRow.contains(red))

        view.close()
        // The redirected stderr received every byte, unchanged.
        _ = fcntl(harness.stderrRead, F_SETFL, O_NONBLOCK)
        var buffer = [UInt8](repeating: 0, count: 4096)
        let count = read(harness.stderrRead, &buffer, buffer.count)
        let received = String(decoding: buffer[0..<max(0, count)], as: UTF8.self)
        #expect(received.hasPrefix(errLine))
        // The file holds both streams, in full.
        let file = harness.logFileText()
        #expect(file.contains(errLine))
        #expect(file.contains(outLine))
        #expect(file.contains("live-trace view opened"))
        #expect(file.contains("live-trace view closed"))
    }

    @Test func aLogFileThatCannotBeOpenedFallsBackToHoldingAndReplaying() throws {
        let harness = try #require(ViewHarness())
        // A path whose parent is a regular file cannot be created.
        let blocker = NSTemporaryDirectory() + "live-trace-blocker-\(UUID().uuidString)"
        FileManager.default.createFile(atPath: blocker, contents: Data())
        defer { try? FileManager.default.removeItem(atPath: blocker) }
        let collector = harness.notes
        let prepared = LiveTraceServerView.prepare(
            options: .init(logPath: blocker + "/x/server.log"), probe: ViewHarness.tty,
            stdoutFD: harness.stdoutTarget, stderrFD: harness.stderrTarget,
            note: { line in collector.add(line) })
        let view = try #require(prepared)
        #expect(view.open(startRenderThread: false))
        #expect(harness.noteLines.contains { $0.contains("could not open") })
        let text = "held for the terminal\n"
        _ = text.withCString { write(harness.stdoutTarget, $0, strlen($0)) }
        _ = harness.waitForFrame(containing: "held for the terminal")
        view.close()
        // The held stdout is written back to the terminal on close, so nothing is lost.
        #expect(harness.terminalOutput(wait: 0.2).contains("held for the terminal"))
    }

    @Test func closingPutsStdoutAndStderrBackAndTheTerminalRight() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        view.modelLoaded(token: UUID(), info: Self.info("ornith", shape: Self.routed))
        #expect(view.open(startRenderThread: false))
        #expect(view.isOpen)
        let opened = harness.terminalOutput(wait: 0.1)
        #expect(opened.contains("\u{1B}[?25l"))
        #expect(opened.contains("\u{1B}[?7l"))
        view.close()
        view.close()
        #expect(!view.isOpen)
        let closed = harness.terminalOutput(wait: 0.1)
        #expect(closed.contains("\u{1B}[30B"))
        #expect(closed.contains("\u{1B}[?7h\u{1B}[?25h"))
        // After close, a write to the captured descriptors goes where it always
        // went: the terminal for stdout, the pipe for stderr.
        let text = "after close\n"
        _ = text.withCString { write(harness.stdoutTarget, $0, strlen($0)) }
        #expect(harness.terminalOutput(wait: 0.1).contains("after close"))
        #expect(!harness.logFileText().contains("after close"))
        // And the guard is disarmed: nothing left to restore.
        #expect(!LiveTraceTerminalGuard.isArmed)
    }

    // MARK: signals and exit

    @Test func aSecondSignalsRestorePutsTheTerminalBackOnceAndStopsTheDrawing() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        view.modelLoaded(token: UUID(), info: Self.info("ornith", shape: Self.routed))
        #expect(view.open(startRenderThread: false))
        view.drawFrameForTesting()
        _ = harness.terminalOutput(wait: 0.05)
        #expect(LiveTraceTerminalGuard.isArmed)

        LiveTraceServerView.restoreTerminal()
        let restored = harness.terminalOutput(wait: 0.1)
        #expect(restored.contains("\u{1B}[30B\r"))
        #expect(restored.hasSuffix("\u{1B}[?7h\u{1B}[?25h"))
        #expect(!LiveTraceTerminalGuard.isArmed)

        // Idempotent, and a render thread caught mid-frame cannot redraw over
        // the shell's prompt.
        LiveTraceServerView.restoreTerminal()
        view.drawFrameForTesting()
        #expect(harness.terminalOutput(wait: 0.1).isEmpty)
        view.close()
    }

    @Test func theRenderThreadDrawsOnItsOwnAndStopsWhenClosed() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        view.modelLoaded(token: UUID(), info: Self.info("qwen3.5-4b", shape: Self.dense))
        #expect(view.open())
        view.requestAccepted(id: "r", client: "c")
        view.requestGenerating(id: "r")
        let generation = view.beginGeneration()
        generation.handle(.token(index: 0, id: 1, delta: "streamed"))
        let drawn = harness.terminalOutput(wait: 0.5)
        #expect(drawn.contains("streamed"))
        generation.close()
        view.close()
        _ = harness.terminalOutput(wait: 0.05)
        Thread.sleep(forTimeInterval: 0.2)
        #expect(harness.terminalOutput(wait: 0.1).isEmpty)
    }

    @Test func anIdleViewStopsRepeatingIdenticalFrames() throws {
        let harness = try #require(ViewHarness())
        let view = try #require(harness.view)
        view.modelLoaded(token: UUID(), info: Self.info("qwen3.5-4b", shape: Self.dense))
        #expect(view.open())
        _ = harness.terminalOutput(wait: 0.3)
        // Quiet for a while: 20 fps of nothing would be 100 KB/s.
        let quiet = harness.terminalOutput(wait: 0.5)
        #expect(quiet.count < 200)
        view.close()
    }
}
