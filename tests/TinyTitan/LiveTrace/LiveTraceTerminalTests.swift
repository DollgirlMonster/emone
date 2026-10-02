import Foundation
import Testing
import TinyTitan

@testable import TinyTitanCLICore
@testable import TinyTitanLiveTrace

/// Opens a pseudo-terminal of a given size, so the TTY paths are exercised
/// against a real one rather than a stub.
struct PseudoTerminal {
    let master: Int32
    let slave: Int32

    init?(cols: UInt16 = 100, rows: UInt16 = 40) {
        var master: Int32 = -1
        var slave: Int32 = -1
        var size = winsize(ws_row: rows, ws_col: cols, ws_xpixel: 0, ws_ypixel: 0)
        guard openpty(&master, &slave, nil, nil, &size) == 0 else { return nil }
        self.master = master
        self.slave = slave
    }

    func close() {
        Darwin.close(master)
        Darwin.close(slave)
    }
}

@Suite struct LiveTraceTerminalTests {
    static let routed = ExpertGridLayoutTests.shape(experts: 512, topK: 10)
    static let tty = TerminalProbe(
        stdoutIsTTY: true, term: "xterm-256color", colorTerm: nil, noColor: false, cols: 100,
        rows: 40)

    // MARK: the flag and TTY gating

    @Test func offByDefaultEvenOnAPerfectTerminal() throws {
        let args = try Args.parse(["--model", "m", "--prompt", "hi"])
        #expect(!args.liveTrace)
        #expect(
            LiveTraceGate.decide(requested: args.liveTrace, probe: Self.tty, shape: Self.routed)
                == .off)
    }

    @Test func oneFlagTurnsItOn() throws {
        let args = try Args.parse(["--model", "m", "--prompt", "hi", "--live-trace"])
        #expect(args.liveTrace)
        #expect(Args.usage.contains("--live-trace"))
        #expect(
            LiveTraceGate.decide(requested: args.liveTrace, probe: Self.tty, shape: Self.routed)
                == .run(plan: .full(width: 78, height: 30), depth: .ansi256))
    }

    @Test func theFlagChangesNothingElseInTheArguments() throws {
        let base = try Args.parse(["--model", "m", "--prompt", "hi", "--seed", "3"])
        var flagged = try Args.parse([
            "--model", "m", "--prompt", "hi", "--seed", "3", "--live-trace",
        ])
        flagged.liveTrace = false
        #expect(flagged == base)
    }

    @Test func scoringHasNoDecodeToShow() {
        #expect(throws: ArgsError.mutuallyExclusive("--live-trace", "--score")) {
            _ = try Args.parse(["--model", "m", "--prompt", "hi", "--live-trace", "--score", "4"])
        }
    }

    @Test func nothingIsDrawnWhenStdoutIsNotATerminal() {
        var probe = Self.tty
        probe.stdoutIsTTY = false
        #expect(
            LiveTraceGate.decide(requested: true, probe: probe, shape: Self.routed)
                == .disabled(reason: "stdout is not a terminal"))
    }

    @Test func aDumbOrUnsetTermDrawsNothing() {
        for term in [nil, "", "dumb"] as [String?] {
            var probe = Self.tty
            probe.term = term
            if case .disabled = LiveTraceGate.decide(
                requested: true, probe: probe, shape: Self.routed)
            {
            } else {
                Issue.record("TERM \(String(describing: term)) should disable the view")
            }
        }
    }

    @Test func noColorKeepsTheViewButDropsColour() {
        var probe = Self.tty
        probe.noColor = true
        #expect(
            LiveTraceGate.decide(requested: true, probe: probe, shape: Self.routed)
                == .run(plan: .full(width: 78, height: 30), depth: .none))
    }

    @Test func aTerminalTooSmallIsDisabledWithAReason() {
        var probe = Self.tty
        probe.cols = 20
        if case .disabled(let reason) = LiveTraceGate.decide(
            requested: true, probe: probe, shape: Self.routed)
        {
            #expect(reason.contains("columns"))
        } else {
            Issue.record("20 columns should disable the view")
        }
    }

    @Test func aPipeIsNotATerminalAndAPseudoTerminalIs() throws {
        var fds: [Int32] = [0, 0]
        #expect(pipe(&fds) == 0)
        let piped = TerminalProbe.current(stdoutFD: fds[1], environment: ["TERM": "xterm"])
        #expect(!piped.stdoutIsTTY)
        close(fds[0])
        close(fds[1])

        let pty = try #require(PseudoTerminal(cols: 123, rows: 45))
        defer { pty.close() }
        let probe = TerminalProbe.current(
            stdoutFD: pty.slave, environment: ["TERM": "xterm-256color", "NO_COLOR": "1"])
        #expect(probe.stdoutIsTTY)
        #expect((probe.cols, probe.rows) == (123, 45))
        #expect(probe.noColor)
        #expect(probe.term == "xterm-256color")
        #expect(
            LiveTraceGate.decide(requested: true, probe: probe, shape: Self.routed)
                == .run(plan: .full(width: 78, height: 30), depth: .none))
        #expect(!TerminalProbe.current(stdoutFD: pty.slave, environment: [:]).noColor)
        #expect(
            TerminalProbe.current(stdoutFD: pty.slave, environment: ["NO_COLOR": ""]).noColor
                == false)
    }

    // MARK: the in-place region

    @Test func regionSequencesParkTheCursorOnItsFirstLine() {
        let region = LiveTraceRegion(fd: -1, height: 3)
        #expect(region.openSequence() == "\u{1B}[?25l\u{1B}[?7l\n\n\n\u{1B}[3A\r")
        #expect(
            region.frameSequence(["a", "b"])
                == "\r\u{1B}[2Ka\u{1B}[0m\n\r\u{1B}[2Kb\u{1B}[0m\n\r\u{1B}[2K\u{1B}[0m\n\u{1B}[3A")
        #expect(region.closeSequence() == "\u{1B}[3B\r\u{1B}[0m\u{1B}[?7h\u{1B}[?25h")
    }

    // MARK: stderr capture

    @Test func logTailKeepsTheMostRecentLines() {
        let tail = LiveTraceLogTail()
        for index in 0..<(LiveTraceLogTail.capacity + 5) { tail.append("line \(index)") }
        let recent = tail.recent()
        #expect(recent.count == LiveTraceLogTail.capacity)
        #expect(recent.last == "line \(LiveTraceLogTail.capacity + 4)")
        #expect(recent.first == "line 5")
    }

    @Test func byteQueueDropsOldestPastItsLimit() {
        var queue = LiveTraceByteQueue(limit: 10)
        queue.push([UInt8](repeating: 1, count: 6))
        queue.push([UInt8](repeating: 2, count: 6))
        #expect(queue.droppedChunks == 1)
        #expect(queue.takeAll() == [[UInt8](repeating: 2, count: 6)])
        #expect(queue.bytes == 0)
    }

    private func readAll(_ fd: Int32) -> String {
        var out = [UInt8]()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let count = read(fd, &buffer, buffer.count)
            if count <= 0 { break }
            out.append(contentsOf: buffer[0..<count])
        }
        return String(decoding: out, as: UTF8.self)
    }

    @Test func captureTeesToANonTerminalSinkUnchangedAndFeedsTheLog() throws {
        var fds: [Int32] = [0, 0]
        #expect(pipe(&fds) == 0)
        let target = dup(fds[1])
        close(fds[1])
        let log = LiveTraceLogTail()
        let capture = try #require(LiveTraceStderrCapture.start(targetFD: target, log: log))
        let text = "first line\nERROR second\r\nthird \u{1B}[31mred\u{1B}[0m\nno newline"
        _ = text.withCString { write(target, $0, strlen($0)) }
        let stopped = capture.stop()
        // Passed through live, so nothing is held back.
        #expect(stopped.held.isEmpty)
        close(target)
        #expect(readAll(fds[0]) == text)
        close(fds[0])
        #expect(log.recent() == ["first line", "ERROR second", "third red", "no newline"])
    }

    @Test func captureHoldsForATerminalAndHandsItBackOnStop() throws {
        let pty = try #require(PseudoTerminal())
        defer { pty.close() }
        let log = LiveTraceLogTail()
        let capture = try #require(LiveTraceStderrCapture.start(targetFD: pty.slave, log: log))
        let text = "[stop=eos new=3tok]\n"
        _ = text.withCString { write(pty.slave, $0, strlen($0)) }
        let stopped = capture.stop()
        #expect(String(decoding: stopped.held.flatMap { $0 }, as: UTF8.self) == text)
        #expect(log.recent() == ["[stop=eos new=3tok]"])
    }

    // MARK: the render thread

    @Test func theRenderThreadDrawsFromTheRingWithoutAModelOrATerminal() throws {
        let shape = Self.routed
        let ring = ExpertTraceRing(shape: shape)
        let path = NSTemporaryDirectory() + "live-trace-\(UUID().uuidString).out"
        let fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o600)
        #expect(fd >= 0)
        defer { unlink(path) }
        let feed = LiveTraceFeed(maxNew: 8)
        let region = LiveTraceRegion(fd: fd, height: 30)
        let renderer = LiveTraceRenderer(
            ring: ring, shape: shape, slotsPerLayer: 96, plan: .full(width: 78, height: 30),
            depth: .truecolor, feed: feed, log: LiveTraceLogTail(), region: region)
        renderer.start()
        feed.notePrefill(done: 5, total: 5)
        feed.noteToken("hello")
        for layer in 0..<4 {
            ring.record(layer: layer, position: 0, experts: [1, 2, 3], missIndices: [0])
        }
        Thread.sleep(forTimeInterval: 0.5)
        feed.finish()
        renderer.stop()
        close(fd)
        let written = try String(contentsOfFile: path, encoding: .utf8)
        // At least a handful of frames at 20 fps, each parking the cursor again.
        #expect(written.components(separatedBy: "\u{1B}[30A").count > 3)
        #expect(written.contains("emone"))
        #expect(written.contains("MISS"))
        #expect(written.contains("hello"))
        #expect(ring.pending == 0)
    }

    @Test func theFeedKeepsEveryTokenInOrder() {
        let feed = LiveTraceFeed(maxNew: 4)
        feed.noteToken("a")
        _ = feed.snapshot()
        for word in ["b", "c", "d"] { feed.noteToken(word) }
        feed.finish()
        #expect(feed.fullText() == "abcd")
        let snapshot = feed.snapshot()
        #expect(snapshot.phase == .done && snapshot.tokens == 4)
        #expect(snapshot.textTail == "abcd")
    }
}
