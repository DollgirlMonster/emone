import Dispatch
import Foundation
import Synchronization
import TinyTitan

/// What the decode task tells the view: phase, prefill progress, and the
/// generated text. The decode thread is the only writer.
///
/// Nothing here can make the decode thread wait for the renderer. Counters are
/// plain atomic stores; the text and token times go through a mutex the writer
/// only *tries* to take, and anything it could not hand over is kept and handed
/// over with the next token.
///
/// unchecked-invariant: `pendingText` and `pendingTicks` are touched only by the
/// writer (the decode task, then `finish` from the same task); `shared` is a
/// `Mutex`; the rest are atomics.
final class LiveTraceFeed: @unchecked Sendable {
    private struct Shared {
        var text = ""
        var ticks: [UInt64] = []
    }

    static let rateWindow = 8
    static let tailCharacters = 4096

    private let phase = Atomic<Int>(0)
    private let prefillDone = Atomic<Int>(0)
    private let prefillTotal = Atomic<Int>(0)
    private let tokens = Atomic<Int>(0)
    private let shared = Mutex(Shared())
    private var pendingText = ""
    private var pendingTicks: [UInt64] = []
    let maxNew: Int

    init(maxNew: Int) { self.maxNew = maxNew }

    // MARK: writer (decode task)

    func notePrefill(done: Int, total: Int) {
        prefillDone.store(done, ordering: .relaxed)
        prefillTotal.store(total, ordering: .relaxed)
        phase.store(LiveTraceFeedSnapshot.Phase.prefill.rawValue, ordering: .relaxed)
    }

    func noteToken(_ delta: String) {
        pendingText += delta
        pendingTicks.append(mach_absolute_time())
        tokens.store(tokens.load(ordering: .relaxed) + 1, ordering: .relaxed)
        phase.store(LiveTraceFeedSnapshot.Phase.decode.rawValue, ordering: .relaxed)
        _ = shared.withLockIfAvailable { handOver(&$0) }
    }

    func noteTail(_ tail: String) {
        pendingText += tail
        _ = shared.withLockIfAvailable { handOver(&$0) }
    }

    func finish() {
        shared.withLock { handOver(&$0) }
        phase.store(LiveTraceFeedSnapshot.Phase.done.rawValue, ordering: .releasing)
    }

    private func handOver(_ shared: inout Shared) {
        shared.text += pendingText
        shared.ticks.append(contentsOf: pendingTicks)
        if shared.ticks.count > Self.rateWindow {
            shared.ticks.removeFirst(shared.ticks.count - Self.rateWindow)
        }
        pendingText.removeAll(keepingCapacity: true)
        pendingTicks.removeAll(keepingCapacity: true)
    }

    // MARK: reader (render thread, or finish after it has stopped)

    func snapshot() -> LiveTraceFeedSnapshot {
        var out = LiveTraceFeedSnapshot()
        out.phase =
            LiveTraceFeedSnapshot.Phase(rawValue: phase.load(ordering: .acquiring)) ?? .starting
        out.prefillDone = prefillDone.load(ordering: .relaxed)
        out.prefillTotal = prefillTotal.load(ordering: .relaxed)
        out.tokens = tokens.load(ordering: .relaxed)
        out.maxNew = maxNew
        shared.withLock { shared in
            out.textTail = String(shared.text.suffix(Self.tailCharacters))
            if shared.ticks.count >= 2, let first = shared.ticks.first, let last = shared.ticks.last
            {
                let nanos = Double(ExpertTraceRing.nanoseconds(fromTicks: last &- first))
                if nanos > 0 { out.tokensPerSecond = Double(shared.ticks.count - 1) * 1e9 / nanos }
            }
        }
        return out
    }

    /// Everything generated, for printing once the view has closed.
    func fullText() -> String { shared.withLock { $0.text } }
}

/// Draws the view on its own thread, never the decode thread. It samples: at
/// 20 frames a second it takes whatever the ring and the feed hold and draws
/// the latest state.
///
/// unchecked-invariant: `model`, `batch`, `frame` and `spinner` belong to the
/// render thread while it runs, and to `stop` after it has been joined; the
/// stop flag is an atomic and the join is a semaphore.
final class LiveTraceRenderer: @unchecked Sendable {
    static let framesPerSecond = 20

    private let ring: ExpertTraceRing?
    private let shape: ExpertTraceShape
    private let plan: LiveTraceViewPlan
    private let depth: LiveTraceColorDepth
    private let feed: LiveTraceFeed
    private let log: LiveTraceLogTail
    private let region: LiveTraceRegion
    private var model: LiveTraceModel?
    private var batch = ExpertTraceBatch()
    private var frame = 0
    private let stopFlag = Atomic<Bool>(false)
    private let finished = DispatchSemaphore(value: 0)

    init(
        ring: ExpertTraceRing?, shape: ExpertTraceShape, slotsPerLayer: Int?,
        plan: LiveTraceViewPlan, depth: LiveTraceColorDepth, feed: LiveTraceFeed,
        log: LiveTraceLogTail, region: LiveTraceRegion
    ) {
        self.ring = ring
        self.shape = shape
        self.plan = plan
        self.depth = depth
        self.feed = feed
        self.log = log
        self.region = region
        if case .full = plan {
            model = LiveTraceModel(shape: shape, slotsPerLayer: slotsPerLayer)
        }
    }

    func start() {
        let thread = Thread { [self] in
            let interval = 1.0 / Double(Self.framesPerSecond)
            while !stopFlag.load(ordering: .acquiring) {
                drawFrame()
                Thread.sleep(forTimeInterval: interval)
            }
            finished.signal()
        }
        thread.qualityOfService = .utility
        thread.name = "emone.live-trace"
        thread.start()
    }

    /// Join the render thread and draw one last frame on the caller's.
    func stop() {
        stopFlag.store(true, ordering: .releasing)
        finished.wait()
        drawFrame()
    }

    func drawFrame() {
        if let ring {
            ring.drain(into: &batch)
            model?.apply(batch)
        }
        frame += 1
        let lines = LiveTraceFrame.compose(
            LiveTraceFrameInput(
                shape: shape, model: model, feed: feed.snapshot(), log: log.recent(),
                plan: plan, spinner: frame / 2, depth: depth))
        region.draw(lines)
    }
}

/// `--live-trace` for one CLI run: gate it, hook the runner, redirect stderr
/// into the log region, draw, and put everything back.
///
/// unchecked-invariant: every stored property is immutable after `begin`; the
/// runner is only touched by `finish`, on the task that drives it.
public final class LiveTraceSession: @unchecked Sendable {
    private let feed: LiveTraceFeed
    private let renderer: LiveTraceRenderer
    private let region: LiveTraceRegion
    private let capture: LiveTraceStderrCapture
    private let stdout: FileHandle
    private let runner: RealForwardRunner

    private init(
        feed: LiveTraceFeed, renderer: LiveTraceRenderer, region: LiveTraceRegion,
        capture: LiveTraceStderrCapture, stdout: FileHandle, runner: RealForwardRunner
    ) {
        self.feed = feed
        self.renderer = renderer
        self.region = region
        self.capture = capture
        self.stdout = stdout
        self.runner = runner
    }

    /// nil when the flag is off, or when this terminal cannot show the view (a
    /// note says why, on stderr, and the run goes on as if the flag were absent).
    public static func begin(
        requested: Bool, runner: RealForwardRunner, maxNew: Int, slotsPerLayer: Int?,
        stdout: FileHandle, stderr: FileHandle
    ) -> LiveTraceSession? {
        let probe = TerminalProbe.current(stdoutFD: stdout.fileDescriptor)
        let shape = runner.expertTraceShape()
        switch LiveTraceGate.decide(requested: requested, probe: probe, shape: shape) {
        case .off:
            return nil
        case .disabled(let reason):
            stderr.write(Data("note: --live-trace is off: \(reason)\n".utf8))
            return nil
        case .run(let plan, let depth):
            guard let height = plan.height else { return nil }
            // A routed model on a terminal too small for the grid gets the
            // compact view; say so, or the missing grid looks like a bug.
            if case .compact = plan, let full = LiveTraceViewPlan.fullHeight(shape: shape) {
                stderr.write(Data((
                    "note: --live-trace grid hidden: the terminal is \(probe.cols)x\(probe.rows); "
                    + "the grid needs at least \(LiveTraceViewPlan.fullWidth)x\(full + 1)\n").utf8))
            }
            let log = LiveTraceLogTail()
            guard let capture = LiveTraceStderrCapture.start(log: log) else {
                stderr.write(Data("note: --live-trace is off: could not capture stderr\n".utf8))
                return nil
            }
            // Only the full view has a grid to feed; the compact view needs no hook.
            var ring: ExpertTraceRing?
            if case .full = plan { ring = runner.enableExpertTrace() }
            let feed = LiveTraceFeed(maxNew: maxNew)
            let region = LiveTraceRegion(fd: stdout.fileDescriptor, height: height)
            let renderer = LiveTraceRenderer(
                ring: ring, shape: shape, slotsPerLayer: slotsPerLayer, plan: plan,
                depth: depth, feed: feed, log: log, region: region)
            region.open()
            renderer.start()
            return LiveTraceSession(
                feed: feed, renderer: renderer, region: region, capture: capture,
                stdout: stdout, runner: runner)
        }
    }

    /// Route the run's progress into the view instead of straight to stdout.
    public func handle(_ progress: RawDecodeProgress) {
        switch progress {
        case .prefill(let done, let total): feed.notePrefill(done: done, total: total)
        case .token(_, _, let delta): feed.noteToken(delta)
        case .tail(let tail): feed.noteTail(tail)
        }
    }

    /// Stop drawing, leave the last frame on screen, restore stderr, then print
    /// the full generated text and any stderr held back, so scrollback holds
    /// them in the order a plain run would have.
    public func finish() {
        runner.disableExpertTrace()
        feed.finish()
        renderer.stop()
        region.close()
        let stopped = capture.stop()
        var text = feed.fullText()
        if !text.isEmpty && !text.hasSuffix("\n") { text += "\n" }
        stdout.write(Data(text.utf8))
        for chunk in stopped.held { liveTraceWriteAll(STDERR_FILENO, chunk) }
        if stopped.droppedChunks > 0 {
            let note = "note: \(stopped.droppedChunks) early stderr chunks were dropped\n"
            liveTraceWriteAll(STDERR_FILENO, Array(note.utf8))
        }
    }
}
