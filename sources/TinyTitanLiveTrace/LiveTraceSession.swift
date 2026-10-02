import Dispatch
import Foundation
import Synchronization
import TinyTitan

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
