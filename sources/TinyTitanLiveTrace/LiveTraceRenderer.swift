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
    private let firstTick = Atomic<UInt64>(0)
    private let lastTick = Atomic<UInt64>(0)
    private let shared = Mutex(Shared())
    private var pendingText = ""
    private var pendingTicks: [UInt64] = []
    let maxNew: Int
    /// A CLI run prints everything it generated once the view closes, so it
    /// keeps all of it. A server's reply can be arbitrarily long and nothing
    /// prints it afterwards: only the tail the view can show is kept.
    private let retainAll: Bool

    init(maxNew: Int, retainAll: Bool = true) {
        self.maxNew = maxNew
        self.retainAll = retainAll
    }

    // MARK: writer (decode task)

    func notePrefill(done: Int, total: Int) {
        prefillDone.store(done, ordering: .relaxed)
        prefillTotal.store(total, ordering: .relaxed)
        phase.store(LiveTraceFeedSnapshot.Phase.prefill.rawValue, ordering: .relaxed)
    }

    func noteToken(_ delta: String) {
        pendingText += delta
        let now = mach_absolute_time()
        pendingTicks.append(now)
        let count = tokens.load(ordering: .relaxed)
        if count == 0 { firstTick.store(now, ordering: .relaxed) }
        lastTick.store(now, ordering: .relaxed)
        tokens.store(count + 1, ordering: .relaxed)
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
        if !retainAll, shared.text.utf8.count > 2 * Self.tailCharacters * 4 {
            shared.text = String(shared.text.suffix(Self.tailCharacters))
        }
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

    /// Tokens per second over the whole decode, first token to last, or nil
    /// with fewer than two tokens. What a server's idle frame calls the last
    /// request's speed.
    var averageTokensPerSecond: Double? {
        let count = tokens.load(ordering: .relaxed)
        guard count >= 2 else { return nil }
        let nanos = Double(
            ExpertTraceRing.nanoseconds(
                fromTicks: lastTick.load(ordering: .relaxed)
                    &- firstTick.load(ordering: .relaxed)))
        return nanos > 0 ? Double(count - 1) * 1e9 / nanos : nil
    }

    var tokenCount: Int { tokens.load(ordering: .relaxed) }

    /// Everything generated, for printing once the view has closed.
    func fullText() -> String { shared.withLock { $0.text } }
}

/// The model-dependent part of the view: which shape to draw, whether there is
/// a ring to draw it from, and the layout that follows. A server swaps this
/// when a catalog switch loads a model of a different shape.
struct LiveTraceScene: Sendable {
    var shape: ExpertTraceShape
    var plan: LiveTraceViewPlan
    /// nil in the compact view, and before the runner has been asked to record.
    var ring: ExpertTraceRing?
    var slotsPerLayer: Int?
    /// The terminal's height, when the log may grow into the rows below the
    /// plan (the server's view); nil keeps the frame at the plan's height.
    var terminalRows: Int?
}

/// What the render thread reads once a frame.
struct LiveTraceSnapshot: Sendable {
    var feed: LiveTraceFeedSnapshot
    /// nil in a CLI run.
    var server: LiveTraceServerStatus?
    /// Bumped by the server for every request, so the view knows to forget the
    /// token in flight.
    var generation = 0
}

/// Draws the view on its own thread, never the decode thread. It samples: at
/// 20 frames a second it takes whatever the ring and the feed hold and draws
/// the latest state.
///
/// A CLI run has one scene for its whole life. A server's scene can be replaced
/// from any thread with `setScene`: the render thread picks the replacement up
/// at the start of its next frame and re-lays the region out if the height
/// changed.
///
/// unchecked-invariant: `scene`, `model`, `batch`, `frame`, `spinner` and the
/// counters belong to the render thread while it runs, and to `stop` after it
/// has been joined; the pending scene is a `Mutex`, the stop flag an atomic and
/// the join a semaphore.
final class LiveTraceRenderer: @unchecked Sendable {
    static let framesPerSecond = 20
    /// Frames between idle heat decay steps: about a quarter of a second, so a
    /// recent favourite halves in roughly six seconds of quiet.
    static let framesPerIdleStep = 5

    private var scene: LiveTraceScene
    private let pending = Mutex<LiveTraceScene?>(nil)
    private let depth: LiveTraceColorDepth
    private let source: @Sendable () -> LiveTraceSnapshot
    private let log: LiveTraceLogTail
    private let region: LiveTraceRegion
    private var model: LiveTraceModel?
    private var batch = ExpertTraceBatch()
    private var frame = 0
    private var lastGeneration = 0
    private var settled = false
    private var idleFrames = 0
    private var lastDrawn: [String]?
    private var lastWasServer = false
    private let stopFlag = Atomic<Bool>(false)
    private let finished = DispatchSemaphore(value: 0)

    /// The CLI's renderer: one feed, one scene.
    convenience init(
        ring: ExpertTraceRing?, shape: ExpertTraceShape, slotsPerLayer: Int?,
        plan: LiveTraceViewPlan, depth: LiveTraceColorDepth, feed: LiveTraceFeed,
        log: LiveTraceLogTail, region: LiveTraceRegion
    ) {
        self.init(
            scene: LiveTraceScene(
                shape: shape, plan: plan, ring: ring, slotsPerLayer: slotsPerLayer),
            depth: depth, log: log, region: region,
            source: { LiveTraceSnapshot(feed: feed.snapshot()) })
    }

    init(
        scene: LiveTraceScene, depth: LiveTraceColorDepth, log: LiveTraceLogTail,
        region: LiveTraceRegion, source: @escaping @Sendable () -> LiveTraceSnapshot
    ) {
        self.scene = scene
        self.depth = depth
        self.log = log
        self.region = region
        self.source = source
        model = Self.makeModel(for: scene)
    }

    private static func makeModel(for scene: LiveTraceScene) -> LiveTraceModel? {
        guard case .full = scene.plan else { return nil }
        return LiveTraceModel(shape: scene.shape, slotsPerLayer: scene.slotsPerLayer)
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

    /// Join the render thread and draw one last frame on the caller's. The
    /// join is bounded: a render thread stuck on a terminal that stopped reading
    /// must not hang shutdown, and then no last frame is drawn over it.
    func stop() {
        stopFlag.store(true, ordering: .releasing)
        guard finished.wait(timeout: .now() + 1) == .success else { return }
        lastDrawn = nil
        drawFrame()
    }

    /// Replace the scene. Callable from any thread; takes effect on the next
    /// frame. Only the latest replacement matters.
    func setScene(_ next: LiveTraceScene) {
        pending.withLock { $0 = next }
    }

    private func applyPendingScene() {
        guard let next = pending.withLock({ pending -> LiveTraceScene? in
            let value = pending
            pending = nil
            return value
        })
        else { return }
        // The same model again (an idle unload and reload) keeps its heat; any
        // other shape starts a fresh picture.
        let sameShape = next.shape == scene.shape && next.slotsPerLayer == scene.slotsPerLayer
        let planChanged = next.plan != scene.plan
        scene = next
        if !(sameShape && !planChanged && model != nil) { model = Self.makeModel(for: next) }
        if let height = next.plan.height, height != region.height {
            region.relayout(to: height)
            lastDrawn = nil
        }
    }

    /// Take the state the view shows from the ring, the feed and the server, and
    /// compose the frame: everything `drawFrame` does but the write.
    func renderFrame() -> [String] {
        applyPendingScene()
        let snapshot = source()
        if snapshot.server != nil, snapshot.generation != lastGeneration {
            lastGeneration = snapshot.generation
            model?.clearLive()
            settled = false
        }
        if let ring = scene.ring {
            ring.drain(into: &batch)
            model?.apply(batch)
        }
        if let server = snapshot.server {
            if server.isIdle {
                // The ring was drained above, so the request's last tokens are
                // in the model; only now is it safe to forget them.
                if !settled {
                    model?.clearLive()
                    settled = true
                }
                idleFrames &+= 1
                if idleFrames % Self.framesPerIdleStep == 0 { model?.idleStep() }
            } else {
                settled = false
            }
        }
        frame += 1
        lastWasServer = snapshot.server != nil
        return LiveTraceFrame.compose(
            LiveTraceFrameInput(
                shape: scene.shape, model: model, feed: snapshot.feed, log: log.recent(),
                plan: scene.plan, spinner: frame / 2, depth: depth, server: snapshot.server,
                extraLogRows: scene.terminalRows.map { rows in
                    // One row stays free below the view, as the plan keeps it.
                    max(0, rows - 1 - (scene.plan.height ?? rows))
                } ?? 0))
    }

    func drawFrame() {
        let lines = renderFrame()
        // A server can sit idle for days; repeating an identical frame 20 times a
        // second would be 100 KB/s of nothing down a tty or an ssh session.
        if lastWasServer, lines == lastDrawn { return }
        // The log grew or shrank to fit its newest line.
        if !lines.isEmpty, lines.count != region.height { region.relayout(to: lines.count) }
        lastDrawn = lines
        region.draw(lines)
    }
}
