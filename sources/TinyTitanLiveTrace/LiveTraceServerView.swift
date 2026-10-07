import Dispatch
import Foundation
import Synchronization
import TinyTitan

/// What a server tells the view about a model it has just made resident.
public struct LiveTraceModelInfo: Sendable, Equatable {
    /// The id clients know it by.
    public var name: String
    /// "gpu" or "cpu".
    public var engine: String
    /// The runner's shape; nil for an engine with no runner to trace (the CPU
    /// engine), which gets the compact view.
    public var shape: ExpertTraceShape?
    /// Routed-expert cache slots per layer, for the view's cached-expert shading.
    public var slotsPerLayer: Int?
    /// Set when the model is routed but its decode cannot feed the grid, with
    /// the reason: batched serving and MTP decode do not go through the traced
    /// path. The view then shows the compact layout and says why.
    public var gridUnavailable: String?

    public init(
        name: String, engine: String, shape: ExpertTraceShape?, slotsPerLayer: Int?,
        gridUnavailable: String? = nil
    ) {
        self.name = name
        self.engine = engine
        self.shape = shape
        self.slotsPerLayer = slotsPerLayer
        self.gridUnavailable = gridUnavailable
    }

    /// The shape of "no model": nothing routed, so the plan is the compact one.
    static func placeholderShape(name: String) -> ExpertTraceShape {
        ExpertTraceShape(
            modelName: name, routedExpertBits: nil, layers: 0, numExperts: 0, topK: 0,
            layerKinds: [], expertBytes: 0)
    }
}

/// One generation, as the view follows it: the decode task feeds prefill and
/// token progress in, and says how it ended.
///
/// Every method is cheap and none waits: progress goes into atomics and a
/// mutex that is only ever *tried*, as in the CLI's feed.
///
/// unchecked-invariant: `summary` is written by the one task driving the
/// generation before `close`, and read by `close`, on that task; `closed` and
/// the feed are atomic or internally synchronised.
public final class LiveTraceGeneration: @unchecked Sendable {
    struct Summary {
        var promptTokens = 0
        var cachedTokens = 0
        var newTokens = 0
    }

    private let view: LiveTraceServerView?
    let client: String?
    let feed = LiveTraceFeed(maxNew: 0, retainAll: false)
    private var summary: Summary?
    private let closed = Atomic<Bool>(false)

    fileprivate init(view: LiveTraceServerView?, client: String?) {
        self.view = view
        self.client = client
    }

    /// Route the generation's progress into the view.
    public func handle(_ progress: RawDecodeProgress) {
        switch progress {
        case .prefill(let done, let total): feed.notePrefill(done: done, total: total)
        case .token(_, _, let delta): feed.noteToken(delta)
        case .tail(let tail): feed.noteTail(tail)
        }
    }

    /// The generation finished normally; these are what the idle frame reports.
    public func complete(promptTokens: Int, cachedTokens: Int, newTokens: Int) {
        summary = Summary(
            promptTokens: promptTokens, cachedTokens: cachedTokens, newTokens: newTokens)
    }

    /// Leave the view: normally, after an error, after a cancel. Idempotent.
    public func close() {
        guard closed.compareExchange(
            expected: false, desired: true, ordering: .acquiringAndReleasing
        ).exchanged
        else { return }
        feed.finish()
        view?.generationEnded(self, summary: summary)
    }

    var rate: Double? { feed.averageTokensPerSecond }
}

/// The server's live view: what a `TinyTitanServer --live-trace` draws in its
/// own terminal while clients talk to it.
///
/// It is told things by the server, through methods that only take a short lock
/// and never touch the terminal; it draws on its own thread, from a snapshot it
/// reads once a frame, and drops frames whenever the terminal is slow. Nothing it
/// computes (heat, the believed-resident set) is read by the engine.
///
/// One model is resident at a time and its shape can change at any switch: the
/// server reports each load and unload, the view re-lays itself out for the new
/// shape, and asks whichever runner serves the next request to record into a
/// fresh ring (`attach`).
///
/// Two stages. `prepare` gates and builds the view and nothing else, so a server
/// can record what loads at startup. `open` redirects stdout and stderr into the
/// view's log tail and starts drawing; `close` puts everything back.
///
/// unchecked-invariant: `state` and `resources` are `Mutex`es and everything else
/// is immutable after `prepare`.
public final class LiveTraceServerView: @unchecked Sendable {
    public struct Options: Sendable {
        /// Where the full log goes while the view owns the terminal; nil takes
        /// `LiveTraceServerView.defaultLogPath`.
        public var logPath: String?
        /// Names the model in the header when the operator gave it an id of its
        /// own (`--model-id`); nil uses the id the model reports.
        public var nameOverride: String?
        public init(logPath: String? = nil, nameOverride: String? = nil) {
            self.logPath = logPath
            self.nameOverride = nameOverride
        }
    }

    private enum Residency {
        case none
        case loading(String)
        case loaded(token: UUID, info: LiveTraceModelInfo)
    }

    private struct Request {
        var client: String?
        var generating = false
        var claimed = false
        let order: Int
    }

    private struct State {
        var residency = Residency.none
        var scene: LiveTraceScene?
        var attachedToken: UUID?
        var requests: [String: Request] = [:]
        var requestCounter = 0
        var active: [LiveTraceGeneration] = []
        var finishedFeed: LiveTraceFeed?
        var last: LiveTraceServerStatus.Last?
        var generation = 0
        var renderer: LiveTraceRenderer?
    }

    private struct Resources {
        var region: LiveTraceRegion
        var renderer: LiveTraceRenderer
        var stdout: LiveTraceStderrCapture
        var stderr: LiveTraceStderrCapture
        var logFile: LiveTraceLogFile?
    }

    let options: Options
    let probe: TerminalProbe
    let depth: LiveTraceColorDepth
    private let stdoutFD: Int32
    private let stderrFD: Int32
    private let note: @Sendable (String) -> Void
    private let state = Mutex(State())
    private let resources = Mutex<Resources?>(nil)

    private init(
        options: Options, probe: TerminalProbe, depth: LiveTraceColorDepth, stdoutFD: Int32,
        stderrFD: Int32, note: @escaping @Sendable (String) -> Void
    ) {
        self.options = options
        self.probe = probe
        self.depth = depth
        self.stdoutFD = stdoutFD
        self.stderrFD = stderrFD
        self.note = note
    }

    public static var defaultLogPath: String { LiveTraceLogFile.defaultPath() }

    /// The default for `note`: one line on the real stderr.
    @Sendable public static func noteToStderr(_ line: String) {
        liveTraceWriteAll(STDERR_FILENO, Array((line + "\n").utf8))
    }

    /// Gate the view. nil when this terminal cannot show it, after one line on
    /// `note` saying why; otherwise a view that has drawn nothing yet.
    public static func prepare(
        options: Options = Options(),
        probe: TerminalProbe? = nil,
        stdoutFD: Int32 = STDOUT_FILENO,
        stderrFD: Int32 = STDERR_FILENO,
        note: @escaping @Sendable (String) -> Void = LiveTraceServerView.noteToStderr
    ) -> LiveTraceServerView? {
        let probe = probe ?? TerminalProbe.current(stdoutFD: stdoutFD)
        switch LiveTraceGate.decideForServer(requested: true, probe: probe) {
        case .off:
            return nil
        case .disabled(let reason):
            note("note: --live-trace is off: \(reason)")
            return nil
        case .run(_, let depth):
            return LiveTraceServerView(
                options: options, probe: probe, depth: depth, stdoutFD: stdoutFD,
                stderrFD: stderrFD, note: note)
        }
    }

    // MARK: layout

    /// The scene for a model: the grid view when it is routed, fits this
    /// terminal and can be fed, the compact one otherwise. The note says why a
    /// routed model did not get the grid.
    func scene(for info: LiveTraceModelInfo) -> (scene: LiveTraceScene, note: String?) {
        let shape = info.shape ?? LiveTraceModelInfo.placeholderShape(name: info.name)
        let plan = LiveTraceViewPlan.plan(
            shape: shape, cols: probe.cols, rows: probe.rows,
            allowGrid: info.gridUnavailable == nil)
        // The gate admitted this terminal for the compact view, so the plan is
        // never unavailable; this is the same view if it somehow were.
        let resolved: LiveTraceViewPlan
        if case .unavailable = plan {
            resolved = .compact(
                width: min(probe.cols, LiveTraceViewPlan.fullWidth),
                height: LiveTraceViewPlan.compactHeight)
        } else {
            resolved = plan
        }
        var note: String?
        if case .compact = resolved, shape.isRouted {
            if let why = info.gridUnavailable {
                note = "note: --live-trace grid hidden for \(info.name): \(why)"
            } else if let full = LiveTraceViewPlan.fullHeight(shape: shape) {
                note =
                    "note: --live-trace grid hidden: the terminal is \(probe.cols)x\(probe.rows); "
                    + "the grid needs at least \(LiveTraceViewPlan.fullWidth)x\(full + 1)"
            }
        }
        return (
            LiveTraceScene(
                shape: shape, plan: resolved, ring: nil, slotsPerLayer: info.slotsPerLayer,
                terminalRows: probe.rows),
            note
        )
    }

    private var placeholderScene: LiveTraceScene {
        scene(for: LiveTraceModelInfo(name: "", engine: "", shape: nil, slotsPerLayer: nil)).scene
    }

    // MARK: the model

    /// A switch has started: the header says which model is being loaded.
    public func modelLoading(_ name: String) {
        state.withLock { $0.residency = .loading(name) }
    }

    /// The load that `modelLoading` announced failed: nothing is resident.
    public func modelLoadFailed() {
        state.withLock { state in
            if case .loading = state.residency { state.residency = .none }
        }
    }

    /// A model is resident. `token` identifies this load, so a stale unload
    /// cannot clear a newer model.
    public func modelLoaded(token: UUID, info: LiveTraceModelInfo) {
        let (next, why) = scene(for: info)
        let renderer = state.withLock { state -> LiveTraceRenderer? in
            state.residency = .loaded(token: token, info: info)
            state.attachedToken = nil
            state.scene = next
            return state.renderer
        }
        renderer?.setScene(next)
        if let why { note(why) }
    }

    /// The model with this token is gone. The layout stays until the next model
    /// replaces it, so a switch does not resize the view twice; the ring goes,
    /// because its runner has.
    public func modelUnloaded(token: UUID) {
        let update = state.withLock { state -> (LiveTraceRenderer?, LiveTraceScene?) in
            guard case .loaded(let current, _) = state.residency, current == token
            else { return (nil, nil) }
            state.residency = .none
            state.attachedToken = nil
            state.scene?.ring = nil
            return (state.renderer, state.scene)
        }
        if let scene = update.1 { update.0?.setScene(scene) }
    }

    /// Ask the runner that is about to serve a request to record its routing
    /// into the view, once per loaded model. The runner is the one the load with
    /// `token` built, so after a switch the ring follows the new runner and the
    /// old one is gone with its model. A compact view never enables the hook.
    ///
    /// Call from the task that drives the runner, before the generation starts.
    public func attach(token: UUID, runner: any ExpertTraceHosting) {
        let published = state.withLock { state -> (LiveTraceRenderer?, LiveTraceScene)? in
            guard case .loaded(let current, _) = state.residency, current == token,
                state.attachedToken != token, var scene = state.scene, case .full = scene.plan
            else { return nil }
            state.attachedToken = token
            scene.ring = runner.enableExpertTrace(capacity: ExpertTraceRing.defaultCapacity)
            state.scene = scene
            return (state.renderer, scene)
        }
        if let published { published.0?.setScene(published.1) }
    }

    // MARK: requests

    /// The server accepted a request. `client` is the model id the client sent.
    public func requestAccepted(id: String, client: String?) {
        state.withLock { state in
            state.requestCounter += 1
            state.requests[id] = Request(client: client, order: state.requestCounter)
        }
    }

    /// The request reached the front of the queue and was handed to the engine.
    public func requestGenerating(id: String) {
        state.withLock { $0.requests[id]?.generating = true }
    }

    /// The request is over, however it ended.
    public func requestClosed(id: String) {
        state.withLock { _ = $0.requests.removeValue(forKey: id) }
    }

    /// The engine starts a generation. It belongs to the oldest request that is
    /// generating and has not been claimed, which with the default one-at-a-time
    /// serving is exactly the request running; with batched serving it is the
    /// same order the requests started in. A generation no request asked for
    /// (the memory subsystem's own) is shown without a client.
    public func beginGeneration() -> LiveTraceGeneration {
        state.withLock { state in
            let claimed = state.requests
                .filter { $0.value.generating && !$0.value.claimed }
                .min { $0.value.order < $1.value.order }
            if let id = claimed?.key { state.requests[id]?.claimed = true }
            let generation = LiveTraceGeneration(
                view: self, client: claimed?.value.client ?? "(internal)")
            state.active.append(generation)
            state.generation += 1
            return generation
        }
    }

    fileprivate func generationEnded(
        _ generation: LiveTraceGeneration, summary: LiveTraceGeneration.Summary?
    ) {
        state.withLock { state in
            state.active.removeAll { $0 === generation }
            state.finishedFeed = generation.feed
            if let summary {
                state.last = LiveTraceServerStatus.Last(
                    tokensPerSecond: generation.rate, newTokens: summary.newTokens,
                    promptTokens: summary.promptTokens, cachedTokens: summary.cachedTokens)
            }
        }
    }

    // MARK: what the renderer reads

    private func title(for residency: Residency) -> String {
        switch residency {
        case .none:
            return "no model loaded"
        case .loading(let name):
            return "loading \(name)…"
        case .loaded(_, let info):
            var text = options.nameOverride ?? info.name
            if let bits = info.shape?.routedExpertBits {
                text += " · \(bits)-bit"
            } else if info.engine == "cpu" {
                text += " · cpu"
            }
            return text
        }
    }

    /// The frame's server fields and the feed to draw, from one consistent read.
    func snapshot() -> LiveTraceSnapshot {
        let (status, feed, generation) = state.withLock {
            state -> (LiveTraceServerStatus, LiveTraceFeed?, Int) in
            var status = LiveTraceServerStatus()
            status.title = title(for: state.residency)
            switch state.residency {
            case .none: status.residency = .none
            case .loading: status.residency = .loading
            case .loaded: status.residency = .loaded
            }
            let generating = state.requests.values.filter(\.generating)
            status.waiting = state.requests.count - generating.count
            status.running = max(state.active.count, generating.count)
            status.last = state.last
            let current = state.active.last
            status.client =
                current?.client
                ?? generating.filter { !$0.claimed }.min { $0.order < $1.order }?.client
            return (status, current?.feed ?? state.finishedFeed, state.generation)
        }
        return LiveTraceSnapshot(
            feed: feed?.snapshot() ?? LiveTraceFeedSnapshot(), server: status,
            generation: generation)
    }

    // MARK: opening and closing

    /// Redirect stdout and stderr into the view, open the region and start
    /// drawing. Returns false, leaving the process exactly as it was, when the
    /// redirect cannot be made.
    ///
    /// Everything the server prints from here goes two places: the view's log
    /// tail and a log file (or, for stderr, straight on to a file the operator
    /// already redirected it to). Both are written by a thread of their own, so
    /// a slow disk never blocks a request.
    @discardableResult
    public func open(startRenderThread: Bool = true) -> Bool {
        resources.withLock { resources in
            guard resources == nil else { return true }
            let logPath = options.logPath ?? Self.defaultLogPath
            // The file is needed for stdout always (it is the terminal) and for
            // stderr when that is the terminal too; a stderr the operator
            // redirected keeps going there untouched.
            let logFile = LiveTraceLogFile.open(path: logPath)
            if logFile == nil {
                note(
                    "note: --live-trace could not open \(logPath); log lines are held and "
                        + "printed when the server stops")
            }
            // Said before the redirect, so it stays in scrollback: afterwards a
            // line on stderr goes into the view's log tail instead.
            if let logFile {
                note("note: --live-trace: the full server log is in \(logFile.path)")
            }
            let log = LiveTraceLogTail()
            guard
                let stderrCapture = LiveTraceStderrCapture.start(
                    targetFD: stderrFD, log: log,
                    fileSinkFD: isatty(stderrFD) != 0 ? logFile?.fd : nil)
            else {
                note("note: --live-trace is off: could not capture stderr")
                logFile?.close()
                return false
            }
            guard
                let stdoutCapture = LiveTraceStderrCapture.start(
                    targetFD: stdoutFD, log: log, fileSinkFD: logFile?.fd)
            else {
                note("note: --live-trace is off: could not capture stdout")
                _ = stderrCapture.stop()
                logFile?.close()
                return false
            }
            let initial = state.withLock { $0.scene } ?? placeholderScene
            guard let height = initial.plan.height else { return false }
            let region = LiveTraceRegion(fd: stdoutCapture.originalFD, height: height)
            let renderer = LiveTraceRenderer(
                scene: initial, depth: depth, log: log, region: region,
                source: { [weak self] in
                    self?.snapshot() ?? LiveTraceSnapshot(feed: LiveTraceFeedSnapshot())
                })
            state.withLock { $0.renderer = renderer }
            region.open()
            if startRenderThread { renderer.start() }
            resources = Resources(
                region: region, renderer: renderer, stdout: stdoutCapture, stderr: stderrCapture,
                logFile: logFile)
            return true
        }
    }

    /// Stop drawing, leave the last frame on screen, restore stdout and stderr,
    /// and write back anything held. Idempotent; safe to call from a signal's
    /// shutdown path.
    public func close() {
        let taken = resources.withLock { resources -> Resources? in
            let taken = resources
            resources = nil
            return taken
        }
        guard let taken else { return }
        state.withLock { $0.renderer = nil }
        taken.renderer.stop()
        taken.region.close()
        let heldOut = taken.stdout.stop()
        let heldErr = taken.stderr.stop()
        for chunk in heldOut.held { liveTraceWriteAll(stdoutFD, chunk) }
        for chunk in heldErr.held { liveTraceWriteAll(stderrFD, chunk) }
        taken.logFile?.close()
        if let path = taken.logFile?.path {
            liveTraceWriteAll(stderrFD, Array("note: server log: \(path)\n".utf8))
        }
    }

    /// Put the terminal back (cursor shown, autowrap on, the cursor below the
    /// view) without closing anything else, if a view is open. For a path that
    /// is about to `exit` and cannot run `close`: a second termination signal.
    /// The same thing runs from an `atexit` handler, so any `exit` is covered;
    /// calling it first just makes the order explicit. A crash is not covered.
    public static func restoreTerminal() { LiveTraceTerminalGuard.restore() }

    /// True between `open` and `close`.
    public var isOpen: Bool { resources.withLock { $0 != nil } }

    // MARK: testing

    /// One frame as the render thread would compose it, without a thread or a
    /// terminal write. Valid between `open(startRenderThread: false)` and `close`.
    func composeFrameForTesting() -> [String] {
        let renderer = state.withLock { $0.renderer }
        return renderer?.renderFrame() ?? []
    }

    /// Compose a frame and write it, as the render thread does each tick.
    func drawFrameForTesting() {
        let renderer = state.withLock { $0.renderer }
        renderer?.drawFrame()
    }

    var currentPlanForTesting: LiveTraceViewPlan? { state.withLock { $0.scene?.plan } }
    var currentRingForTesting: ExpertTraceRing? { state.withLock { $0.scene?.ring } }
}
