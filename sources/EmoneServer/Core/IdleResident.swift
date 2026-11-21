import Foundation

/// One lazily loaded value that is released after an idle window: the
/// residency policy `ManagedModelBackend` applies to the primary model, as a
/// reusable piece for the models that sit beside it -- a side chat model
/// (session titles) and the System One decision model.
///
/// `use` loads on first need (concurrent first callers share one load), counts
/// the caller in flight while its body runs, and refreshes the idle clock when
/// it leaves. A reaper wakes once per idle window, not on a tick, and releases
/// the value only when nothing is in flight and nothing is loading. A failed
/// load leaves nothing resident, so the next caller retries.
actor IdleResident<Value: Sendable> {
    typealias Loader = @Sendable () async throws -> Value

    enum ReaperStep: Equatable {
        case sleep(Duration)
        case stop
    }

    let name: String
    private let loader: Loader
    private let idleTimeout: Duration?
    private var value: Value?
    private var loadTask: Task<Value, any Error>?
    private var inFlight = 0
    private var lastActivity = ContinuousClock.now
    private var reaper: Task<Void, Never>?

    /// - Parameters:
    ///   - name: for the residency log line.
    ///   - idleTimeout: release after this long unused; nil keeps the value
    ///     once loaded.
    init(name: String, idleTimeout: Duration?, loader: @escaping Loader) {
        self.name = name
        self.idleTimeout = idleTimeout
        self.loader = loader
    }

    /// Runs `body` with the resident value, loading it first if needed.
    func use<T: Sendable>(_ body: @Sendable (Value) async throws -> T) async throws -> T {
        inFlight += 1
        defer {
            inFlight -= 1
            lastActivity = .now
        }
        return try await body(try await resident())
    }

    /// The value when it is already loaded, without loading it or counting a
    /// caller in flight; for questions that must not force a load.
    var loadedValue: Value? { value }

    private func resident() async throws -> Value {
        if let value { return value }
        if let loadTask { return try await loadTask.value }
        let loader = self.loader
        let task = Task { try await loader() }
        loadTask = task
        defer { loadTask = nil }
        let loaded = try await task.value
        value = loaded
        lastActivity = .now
        startReaper()
        ServerLog.residency("loaded \(name)")
        return loaded
    }

    func reaperStep(now: ContinuousClock.Instant = .now) -> ReaperStep {
        guard let idleTimeout, value != nil else { return .stop }
        guard inFlight == 0, loadTask == nil else { return .sleep(idleTimeout) }
        let idleFor = now - lastActivity
        guard idleFor >= idleTimeout else { return .sleep(idleTimeout - idleFor) }
        return .stop
    }

    private func startReaper() {
        guard idleTimeout != nil, reaper == nil else { return }
        reaper = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                switch await self.reaperStep() {
                case .sleep(let duration):
                    try? await Task.sleep(for: duration)
                case .stop:
                    await self.unloadIfIdle()
                    return
                }
            }
        }
    }

    /// Re-checks under isolation: the reaper decided before its sleep.
    func unloadIfIdle() {
        reaper = nil
        // Without a timeout `.stop` only means "nothing to schedule".
        guard idleTimeout != nil, value != nil else { return }
        guard case .stop = reaperStep() else {
            startReaper()
            return
        }
        value = nil  // the last strong reference: the weights go here
        ServerLog.residency("unloaded \(name)")
    }

    /// Releases the value now if nothing is using it; true when it did.
    func unloadNow() -> Bool {
        guard value != nil, inFlight == 0, loadTask == nil else { return false }
        value = nil
        reaper?.cancel()
        reaper = nil
        ServerLog.residency("unloaded \(name)")
        return true
    }

    func shutdown() {
        reaper?.cancel()
        reaper = nil
        value = nil
    }

    // MARK: - Test hooks

    var isLoaded: Bool { value != nil }
    var hasReaper: Bool { reaper != nil }

    func backdateActivity(by duration: Duration) {
        lastActivity = .now - duration
    }
}
