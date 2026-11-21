import Foundation
import Emone

/// What the host needs from a loaded decision model. `DecisionModel` is the
/// real one; tests inject a stub so residency can be checked without a model.
public protocol DecisionScoring: AnyObject, Sendable {
    func score(state: [Int32], questions: [DecisionModel.Question]) async throws -> (
        answers: [DecisionModel.Answer], timing: DecisionModel.Timing
    )
}

extension DecisionModel: DecisionScoring {}

/// Keeps the System One decision model (S1) resident only while it is used:
/// loaded on the first request, released after `idleTimeout` without one --
/// the same policy `ManagedModelBackend` applies to the primary model (S2),
/// and for the same reason: an idle model's weights and KV are memory the
/// primary's expert cache could have.
///
/// Requests are serialised. A decision model owns one sequence (its resident
/// state and branch point), and actor re-entrancy alone would let a second
/// request start a prefill while the first is suspended inside its own, so a
/// FIFO gate admits one scorer call at a time.
///
/// The `MetalContext` is built once and kept across unloads, as the primary's
/// is: a command queue has no deinit-safe teardown, and the shader library
/// would otherwise be recompiled on every reload. The unload releases the
/// model, its runner and the cached state; those are the memory that matters.
public actor DecisionModelHost {
    public typealias Loader = @Sendable (MetalContext?) async throws -> any DecisionScoring

    private let loader: Loader
    private let makeContext: @Sendable () throws -> MetalContext?
    private let idleTimeout: Duration?
    private var context: MetalContext?
    private var slot: IdleResident<any DecisionScoring>?
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    /// - Parameters:
    ///   - directory: the decision model install (has `decision_head.safetensors`).
    ///   - idleTimeout: release after this long without a request; nil keeps
    ///     the model resident once loaded.
    public init(directory: URL, idleTimeout: Duration?) {
        self.init(
            idleTimeout: idleTimeout,
            makeContext: { try MetalContext() },
            loader: { context in
                guard let context else {
                    throw DecisionModel.LoadError.notADecisionModel("no Metal")
                }
                return try DecisionModel.load(directory: directory, context: context)
            })
    }

    /// A runtime-LoRA decision adapter laid over `base` (Emone-lmw.11):
    /// the base's weights are the side chat model's own while it is loaded
    /// (`ModelWeightsRegistry`), so S1 costs its adapter, KV and state.
    public init(adapter: URL, base: URL, idleTimeout: Duration?) {
        self.init(
            idleTimeout: idleTimeout,
            makeContext: { try MetalContext() },
            loader: { context in
                guard let context else {
                    throw DecisionModel.LoadError.notADecisionModel("no Metal")
                }
                return try DecisionModel.load(adapter: adapter, base: base, context: context)
            })
    }

    init(
        idleTimeout: Duration?,
        makeContext: @escaping @Sendable () throws -> MetalContext?,
        loader: @escaping Loader
    ) {
        self.idleTimeout = idleTimeout
        self.makeContext = makeContext
        self.loader = loader
    }

    /// Scores `questions` against `state`, loading the model first if needed.
    public func score(state: [Int32], questions: [DecisionModel.Question]) async throws -> (
        answers: [DecisionModel.Answer], timing: DecisionModel.Timing
    ) {
        let slot = try residency()
        await enterGate()
        defer { leaveGate() }
        return try await slot.use { model in
            try await model.score(state: state, questions: questions)
        }
    }

    /// The residency slot, made on first use with the shared Metal context.
    private func residency() throws -> IdleResident<any DecisionScoring> {
        if let slot { return slot }
        if context == nil { context = try makeContext() }
        let loader = self.loader
        let context = self.context
        let made = IdleResident<any DecisionScoring>(
            name: "decision model", idleTimeout: idleTimeout,
            loader: { try await loader(context) })
        slot = made
        return made
    }

    private func enterGate() async {
        guard busy else {
            busy = true
            return
        }
        await withCheckedContinuation { waiters.append($0) }
    }

    /// Hands the gate straight to the next waiter, so it stays held and no
    /// newcomer can slip in between.
    private func leaveGate() {
        if waiters.isEmpty {
            busy = false
        } else {
            waiters.removeFirst().resume()
        }
    }

    public func shutdown() async {
        await slot?.shutdown()
    }

    // MARK: - Test hooks

    var isLoaded: Bool { get async { await slot?.isLoaded ?? false } }
    var hasReaper: Bool { get async { await slot?.hasReaper ?? false } }
    func reaperStep() async -> IdleResident<any DecisionScoring>.ReaperStep {
        await slot?.reaperStep() ?? .stop
    }
    func backdateActivity(by duration: Duration) async {
        await slot?.backdateActivity(by: duration)
    }
    func unloadIfIdle() async { await slot?.unloadIfIdle() }
}
