import Foundation
import Testing

@testable import Emone
@testable import EmoneServerCore

/// A scorer that records loads and calls, and can hold a call open so the
/// gate and the in-flight guard can be observed.
/// unchecked-invariant: every member is read and written under `lock`.
private final class StubScorer: DecisionScoring, @unchecked Sendable {
    private let lock = NSLock()
    private var _active = 0
    private var _maxActive = 0
    private var _calls = 0
    private var _held: CheckedContinuation<Void, Never>?
    private var _holdNext = false

    var maxActive: Int { lock.withLock { _maxActive } }
    var calls: Int { lock.withLock { _calls } }
    var isHolding: Bool { lock.withLock { _held != nil } }
    func holdNext() { lock.withLock { _holdNext = true } }
    func release() { lock.withLock { _held }?.resume() }

    func score(state: [Int32], questions: [DecisionModel.Question]) async throws -> (
        answers: [DecisionModel.Answer], timing: DecisionModel.Timing
    ) {
        let hold = lock.withLock {
            _active += 1
            _calls += 1
            _maxActive = max(_maxActive, _active)
            defer { _holdNext = false }
            return _holdNext
        }
        if hold {
            await withCheckedContinuation { c in lock.withLock { _held = c } }
            lock.withLock { _held = nil }
        } else {
            await Task.yield()
        }
        lock.withLock { _active -= 1 }
        let answer = DecisionModel.Answer(logits: [0, 0], probabilities: [0.5, 0.5])
        return (questions.map { _ in answer }, DecisionModel.Timing())
    }
}

/// unchecked-invariant: every member is read and written under `lock`.
private final class Loads: @unchecked Sendable {
    private let lock = NSLock()
    private var _count = 0
    private var _failNext = false
    var count: Int { lock.withLock { _count } }
    func failNext() { lock.withLock { _failNext = true } }
    func record() throws {
        try lock.withLock {
            _count += 1
            if _failNext {
                _failNext = false
                throw DecisionModel.LoadError.notADecisionModel("stub failure")
            }
        }
    }
}

private func host(idle: Duration?, scorer: StubScorer, loads: Loads) -> DecisionModelHost {
    DecisionModelHost(
        idleTimeout: idle, makeContext: { nil },
        loader: { _ in
            try loads.record()
            return scorer
        })
}

private let question = DecisionModel.Question(instruction: [1], options: [[2], [3]])

@Suite("Decision model host")
struct DecisionModelHostTests {
    @Test func loadsLazilyAndOnceForConcurrentRequests() async throws {
        let scorer = StubScorer()
        let loads = Loads()
        let h = host(idle: nil, scorer: scorer, loads: loads)
        #expect(await h.isLoaded == false)
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<8 {
                group.addTask { _ = try await h.score(state: [9], questions: [question]) }
            }
            try await group.waitForAll()
        }
        #expect(loads.count == 1)
        #expect(scorer.calls == 8)
        #expect(await h.isLoaded)
    }

    /// The model owns one sequence: two requests must never be inside it at
    /// once, even though the actor would let them interleave at an await.
    @Test func requestsAreSerialised() async throws {
        let scorer = StubScorer()
        let h = host(idle: nil, scorer: scorer, loads: Loads())
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<16 {
                group.addTask { _ = try await h.score(state: [9], questions: [question]) }
            }
            try await group.waitForAll()
        }
        #expect(scorer.maxActive == 1)
    }

    @Test func unloadsAfterTheIdleWindowAndReloadsOnDemand() async throws {
        let scorer = StubScorer()
        let loads = Loads()
        let h = host(idle: .seconds(60), scorer: scorer, loads: loads)
        _ = try await h.score(state: [9], questions: [question])
        #expect(await h.hasReaper)
        #expect(await h.reaperStep() != .stop)
        await h.backdateActivity(by: .seconds(61))
        #expect(await h.reaperStep() == .stop)
        await h.unloadIfIdle()
        #expect(await h.isLoaded == false)
        _ = try await h.score(state: [9], questions: [question])
        #expect(loads.count == 2)
    }

    @Test func aRequestInFlightBlocksTheUnload() async throws {
        let scorer = StubScorer()
        let h = host(idle: .seconds(60), scorer: scorer, loads: Loads())
        _ = try await h.score(state: [9], questions: [question])
        scorer.holdNext()
        let pending = Task { try await h.score(state: [9], questions: [question]) }
        while !scorer.isHolding { await Task.yield() }
        await h.backdateActivity(by: .seconds(61))
        #expect(await h.reaperStep() == .sleep(.seconds(60)))
        await h.unloadIfIdle()
        #expect(await h.isLoaded)
        scorer.release()
        _ = try await pending.value
    }

    @Test func aFailedLoadIsRetriedByTheNextRequest() async throws {
        let scorer = StubScorer()
        let loads = Loads()
        loads.failNext()
        let h = host(idle: nil, scorer: scorer, loads: loads)
        await #expect(throws: DecisionModel.LoadError.self) {
            _ = try await h.score(state: [9], questions: [question])
        }
        #expect(await h.isLoaded == false)
        _ = try await h.score(state: [9], questions: [question])
        #expect(loads.count == 2)
        #expect(await h.isLoaded)
    }

    @Test func withoutATimeoutTheModelStays() async throws {
        let h = host(idle: nil, scorer: StubScorer(), loads: Loads())
        _ = try await h.score(state: [9], questions: [question])
        #expect(await h.hasReaper == false)
        #expect(await h.reaperStep() == .stop)
        await h.unloadIfIdle()
        // `.stop` with no timeout means "nothing to schedule", never "unload".
        #expect(await h.isLoaded)
    }
}
