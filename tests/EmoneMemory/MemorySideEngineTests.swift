import Foundation
import Testing

@testable import EmoneMemory

/// The side-engine port, and the one judgement wired to a write so far.
///
/// The engine itself needs a model and lives in the runtime target; what is
/// here is the contract the memory path depends on, driven by a stub: a
/// `true` verdict stops a near-duplicate, a `nil` verdict changes nothing, and
/// no engine at all is the control.
@Suite struct MemorySideEngineTests {
    private func configuration(workspace: String = "repo-a") -> MemoryConfiguration {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = workspace
        configuration.user = "local"
        return configuration
    }

    private func fact(_ key: String, _ value: String) throws -> MemoryRecord {
        MemoryRecord(key: try MemoryKey(validating: key), value: value)
    }

    @Test func aFactUnderANewKeyThatAnExistingOneAlreadySaysIsNotStored() async throws {
        let store = InMemoryStore()
        let engine = StubSideEngine(duplicates: true)
        let log = LogCollector()
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine,
            log: { log.append($0) })
        let context = try #require(await service.beginSession(id: "s-dup"))
        try await store.set(try fact("characters/marcus/eyes", "grey"), in: context.scope)

        let written = await service.storeConsolidation(
            [try fact("characters/marcus/eye_colour", "grey")], in: context)

        #expect(written == 0)
        #expect(engine.pairs.count == 1)
        // Stored first, incoming second: the order the prompts were measured
        // in, and the one the 4B answers YES to.
        #expect(engine.pairs.first?.0.key == "characters/marcus/eyes")
        #expect(engine.pairs.first?.1.key == "characters/marcus/eye_colour")
        let stored = try await store.get(
            try MemoryKey(validating: "characters/marcus/eye_colour"),
            in: context.scope)
        #expect(stored == nil)
        #expect(log.messages().contains { $0.contains("near-duplicate stopped") })
        #expect(log.messages().contains { $0.contains("stopped 1 near-duplicate") })
    }

    @Test func withoutAnEngineTheSameFactIsStored() async throws {
        let store = InMemoryStore()
        let service = MemoryService(configuration: configuration(), durableStore: store)
        let context = try #require(await service.beginSession(id: "s-plain"))
        try await store.set(try fact("characters/marcus/eyes", "grey"), in: context.scope)

        let written = await service.storeConsolidation(
            [try fact("characters/marcus/eye_colour", "grey")], in: context)

        #expect(written == 1)
        let stored = try await store.get(
            try MemoryKey(validating: "characters/marcus/eye_colour"),
            in: context.scope)
        #expect(stored?.value == "grey")
    }

    @Test func aNoDecisionFromTheEngineStoresTheFact() async throws {
        let store = InMemoryStore()
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: StubSideEngine(duplicates: nil))
        let context = try #require(await service.beginSession(id: "s-nil"))
        try await store.set(try fact("characters/marcus/eyes", "grey"), in: context.scope)

        let written = await service.storeConsolidation(
            [try fact("characters/marcus/eye_colour", "grey")], in: context)

        #expect(written == 1)
    }

    @Test func onlyFactsInTheSameLeadingSegmentAreCompared() async throws {
        let store = InMemoryStore()
        let engine = StubSideEngine(duplicates: true)
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine)
        let context = try #require(await service.beginSession(id: "s-category"))
        try await store.set(try fact("gotchas/other", "grey"), in: context.scope)

        let written = await service.storeConsolidation(
            [try fact("characters/marcus/eyes", "grey")], in: context)

        #expect(written == 1)
        #expect(engine.pairs.isEmpty)
    }

    @Test func theSameKeyIsNeverComparedWithItself() async throws {
        let store = InMemoryStore()
        let engine = StubSideEngine(duplicates: true)
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine)
        let context = try #require(await service.beginSession(id: "s-same-key"))
        // Same key, different value: the unchanged check does not fire, and
        // the duplication check must not ask the engine about a key against
        // itself.
        try await store.set(try fact("characters/marcus/eyes", "grey"), in: context.scope)

        let written = await service.storeConsolidation(
            [try fact("characters/marcus/eyes", "hazel")], in: context)

        #expect(written == 1)
        #expect(engine.pairs.isEmpty)
    }

    @Test func aContradictionIsRecordedAndTheWriteStillStands() async throws {
        let store = InMemoryStore()
        let engine = StubSideEngine(duplicates: false, contradicts: true)
        let log = LogCollector()
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine,
            log: { log.append($0) })
        let context = try #require(await service.beginSession(id: "s-conflict"))
        try await store.set(try fact("characters/marcus/eyes", "grey"), in: context.scope)

        let written = await service.storeConsolidation(
            [try fact("characters/marcus/eye_colour", "hazel")], in: context)

        // Advisory only: disagreement is not supersession, and T4, which would
        // tell them apart, is not ready. The fact is stored and the conflict
        // is logged.
        #expect(written == 1)
        let stored = try await store.get(
            try MemoryKey(validating: "characters/marcus/eye_colour"),
            in: context.scope)
        #expect(stored?.value == "hazel")
        #expect(log.messages().contains { $0.contains("possible conflict") })
        #expect(log.messages().contains { $0.contains("recorded 1 possible conflict") })
    }

    @Test func aRuleIsFoundByTheAttributesLastSegment() throws {
        let eyes = try MemoryKey(validating: "characters/marcus/eyes")
        #expect(MemoryRuleLookup.ruleKey(for: eyes)?.rawValue == "rules/eyes")
        let facts = [try fact("rules/eyes", "eye colour is fixed and must never change.")]
        #expect(
            MemoryRuleLookup.rule(for: eyes, among: facts)
                == "eye colour is fixed and must never change.")
    }

    @Test func aSingleSegmentKeyHasNoRule() throws {
        let key = try MemoryKey(validating: "decisions")
        #expect(MemoryRuleLookup.ruleKey(for: key) == nil)
        #expect(MemoryRuleLookup.rule(for: key, among: [try fact("rules/decisions", "x")]) == nil)
    }

    @Test func anotherAttributesRuleDoesNotMatch() throws {
        let eyes = try MemoryKey(validating: "characters/marcus/eyes")
        let facts = [try fact("rules/eyes_colour", "fixed")]
        #expect(MemoryRuleLookup.rule(for: eyes, among: facts) == nil)
    }

    @Test func aStoredRuleStopsAChangeItFixes() async throws {
        let store = InMemoryStore()
        let engine = StubSideEngine(
            duplicates: false, contradicts: false,
            supersedes: .conflict)
        let log = LogCollector()
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine,
            log: { log.append($0) })
        let context = try #require(await service.beginSession(id: "s-rule"))
        try await store.set(try fact("characters/marcus/eyes", "grey"), in: context.scope)
        try await store.set(
            try fact("rules/eyes", "eye colour is fixed and must never change."),
            in: context.scope)

        let written = await service.storeConsolidation(
            [try fact("characters/marcus/eyes", "hazel")], in: context)

        #expect(written == 0)
        let stored = try await store.get(
            try MemoryKey(validating: "characters/marcus/eyes"),
            in: context.scope)
        #expect(stored?.value == "grey")
        #expect(engine.supersessionQuestions == 1)
        #expect(log.messages().contains { $0.contains("rule conflict") })
        #expect(log.messages().contains { $0.contains("stopped 1 change") })
    }

    @Test func withoutARuleAChangedValueIsStillStored() async throws {
        let store = InMemoryStore()
        // `.conflict` is what the engine would say, but it is never asked: no
        // rule is filed for this attribute.
        let engine = StubSideEngine(
            duplicates: false, contradicts: false,
            supersedes: .conflict)
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine)
        let context = try #require(await service.beginSession(id: "s-no-rule"))
        try await store.set(try fact("characters/marcus/eyes", "grey"), in: context.scope)

        let written = await service.storeConsolidation(
            [try fact("characters/marcus/eyes", "hazel")], in: context)

        #expect(written == 1)
        #expect(engine.supersessionQuestions == 0)
    }

    @Test func anUpdateUnderARuleIsStored() async throws {
        let store = InMemoryStore()
        let engine = StubSideEngine(
            duplicates: false, contradicts: false,
            supersedes: .update)
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine)
        let context = try #require(await service.beginSession(id: "s-update"))
        try await store.set(try fact("state/inn", "standing"), in: context.scope)
        try await store.set(try fact("rules/inn", "the inn may burn."), in: context.scope)

        let written = await service.storeConsolidation(
            [try fact("state/inn", "burned to the ground")], in: context)

        #expect(written == 1)
        #expect(engine.supersessionQuestions == 1)
    }

    @Test func aPersonsOwnChangeIsNotHeldByARule() async throws {
        let store = InMemoryStore()
        let engine = StubSideEngine(
            duplicates: false, contradicts: false,
            supersedes: .conflict)
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine)
        let context = try #require(await service.beginSession(id: "s-person-rule"))
        try await store.set(try fact("characters/marcus/eyes", "grey"), in: context.scope)
        try await store.set(
            try fact("rules/eyes", "eye colour is fixed and must never change."),
            in: context.scope)

        var asserted = try fact("characters/marcus/eyes", "hazel")
        asserted.isUserAsserted = true
        let written = await service.storeConsolidation([asserted], in: context)

        #expect(written == 1)
        #expect(engine.supersessionQuestions == 0)
    }

    @Test func aGlobalFactIsCheckedAgainstTheSharedWorkspace() async throws {
        let store = InMemoryStore()
        let engine = StubSideEngine(duplicates: true)
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine)
        let context = try #require(await service.beginSession(id: "s-global"))
        let shared = try #require(await service.configuration.sharedScope)
        try await store.set(try fact("language/replies", "in German"), in: shared)

        var global = try fact("language/response_language", "German")
        global.isGlobal = true
        let written = await service.storeConsolidation([global], in: context)

        #expect(written == 0)
        #expect(engine.pairs.first?.0.key == "language/replies")
        #expect(engine.pairs.first?.1.key == "language/response_language")
    }

    @Test func aFactNotWorthKeepingIsNotStored() async throws {
        let store = InMemoryStore()
        let engine = StubSideEngine(duplicates: false, contradicts: false, durable: false)
        let log = LogCollector()
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine,
            log: { log.append($0) })
        let context = try #require(await service.beginSession(id: "s-durable"))

        let written = await service.storeConsolidation(
            [try fact("chapters/note", "Chapter 12: Ines turned the pages.")], in: context)

        #expect(written == 0)
        let stored = try await store.get(
            try MemoryKey(validating: "chapters/note"),
            in: context.scope)
        #expect(stored == nil)
        #expect(log.messages().contains { $0.contains("not worth keeping") })
        #expect(log.messages().contains { $0.contains("dropped 1 fact") })
        // The check ends at the durability answer, so no candidate question
        // was asked.
        #expect(engine.questions == 1)
        #expect(engine.pairs.isEmpty)
    }

    @Test func aUsersOwnStatementIsNeverDropped() async throws {
        let store = InMemoryStore()
        let engine = StubSideEngine(duplicates: false, contradicts: false, durable: false)
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine)
        let context = try #require(await service.beginSession(id: "s-asserted"))

        var asserted = try fact("decisions/sync", "Keep the queue single-threaded.")
        asserted.isUserAsserted = true
        let written = await service.storeConsolidation([asserted], in: context)

        // Durability is not the engine's question about the person's words, so
        // it is not even asked.
        #expect(written == 1)
        #expect(engine.durabilityQuestions == 0)
    }

    @Test func theQuestionBudgetSpansTheWholeConsolidation() async throws {
        let store = InMemoryStore()
        let engine = StubSideEngine(duplicates: false, contradicts: false)
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine)
        let context = try #require(await service.beginSession(id: "s-budget"))
        for index in 0..<20 {
            try await store.set(
                try fact("characters/p\(index)", "value \(index)"),
                in: context.scope)
        }
        let records = try (0..<5).map { try fact("characters/new_\($0)", "other \($0)") }

        let written = await service.storeConsolidation(records, in: context)

        // Five facts and twenty candidates: the cost is capped by the budget
        // for the consolidation, not by the candidate count.
        #expect(written == 5)
        #expect(engine.questions <= MemoryService.maximumSideEngineQuestions)
    }
}

/// Answers every question the same way, and counts what it was asked.
///
/// unchecked-invariant: the counters are only ever touched under `lock`.
private final class StubSideEngine: MemorySideEngine, @unchecked Sendable {
    private let lock = NSLock()
    private let duplicateAnswer: Bool?
    private let contradictionAnswer: Bool?
    private let durabilityAnswer: Bool?
    private let supersessionAnswer: MemorySupersession?
    private var asked: [(MemoryFact, MemoryFact)] = []
    private var contradictionAsked = 0
    private var durabilityAsked = 0
    private var supersessionAsked = 0

    init(
        duplicates duplicateAnswer: Bool?,
        contradicts contradictionAnswer: Bool? = nil,
        durable durabilityAnswer: Bool? = nil,
        supersedes supersessionAnswer: MemorySupersession? = nil
    ) {
        self.duplicateAnswer = duplicateAnswer
        self.contradictionAnswer = contradictionAnswer
        self.durabilityAnswer = durabilityAnswer
        self.supersessionAnswer = supersessionAnswer
    }

    var pairs: [(MemoryFact, MemoryFact)] { lock.withLock { asked } }
    var durabilityQuestions: Int { lock.withLock { durabilityAsked } }
    var supersessionQuestions: Int { lock.withLock { supersessionAsked } }
    /// Every question of any kind: the budget counts them all.
    var questions: Int {
        lock.withLock { asked.count + contradictionAsked + durabilityAsked + supersessionAsked }
    }

    func isDurable(_ fact: MemoryFact) async -> Bool? {
        lock.withLock { durabilityAsked += 1 }
        return durabilityAnswer
    }

    /// Mirrors the adapter: no rule, no answer.
    func supersedes(
        _ stored: MemoryFact, _ new: MemoryFact,
        rule: String?
    ) async -> MemorySupersession? {
        lock.withLock { supersessionAsked += 1 }
        return rule == nil ? nil : supersessionAnswer
    }

    func duplicates(_ stored: MemoryFact, _ new: MemoryFact) async -> Bool? {
        lock.withLock { asked.append((stored, new)) }
        return duplicateAnswer
    }

    func contradicts(_ stored: MemoryFact, _ new: MemoryFact) async -> Bool? {
        lock.withLock { contradictionAsked += 1 }
        return contradictionAnswer
    }

    /// T7 is not wired to a write, and these tests never search. The
    /// background caller has its own tests in `MemoryRetrievalTests`.
    func couldAnswer(_ question: String, _ fact: MemoryFact) async -> Bool? { nil }
}

/// Collects the service's log events.
///
/// unchecked-invariant: `events` is only ever touched under `lock`.
private final class LogCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [MemoryLogEvent] = []

    func append(_ event: MemoryLogEvent) {
        lock.withLock { events.append(event) }
    }

    func messages() -> [String] {
        lock.withLock { events.map(\.message) }
    }
}
