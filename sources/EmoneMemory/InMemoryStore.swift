import Foundation

/// A `MemoryStore` held in process.
///
/// It is the reference implementation: every rule the Valkey backend has to
/// honour (scope isolation, value limits, bootstrap bounds, ranking) is
/// implemented here once, in plain Swift, and the conformance tests run
/// against both. It is also what the server falls back to when Valkey is
/// configured but unreachable and `degradeToLocal` is set, and what the
/// tests use so the suite needs no server.
public actor InMemoryStore: MemoryStore {
    private var records: [MemoryScope: [MemoryKey: MemoryRecord]] = [:]
    private var sessions: [MemoryScope: [MemorySession]] = [:]
    private let limits: MemoryLimits

    public init(limits: MemoryLimits = .init()) {
        self.limits = limits
    }

    /// Every record in a scope, for tests and for the consolidation hook.
    public func allRecords(in scope: MemoryScope) -> [MemoryRecord] {
        Array(records[scope]?.values ?? [:].values)
    }

    public func get(_ key: MemoryKey, in scope: MemoryScope) async throws -> MemoryRecord? {
        records[scope]?[key]
    }

    public func set(_ record: MemoryRecord, in scope: MemoryScope) async throws {
        try limits.validate(value: record.value)
        var scoped = records[scope] ?? [:]
        // A rewrite keeps the original creation time: the model is updating a
        // fact, not making a new one, and "known since" is the useful date.
        var stored = record
        if let existing = scoped[record.key] {
            stored.createdAt = existing.createdAt
        }
        stored.updatedAt = Date()
        scoped[record.key] = stored
        records[scope] = scoped
    }

    @discardableResult
    public func delete(_ key: MemoryKey, in scope: MemoryScope) async throws -> Bool {
        guard records[scope]?[key] != nil else { return false }
        records[scope]?[key] = nil
        return true
    }

    public func exists(_ key: MemoryKey, in scope: MemoryScope) async throws -> Bool {
        records[scope]?[key] != nil
    }

    public func list(prefix: String, limit: Int, in scope: MemoryScope) async throws -> [MemoryKey]
    {
        let scoped = records[scope] ?? [:]
        return scoped.values
            .filter { prefix.isEmpty || $0.key.rawValue.hasPrefix(prefix) }
            .sorted { $0.updatedAt > $1.updatedAt }
            .prefix(max(0, min(limit, limits.maximumListResults)))
            .map(\.key)
    }

    public func search(_ query: MemoryQuery, in scope: MemoryScope) async throws -> [MemoryRecord] {
        let scoped = records[scope] ?? [:]
        let ranked = MemoryRanking.rank(Array(scoped.values), for: query)
        return Array(ranked.prefix(max(0, min(query.limit, limits.maximumSearchResults))))
    }

    @discardableResult
    public func append(_ text: String, to key: MemoryKey, in scope: MemoryScope) async throws
        -> MemoryRecord
    {
        let existing = records[scope]?[key]
        let combined = existing.map { $0.value.isEmpty ? text : $0.value + "\n" + text } ?? text
        try limits.validate(value: combined)
        var record = existing ?? MemoryRecord(key: key, value: "")
        record.value = combined
        record.updatedAt = Date()
        var scoped = records[scope] ?? [:]
        scoped[key] = record
        records[scope] = scoped
        return record
    }

    public func sessionInit(_ session: MemorySession, in scope: MemoryScope) async throws
        -> MemoryBootstrap
    {
        sessions[scope, default: []].append(session)
        let scoped = Array((records[scope] ?? [:]).values)
        return MemoryBootstrap.build(from: scoped, limits: limits)
    }

    /// Sessions recorded in a scope, for tests.
    public func recordedSessions(in scope: MemoryScope) -> [MemorySession] {
        sessions[scope] ?? []
    }
}

/// Size and count bounds shared by every backend.
///
/// These exist because the model chooses what to write. Without a ceiling on
/// value size a single tool call can put a source file in the store, and
/// without one on result counts a search can return the store.
public struct MemoryLimits: Sendable, Equatable {
    public var maximumValueBytes: Int
    public var maximumSearchResults: Int
    public var maximumListResults: Int
    public var bootstrapRecords: Int
    public var bootstrapBytes: Int

    public init(
        maximumValueBytes: Int = 64 * 1024,
        maximumSearchResults: Int = 50,
        maximumListResults: Int = 200,
        bootstrapRecords: Int = 60,
        bootstrapBytes: Int = 16 * 1024
    ) {
        self.maximumValueBytes = maximumValueBytes
        self.maximumSearchResults = maximumSearchResults
        self.maximumListResults = maximumListResults
        self.bootstrapRecords = bootstrapRecords
        self.bootstrapBytes = bootstrapBytes
    }

    public func validate(value: String) throws {
        let bytes = value.utf8.count
        guard bytes <= maximumValueBytes else {
            throw MemoryError.valueTooLarge(bytes: bytes, limit: maximumValueBytes)
        }
    }
}

extension MemoryBootstrap {
    /// The bounded bootstrap set: most important first, then most recent,
    /// cut by whichever limit binds first.
    ///
    /// Both limits are enforced here rather than at the call site so every
    /// backend gets the same ceiling; the count alone is not enough, because
    /// twenty records of 64 KB would still be 1.2 MB of context.
    /// Applies the count and byte caps to records already in the order they
    /// should appear. For a ranking done elsewhere -- by relevance to the
    /// request -- that must not be re-sorted here.
    static func build(
        ordered records: [MemoryRecord], limits: MemoryLimits,
        recent: [MemoryRecord] = []
    ) -> MemoryBootstrap {
        var chosen: [MemoryRecord] = []
        var bytes = 0
        for record in records {
            guard chosen.count < limits.bootstrapRecords else { break }
            let size = record.key.rawValue.utf8.count + record.value.utf8.count
            guard bytes + size <= limits.bootstrapBytes else { continue }
            chosen.append(record)
            bytes += size
        }
        return MemoryBootstrap(
            records: chosen,
            omittedCount: records.count - chosen.count,
            totalBytes: bytes,
            recent: recent)
    }

    static func build(from records: [MemoryRecord], limits: MemoryLimits) -> MemoryBootstrap {
        // Importance first; among equals the OLDER fact wins. A bible written
        // in session one and the state of session nine compete for the same
        // slots, and the foundation is the one a session cannot do without.
        // Newest-first here is how a novel's character eye colours were
        // crowded out of the bootstrap by the fourth session.
        let ordered = records.sorted { left, right in
            let leftImportance = left.importance ?? 0
            let rightImportance = right.importance ?? 0
            if leftImportance != rightImportance { return leftImportance > rightImportance }
            if left.createdAt != right.createdAt { return left.createdAt < right.createdAt }
            return left.key.rawValue < right.key.rawValue
        }
        var chosen: [MemoryRecord] = []
        var bytes = 0
        for record in ordered {
            guard chosen.count < limits.bootstrapRecords else { break }
            let size = record.key.rawValue.utf8.count + record.value.utf8.count
            guard bytes + size <= limits.bootstrapBytes else { continue }
            chosen.append(record)
            bytes += size
        }
        return MemoryBootstrap(
            records: chosen,
            omittedCount: records.count - chosen.count,
            totalBytes: bytes)
    }
}

/// Ranking for `search`, kept apart from storage so the retrieval strategy
/// can change without the backends changing.
///
/// Today: filter by prefix, tags and importance, then score by where the
/// query's terms appear. Deliberately not a vector index; the interface is
/// what allows one later, and adding one now would be a dependency and an
/// index to maintain for a store that holds a few hundred short facts.
public enum MemoryRanking {
    /// Rank a query's matches, best first, truncated to `query.limit`.
    ///
    /// `limit` is applied here rather than left to the caller because the two
    /// callers disagreed about it: the in-memory store prefixed the ranked
    /// array, while `ContinuityStore` set `bounded.limit` and passed it in —
    /// where it was computed and then ignored, so the durable backend returned
    /// every match (up to its 2000-candidate scan) for a query that asked for
    /// ten. Both backends now get the same answer, and clamping to
    /// `MemoryLimits.maximumSearchResults` stays the caller's business because
    /// only the caller holds the limits.
    public static func rank(_ records: [MemoryRecord], for query: MemoryQuery) -> [MemoryRecord] {
        let ranked: [MemoryRecord]
        let terms = (query.text ?? "")
            .lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
            .filter { $0.count > 2 }
        let candidates = records.filter { record in
            if let prefix = query.prefix, !prefix.isEmpty,
                !record.key.rawValue.hasPrefix(prefix)
            {
                return false
            }
            if let minimum = query.minimumImportance, (record.importance ?? 0) < minimum {
                return false
            }
            if !query.tags.isEmpty {
                let lowered = Set(record.tags.map { $0.lowercased() })
                guard query.tags.contains(where: { lowered.contains($0.lowercased()) }) else {
                    return false
                }
            }
            return true
        }
        if terms.isEmpty {
            ranked = candidates.sorted { scoreWithoutText($0) > scoreWithoutText($1) }
        } else {
            // How many candidates carry each term, so `textScore` can weight a
            // rare one above a common one. Read off the store rather than
            // hardcoded: "town" naming one fact and appearing in another's
            // value is what put the town above the rain rule.
            //
            // One lowered haystack at a time, not one per candidate: the
            // durable store scans up to 2000 records and a value may be 64 KiB,
            // so holding them all at once would be hundreds of megabytes for a
            // search that used to allocate one at a time.
            var documentFrequency: [String: Int] = [:]
            for record in candidates {
                let text = Self.haystack(record)
                for term in terms where text.contains(term) {
                    documentFrequency[term, default: 0] += 1
                }
            }
            let documents = Double(max(1, candidates.count))
            let scored = candidates.compactMap { record -> (MemoryRecord, Double)? in
                let score = textScore(
                    record, terms: terms,
                    documentFrequency: documentFrequency,
                    documents: documents)
                return score > 0 ? (record, score) : nil
            }
            ranked =
                scored
                .sorted { left, right in
                    if left.1 != right.1 { return left.1 > right.1 }
                    return left.0.updatedAt > right.0.updatedAt
                }
                .map(\.0)
        }
        return Array(ranked.prefix(max(0, query.limit)))
    }

    private static func scoreWithoutText(_ record: MemoryRecord) -> Double {
        (record.importance ?? 0) * 1000 + record.updatedAt.timeIntervalSince1970 / 1_000_000_000
    }

    /// Everything a term is matched against, lowered once.
    ///
    /// Tags are in here because they are matched in `textScore`: leaving them
    /// out made a tag-only query score zero for every candidate, which drops
    /// the fact entirely rather than ranking it.
    private static func haystack(_ record: MemoryRecord) -> String {
        (record.key.rawValue + " " + record.value + " "
            + record.tags.joined(separator: " ")).lowercased()
    }

    /// A term in the key counts for more than one in the body: the model
    /// names a memory for what it is about, so "decisions/sync" matching
    /// "sync" is a stronger signal than the word appearing in a sentence.
    ///
    /// Both are scaled by the term's inverse document frequency over the
    /// candidates, and a value match is worth two rather than one. Measured on
    /// the authored recall set (`benchmark/side_engine_recall.py
    /// --baseline`), that is what lifts paraphrase recall@1 from 1 of 4 to 3 of
    /// 4 while the ten questions phrased in the store's own words stay at 10 of
    /// 10. Without it a question naming a *common* word won: "does it ever rain
    /// in this town?" matched `setting/town`'s key for 3 and the rain rule's
    /// value for 1, so the town won.
    ///
    /// Smoothed (`+ 1`) so a term present in every candidate still counts, which
    /// keeps the key's three-to-one ordering intact among common terms.
    private static func textScore(
        _ record: MemoryRecord, terms: [String],
        documentFrequency: [String: Int],
        documents: Double
    ) -> Double {
        let key = record.key.rawValue.lowercased()
        let value = record.value.lowercased()
        let tags = record.tags.map { $0.lowercased() }
        var score = 0.0
        for term in terms {
            let frequency = documentFrequency[term] ?? 0
            guard frequency > 0 else { continue }
            let weight = log(documents / Double(frequency)) + 1
            if key.contains(term) { score += 3 * weight }
            if tags.contains(where: { $0.contains(term) }) { score += 2 * weight }
            if value.contains(term) { score += 2 * weight }
        }
        guard score > 0 else { return 0 }
        return score + (record.importance ?? 0)
    }
}
