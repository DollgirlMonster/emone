import Foundation
import Testing

@testable import EmoneMemory

/// What `memory_search` returns, and why.
///
/// The measured weakness was a *common* word winning: a question naming "town"
/// scored 3 against `setting/town`'s key while the rain rule's value scored 1,
/// so the town came first on a question about rain. These pin the fix — a term
/// weighted by its inverse document frequency over the candidates, and a value
/// match worth two rather than one — and the ordering it must not change.
///
/// `benchmark/side_engine_recall.py --baseline` mirrors this scorer and measures
/// it: the ten questions phrased in the store's own words stay at 10 of 10, and
/// the authored paraphrases go from 1 of 4 to 3 of 4 at rank 1. The one that
/// remains needs a word the fact does not contain at all ("boat" against a
/// ferry rule), which no term weighting reaches.
@Suite struct MemoryRankingTests {
    private func record(
        _ key: String, _ value: String,
        tags: [String] = []
    ) throws -> MemoryRecord {
        MemoryRecord(key: try MemoryKey(validating: key), value: value, tags: tags)
    }

    private func ranked(_ records: [MemoryRecord], _ text: String) -> [String] {
        MemoryRanking.rank(records, for: MemoryQuery(text: text, limit: 10))
            .map(\.key.rawValue)
    }

    private func rainStore() throws -> [MemoryRecord] {
        [
            try record("setting/town", "Ashgrove"),
            try record("characters/ines/role", "the town archivist"),
            try record("rules/weather", "it never rains"),
        ]
    }

    @Test func aRareWordBeatsACommonOne() throws {
        // "ever" and "rain" each appear in one fact; "town" names one and sits
        // in another's value.
        #expect(
            ranked(try rainStore(), "does it ever rain in this town").first
                == "rules/weather")
    }

    @Test func theKeyStillOutranksTheValue() throws {
        let store = [
            try record("rules/ferry", "runs only on Sundays"),
            try record("notes/boats", "the ferry is a boat"),
        ]
        #expect(ranked(store, "ferry").first == "rules/ferry")
    }

    @Test func aValueMatchIsReturnedWhenNothingElseMatches() throws {
        #expect(ranked(try rainStore(), "rain") == ["rules/weather"])
    }

    @Test func aTermInEveryCandidateStillCounts() throws {
        // Both candidates carry "town", so the weight smooths to 1 rather than
        // to zero; the key match still wins.
        #expect(ranked(try rainStore(), "town").first == "setting/town")
    }

    @Test func aFactSharingNoTermIsNotReturned() throws {
        #expect(ranked(try rainStore(), "quantum").isEmpty)
    }

    @Test func aTagMatchStillCounts() throws {
        let store = [
            try record("plot/turn", "the brother returns", tags: ["pivot"]),
            try record("plot/open", "a storm"),
        ]
        #expect(ranked(store, "pivot") == ["plot/turn"])
    }

    @Test func withoutTextTheImportanceOrderStands() throws {
        let store = [
            try record("a/low", "x", tags: []),
            try record("b/high", "y"),
        ]
        var records = store
        records[1].importance = 0.9
        records[0].importance = 0.1
        #expect(ranked(records, "").first == "b/high")
    }
}
