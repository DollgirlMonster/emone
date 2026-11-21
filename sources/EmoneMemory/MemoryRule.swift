import Foundation

/// The stored rule that fixes one attribute, if there is one.
///
/// A rule is filed under `rules/<attribute>`, where the attribute is the last
/// segment of the key it governs: `rules/eyes` fixes `characters/marcus/eyes`.
/// That is the whole convention, and the lookup is a key match rather than a
/// model call on purpose — it runs on the write path, a wrong rule could block
/// a real update, and the side-engine reading a rule only helps if the rule is
/// the right one.
///
/// A rule *about* a subject rather than an attribute (`rules/ferry = runs only
/// on Sundays`) governs no key's value and is deliberately not matched:
/// supersession is about a value that changed, and only an attribute has one.
public enum MemoryRuleLookup {
    /// The segment every rule is filed under.
    public static let category = "rules"

    /// The rule text governing `key`, or nil.
    ///
    /// `facts` is the scope's facts. The first exact `rules/<attribute>` match
    /// wins, whatever its namespace, because a rule is about the attribute
    /// rather than about one holder of it.
    public static func rule(for key: MemoryKey, among facts: [MemoryRecord]) -> String? {
        guard let wanted = ruleKey(for: key)?.rawValue else { return nil }
        return facts.first { $0.key.rawValue == wanted }?.value
    }

    /// The key a rule for this key would be filed under, or nil.
    ///
    /// A single-segment key names no attribute — `decisions` has no
    /// `rules/decisions` shape — so it has no rule.
    public static func ruleKey(for key: MemoryKey) -> MemoryKey? {
        let segments = key.rawValue.split(separator: "/")
        guard segments.count >= 2, let attribute = segments.last else { return nil }
        return try? MemoryKey(validating: "\(category)/\(attribute)")
    }
}
