import Foundation

/// Where to place the token-prefix "frontier" chunk checkpoints.
///
/// The follow-up prompt cache keys on whole-message equality, so an edit to an
/// earlier message (a mutated system prompt -- a memory-log append, a clock
/// rolling over) drops a turn to a full re-prefill even when the render still
/// shares tens of thousands of leading tokens with the previous one. This
/// tracks the leading token run recent renders agree on (the frontier), so a
/// prefill can checkpoint its KV+GDN state at chunk boundaries on it and a
/// later divergent-tail request can resume from the deepest one instead of
/// position zero.
///
/// Pure value logic, no model state -- the snapshots live in
/// `ServerPromptStateStore`, and which of them exist is the caller's to say.
struct FrontierTracker: Equatable {
    /// Prefill chunk width. Every checkpoint position is a multiple of this
    /// past where its prefill started, so the split prefill runs the same
    /// spans a single call would.
    let chunkTokens: Int

    /// Leading token run the observed renders still agree on.
    private(set) var frontier: [Int32] = []

    init(chunkTokens: Int) {
        self.chunkTokens = max(0, chunkTokens)
    }

    /// Fold a new render into the frontier.
    ///
    /// Within one conversation the frontier only shrinks, to the point the new
    /// render diverges. But a render that shares less than one chunk with it --
    /// a different client, or the same client's side request with its own
    /// system prompt (a title or summary call) -- reseeds it instead. A
    /// frontier shorter than a chunk can hold no checkpoint, so letting one
    /// foreign request shrink it that far would switch capture off for the
    /// rest of the process.
    mutating func observe(_ render: [Int32]) {
        let shared = commonPrefixLength(frontier, render)
        if frontier.isEmpty || shared < max(1, chunkTokens) {
            frontier = render
        } else if shared < frontier.count {
            frontier.removeLast(frontier.count - shared)
        }
    }

    /// Recent renders that were prefilled verbatim -- new conversations, and
    /// side requests -- as opposed to message-shaped continuations. Bounded,
    /// newest last, so one side call (a title, a classifier) cannot evict the
    /// conversation start it would otherwise be compared against.
    private(set) var recentVerbatimRenders: [[Int32]] = []
    static let recentVerbatimLimit = 4

    /// The longest prefix `render` shares with a recent verbatim render: the
    /// part of a new conversation proven to repeat across conversations. It
    /// stops wherever the client's first turn first varies -- a date inside the
    /// system prompt, or past it through an AGENTS.md the client inlines -- so
    /// it is where a cross-conversation checkpoint belongs, which the end of
    /// the system block is not when the block itself varies.
    func provenSharedPrefix(_ render: [Int32]) -> Int {
        recentVerbatimRenders.map { commonPrefixLength($0, render) }.max() ?? 0
    }

    mutating func rememberVerbatimRender(_ render: [Int32]) {
        recentVerbatimRenders.append(render)
        if recentVerbatimRenders.count > Self.recentVerbatimLimit {
            recentVerbatimRenders.removeFirst()
        }
    }

    /// Where a verbatim prefill should take its unaligned anchor: the prefix
    /// proven shared with a recent conversation when there is a useful one,
    /// else the end of the system block (the first conversation's best guess).
    static func anchor(proven: Int, systemBlockEnd: Int?) -> Int? {
        proven >= anchorMinimumGain ? proven : systemBlockEnd
    }

    /// The chunk-aligned positions this prefill should checkpoint, deepest
    /// first. Each one is on the frontier (a real shared prefix that long), a
    /// whole number of chunks past `resumeFrom`, strictly inside the render,
    /// and not in `held` (positions whose checkpoint for this very prefix
    /// already exists).
    ///
    /// A halving ladder, not every chunk: the deepest boundary, then the one
    /// half as many chunks in, a quarter, and so on down to one chunk. A
    /// snapshot is the whole prefix state (KV plus GDN), not a delta, so one
    /// per chunk writes a quadratic total -- about 24 GB for a 128K prompt on
    /// a 35B-A3B -- while the ladder stays under about twice the deepest one.
    /// Coverage is not lost for long: when a later render diverges at M, the
    /// frontier shrinks to M and the deepest boundary under M is the first
    /// rung of that request's own ladder.
    ///
    /// `anchor` is where the render's leading system block ends -- the prefix
    /// every new conversation from the same client shares, and so the one
    /// checkpoint a first turn most wants. It is taken even though it is not
    /// chunk-aligned, which costs this prefill one extra partial chunk pass:
    /// on a model with 16K chunks a whole agent system prompt fits inside the
    /// first chunk, so without it no checkpoint could ever cover that prompt.
    /// It is skipped when it is already held, or when an aligned boundary sits
    /// within `anchorMinimumGain` tokens under it (small chunks already cover
    /// it). Ladder rungs past the anchor then count chunks from the anchor,
    /// since the prefill resumes from there.
    func captureTargets(
        render: [Int32], resumeFrom: Int, held: Set<Int>, anchor: Int? = nil
    ) -> [Int] {
        guard chunkTokens > 0 else { return [] }
        let shared = min(commonPrefixLength(frontier, render), render.count - 1)
        if let anchor, anchor > resumeFrom, anchor < render.count, !held.contains(anchor),
            (anchor - resumeFrom) % chunkTokens >= Self.anchorMinimumGain
        {
            return ladder(from: anchor, to: shared, held: held) + [anchor]
                + ladder(from: resumeFrom, to: min(shared, anchor - 1), held: held)
        }
        return ladder(from: resumeFrom, to: shared, held: held)
    }

    /// Smallest distance past the deepest aligned boundary for which an
    /// unaligned anchor checkpoint is worth its extra prefill pass.
    static let anchorMinimumGain = 1_024

    /// Halving ladder of whole-chunk positions past `start`, up to `limit`.
    private func ladder(from start: Int, to limit: Int, held: Set<Int>) -> [Int] {
        guard limit > start else { return [] }
        var targets: [Int] = []
        var chunks = (limit - start) / chunkTokens
        while chunks > 0 {
            let position = start + chunks * chunkTokens
            if !held.contains(position) { targets.append(position) }
            chunks /= 2
        }
        return targets
    }

    private func commonPrefixLength(_ a: [Int32], _ b: [Int32]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n, a[i] == b[i] { i += 1 }
        return i
    }
}
