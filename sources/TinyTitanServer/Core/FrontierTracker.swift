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
    func captureTargets(render: [Int32], resumeFrom: Int, held: Set<Int>) -> [Int] {
        guard chunkTokens > 0 else { return [] }
        let shared = min(commonPrefixLength(frontier, render), render.count - 1)
        guard shared > resumeFrom else { return [] }
        var targets: [Int] = []
        var chunks = (shared - resumeFrom) / chunkTokens
        while chunks > 0 {
            let position = resumeFrom + chunks * chunkTokens
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
