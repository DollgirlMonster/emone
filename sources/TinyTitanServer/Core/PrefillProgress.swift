import Synchronization

/// Coarse progress of the current generation, for a client progress bar. A
/// long prefill (tens of thousands of tokens, no output yet) otherwise looks
/// like a hang. Read via `GET /v1/prefill-progress`.
///
/// `phase` is `idle` between requests, `prefill` while the prompt is being
/// processed, `decode` once tokens are being produced. `generation` bumps once
/// per request so a poller can tell a new turn from a stale reading. With more
/// than one sequence in flight it follows the most recently started one: an
/// update or end from an older generation is ignored rather than clobbering it.
public struct PrefillProgress: Sendable, Codable, Equatable {
    public enum Phase: String, Sendable, Codable { case idle, prefill, decode }
    public var phase: Phase
    public var done: Int
    public var total: Int
    public var generation: Int

    public static let idle = PrefillProgress(phase: .idle, done: 0, total: 0, generation: 0)
}

enum PrefillProgressMonitor {
    static let shared = Mutex(PrefillProgress.idle)

    /// Start a generation's progress; returns its generation number, which the
    /// caller hands back to the other updates.
    @discardableResult
    static func begin(total: Int, cached: Int) -> Int {
        shared.withLock {
            $0 = PrefillProgress(
                phase: .prefill, done: min(cached, total),
                total: total, generation: $0.generation &+ 1)
            return $0.generation
        }
    }

    static func prefill(done: Int, total: Int, generation: Int) {
        shared.withLock {
            guard $0.generation == generation, $0.phase == .prefill else { return }
            $0.done = done
            $0.total = total
        }
    }

    static func decoding(generation: Int) {
        shared.withLock {
            guard $0.generation == generation, $0.phase == .prefill else { return }
            $0.phase = .decode
            $0.done = $0.total
        }
    }

    static func end(generation: Int) {
        shared.withLock {
            guard $0.generation == generation else { return }
            $0.phase = .idle
            $0.done = 0
            $0.total = 0
        }
    }

    static var snapshot: PrefillProgress { shared.withLock { $0 } }
}
