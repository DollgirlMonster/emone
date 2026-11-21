import Foundation

/// One loaded `Model`, held by every runner over it (Emone-lmw.11). A
/// `Model` is a value whose weights sit in reference-counted Metal buffers, so
/// the runners' copies share one set of weights; this object is what the
/// registry tracks to know whether any of them is still alive.
///
/// unchecked-invariant: `model` is a `let` set once in `init`; its buffers are
/// read-only after the load.
public final class SharedModelWeights: @unchecked Sendable {
    public let model: Model
    public let directory: URL

    init(model: Model, directory: URL) {
        self.model = model
        self.directory = directory
    }
}

/// Lets two runners over the same install share its weights however their
/// lifetimes interleave: the side chat model and the System One adapter laid
/// over it each ask for the install, and whichever comes second gets the live
/// weights instead of a second copy. Entries are weak, so the weights go when
/// the last runner holding them does -- each user keeps its own idle unload.
///
/// Only for installs without streamed experts (the dense family): a routed
/// model's load also fixes its expert streaming and cache policy, which two
/// users would have to agree on.
///
/// unchecked-invariant: `entries` is only touched under `lock`.
public final class ModelWeightsRegistry: @unchecked Sendable {
    public static let shared = ModelWeightsRegistry()

    private final class Entry {
        weak var weights: SharedModelWeights?
        init(_ weights: SharedModelWeights) { self.weights = weights }
    }

    private let lock = NSLock()
    private var entries: [String: Entry] = [:]

    /// The live weights of `directory`, or `load()`'s, registered for the
    /// next caller. Loads run under the lock, so two first callers never load
    /// the same install twice.
    public func weights(
        directory: URL, load: () throws -> Model
    ) throws -> SharedModelWeights {
        let key = directory.standardizedFileURL.resolvingSymlinksInPath().path
        lock.lock()
        defer { lock.unlock() }
        if let live = entries[key]?.weights { return live }
        let made = SharedModelWeights(model: try load(), directory: directory)
        entries[key] = Entry(made)
        return made
    }

    /// True when some runner still holds `directory`'s weights.
    public func isLive(directory: URL) -> Bool {
        let key = directory.standardizedFileURL.resolvingSymlinksInPath().path
        lock.lock()
        defer { lock.unlock() }
        return entries[key]?.weights != nil
    }
}
