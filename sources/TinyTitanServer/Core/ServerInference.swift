import CryptoKit
import Foundation
import TinyTitan

/// A model the server was asked to load but this build cannot run.
enum ServerInferenceError: Error, CustomStringConvertible {
    case unsupportedModel(String)

    var description: String {
        switch self {
        case .unsupportedModel(let detail): return detail
        }
    }
}

/// Prefill chunk size when the caller has not asked for one. Both families
/// with a long-chunk default want it for the same reason -- routed experts are
/// what prefill spends itself on, and a longer chunk amortizes them.
func defaultPrefillChunkTokens(family: ModelFamily, fallback: Int) -> Int {
    switch family {
    case .qwen36: return RuntimeConfiguration.qwenLongPrefillChunkTokens
    // 4,096 for Qwen3.8 too. Prefill's expert cache is inert -- a chunk routes
    // essentially every expert in a layer against 96 slots, so the hit rate is
    // 0.6% and each chunk re-streams what the last evicted. The cost is
    // therefore proportional to the chunk *count*: an 8k prompt is 5 chunks at
    // 2,048 and 3 at 4,096, measured at 167.5 -> 111.0 GiB of expert reads and
    // 506.4 -> 450.5 s of prefill (-11%), with identical output on a
    // multi-chunk prompt. The 2,048 here predates that measurement.
    //
    // It is a trade, not free. A/B/A on one machine state, 0.4% drift
    // between the repeated arms: decode 6.97 / 7.19 / 7.00 tok/s at
    // 4096 / 2048 / 4096, so 2,048 decodes ~3% faster -- the KV ring is
    // sized from the chunk and this machine feels the reservation.
    // 4,096 still wins for the long-prompt case it is chosen for: 56 s
    // of prefill on a 10k prompt against ~2 s of a 512-token generation.
    // A short-prompt, long-generation workload would want 2,048 back.
    case .qwen38flash: return RuntimeConfiguration.qwenLongPrefillChunkTokens
    default: return fallback
    }
}

public enum ServerInferenceEvent: Equatable, Sendable {
    case content(String)
    /// Thought text from inside the model's `<think>` block. Kept apart from
    /// `content` so each surface can put it where its clients look for
    /// reasoning, and so nothing that judges the answer ever reads it.
    case reasoning(String)
    case toolCall(ParsedToolCall)
}

public struct ServerCompletion: Equatable, Sendable {
    public let content: String
    /// Everything the model thought, in order; empty with thinking off.
    /// `usage.completionTokens` already counts these tokens, as it always
    /// has -- only where the text goes has changed.
    public let reasoning: String
    /// Characters of `reasoning` the model wrote although this request's
    /// render had thinking off.
    ///
    /// Not a fault of the runtime -- some installs think with the switch off,
    /// measured on Qwen AgentWorld 35B-A3B 8-bit -- but not something to leave
    /// silent either: the operator asked for no thought, paid tokens for one,
    /// and a client that caps tokens gets an empty answer rather than a short
    /// one. Carried as a count, like the watchdog trips, so the HTTP layer can
    /// log it where generated text does not belong.
    public let unrequestedReasoning: Int
    public let toolCalls: [ParsedToolCall]
    public let finishReason: String
    public let usage: OpenAIUsage
    /// What the watchdogs saw, empty when they are off. Carried on the
    /// completion so the HTTP layer, which owns the request id, can log them
    /// on the one line that already reports how the request ended.
    public let watchdogTrips: [WatchdogSet.Trip]
    /// The client stop string that ended generation, when one did. OpenAI
    /// folds this into finish_reason "stop"; the Anthropic Messages API
    /// distinguishes it as stop_reason "stop_sequence" and names the string.
    public let stopSequence: String?

    public init(
        content: String,
        toolCalls: [ParsedToolCall],
        finishReason: String,
        usage: OpenAIUsage,
        watchdogTrips: [WatchdogSet.Trip] = [],
        stopSequence: String? = nil,
        reasoning: String = "",
        unrequestedReasoning: Int = 0
    ) {
        self.content = content
        self.reasoning = reasoning
        self.unrequestedReasoning = unrequestedReasoning
        self.toolCalls = toolCalls
        self.finishReason = finishReason
        self.usage = usage
        self.watchdogTrips = watchdogTrips
        self.stopSequence = stopSequence
    }
}

/// A backend that can count the prompt tokens a request would occupy
/// without generating. Kept apart from `ServerInferenceBackend` so wrappers
/// and test doubles that cannot count are not forced to pretend; the
/// Anthropic `count_tokens` endpoint answers 501 when the backend lacks it.
public protocol PromptTokenCounting: Sendable {
    func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int
}

/// A backend that can say which prompt-cache mode it is really running.
///
/// Kept apart from `ServerInferenceBackend` for the same reason
/// `PromptTokenCounting` is: an engine with no prompt cache (the CPU backend)
/// and the test doubles should not have to answer for a cache they do not have.
/// One resident slot serves a whole catalog, so a residency line that named the
/// mode from the server's flags rather than from the backend that just loaded
/// would report the previous model's cache after a switch.
public protocol PromptCacheDescribing: Sendable {
    /// The mode in force for this backend, never the one requested.
    var promptCacheMode: ServerPromptCacheMode { get }
}

public protocol ServerInferenceBackend: Sendable {
    /// The backend's configured context window, used to validate
    /// max_tokens/max_completion_tokens against the session's maxContext (S11).
    var maximumContext: Int { get }
    /// Sampling values used for whatever the request omits. A family whose
    /// model card differs from the house settings reports its own here, so a
    /// client that sends no temperature gets what the model was tuned for
    /// rather than what the last family to need tuning wanted.
    var samplingDefaults: GenerationDefaults.Sampling { get }
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion
}

extension ServerInferenceBackend {
    public var maximumContext: Int {
        RuntimeConfiguration.supportedContextTokens.max() ?? 262_144
    }
    public var samplingDefaults: GenerationDefaults.Sampling { GenerationDefaults.house }
}

/// A backend that owns the model's residency and can release it on demand.
///
/// Kept separate from `ServerInferenceBackend` rather than added to it with a
/// `false`-returning default: exactly one backend manages residency, and the
/// wrapper design exists so the HTTP layer stays unaware of loading at all.
/// Folding it into the inference protocol would make every conforming type —
/// including the plain session and every test stub — carry a member that only
/// answers "not me".
public protocol ResidencyManaging: Sendable {
    /// Releases the model's memory, waiting for in-flight requests to drain
    /// first. Returns true when a resident model was actually released.
    func unload() async -> Bool
}

/// Whether a client generation is running, readable without awaiting the
/// coordinator.
///
/// The CPU side-engine needs this before every token it produces, from
/// whatever thread it happens to be on, and `await`ing an actor to decide
/// how wide to run a GEMV would cost more than the decision is worth.
///
/// unchecked-invariant: `depth` is only ever read or written under `lock`.
public final class GenerationSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var depth = 0

    public init() {}

    /// True while at least one client generation is in flight.
    public var isBusy: Bool { lock.withLock { depth > 0 } }

    func enter() { lock.withLock { depth += 1 } }
    func leave() { lock.withLock { depth = max(0, depth - 1) } }
}

public actor ServerCoordinator {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
    }

    private let queueLimit: Int
    /// How many generations may run at once. One is the historical
    /// single-generation server; more lets the batched slots through while the
    /// excess still queues. The engine's `ForwardStepGate` keeps their forward
    /// passes from interleaving.
    private let width: Int
    private var admittedCount = 0
    private var activeCount = 0
    private var waiters: [Waiter] = []
    private var shuttingDown = false
    /// Raised for the duration of every client generation. The side-engine
    /// reads it to choose its width: one thread while a person is waiting,
    /// four in the gaps.
    public nonisolated let generating = GenerationSignal()

    public init(queueLimit: Int, width: Int = 1) {
        self.queueLimit = queueLimit
        self.width = max(1, width)
    }

    public func run<T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await runPreparing(
            onQueued: onQueued,
            prepare: { () },
            operation: { _ in try await operation() })
    }

    func runPreparing<Prepared: Sendable, T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        prepare: @escaping @Sendable () async throws -> Prepared,
        operation: @escaping @Sendable (Prepared) async throws -> T
    ) async throws -> T {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        // S6: at most `width` running and `queueLimit` queued behind them, so
        // `width + queueLimit` admitted. Width 1 reproduces the original
        // single-generation bound exactly.
        guard admittedCount < width + queueLimit else {
            // Shed load rather than queue without bound.
            throw ServerRequestError.queueFull
        }
        admittedCount += 1
        defer { admittedCount -= 1 }

        let prepared = try await prepare()
        try Task.checkCancellation()
        try await acquire(onQueued: onQueued)
        defer { release() }
        generating.enter()
        defer { generating.leave() }
        return try await operation(prepared)
    }

    private func acquire(onQueued: @escaping @Sendable () -> Void) async throws {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        if activeCount < width {
            activeCount += 1
            return
        }
        guard waiters.count < queueLimit else { throw ServerRequestError.queueFull }
        onQueued()
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(Waiter(id: id, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        if Task.isCancelled {
            release()
            throw CancellationError()
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }

    private func release() {
        if waiters.isEmpty {
            activeCount = max(0, activeCount - 1)
        } else {
            waiters.removeFirst().continuation.resume()
        }
    }

    public func shutdown() {
        shuttingDown = true
        let queued = waiters
        waiters.removeAll()
        for waiter in queued {
            waiter.continuation.resume(throwing: CancellationError())
        }
    }

    public var queuedCount: Int { waiters.count }
    public var isActive: Bool { activeCount > 0 }
    /// Running generations, for tests and the readiness view.
    public var runningCount: Int { activeCount }
    /// The admission width this coordinator was built with.
    public var concurrencyWidth: Int { width }
}

/// Snapshot of the runner's lifetime stage counters at request start, so the
/// TINYTITAN_RUNNER_STATS footer can report this request's per-stage deltas.
private struct RunnerCounterSnapshot {
    let cb1: UInt64
    let io: UInt64
    let cb2: UInt64
    let head: UInt64
    let headFused: UInt64
    let rdadvise: UInt64
    let rdadviseCalls: UInt64
    let rdadviseBytes: UInt64
    let wait: UInt64
    let body: UInt64
    let prefetchIssued: UInt64
    let prefetchAdopted: UInt64
    let preamble: UInt64
    let preambleRelease: UInt64
    let preamblePin: UInt64
    let preambleReserve: UInt64
    let embed: UInt64
    let gather: UInt64
    let loopSample: UInt64
    let loopProgress: UInt64
    let loopOther: UInt64
    let missIo: UInt64
    let exposedIo: UInt64
    let hitFixupLayers: UInt64
    let routerReadback: UInt64
    let cachePlan: UInt64
    let ioQueue: UInt64
    let ioCompletionToFixup: UInt64
    let ioHostWaits: UInt64
    let ioHostWaitsAvoided: UInt64
    let gpuClassifiedHits: UInt64
    let gpuClassifiedMisses: UInt64
    let gpuAllHitLayers: UInt64
    let expertStreaming: ExpertStreamingStatistics
}

/// One mid-prefill checkpoint, held until the generation ends and it is banked.
private struct FrontierCapture: Sendable {
    let position: Int
    let snapshot: InferenceStateSnapshot
}

/// Per-generation decode state that `runRawCompletion`'s progress closure
/// mutates. Boxed so the closure captures a reference the compiler can send
/// into the nonisolated call; Swift 6.4 rejects sending the captured mutable
/// struct itself.
///
/// unchecked-invariant: one box per generation, and the coordinator plus the
/// session's slot pool guarantee one generation per occurrence, so exactly one
/// task ever touches a given box.
private final class GenerationDecodeState: @unchecked Sendable {
    /// The decoder is per-generation state too, and holding it here is what
    /// lets the progress closure capture only this box: a closure that also
    /// captured the decoder directly is not Sendable, and Swift 6.4 refuses to
    /// send it into the nonisolated completion call.
    let decoder: StructuredAssistantDecoder
    var output: AssistantOutput
    var decodingError: Error?
    var shouldStop = false
    /// Frontier checkpoints captured mid-prefill, banked after the generation.
    var frontierCaptures: [FrontierCapture] = []

    init(decoder: StructuredAssistantDecoder, output: AssistantOutput) {
        self.decoder = decoder
        self.output = output
    }
}

public actor ServerModelSession: ServerInferenceBackend, PromptTokenCounting, PromptCacheDescribing
{
    /// Manifest-derived API model identifier used when --model-id is absent.
    public nonisolated let defaultModelID: String
    /// The session's configured context window; the HTTP layer validates
    /// max_tokens against it (S11).
    public nonisolated var maximumContext: Int { maxContext }
    public nonisolated var samplingDefaults: GenerationDefaults.Sampling {
        profileSampling
    }
    private nonisolated let profileSampling: GenerationDefaults.Sampling
    private nonisolated let modelFamily: ModelFamily

    private let context: MetalContext
    private let model: Model
    private let tokenizer: GFTokenizer
    /// The tokenizer's vocabulary as byte strings, built the first time a
    /// request asks for structured output and kept for the life of the model.
    /// The bytes of a token id do not change with the reasoning level a request
    /// re-renders at, so one table serves every request this session handles.
    private var jsonTokenTable: JSONTokenTable?
    /// Where the tokenizer came from and the reasoning it was rendered at.
    ///
    /// Kept so a request that asks for a different thinking mode or effort can
    /// resolve its own tokenizer through the shared
    /// `(folder, thinking, effort)` cache, instead of being pinned to whatever
    /// the model was loaded with. `nil` reasoning means the session's own.
    private let tokenizerFolder: URL
    private nonisolated let loadedReasoning: RequestReasoning
    private let runner: RealForwardRunner
    private let mtpDecoder: StreamingMTPDecoder?
    /// One raw-completion scratch per slot: its own logits/probs/token buffers
    /// and its own sampler, so concurrent slots cannot sample from each other's
    /// logits.
    private let scratches: [RawCompletionScratch]
    /// Slots not currently held by a generation. Bounded by the coordinator's
    /// width; the waiter queue is a safety net if width ever exceeds slots.
    private var freeSlots: [Int]
    private var slotWaiters: [SlotWaiter] = []
    private let prefillConfig: PrefillRuntimeConfig
    // Long prompts are prefilled chunk by chunk — small enough to keep expert
    // reads tight.
    public nonisolated let prefillChunkTokens: Int
    /// Routed-expert slots per layer actually in force, so the ready banner can
    /// report the streaming budget rather than leaving the user to infer it.
    public nonisolated let expertCacheSlots: Int
    /// How many sequences this session runs at once.
    public nonisolated let slots: Int
    private let maxContext: Int
    public nonisolated let promptCacheMode: ServerPromptCacheMode
    private let promptCacheDomain: ServerPromptCacheDomain
    private var promptCache: ServerPromptCache
    private let promptStateStore: ServerPromptStateStore?
    private var activePromptCacheEntryID: UUID?
    /// Where the token-prefix frontier checkpoints go (see FrontierTracker),
    /// and the ones that exist. Each `frontierEntries` row is a chunk-aligned
    /// KV+GDN snapshot in `promptStateStore` whose `tokens` is the exact
    /// prefix it covers; a later request that still begins with them restores
    /// the deepest match and resumes from there, whatever changed after it --
    /// the mutated-system-prompt case the message-shaped cache misses because
    /// it keys on whole-message equality. They share the store's RAM and SSD
    /// budgets and its oldest-first eviction with the message-shaped entries.
    private var frontier: FrontierTracker
    private var frontierEntries: [FrontierEntry] = []
    /// Bumped on every frontier restore or store, so the row cap drops the
    /// least recently useful checkpoint first.
    private var frontierClock = 0
    /// Concise-mode system prompt injected into every completion, or nil when
    /// concise mode is off. Selected per quantization (see ConcisePrompt).
    private nonisolated let concisePrompt: String?

    private struct SlotWaiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
    }

    private struct FrontierEntry {
        let tokens: [Int32]
        let id: UUID
        var lastUse: Int
    }

    /// A pure function of its arguments, so a caller can reproduce the
    /// effective cache mode for the startup banner without loading a model.
    public static func effectivePromptCacheMode(
        requested: ServerPromptCacheMode,
        mtpEnabled: Bool,
        slots: Int = 1
    ) -> ServerPromptCacheMode {
        // A target-only snapshot cannot restore the draft stream. Keeping a
        // cache allocated while MTP is active would spend memory on entries
        // that must never be consumed or published.
        guard !mtpEnabled else { return .off }
        // The cache holds one sequence's KV prefix, and its snapshot/restore and
        // `activePromptCacheEntryID` are session-wide. With more than one slot
        // that entry could be restored into the wrong sequence, which produces
        // plausible wrong output rather than an error, so batching runs with the
        // cache off and re-prefills each turn until it is slot-keyed.
        return slots > 1 ? .off : requested
    }

    /// The prompt cache a catalog server's *initial* model will really run,
    /// which is what the routing banner states beside `engine=`.
    ///
    /// The extra input over `effectivePromptCacheMode` is the engine: a CPU
    /// entry has no cache at all, so it reports `.off` rather than a mode the
    /// server asked for that could never exist. The catalog loader never
    /// attaches MTP, so the engine and the width are the whole rule. Kept here
    /// rather than in the executable so both arms are testable without a
    /// catalog on disk.
    public static func initialPromptCacheMode(
        backend: ModelCatalog.Backend,
        requested: ServerPromptCacheMode,
        maxConcurrentSequences: Int
    ) -> ServerPromptCacheMode {
        guard backend != .cpu else { return .off }
        return effectivePromptCacheMode(
            requested: requested, mtpEnabled: false,
            slots: maxConcurrentSequences)
    }

    /// lint:allow-long a sequential construction pipeline: tokenizer, Metal
    /// context, runtime config, model, optional MTP sidecar, runner, scratch.
    /// Each step consumes the last, so extracting any of them would return a
    /// tuple straight back into the next -- the same shape as Model.load.
    public static func load(
        modelDirectory: URL,
        maxContext: Int,
        slots: Int = 1,
        promptCacheMode: ServerPromptCacheMode = .multiPrefix,
        promptCacheMaximumEntries: Int = 4,
        promptCacheMemoryLimitBytes: Int = 256 * 1_048_576,
        promptCacheDiskDirectory: URL? = nil,
        promptCacheDiskLimitBytes: Int = 8_192 * 1_048_576,
        promptCacheMemoryTTLSeconds: Int = 0,
        prefillChunkTokens requestedPrefillChunkTokens: Int? = nil,
        kvCachePrecision: KVCachePrecision = .int8,
        ropeScalingMode: RuntimeRoPEScalingMode = .none,
        thinkingMode: ModelThinkingMode = .off,
        reasoningEffort: ModelReasoningEffort? = nil,
        expertCacheSlots requestedExpertCacheSlots: Int? = nil,
        expertCacheBudgetBytes: Int? = nil,
        mtpModelDirectory: URL? = nil,
        mtpMemoryMiB: Int = StreamingMTPMemoryPlan.defaultBudgetMiB,
        reusingContext: MetalContext? = nil
    ) async throws -> ServerModelSession {
        let tokenizerFolder = GFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
        guard let tokenizerFolder else {
            throw GFTokenizerError.missingToolTemplate
        }
        let templateURL = tokenizerFolder.appendingPathComponent("chat_template.jinja")
        guard FileManager.default.fileExists(atPath: templateURL.path) else {
            throw GFTokenizerError.missingToolTemplate
        }
        // Reasoning effort is defined per family; reject it before the
        // tokenizer bakes an unsupported control into its rendering. An
        // unreadable manifest is left for Model.load, which reports it better.
        if reasoningEffort != nil,
            let family = try? ManifestReader.peekFamily(directoryURL: modelDirectory)
        {
            try family.validateReasoning(
                thinkingMode: thinkingMode,
                effort: reasoningEffort)
        }
        let tokenizer = try await GFTokenizer.load(
            from: tokenizerFolder,
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort)
        // A caller managing model residency supplies its own context so one
        // MTLCommandQueue and one compiled shader library survive across
        // unload/reload cycles (MetalContext.deinit documents that queue
        // teardown is not deinit-safe). Nil for every ordinary caller.
        let context = try reusingContext ?? MetalContext()
        let loadRuntime = try RuntimeConfiguration(
            forceLogitsHead: true,
            decodeExpertExecution: try RuntimeDecodeExpertExecution.environmentValue(),
            expertIOSynchronization: try RuntimeExpertIOSynchronization.environmentValue(),
            expertIOSubmission: try RuntimeExpertIOSubmission.environmentValue())
        let slotOverride = ProcessInfo.processInfo.environment["TINYTITAN_EXPERT_CACHE_SLOTS"]
            .flatMap(Int.init)
        // Precedence: --expert-cache-slots flag, then the env override, then a
        // count derived from the model's own expert stride against a 1 GiB budget.
        //
        // Derived rather than fixed because the right count depends on the
        // quantisation: 1 GiB is 16 slots at 4-bit and 8 at 8-bit, which are the
        // measured optima for each. The previous fixed default of 64 was slower
        // *and* larger than either -- benchmarked at the shipped 262144 context,
        // 4-bit managed 9.85 tok/s at 64 slots against 13.61 at 16.
        // The architecture comes from the manifest rather than being assumed,
        // exactly as the CLI resolves it: a payload of any other family should
        // load, not fail on a dimension mismatch.
        let modelFamily = try ManifestReader.peekFamily(directoryURL: modelDirectory)
        let expectedArch: ArchConfig
        do {
            // The family's preset, or -- for a family with more than one
            // geometry, like the dense Qwen 3.5 models -- the manifest's own
            // declaration. `ServerInferenceError` keeps the shape callers
            // expect; the reason travels in the message.
            expectedArch = try ArchConfig.resolved(
                forFamily: modelFamily,
                directoryURL: modelDirectory)
        } catch {
            throw ServerInferenceError.unsupportedModel("\(error)")
        }
        let derivedSlots: Int
        // An explicit --ram-budget names what the whole server may hold; this
        // only supplies the cache default, and it is clamped so a family tuned
        // on a 24 GiB machine cannot hand a smaller one a budget it has no room
        // for.
        let tunedBudget: Int
        if let identity = try? ManifestReader.peekIdentity(directoryURL: modelDirectory) {
            tunedBudget = RuntimeConfiguration.affordableExpertCacheBudget(
                ModelProfile.resolve(identity: identity).expertCacheBudgetBytes)
        } else {
            tunedBudget = RuntimeConfiguration.defaultExpertCacheBudgetBytes
        }
        let loadedManifest = try? ManifestReader.load(
            directoryURL: modelDirectory,
            expecting: expectedArch)
        // Bytes one expert occupies in the cache, across every layer.
        let cachePerSlotBytes =
            loadedManifest.map {
                Double($0.expertStride) * Double($0.arch.numLayers)
            } ?? 0
        let residentFloor = RuntimeConfiguration.residentFloorBytes(
            residentWeightBytes: loadedManifest?.files["model_weights.bin"]
                .map { Int($0.size) } ?? 0)
        let gib = { (bytes: Double) in bytes / 1_073_741_824 }
        let slotsGib = { (slots: Int) in gib(Double(slots) * cachePerSlotBytes) }
        if let explicitTarget = expertCacheBudgetBytes {
            // The flag is a target for the whole process, not just the cache:
            // the weights and the runtime are resident either way, so the cache
            // is what is left, and the slot count steps down to stay inside the
            // number the user named.
            derivedSlots = RuntimeConfiguration.expertCacheSlotsFitting(
                expertStrideBytes: loadedManifest?.expertStride ?? 0,
                layers: loadedManifest?.arch.numLayers ?? 0,
                cacheBytes: explicitTarget - residentFloor)
            let targetGib = gib(Double(explicitTarget))
            let floorGib = gib(Double(residentFloor))
            let cacheGib = slotsGib(derivedSlots)
            print(
                String(
                    format: "TinyTitan ram target=%.2fG cache=%.2fG slots=%d "
                        + "resident_floor=%.2fG estimate=%.2fG",
                    targetGib, cacheGib, derivedSlots, floorGib,
                    floorGib + cacheGib))
            if targetGib < floorGib + cacheGib {
                print(
                    String(
                        format: "TinyTitan ram warning: %.2fG is below this install's "
                            + "%.2fG floor (%.2fG resident + the %d-slot minimum cache); "
                            + "the cache is already at its smallest.",
                        targetGib, floorGib + cacheGib, floorGib, derivedSlots))
            }
        } else if let manifest = loadedManifest {
            derivedSlots = RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: manifest.expertStride,
                layers: manifest.arch.numLayers,
                budgetBytes: tunedBudget)
            let floorGib = gib(Double(residentFloor))
            print(
                String(
                    format: "TinyTitan ram profile cache=%.2fG slots=%d "
                        + "resident_floor=%.2fG estimate=%.2fG (cache budget, not a "
                        + "process target)",
                    slotsGib(derivedSlots), derivedSlots, floorGib,
                    floorGib + slotsGib(derivedSlots)))
        } else {
            // Unreadable manifest means the load below will fail with a better
            // message than anything this could throw, so pick the safe small end.
            derivedSlots = RuntimeConfiguration.allowedExpertCacheSlots.first ?? 8
        }
        let loadSlots = requestedExpertCacheSlots ?? slotOverride ?? derivedSlots
        let model = try Model.load(
            directoryURL: modelDirectory,
            device: context.device,
            expecting: expectedArch,
            streamingMode: .pread(slotCount: loadSlots),
            expertCachePolicy: loadRuntime.modelExpertCachePolicy,
            integrityPolicy: .resolved(directoryURL: modelDirectory))
        let runtime = try RuntimeConfiguration(
            expertCacheSlots: loadSlots,
            expertCachePolicy: loadRuntime.expertCachePolicy,
            rdadvisePolicy: ProcessInfo.processInfo.environment["TINYTITAN_RDADVISE_POLICY"]
                .map(RDAdvicePolicyMode.parse)
                ?? loadRuntime.rdadvisePolicy,
            prefillChunkTokens: requestedPrefillChunkTokens
                ?? ModelProfile.resolve(
                    modelID: model.modelID, family: model.config.family,
                    weightBits: model.routedExpertWeightBits
                ).prefillChunkTokens.map {
                    RuntimeConfiguration.profilePrefillChunk($0, forContext: maxContext)
                }
                ?? defaultPrefillChunkTokens(
                    family: model.config.family,
                    fallback: loadRuntime.prefillChunkTokens),
            prefillAttentionPath: loadRuntime.prefillAttentionPath,
            forceLogitsHead: true,
            decodeExpertExecution: loadRuntime.decodeExpertExecution,
            expertIOSynchronization: loadRuntime.expertIOSynchronization,
            expertIOSubmission: loadRuntime.expertIOSubmission,
            kvCachePrecision: kvCachePrecision,
            ropeScalingMode: ropeScalingMode,
            yarnContextTokens: ropeScalingMode == .yarn
                ? maxContext : RuntimeConfiguration.defaultYaRNContextTokens)
        // MTP owns the target's runner and is single-sequence, so it cannot
        // batch. Otherwise the requested width is capped by what the worst-case
        // per-slot stores can hold beside the wired expert cache: the whole
        // point of the cap is that the cache cannot be paged out to rescue an
        // over-commit. The clamp is per load, so a catalog switch re-evaluates
        // it against the model actually being loaded.
        let requestedSlots = mtpModelDirectory == nil ? slots : 1
        let perSlotBytes = BatchedMemoryBudget.perSlotBytes(
            config: model.config,
            maxContext: maxContext,
            precision: runtime.kvCachePrecision,
            fp16RingEnabled: runtime.fp16RingEnabled,
            slidingWindow: model.config.slidingWindow,
            maxPrefillChunkTokens: runtime.prefillChunkTokens,
            vocab: model.config.vocabSize)
        let slotBudget = BatchedMemoryBudget.slotBudgetBytes(
            physicalMemory: ProcessInfo.processInfo.physicalMemory,
            // A dense family has no routed experts, so it has no expert cache to
            // hold back; see `expertCacheHeldBack`.
            expertCacheBudgetBytes: BatchedMemoryBudget.expertCacheHeldBack(
                numExperts: model.config.numExperts,
                configured: expertCacheBudgetBytes ?? tunedBudget))
        let effectiveSlots = BatchedMemoryBudget.effectiveSlots(
            requested: requestedSlots,
            perSlotBytes: perSlotBytes,
            budgetBytes: slotBudget)
        if effectiveSlots < requestedSlots {
            FileHandle.standardError.write(
                Data(
                    ("TinyTitan batch width \(requestedSlots) exceeds the memory budget "
                        + "(per-slot \(perSlotBytes / 1_048_576) MiB, budget "
                        + "\(slotBudget / 1_048_576) MiB); serving \(effectiveSlots) at once\n")
                        .utf8))
        }
        let mtpDecoder: StreamingMTPDecoder?
        let runner: RealForwardRunner
        if let mtpModelDirectory {
            let sidecarFamily = try ManifestReader.peekFamily(
                directoryURL: mtpModelDirectory)
            guard let sidecarArch = ArchConfig.knownArchitectures[sidecarFamily] else {
                throw ServerInferenceError.unsupportedModel(
                    "MTP sidecar declares family \(sidecarFamily.rawValue), "
                        + "which this runtime does not implement")
            }
            let sidecar = try Model.load(
                directoryURL: mtpModelDirectory,
                device: context.device,
                expecting: sidecarArch,
                streamingMode: .pread(slotCount: StreamingMTPMemoryPlan.expertSlots),
                expertCachePolicy: runtime.modelExpertCachePolicy,
                integrityPolicy: .resolved(directoryURL: mtpModelDirectory))
            let decoder = try StreamingMTPDecoder(
                targetModel: model,
                mtpSidecar: sidecar,
                context: context,
                maxContext: maxContext,
                memoryBudgetMiB: mtpMemoryMiB,
                runtimeConfiguration: runtime)
            mtpDecoder = decoder
            runner = decoder.target
        } else {
            mtpDecoder = nil
            runner = try RealForwardRunner(
                model: model,
                context: context,
                maxContext: maxContext,
                slots: effectiveSlots,
                runtimeConfiguration: runtime)
        }
        let scratches = try (0..<effectiveSlots).map { _ in
            try RawCompletionScratch(
                context: context, vocab: model.config.vocabSize,
                logitSoftcap: Float(model.config.finalLogitSoftcap))
        }
        let templateDigest = SHA256.hash(data: try Data(contentsOf: templateURL))
            .map { String(format: "%02x", $0) }
            .joined()
        let runtimeIdentity = [
            String(runtime.expertCacheSlots),
            runtime.expertCachePolicy.rawValue,
            runtime.rdadvisePolicy.rawValue,
            runtime.prefillPolicy.rawValue,
            String(runtime.prefillChunkTokens),
            runtime.headPath.rawValue,
            String(runtime.kvCachePrecision.rawValue),
            runtime.ropeScalingMode.rawValue,
            String(runtime.yarnContextTokens),
        ].joined(separator: ":")
        let runtimeDigest = SHA256.hash(data: Data(runtimeIdentity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        let promptCacheDomain = ServerPromptCacheDomain(
            modelID: model.modelID,
            sourceSnapshotHash: model.sourceSnapshotHash,
            runtimeProfileHash: runtimeDigest,
            maximumContext: maxContext,
            kvStorage: runtime.kvCachePrecision.label,
            fp16RingEnabled: runtime.fp16RingEnabled,
            templateSHA256: templateDigest)
        let effectivePromptCacheMode = Self.effectivePromptCacheMode(
            requested: promptCacheMode,
            mtpEnabled: mtpDecoder != nil,
            slots: effectiveSlots)
        let promptStateStore: ServerPromptStateStore?
        let promptCache: ServerPromptCache
        var frontierPrefixEntries: [ServerPromptCacheEntry] = []
        if effectivePromptCacheMode == .multiPrefix {
            let store = try ServerPromptStateStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: promptCacheMemoryLimitBytes,
                    diskDirectory: promptCacheDiskDirectory,
                    diskLimitBytes: promptCacheDiskLimitBytes,
                    memoryTTLSeconds: promptCacheMemoryTTLSeconds))
            let loaded = store.loadEntries(domain: promptCacheDomain)
            // Frontier checkpoints share the store but are matched by token
            // prefix, not message shape, so they stay out of the
            // ServerPromptCache trie and out of its entry count.
            frontierPrefixEntries = loaded.filter(Self.isFrontierEntry)
            if frontierPrefixEntries.count > Self.frontierMaximumEntries {
                store.remove(
                    entryIDs: frontierPrefixEntries
                        .dropLast(Self.frontierMaximumEntries)
                        .map(\.id))
                frontierPrefixEntries = Array(
                    frontierPrefixEntries.suffix(Self.frontierMaximumEntries))
            }
            let persisted = loaded.filter { !Self.isFrontierEntry($0) }
            if persisted.count > promptCacheMaximumEntries {
                store.remove(
                    entryIDs:
                        persisted
                        .dropLast(promptCacheMaximumEntries)
                        .map(\.id))
            }
            promptStateStore = store
            promptCache = ServerPromptCache(
                maximumEntries: promptCacheMaximumEntries,
                entries: persisted)
        } else {
            promptStateStore = nil
            promptCache = ServerPromptCache(maximumEntries: 1)
        }
        return ServerModelSession(
            context: context,
            model: model,
            tokenizer: tokenizer,
            tokenizerFolder: tokenizerFolder,
            loadedReasoning: RequestReasoning(
                thinkingMode: thinkingMode,
                effort: reasoningEffort),
            runner: runner,
            mtpDecoder: mtpDecoder,
            scratches: scratches,
            prefillConfig: runtime.prefillConfig,
            expertCacheSlots: loadSlots,
            slots: effectiveSlots,
            maxContext: maxContext,
            promptCacheMode: effectivePromptCacheMode,
            promptCacheDomain: promptCacheDomain,
            promptCache: promptCache,
            promptStateStore: promptStateStore,
            frontierPrefixEntries: frontierPrefixEntries,
            concisePrompt: conciseModeEnabled()
                ? ConcisePrompt.standard : nil)
    }

    private init(
        context: MetalContext,
        model: Model,
        tokenizer: GFTokenizer,
        tokenizerFolder: URL,
        loadedReasoning: RequestReasoning,
        runner: RealForwardRunner,
        mtpDecoder: StreamingMTPDecoder?,
        scratches: [RawCompletionScratch],
        prefillConfig: PrefillRuntimeConfig,
        expertCacheSlots: Int,
        slots: Int,
        maxContext: Int,
        promptCacheMode: ServerPromptCacheMode,
        promptCacheDomain: ServerPromptCacheDomain,
        promptCache: ServerPromptCache,
        promptStateStore: ServerPromptStateStore?,
        frontierPrefixEntries: [ServerPromptCacheEntry] = [],
        concisePrompt: String?
    ) {
        self.context = context
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizerFolder = tokenizerFolder
        self.loadedReasoning = loadedReasoning
        self.modelFamily = model.config.family
        self.profileSampling =
            ModelProfile.resolve(
                modelID: model.modelID, family: model.config.family,
                weightBits: model.routedExpertWeightBits
            ).sampling
        self.defaultModelID = ServerModelIdentity.apiModelID(
            manifestModelID: model.modelID,
            family: model.config.family,
            weightBits: model.routedExpertWeightBits)
        self.runner = runner
        self.mtpDecoder = mtpDecoder
        self.scratches = scratches
        self.slots = max(1, slots)
        self.freeSlots = Array(0..<max(1, slots))
        self.prefillConfig = prefillConfig
        self.prefillChunkTokens = prefillConfig.chunkTokens
        self.expertCacheSlots = expertCacheSlots
        self.maxContext = maxContext
        self.promptCacheMode = promptCacheMode
        self.promptCacheDomain = promptCacheDomain
        self.promptCache = promptCache
        self.promptStateStore = promptStateStore
        self.frontier = FrontierTracker(chunkTokens: prefillConfig.chunkTokens)
        // Oldest first, as the store listed them, so the cap drops those first.
        self.frontierEntries = frontierPrefixEntries.enumerated().map {
            FrontierEntry(tokens: $1.kvBackedTokenIDs, id: $1.id, lastUse: $0)
        }
        self.frontierClock = frontierPrefixEntries.count
        self.concisePrompt = concisePrompt
    }

    /// Take a slot for one generation. Actor-isolated, so the free list and the
    /// waiter queue never race; the coordinator's width normally keeps a slot
    /// free, and the wait exists only so a wider coordinator degrades to
    /// queueing instead of failing.
    private func acquireSlot() async throws -> Int {
        if let slot = freeSlots.popLast() { return slot }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                slotWaiters.append(SlotWaiter(id: id, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancelSlotWaiter(id) }
        }
        if Task.isCancelled { throw CancellationError() }
        guard let slot = freeSlots.popLast() else {
            throw ServerRequestError.queueFull
        }
        return slot
    }

    private func cancelSlotWaiter(_ id: UUID) {
        guard let index = slotWaiters.firstIndex(where: { $0.id == id }) else { return }
        slotWaiters.remove(at: index).continuation.resume(throwing: CancellationError())
    }

    private func releaseSlot(_ slot: Int) {
        freeSlots.append(slot)
        if !slotWaiters.isEmpty {
            slotWaiters.removeFirst().continuation.resume()
        }
    }

    /// TINYTITAN_CONCISE_MODE=1 (or "on") enables concise mode; the per-quant
    /// system prompt is then injected into every completion.
    private static func conciseModeEnabled() -> Bool {
        switch ProcessInfo.processInfo.environment["TINYTITAN_CONCISE_MODE"]?.lowercased() {
        case "1", "on", "true", "yes": return true
        default: return false
        }
    }

    /// Render a validated request into prompt tokens.
    ///
    /// TINYTITAN_STRIP_CLI_PROMPT: drop the coding-CLI's system/developer guidance,
    /// tool definitions, tool-call history, and in-message <system-reminder>
    /// scaffolding, keeping only the real user/assistant conversation (see
    /// CLIStrip). Guards ensure the real prompt can never be stripped into an
    /// empty turn or an empty request. Runs when the request names the
    /// "<model>-fast" alias or TINYTITAN_STRIP_CLI_PROMPT is set.
    ///
    /// Returns the encoded prompt alongside the `cacheRequest` — the post-strip
    /// view the prompt cache must key on. Cache entries describe a KV range
    /// prefilled from the filtered messages, and the cache's text-continuation
    /// path re-renders the tail with the same template, so matching or
    /// publishing against the raw request would splice an unstripped tail onto
    /// a stripped prefix, silently losing the "-fast" alias's strip on every
    /// cached continuation turn.
    ///
    /// `renderTokenizer` is the one this request's reasoning resolves to, so a
    /// mid-session switch renders through the right template instead of the
    /// one the model happened to load with.
    private func preparePrompt(
        _ request: ValidatedChatRequest,
        renderTokenizer: GFTokenizer
    ) throws -> (
        promptIDs: [Int32],
        cacheRequest: ValidatedChatRequest,
        needsToolTemplate: Bool
    ) {
        let filteredMessages: [GFTokenizer.Message]
        let filteredTools: [GFTokenizer.FunctionDefinition]
        var stripStats: CLIStrip.Stats?
        if request.stripCLIPrompt || CLIStrip.isEnabled() {
            let filtered = CLIStrip.filter(
                messages: request.messages,
                tools: request.tools)
            filteredMessages = filtered.messages
            filteredTools = filtered.tools
            stripStats = filtered.stats
        } else {
            filteredMessages = request.messages
            filteredTools = request.tools
        }
        let cacheRequest = request.replacingMessages(
            filteredMessages,
            tools: filteredTools)
        let needsToolTemplate = usesToolTemplate(
            messages: filteredMessages,
            tools: filteredTools)
        let effectiveMessages =
            concisePrompt.map {
                ConcisePrompt.appendingSystemPrompt($0, to: filteredMessages)
            } ?? filteredMessages
        let promptIDs = try encodePrompt(
            with: renderTokenizer,
            messages: effectiveMessages,
            tools: filteredTools,
            usesToolTemplate: needsToolTemplate)
        if let stats = stripStats {
            ServerLog.strip(
                stats: stats,
                promptTokens: promptIDs.count)
        }
        guard promptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }
        return (promptIDs, cacheRequest, needsToolTemplate)
    }

    /// Decide where this request's prefill starts: from scratch, or resumed on
    /// a cache entry whose KV is live or restorable.
    ///
    /// Mutates the cache and `activePromptCacheEntryID`, so it must run on the
    /// actor and before any generation begins.
    /// Whether a reasoning-level change forbids reusing any cached prefix.
    ///
    /// A request rendered at a different level than the session loaded at must
    /// not splice onto a cached KV range, and this is what makes the claim on
    /// `ValidatedChatRequest.reasoning` true.
    ///
    /// Comparing rendered token IDs is enough for the direct-prefix path, which
    /// is what that claim was written against: a level change renders different
    /// IDs, so it misses. It is *not* enough for a text continuation, which
    /// re-renders only the tail -- `matchTextContinuation` calls
    /// `applyChatTemplate` again -- and that used to happen with the session's
    /// tokenizer. The model then saw a generation prompt built for the loaded
    /// level while the decoder was built for the requested one, so either
    /// chain-of-thought leaked into `content` or the whole answer was reported
    /// as reasoning with `content` empty.
    ///
    /// Re-prefilling is the correct cost of a switch: the cached KV belongs to a
    /// different render, and the honest outcome is a miss.
    private func reasoningForbidsCacheReuse(_ requested: RequestReasoning?) -> Bool {
        guard let requested else { return false }
        return !requested.matches(loadedReasoning)
    }

    private func resolveCacheStart(
        cacheRequest: ValidatedChatRequest,
        promptIDs: [Int32],
        requestedReasoning: RequestReasoning?
    ) async throws -> (
        effectivePromptIDs: [Int32], start: RawCompletionStart, captureBoundaries: [Int]
    ) {
        if reasoningForbidsCacheReuse(requestedReasoning) {
            promptCache.invalidate()
            activePromptCacheEntryID = nil
            return (promptIDs, .reset, [])
        }
        let frontierOn = frontierCacheActive
        if frontierOn { frontier.observe(promptIDs) }
        var (effectivePromptIDs, completionStart) = try await messageShapedStart(
            cacheRequest: cacheRequest, promptIDs: promptIDs)
        // Token-prefix frontier fallback. The message-shaped cache above keys
        // on whole-message equality, so an edit to any earlier message (a
        // mutated system prompt) drops it to a full re-prefill. A checkpoint
        // taken at a chunk boundary on the shared frontier can still be an
        // exact token prefix of this render: restore it and resume there. It
        // only wins when it is deeper than what the message-shaped path found.
        if frontierOn,
            let restored = await restoreFrontierCheckpoint(
                promptIDs: promptIDs,
                beating: Self.resumePosition(completionStart))
        {
            effectivePromptIDs = promptIDs
            completionStart = .resume(cachedPromptTokens: restored)
        }
        // S12: an identical-prompt replay whose render equals the entry's
        // KV-backed prefix has nothing to prefill (cached == prompt count).
        // The continuation API requires cached < prompt count (it must
        // prefill at least one token), so resume as a full prefill; the
        // entry stays active for later extending requests.
        if case .resume(let cached) = completionStart,
            cached >= effectivePromptIDs.count
        {
            completionStart = .reset
        }
        guard effectivePromptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "effective prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }
        // A split prefill indexes the render by position, so it is only safe
        // when the runner prefills the render verbatim -- not the spliced
        // (KV-backed + bridge) array a message-shaped continuation hands it.
        let captureBoundaries =
            frontierOn && effectivePromptIDs == promptIDs
            ? frontier.captureTargets(
                render: promptIDs,
                resumeFrom: Self.resumePosition(completionStart),
                held: heldFrontierPositions(for: promptIDs))
            : []
        return (effectivePromptIDs, completionStart, captureBoundaries)
    }

    /// The message-shaped cache's decision: match on whole-message equality
    /// and, on a hit, resume the live KV or restore a persisted snapshot.
    /// Mutates `promptCache` and `activePromptCacheEntryID`.
    private func messageShapedStart(
        cacheRequest: ValidatedChatRequest,
        promptIDs: [Int32]
    ) async throws -> (effectivePromptIDs: [Int32], start: RawCompletionStart) {
        let effectivePromptIDs: [Int32]
        let completionStart: RawCompletionStart
        if promptCacheMode == .singlePrefix {
            switch promptCache.match(
                domain: promptCacheDomain,
                request: cacheRequest,
                renderedPromptIDs: promptIDs,
                tokenizer: tokenizer)
            {
            case .miss:
                promptCache.invalidate()
                effectivePromptIDs = promptIDs
                completionStart = .reset
            case .hit(_, let effective, let cached):
                if runner.continuationPosition != cached {
                    // S15: the live KV no longer sits at the cached entry's
                    // position; re-prefill instead of resuming from a stale
                    // or mismatched in-memory state.
                    promptCache.invalidate()
                    effectivePromptIDs = promptIDs
                    completionStart = .reset
                } else {
                    effectivePromptIDs = effective
                    completionStart = .resume(cachedPromptTokens: cached)
                }
            }
        } else if promptCacheMode == .multiPrefix {
            switch promptCache.match(
                domain: promptCacheDomain,
                request: cacheRequest,
                renderedPromptIDs: promptIDs,
                tokenizer: tokenizer)
            {
            case .miss:
                activePromptCacheEntryID = nil
                effectivePromptIDs = promptIDs
                completionStart = .reset
            case .hit(let entryID, let effective, let cached):
                if entryID == activePromptCacheEntryID,
                    runner.continuationPosition == cached
                {
                    // S15: tier=live is only trusted when the in-memory KV
                    // still matches the entry (same entry id and the KV
                    // cursor sits exactly at the request's expected
                    // position). Anything else falls through to a snapshot
                    // restore or a full prefill instead of resuming from a
                    // stale or mismatched KV.
                    print(
                        "TinyTitan prompt_cache hit tier=live "
                            + "cached_tokens=\(cached) entry=\(entryID.uuidString.lowercased())")
                } else {
                    do {
                        guard let promptStateStore else {
                            throw ServerPromptStateStoreError.missing(entryID)
                        }
                        let tier = try await promptStateStore.restore(
                            entryID: entryID,
                            into: runner)
                        print(
                            "TinyTitan prompt_cache hit tier=\(tier) "
                                + "cached_tokens=\(cached) entry=\(entryID.uuidString.lowercased())"
                        )
                    } catch {
                        // Drop the stale entry and prefill from scratch rather
                        // than trust it.
                        FileHandle.standardError.write(
                            Data(
                                ("TinyTitan prompt_cache restore_failed "
                                    + "entry=\(entryID.uuidString.lowercased()) error=\(error)\n")
                                    .utf8))
                        promptStateStore?.remove(entryIDs: [entryID])
                        promptCache.remove(entryIDs: [entryID])
                        activePromptCacheEntryID = nil
                        effectivePromptIDs = promptIDs
                        completionStart = .reset
                        break
                    }
                }
                activePromptCacheEntryID = entryID
                effectivePromptIDs = effective
                completionStart = .resume(cachedPromptTokens: cached)
            }
        } else {
            promptCache.invalidate()
            activePromptCacheEntryID = nil
            effectivePromptIDs = promptIDs
            completionStart = .reset
        }
        return (effectivePromptIDs, completionStart)
    }

    /// Whether this session places and restores frontier checkpoints: the
    /// multi-prefix store is in use *with an SSD tier*, and
    /// `TINYTITAN_FRONTIER_CACHE` does not switch it off. RAM alone is left to
    /// the message-shaped entries: at the default 256 MiB budget one long
    /// prompt's checkpoints would evict them on arrival.
    private var frontierCacheActive: Bool {
        guard let promptStateStore, promptStateStore.persistsToDisk else { return false }
        return Self.frontierCacheEnabled()
    }

    /// Off-switch for the token-prefix frontier checkpoints, matching the
    /// codebase's other opt-out experiments. Default on; `off`/`0`/`false`/`no`
    /// disables both capture and restore.
    static func frontierCacheEnabled(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        switch environment["TINYTITAN_FRONTIER_CACHE"]?.lowercased() {
        case "off", "0", "false", "no": return false
        default: return true
        }
    }

    /// Ceiling on one frontier checkpoint. A larger one is skipped rather than
    /// allocated; the set as a whole is bounded by the store's budgets.
    static let frontierCheckpointMaxBytes = 2 * 1_024 * 1_048_576
    /// Ceiling on how many frontier checkpoints are kept, least recently used
    /// dropped first, so they cannot crowd the message-shaped entries out of
    /// the store by count.
    static let frontierMaximumEntries = 16

    /// A persisted entry is a frontier checkpoint when it carries a pure token
    /// prefix and no message shape: no input messages, no uncommitted boundary,
    /// every KV row backed by a token. A chat entry always has a message.
    static func isFrontierEntry(_ entry: ServerPromptCacheEntry) -> Bool {
        entry.inputMessages.isEmpty
            && entry.uncommittedBoundaryTokenIDs.isEmpty
            && entry.kvPosition > 0
            && entry.kvPosition == entry.kvBackedTokenIDs.count
    }

    static func makeFrontierEntry(
        domain: ServerPromptCacheDomain, tokens: [Int32]
    ) -> ServerPromptCacheEntry {
        ServerPromptCacheEntry(
            id: UUID(),
            domain: domain,
            inputMessages: [],
            tools: [],
            assistantTurn: CachedAssistantTurn(
                message: GFTokenizer.Message(role: .assistant, content: ""),
                rawStopReason: .endOfTurn),
            kvBackedTokenIDs: tokens,
            uncommittedBoundaryTokenIDs: [],
            kvPosition: tokens.count)
    }

    private static func resumePosition(_ start: RawCompletionStart) -> Int {
        if case .resume(let count) = start { return count }
        return 0
    }

    /// Positions whose checkpoint for exactly this render's prefix exists.
    private func heldFrontierPositions(for promptIDs: [Int32]) -> Set<Int> {
        Set(
            frontierEntries.lazy
                .filter { $0.tokens.count <= promptIDs.count }
                .filter { promptIDs.prefix($0.tokens.count).elementsEqual($0.tokens) }
                .map(\.tokens.count))
    }

    /// Restore the deepest frontier checkpoint that is an exact prefix of this
    /// render and deeper than `cached`, and return its position; nil when
    /// there is none or the restore failed (the runner is then reset, so the
    /// request prefills from scratch).
    private func restoreFrontierCheckpoint(promptIDs: [Int32], beating cached: Int) async -> Int? {
        guard let store = promptStateStore else { return nil }
        // Rows the store evicted on its own are gone; forget them first.
        frontierEntries.removeAll { !store.contains($0.id) }
        let candidates = frontierEntries.indices
            .filter {
                let n = frontierEntries[$0].tokens.count
                return n > cached && n < promptIDs.count
                    && promptIDs.prefix(n).elementsEqual(frontierEntries[$0].tokens)
            }
            .sorted { frontierEntries[$0].tokens.count > frontierEntries[$1].tokens.count }
        guard let index = candidates.first else { return nil }
        let entry = frontierEntries[index]
        do {
            let tier = try await store.restore(entryID: entry.id, into: runner)
            if promptCacheMode == .singlePrefix { promptCache.invalidate() }
            activePromptCacheEntryID = nil
            frontierClock += 1
            if let row = frontierEntries.firstIndex(where: { $0.id == entry.id }) {
                frontierEntries[row].lastUse = frontierClock
            }
            print(
                "TinyTitan prompt_cache hit tier=frontier-\(tier) "
                    + "cached_tokens=\(entry.tokens.count)")
            return entry.tokens.count
        } catch {
            FileHandle.standardError.write(
                Data(
                    ("TinyTitan prompt_cache frontier_restore_failed "
                        + "entry=\(entry.id.uuidString.lowercased()) error=\(error)\n").utf8))
            store.remove(entryIDs: [entry.id])
            frontierEntries.removeAll { $0.id == entry.id }
            activePromptCacheEntryID = nil
            runner.reset()
            return nil
        }
    }

    /// Persist one chunk checkpoint, then register its row. Awaited (the write
    /// runs on the store's serial disk queue, off this actor) so a request
    /// that arrives right after cannot race a half-written snapshot. A row is
    /// registered only once its backing exists, and the least recently used
    /// rows past `frontierMaximumEntries` are dropped with their snapshots.
    private func persistFrontierCheckpoint(
        tokens: [Int32], snapshot: InferenceStateSnapshot
    ) async {
        guard let store = promptStateStore, !tokens.isEmpty,
            snapshot.descriptor.position == tokens.count,
            !frontierEntries.contains(where: { $0.tokens == tokens })
        else { return }
        let entry = Self.makeFrontierEntry(domain: promptCacheDomain, tokens: tokens)
        let saved = await store.save(entry: entry, snapshot: snapshot)
        if let diskError = saved.diskError {
            FileHandle.standardError.write(
                Data(("TinyTitan prompt_cache frontier disk_write_failed error=\(diskError)\n").utf8))
        }
        guard store.contains(entry.id) else { return }  // evicted on arrival
        frontierClock += 1
        frontierEntries.append(FrontierEntry(tokens: tokens, id: entry.id, lastUse: frontierClock))
        if frontierEntries.count > Self.frontierMaximumEntries {
            let overflow = frontierEntries
                .sorted { $0.lastUse < $1.lastUse }
                .prefix(frontierEntries.count - Self.frontierMaximumEntries)
                .map(\.id)
            store.remove(entryIDs: overflow)
            let dropped = Set(overflow)
            frontierEntries.removeAll { dropped.contains($0.id) }
        }
        print(
            "TinyTitan prompt_cache frontier_stored tokens=\(tokens.count) "
                + "state_bytes=\(snapshot.payload.count)")
    }

    /// lint:allow-long the request orchestrator: prompt preparation, cache
    /// resolution, decode, publish, and the completion. Each of those is its
    /// own method; what remains is the sequence plus a nested failure builder
    /// that closes over eight locals -- hoisting it would mean an
    /// eight-parameter signature for a twenty-line body.
    public func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        // One slot per in-flight generation. The coordinator bounds concurrency
        // to this session's width, so a slot is normally free immediately; the
        // wait is a safety net if the two ever disagree.
        let slot = try await acquireSlot()
        defer { releaseSlot(slot) }
        // Stage-split measurement (TINYTITAN_RUNNER_STATS): snapshot the runner's
        // lifetime counters so the footer can report this request's delta.
        let runnerSnapshot = RunnerCounterSnapshot(
            cb1: runner.totalCb1Nanos,
            io: runner.totalIoNanos,
            cb2: runner.totalCb2Nanos,
            head: runner.totalHeadNanos,
            headFused: runner.totalHeadFusedNanos,
            rdadvise: runner.totalRDAdviseNanos,
            rdadviseCalls: runner.totalRDAdviseCalls,
            rdadviseBytes: runner.totalRDAdviseBytes,
            wait: runner.totalWaitNanos,
            body: runner.totalBodyNanos,
            prefetchIssued: runner.totalPrefetchIssued,
            prefetchAdopted: runner.totalPrefetchAdopted,
            preamble: runner.totalPreambleNanos,
            preambleRelease: runner.totalPreambleReleaseNanos,
            preamblePin: runner.totalPreamblePinNanos,
            preambleReserve: runner.totalPreambleReserveNanos,
            embed: runner.totalEmbedNanos,
            gather: runner.totalGatherNanos,
            loopSample: runner.totalLoopSampleNanos,
            loopProgress: runner.totalLoopProgressNanos,
            loopOther: runner.totalLoopOtherNanos,
            missIo: runner.totalMissIoNanos,
            exposedIo: runner.totalExposedIoNanos,
            hitFixupLayers: runner.totalHitFixupLayers,
            routerReadback: runner.totalRouterReadbackNanos,
            cachePlan: runner.totalCachePlanNanos,
            ioQueue: runner.totalIOQueueNanos,
            ioCompletionToFixup: runner.totalIOCompletionToFixupSubmitNanos,
            ioHostWaits: runner.totalExpertIOHostWaits,
            ioHostWaitsAvoided: runner.totalExpertIOHostWaitsAvoided,
            gpuClassifiedHits: runner.totalGPUClassifiedHits,
            gpuClassifiedMisses: runner.totalGPUClassifiedMisses,
            gpuAllHitLayers: runner.totalGPUResidencyAllHitLayers,
            expertStreaming: runner.expertStreamingStatistics())
        runner.resetKernelGPUTimings()
        var completed = false
        defer {
            if !completed {
                if promptCacheMode == .singlePrefix {
                    promptCache.invalidate()
                }
                activePromptCacheEntryID = nil
                // One sequence failed; only its slot's KV/GDN is suspect. A
                // whole-runner reset would wipe the other slots' live state.
                if slots > 1 {
                    runner.reset(slot: slot)
                } else {
                    runner.reset()
                }
                mtpDecoder?.reset()
            }
        }
        // B6: an engine-internal generation is never watched. Everything
        // else gets the configured set, which is inert unless the operator
        // turned watchdogs on.
        let watchdogs =
            request.isEngineInternal
            ? WatchdogSupervisor.inert
            : WatchdogSupervisor(configuration: WatchdogConfiguration.shared)
        let watchdogTicker = watchdogs.startTicker()
        defer { watchdogTicker?.cancel() }
        // B2: a tool loop shows up in the incoming message history, not in
        // the output stream, so it is judged before anything is generated.
        if !request.isEngineInternal {
            watchdogs.record(
                pingPong: PingPongWatchdog.inspect(
                    request.messages, configuration: watchdogs.configuration))
        }
        // There is no safe intervention from here -- withholding the tools
        // leaves a tool-templated prompt with a decoder that allows none,
        // which fails the request outright. `WatchdogKind.canAct` carries the
        // reasoning; ping-pong observes, and the client, which owns the loop,
        // decides.
        // A request that names a different thinking mode or effort than the
        // session loaded at is a mid-session switch: resolve the tokenizer for
        // it here, so the render, the special tokens and the decoder all
        // follow the switch. `nil` (the common case) reuses the session's.
        let renderTokenizer = try await resolvedTokenizer(for: request.reasoning)
        let prepared = try preparePrompt(request, renderTokenizer: renderTokenizer)
        let promptIDs = prepared.promptIDs
        let cacheRequest = prepared.cacheRequest
        let needsToolTemplate = prepared.needsToolTemplate

        let resolved = try await resolveCacheStart(
            cacheRequest: cacheRequest,
            promptIDs: promptIDs,
            requestedReasoning: request.reasoning)
        let effectivePromptIDs = resolved.effectivePromptIDs
        let completionStart = resolved.start

        var config = request.generationConfig
        config.maxNewTokens = min(
            request.maximumCompletionTokens,
            maxContext - effectivePromptIDs.count)
        config.stopStrings = []
        // Structured output is a per-request grammar: a fresh constraint per
        // request (its state is the document parsed so far), over a table that
        // is built once per model.
        if let node = request.jsonSchema {
            config.constraint = JSONConstraint(
                table: structuredOutputTable(), node: node,
                vocab: model.config.vocabSize)
        }

        // The full render, not the cache-trimmed suffix, decides whether the
        // generation prompt left a thought open; both end in the same
        // generation prompt, but only the render is always whole. The decoder
        // runs for every generation, so a thought the *model* opens while the
        // switch is off is still split out of the answer rather than streamed
        // as it.
        let decoder = StructuredAssistantDecoder.forGeneration(
            tokenizer: renderTokenizer,
            promptIDs: promptIDs,
            allowedTools: needsToolTemplate ? Set(request.tools.map(\.name)) : nil)
        // The stall clock starts at the first visible token, so a long
        // thought before the answer cannot trip it. Reasoning is watched for
        // loops alone, in a window of its own.
        let state = GenerationDecodeState(
            decoder: decoder,
            output: AssistantOutput(
                stops: request.generationConfig.stopStrings,
                onEvent: onEvent,
                observeVisible: { watchdogs.observe($0) },
                observeReasoning: { watchdogs.observeReasoning($0) }))

        // MTP drafts several tokens ahead of the sampler and never consults a
        // grammar, so a constrained request takes the ordinary decode path
        // (`runRawCompletion` refuses the MTP producer outright).
        let activeProducer: any LogitProducer =
            if config.isPureGreedy,
                config.constraint == nil,
                let mtpDecoder,
                promptIDs.count + config.maxNewTokens
                    <= mtpDecoder.draftMaxContext
            {
                mtpDecoder
            } else {
                runner
            }
        let activeStart: RawCompletionStart =
            activeProducer is StreamingMTPDecoder
            ? .reset : completionStart
        let activePromptIDs =
            activeProducer is StreamingMTPDecoder
            ? promptIDs : effectivePromptIDs
        // Frontier checkpointing rides only the plain runner path.
        let captureBoundaries =
            activeProducer is StreamingMTPDecoder ? [] : resolved.captureBoundaries
        let progressGeneration = PrefillProgressMonitor.begin(
            total: activePromptIDs.count,
            cached: Self.resumePosition(activeStart))
        defer { PrefillProgressMonitor.end(generation: progressGeneration) }
        // `@Sendable`: `runRawCompletion` is @concurrent, so a progress closure
        // that is still actor-isolated cannot be sent into it (Swift 6.4).
        // Everything these touch lives in the Sendable box above.
        let publish: @Sendable ([StructuredAssistantEvent], Bool) -> Void = { events, isToken in
            state.output.publish(events, isToken: isToken)
            if state.output.isStopped { state.shouldStop = true }
        }
        // `renderTokenizer` is the one this request's reasoning resolves to, and
        // it is already what rendered the prompt and what the assistant decoder
        // was built with. `tokenizer` is the session's -- the level the model was
        // *loaded* at -- so a mid-session reasoning switch had the decoder on one
        // tokenizer and the detokenizer plus the stop-id check on another. Their
        // stop ids and special tokens coincide across a loaded folder today,
        // which is why this looked harmless; it is not guaranteed, and the
        // generation loop is the wrong place to rely on it.
        let result = try await runRawCompletion(
            producer: activeProducer,
            tokenizer: renderTokenizer,
            promptIds: activePromptIDs,
            config: config,
            context: context,
            scratch: scratches[slot],
            prefillConfig: prefillConfig,
            start: activeStart,
            slot: slot,
            captureBoundaries: captureBoundaries,
            captureMaxBytes: min(
                Self.frontierCheckpointMaxBytes,
                promptStateStore?.maximumSnapshotBytes ?? 0),
            // Always passed: with no boundaries `runRawCompletion` never calls
            // it. (A `cond ? nil : { ... }` here types the literal before the
            // annotation and loses `@Sendable`.)
            onCapture: { @Sendable (position: Int, snapshot: InferenceStateSnapshot) in
                state.frontierCaptures.append(
                    FrontierCapture(position: position, snapshot: snapshot))
            },
            // A watchdog stop is polled here, between tokens, alongside the
            // stop-string matcher's own flag.
            shouldStop: { @Sendable in state.shouldStop || watchdogs.wantsStop },
            onProgress: { @Sendable progress in
                if case .prefill(let done, let total) = progress {
                    PrefillProgressMonitor.prefill(
                        done: done, total: total, generation: progressGeneration)
                } else {
                    PrefillProgressMonitor.decoding(generation: progressGeneration)
                }
                guard state.decodingError == nil else { return }
                do {
                    switch progress {
                    case .prefill:
                        break
                    case .token(_, let tokenID, let delta):
                        publish(try state.decoder.consume(tokenID: tokenID, delta: delta), true)
                    case .tail(let text):
                        publish(try state.decoder.consumeTail(text), false)
                    }
                } catch {
                    state.decodingError = error
                    state.shouldStop = true
                }
            })
        // Bank the checkpoints captured mid-prefill. Each is a pure prefix
        // state, so it stands however the rest of this turn goes.
        let captures = state.frontierCaptures
        state.frontierCaptures = []
        for capture in captures {
            await persistFrontierCheckpoint(
                tokens: Array(promptIDs.prefix(capture.position)), snapshot: capture.snapshot)
        }
        emitGenerationDiagnostics(
            activeProducer: activeProducer,
            result: result,
            snapshot: runnerSnapshot)
        func structuredFailure(
            kind: StructuredOutputFailureKind,
            cause: StructuredOutputFailureCause
        ) -> StructuredOutputFailure {
            StructuredOutputFailure(
                kind: kind,
                cause: cause,
                diagnostics: StructuredOutputFailureDiagnostics(
                    renderedPromptIDs: promptIDs,
                    effectivePromptIDs: effectivePromptIDs,
                    result: result,
                    maxCompletionTokens: config.maxNewTokens,
                    decodedCalls: state.output.calls.count,
                    visibleBytes: state.output.content.utf8.count,
                    stopStringMatched: state.output.isStopped,
                    toolStartID: tokenizer.toolCallStartID,
                    toolEndID: tokenizer.toolCallEndID,
                    toolResponseID: tokenizer.toolResponseID,
                    toolResponseEndID: tokenizer.toolResponseEndID))
        }
        if let decodingError = state.decodingError {
            throw structuredFailure(
                kind: .decoderConsume,
                cause: .classify(decodingError))
        }
        do {
            try decoder.finish()
        } catch {
            throw structuredFailure(
                kind: .decoderFinish,
                cause: .classify(error))
        }
        if needsToolTemplate, result.reason == .toolCalls, state.output.calls.isEmpty {
            throw structuredFailure(kind: .orphanToolResponse, cause: .none)
        }
        state.output.finish()
        var content = state.output.content
        let calls = state.output.calls
        var reason: String
        if !calls.isEmpty {
            reason = "tool_calls"
        } else if result.reason == .maxTokens {
            reason = "length"
        } else {
            reason = "stop"
        }
        // The *last user message*, not the whole prompt: a long system
        // prompt in front of "hi" is still a short question, and an agent
        // harness puts a long system prompt in front of everything.
        let asked = request.messages.last { $0.role == .user }?.content?.utf8.count ?? 0
        watchdogs.finish(
            visibleBytes: content.utf8.count,
            requestBytes: asked,
            finishReason: reason)
        // B4: neither protocol has an honest reason for "the server stopped
        // this", and inventing one breaks clients. The mapping and the note
        // live in `WatchdogSet.resolve`, which is testable without a model.
        let outcome = watchdogs.resolve(content: content, finishReason: reason)
        // The cache entry must carry what was GENERATED. The note is written
        // by the server after the fact and has no tokens behind it in the KV
        // range, so publishing it would leave an entry whose text and KV
        // disagree, and a later continuation would splice the difference in.
        let generated = content
        if let note = outcome.note {
            content = outcome.content
            reason = outcome.finishReason
            onEvent(.content(note))
        }
        publishCacheEntry(
            cacheRequest: cacheRequest,
            content: generated,
            calls: calls,
            result: result,
            stopStringFiltered: state.output.isStopped)
        completed = true
        return ServerCompletion(
            content: content,
            toolCalls: calls,
            finishReason: reason,
            // S26: completion_tokens reports the number of GENERATED tokens,
            // matching OpenAI's "completion_tokens = tokens in the generated
            // completion". A stop-string-hidden suffix is therefore counted as
            // generated even though it is filtered from the visible content.
            usage: OpenAIUsage(
                promptTokens: result.prefillTokens,
                completionTokens: result.newTokens,
                totalTokens: result.prefillTokens + result.newTokens,
                cachedTokens: result.cachedPromptTokens,
                reasoningTokens: state.output.reasoningTokens),
            watchdogTrips: watchdogs.trips,
            stopSequence: state.output.matchedStop,
            reasoning: state.output.reasoning,
            // The render's mode, not the loaded session's: a request that
            // switched thinking off per request is the one whose thought is
            // unrequested.
            unrequestedReasoning: renderTokenizer.thinkingMode.isEnabled
                ? 0 : state.output.reasoning.count)
    }

    /// Publish this turn's KV range to the prompt cache, and persist a snapshot
    /// so a later request can resume from it without re-prefilling.
    ///
    /// Every failure path here degrades to "no cache entry" rather than to a
    /// broken one: an entry whose snapshot cannot be captured or verified is
    /// removed again, so the next hit re-prefills instead of attempting a
    /// doomed restore.
    private func publishCacheEntry(
        cacheRequest: ValidatedChatRequest,
        content: String,
        calls: [ParsedToolCall],
        result: RawDecodeResult,
        stopStringFiltered: Bool
    ) {
        if mtpDecoder != nil {
            // Native MTP keeps a second KV stream. Until both states are
            // persisted atomically, do not publish target-only cache entries.
            promptCache.invalidate()
            activePromptCacheEntryID = nil
        } else if promptCacheMode == .singlePrefix {
            let publication = promptCache.publish(
                domain: promptCacheDomain,
                request: cacheRequest,
                content: content,
                calls: calls,
                result: result,
                stopStringFiltered: stopStringFiltered)
            if publication == nil { promptCache.invalidate() }
        } else if promptCacheMode == .multiPrefix {
            let previousActive = activePromptCacheEntryID
            if let publication = promptCache.publish(
                domain: promptCacheDomain,
                request: cacheRequest,
                content: content,
                calls: calls,
                result: result,
                stopStringFiltered: stopStringFiltered)
            {
                promptStateStore?.remove(entryIDs: publication.evictedEntryIDs)
                do {
                    guard let promptStateStore else {
                        throw ServerPromptStateStoreError.missing(
                            publication.entry.id)
                    }
                    // S2: capture is bounded by the store's hard snapshot cap;
                    // the payload is a plain Data copy, so the disk write can
                    // proceed off the actor (dedicated store disk queue) while
                    // the next request starts. Concurrent saves serialize on
                    // the queue, so a later generation's snapshot can never
                    // clobber an in-flight write. The entry is already in the
                    // in-memory cache; a request that races the write simply
                    // misses and re-prefills (restore failure self-heals).
                    let snapshot = try runner.captureInferenceState(
                        maximumBytes: promptStateStore.maximumSnapshotBytes)
                    guard snapshot.descriptor.position == publication.entry.kvPosition else {
                        throw InferenceStateSnapshotError.invalidPosition(
                            snapshot.descriptor.position)
                    }
                    let entry = publication.entry
                    Task.detached(priority: .utility) { [promptStateStore] in
                        let saved = await promptStateStore.save(
                            entry: entry,
                            snapshot: snapshot)
                        if let diskError = saved.diskError {
                            FileHandle.standardError.write(
                                Data(
                                    ("TinyTitan prompt_cache disk_write_failed error=\(diskError)\n")
                                        .utf8))
                        }
                        print(
                            "TinyTitan prompt_cache stored "
                                + "tokens=\(entry.kvPosition) "
                                + "state_bytes=\(snapshot.payload.count) "
                                + "ram_bytes=\(saved.memoryBytes) "
                                + "disk_bytes=\(saved.diskBytes) "
                                + "entry=\(entry.id.uuidString.lowercased())")
                    }
                } catch {
                    // S24: a snapshot that cannot be captured or verified is
                    // never left published without backing; drop the entry so
                    // the next hit re-prefills instead of a doomed restore.
                    FileHandle.standardError.write(
                        Data(
                            ("TinyTitan prompt_cache snapshot_failed error=\(error)\n").utf8))
                    promptCache.remove(entryIDs: [publication.entry.id])
                    activePromptCacheEntryID = nil
                }
                if let previousActive,
                    previousActive != publication.entry.id,
                    promptStateStore?.contains(previousActive) != true
                {
                    promptCache.remove(entryIDs: [previousActive])
                }
                activePromptCacheEntryID = publication.entry.id
            } else {
                activePromptCacheEntryID = nil
            }
        }
    }

    private func usesToolTemplate(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition]
    ) -> Bool {
        Self.usesToolTemplate(messages: messages, tools: tools)
    }

    private static func usesToolTemplate(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition]
    ) -> Bool {
        !tools.isEmpty
            || messages.contains {
                $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
            }
    }

    /// Prompt tokens of a request as generation would render it — the same
    /// encoding, minus the generation.
    ///
    /// The tokenizer is resolved per request for the same reason `generate`
    /// resolves one: a request that names a different thinking mode or effort is
    /// rendered at that level, and an effort sentence is tens of tokens on a model
    /// that has levels. Counting with the session's tokenizer reported the loaded
    /// level's number for a request that would not be rendered at it.
    public func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int {
        try Self.promptTokenCount(
            request,
            tokenizer: try await resolvedTokenizer(for: request.reasoning),
            concisePrompt: concisePrompt)
    }

    /// The count from a tokenizer alone, which is how the router answers for a
    /// GPU model that is not the one loaded.
    ///
    /// `concisePrompt` is passed in rather than read because it is a property of a
    /// *loaded session*; the router's path has no session, so it counts without
    /// one and is the one case that can differ from what a concise-mode server
    /// would spend.
    static func promptTokenCount(
        _ request: ValidatedChatRequest,
        tokenizer: GFTokenizer,
        concisePrompt: String? = nil
    ) throws -> Int {
        // The same two transformations `preparePrompt` applies, in the same order.
        // This used to encode the request verbatim, so the count was inflated for
        // the `<model>-fast` alias and whenever `TINYTITAN_STRIP_CLI_PROMPT` is set,
        // and under-reported in concise mode — the opposite direction in each case,
        // which is why neither showed up as a single discrepancy.
        let filteredMessages: [GFTokenizer.Message]
        let filteredTools: [GFTokenizer.FunctionDefinition]
        if request.stripCLIPrompt || CLIStrip.isEnabled() {
            let filtered = CLIStrip.filter(
                messages: request.messages,
                tools: request.tools)
            filteredMessages = filtered.messages
            filteredTools = filtered.tools
        } else {
            filteredMessages = request.messages
            filteredTools = request.tools
        }
        let messages =
            concisePrompt.map {
                ConcisePrompt.appendingSystemPrompt($0, to: filteredMessages)
            } ?? filteredMessages
        return try encodePrompt(
            tokenizer: tokenizer, messages: messages, tools: filteredTools,
            usesToolTemplate: usesToolTemplate(
                messages: filteredMessages,
                tools: filteredTools)
        ).count
    }

    /// The vocabulary-as-bytes table, built on first use.
    private func structuredOutputTable() -> JSONTokenTable {
        if let jsonTokenTable { return jsonTokenTable }
        let table = JSONTokenTable(tokenizer: tokenizer)
        jsonTokenTable = table
        return table
    }

    /// The tokenizer this request should be rendered with.
    ///
    /// Almost every request takes the session's own, which keeps the render,
    /// the special tokens and the prompt cache exactly as they were. A request
    /// that named a different thinking mode or effort -- a mid-session switch,
    /// including turning thinking off -- gets a tokenizer for that
    /// configuration. The load coordinator caches by
    /// `(folder, thinking, effort)`, so this is a dictionary lookup once a
    /// level has been used, and a real load the first time.
    ///
    /// The returned tokenizer carries the think-block and stop token IDs for
    /// its own mode, so decode and the assistant decoder follow the switch
    /// rather than only the prompt text.
    private func resolvedTokenizer(
        for reasoning: RequestReasoning?
    ) async throws -> GFTokenizer {
        guard let reasoning, !reasoning.matches(loadedReasoning) else {
            return tokenizer
        }
        return try await GFTokenizer.load(
            from: tokenizerFolder,
            thinkingMode: reasoning.thinkingMode,
            reasoningEffort: reasoning.effort)
    }

    private func encodePrompt(
        with renderTokenizer: GFTokenizer,
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        usesToolTemplate: Bool
    ) throws -> [Int32] {
        try Self.encodePrompt(
            tokenizer: renderTokenizer, messages: messages, tools: tools,
            usesToolTemplate: usesToolTemplate)
    }

    private static func encodePrompt(
        tokenizer: GFTokenizer,
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        usesToolTemplate: Bool
    ) throws -> [Int32] {
        if usesToolTemplate {
            return try tokenizer.encodeToolChat(messages: messages, tools: tools)
        }
        let rendered = try tokenizer.applyChatTemplate(messages)
        return tokenizer.encode(rendered, addBOS: false)
    }

    /// Optional per-request diagnostics: MTP acceptance, the TINYTITAN_RUNNER_STATS
    /// stage split, and the TINYTITAN_KERNEL_STATS GPU breakdown. All three are
    /// env-gated and read-only, so they stay out of the generation path proper.
    private func emitGenerationDiagnostics(
        activeProducer: any LogitProducer,
        result: RawDecodeResult,
        snapshot runnerSnapshot: RunnerCounterSnapshot
    ) {
        if let activeMTP = activeProducer as? StreamingMTPDecoder {
            let stats = activeMTP.statistics
            let decodeRate =
                result.decodeSeconds > 0
                ? Double(result.newTokens) / result.decodeSeconds : 0
            print(
                String(
                    format:
                        "TinyTitan mtp drafted=%d accepted=%d acceptance=%.1f%% "
                        + "target_passes=%d emitted_per_pass=%.3f "
                        + "prefill_s=%.3f decode_s=%.3f decode_tok_s=%.3f "
                        + "memory_required_mib=%.1f memory_budget_mib=%.1f",
                    stats.draftedTokens,
                    stats.acceptedTokens,
                    stats.acceptanceRate * 100,
                    stats.targetBackbonePasses,
                    stats.emittedTokensPerTargetPass,
                    result.prefillSeconds,
                    result.decodeSeconds,
                    decodeRate,
                    Double(activeMTP.memoryPlan.requiredBytes) / 1_048_576,
                    Double(activeMTP.memoryPlan.budgetBytes) / 1_048_576))
            if ProcessInfo.processInfo.environment["TINYTITAN_RUNNER_STATS"] != nil,
                stats.targetBackbonePasses > 0
            {
                // Per-pass phase attribution for the Track B1 investigation:
                // where a verify pass's wall time actually goes. Milliseconds
                // averaged over the request's target passes.
                let passes = Double(stats.targetBackbonePasses)
                let ms: (UInt64) -> Double = { Double($0) / passes / 1_000_000 }
                print(
                    String(
                        format:
                            "TinyTitan mtp-phases per_pass_ms proposal=%.3f checkpoint=%.3f "
                            + "verify=%.3f verify_backbone=%.3f verify_head=%.3f "
                            + "verify_argmax=%.3f commit=%.3f rollback=%.3f passes=%d",
                        ms(stats.proposalNanos),
                        ms(stats.checkpointNanos),
                        ms(stats.verifyNanos),
                        ms(stats.verifyBackboneNanos),
                        ms(stats.verifyHeadNanos),
                        ms(stats.verifyArgmaxNanos),
                        ms(stats.commitNanos),
                        ms(stats.rollbackNanos),
                        stats.targetBackbonePasses))
            }
        } else {
            let decodeRate =
                result.decodeSeconds > 0
                ? Double(result.newTokens) / result.decodeSeconds : 0
            print(
                String(
                    format:
                        "TinyTitan generation prefill_s=%.3f decode_s=%.3f decode_tok_s=%.3f",
                    result.prefillSeconds,
                    result.decodeSeconds,
                    decodeRate))
        }
        if ProcessInfo.processInfo.environment["TINYTITAN_RUNNER_STATS"] != nil {
            emitRunnerDiagnostics(result: result, snapshot: runnerSnapshot)
        }
        if ProcessInfo.processInfo.environment["TINYTITAN_KERNEL_STATS"] != nil {
            emitKernelDiagnostics(result: result)
        }
    }

    private func emitRunnerDiagnostics(
        result: RawDecodeResult,
        snapshot: RunnerCounterSnapshot
    ) {
        let tokens = max(1, result.newTokens)
        let ms: (UInt64, UInt64) -> Double = { delta, base in
            Double(delta > base ? delta - base : 0) / Double(tokens) / 1_000_000
        }
        let missIoNanos = runner.totalMissIoNanos - snapshot.missIo
        let exposedIoNanos = runner.totalExposedIoNanos - snapshot.exposedIo
        let hiddenPercent =
            missIoNanos == 0
            ? 100.0
            : 100 * (1 - Double(exposedIoNanos) / Double(missIoNanos))
        let expert = runner.expertStreamingStatistics()
            .subtracting(snapshot.expertStreaming)
        let gpuHits = runner.totalGPUClassifiedHits - snapshot.gpuClassifiedHits
        let gpuMisses = runner.totalGPUClassifiedMisses - snapshot.gpuClassifiedMisses
        let gpuAllHit = runner.totalGPUResidencyAllHitLayers - snapshot.gpuAllHitLayers
        print(
            String(
                format: "TinyTitan runner cb1_ms=%.3f io_ms=%.3f cb2_ms=%.3f "
                    + "head_ms=%.3f head_fused_ms=%.3f rdadvise_ms=%.3f "
                    + "wait_ms=%.3f body_ms=%.3f rdadvise_calls=%llu rdadvise_mib=%.1f "
                    + "expert_hit_rate=%.4f expert_hits=%llu expert_misses=%llu "
                    + "expert_evictions=%llu expert_reloads=%llu expert_read_mib=%.1f "
                    + "expert_load_p50_ms=%.3f expert_load_p95_ms=%.3f "
                    + "expert_load_p99_ms=%.3f io_hidden_pct=%.2f hit_fixup_layers=%llu "
                    + "router_readback_ms=%.4f cache_plan_ms=%.4f io_queue_ms=%.4f "
                    + "io_completion_to_fixup_ms=%.4f io_host_waits=%llu "
                    + "io_host_waits_avoided=%llu gpu_classified_hits=%llu "
                    + "gpu_classified_misses=%llu gpu_all_hit_layers=%llu "
                    + "prefetch_issued_per_token=%.2f prefetch_adopted_per_token=%.2f "
                    + "pre_ms=%.3f pre_release_ms=%.3f pre_pin_ms=%.3f pre_reserve_ms=%.3f "
                    + "embed_ms=%.3f gather_ms=%.3f loop_sample_ms=%.3f "
                    + "loop_progress_ms=%.3f loop_other_ms=%.3f",
                ms(runner.totalCb1Nanos, snapshot.cb1),
                ms(runner.totalIoNanos, snapshot.io),
                ms(runner.totalCb2Nanos, snapshot.cb2),
                ms(runner.totalHeadNanos, snapshot.head),
                ms(runner.totalHeadFusedNanos, snapshot.headFused),
                ms(runner.totalRDAdviseNanos, snapshot.rdadvise),
                ms(runner.totalWaitNanos, snapshot.wait),
                ms(runner.totalBodyNanos, snapshot.body),
                runner.totalRDAdviseCalls - snapshot.rdadviseCalls,
                Double(runner.totalRDAdviseBytes - snapshot.rdadviseBytes) / 1_048_576,
                expert.hitRate, expert.hits, expert.misses, expert.evictions,
                expert.reloads, Double(expert.bytesRead) / 1_048_576,
                Double(expert.loadLatencyPercentile(0.50)) / 1_000_000,
                Double(expert.loadLatencyPercentile(0.95)) / 1_000_000,
                Double(expert.loadLatencyPercentile(0.99)) / 1_000_000,
                hiddenPercent, runner.totalHitFixupLayers - snapshot.hitFixupLayers,
                ms(runner.totalRouterReadbackNanos, snapshot.routerReadback),
                ms(runner.totalCachePlanNanos, snapshot.cachePlan),
                ms(runner.totalIOQueueNanos, snapshot.ioQueue),
                ms(runner.totalIOCompletionToFixupSubmitNanos, snapshot.ioCompletionToFixup),
                runner.totalExpertIOHostWaits - snapshot.ioHostWaits,
                runner.totalExpertIOHostWaitsAvoided - snapshot.ioHostWaitsAvoided,
                gpuHits, gpuMisses, gpuAllHit,
                Double(runner.totalPrefetchIssued &- snapshot.prefetchIssued) / Double(tokens),
                Double(runner.totalPrefetchAdopted &- snapshot.prefetchAdopted) / Double(tokens),
                ms(runner.totalPreambleNanos, snapshot.preamble),
                ms(runner.totalPreambleReleaseNanos, snapshot.preambleRelease),
                ms(runner.totalPreamblePinNanos, snapshot.preamblePin),
                ms(runner.totalPreambleReserveNanos, snapshot.preambleReserve),
                ms(runner.totalEmbedNanos, snapshot.embed),
                ms(runner.totalGatherNanos, snapshot.gather),
                ms(runner.totalLoopSampleNanos, snapshot.loopSample),
                ms(runner.totalLoopProgressNanos, snapshot.loopProgress),
                ms(runner.totalLoopOtherNanos, snapshot.loopOther)))
        if let ring = runner.prefetchRingSummary { print("TinyTitan \(ring)") }
    }

    private func emitKernelDiagnostics(result: RawDecodeResult) {
        let tokens = max(1, result.newTokens)
        let summary = runner.kernelGPUTimingSummary()
        let totalGPU = summary.reduce(0) { $0 + $1.millis }
        for entry in summary {
            print(
                String(
                    format: "TinyTitan kernel role=%@ gpu_ms=%.3f per_token_ms=%.3f count=%d",
                    entry.role, entry.millis, entry.millis / Double(tokens), entry.count))
        }
        // Role sums overlap by design. Merged busy/span is the actual queue
        // occupancy and distinguishes useful concurrency from idle gaps.
        let occupancy = runner.kernelGPUOccupancy()
        print(
            String(
                format: "TinyTitan kernel total_gpu_ms=%.3f gpu_share_of_decode=%.1f%%",
                totalGPU,
                result.decodeSeconds > 0
                    ? totalGPU / (result.decodeSeconds * 1000) * 100 : 0))
        for gap in runner.kernelGPUGaps().prefix(8) {
            print(
                String(
                    format: "TinyTitan gap %@ total_ms=%.1f per_token_ms=%.3f count=%d",
                    gap.transition, gap.millis, gap.millis / Double(tokens), gap.count))
        }
        print(
            String(
                format: "TinyTitan kernel busy_ms=%.3f span_ms=%.3f "
                    + "occupancy=%.1f%% busy_share_of_decode=%.1f%% busy_per_token_ms=%.3f",
                occupancy.busyMillis, occupancy.spanMillis,
                occupancy.spanMillis > 0
                    ? occupancy.busyMillis / occupancy.spanMillis * 100 : 0,
                result.decodeSeconds > 0
                    ? occupancy.busyMillis / (result.decodeSeconds * 1000) * 100 : 0,
                occupancy.busyMillis / Double(tokens)))
    }
}
