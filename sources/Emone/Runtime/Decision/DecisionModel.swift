import Accelerate
import Foundation
import Metal

/// A System One decision model (Kev: github.com/jaredpalmer/kev): a dense
/// Qwen 3.5 backbone whose LoRA was merged at install time, plus a pointer
/// head that scores a question's options from final-normed hidden states.
/// It never samples a token, so it loads without the chat tokenizer and never
/// runs the LM head.
///
/// One request is a *state* and some *questions*. The state is prefilled once
/// and becomes a branch point; each question then runs as its own causal row
/// continuing it -- `[<q> instruction (<opt> option </opt>)... <decide>]` --
/// and the head compares each `</opt>` row with the `<decide>` row. That is
/// Kev's row form, the only one a Gated-DeltaNet backbone can run (its
/// recurrence cannot honour a block-causal mask), and its prefix cache: the
/// last state stays resident, so a second request about the same state pays
/// only for its questions.
///
/// Not thread-safe: one request at a time, which the server's host enforces.
///
/// unchecked-invariant: every mutable member (the runner's sequence, the cached
/// state and branch point, the scratch buffer) is touched only inside `score`,
/// and the owner serialises calls to it (`DecisionModelHost`'s FIFO gate).
public final class DecisionModel: @unchecked Sendable {
    public let directory: URL
    public let head: PointerHead
    public let delimiters: Delimiters
    public let hiddenSize: Int
    public let maxContext: Int
    let model: Model
    let runner: RealForwardRunner
    let prefillConfig: PrefillRuntimeConfig
    private var cachedState: [Int32]?
    private var cachedPoint: DecisionBranchPoint?

    /// The five delimiter token ids, in Kev's order.
    public struct Delimiters: Sendable, Equatable {
        public let state, question, optionOpen, optionClose, decide: Int32
    }

    /// One question as token ids: the instruction and each option's text,
    /// already tokenized with the delimiters escaped (the caller's job: the
    /// ids must never contain a delimiter, or an option boundary is forged).
    public struct Question: Sendable, Equatable {
        public let instruction: [Int32]
        public let options: [[Int32]]
        public init(instruction: [Int32], options: [[Int32]]) {
            self.instruction = instruction
            self.options = options
        }
    }

    public struct Answer: Sendable, Equatable {
        public let logits: [Float]
        public let probabilities: [Float]
    }

    /// Where the time went, for the server log and the benchmark.
    public struct Timing: Sendable, Equatable {
        public var statePrefillNanos: UInt64 = 0
        public var stateReused = false
        public var branchNanos: [UInt64] = []
        public var restoreNanos: UInt64 = 0
        public var headNanos: UInt64 = 0
    }

    public enum LoadError: Error, CustomStringConvertible {
        case notADecisionModel(String)
        case badHead(String)
        public var description: String {
            switch self {
            case .notADecisionModel(let reason), .badHead(let reason): reason
            }
        }
    }

    public static let headFile = "decision_head.safetensors"

    /// Loads the install at `directory`: backbone, runner and head. The
    /// install receipt is verified like any other model's.
    public static func load(
        directory: URL, context: MetalContext, maxContext: Int = 16_384,
        prefillChunkTokens: Int = 4_096
    ) throws -> DecisionModel {
        let headURL = directory.appendingPathComponent(headFile)
        guard FileManager.default.fileExists(atPath: headURL.path) else {
            throw LoadError.notADecisionModel("\(directory.path) has no \(headFile)")
        }
        let identity = try ManifestReader.peekIdentity(directoryURL: directory)
        guard identity.family == .qwen35Dense else {
            throw LoadError.notADecisionModel(
                "decision models are dense Qwen 3.5; this is \(identity.family.rawValue)")
        }
        let arch = try ArchConfig.resolved(forFamily: identity.family, directoryURL: directory)
        let model = try Model.load(
            directoryURL: directory, device: context.device, expecting: arch,
            integrityPolicy: .resolved(directoryURL: directory))
        let head = try PointerHead(contentsOf: headURL, hiddenSize: model.config.hiddenSize)
        let delimiters = try Self.delimiterIDs(
            names: head.delimiterNames,
            tokenizer: directory.appendingPathComponent("tokenizer/tokenizer.json"))
        let runtime = try RuntimeConfiguration(prefillChunkTokens: prefillChunkTokens)
        try runtime.validate(maxContext: maxContext)
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: maxContext, runtimeConfiguration: runtime)
        if let refusal = runner.decisionReadoutRefusal {
            throw LoadError.notADecisionModel(refusal)
        }
        return DecisionModel(
            directory: directory, head: head, delimiters: delimiters, model: model,
            runner: runner, prefillConfig: runtime.prefillConfig, maxContext: maxContext)
    }

    /// A decision adapter package (`LoRAAdapter`) laid over a base install at
    /// request time (Emone-lmw.11): the base's weights come from
    /// `ModelWeightsRegistry`, so they are the side chat model's own when it is
    /// loaded, and this scorer gets its own runner, KV and Gated-DeltaNet
    /// state, the adapter's measured profile row and the adapter's delta.
    public static func load(
        adapter adapterDirectory: URL, base baseDirectory: URL, context: MetalContext,
        maxContext: Int = 16_384, prefillChunkTokens: Int = 4_096
    ) throws -> DecisionModel {
        let identity = try ManifestReader.peekIdentity(directoryURL: baseDirectory)
        guard identity.family == .qwen35Dense else {
            throw LoadError.notADecisionModel(
                "decision adapters lay over dense Qwen 3.5; this base is \(identity.family.rawValue)"
            )
        }
        let weights = try ModelWeightsRegistry.shared.weights(directory: baseDirectory) {
            let arch = try ArchConfig.resolved(
                forFamily: identity.family, directoryURL: baseDirectory)
            return try Model.load(
                directoryURL: baseDirectory, device: context.device, expecting: arch,
                integrityPolicy: .resolved(directoryURL: baseDirectory))
        }
        let model = weights.model
        let runtime = try RuntimeConfiguration(prefillChunkTokens: prefillChunkTokens)
        try runtime.validate(maxContext: maxContext)
        let adapter = try LoRAAdapter(
            directory: adapterDirectory, baseModelID: model.modelID, device: context.device,
            maxTokens: runtime.prefillConfig.chunkTokens)
        let head = try PointerHead(
            contentsOf: adapterDirectory.appendingPathComponent(LoRAAdapter.headFile),
            hiddenSize: model.config.hiddenSize)
        let delimiters = try Self.delimiterIDs(
            names: head.delimiterNames,
            tokenizer: baseDirectory.appendingPathComponent("tokenizer/tokenizer.json"))
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: maxContext, runtimeConfiguration: runtime,
            profileModelID: adapter.modelID, sharedWeights: weights)
        if let refusal = runner.decisionReadoutRefusal {
            throw LoadError.notADecisionModel(refusal)
        }
        try runner.setLoRAAdapter(adapter)
        return DecisionModel(
            directory: adapterDirectory, head: head, delimiters: delimiters, model: model,
            runner: runner, prefillConfig: runtime.prefillConfig, maxContext: maxContext)
    }

    private init(
        directory: URL, head: PointerHead, delimiters: Delimiters, model: Model,
        runner: RealForwardRunner, prefillConfig: PrefillRuntimeConfig, maxContext: Int
    ) {
        self.directory = directory
        self.head = head
        self.delimiters = delimiters
        self.model = model
        self.runner = runner
        self.prefillConfig = prefillConfig
        self.hiddenSize = model.config.hiddenSize
        self.maxContext = maxContext
    }

    /// The delimiter names' ids, read from the tokenizer's added tokens -- the
    /// one place they are defined -- rather than through the chat tokenizer,
    /// which a Base checkpoint's vocabulary cannot satisfy.
    static func delimiterIDs(names: [String], tokenizer: URL) throws -> Delimiters {
        guard names.count == 5 else {
            throw LoadError.badHead("expected five delimiters, got \(names.count)")
        }
        let data = try Data(contentsOf: tokenizer)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let added = json["added_tokens"] as? [[String: Any]]
        else { throw LoadError.badHead("tokenizer.json has no added_tokens") }
        var ids: [String: Int32] = [:]
        for entry in added {
            if let content = entry["content"] as? String, let id = entry["id"] as? Int {
                ids[content] = Int32(id)
            }
        }
        let resolved = try names.map { name -> Int32 in
            guard let id = ids[name] else {
                throw LoadError.badHead("delimiter \(name) is not a tokenizer special token")
            }
            return id
        }
        return Delimiters(
            state: resolved[0], question: resolved[1], optionOpen: resolved[2],
            optionClose: resolved[3], decide: resolved[4])
    }

    /// Per-role GPU milliseconds since the last call, when the process runs
    /// with EMONE_KERNEL_STATS (empty otherwise); resets the tally.
    public func drainKernelGPUSummary() -> [(role: String, millis: Double, count: Int)] {
        defer { runner.resetKernelGPUTimings() }
        return runner.kernelGPUTimingSummary()
    }

    /// The state row Kev encodes: `<state>` then the state's tokens.
    public func stateRow(_ state: [Int32]) -> [Int32] { Self.stateRow(state, delimiters) }

    public func branchRow(_ question: Question) -> (ids: [Int32], decide: Int, options: [Int]) {
        Self.branchRow(question, delimiters)
    }

    static func stateRow(_ state: [Int32], _ d: Delimiters) -> [Int32] { [d.state] + state }

    /// One question's branch row and its readout offsets within the row:
    /// `<decide>` last, and each option's `</opt>` (kev/model.py `encode`).
    static func branchRow(_ question: Question, _ d: Delimiters) -> (
        ids: [Int32], decide: Int, options: [Int]
    ) {
        var ids = [d.question] + question.instruction
        var closes: [Int] = []
        for option in question.options {
            ids.append(d.optionOpen)
            ids += option
            ids.append(d.optionClose)
            closes.append(ids.count - 1)
        }
        ids.append(d.decide)
        return (ids, ids.count - 1, closes)
    }

    /// Scores every question against `state` (token ids, delimiters escaped).
    public func score(state: [Int32], questions: [Question]) async throws -> (
        answers: [Answer], timing: Timing
    ) {
        var timing = Timing()
        let row = stateRow(state)
        let branches = questions.map(branchRow)
        let longest = branches.map(\.ids.count).max() ?? 0
        guard row.count + longest <= maxContext else {
            throw DecisionReadoutError.unsupported(
                "state of \(row.count) tokens plus a \(longest)-token question exceeds \(maxContext)"
            )
        }
        guard branches.allSatisfy({ !$0.options.isEmpty }) else {
            throw DecisionReadoutError.unsupported("every question needs at least one option")
        }
        let point = try await prefillState(row, timing: &timing)
        var answers: [Answer] = []
        answers.reserveCapacity(branches.count)
        // The state prefill leaves the sequence at the point already, so the
        // first branch needs no restore; every later one returns to it.
        var atPoint = !timing.stateReused
        for branch in branches {
            if !atPoint {
                let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                try await runner.restoreDecisionBranchPoint(point)
                timing.restoreNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
            }
            atPoint = false
            let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let rows = try await readout(
                tokens: branch.ids, from: point.position,
                offsets: [branch.decide] + branch.options)
            timing.branchNanos.append(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
            let headStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let logits = head.logits(decide: rows[0], options: Array(rows.dropFirst()))
            answers.append(Answer(logits: logits, probabilities: PointerHead.softmax(logits)))
            timing.headNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - headStarted
        }
        return (answers, timing)
    }

    /// The branch point for `row`: the resident one when the state is the
    /// cached state (restored on first use), otherwise a fresh prefill.
    private func prefillState(_ row: [Int32], timing: inout Timing) async throws
        -> DecisionBranchPoint
    {
        if let cachedState, let cachedPoint, cachedState == row {
            timing.stateReused = true
            return cachedPoint
        }
        cachedState = nil
        cachedPoint = nil
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        await runner.resetSequence(slot: 0)
        // Read the last state row: the readout's only job here is to stand in
        // for the LM head, which would otherwise project 248k logits nobody
        // reads.
        _ = try await readout(tokens: row, from: 0, offsets: [row.count - 1])
        let point = try await runner.makeDecisionBranchPoint()
        timing.statePrefillNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
        cachedState = row
        cachedPoint = point
        return point
    }

    /// Prefills `tokens` at `from` and returns the final-normed rows at
    /// `offsets` (relative to `from`).
    private func readout(tokens: [Int32], from position: Int, offsets: [Int]) async throws
        -> [[Float]]
    {
        try runner.armDecisionReadout(positions: offsets.map { position + $0 })
        do {
            _ = try await runner.prefillChunked(
                tokens: tokens[...], startPosition: position, outputMode: .logits,
                config: prefillConfig, into: try readoutScratch(), onProgress: { _ in })
            return try runner.finishDecisionReadout()
        } catch {
            runner.disarmDecisionReadout()
            // A failed prefill leaves the sequence mid-row; the cached point
            // is still intact, but drop it so nothing trusts a half state.
            cachedState = nil
            cachedPoint = nil
            throw error
        }
    }

    private var unusedLogits: MTLBuffer?
    /// Prefill's logits destination, never written while a readout is armed.
    private func readoutScratch() throws -> MTLBuffer {
        if let unusedLogits { return unusedLogits }
        guard let buffer = runner.ctx.device.makeBuffer(length: 64, options: .storageModeShared)
        else { throw ModelError.residentBufferWrapFailed }
        unusedLogits = buffer
        return buffer
    }
}

/// Kev's pointer head: `logit_i = (Wk h_i + bk) . (Wq h_decide + bq) / sqrt(d)
/// / T`, over the `</opt>` rows h_i. Float32 throughout, as Kev scores.
public struct PointerHead: Sendable {
    public let hiddenSize: Int
    public let headDim: Int
    public let temperature: Float
    public let delimiterNames: [String]
    let qWeight: [Float]
    let qBias: [Float]
    let kWeight: [Float]
    let kBias: [Float]

    public init(contentsOf url: URL, hiddenSize: Int) throws {
        let data = try Data(contentsOf: url)
        guard data.count >= 8 else { throw DecisionModel.LoadError.badHead("head file too short") }
        let length = data.withUnsafeBytes {
            Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)))
        }
        guard length > 0, 8 + length <= data.count,
            let header = try JSONSerialization.jsonObject(with: data[8..<(8 + length)])
                as? [String: Any],
            let meta = header["__metadata__"] as? [String: String]
        else { throw DecisionModel.LoadError.badHead("unreadable safetensors header") }
        guard meta["format"] == "kev-pointer-v1",
            let temperature = meta["temperature"].flatMap(Float.init), temperature > 0,
            let headDim = meta["head_dim"].flatMap(Int.init), headDim > 0,
            let delimiterJSON = meta["delimiters"]?.data(using: .utf8),
            let delimiters = try JSONSerialization.jsonObject(with: delimiterJSON) as? [String]
        else { throw DecisionModel.LoadError.badHead("head metadata is not kev-pointer-v1") }
        let payload = data[(8 + length)...]
        func tensor(_ name: String, _ count: Int) throws -> [Float] {
            guard let entry = header[name] as? [String: Any],
                entry["dtype"] as? String == "F32",
                let offsets = entry["data_offsets"] as? [Int], offsets.count == 2,
                offsets[1] - offsets[0] == count * 4,
                payload.startIndex + offsets[1] <= payload.endIndex
            else {
                throw DecisionModel.LoadError.badHead(
                    "\(name) is missing or not \(count) F32 values")
            }
            let start = payload.startIndex + offsets[0]
            return payload[start..<(start + count * 4)].withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
        }
        self.hiddenSize = hiddenSize
        self.headDim = headDim
        self.temperature = temperature
        self.delimiterNames = delimiters
        self.qWeight = try tensor("q.weight", headDim * hiddenSize)
        self.qBias = try tensor("q.bias", headDim)
        self.kWeight = try tensor("k.weight", headDim * hiddenSize)
        self.kBias = try tensor("k.bias", headDim)
    }

    /// For tests: a head from explicit weights.
    init(
        hiddenSize: Int, headDim: Int, temperature: Float, qWeight: [Float], qBias: [Float],
        kWeight: [Float], kBias: [Float]
    ) {
        self.hiddenSize = hiddenSize
        self.headDim = headDim
        self.temperature = temperature
        self.delimiterNames = []
        self.qWeight = qWeight
        self.qBias = qBias
        self.kWeight = kWeight
        self.kBias = kBias
    }

    /// `y = W x + b` for a [headDim x hiddenSize] row-major W.
    private func project(_ weight: [Float], _ bias: [Float], _ x: [Float]) -> [Float] {
        var y = [Float](repeating: 0, count: headDim)
        // [headDim x hiddenSize] . [hiddenSize x 1]
        vDSP_mmul(
            weight, 1, x, 1, &y, 1, vDSP_Length(headDim), 1, vDSP_Length(hiddenSize))
        vDSP_vadd(y, 1, bias, 1, &y, 1, vDSP_Length(headDim))
        return y
    }

    public func logits(decide: [Float], options: [[Float]]) -> [Float] {
        let q = project(qWeight, qBias, decide)
        let scale = 1 / (Float(headDim).squareRoot() * temperature)
        return options.map { option in
            let k = project(kWeight, kBias, option)
            var dot: Float = 0
            vDSP_dotpr(k, 1, q, 1, &dot, vDSP_Length(headDim))
            return dot * scale
        }
    }

    public static func softmax(_ logits: [Float]) -> [Float] {
        guard let top = logits.max() else { return [] }
        let exps = logits.map { Foundation.exp($0 - top) }
        let total = exps.reduce(0, +)
        return exps.map { $0 / total }
    }
}
