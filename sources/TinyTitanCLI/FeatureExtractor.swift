import Foundation
import Metal
import TinyTitan

/// The model-side half of `TinyTitanCLI features` (see `FeaturesCommand`):
/// loads the MoE, then per row prefills the state once, branches each
/// question from it, and appends the rows and the letter logits.
///
/// unchecked-invariant: one extractor runs one sweep on one task; nothing is
/// shared with another thread.
final class FeatureExtractor: @unchecked Sendable {
    let o: FeaturesCommand.Options
    let stdout: FileHandle

    init(options: FeaturesCommand.Options, stdout: FileHandle) {
        self.o = options
        self.stdout = stdout
    }

    func say(_ line: String) { stdout.write(Data((line + "\n").utf8)) }
    static func now() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
    static func ms(_ nanos: UInt64) -> String { String(format: "%.1f", Double(nanos) / 1e6) }

    struct Loaded {
        let model: Model
        let runner: RealForwardRunner
        let config: PrefillRuntimeConfig
        let logits: MTLBuffer
        let vocab: Int
    }

    func load() throws -> Loaded {
        let url = URL(fileURLWithPath: o.model)
        let identity = try ManifestReader.peekIdentity(directoryURL: url)
        let arch = try ArchConfig.resolved(forFamily: identity.family, directoryURL: url)
        let manifest = try ManifestReader.load(directoryURL: url, expecting: arch)
        let profile = ModelProfile.resolve(identity: identity)
        let slots = RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: manifest.expertStride, layers: manifest.arch.numLayers,
            budgetBytes: RuntimeConfiguration.affordableExpertCacheBudget(
                profile.expertCacheBudgetBytes))
        let chunk =
            profile.prefillChunkTokens.map {
                RuntimeConfiguration.profilePrefillChunk($0, forContext: o.maxContext)
            } ?? RuntimeConfiguration.qwenLongPrefillChunkTokens
        let context = try MetalContext()
        let started = Self.now()
        let model = try Model.load(
            directoryURL: url, device: context.device, expecting: arch,
            streamingMode: .pread(slotCount: slots),
            integrityPolicy: .resolved(directoryURL: url))
        let runtime = try RuntimeConfiguration(
            expertCacheSlots: slots, prefillChunkTokens: chunk, forceLogitsHead: true)
        try runtime.validate(maxContext: o.maxContext)
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: o.maxContext, runtimeConfiguration: runtime)
        if let refusal = runner.decisionReadoutRefusal {
            throw DecisionReadoutError.unsupported(
                "\(refusal) (run with TINYTITAN_PREFILL_ANE=off)")
        }
        guard o.layers.allSatisfy({ $0 < model.config.numLayers }) else {
            throw DecisionReadoutError.unsupported(
                "--layers must be below \(model.config.numLayers): \(o.layers)")
        }
        let vocab = model.config.vocabSize
        guard
            let logits = context.device.makeBuffer(
                length: vocab * MemoryLayout<Float16>.stride, options: .storageModeShared)
        else { throw DecisionReadoutError.unsupported("logits buffer allocation failed") }
        say(
            "# loaded \(identity.family.rawValue) in \(Self.ms(Self.now() - started)) ms; "
                + "slots=\(slots) chunk=\(chunk) maxContext=\(o.maxContext) layers=\(o.layers)")
        return Loaded(
            model: model, runner: runner, config: runtime.prefillConfig, logits: logits,
            vocab: vocab)
    }

    /// Prefills `tokens` at `from` with a readout armed; returns its rows.
    func readout(
        _ m: Loaded, _ tokens: ArraySlice<Int32>, from: Int, offsets: [Int], blocks: [Int],
        keepHead: Bool
    ) async throws -> DecisionReadoutRows {
        try m.runner.armDecisionReadout(
            positions: offsets.map { from + $0 }, blockLayers: blocks, keepHead: keepHead)
        do {
            _ = try await m.runner.prefillChunked(
                tokens: tokens, startPosition: from, outputMode: .logits, config: m.config,
                into: m.logits, onProgress: { _ in })
            return try m.runner.finishDecisionReadoutRows()
        } catch {
            m.runner.disarmDecisionReadout()
            throw error
        }
    }

    func run() async throws -> Int32 {
        let m = try load()
        let tokenizer = try await GFTokenizer.load(forModelDirectory: URL(fileURLWithPath: o.model))
        let encode: (String) -> [Int32] = { tokenizer.encode($0, addBOS: false) }
        // Single-token option letters, for the zero-shot readout.
        let letterIDs: [Int32?] = FeaturesCommand.letters(26).map {
            let ids = encode($0)
            return ids.count == 1 ? ids[0] : nil
        }
        let outDir = URL(fileURLWithPath: o.out, isDirectory: true)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let blobURL = outDir.appendingPathComponent("\(o.split).f16")
        let indexURL = outDir.appendingPathComponent("\(o.split).jsonl")
        FileManager.default.createFile(atPath: blobURL.path, contents: nil)
        FileManager.default.createFile(atPath: indexURL.path, contents: nil)
        let blob = try FileHandle(forWritingTo: blobURL)
        let index = try FileHandle(forWritingTo: indexURL)
        defer {
            try? blob.close()
            try? index.close()
        }
        var tally = Tally()
        let lines = try String(contentsOfFile: o.data, encoding: .utf8).split(separator: "\n")
        for (rowNumber, line) in lines.enumerated() {
            if let limit = o.limit, tally.questions >= limit { break }
            try await extractRow(
                m, String(line), rowNumber: rowNumber, encode: encode, letterIDs: letterIDs,
                blob: blob, index: index, tally: &tally)
        }
        say(
            "# \(o.split): \(tally.questions) questions from \(tally.rows) rows, "
                + "\(tally.rowsWritten) rows of features, skipped \(tally.skipped) "
                + "(over context or malformed), capped \(tally.capped) by --max-per-source; "
                + "self-check worst cosine \(String(format: "%.6f", tally.worstCosine))")
        return 0
    }

    struct Tally {
        var rows = 0
        var questions = 0
        var rowsWritten = 0
        var skipped = 0
        var capped = 0
        var perSource: [String: Int] = [:]
        var checked = 0
        var worstCosine: Double = 1
    }
}
