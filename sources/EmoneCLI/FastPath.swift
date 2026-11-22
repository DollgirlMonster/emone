import Accelerate
import Foundation
import Metal
import Emone

/// `EmoneCLI fastpath`: the Emone-lmw.13 timing spike. Is a System One
/// decision on the resident MoE, branching from context the System 2
/// conversation already holds, faster end to end than Rev-0.8B, which has to
/// prefill that context itself?
///
/// Two arms, one model per process:
///
/// - `--arm rev --model <base> --adapter <package>`: Rev over its base, cold
///   every time (each rep's state starts at a different corpus offset, so the
///   scorer's resident state never matches). The timed span is `score`:
///   prefill of N shared + M new tokens, the branch, the head.
/// - `--arm moe --model <install>`: the context is prefilled once per N
///   (untimed, it is what System 2 already paid) and becomes a branch point.
///   The timed span is what a decision adds: M new tokens plus the branch
///   row with readout, a head over the rows, and the restore back to the
///   point so System 2 continues where it was.
///
/// No trained MoE head exists, so that arm scores with a random pointer head
/// of Kev's shape: timing only, the probabilities mean nothing. Correctness
/// for the MoE arm: the logits System 2 sees for a fixed probe after the point
/// must be byte-identical before and after the decisions.
public enum FastPathCommand {
    public static let usage = """
        usage: EmoneCLI fastpath --arm rev --model <base> --adapter <package> --corpus <text> [grid]
               EmoneCLI fastpath --arm moe --model <install> --corpus <text> [grid]
          grid: --n 1024,4096 --m 0,512 --k 2,8,16 --reps 5 --chunk <tokens>
        """

    struct Options: Equatable {
        var arm = ""
        var model = ""
        var adapter: String?
        var corpus = ""
        var ns = [1_024, 4_096, 16_384, 65_536]
        var ms = [0, 512, 2_048, 8_192]
        var ks = [2, 8, 16]
        var reps = 5
        var chunk: Int?
    }

    static func parse(_ argv: [String]) -> Options? {
        var options = Options()
        var index = 0
        func list(_ value: String) -> [Int]? {
            let items = value.split(separator: ",").compactMap { Int($0) }
            return items.isEmpty || items.contains(where: { $0 < 0 }) ? nil : items
        }
        while index < argv.count {
            guard index + 1 < argv.count else { return nil }
            let value = argv[index + 1]
            switch argv[index] {
            case "--arm": options.arm = value
            case "--model": options.model = value
            case "--adapter": options.adapter = value
            case "--corpus": options.corpus = value
            case "--n": guard let v = list(value) else { return nil }; options.ns = v.sorted()
            case "--m": guard let v = list(value) else { return nil }; options.ms = v
            case "--k":
                guard let v = list(value), !v.contains(0) else { return nil }; options.ks = v
            case "--reps": guard let v = Int(value), v > 0 else { return nil }; options.reps = v
            case "--chunk": guard let v = Int(value), v > 0 else { return nil }; options.chunk = v
            default: return nil
            }
            index += 2
        }
        guard ["rev", "moe"].contains(options.arm), !options.model.isEmpty,
            !options.corpus.isEmpty, (options.arm == "rev") == (options.adapter != nil),
            options.ns.first ?? 0 > 0
        else { return nil }
        return options
    }

    /// A question shaped like a real one: a 12-token instruction and `k`
    /// options of 6 tokens, cut from `pool` so routing sees real text.
    static func question(k: Int, pool: [Int32]) -> DecisionModel.Question {
        precondition(pool.count >= 12 + 6 * k, "pool too short for \(k) options")
        let instruction = Array(pool[0..<12])
        let options = (0..<k).map { Array(pool[(12 + 6 * $0)..<(18 + 6 * $0)]) }
        return .init(instruction: instruction, options: options)
    }

    /// The branch row the MoE arm prefills, the same length and readout
    /// layout as Rev's (`DecisionModel.branchRow`): `marker` stands in for
    /// each delimiter, since this model has no trained ones.
    static func branchRow(_ q: DecisionModel.Question, marker: Int32) -> (
        ids: [Int32], readouts: [Int]
    ) {
        var ids = [marker] + q.instruction
        var closes: [Int] = []
        for option in q.options {
            ids.append(marker)
            ids += option
            ids.append(marker)
            closes.append(ids.count - 1)
        }
        ids.append(marker)
        return (ids, [ids.count - 1] + closes)
    }

    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return .nan }
        let mid = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
    }

    public static func run(
        _ argv: [String], stdout: FileHandle = .standardOutput,
        stderr: FileHandle = .standardError
    ) async -> Int32 {
        guard let options = parse(argv) else {
            stderr.write(Data("\(usage)\n".utf8))
            return 2
        }
        do {
            let text = try String(contentsOfFile: options.corpus, encoding: .utf8)
            let say: (String) -> Void = { stdout.write(Data(($0 + "\n").utf8)) }
            say("# fastpath arm=\(options.arm) model=\(options.model) reps=\(options.reps)")
            say("arm\tn\tm\tk\trep\ttotal_ms\tdetail")
            if options.arm == "rev" {
                try await runRev(options, text: text, say: say)
            } else {
                return try await runMoE(options, text: text, say: say)
            }
            return 0
        } catch {
            stderr.write(Data("error: \(error)\n".utf8))
            return 1
        }
    }

    static func now() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
    static func ms(_ nanos: UInt64) -> Double { Double(nanos) / 1e6 }
    static func f1(_ value: Double) -> String { String(format: "%.1f", value) }

    /// The corpus as this model's tokens, long enough for the whole grid.
    static func tokens(_ text: String, model: URL, need: Int) async throws -> [Int32] {
        let tokenizer = try await GFTokenizer.load(forModelDirectory: model)
        let ids = tokenizer.encode(text, addBOS: false)
        guard ids.count >= need else {
            throw DecisionReadoutError.unsupported(
                "corpus is \(ids.count) tokens; the grid needs \(need)")
        }
        return ids
    }

    static func needed(_ o: Options) -> Int {
        (o.ns.last ?? 0) + (o.ms.max() ?? 0) + 12 + 6 * (o.ks.max() ?? 0) + 64 * (o.reps + 1)
    }

    static func runRev(_ o: Options, text: String, say: (String) -> Void) async throws {
        let base = URL(fileURLWithPath: o.model)
        let context = try MetalContext()
        let maxContext = (o.ns.last ?? 0) + (o.ms.max() ?? 0) + 512
        let decision = try DecisionModel.load(
            adapter: URL(fileURLWithPath: o.adapter ?? ""), base: base, context: context,
            maxContext: maxContext, prefillChunkTokens: o.chunk ?? 4_096)
        let ids = try await tokens(text, model: base, need: needed(o))
        // Rev scores real requests with escaped text; delimiters never occur
        // in these ids because the corpus is prose, so no escaping is needed.
        let questionPool = Array(ids.suffix(12 + 6 * (o.ks.max() ?? 0)))
        for n in o.ns {
            for m in o.ms {
                for k in o.ks {
                    let q = question(k: k, pool: questionPool)
                    var totals: [Double] = []
                    for rep in 0...o.reps {
                        // A different offset each rep: the scorer keeps the
                        // last state resident, and Rev never has this one.
                        let start = 64 * rep
                        let state = Array(ids[start..<(start + n + m)])
                        let started = now()
                        let (_, timing) = try await decision.score(state: state, questions: [q])
                        let total = ms(now() - started)
                        let detail =
                            "prefill=\(f1(ms(timing.statePrefillNanos))) "
                            + "branch=\(f1(ms(timing.branchNanos.reduce(0, +)))) "
                            + "head=\(f1(ms(timing.headNanos)))"
                        say(
                            "rev\t\(n)\t\(m)\t\(k)\t\(rep == 0 ? "warm" : "\(rep)")\t\(f1(total))\t\(detail)"
                        )
                        if rep > 0 { totals.append(total) }
                    }
                    say(
                        "SUMMARY\trev\t\(n)\t\(m)\t\(k)\tmedian=\(f1(median(totals)))\tmin=\(f1(totals.min() ?? .nan))\tmax=\(f1(totals.max() ?? .nan))"
                    )
                }
            }
        }
    }

    // lint:allow-long one sweep: load, then per N extend the context, probe, time every (M, k), re-probe
    static func runMoE(_ o: Options, text: String, say: (String) -> Void) async throws -> Int32 {
        let url = URL(fileURLWithPath: o.model)
        let identity = try ManifestReader.peekIdentity(directoryURL: url)
        let arch = try ArchConfig.resolved(forFamily: identity.family, directoryURL: url)
        let manifest = try ManifestReader.load(directoryURL: url, expecting: arch)
        let slots = RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: manifest.expertStride, layers: manifest.arch.numLayers,
            budgetBytes: RuntimeConfiguration.affordableExpertCacheBudget(
                ModelProfile.resolve(identity: identity).expertCacheBudgetBytes))
        let maxContext = (o.ns.last ?? 0) + (o.ms.max() ?? 0) + 512
        let chunk =
            o.chunk ?? ModelProfile.resolve(identity: identity).prefillChunkTokens.map {
                RuntimeConfiguration.profilePrefillChunk($0, forContext: maxContext)
            } ?? RuntimeConfiguration.qwenLongPrefillChunkTokens
        let context = try MetalContext()
        let loadStarted = now()
        let model = try Model.load(
            directoryURL: url, device: context.device, expecting: arch,
            streamingMode: .pread(slotCount: slots),
            integrityPolicy: .resolved(directoryURL: url))
        let runtime = try RuntimeConfiguration(
            expertCacheSlots: slots, prefillChunkTokens: chunk, forceLogitsHead: true)
        try runtime.validate(maxContext: maxContext)
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: maxContext, runtimeConfiguration: runtime)
        if let refusal = runner.decisionReadoutRefusal {
            throw DecisionReadoutError.unsupported(
                "\(refusal) (run with EMONE_PREFILL_ANE=off)")
        }
        say(
            "# loaded \(identity.family.rawValue) in \(f1(ms(now() - loadStarted))) ms; "
                + "slots=\(slots) chunk=\(chunk) maxContext=\(maxContext)")
        let ids = try await tokens(text, model: url, need: needed(o))
        let hidden = model.config.hiddenSize
        let head = RandomPointerHead(hiddenSize: hidden)
        let config = runtime.prefillConfig
        let vocabBytes = model.config.vocabSize * MemoryLayout<Float>.stride
        guard
            let logits = context.device.makeBuffer(length: vocabBytes, options: .storageModeShared)
        else { throw DecisionReadoutError.unsupported("logits buffer allocation failed") }

        // Readout prefill: rows at `offsets` past `from`, LM head skipped.
        func readout(_ tokens: ArraySlice<Int32>, from: Int, offsets: [Int]) async throws
            -> [[Float]]
        {
            try runner.armDecisionReadout(positions: offsets.map { from + $0 })
            do {
                _ = try await runner.prefillChunked(
                    tokens: tokens, startPosition: from, outputMode: .logits, config: config,
                    into: logits, onProgress: { _ in })
                return try runner.finishDecisionReadout()
            } catch {
                runner.disarmDecisionReadout()
                throw error
            }
        }
        // What System 2 sees next: the logits after a fixed 16-token probe.
        let probe = ids[(ids.count - 16)...]
        func probeLogits(at point: DecisionBranchPoint) async throws -> Data {
            memset(logits.contents(), 0, vocabBytes)
            _ = try await runner.prefillChunked(
                tokens: probe, startPosition: point.position, outputMode: .logits,
                config: config, into: logits, onProgress: { _ in })
            let bytes = Data(bytes: logits.contents(), count: vocabBytes)
            try await runner.restoreDecisionBranchPoint(point)
            return bytes
        }

        let questionPool = Array(ids.suffix(12 + 6 * (o.ks.max() ?? 0) + 16).dropLast(16))
        // New content M comes from just past the largest N, so it is text
        // the context has not seen.
        let newBase = o.ns.last ?? 0
        var position = 0
        var mismatches = 0
        await runner.resetSequence(slot: 0)
        for n in o.ns {
            let extendStarted = now()
            _ = try await readout(ids[position..<n], from: position, offsets: [n - 1 - position])
            say(
                "# context \(position)->\(n) tokens prefilled in \(f1(ms(now() - extendStarted))) ms (System 2's cost, untimed below)"
            )
            position = n
            let point = try await runner.makeDecisionBranchPoint()
            let before = try await probeLogits(at: point)
            for m in o.ms {
                for k in o.ks {
                    let q = question(k: k, pool: questionPool)
                    let branch = branchRow(q, marker: q.instruction[0])
                    let row = Array(ids[newBase..<(newBase + m)]) + branch.ids
                    var totals: [Double] = []
                    for rep in 0...o.reps {
                        let started = now()
                        let rows = try await readout(
                            row[...], from: n, offsets: branch.readouts.map { m + $0 })
                        let decided = now()
                        _ = head.logits(decide: rows[0], options: Array(rows.dropFirst()))
                        let headDone = now()
                        try await runner.restoreDecisionBranchPoint(point)
                        let done = now()
                        let total = ms(done - started)
                        let detail =
                            "prefill+branch=\(f1(ms(decided - started))) "
                            + "head=\(f1(ms(headDone - decided))) restore=\(f1(ms(done - headDone)))"
                        say(
                            "moe\t\(n)\t\(m)\t\(k)\t\(rep == 0 ? "warm" : "\(rep)")\t\(f1(total))\t\(detail)"
                        )
                        if rep > 0 { totals.append(total) }
                    }
                    say(
                        "SUMMARY\tmoe\t\(n)\t\(m)\t\(k)\tmedian=\(f1(median(totals)))\tmin=\(f1(totals.min() ?? .nan))\tmax=\(f1(totals.max() ?? .nan))"
                    )
                }
            }
            let after = try await probeLogits(at: point)
            let same = before == after
            if !same { mismatches += 1 }
            say(
                "# N=\(n): System 2 probe logits after decisions \(same ? "byte-identical" : "DIFFER")"
            )
        }
        say("# probe mismatches: \(mismatches)")
        return mismatches == 0 ? 0 : 3
    }
}

/// Kev's pointer-head shape (d=256, q/k projections with bias) with seeded
/// random weights: costs what a trained head costs, means nothing.
struct RandomPointerHead {
    let hiddenSize: Int
    let headDim = 256
    let q: [Float]
    let k: [Float]

    init(hiddenSize: Int, seed: UInt64 = 0x5EED) {
        self.hiddenSize = hiddenSize
        var state = seed
        func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int64(bitPattern: state >> 11) % 2_000 - 1_000) / 1e5
        }
        q = (0..<(256 * hiddenSize)).map { _ in next() }
        k = (0..<(256 * hiddenSize)).map { _ in next() }
    }

    private func project(_ w: [Float], _ x: [Float]) -> [Float] {
        var y = [Float](repeating: 0, count: headDim)
        vDSP_mmul(w, 1, x, 1, &y, 1, vDSP_Length(headDim), 1, vDSP_Length(hiddenSize))
        return y
    }

    func logits(decide: [Float], options: [[Float]]) -> [Float] {
        let qv = project(q, decide)
        return options.map { option in
            var dot: Float = 0
            vDSP_dotpr(project(k, option), 1, qv, 1, &dot, vDSP_Length(headDim))
            return dot / 16
        }
    }
}
