import Foundation
import Emone

/// `EmoneCLI decide --model <decision install> --fixture <json> [--repeat N]`:
/// replays a Kev reference fixture (benchmark/kev/kev_reference.py writes one)
/// through the decision scorer and reports parity and timing. The fixture
/// carries token ids, so this checks the engine and the head, not a tokenizer.
///
/// Questions sharing a state run as one request, as a server would send them,
/// so the state prefill and the branch rollback are both exercised; `--repeat`
/// re-sends every request, which measures the resident-state path.
public enum DecideCommand {
    public static let usage = """
        usage: EmoneCLI decide --model <decision-model dir> --fixture <fixture.json> [--repeat N]
               EmoneCLI decide --model <base install> --adapter <adapter package> --fixture ...
        """

    struct Fixture: Decodable {
        struct Question: Decodable {
            let key: String
            let ids: [Int32]
            let decide: Int
            let opts: [Int]
            let probs_kev: [Float]
        }
        let questions: [Question]
    }

    public static func run(
        _ argv: [String], stdout: FileHandle = .standardOutput,
        stderr: FileHandle = .standardError
    ) async -> Int32 {
        var model: String?
        var fixture: String?
        var adapter: String?
        var repeats = 3
        var index = 0
        while index < argv.count {
            let flag = argv[index]
            let value = index + 1 < argv.count ? argv[index + 1] : nil
            switch (flag, value) {
            case ("--model", let v?): model = v
            case ("--fixture", let v?): fixture = v
            case ("--adapter", let v?): adapter = v
            case ("--repeat", let v?): repeats = max(1, Int(v) ?? 1)
            default:
                stderr.write(Data("\(usage)\n".utf8))
                return 2
            }
            index += 2
        }
        guard let model, let fixture else {
            stderr.write(Data("\(usage)\n".utf8))
            return 2
        }
        do {
            return try await drive(
                model: URL(fileURLWithPath: model), fixture: URL(fileURLWithPath: fixture),
                adapter: adapter.map { URL(fileURLWithPath: $0) }, repeats: repeats, stdout: stdout)
        } catch {
            stderr.write(Data("error: \(error)\n".utf8))
            return 1
        }
    }

    static func drive(
        model url: URL, fixture: URL, adapter: URL? = nil, repeats: Int, stdout: FileHandle
    ) async throws -> Int32 {
        func say(_ line: String) { stdout.write(Data((line + "\n").utf8)) }
        let loadStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let context = try MetalContext()
        let decision =
            try adapter.map { try DecisionModel.load(adapter: $0, base: url, context: context) }
            ?? DecisionModel.load(directory: url, context: context)
        let name =
            adapter.map { "\($0.lastPathComponent) over \(url.lastPathComponent)" }
            ?? url.lastPathComponent
        say("loaded \(name) in \(ms(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - loadStarted)) ms")
        let fixtureData = try Data(contentsOf: fixture)
        let rows = try JSONDecoder().decode(Fixture.self, from: fixtureData).questions
        if let requests = try OrderedJSON.parse(fixtureData)["requests"] {
            say(
                try await checkEncoding(
                    requests, rows: rows, decision: decision, tokenizerFrom: url))
        }
        let requests = try group(rows, delimiters: decision.delimiters)
        var worst: Float = 0
        var flips = 0
        for pass in 0..<repeats {
            for request in requests {
                let (answers, timing) = try await decision.score(
                    state: request.state, questions: request.questions.map(\.question))
                for (item, answer) in zip(request.questions, answers) {
                    let dp =
                        zip(answer.probabilities, item.reference).map { abs($0 - $1) }.max() ?? 0
                    worst = max(worst, dp)
                    if argmax(answer.probabilities) != argmax(item.reference) { flips += 1 }
                    if pass == 0 {
                        say(
                            "\(item.key.padding(toLength: 22, withPad: " ", startingAt: 0)) "
                                + "ours=\(fmt(answer.probabilities)) kev=\(fmt(item.reference)) "
                                + "max|dp|=\(String(format: "%.4f", dp))")
                    }
                }
                say(
                    "  pass \(pass) state \(request.state.count + 1) tok "
                        + (timing.stateReused
                            ? "(resident)" : "prefill \(ms(timing.statePrefillNanos)) ms")
                        + ", branches \(timing.branchNanos.map(ms).joined(separator: "/")) ms"
                        + ", restore \(ms(timing.restoreNanos)) ms, head \(ms(timing.headNanos)) ms"
                )
            }
        }
        let gpu = decision.drainKernelGPUSummary()
        if !gpu.isEmpty {
            say("GPU time by role over all passes (EMONE_KERNEL_STATS):")
            for row in gpu {
                say(
                    "  \(row.role.padding(toLength: 28, withPad: " ", startingAt: 0)) "
                        + "\(String(format: "%8.1f", row.millis)) ms  x\(row.count)")
            }
        }
        say(
            "max |dp| vs kev over \(repeats) passes: \(String(format: "%.5f", worst)); argmax flips: \(flips)"
        )
        return flips == 0 ? 0 : 3
    }

    struct Item {
        let key: String
        let question: DecisionModel.Question
        let reference: [Float]
    }

    struct Request {
        let state: [Int32]
        var questions: [Item]
    }

    /// The engine's own request encoding (`SystemOneRequest`, the base
    /// install's tokenizer) against the fixture's token ids, which Kev's
    /// `kev.api.to_record` + `kev.model.encode` produced: every question's
    /// row must match exactly, or the model reads different text.
    static func checkEncoding(
        _ requests: OrderedJSON, rows: [Fixture.Question], decision: DecisionModel,
        tokenizerFrom directory: URL
    ) async throws -> String {
        guard case .array(let items) = requests else {
            return "encoding: fixture requests unreadable"
        }
        let tokenizer = try await GFTokenizer.load(forModelDirectory: directory)
        var ours: [[Int32]] = []
        for item in items {
            let request = try SystemOneRequest(json: item)
            let (state, questions) = request.encoded { tokenizer.encode($0) }
            ours += questions.map { decision.stateRow(state) + decision.branchRow($0).ids }
        }
        let theirs = rows.map(\.ids)
        let matched = zip(ours, theirs).filter { $0 == $1 }.count
        guard ours.count == theirs.count, matched == theirs.count else {
            let first =
                Array(zip(ours, theirs)).firstIndex { $0 != $1 } ?? min(ours.count, theirs.count)
            return "encoding: \(matched) of \(theirs.count) rows match Kev's token ids; "
                + "first difference at row \(first)"
        }
        return "encoding: all \(theirs.count) rows match Kev's token ids exactly"
    }

    /// Splits each fixture row back into state and question at the
    /// delimiters, and groups questions that share a state.
    static func group(_ rows: [Fixture.Question], delimiters d: DecisionModel.Delimiters) throws
        -> [Request]
    {
        var requests: [Request] = []
        for row in rows {
            guard row.ids.first == d.state, let q = row.ids.firstIndex(of: d.question),
                row.ids.last == d.decide, row.decide == row.ids.count - 1
            else { throw DecisionReadoutError.unsupported("\(row.key): not a Kev row") }
            let state = Array(row.ids[1..<q])
            var instruction: [Int32] = []
            var options: [[Int32]] = []
            var current: [Int32]?
            for token in row.ids[(q + 1)..<(row.ids.count - 1)] {
                switch token {
                case d.optionOpen: current = []
                case d.optionClose:
                    options.append(current ?? [])
                    current = nil
                default:
                    if current != nil { current?.append(token) } else { instruction.append(token) }
                }
            }
            guard options.count == row.opts.count else {
                throw DecisionReadoutError.unsupported("\(row.key): option count mismatch")
            }
            let item = Item(
                key: row.key, question: .init(instruction: instruction, options: options),
                reference: row.probs_kev)
            if let last = requests.indices.last, requests[last].state == state {
                requests[last].questions.append(item)
            } else {
                requests.append(Request(state: state, questions: [item]))
            }
        }
        return requests
    }

    static func argmax(_ values: [Float]) -> Int {
        values.indices.max { values[$0] < values[$1] } ?? 0
    }

    static func fmt(_ values: [Float]) -> String {
        "[" + values.map { String(format: "%.4f", $0) }.joined(separator: ", ") + "]"
    }

    static func ms(_ nanos: UInt64) -> String { String(format: "%.1f", Double(nanos) / 1e6) }
}
