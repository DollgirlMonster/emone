import Foundation
import Metal
import Emone

/// `EmoneCLI features`: feature extraction for Emone-lmw.15, one
/// general decision head on a frozen MoE that reuses the held context.
///
/// Each row's state is prefilled once, the way a System Two conversation
/// already holds it, and becomes a branch point. Each question then runs as
/// a branch from that point with frozen weights: a user turn holding the
/// question and lettered options, then the thinking-off assistant prefix. The
/// branch reads, at the decide position (its last token) and at each option
/// line's last token:
///
/// - the raw residual after each requested block (`--layers`), and
/// - the final-normed rows (what a head on the last layer would read),
///
/// and keeps the LM head, so the same branch also yields the zero-shot letter
/// readout: the next-token logits of the option letters, for options <= 26.
///
/// Output under `--out`: `<split>.f16` (rows, fp16, row-major) and
/// `<split>.jsonl` (one record per question: where its rows start, the label,
/// the letter logits). Per question the rows are `layers.count + 1` blocks of
/// `k + 1` rows each (decide first, then the options in order); the last
/// block is the final-normed one.
public enum FeaturesCommand {
    public static let usage = """
        usage: EmoneCLI features --model <MoE install> --data <rows.jsonl> --split <name> --out <dir>
               [--layers 8,16,24,32,39] [--max-per-source N] [--limit N] [--max-context N] [--self-check N]
        """

    struct Options: Equatable {
        var model = ""
        var data = ""
        var split = ""
        var out = ""
        var layers = [8, 16, 20, 24, 28, 32, 36, 39]
        var maxPerSource: Int?
        var limit: Int?
        var maxContext = 16_384
        var selfCheck = 0
    }

    static func parse(_ argv: [String]) -> Options? {
        var o = Options()
        var index = 0
        while index < argv.count {
            guard index + 1 < argv.count else { return nil }
            let value = argv[index + 1]
            switch argv[index] {
            case "--model": o.model = value
            case "--data": o.data = value
            case "--split": o.split = value
            case "--out": o.out = value
            case "--layers":
                let layers = value.split(separator: ",").compactMap { Int($0) }
                guard !layers.isEmpty, layers.allSatisfy({ $0 >= 0 }) else { return nil }
                o.layers = layers
            case "--max-per-source":
                guard let v = Int(value), v > 0 else { return nil }; o.maxPerSource = v
            case "--limit": guard let v = Int(value), v > 0 else { return nil }; o.limit = v
            case "--max-context":
                guard let v = Int(value), v >= 1_024 else { return nil }; o.maxContext = v
            case "--self-check":
                guard let v = Int(value), v >= 0 else { return nil }; o.selfCheck = v
            default: return nil
            }
            index += 2
        }
        guard !o.model.isEmpty, !o.data.isEmpty, !o.split.isEmpty, !o.out.isEmpty else {
            return nil
        }
        return o
    }

    /// One question, ready to render: its options' text and the label's index.
    struct Question: Equatable {
        let name: String
        let kind: String
        let source: String
        let instructions: String
        let options: [String]
        let label: Int?
    }

    enum RowError: Error, CustomStringConvertible {
        case malformed(String)
        var description: String {
            switch self {
            case .malformed(let why): why
            }
        }
    }

    /// The state text and questions of one Kev/TypeSafe-shaped row:
    /// `{"state": ..., "questions": {name: {type, instructions, criteria?, label?}}}`.
    /// Choice criteria keep the row's own key order (an object's order is the
    /// options' order); a noul is Yes then No; a score keeps its levels' order.
    /// Text is Kev's rendering (`KevRender`: objects become labelled fields,
    /// arrays bullet lists), the same text Rev reads, with special-token
    /// spellings escaped so no field can forge a turn marker.
    static func row(_ json: OrderedJSON) throws -> (state: String, questions: [Question]) {
        guard let stateValue = json["state"], stateValue != .null else {
            throw RowError.malformed("row has no state")
        }
        guard case .object(let entries)? = json["questions"] else {
            throw RowError.malformed("row has no questions object")
        }
        let state = KevRender.escapeSpecialTokens(KevRender.render(stateValue))
        return (state, try entries.map { try question(name: $0.0, $0.1) })
    }

    static func question(name: String, _ q: OrderedJSON) throws -> Question {
        guard case .string(let kind)? = q["type"], let instructionValue = q["instructions"]
        else { throw RowError.malformed("\(name): no type or instructions") }
        let instructions = KevRender.escapeSpecialTokens(KevRender.render(instructionValue))
        let source: String = {
            if case .string(let s)? = q["src"] { return s }
            return ""
        }()
        var options: [String] = []
        var label: Int?
        switch (kind, q["criteria"]) {
        case ("choice", .object(let pairs)?):
            options = pairs.map { KevRender.optionText($0.0, $0.1) }
            if case .string(let key)? = q["label"] { label = pairs.firstIndex { $0.0 == key } }
            if case .integer(let digits)? = q["label"] { label = Int(digits) }
        case ("choice", .array(let items)?), ("score", .array(let items)?):
            options = items.map { KevRender.render($0) }
            if case .integer(let digits)? = q["label"] { label = Int(digits) }
            if case .string(let key)? = q["label"] { label = options.firstIndex(of: key) }
        case ("noul", let criteria):
            options = [
                KevRender.optionText("Yes", criteria?["true"]),
                KevRender.optionText("No", criteria?["false"]),
            ]
            if case .bool(let yes)? = q["label"] { label = yes ? 0 : 1 }
        default:
            throw RowError.malformed("\(name): unsupported \(kind) question or criteria")
        }
        guard options.count >= 2 else {
            throw RowError.malformed("\(name): fewer than two options")
        }
        if let label, !options.indices.contains(label) {
            throw RowError.malformed("\(name): label \(label) is not an option index")
        }
        return Question(
            name: name, kind: kind, source: source, instructions: instructions,
            options: options.map(KevRender.escapeSpecialTokens), label: label)
    }

    /// Option labels: A to Z, then AA, AB, ... for longer lists.
    static func letters(_ count: Int) -> [String] {
        let alphabet = (65...90).map { String(UnicodeScalar(UInt8($0))) }
        return (0..<count).map { i in
            i < 26 ? alphabet[i] : alphabet[i / 26 - 1] + alphabet[i % 26]
        }
    }

    /// The held state, as a System Two conversation would hold a user turn.
    static func stateText(_ state: String) -> String {
        "<|im_start|>user\n\(state)<|im_end|>\n"
    }

    /// A question's branch: token ids, each option line's last token, and the
    /// decide position (the last token), all relative to the branch start.
    /// The pieces are tokenized separately, so the positions are exact.
    static func branch(_ q: Question, encode: (String) -> [Int32]) -> (
        ids: [Int32], optionEnds: [Int], decide: Int
    ) {
        var ids = encode("<|im_start|>user\nQuestion: \(q.instructions)\nOptions:\n")
        var ends: [Int] = []
        for (letter, option) in zip(letters(q.options.count), q.options) {
            ids += encode("\(letter). \(option)\n")
            ends.append(ids.count - 1)
        }
        ids += encode(
            "Answer with the letter of the best option.<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return (ids, ends, ids.count - 1)
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
            return try await FeatureExtractor(options: options, stdout: stdout).run()
        } catch {
            stderr.write(Data("error: \(error)\n".utf8))
            return 1
        }
    }
}
