import Foundation

/// JSON as Kev's Python server reads it (Emone-lmw.5): objects keep their
/// key order, because Kev's `render` and the question order both follow it,
/// and numbers keep whether they were written as integers, because Python's
/// `str()` prints `3` and `3.0` differently. `JSONSerialization` keeps
/// neither, and the rendered text is what the model reads.
public indirect enum OrderedJSON: Equatable, Sendable {
    case object([(String, OrderedJSON)])
    case array([OrderedJSON])
    case string(String)
    /// An integer literal's digits (a JSON integer, Python `int`).
    case integer(String)
    case number(Double)
    case bool(Bool)
    case null

    public static func == (lhs: OrderedJSON, rhs: OrderedJSON) -> Bool {
        switch (lhs, rhs) {
        case (.object(let a), .object(let b)):
            return a.count == b.count && zip(a, b).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 }
        case (.array(let a), .array(let b)): return a == b
        case (.string(let a), .string(let b)): return a == b
        case (.integer(let a), .integer(let b)): return a == b
        case (.number(let a), .number(let b)): return a == b
        case (.bool(let a), .bool(let b)): return a == b
        case (.null, .null): return true
        default: return false
        }
    }

    public subscript(key: String) -> OrderedJSON? {
        guard case .object(let pairs) = self else { return nil }
        return pairs.last { $0.0 == key }?.1
    }

    public struct ParseError: Error, CustomStringConvertible {
        public let description: String
    }

    public static func parse(_ data: Data) throws -> OrderedJSON {
        var parser = Parser(bytes: Array(data))
        parser.skipSpace()
        let value = try parser.value(depth: 0)
        parser.skipSpace()
        guard parser.index == parser.bytes.count else { throw parser.fail("trailing data") }
        return value
    }

    private struct Parser {
        let bytes: [UInt8]
        var index = 0
        static let maxDepth = 128

        func fail(_ what: String) -> ParseError {
            ParseError(description: "invalid JSON at byte \(index): \(what)")
        }

        mutating func skipSpace() {
            while index < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[index]) {
                index += 1
            }
        }

        mutating func literal(_ word: String, _ value: OrderedJSON) throws -> OrderedJSON {
            let w = Array(word.utf8)
            guard index + w.count <= bytes.count, Array(bytes[index..<(index + w.count)]) == w
            else { throw fail("expected \(word)") }
            index += w.count
            return value
        }

        mutating func value(depth: Int) throws -> OrderedJSON {
            guard depth < Self.maxDepth else { throw fail("nested too deeply") }
            guard index < bytes.count else { throw fail("unexpected end") }
            switch bytes[index] {
            case UInt8(ascii: "{"): return try object(depth: depth)
            case UInt8(ascii: "["): return try array(depth: depth)
            case UInt8(ascii: "\""): return .string(try string())
            case UInt8(ascii: "t"): return try literal("true", .bool(true))
            case UInt8(ascii: "f"): return try literal("false", .bool(false))
            case UInt8(ascii: "n"): return try literal("null", .null)
            default: return try number()
            }
        }

        /// Python's `json.loads`: a repeated key keeps its first position and
        /// takes the last value.
        mutating func object(depth: Int) throws -> OrderedJSON {
            index += 1
            var pairs: [(String, OrderedJSON)] = []
            skipSpace()
            if index < bytes.count, bytes[index] == UInt8(ascii: "}") {
                index += 1
                return .object(pairs)
            }
            while true {
                skipSpace()
                guard index < bytes.count, bytes[index] == UInt8(ascii: "\"") else {
                    throw fail("expected a key")
                }
                let key = try string()
                skipSpace()
                guard index < bytes.count, bytes[index] == UInt8(ascii: ":") else {
                    throw fail("expected :")
                }
                index += 1
                skipSpace()
                let v = try value(depth: depth + 1)
                if let at = pairs.firstIndex(where: { $0.0 == key }) {
                    pairs[at].1 = v
                } else {
                    pairs.append((key, v))
                }
                skipSpace()
                guard index < bytes.count else { throw fail("unterminated object") }
                if bytes[index] == UInt8(ascii: ",") {
                    index += 1
                    continue
                }
                guard bytes[index] == UInt8(ascii: "}") else { throw fail("expected , or }") }
                index += 1
                return .object(pairs)
            }
        }

        mutating func array(depth: Int) throws -> OrderedJSON {
            index += 1
            var items: [OrderedJSON] = []
            skipSpace()
            if index < bytes.count, bytes[index] == UInt8(ascii: "]") {
                index += 1
                return .array(items)
            }
            while true {
                skipSpace()
                items.append(try value(depth: depth + 1))
                skipSpace()
                guard index < bytes.count else { throw fail("unterminated array") }
                if bytes[index] == UInt8(ascii: ",") {
                    index += 1
                    continue
                }
                guard bytes[index] == UInt8(ascii: "]") else { throw fail("expected , or ]") }
                index += 1
                return .array(items)
            }
        }

        mutating func hex4() throws -> UInt32 {
            guard index + 4 <= bytes.count,
                let v = UInt32(
                    bytes[index..<(index + 4)].lossyUTF8String, radix: 16)
            else { throw fail("bad \\u escape") }
            index += 4
            return v
        }

        mutating func string() throws -> String {
            index += 1
            var out = [UInt8]()
            while true {
                guard index < bytes.count else { throw fail("unterminated string") }
                let b = bytes[index]
                index += 1
                if b == UInt8(ascii: "\"") { break }
                guard b >= 0x20 else { throw fail("control character in string") }
                guard b == UInt8(ascii: "\\") else {
                    out.append(b)
                    continue
                }
                guard index < bytes.count else { throw fail("unterminated escape") }
                let e = bytes[index]
                index += 1
                switch e {
                case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"): out.append(e)
                case UInt8(ascii: "b"): out.append(0x08)
                case UInt8(ascii: "f"): out.append(0x0C)
                case UInt8(ascii: "n"): out.append(0x0A)
                case UInt8(ascii: "r"): out.append(0x0D)
                case UInt8(ascii: "t"): out.append(0x09)
                case UInt8(ascii: "u"):
                    var scalar = try hex4()
                    if (0xD800..<0xDC00).contains(scalar), index + 6 <= bytes.count,
                        bytes[index] == UInt8(ascii: "\\"), bytes[index + 1] == UInt8(ascii: "u")
                    {
                        let save = index
                        index += 2
                        let low = try hex4()
                        if (0xDC00..<0xE000).contains(low) {
                            scalar = 0x10000 + ((scalar - 0xD800) << 10) + (low - 0xDC00)
                        } else {
                            index = save
                        }
                    }
                    // A lone surrogate has no UTF-8 form; Python keeps it, and
                    // the tokenizer could not read it either way.
                    let s = Unicode.Scalar(scalar) ?? "\u{FFFD}"
                    out += Array(String(Character(s)).utf8)
                default: throw fail("bad escape")
                }
            }
            return out.lossyUTF8String
        }

        mutating func number() throws -> OrderedJSON {
            let start = index
            var isInteger = true
            if index < bytes.count, bytes[index] == UInt8(ascii: "-") { index += 1 }
            func digits() -> Int {
                let s = index
                while index < bytes.count, (0x30...0x39).contains(bytes[index]) { index += 1 }
                return index - s
            }
            let intDigits = digits()
            guard intDigits > 0 else { throw fail("expected a value") }
            if intDigits > 1, bytes[index - intDigits] == 0x30 { throw fail("leading zero") }
            if index < bytes.count, bytes[index] == UInt8(ascii: ".") {
                index += 1
                isInteger = false
                guard digits() > 0 else { throw fail("expected digits after .") }
            }
            if index < bytes.count, bytes[index] | 0x20 == UInt8(ascii: "e") {
                index += 1
                isInteger = false
                if index < bytes.count,
                    bytes[index] == UInt8(ascii: "+") || bytes[index] == UInt8(ascii: "-")
                {
                    index += 1
                }
                guard digits() > 0 else { throw fail("expected exponent digits") }
            }
            let text = bytes[start..<index].lossyUTF8String
            if isInteger { return .integer(text == "-0" ? "0" : text) }
            guard let v = Double(text), v.isFinite else { throw fail("number out of range") }
            return .number(v)
        }
    }
}

/// Kev's request rendering (kev/api.py), byte for byte: the text the model
/// reads is a function of it, and Rev was trained on Kev's.
public enum KevRender {
    /// Python's `str(float)` (`repr`): the shortest round-trip digits, fixed
    /// notation while the decimal point sits within 16 digits and at least 4
    /// places from the first, `d.ddde+XX` otherwise.
    public static func pythonFloat(_ value: Double) -> String {
        if value == 0 { return value.sign == .minus ? "-0.0" : "0.0" }
        let negative = value < 0
        var text = "\(abs(value))"
        var exponent = 0
        if let e = text.firstIndex(where: { $0 == "e" || $0 == "E" }) {
            exponent = Int(text[text.index(after: e)...]) ?? 0
            text = String(text[..<e])
        }
        let parts = text.split(separator: ".", omittingEmptySubsequences: false)
        let whole = String(parts[0])
        let fraction = parts.count > 1 ? String(parts[1]) : ""
        var digits = whole + fraction
        // value = 0.d1d2... x 10^decpt
        var decpt = whole.count + exponent
        while digits.hasPrefix("0") && digits.count > 1 {
            digits.removeFirst()
            decpt -= 1
        }
        while digits.hasSuffix("0") && digits.count > 1 { digits.removeLast() }
        let sign = negative ? "-" : ""
        if decpt > -4 && decpt <= 16 {
            if decpt <= 0 { return sign + "0." + String(repeating: "0", count: -decpt) + digits }
            if decpt >= digits.count {
                return sign + digits + String(repeating: "0", count: decpt - digits.count) + ".0"
            }
            let cut = digits.index(digits.startIndex, offsetBy: decpt)
            return sign + digits[..<cut] + "." + digits[cut...]
        }
        let e = decpt - 1
        let mantissa = digits.count == 1 ? digits : "\(digits.prefix(1)).\(digits.dropFirst())"
        let magnitude = abs(e)
        return sign + mantissa + "e" + (e < 0 ? "-" : "+") + (magnitude < 10 ? "0" : "")
            + "\(magnitude)"
    }

    /// `render(v, indent)`: str | object | array flattened into the text the
    /// model sees, field names kept as labels.
    public static func render(_ value: OrderedJSON, indent: Int = 0) -> String {
        let pad = String(repeating: "  ", count: indent)
        switch value {
        case .null: return ""
        case .string(let s): return s
        case .integer(let s): return s
        case .number(let d): return pythonFloat(d)
        case .bool(let b): return b ? "True" : "False"
        case .array(let items):
            return items.map { "\(pad)- " + lstrip(render($0, indent: indent + 1)) }
                .joined(separator: "\n")
        case .object(let pairs):
            return pairs.map { key, x in
                switch x {
                case .object, .array: return "\(pad)\(key):\n" + render(x, indent: indent + 1)
                default: return "\(pad)\(key): " + render(x)
                }
            }.joined(separator: "\n")
        }
    }

    /// Python's `str.lstrip()`: leading whitespace, Unicode's definition.
    static func lstrip(_ s: String) -> String {
        String(s.drop { $0.isWhitespace })
    }

    public static func optionText(_ name: String, _ description: OrderedJSON?) -> String {
        guard let description else { return name }
        if case .null = description { return name }
        if case .string(let s) = description, s.isEmpty { return name }
        return "\(name): \(render(description))"
    }

    /// `user_tokens`'s rewrite: `<|name|>` becomes `<¦name¦>` before
    /// tokenizing, so caller text can never produce a delimiter or control
    /// token and an option boundary cannot be forged.
    public static func escapeSpecialTokens(_ text: String) -> String {
        text.replacing(/<\|([A-Za-z0-9_]+)\|>/) { "<¦\($0.output.1)¦>" }
    }
}

/// A TypeSafe System One request (`POST /v1/systemone`) as Kev reads it:
/// a state and named Noul / Choice / Score questions, plus Emone's one
/// extension, a request-level `temperature`.
public struct SystemOneRequest: Sendable {
    public enum Kind: String, Sendable { case noul, choice, score }

    public struct Question: Sendable {
        public let id: String
        public let kind: Kind
        public let instruction: String
        /// The rendered option texts, in the order the probabilities come back.
        public let options: [String]
        /// The names the probabilities are reported under (`question_keys`).
        public let keys: [String]
    }

    public struct Invalid: Error, CustomStringConvertible {
        public let description: String
    }

    public static let maximumOptions = 255

    public let model: String
    public let state: String
    public let questions: [Question]
    public let temperature: Float?

    public init(json: OrderedJSON) throws {
        guard case .object = json else {
            throw Invalid(description: "the body must be a JSON object")
        }
        guard let state = json["state"] else { throw Invalid(description: "state is required") }
        if let model = json["model"] {
            guard case .string(let name) = model else {
                throw Invalid(description: "model must be a string")
            }
            self.model = name
        } else {
            self.model = "kev-latest"
        }
        switch json["temperature"] {
        case nil, .null?: temperature = nil
        case .integer(let s)?:
            guard let t = Float(s), t > 0 else {
                throw Invalid(description: "temperature must be positive")
            }
            temperature = t
        case .number(let d)?:
            guard d > 0, d.isFinite else {
                throw Invalid(description: "temperature must be positive")
            }
            temperature = Float(d)
        default: throw Invalid(description: "temperature must be a number")
        }
        guard case .object(let pairs)? = json["questions"], !pairs.isEmpty else {
            throw Invalid(description: "questions must be a non-empty object")
        }
        self.state = KevRender.render(state)
        self.questions = try pairs.map { try Self.question(id: $0.0, $0.1) }
    }

    /// One question, rendered and keyed the way `kev.api.to_record` and
    /// `question_keys` do it.
    static func question(id: String, _ q: OrderedJSON) throws -> Question {
        guard case .object = q, case .string(let type)? = q["type"], let kind = Kind(rawValue: type)
        else {
            throw Invalid(description: "question \(id): type must be noul, choice or score")
        }
        let instruction = KevRender.render(q["instructions"] ?? .null)
        switch kind {
        case .noul:
            var criteria: [(String, OrderedJSON)] = []
            switch q["criteria"] {
            case nil, .null?: break
            case .object(let pairs)?: criteria = pairs
            default: throw Invalid(description: "question \(id): noul criteria must be an object")
            }
            func described(_ key: String) -> OrderedJSON? { criteria.last { $0.0 == key }?.1 }
            return Question(
                id: id, kind: kind, instruction: instruction,
                options: [
                    KevRender.optionText("no", described("false")),
                    KevRender.optionText("yes", described("true")),
                ],
                keys: ["false", "true"])
        case .choice:
            guard case .object(let pairs)? = q["criteria"],
                (1...maximumOptions).contains(pairs.count)
            else {
                throw Invalid(
                    description:
                        "question \(id): choice criteria must be an object of 1..\(maximumOptions) options"
                )
            }
            return Question(
                id: id, kind: kind, instruction: instruction,
                options: pairs.map { KevRender.optionText($0.0, $0.1) }, keys: pairs.map(\.0))
        case .score:
            guard case .array(let levels)? = q["criteria"],
                (1...maximumOptions).contains(levels.count)
            else {
                throw Invalid(
                    description:
                        "question \(id): score criteria must be a list of 1..\(maximumOptions) levels"
                )
            }
            return Question(
                id: id, kind: kind, instruction: instruction,
                options: levels.map { KevRender.render($0) },
                keys: levels.indices.map(String.init))
        }
    }

    /// The scorer's input: the state's ids and every question's, through
    /// `tokenize` (the base install's tokenizer, no special tokens added)
    /// after `escapeSpecialTokens` -- Kev's `user_tokens`.
    public func encoded(
        tokenize: (String) -> [Int32]
    ) -> (state: [Int32], questions: [DecisionModel.Question]) {
        func user(_ text: String) -> [Int32] { tokenize(KevRender.escapeSpecialTokens(text)) }
        let rows = questions.map { q in
            DecisionModel.Question(
                instruction: user(q.instruction), options: q.options.map { user($0) })
        }
        return (user(state), rows)
    }
}

extension OrderedJSON {
    /// Compact JSON with the keys in order, for a response body.
    public func serialized() -> String { write(python: false) }

    /// Python's `json.dumps(v)` with its defaults (`", "` and `": "`,
    /// `ensure_ascii`), which is what Kev tokenizes for `output_tokens`.
    public func pythonDumps() -> String { write(python: true) }

    private func write(python: Bool) -> String {
        let comma = python ? ", " : ","
        let colon = python ? ": " : ":"
        switch self {
        case .null: return "null"
        case .bool(let b): return b ? "true" : "false"
        case .integer(let s): return s
        case .number(let d): return KevRender.pythonFloat(d)
        case .string(let s): return Self.quoted(s, asciiOnly: python)
        case .array(let items):
            return "[" + items.map { $0.write(python: python) }.joined(separator: comma) + "]"
        case .object(let pairs):
            return "{"
                + pairs.map {
                    Self.quoted($0.0, asciiOnly: python) + colon + $0.1.write(python: python)
                }
                .joined(separator: comma) + "}"
        }
    }

    static func quoted(_ s: String, asciiOnly: Bool) -> String {
        var out = "\""
        for unit in s.unicodeScalars {
            switch unit {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if unit.value < 0x20 || (asciiOnly && unit.value > 0x7E) {
                    for code in String(unit).utf16 { out += String(format: "\\u%04x", code) }
                } else {
                    out.unicodeScalars.append(unit)
                }
            }
        }
        return out + "\""
    }
}

/// Kev's answers (kev/api.py `to_answers`), from the scorer's probabilities.
public enum SystemOneAnswers {
    /// `round(float(x), 4)`.
    static func round4(_ x: Double) -> Double { (x * 10_000).rounded() / 10_000 }

    /// `choice_confidence`: (p_max - 1/K) / (1 - 1/K), on p normalised to 1.
    public static func choiceConfidence(_ p: [Double]) -> Double {
        let k = Double(p.count)
        guard p.count > 1 else { return 1 }
        return ((normalized(p).max() ?? 0) - 1 / k) / (1 - 1 / k)
    }

    /// `score_confidence`: max(0, 1 - E|level - mode| / D), D the mean absolute
    /// deviation of a uniform distribution over the levels.
    public static func scoreConfidence(_ p: [Double]) -> Double {
        let levels = p.count
        guard levels > 1 else { return 1 }
        let q = normalized(p)
        let mode = q.indices.max { q[$0] < q[$1] || (q[$0] == q[$1] && $0 > $1) } ?? 0
        let center = Double(levels - 1) / 2
        let spread = (0..<levels).reduce(0.0) { $0 + abs(Double($1) - center) } / Double(levels)
        let deviation = q.enumerated().reduce(0.0) {
            $0 + $1.element * abs(Double($1.offset - mode))
        }
        return max(0, 1 - deviation / spread)
    }

    static func normalized(_ p: [Double]) -> [Double] {
        let total = p.reduce(0, +)
        return total == 0
            ? Array(repeating: 1 / Double(p.count), count: p.count) : p.map { $0 / total }
    }

    /// Probabilities at `temperature` from the head's logits, which it
    /// returned at its own `headTemperature`.
    public static func probabilities(
        logits: [Float], headTemperature: Float, temperature: Float
    ) -> [Double] {
        let scaled = logits.map { Double($0) * Double(headTemperature) / Double(temperature) }
        let top = scaled.max() ?? 0
        let e = scaled.map { exp($0 - top) }
        let sum = e.reduce(0, +)
        return e.map { $0 / sum }
    }

    /// One question's answer object, keys in Kev's order.
    public static func answer(_ q: SystemOneRequest.Question, _ p: [Double]) -> OrderedJSON {
        let best = p.indices.max { p[$0] < p[$1] || (p[$0] == p[$1] && $0 > $1) } ?? 0
        switch q.kind {
        case .noul:
            return .object([
                ("type", .string("noul")), ("noul", .number(round4(p.count > 1 ? p[1] : 0))),
            ])
        case .choice:
            return .object([
                ("type", .string("choice")), ("choice", .string(q.keys[best])),
                ("confidence", .number(round4(choiceConfidence(p)))),
                ("probabilities", .object(zip(q.keys, p).map { ($0, .number(round4($1))) })),
            ])
        case .score:
            let score = p.enumerated().reduce(0.0) { $0 + Double($1.offset) * $1.element }
            return .object([
                ("type", .string("score")), ("score", .number(round4(score))),
                ("legend", .object(zip(q.keys, q.options).map { ($0, .string($1)) })),
                ("probabilities", .object(zip(q.keys, p).map { ($0, .number(round4($1))) })),
                ("confidence", .number(round4(scoreConfidence(p)))),
            ])
        }
    }
}
