import Foundation
import Testing

@testable import Emone

/// Kev's head and hidden-state fixtures are safetensors, which `.gitignore`
/// keeps out of the repository, so a fresh checkout (CI) does not have them.
/// The comparison against them is skipped there, and the skip is recorded.
private let kevTensorsPresent: Bool = {
    let kev = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().appendingPathComponent("benchmark/kev")
    return ["kev08b_head.safetensors", "kev08b_hidden.safetensors"].allSatisfy {
        FileManager.default.fileExists(atPath: kev.appendingPathComponent($0).path)
    }
}()

/// The decision model's host-side pieces, without a model: the pointer head's
/// arithmetic against Kev's own numbers, the head file's validation, the
/// delimiter lookup and the row layout the readout offsets come from.
@Suite("Decision model")
struct DecisionModelTests {
    static let kev = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().appendingPathComponent("benchmark/kev")

    /// F32 tensors and metadata of a small safetensors file.
    static func readSafetensors(_ url: URL) throws -> (
        tensors: [String: [Float]], metadata: [String: String]
    ) {
        let data = try Data(contentsOf: url)
        let length = data.withUnsafeBytes { Int($0.loadUnaligned(as: UInt64.self)) }
        let header = try #require(
            try JSONSerialization.jsonObject(with: data[8..<(8 + length)]) as? [String: Any])
        var tensors: [String: [Float]] = [:]
        for (name, value) in header where name != "__metadata__" {
            let entry = try #require(value as? [String: Any])
            let offsets = try #require(entry["data_offsets"] as? [Int])
            let start = 8 + length + offsets[0]
            tensors[name] = data[start..<(8 + length + offsets[1])].withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
        }
        return (tensors, header["__metadata__"] as? [String: String] ?? [:])
    }

    /// Kev's head on Kev's own fp32 final-normed rows (the reference fixture):
    /// the Swift arithmetic must reproduce its logits, so any gap the engine
    /// shows is the backbone's and never the head's.
    @Test(.enabled(if: kevTensorsPresent, "Needs benchmark/kev/kev08b_{head,hidden}.safetensors"))
    func theHeadReproducesKevsLogitsFromItsHiddenStates() throws {
        let head = try Self.readSafetensors(
            Self.kev.appendingPathComponent("kev08b_head.safetensors"))
        let hidden = try Self.readSafetensors(
            Self.kev.appendingPathComponent("kev08b_hidden.safetensors")
        ).tensors
        let pointer = PointerHead(
            hiddenSize: 1_024, headDim: 256,
            temperature: try #require(head.metadata["temperature"].flatMap(Float.init)),
            qWeight: try #require(head.tensors["q.weight"]),
            qBias: try #require(head.tensors["q.bias"]),
            kWeight: try #require(head.tensors["k.weight"]),
            kBias: try #require(head.tensors["k.bias"]))
        let fixture = try #require(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: Self.kev.appendingPathComponent("kev08b_fixture.json")))
                as? [String: Any])
        let questions = try #require(fixture["questions"] as? [[String: Any]])
        #expect(questions.count == 5)
        for question in questions {
            let key = try #require(question["key"] as? String)
            let expected = try #require(question["logits_row"] as? [Double]).map(Float.init)
            let decide = try #require(hidden["\(key).h_decide"])
            let flat = try #require(hidden["\(key).h_opts"])
            let options = stride(from: 0, to: flat.count, by: 1_024).map {
                Array(flat[$0..<($0 + 1_024)])
            }
            let logits = pointer.logits(decide: decide, options: options)
            #expect(logits.count == expected.count, "\(key)")
            for (got, want) in zip(logits, expected) {
                #expect(abs(got - want) < 1e-4, "\(key): \(got) vs \(want)")
            }
            let probs = PointerHead.softmax(logits)
            let reference = try #require(question["probs_row"] as? [Double]).map(Float.init)
            for (got, want) in zip(probs, reference) { #expect(abs(got - want) < 1e-5, "\(key)") }
        }
    }

    @Test func softmaxIsStableAndNormalised() {
        let p = PointerHead.softmax([1_000, 999, -1_000])
        #expect(abs(p.reduce(0, +) - 1) < 1e-6)
        #expect(p[0] > p[1] && p[2] == 0)
        #expect(PointerHead.softmax([]).isEmpty)
    }

    private func writeHead(
        _ url: URL, metadata: [String: String], hidden: Int = 8, headDim: Int = 4,
        omit: String? = nil
    ) throws {
        var header: [String: Any] = ["__metadata__": metadata]
        var payload = Data()
        for (name, count) in [
            ("q.weight", headDim * hidden), ("q.bias", headDim), ("k.weight", headDim * hidden),
            ("k.bias", headDim),
        ] where name != omit {
            let values = (0..<count).map { Float($0) / 100 }
            header[name] = [
                "dtype": "F32", "shape": [count],
                "data_offsets": [payload.count, payload.count + count * 4],
            ]
            values.withUnsafeBytes { payload.append(contentsOf: $0) }
        }
        var json = try JSONSerialization.data(withJSONObject: header)
        json.append(contentsOf: [UInt8](repeating: 0x20, count: (8 - json.count % 8) % 8))
        var file = withUnsafeBytes(of: UInt64(json.count).littleEndian) { Data($0) }
        file.append(json)
        file.append(payload)
        try file.write(to: url)
    }

    private var goodMetadata: [String: String] {
        [
            "format": "kev-pointer-v1", "temperature": "2.0", "head_dim": "4",
            "delimiters":
                #"["<|fim_prefix|>","<|fim_middle|>","<|box_start|>","<|box_end|>","<|fim_suffix|>"]"#,
        ]
    }

    @Test func aWellFormedHeadLoads() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "head-\(UUID()).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try writeHead(url, metadata: goodMetadata)
        let head = try PointerHead(contentsOf: url, hiddenSize: 8)
        #expect(head.temperature == 2 && head.headDim == 4)
        #expect(head.delimiterNames.count == 5)
        #expect(
            head.logits(
                decide: [Float](repeating: 1, count: 8), options: [[Float](repeating: 0, count: 8)]
            ).count == 1)
    }

    /// Each of these loads without error in a naive reader and scores garbage.
    @Test func aMalformedHeadIsRefused() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "head-\(UUID()).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        var cases: [(String, [String: String], Int, String?)] = []
        var other = goodMetadata
        other["format"] = "something-else"
        cases.append(("format", other, 8, nil))
        var cold = goodMetadata
        cold["temperature"] = "0"
        cases.append(("temperature", cold, 8, nil))
        cases.append(("width", goodMetadata, 16, nil))
        cases.append(("missing tensor", goodMetadata, 8, "k.bias"))
        for (name, metadata, hidden, omit) in cases {
            try writeHead(url, metadata: metadata, omit: omit)
            #expect(throws: DecisionModel.LoadError.self, "\(name)") {
                _ = try PointerHead(contentsOf: url, hiddenSize: hidden)
            }
        }
    }

    @Test func delimitersResolveFromTheTokenizersAddedTokens() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "tok-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let names = ["<a>", "<b>", "<c>", "<d>", "<e>"]
        let added = names.enumerated().map { ["content": $0.element, "id": 100 + $0.offset] }
        try JSONSerialization.data(withJSONObject: ["added_tokens": added]).write(to: url)
        let d = try DecisionModel.delimiterIDs(names: names, tokenizer: url)
        #expect(
            [d.state, d.question, d.optionOpen, d.optionClose, d.decide] == [
                100, 101, 102, 103, 104,
            ])
        #expect(throws: DecisionModel.LoadError.self) {
            _ = try DecisionModel.delimiterIDs(
                names: names.dropLast() + ["<missing>"], tokenizer: url)
        }
    }

    /// The layout kev/model.py `encode` produces, which the readout offsets
    /// index into: `<q> instr (<opt> option </opt>)... <decide>`.
    @Test func theBranchRowIsKevsLayout() {
        let d = DecisionModel.Delimiters(
            state: 1, question: 2, optionOpen: 3, optionClose: 4, decide: 5)
        #expect(DecisionModel.stateRow([9, 9], d) == [1, 9, 9])
        let row = DecisionModel.branchRow(
            DecisionModel.Question(instruction: [10, 11], options: [[20], [30, 31]]), d)
        #expect(row.ids == [2, 10, 11, 3, 20, 4, 3, 30, 31, 4, 5])
        #expect(row.options == [5, 9])
        #expect(row.decide == 10)
    }
}
