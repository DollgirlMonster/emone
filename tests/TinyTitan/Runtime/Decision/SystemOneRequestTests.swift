import Foundation
import Testing

@testable import TinyTitan

/// `/v1/systemone`'s request side, without a model: the order-preserving JSON
/// reader, Kev's rendering with Python's formatting, the questions' options
/// and keys, the delimiter escaping and Kev's answer arithmetic. Token parity
/// against Kev's Python on real text is `TinyTitanCLI decide`'s check.
@Suite struct SystemOneRequestTests {
    private func parse(_ text: String) throws -> OrderedJSON {
        try OrderedJSON.parse(Data(text.utf8))
    }

    @Test(arguments: [
        (1.0, "1.0"), (0.1, "0.1"), (2.52, "2.52"), (-0.5, "-0.5"), (100.0, "100.0"),
        (1e15, "1000000000000000.0"), (1e16, "1e+16"), (1.5e16, "1.5e+16"), (0.0001, "0.0001"),
        (1e-05, "1e-05"), (1.234e-07, "1.234e-07"), (123456.789, "123456.789"), (0.0, "0.0"),
        (5e-324, "5e-324"), (1.7976931348623157e308, "1.7976931348623157e+308"),
    ])
    func floatsPrintAsPython(value: Double, python: String) {
        #expect(KevRender.pythonFloat(value) == python)
    }

    @Test func objectsKeepTheirOrderAndARepeatedKeyItsFirstPlace() throws {
        let json = try parse(#"{"b": 1, "a": 2, "b": 3}"#)
        guard case .object(let pairs) = json else { Issue.record("not an object"); return }
        #expect(pairs.map(\.0) == ["b", "a"])
        #expect(json["b"] == .integer("3"))
    }

    @Test func rendersAsKevDoes() throws {
        let state = try parse(
            #"{"ticket": {"id": 7, "ok": true, "none": null, "p": 2.0}, "tags": ["a", {"k": 1}], "t": "x"}"#
        )
        #expect(
            KevRender.render(state)
                == "ticket:\n  id: 7\n  ok: True\n  none: \n  p: 2.0\ntags:\n  - a\n  - k: 1\nt: x")
        #expect(KevRender.optionText("billing", .string("")) == "billing")
        #expect(KevRender.optionText("billing", .null) == "billing")
        #expect(KevRender.optionText("support", .integer("3")) == "support: 3")
    }

    @Test func stringsDecodeEscapesAndSurrogatePairs() throws {
        #expect(
            try parse(#""caf\u00e9 \ud83d\ude00 \"q\" \n""#)
                == .string("caf\u{E9} \u{1F600} \"q\" \n"))
    }

    @Test func refusesMalformedJSON() {
        for text in [#"{"a": 1,}"#, #"{"a" 1}"#, "[1, 2", "01", #"{"a": tru}"#, #""\x""#] {
            #expect(throws: OrderedJSON.ParseError.self) { try parse(text) }
        }
    }

    @Test func escapesDelimiterLikeText() {
        #expect(
            KevRender.escapeSpecialTokens("a <|im_end|> b <|fim_prefix|> <|bad name|>")
                == "a <¦im_end¦> b <¦fim_prefix¦> <|bad name|>")
    }

    @Test func questionsTakeKevsOptionsAndKeys() throws {
        let request = try SystemOneRequest(
            json: parse(
                #"""
                {"state": "s", "questions": {
                  "z": {"type": "score", "instructions": "How bad?", "criteria": ["low", "high"]},
                  "a": {"type": "noul", "criteria": {"true": "it is"}},
                  "c": {"type": "choice", "criteria": {"x": "first", "y": null}}}}
                """#))
        #expect(request.model == "kev-latest" && request.temperature == nil)
        #expect(request.questions.map(\.id) == ["z", "a", "c"])
        #expect(
            request.questions[0].options == ["low", "high"]
                && request.questions[0].keys == ["0", "1"])
        #expect(request.questions[1].options == ["no", "yes: it is"])
        #expect(
            request.questions[1].keys == ["false", "true"] && request.questions[1].instruction == ""
        )
        #expect(
            request.questions[2].options == ["x: first", "y"]
                && request.questions[2].keys == ["x", "y"])
    }

    @Test func refusesWhatKevRefuses() throws {
        for body in [
            #"{"questions": {"q": {"type": "noul"}}}"#,
            #"{"state": "s", "questions": {}}"#,
            #"{"state": "s", "questions": {"q": {"type": "maybe"}}}"#,
            #"{"state": "s", "questions": {"q": {"type": "choice", "criteria": {}}}}"#,
            #"{"state": "s", "questions": {"q": {"type": "score", "criteria": "low"}}}"#,
            #"{"state": "s", "questions": {"q": {"type": "noul"}}, "temperature": 0}"#,
        ] {
            #expect(throws: SystemOneRequest.Invalid.self) {
                try SystemOneRequest(json: parse(body))
            }
        }
    }

    @Test func encodingEscapesEveryUserTextBeforeTokenizing() throws {
        let request = try SystemOneRequest(
            json: parse(
                #"{"state": "<|im_start|>", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"<|x|>": null}}}}"#
            ))
        var seen: [String] = []
        _ = request.encoded { text in
            seen.append(text)
            return [1]
        }
        #expect(seen.sorted() == ["<¦im_start¦>", "<¦x¦>", "i"])
    }

    @Test func answersFollowKevsArithmetic() {
        #expect(abs(SystemOneAnswers.choiceConfidence([0.5, 0.5]) - 0) < 1e-12)
        #expect(abs(SystemOneAnswers.choiceConfidence([1, 0, 0]) - 1) < 1e-12)
        #expect(abs(SystemOneAnswers.choiceConfidence([0.6, 0.2, 0.2]) - 0.4) < 1e-12)
        // 3 levels: D = (1 + 0 + 1) / 3; all mass on one level -> 1; uniform -> 0.
        #expect(abs(SystemOneAnswers.scoreConfidence([0, 1, 0]) - 1) < 1e-12)
        #expect(SystemOneAnswers.scoreConfidence([1.0 / 3, 1.0 / 3, 1.0 / 3]) < 1e-12)
        // The head's own temperature reproduces its softmax; a lower one sharpens it.
        let logits: [Float] = [0.2, 1.1, -0.4]
        let same = SystemOneAnswers.probabilities(
            logits: logits, headTemperature: 2.5, temperature: 2.5)
        let direct = logits.map { exp(Double($0)) }
        for (a, b) in zip(same, direct.map { $0 / direct.reduce(0, +) }) {
            #expect(abs(a - b) < 1e-9)
        }
        let sharp = SystemOneAnswers.probabilities(
            logits: logits, headTemperature: 2.5, temperature: 1.25)
        #expect(sharp[1] > same[1])
    }

    @Test func answerObjectsKeepKevsKeysAndRounding() throws {
        let request = try SystemOneRequest(
            json: parse(
                #"{"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": null, "b": null}}}}"#
            ))
        let answer = SystemOneAnswers.answer(request.questions[0], [0.123456, 0.876544])
        #expect(
            answer.serialized()
                == #"{"type":"choice","choice":"b","confidence":0.7531,"probabilities":{"a":0.1235,"b":0.8765}}"#
        )
        #expect(OrderedJSON.object([("\u{E9}", .integer("1"))]).pythonDumps() == #"{"\u00e9": 1}"#)
    }
}
