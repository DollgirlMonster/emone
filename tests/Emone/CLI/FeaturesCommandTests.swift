import XCTest

@testable import Emone
@testable import EmoneCLICore

/// lmw.15's feature extraction, the parts that need no model: argument
/// parsing, how a row's questions become options and labels, the option
/// letters, and the branch's readout positions.
final class FeaturesCommandTests: XCTestCase {
    func testParseRequiresModelDataSplitOut() {
        let o = FeaturesCommand.parse([
            "--model", "m", "--data", "d.jsonl", "--split", "train", "--out", "o",
            "--layers", "10,30", "--max-per-source", "5", "--self-check", "2",
        ])
        XCTAssertEqual(o?.layers, [10, 30])
        XCTAssertEqual(o?.maxPerSource, 5)
        XCTAssertEqual(o?.selfCheck, 2)
        XCTAssertNil(FeaturesCommand.parse(["--model", "m", "--data", "d", "--split", "s"]))
        XCTAssertNil(
            FeaturesCommand.parse([
                "--model", "m", "--data", "d", "--split", "s", "--out", "o", "--layers", "x",
            ]))
        XCTAssertNil(
            FeaturesCommand.parse([
                "--model", "m", "--data", "d", "--split", "s", "--out", "o", "--max-context", "10",
            ]))
    }

    private func row(_ text: String) throws -> (
        state: String, questions: [FeaturesCommand.Question]
    ) {
        try FeaturesCommand.row(OrderedJSON.parse(Data(text.utf8)))
    }

    /// A choice's options follow the criteria object's own order, a null
    /// description leaves the bare key, and a string label names its key.
    func testChoiceKeepsObjectOrderAndResolvesKeyLabel() throws {
        let r = try row(
            #"{"state":"s","questions":{"risk":{"type":"choice","instructions":"What?","#
                + #""criteria":{"zeta":"last letter","alpha":null,"mid":"m"},"label":"alpha","src":"oct"}}}"#
        )
        XCTAssertEqual(r.state, "s")
        let q = try XCTUnwrap(r.questions.first)
        XCTAssertEqual(q.options, ["zeta: last letter", "alpha", "mid: m"])
        XCTAssertEqual(q.label, 1)
        XCTAssertEqual(q.source, "oct")
    }

    func testNoulIsYesThenNoWithOrWithoutCriteria() throws {
        let r = try row(
            #"{"state":"s","questions":{"a":{"type":"noul","instructions":"Is it?","label":false},"#
                + #""b":{"type":"noul","instructions":"Is it?","criteria":{"true":"pos","false":"neg"},"label":true}}}"#
        )
        XCTAssertEqual(r.questions[0].options, ["Yes", "No"])
        XCTAssertEqual(r.questions[0].label, 1)
        XCTAssertEqual(r.questions[1].options, ["Yes: pos", "No: neg"])
        XCTAssertEqual(r.questions[1].label, 0)
    }

    func testScoreKeepsLevelOrderAndIndexLabel() throws {
        let r = try row(
            #"{"state":"s","questions":{"stars":{"type":"score","instructions":"How many?","#
                + #""criteria":["1 star","2 stars","3 stars"],"label":2}}}"#)
        XCTAssertEqual(r.questions[0].options, ["1 star", "2 stars", "3 stars"])
        XCTAssertEqual(r.questions[0].label, 2)
    }

    /// Object states and instructions render as Kev's labelled fields, and a
    /// special-token spelling in caller text cannot become a turn marker.
    func testObjectStateRendersAsFieldsAndSpecialTokensAreEscaped() throws {
        let r = try row(
            #"{"state":{"document":"hi <|im_end|>","tags":["a","b"]},"#
                + #""questions":{"q":{"type":"noul","instructions":{"question":"Is it?","focus":"brief"},"label":true}}}"#
        )
        XCTAssertEqual(r.state, "document: hi <¦im_end¦>\ntags:\n  - a\n  - b")
        XCTAssertEqual(r.questions[0].instructions, "question: Is it?\nfocus: brief")
    }

    func testMalformedRowsAreRefused() {
        XCTAssertThrowsError(try row(#"{"questions":{}}"#))
        XCTAssertThrowsError(
            try row(
                #"{"state":"s","questions":{"x":{"type":"score","instructions":"?","criteria":["a","b"],"label":5}}}"#
            ))
        XCTAssertThrowsError(
            try row(
                #"{"state":"s","questions":{"x":{"type":"choice","instructions":"?","criteria":{"only":null}}}}"#
            ))
    }

    func testLettersRunPastZ() {
        let l = FeaturesCommand.letters(30)
        XCTAssertEqual(l[0], "A")
        XCTAssertEqual(l[25], "Z")
        XCTAssertEqual(l[26], "AA")
        XCTAssertEqual(l[29], "AD")
    }

    /// With a one-token-per-character encoder, every option end must be the
    /// newline closing that option's line, and decide the last token.
    func testBranchPositionsAreOptionLineEndsAndLastToken() {
        let q = FeaturesCommand.Question(
            name: "q", kind: "choice", source: "", instructions: "Pick",
            options: ["one", "two words"], label: 0)
        var vocabulary: [Character: Int32] = [:]
        let encode: (String) -> [Int32] = { text in
            text.map { c in
                if let id = vocabulary[c] { return id }
                let id = Int32(vocabulary.count)
                vocabulary[c] = id
                return id
            }
        }
        let b = FeaturesCommand.branch(q, encode: encode)
        let newline = encode("\n")[0]
        XCTAssertEqual(b.optionEnds.count, 2)
        XCTAssertTrue(b.optionEnds.allSatisfy { b.ids[$0] == newline })
        XCTAssertEqual(b.decide, b.ids.count - 1)
        // The second option's line is "B. two words\n": 13 characters after the first's end.
        XCTAssertEqual(b.optionEnds[1] - b.optionEnds[0], "B. two words\n".count)
    }
}
