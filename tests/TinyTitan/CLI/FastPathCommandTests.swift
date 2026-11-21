import XCTest

@testable import TinyTitan
@testable import TinyTitanCLICore

/// The lmw.13 timing harness's pure pieces: argument parsing, the question
/// and branch-row layout the MoE arm prefills, and the summary statistic.
/// None of this loads a model.
final class FastPathCommandTests: XCTestCase {
    func testParsesBothArmsAndRejectsMismatchedAdapter() {
        let rev = FastPathCommand.parse([
            "--arm", "rev", "--model", "b", "--adapter", "a", "--corpus", "c", "--n", "4096,1024",
            "--m", "0", "--k", "2", "--reps", "3",
        ])
        XCTAssertEqual(
            rev?.ns, [1_024, 4_096], "N is sorted: the MoE arm extends the context in order")
        XCTAssertEqual(rev?.ms, [0])
        XCTAssertEqual(rev?.reps, 3)
        XCTAssertNotNil(FastPathCommand.parse(["--arm", "moe", "--model", "m", "--corpus", "c"]))
        XCTAssertNil(
            FastPathCommand.parse([
                "--arm", "moe", "--model", "m", "--adapter", "a", "--corpus", "c",
            ]),
            "an adapter belongs to the rev arm only")
        XCTAssertNil(FastPathCommand.parse(["--arm", "rev", "--model", "b", "--corpus", "c"]))
        XCTAssertNil(
            FastPathCommand.parse(["--arm", "moe", "--model", "m", "--corpus", "c", "--k", "0"]))
        XCTAssertNil(
            FastPathCommand.parse(["--arm", "moe", "--model", "m", "--corpus", "c", "--n", "0"]))
        XCTAssertNil(FastPathCommand.parse(["--arm", "moe", "--model"]))
    }

    /// The MoE arm's row must be exactly as long as Rev's and read the same
    /// offsets, or the two arms time different work.
    func testBranchRowMatchesRevLayout() {
        let pool = (0..<200).map(Int32.init)
        for k in [2, 8, 16] {
            let q = FastPathCommand.question(k: k, pool: pool)
            XCTAssertEqual(q.instruction.count, 12)
            XCTAssertEqual(q.options.count, k)
            XCTAssertTrue(q.options.allSatisfy { $0.count == 6 })
            let ours = FastPathCommand.branchRow(q, marker: -1)
            let d = DecisionModel.Delimiters(
                state: -2, question: -3, optionOpen: -4, optionClose: -5, decide: -6)
            let revRow = DecisionModel.branchRow(q, d)
            XCTAssertEqual(ours.ids.count, revRow.ids.count)
            XCTAssertEqual(ours.readouts, [revRow.decide] + revRow.options)
            // 1 + 12 + k * (6 + 2) + 1
            XCTAssertEqual(ours.ids.count, 14 + 8 * k)
        }
    }

    func testMedian() {
        XCTAssertEqual(FastPathCommand.median([3, 1, 2]), 2)
        XCTAssertEqual(FastPathCommand.median([4, 1, 3, 2]), 2.5)
        XCTAssertTrue(FastPathCommand.median([]).isNaN)
    }
}
