import Testing

@testable import TinyTitan

/// The probe's verdict rests on two numbers per row: which token wins and by
/// how much. Both must be exact, including at a tie, or a near-tie flip could
/// be misread as a disagreement (or hidden as an agreement).
@Suite struct VerifyPairProbeTests {
    static func top2(_ values: [Float]) -> (token: Int32, margin: Float) {
        let halves = values.map { Float16($0) }
        return halves.withUnsafeBufferPointer { VerifyPairProbe.top2($0) }
    }

    @Test func theWinnerAndItsLeadOverTheRunnerUp() {
        let pick = Self.top2([0.5, 3.0, -1.0, 2.25, 1.0])
        #expect(pick.token == 1)
        #expect(pick.margin == 0.75)
    }

    @Test func aRunnerUpAfterTheWinnerStillCounts() {
        // The runner-up comes after the winner, so it is only ever compared
        // against `second`, never promoted.
        let pick = Self.top2([4.0, 1.0, 3.5])
        #expect(pick.token == 0)
        #expect(pick.margin == 0.5)
    }

    @Test func anExactTieKeepsTheFirstIndexWithZeroMargin() {
        // The same rule as the runner's argmax: strictly greater wins.
        let pick = Self.top2([1.0, 2.0, 2.0])
        #expect(pick.token == 1)
        #expect(pick.margin == 0)
    }

    @Test func mismatchesAreThePositionsWhereThePathsPickDifferently() {
        let rows = [
            VerifyPairProbe.Row(decodeToken: 5, decodeMargin: 1, verifyToken: 5, verifyMargin: 1),
            VerifyPairProbe.Row(
                decodeToken: 7, decodeMargin: 0.1, verifyToken: 9, verifyMargin: 0.05),
            VerifyPairProbe.Row(decodeToken: 2, decodeMargin: 3, verifyToken: 2, verifyMargin: 3),
        ]
        #expect(VerifyPairProbe(contextTokens: 4, rows: rows).mismatches == [1])
    }
}
