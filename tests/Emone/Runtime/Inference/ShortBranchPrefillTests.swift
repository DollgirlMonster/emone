import Testing

@testable import Emone

/// The short-branch prefill's pure parts: how a chunk's rows split into the
/// routed-rows kernels' blocks, and the GPU span sum the trace reports. The
/// path itself needs a GPU and a model, so it is checked on the M1 Max
/// (docs/short-branch-prefill.md), not here.
@Suite struct ShortBranchPrefillTests {
    @Test func rowsWithinOneBlockAreOneCall() {
        #expect(
            ResidentRowBlocks.blocks(rows: 16, blockRows: 64)
                == [ResidentRowBlocks.Block(index: 0, start: 0, count: 16)])
        #expect(
            ResidentRowBlocks.blocks(rows: 64, blockRows: 64)
                == [ResidentRowBlocks.Block(index: 0, start: 0, count: 64)])
    }

    @Test func aTypicalQuestionRowSplitsInTwo() {
        // 72 rows, the lmw.15 median question branch.
        #expect(
            ResidentRowBlocks.blocks(rows: 72, blockRows: 64) == [
                ResidentRowBlocks.Block(index: 0, start: 0, count: 64),
                ResidentRowBlocks.Block(index: 1, start: 64, count: 8),
            ])
    }

    @Test func theCapFillsExactlyTheScratchBlocks() {
        let cap = RealForwardRunner.residentPrefillMaxBlockedRows
        let blocks = ResidentRowBlocks.blocks(rows: cap, blockRows: MoE.maxRoutedRows)
        // The scratch holds cap / maxRoutedRows argument ranges; the last
        // block must land in the last of them, and every row is covered once.
        #expect(blocks.count == cap / MoE.maxRoutedRows)
        #expect(blocks.last?.index == cap / MoE.maxRoutedRows - 1)
        #expect(blocks.map(\.count).reduce(0, +) == cap)
        #expect(blocks.allSatisfy { $0.count <= MoE.maxRoutedRows })
        // Each block's pairs fit the kernels' 512-pointer argument struct.
        #expect(blocks.allSatisfy { $0.count * 8 <= 512 })
    }

    @Test func noRowsNoBlocks() {
        #expect(ResidentRowBlocks.blocks(rows: 0, blockRows: 64).isEmpty)
        #expect(ResidentRowBlocks.blocks(rows: 5, blockRows: 0).isEmpty)
    }

    @Test func gpuSpanSeparatesBusyFromElapsed() {
        var span = ResidentPrefillGPUSpan()
        #expect(span.elapsedSeconds == 0)
        span.add(start: 10.000, end: 10.004)
        span.add(start: 10.006, end: 10.010)
        // A buffer with no timestamps (never ran) is ignored.
        span.add(start: 0, end: 0)
        #expect(span.buffers == 2)
        #expect(abs(span.busySeconds - 0.008) < 1e-9)
        #expect(abs(span.elapsedSeconds - 0.010) < 1e-9)
    }

    @Test func defaultOptionsAskForTheCapAndAWarm() {
        let options = ShortBranchPrefillOptions()
        #expect(options.maxRows == RealForwardRunner.residentPrefillMaxBlockedRows)
        #expect(options.warmToResidency)
    }

    @Test func theThresholdIsTheMeasuredCrossoverAtBothEnds() {
        #expect(ShortBranchRouting.groupedMinRows(heldContext: 0) == 15)
        #expect(ShortBranchRouting.groupedMinRows(heldContext: 1024) == 15)
        #expect(ShortBranchRouting.groupedMinRows(heldContext: 4096) == 15)
        #expect(ShortBranchRouting.groupedMinRows(heldContext: 16384) == 21)
        #expect(ShortBranchRouting.groupedMinRows(heldContext: 65536) == 21)
        // Interpolated in log2(N): 8192 is halfway from 4096 to 16384.
        #expect(ShortBranchRouting.groupedMinRows(heldContext: 8192) == 18)
        #expect(ShortBranchRouting.groupedEdge(rows: 64, heldContext: 1024) == 32)
    }

    @Test func theThresholdNeverFallsAsTheContextGrows() {
        var last = 0
        for context in stride(from: 0, through: 70_000, by: 512) {
            let rows = ShortBranchRouting.groupedMinRows(heldContext: context)
            #expect(rows >= last)
            last = rows
        }
    }

    @Test func theOverrideTakesMinRowsFirstThenTheSwitch() {
        typealias R = ShortBranchRouting
        #expect(R.override(environment: [:]) == nil)
        #expect(R.override(environment: ["EMONE_SHORT_BRANCH_GROUPED_MIN_ROWS": "0"]) == 0)
        #expect(R.override(environment: ["EMONE_SHORT_BRANCH_GROUPED_MIN_ROWS": " 48 "]) == 48)
        #expect(R.override(environment: ["EMONE_SHORT_BRANCH_GROUPED": "1"]) == 0)
        #expect(R.override(environment: ["EMONE_SHORT_BRANCH_GROUPED": "0"]) == Int.max)
        #expect(R.override(environment: ["EMONE_SHORT_BRANCH_GROUPED": "yes"]) == nil)
        #expect(
            R.override(environment: [
                "EMONE_SHORT_BRANCH_GROUPED_MIN_ROWS": "20", "EMONE_SHORT_BRANCH_GROUPED": "0",
            ]) == 20)
        // Unparseable or negative counts are ignored, not read as 0.
        #expect(R.override(environment: ["EMONE_SHORT_BRANCH_GROUPED_MIN_ROWS": "-1"]) == nil)
        #expect(R.override(environment: ["EMONE_SHORT_BRANCH_GROUPED_MIN_ROWS": "lots"]) == nil)
        #expect(R.groupedMinRows(heldContext: 1024, override: 5) == 5)
        #expect(R.groupedMinRows(heldContext: 1024, override: nil) == 15)
    }

    @Test func theGroupedGridCoversTheWorstCaseWorkList() {
        // One partial item per live expert plus one per full 64 rows.
        #expect(ShortBranchGroupedMoE.workCapacity(pairs: 64, numExperts: 256) == 66)
        #expect(ShortBranchGroupedMoE.workCapacity(pairs: 1024, numExperts: 256) == 273)
    }

    @Test func theEdgeOverrideTakesOnlyAnEdgeWithKernels() {
        typealias R = ShortBranchRouting
        #expect(R.edgeOverride(environment: [:]) == nil)
        #expect(R.edgeOverride(environment: ["EMONE_SHORT_BRANCH_GROUPED_EDGE": "16"]) == 16)
        #expect(R.edgeOverride(environment: ["EMONE_SHORT_BRANCH_GROUPED_EDGE": " 32"]) == 32)
        #expect(R.edgeOverride(environment: ["EMONE_SHORT_BRANCH_GROUPED_EDGE": "64"]) == 64)
        #expect(R.edgeOverride(environment: ["EMONE_SHORT_BRANCH_GROUPED_EDGE": "8"]) == nil)
        #expect(R.edgeOverride(environment: ["EMONE_SHORT_BRANCH_GROUPED_EDGE": "big"]) == nil)
        for rows in [1, 8, 16, 37, 64, 128] {
            for context in [0, 1024, 16384, 65536] {
                #expect(
                    ShortBranchGroupedMoE.edges.contains(
                        R.groupedEdge(rows: rows, heldContext: context)))
            }
        }
    }

    @Test func aSmallerEdgeSizesTheGridForMoreItems() {
        #expect(ShortBranchGroupedMoE.workCapacity(pairs: 1024, numExperts: 256, edge: 16) == 321)
        #expect(ShortBranchGroupedMoE.workCapacity(pairs: 1024, numExperts: 256, edge: 32) == 289)
        #expect(ShortBranchGroupedMoE.workCapacity(pairs: 64, numExperts: 256, edge: 16) == 69)
    }
}
