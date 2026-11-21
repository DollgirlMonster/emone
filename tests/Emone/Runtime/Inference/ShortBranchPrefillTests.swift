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
}
