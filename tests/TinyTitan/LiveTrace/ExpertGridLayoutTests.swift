import Testing
import TinyTitan

@testable import TinyTitanLiveTrace

@Suite struct ExpertGridLayoutTests {
    static func shape(experts: Int, topK: Int, layers: Int = 48) -> ExpertTraceShape {
        ExpertTraceShape(
            modelName: "m", routedExpertBits: 4, layers: layers, numExperts: experts,
            topK: topK, layerKinds: (0..<layers).map { $0 % 4 == 3 ? 1 : 2 },
            expertBytes: 1_000_000)
    }

    @Test func two56ExpertsAreSixteenBySixteenTwoCharacterCells() throws {
        let grid = try #require(ExpertGridLayout.make(numExperts: 256))
        #expect((grid.cols, grid.rows, grid.cellWidth, grid.bin) == (16, 16, 2, 1))
        #expect(grid.bodyWidth == 32)
    }

    @Test func five12ExpertsAreThirtyTwoBySixteenOneCharacterCells() throws {
        let grid = try #require(ExpertGridLayout.make(numExperts: 512))
        #expect((grid.cols, grid.rows, grid.cellWidth, grid.bin) == (32, 16, 1, 1))
        // Same footprint as the 256-expert grid, so the router panel never moves.
        #expect(grid.bodyWidth == 32)
    }

    @Test func fewerExpertsGiveFewerRows() throws {
        let grid = try #require(ExpertGridLayout.make(numExperts: 128))
        #expect((grid.cols, grid.rows, grid.cellWidth) == (16, 8, 2))
        let small = try #require(ExpertGridLayout.make(numExperts: 60))
        #expect((small.cols, small.rows) == (16, 4))
        let tiny = try #require(ExpertGridLayout.make(numExperts: 8))
        #expect((tiny.cols, tiny.rows) == (8, 1))
        let odd = try #require(ExpertGridLayout.make(numExperts: 300))
        #expect((odd.cols, odd.rows, odd.cellWidth) == (32, 10, 1))
    }

    @Test func moreThan512ExpertsAreBinned() throws {
        let grid = try #require(ExpertGridLayout.make(numExperts: 1024))
        #expect((grid.cols, grid.rows, grid.cellWidth, grid.bin) == (32, 16, 1, 2))
        #expect(grid.cell(of: 1023) == 511)
        let huge = try #require(ExpertGridLayout.make(numExperts: 4000))
        #expect(huge.rows <= ExpertGridLayout.maxRows)
        #expect(huge.bodyWidth <= ExpertGridLayout.maxGridChars)
    }

    @Test func aDenseModelHasNoGrid() {
        #expect(ExpertGridLayout.make(numExperts: 0) == nil)
    }

    @Test(arguments: [8, 60, 128, 256, 300, 512, 1024, 4000])
    func everyExpertLandsInExactlyOneCell(experts: Int) throws {
        let grid = try #require(ExpertGridLayout.make(numExperts: experts))
        var covered = [Int](repeating: 0, count: experts)
        for cell in 0..<(grid.cols * grid.rows) {
            guard let range = grid.experts(inCell: cell) else { continue }
            for expert in range {
                covered[expert] += 1
                #expect(grid.cell(of: expert) == cell)
            }
        }
        #expect(covered.allSatisfy { $0 == 1 })
        #expect(grid.rows * grid.cols >= grid.usedCells)
        #expect(grid.bodyWidth <= ExpertGridLayout.maxGridChars)
        #expect(grid.rows <= ExpertGridLayout.maxRows)
    }

    // MARK: view plan

    @Test func qwen38ShapeIsThirtyRowsAndNeedsThirtyOne() {
        let shape = Self.shape(experts: 512, topK: 10)
        #expect(LiveTraceViewPlan.fullHeight(shape: shape) == 30)
        #expect(
            LiveTraceViewPlan.plan(shape: shape, cols: 78, rows: 31)
                == .full(width: 78, height: 30))
        #expect(
            LiveTraceViewPlan.plan(shape: shape, cols: 200, rows: 60)
                == .full(width: 78, height: 30))
    }

    @Test func topKAndExpertCountFollowTheModel() {
        // top-8 and 256 experts: same footprint as the shipped 35B models.
        #expect(LiveTraceViewPlan.fullHeight(shape: Self.shape(experts: 256, topK: 8)) == 30)
        // A 64-expert model has a short grid, so the router panel sets the height.
        let small = Self.shape(experts: 64, topK: 6)
        let grid = ExpertGridLayout.make(numExperts: 64)!
        #expect(LiveTraceViewPlan.gridBlockHeight(grid: grid, topK: 6) == 10 + 3)
        #expect(LiveTraceViewPlan.fullHeight(shape: small) == 1 + 1 + 13 + 1 + 1 + 1 + 2 + 1 + 4)
        // More picks than the panel lists are capped, not allowed to grow it.
        #expect(LiveTraceViewPlan.pickRows(topK: 64) == LiveTraceViewPlan.maxPickRows)
    }

    @Test func aShortTerminalFallsBackToTheCompactView() {
        let shape = Self.shape(experts: 512, topK: 10)
        #expect(
            LiveTraceViewPlan.plan(shape: shape, cols: 78, rows: 30)
                == .compact(width: 78, height: LiveTraceViewPlan.compactHeight))
        #expect(
            LiveTraceViewPlan.plan(shape: shape, cols: 77, rows: 60)
                == .compact(width: 77, height: LiveTraceViewPlan.compactHeight))
    }

    @Test func aDenseModelGetsOutputAndLogOnly() {
        let dense = Self.shape(experts: 0, topK: 0)
        #expect(LiveTraceViewPlan.fullHeight(shape: dense) == nil)
        #expect(
            LiveTraceViewPlan.plan(shape: dense, cols: 120, rows: 50)
                == .compact(width: 78, height: 9))
    }

    @Test func aTinyTerminalHasNoView() {
        let shape = Self.shape(experts: 512, topK: 10)
        if case .unavailable = LiveTraceViewPlan.plan(shape: shape, cols: 39, rows: 50) {
        } else {
            Issue.record("39 columns should be unavailable")
        }
        if case .unavailable = LiveTraceViewPlan.plan(shape: shape, cols: 100, rows: 9) {
        } else {
            Issue.record("9 rows should be unavailable")
        }
        // 0x0 is what a failed size query reports.
        if case .unavailable = LiveTraceViewPlan.plan(shape: shape, cols: 0, rows: 0) {
        } else {
            Issue.record("an unknown size should be unavailable")
        }
    }
}
