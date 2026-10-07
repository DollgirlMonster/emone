import Foundation
import Testing
import TinyTitan

@testable import TinyTitanLiveTrace

/// A synthetic event stream: no model, no ring, no clock.
struct SyntheticTrace {
    var batch = ExpertTraceBatch()

    /// Ticks for a duration in nanoseconds, through the same timebase the
    /// reducer converts back with, so tok/s checks hold on any Mac.
    static func ticks(nanos: UInt64) -> UInt64 {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return nanos * UInt64(info.denom) / UInt64(info.numer)
    }

    mutating func add(
        position: Int, layer: Int, ids: [Int], misses: [Int] = [], atNanos nanos: UInt64
    ) {
        var mask: UInt64 = 0
        for index in misses { mask |= 1 << UInt64(index) }
        batch.events.append(
            ExpertTraceEvent(
                ticks: Self.ticks(nanos: nanos), position: Int32(position), layer: Int32(layer),
                count: Int32(ids.count), idStart: Int32(batch.ids.count), missMask: mask))
        batch.ids.append(contentsOf: ids.map { UInt32($0) })
    }
}

@Suite struct LiveTraceModelTests {
    static let shape = ExpertGridLayoutTests.shape(experts: 512, topK: 10)

    @Test func aDenseShapeHasNoModel() {
        #expect(
            LiveTraceModel(
                shape: ExpertGridLayoutTests.shape(experts: 0, topK: 0), slotsPerLayer: 8) == nil)
    }

    @Test func picksCarryHitAndMissFromTheMask() throws {
        var model = try #require(LiveTraceModel(shape: Self.shape, slotsPerLayer: 96))
        var trace = SyntheticTrace()
        trace.add(position: 0, layer: 5, ids: [10, 20, 30], misses: [1], atNanos: 1_000)
        model.apply(trace.batch)
        #expect(model.layer == 5)
        #expect(
            model.picks == [
                .init(expert: 10, miss: false), .init(expert: 20, miss: true),
                .init(expert: 30, miss: false),
            ])
        #expect(model.layerMisses[5] == 1)
        #expect(model.layerMisses[4] == -1)
        #expect(model.layerHitFraction == 2.0 / 3.0)
    }

    @Test func aNewTokenClearsTheRibbonAndRecordsRates() throws {
        var model = try #require(LiveTraceModel(shape: Self.shape, slotsPerLayer: 96))
        var trace = SyntheticTrace()
        // Token 0 starts at t=0 and token 1 at t=50 ms: 20 tok/s. Token 0 read
        // 4 experts of 1 MB in that time: 4 MB / 50 ms = 0.08 GB/s.
        trace.add(position: 0, layer: 0, ids: [1, 2], misses: [0, 1], atNanos: 0)
        trace.add(position: 0, layer: 1, ids: [3, 4], misses: [0, 1], atNanos: 10_000_000)
        trace.add(position: 1, layer: 0, ids: [1, 2], atNanos: 50_000_000)
        model.apply(trace.batch)
        #expect(model.position == 1)
        #expect(model.layerMisses[0] == 0)
        #expect(model.layerMisses[1] == -1)
        #expect(model.tokensSeen == 1)
        let rate = try #require(model.tokensPerSecond.last)
        #expect(abs(rate - 20) < 0.01)
        let ssd = try #require(model.ssdGBPerSecond.last)
        #expect(abs(ssd - 0.08) < 0.0001)
    }

    @Test func hitRateWindowsOverRecentTokens() throws {
        var model = try #require(LiveTraceModel(shape: Self.shape, slotsPerLayer: 96))
        var trace = SyntheticTrace()
        #expect(model.hitRate == nil)
        trace.add(position: 0, layer: 0, ids: [1, 2, 3, 4], misses: [0], atNanos: 0)
        trace.add(position: 1, layer: 0, ids: [1, 2, 3, 4], atNanos: 1_000)
        model.apply(trace.batch)
        // token 0: 3 of 4 hit; token 1 so far: 4 of 4.
        #expect(model.hitRate == 7.0 / 8.0)
    }

    @Test func cachedShadingMirrorsPicksAndEvictsLeastRecentlyPicked() throws {
        var model = try #require(LiveTraceModel(shape: Self.shape, slotsPerLayer: 2))
        var trace = SyntheticTrace()
        trace.add(position: 0, layer: 0, ids: [1], misses: [0], atNanos: 0)
        trace.add(position: 1, layer: 0, ids: [2], misses: [0], atNanos: 100)
        model.apply(trace.batch)
        var cells = model.cells(layer: 0)
        // Expert 1 is cached but not picked now; 2 is the live pick.
        #expect(cells[1].cached && cells[1].pick == .none)
        #expect(cells[2].cached && cells[2].pick == .miss)
        // A third resident pushes out the least recently picked (expert 1).
        var more = SyntheticTrace()
        more.add(position: 2, layer: 0, ids: [3], misses: [0], atNanos: 200)
        model.apply(more.batch)
        cells = model.cells(layer: 0)
        #expect(!cells[1].cached)
        #expect(cells[2].cached)
        #expect(cells[3].pick == .miss)
        // Evicted does not mean cold: heat is pick history, residency is separate.
        #expect(cells[1].heat > 0)
        // Other layers are untouched.
        #expect(model.cells(layer: 1).allSatisfy { !$0.cached && $0.heat == 0 && $0.pick == .none })
    }

    @Test func bothLayersAreIndependent() throws {
        var model = try #require(LiveTraceModel(shape: Self.shape, slotsPerLayer: 96))
        var trace = SyntheticTrace()
        trace.add(position: 0, layer: 0, ids: [4, 5], misses: [1], atNanos: 0)
        trace.add(position: 1, layer: 0, ids: [4], atNanos: 100)
        model.apply(trace.batch)
        let cells = model.cells(layer: 0)
        // 4: picked now (hit) over two picks' worth of heat. 5: no pick, still warm.
        #expect(cells[4].pick == .hit && cells[4].heat > cells[5].heat)
        #expect(cells[5].pick == .none && cells[5].heat > 0)
        #expect(cells[6].heat == 0)
        #expect(cells[0].pick == .none)
    }

    @Test func heatIsLogScaledAndFadesWithTokens() {
        let heat = RecentPickHeat(layers: 2, experts: 8, halfLifeTokens: 24)
        #expect(heat.heat(layer: 0, expert: 3) == 0)
        heat.notePick(layer: 0, expert: 3)
        let one = heat.heat(layer: 0, expert: 3)
        #expect(one > 0.15 && one < 0.25)
        // Log scale: ten picks in a row is well short of ten times one pick.
        for _ in 0..<9 { heat.notePick(layer: 0, expert: 3) }
        let ten = heat.heat(layer: 0, expert: 3)
        #expect(ten > one && ten < one * 4)
        #expect(ten <= 1)
        // A steady favourite saturates at 1; nothing exceeds it.
        for _ in 0..<2000 { heat.advanceToken(); heat.notePick(layer: 0, expert: 4) }
        #expect(heat.heat(layer: 0, expert: 4) > 0.97 && heat.heat(layer: 0, expert: 4) <= 1)
        // Other layers and experts are cold.
        #expect(heat.heat(layer: 1, expert: 3) == 0)
    }

    @Test func aPickHalvesItsScoreEveryHalfLife() {
        let heat = RecentPickHeat(layers: 1, experts: 4, halfLifeTokens: 24)
        heat.notePick(layer: 0, expert: 1)
        for _ in 0..<24 { heat.advanceToken() }
        let ceiling = 1 / (1 - pow(0.5, 1.0 / 24.0))
        let expected = log1p(0.5) / log1p(ceiling)
        #expect(abs(heat.heat(layer: 0, expert: 1) - expected) < 0.001)
    }

    @Test func theHeatSourceIsReplaceable() throws {
        final class Fixed: ExpertHeatSource {
            var advanced = 0
            var picks = 0
            func advanceToken() { advanced += 1 }
            func notePick(layer: Int, expert: Int) { picks += 1 }
            func heat(layer: Int, expert: Int) -> Double { expert == 9 ? 0.5 : 0 }
        }
        let fixed = Fixed()
        var model = try #require(LiveTraceModel(shape: Self.shape, slotsPerLayer: 96, heat: fixed))
        var trace = SyntheticTrace()
        trace.add(position: 0, layer: 0, ids: [1, 2], atNanos: 0)
        trace.add(position: 1, layer: 0, ids: [3], atNanos: 100)
        model.apply(trace.batch)
        #expect(fixed.advanced == 2 && fixed.picks == 3)
        #expect(model.cells(layer: 0)[9].heat == 0.5)
    }

    @Test func aMissWinsInsideABinnedCell() throws {
        let shape = ExpertGridLayoutTests.shape(experts: 1024, topK: 4)
        var model = try #require(LiveTraceModel(shape: shape, slotsPerLayer: 100))
        var trace = SyntheticTrace()
        // Experts 10 and 11 share a cell at bin size 2.
        trace.add(position: 0, layer: 0, ids: [10, 11], misses: [1], atNanos: 0)
        model.apply(trace.batch)
        #expect(model.cells(layer: 0)[5].pick == .miss)
    }

    @Test func outOfRangeLayersAndExpertsAreSkipped() throws {
        var model = try #require(LiveTraceModel(shape: Self.shape, slotsPerLayer: 96))
        var trace = SyntheticTrace()
        trace.add(position: 0, layer: 99, ids: [1], atNanos: 0)
        trace.add(position: 0, layer: 2, ids: [1, 9999], atNanos: 1)
        model.apply(trace.batch)
        #expect(model.layer == 2)
        #expect(model.picks == [.init(expert: 1, miss: false)])
    }

    @Test func gridFollowsTheModelsTopKAndExpertCount() throws {
        var model = try #require(LiveTraceModel(shape: Self.shape, slotsPerLayer: 96))
        var trace = SyntheticTrace()
        trace.add(
            position: 0, layer: 0, ids: [0, 63, 64, 255, 256, 300, 400, 450, 500, 511],
            misses: [9], atNanos: 0)
        model.apply(trace.batch)
        #expect(model.picks.count == 10)
        let cells = model.cells(layer: 0)
        #expect(cells.count == 32 * 16)
        // 512 experts at one expert per cell: the last expert is the last cell.
        #expect(cells[511].pick == .miss)
        #expect(cells[0].pick == .hit)
    }
}
