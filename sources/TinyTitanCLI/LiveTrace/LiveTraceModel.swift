import Foundation
import TinyTitan

/// What one grid cell shows: two independent layers.
struct ExpertCell: Equatable, Sendable {
    enum Pick: Equatable, Sendable { case none, hit, miss }

    /// Foreground layer: the current token's pick in this cell.
    var pick = Pick.none
    /// Background layer: heat from the model's `ExpertHeatSource`, 0...1.
    var heat = 0.0
    /// Believed resident in the layer's cache (a mirror, see `LiveTraceModel`).
    var cached = false
    /// Past the last expert.
    var padding = false
}

/// Where the grid's background heat comes from. The view asks only this, so a
/// long-run accumulated map can replace the recent window without touching the
/// model or the frame.
///
/// Heat is display only. It is computed on the render thread from the hit/miss
/// events the view already receives, and nothing the engine does reads it:
/// not routing, caching, residency, eviction, prefetch, or any other decision.
protocol ExpertHeatSource: AnyObject {
    /// A token boundary: the next picks belong to a new token.
    func advanceToken()
    /// The router picked `expert` in `layer` for the current token.
    func notePick(layer: Int, expert: Int)
    /// Heat of one expert, 0 (cold) to 1 (hottest).
    func heat(layer: Int, expert: Int) -> Double
}

/// Heat from recent picks: an exponentially decaying pick count per
/// (layer, expert), halving every `halfLifeTokens`, shown on a log scale so one
/// pick is visible and a steady favourite saturates instead of washing out the
/// rest. Decay is applied lazily, on read and on write.
final class RecentPickHeat: ExpertHeatSource {
    private let experts: Int
    private let decay: Float
    private let ceiling: Float
    private var score: [Float]
    private var stamp: [Int32]
    private var token: Int32 = 0

    init(layers: Int, experts: Int, halfLifeTokens: Double = 24) {
        self.experts = experts
        decay = Float(pow(0.5, 1 / halfLifeTokens))
        // The score of an expert picked on every token, in the long run.
        ceiling = 1 / (1 - decay)
        score = [Float](repeating: 0, count: layers * experts)
        stamp = [Int32](repeating: 0, count: layers * experts)
    }

    func advanceToken() { token &+= 1 }

    private func current(_ index: Int) -> Float {
        let age = Float(token &- stamp[index])
        return score[index] * pow(decay, age)
    }

    func notePick(layer: Int, expert: Int) {
        let index = layer * experts + expert
        score[index] = current(index) + 1
        stamp[index] = token
    }

    func heat(layer: Int, expert: Int) -> Double {
        let value = current(layer * experts + expert)
        guard value > 0 else { return 0 }
        return Double(min(1, log1p(value) / log1p(ceiling)))
    }
}

/// The render thread's picture of the run, built only from drained ring events.
///
/// Nothing in here reads the engine. "Cached" is mirrored: an expert becomes
/// resident when the router picks it and leaves when its layer holds more than
/// `slotsPerLayer` residents, least recently picked first. The real cache also
/// weighs use counts and prefetch adoption, and decode inherits whatever
/// prefill left resident, so the shading is an estimate; the hit and miss on a
/// pick are the engine's own.
struct LiveTraceModel {
    static let historyLength = 64
    static let windowTokens = 16

    struct Pick: Equatable, Sendable {
        let expert: Int
        let miss: Bool
    }

    let shape: ExpertTraceShape
    let grid: ExpertGridLayout
    let slotsPerLayer: Int

    /// Position of the token being routed, or -1 before the first event.
    private(set) var position: Int32 = -1
    /// The layer of the latest event.
    private(set) var layer = 0
    private(set) var picks: [Pick] = []
    /// Misses per layer for the current token; -1 for layers not reached yet.
    private(set) var layerMisses: [Int16]
    private(set) var tokensSeen = 0
    private(set) var tokensPerSecond: [Double] = []
    /// SSD read rate per token, GB/s.
    private(set) var ssdGBPerSecond: [Double] = []
    private(set) var droppedEvents: UInt64 = 0

    /// Background heat for the grid; replaceable.
    let heatSource: ExpertHeatSource
    private var resident: [Bool]
    private var lastUse: [UInt32]
    private var residentCount: [Int]
    private var useClock: UInt32 = 0
    private var tokenFirstTicks: UInt64 = 0
    private var tokenHits = 0
    private var tokenPicks = 0
    private var tokenMisses = 0
    private var window: [(hits: Int, picks: Int)] = []

    /// nil for a model with no routed experts.
    init?(shape: ExpertTraceShape, slotsPerLayer: Int?, heat: ExpertHeatSource? = nil) {
        guard shape.isRouted, let grid = ExpertGridLayout.make(numExperts: shape.numExperts)
        else { return nil }
        self.shape = shape
        self.grid = grid
        self.slotsPerLayer = max(1, slotsPerLayer ?? shape.numExperts)
        let cells = shape.layers * shape.numExperts
        layerMisses = [Int16](repeating: -1, count: shape.layers)
        heatSource =
            heat ?? RecentPickHeat(layers: shape.layers, experts: shape.numExperts)
        resident = [Bool](repeating: false, count: cells)
        lastUse = [UInt32](repeating: 0, count: cells)
        residentCount = [Int](repeating: 0, count: shape.layers)
    }

    var hasEvents: Bool { position >= 0 }

    /// Hit fraction over the last few tokens plus the one in flight, or nil
    /// before any pick.
    var hitRate: Double? {
        var hits = tokenHits
        var total = tokenPicks
        for entry in window {
            hits += entry.hits
            total += entry.picks
        }
        return total > 0 ? Double(hits) / Double(total) : nil
    }

    /// Hit fraction of the layer in flight.
    var layerHitFraction: Double? {
        guard !picks.isEmpty else { return nil }
        return Double(picks.filter { !$0.miss }.count) / Double(picks.count)
    }

    /// Mean of the last `count` per-token rates, or nil with none.
    static func recent(_ history: [Double], count: Int = 8) -> Double? {
        guard !history.isEmpty else { return nil }
        let tail = history.suffix(count)
        return tail.reduce(0, +) / Double(tail.count)
    }

    mutating func apply(_ batch: ExpertTraceBatch) {
        droppedEvents = batch.dropped
        for event in batch.events {
            let eventLayer = Int(event.layer)
            guard eventLayer >= 0, eventLayer < shape.layers else { continue }
            if event.position != position { startToken(at: event) }
            layer = eventLayer
            applyPicks(event: event, ids: batch.ids(of: event))
        }
    }

    private mutating func startToken(at event: ExpertTraceEvent) {
        if position >= 0 { finishToken(nextTicks: event.ticks) }
        heatSource.advanceToken()
        position = event.position
        tokenFirstTicks = event.ticks
        tokenHits = 0
        tokenPicks = 0
        tokenMisses = 0
        for index in layerMisses.indices { layerMisses[index] = -1 }
    }

    /// A token ends when the next one's first layer arrives, so its period is
    /// the whole step: embedding, every layer, head and sampling.
    private mutating func finishToken(nextTicks: UInt64) {
        let ticks = nextTicks &- tokenFirstTicks
        let nanos = Double(ExpertTraceRing.nanoseconds(fromTicks: ticks))
        if nanos > 0 {
            push(&tokensPerSecond, 1e9 / nanos)
            push(
                &ssdGBPerSecond,
                Double(tokenMisses) * Double(shape.expertBytes) / nanos)
        }
        window.append((tokenHits, tokenPicks))
        if window.count > Self.windowTokens { window.removeFirst() }
        tokensSeen += 1
    }

    private func push(_ history: inout [Double], _ value: Double) {
        history.append(value)
        if history.count > Self.historyLength { history.removeFirst() }
    }

    private mutating func applyPicks(event: ExpertTraceEvent, ids: ArraySlice<UInt32>) {
        picks.removeAll(keepingCapacity: true)
        var misses = 0
        useClock &+= 1
        for (index, raw) in ids.enumerated() {
            let expert = Int(raw)
            guard expert < shape.numExperts else { continue }
            let miss = index < 64 && (event.missMask >> UInt64(index)) & 1 == 1
            picks.append(Pick(expert: expert, miss: miss))
            if miss { misses += 1 }
            touch(layer: layer, expert: expert)
        }
        trimResidents(layer: layer)
        layerMisses[layer] = Int16(misses)
        tokenHits += picks.count - misses
        tokenPicks += picks.count
        tokenMisses += misses
    }

    private mutating func touch(layer: Int, expert: Int) {
        let index = layer * shape.numExperts + expert
        heatSource.notePick(layer: layer, expert: expert)
        lastUse[index] = useClock
        if !resident[index] {
            resident[index] = true
            residentCount[layer] += 1
        }
    }

    private mutating func trimResidents(layer: Int) {
        let base = layer * shape.numExperts
        while residentCount[layer] > slotsPerLayer {
            var victim = -1
            var oldest = UInt32.max
            for expert in 0..<shape.numExperts where resident[base + expert] {
                if lastUse[base + expert] < oldest {
                    oldest = lastUse[base + expert]
                    victim = expert
                }
            }
            // Everything resident was picked by this very event: nothing to evict.
            guard victim >= 0, oldest != useClock else { break }
            resident[base + victim] = false
            residentCount[layer] -= 1
        }
    }

    /// Both layers of every cell of the grid for `layer`, row-major,
    /// `grid.rows * grid.cols` entries. A binned cell shows its hottest expert's
    /// heat, and a miss wins over a hit among the picks in it.
    func cells(layer: Int) -> [ExpertCell] {
        let count = grid.rows * grid.cols
        var cells = [ExpertCell](repeating: ExpertCell(), count: count)
        let base = layer * shape.numExperts
        for cell in 0..<count {
            guard let experts = grid.experts(inCell: cell) else {
                cells[cell].padding = true
                continue
            }
            for expert in experts {
                cells[cell].heat = max(
                    cells[cell].heat, heatSource.heat(layer: layer, expert: expert))
                if resident[base + expert] { cells[cell].cached = true }
            }
        }
        // The picks belong to the layer in flight only.
        for pick in picks where layer == self.layer {
            let cell = grid.cell(of: pick.expert)
            if pick.miss {
                cells[cell].pick = .miss
            } else if cells[cell].pick == .none {
                cells[cell].pick = .hit
            }
        }
        return cells
    }
}
