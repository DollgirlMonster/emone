import Foundation
import Metal

/// A short prefill off a held sequence: a Rev2 System One question row (50-130
/// tokens) or a DFlash verify block (~16 rows).
///
/// Measured on Qwen 3.6 35B-A3B 4-bit (M1 Max, 825 lmw.15 questions, CLI
/// `features`): ~390 ms fixed per call and 1.33 ms per token, so a short
/// decision is ~85% fixed cost. Two things in the ordinary chunk path put a
/// short chunk on the host-planned routed tiles, which wait on the GPU several
/// times in each of the 40 layers (router readback, shared branch, every tile,
/// the tail):
///
/// - the per-row routed stage (`encodeResidentRoutedRowsPrefill`, no readback)
///   takes at most 64 rows, and a typical question row is 72;
/// - it needs every routed expert of every layer resident, and nothing on the
///   CLI's path fills the cache: prefill asks with `warm: false`, the idle warm
///   is the server's, and a `features` run never decodes.
///
/// This entry point removes both for one call. It raises the row limit to
/// `residentPrefillMaxBlockedRows` (128: the routed rows run as blocks of 64),
/// fills the cache first when it can hold every expert, and leaves the last
/// layer unwaited so the head's wait is the call's only one. Everything else is
/// the ordinary chunk: same scratch, same attention / GDN / shared-expert
/// kernels, same KV and GDN writes, same head. The per-row routed arithmetic is
/// decode's, which rounds differently from the tiles (see
/// `RealForwardRunner+ResidentPrefill`), so a branch that used to fall back to
/// tiles now produces slightly different rows. Features captured on the old
/// path (lmw.15) are not bit-comparable with rows read through this one.
///
/// What it assumes of the caller (the fork / branch-point code):
/// - the KV cursor of `slot` is at `startPosition` (the held point), and the
///   caller has already saved whatever it will restore: this call advances the
///   cursor by `tokens.count`, writes KV rows [startPosition, +count) and
///   updates the GDN state and conv tails in place, exactly as `prefillChunked`
///   does. Rewind (KV cursor back, GDN state copied back) is the caller's.
/// - it holds no `forwardStepGate` when it calls `prefillShortBranch`; a
///   caller already under the gate calls `runShortBranchPrefill`.
/// - the model is one whose cache can hold every routed expert (Qwen 3.6 /
///   Ornith 4-bit on 64 GB). For a model that streams experts from SSD (the
///   125B), routing has to reach the host to plan the reads, so the call takes
///   the tile path and the report says why. No QSA / PLE branch support is
///   added here: those families are refused by the fork code, not by this.
///
/// `EMONE_SHORT_BRANCH=1` routes every short `prefillChunked` call here,
/// so an existing harness can A/B the path without code changes.
/// `EMONE_SHORT_BRANCH_TRACE=1` writes one stderr line per call: rows, the
/// routed stage it ran and why, warm time, wall time, and the GPU's busy and
/// elapsed time over the layers, which is the split the fixed-cost hypothesis
/// needs (wall far above GPU elapsed = host waits; GPU busy near elapsed and
/// still large = kernel cost at small t).
public struct ShortBranchPrefillOptions: Sendable, Equatable {
    /// Longest call that is still a short branch; longer calls go through
    /// the ordinary path untouched. Capped at 128.
    public var maxRows: Int
    /// Fill the routed-expert cache before the call when it can hold every
    /// expert and is not full. The first call after a load pays for the
    /// read (~14 GB on Qwen 3.6 4-bit, ~7 s cold); later calls nothing.
    public var warmToResidency: Bool

    public init(maxRows: Int = 128, warmToResidency: Bool = true) {
        self.maxRows = maxRows
        self.warmToResidency = warmToResidency
    }
}

/// Which routed stage a short branch ran.
public enum ShortBranchRoutedStage: Sendable, Equatable {
    /// Per-row decode kernels, routing resolved on the GPU, no host wait.
    case residentRows
    /// The routed stage grouped by expert on the GPU (EMONE_SHORT_BRANCH_GROUPED).
    case residentGrouped
    /// The host-planned tiles, with the reason the per-row stage was refused.
    case hostPlannedTiles(reason: String)
}

public struct ShortBranchPrefillReport: Sendable, Equatable {
    public let result: PrefillResult
    public let rows: Int
    public let routedStage: ShortBranchRoutedStage
    /// Experts read into the cache by this call's warm, and its time.
    public let warmedExperts: UInt64
    public let warmNanos: UInt64
    /// The whole call, warm excluded.
    public let wallNanos: UInt64
    /// The layers' command buffers on the GPU (resident stage only; 0 on the
    /// tile path, whose buffers are timed under EMONE_KERNEL_STATS).
    public let gpuBusyNanos: UInt64
    public let gpuElapsedNanos: UInt64
    /// Rows above which this call's routed stage was routed to the grouped stage.
    public let groupedMinRows: Int
    /// Rows per QMM tile of the grouped stage (meaningful when it ran).
    public let groupedEdge: Int
}

/// Which routed stage a resident short branch takes: per row up to
/// `groupedMinRows(heldContext:)` rows, grouped by expert above, on 32-row
/// QMM tiles (`groupedEdge`).
///
/// The per-row stage reads every picked expert once per row; the grouped one
/// once per call but pays a fixed cost for its tiles, so it wins past a
/// crossover that moves up with the held context. Measured on the M1 Max,
/// Qwen 3.6 35B-A3B 4-bit, `fastpath` exact-row sweep, 2026-11-21 (median of 5
/// per process, median of 3 interleaved rounds; per-row / grouped at edge 32,
/// ms):
/// - N = 1024: 16 rows 137.1 / 137.9, 24 rows 178.2 / 154.1 -> 15 rows (a
///   tie at 16).
/// - N = 4096 (two rounds): 16 rows 139.5-143.4 / 136.2-136.4 -> grouped from
///   16; 8 rows not measured, so 15 as at 1k.
/// - N = 16384: 16 rows 148.7 / 163.5, 24 rows 191.5 / 184.8 -> 21 rows.
/// Between 4k and 16k the threshold is interpolated in log2(N), not measured.
/// Edge 32 is the fastest grouped edge from 16 rows up (16 is within 1.7 ms at
/// 8 rows, where per-row wins anyway; 64 is 100-130 ms slower at 96-128 rows).
public enum ShortBranchRouting {
    static let nearContext = 4096
    static let nearRows = 15
    static let farContext = 16384
    static let farRows = 21

    /// Rows above which the grouped stage is faster at `heldContext` tokens.
    public static func groupedMinRows(heldContext: Int) -> Int {
        if heldContext <= nearContext { return nearRows }
        if heldContext >= farContext { return farRows }
        let octaves = log2(Double(heldContext) / Double(nearContext))
        let span = log2(Double(farContext) / Double(nearContext))
        return Int((Double(nearRows) + Double(farRows - nearRows) * octaves / span).rounded())
    }

    /// The threshold override: `EMONE_SHORT_BRANCH_GROUPED_MIN_ROWS=<n>` (0:
    /// always grouped, a huge value: never), else `EMONE_SHORT_BRANCH_GROUPED=1`
    /// (always) or `=0` (never). Nil leaves the measured threshold.
    public static func override(environment: [String: String]) -> Int? {
        if let raw = environment["EMONE_SHORT_BRANCH_GROUPED_MIN_ROWS"],
            let rows = Int(raw.trimmingCharacters(in: .whitespaces)), rows >= 0
        {
            return rows
        }
        switch environment["EMONE_SHORT_BRANCH_GROUPED"] {
        case "1": return 0
        case "0": return Int.max
        default: return nil
        }
    }

    public static func groupedMinRows(heldContext: Int, override: Int?) -> Int {
        override ?? groupedMinRows(heldContext: heldContext)
    }

    /// The grouped stage's work-item edge (rows per QMM tile): 32 at every
    /// measured size (see the type comment).
    public static func groupedEdge(rows: Int, heldContext: Int) -> Int { 32 }

    /// `EMONE_SHORT_BRANCH_GROUPED_EDGE=16|32|64`; nil for unset or any other value.
    public static func edgeOverride(environment: [String: String]) -> Int? {
        guard let raw = environment["EMONE_SHORT_BRANCH_GROUPED_EDGE"],
            let edge = Int(raw.trimmingCharacters(in: .whitespaces)),
            ShortBranchGroupedMoE.edges.contains(edge)
        else { return nil }
        return edge
    }
}

extension RealForwardRunner {
    static let shortBranchTrace =
        ProcessInfo.processInfo.environment["EMONE_SHORT_BRANCH_TRACE"] == "1"
    /// EMONE_SHORT_BRANCH=1 sends every `prefillChunked` call of at most
    /// `residentPrefillMaxBlockedRows` tokens through this path (opt-in A/B
    /// switch; off by default, so the ordinary path is unchanged).
    static let shortBranchForChunkedPrefill =
        ProcessInfo.processInfo.environment["EMONE_SHORT_BRANCH"] == "1"
    static let shortBranchGroupedOverride = ShortBranchRouting.override(
        environment: ProcessInfo.processInfo.environment)
    static let shortBranchGroupedEdgeOverride = ShortBranchRouting.edgeOverride(
        environment: ProcessInfo.processInfo.environment)

    /// Prefills a short branch of `tokens` at `startPosition` of `slot`,
    /// holding the forward gate for the call. See the type comment above for
    /// what it changes and what it assumes.
    public func prefillShortBranch(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        slot: Int = 0,
        outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig,
        into logits: MTLBuffer,
        options: ShortBranchPrefillOptions = ShortBranchPrefillOptions()
    ) async throws -> ShortBranchPrefillReport {
        try await forwardStepGate.acquire()
        forwardActivityCount &+= 1
        do {
            let report = try await runShortBranchPrefill(
                tokens: tokens, startPosition: startPosition, slot: slot,
                outputMode: outputMode, config: config, into: logits, options: options)
            await forwardStepGate.release()
            return report
        } catch {
            await forwardStepGate.release()
            throw error
        }
    }

    /// `prefillShortBranch` for a caller that already holds the forward gate.
    func runShortBranchPrefill(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        slot: Int = 0,
        outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig,
        into logits: MTLBuffer,
        options: ShortBranchPrefillOptions
    ) async throws -> ShortBranchPrefillReport {
        let rowLimit = min(max(0, options.maxRows), Self.residentPrefillMaxBlockedRows)
        let t = tokens.count
        // One span, so the deferred last-layer wait is followed by this call's
        // head and nothing else. A longer call is not a short branch.
        let short = t > 0 && t <= rowLimit && t <= config.chunkTokens

        var warmedExperts: UInt64 = 0
        var warmNanos: UInt64 = 0
        if short, options.warmToResidency {
            (warmedExperts, warmNanos) = try warmForShortBranch(tokenCount: t, rowLimit: rowLimit)
        }

        let groupedMinRows = ShortBranchRouting.groupedMinRows(
            heldContext: startPosition, override: Self.shortBranchGroupedOverride)
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        residentPrefillDeferLastWait = short
        shortBranchGroupedChosen = short && t > groupedMinRows
        shortBranchGroupedEdge =
            Self.shortBranchGroupedEdgeOverride
            ?? ShortBranchRouting.groupedEdge(rows: t, heldContext: startPosition)
        let result: PrefillResult
        let span: ResidentPrefillGPUSpan
        do {
            result = try await runPrefillChunked(
                tokens: tokens, startPosition: startPosition, slot: slot,
                outputMode: outputMode, config: config, into: logits,
                residentRowLimit: short ? rowLimit : nil,
                onProgress: { _ in })
            residentPrefillDeferLastWait = false
            shortBranchGroupedChosen = false
            // The head has been awaited, so this only checks and times.
            span = try finishResidentPrefillBuffers()
        } catch {
            residentPrefillDeferLastWait = false
            shortBranchGroupedChosen = false
            residentPrefillRanGrouped = false
            _ = try? finishResidentPrefillBuffers()
            throw error
        }
        let wall = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started

        let stage: ShortBranchRoutedStage =
            short && residentPrefillChunk
            ? (residentPrefillRanGrouped ? .residentGrouped : .residentRows)
            : .hostPlannedTiles(
                reason: shortBranchTileReason(tokenCount: t, rowLimit: rowLimit, short: short))
        residentPrefillChunk = false
        residentPrefillRanGrouped = false
        let report = ShortBranchPrefillReport(
            result: result, rows: t, routedStage: stage,
            warmedExperts: warmedExperts, warmNanos: warmNanos, wallNanos: wall,
            gpuBusyNanos: UInt64(max(0, span.busySeconds) * 1e9),
            gpuElapsedNanos: UInt64(max(0, span.elapsedSeconds) * 1e9),
            groupedMinRows: groupedMinRows, groupedEdge: shortBranchGroupedEdge)
        if Self.shortBranchTrace { writeShortBranchTrace(report, startPosition: startPosition) }
        if Self.shortBranchRouteStats, stage == .residentRows || stage == .residentGrouped {
            writeShortBranchRouteStats(rows: t, startPosition: startPosition)
        }
        return report
    }

    /// Fills the routed-expert cache when the per-row stage would otherwise be
    /// refused only for residency and the cache has a slot for every expert.
    /// Runs under the caller's gate, in batches the idle warm uses.
    private func warmForShortBranch(tokenCount: Int, rowLimit: Int) throws -> (UInt64, UInt64) {
        guard residentPrefillRefusal(tokenCount: tokenCount, rowLimit: rowLimit) == nil,
            let slots = model.routedExpertCacheSlotCount(), slots >= cfg.numExperts,
            try !prepareResidentDecode(warm: false)
        else { return (0, 0) }
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let before = totalResidentWarmedExperts
        var layer = 0
        while try !warmIdleBatch(from: &layer) {}
        return (
            totalResidentWarmedExperts &- before,
            clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
        )
    }

    private func shortBranchTileReason(tokenCount: Int, rowLimit: Int, short: Bool) -> String {
        if !short {
            return "\(tokenCount) rows is not a short branch "
                + "(limit \(rowLimit), chunk \(prefillScratch?.layout.chunkTokens ?? 0))"
        }
        if let refusal = residentPrefillRefusal(tokenCount: tokenCount, rowLimit: rowLimit) {
            return refusal
        }
        if let slots = model.routedExpertCacheSlotCount(), slots < cfg.numExperts {
            return "expert cache holds \(slots) of \(cfg.numExperts) experts per layer"
        }
        return "not every layer's experts are resident"
    }

    /// One line per call under EMONE_SHORT_BRANCH_ROUTE_STATS=1: per layer,
    /// distinct experts, the most rows one expert took, and the 32-row work
    /// items the grouped stage uses against its padded grid.
    private func writeShortBranchRouteStats(rows t: Int, startPosition: Int) {
        guard let log = shortBranchRouteStatsLog else { return }
        let k = cfg.topKExperts
        let stride = Self.residentPrefillMaxBlockedRows * k
        let ids = log.contents().assumingMemoryBound(to: UInt32.self)
        var distinct: [Int] = []
        var maxRows: [Int] = []
        var items: [Int] = []
        for L in 0..<cfg.numLayers {
            let stats = ShortBranchRouteStats(
                ids: (0..<(t * k)).map { ids[L * stride + $0] }, numExperts: cfg.numExperts,
                edge: 32)
            distinct.append(stats.distinct)
            maxRows.append(stats.maxRows)
            items.append(stats.items)
        }
        let capacity = ShortBranchGroupedMoE.workCapacity(
            pairs: t * k, numExperts: cfg.numExperts, edge: 32)
        let join = { (v: [Int]) in v.map(String.init).joined(separator: ",") }
        let line =
            "[short-branch-routes] rows=\(t) start=\(startPosition) picks=\(t * k)"
            + " distinct_sum=\(distinct.reduce(0, +)) items32_sum=\(items.reduce(0, +))"
            + " grid32=\(capacity) distinct=\(join(distinct)) maxrows=\(join(maxRows))\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    private func writeShortBranchTrace(_ report: ShortBranchPrefillReport, startPosition: Int) {
        let ms = { (n: UInt64) in String(format: "%.1f", Double(n) / 1e6) }
        let stage: String
        switch report.routedStage {
        case .residentRows: stage = "resident_rows"
        case .residentGrouped: stage = "resident_grouped"
        case .hostPlannedTiles(let reason): stage = "tiles (\(reason))"
        }
        let line =
            "[short-branch] rows=\(report.rows) start=\(startPosition) stage=\(stage)"
            + " warm_ms=\(ms(report.warmNanos)) warmed=\(report.warmedExperts)"
            + " wall_ms=\(ms(report.wallNanos))"
            + " gpu_busy_ms=\(ms(report.gpuBusyNanos))"
            + " gpu_elapsed_ms=\(ms(report.gpuElapsedNanos))"
            + " grouped_min_rows=\(report.groupedMinRows == Int.max ? "never" : String(report.groupedMinRows))"
            + " grouped_edge=\(report.groupedEdge)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }
}

/// Routing shape of one layer's picks: distinct experts, the largest expert's
/// rows, and the work items the grouped stage makes at `edge` rows each.
struct ShortBranchRouteStats: Equatable {
    let distinct: Int
    let maxRows: Int
    let items: Int

    init(distinct: Int, maxRows: Int, items: Int) {
        self.distinct = distinct
        self.maxRows = maxRows
        self.items = items
    }

    init(ids: [UInt32], numExperts: Int, edge: Int) {
        var counts = [Int](repeating: 0, count: max(1, numExperts))
        for id in ids { counts[min(Int(id), counts.count - 1)] += 1 }
        distinct = counts.filter { $0 > 0 }.count
        maxRows = counts.max() ?? 0
        items = counts.reduce(0) { $0 + ($1 + edge - 1) / edge }
    }
}
