import Foundation
import Metal
import Emone

extension FeatureExtractor {
    // lint:allow-long one row's whole lifecycle: filter, prefill, branch per question, write, self-check
    /// One row: prefill its state once, branch each question from it, write
    /// the rows and a record per question; optionally re-derive the first
    /// question's rows from one full prefill and compare (`--self-check`).
    func extractRow(
        _ m: Loaded, _ line: String, rowNumber: Int, encode: (String) -> [Int32],
        letterIDs: [Int32?], blob: FileHandle, index: FileHandle, tally: inout Tally
    ) async throws {
        let json: OrderedJSON
        let parsed: (state: String, questions: [FeaturesCommand.Question])
        do {
            json = try OrderedJSON.parse(Data(line.utf8))
            parsed = try FeaturesCommand.row(json)
        } catch {
            tally.skipped += 1
            if tally.skipped <= 5 { say("# skipped row \(rowNumber): \(error)") }
            return
        }
        var questions = parsed.questions
        if let cap = o.maxPerSource {
            let before = questions.count
            questions = questions.filter { tally.perSource[$0.source, default: 0] < cap }
            tally.capped += before - questions.count
        }
        guard !questions.isEmpty else { return }
        let state = encode(FeaturesCommand.stateText(parsed.state))
        let branches = questions.map { FeaturesCommand.branch($0, encode: encode) }
        guard state.count + (branches.map(\.ids.count).max() ?? 0) < o.maxContext else {
            tally.skipped += questions.count
            return
        }
        let metaID: String? = {
            if case .string(let id)? = json["_meta"]?["id"] { return id }
            return nil
        }()
        tally.rows += 1
        await m.runner.resetSequence(slot: 0)
        let stateStarted = Self.now()
        _ = try await readout(
            m, state[...], from: 0, offsets: [state.count - 1], blocks: [], keepHead: false)
        let point = try await m.runner.makeDecisionBranchPoint()
        let stateNanos = Self.now() - stateStarted
        var firstRows: DecisionReadoutRows?
        for (qi, (q, b)) in zip(questions, branches).enumerated() {
            if let limit = o.limit, tally.questions >= limit { break }
            if qi > 0 { try await m.runner.restoreDecisionBranchPoint(point) }
            let started = Self.now()
            let rows = try await readout(
                m, b.ids[...], from: state.count, offsets: [b.decide] + b.optionEnds,
                blocks: o.layers, keepHead: true)
            let branchNanos = Self.now() - started
            if qi == 0 { firstRows = rows }
            let letters = letterLogits(m, count: q.options.count, ids: letterIDs)
            let written = try write(rows, to: blob)
            let record: [String: Any] = [
                "split": o.split, "row": rowNumber, "id": metaID ?? NSNull(), "q": q.name,
                "kind": q.kind, "src": q.source, "k": q.options.count,
                "label": q.label ?? NSNull(), "offset": tally.rowsWritten, "rows": written,
                "state_tokens": state.count, "branch_tokens": b.ids.count,
                "state_ms": Double(stateNanos) / 1e6, "branch_ms": Double(branchNanos) / 1e6,
                "letters": letters ?? NSNull(),
            ]
            var line = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
            line.append(0x0A)
            index.write(line)
            tally.rowsWritten += written
            tally.questions += 1
            tally.perSource[q.source, default: 0] += 1
        }
        if tally.checked < o.selfCheck, let firstRows, let b = branches.first {
            let cosine = try await selfCheck(m, state: state, branch: b, against: firstRows)
            tally.worstCosine = min(tally.worstCosine, cosine)
            tally.checked += 1
            say("# self-check row \(rowNumber): worst row cosine \(String(format: "%.6f", cosine))")
        }
    }

    /// The zero-shot letter readout: the LM head's logits for the option
    /// letters at the decide position; nil past 26 options or for a letter
    /// that is not one token.
    func letterLogits(_ m: Loaded, count: Int, ids: [Int32?]) -> [Double]? {
        guard count <= ids.count else { return nil }
        let base = m.logits.contents().bindMemory(to: Float16.self, capacity: m.vocab)
        var values: [Double] = []
        for id in ids.prefix(count) {
            guard let id, Int(id) < m.vocab else { return nil }
            values.append(Double(base[Int(id)]))
        }
        return values
    }

    /// Appends every block's rows, then the final-normed rows, as fp16.
    func write(_ rows: DecisionReadoutRows, to blob: FileHandle) throws -> Int {
        var count = 0
        var data = Data()
        for block in rows.blocks + [rows.final] {
            for row in block {
                let half = row.map { Float16($0) }
                half.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
                count += 1
            }
        }
        try blob.write(contentsOf: data)
        return count
    }

    /// Re-derives the first question's rows from one prefill of state plus
    /// branch (the long-chunk path) and returns the worst cosine between
    /// those rows and the branched ones (the short-chunk path).
    func selfCheck(
        _ m: Loaded, state: [Int32], branch b: (ids: [Int32], optionEnds: [Int], decide: Int),
        against branched: DecisionReadoutRows
    ) async throws -> Double {
        await m.runner.resetSequence(slot: 0)
        let full = try await readout(
            m, (state + b.ids)[...], from: 0,
            offsets: ([b.decide] + b.optionEnds).map { state.count + $0 },
            blocks: o.layers, keepHead: false)
        // Per block: the worst cosine against the same block of the full
        // prefill, and against the block one step shallower (a capture that
        // ran before its layer finished would match that one instead).
        let a = branched.blocks + [branched.final]
        let c = full.blocks + [full.final]
        var worst = 1.0
        var report: [String] = []
        for i in a.indices {
            let same = zip(a[i], c[i]).map { Self.cosine($0, $1) }.min() ?? 1
            worst = min(worst, same)
            let shallower =
                i > 0 && i < o.layers.count
                ? zip(a[i], c[i - 1]).map { Self.cosine($0, $1) }.min() ?? 0 : .nan
            let name = i < o.layers.count ? "L\(o.layers[i])" : "final"
            report.append(
                "\(name) \(String(format: "%.4f", same))/\(String(format: "%.4f", shallower))")
        }
        say("#   per block (same / one shallower): " + report.joined(separator: "  "))
        return worst
    }

    static func cosine(_ x: [Float], _ y: [Float]) -> Double {
        var dot = 0.0
        var nx = 0.0
        var ny = 0.0
        for (a, b) in zip(x, y) {
            dot += Double(a) * Double(b)
            nx += Double(a) * Double(a)
            ny += Double(b) * Double(b)
        }
        return nx > 0 && ny > 0 ? dot / (nx.squareRoot() * ny.squareRoot()) : 0
    }
}
