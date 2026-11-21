import TinyTitan

/// One terminal line built from styled spans, tracking its visible width so
/// columns can be aligned without counting escape sequences.
struct LiveTraceLine: Sendable {
    let depth: LiveTraceColorDepth
    private(set) var text = ""
    private(set) var width = 0

    init(depth: LiveTraceColorDepth) { self.depth = depth }

    mutating func add(_ string: String, _ style: LiveTraceStyle = .plain) {
        guard !string.isEmpty else { return }
        let sequence = LiveTraceTheme.foreground(style, depth: depth)
        if sequence.isEmpty {
            text += string
        } else {
            text += sequence + string + LiveTraceTheme.reset
        }
        width += LiveTraceText.width(of: string)
    }

    /// A grid cell: `glyph` in a foreground role over a heat background.
    mutating func addCell(
        _ glyph: String, foreground: LiveTraceStyle, heat: Double
    ) {
        let background = LiveTraceTheme.heatBackground(heat, depth: depth)
        let sequence = LiveTraceTheme.foreground(foreground, depth: depth)
        if background.isEmpty && sequence.isEmpty {
            text += glyph
        } else {
            text += background + sequence + glyph + LiveTraceTheme.reset
        }
        width += LiveTraceText.width(of: glyph)
    }

    mutating func add(_ other: LiveTraceLine) {
        text += other.text
        width += other.width
    }

    mutating func pad(to target: Int) {
        if width < target { add(String(repeating: " ", count: target - width)) }
    }
}

/// What the frame needs from the run besides the routing model.
struct LiveTraceFeedSnapshot: Sendable, Equatable {
    enum Phase: Int, Sendable { case starting, prefill, decode, done }
    var phase: Phase = .starting
    var prefillDone = 0
    var prefillTotal = 0
    var tokens = 0
    var maxNew = 0
    /// Tail of the generated text; the view shows its last two wrapped lines.
    var textTail = ""
    /// Tokens per second over the last few tokens, from arrival times.
    var tokensPerSecond: Double?
}

/// What a server adds to a frame: which model is resident, which client the
/// request on screen came from, how many are waiting, and what the last one did.
/// nil in a CLI run, whose frame is exactly one request in one model.
struct LiveTraceServerStatus: Sendable, Equatable {
    /// What the last finished request did, for the idle frame.
    struct Last: Sendable, Equatable {
        var tokensPerSecond: Double?
        var newTokens = 0
        var promptTokens = 0
        var cachedTokens = 0

        /// Share of the prompt that came from the prompt cache, 0...1.
        var cacheHit: Double? {
            promptTokens > 0 ? min(1, Double(cachedTokens) / Double(promptTokens)) : nil
        }
    }

    enum Residency: Sendable, Equatable { case none, loading, loaded }

    var residency = Residency.none
    /// The header's model text: the resident model and its width, `loading X`
    /// while a switch is in progress, or that nothing is loaded.
    var title = "no model loaded"
    /// The model id the client asked for, for the request on screen.
    var client: String?
    /// Requests admitted and not yet generating: the queue as a client sees it.
    var waiting = 0
    /// Generations in flight. More than one only with batched serving.
    var running = 0
    var last: Last?

    var isIdle: Bool { running == 0 }
}

struct LiveTraceFrameInput {
    let shape: ExpertTraceShape
    let model: LiveTraceModel?
    let feed: LiveTraceFeedSnapshot
    /// Recent stderr lines, oldest first.
    let log: [String]
    let plan: LiveTraceViewPlan
    let spinner: Int
    let depth: LiveTraceColorDepth
    /// Set by a server's view; nil keeps the CLI's frame.
    var server: LiveTraceServerStatus?
    /// Rows the log may grow by, beyond `LiveTraceViewPlan.logLines`, so the
    /// newest line shows whole (an error's cause is at its start, its hashes at
    /// its end). The log is the bottom item, so growing moves nothing else. 0
    /// keeps the frame exactly `plan.height`.
    var extraLogRows = 0
}

/// Composes one frame of the live trace view as a fixed number of ANSI lines.
///
/// Pure: the same input always gives the same frame, which is what lets the
/// frame be pinned by a snapshot test. Time, the terminal and the engine are
/// all outside it.
///
/// Every grid cell has two layers. The background carries heat, from the
/// model's `ExpertHeatSource`; the foreground glyph carries the current
/// token's picks, a centred block for a cache hit and a solid red block for an
/// SSD read. Without colour the background cannot be drawn, so a cell that was
/// not picked shows its heat as a block-height glyph instead.
enum LiveTraceFrame {
    static let blocks = ["·", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    static let spinnerGlyphs = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    static let gridColumnWidth = ExpertGridLayout.maxGridChars + 2
    static let gap = 4

    /// The frame, `plan.height` lines plus however many of `extraLogRows` the
    /// newest log line needs, each at most `plan.width` cells.
    static func compose(_ input: LiveTraceFrameInput) -> [String] {
        guard let width = input.plan.width, let planHeight = input.plan.height else { return [] }
        let log = logLines(input, width: width)
        let height = planHeight + max(0, log.count - LiveTraceViewPlan.logLines)
        var lines: [LiveTraceLine] = [header(input, width: width)]
        if case .full = input.plan, let model = input.model {
            lines.append(blank(input))
            lines.append(contentsOf: gridAndPanel(input, model: model, width: width))
            lines.append(blank(input))
            lines.append(ribbon(input, model: model, width: width))
        }
        lines.append(
            rule(
                input.server?.isIdle == true ? "output · last reply" : "output", width: width,
                depth: input.depth))
        lines.append(contentsOf: outputLines(input, width: width))
        lines.append(rule("log", width: width, depth: input.depth))
        lines.append(contentsOf: log)
        // The plan's height is derived from the same pieces, and a test pins that
        // they agree; if they ever did not, a frame that is the wrong height
        // would tear the in-place redraw, so make it the right one.
        while lines.count < height { lines.append(blank(input)) }
        return lines.prefix(height).map(\.text)
    }

    private static func blank(_ input: LiveTraceFrameInput) -> LiveTraceLine {
        LiveTraceLine(depth: input.depth)
    }

    private static func rule(_ title: String, width: Int, depth: LiveTraceColorDepth)
        -> LiveTraceLine
    {
        var line = LiveTraceLine(depth: depth)
        let label = "─ \(title) "
        line.add(label + String(repeating: "─", count: max(0, width - label.count)), .dim)
        return line
    }

    // MARK: header

    static func header(_ input: LiveTraceFrameInput, width: Int) -> LiveTraceLine {
        if let server = input.server { return serverHeader(input, server: server, width: width) }
        let feed = input.feed
        var right = LiveTraceLine(depth: input.depth)
        switch feed.phase {
        case .starting, .prefill:
            right.add("prefill ", .plain)
            if feed.prefillTotal > 0 {
                right.add("\(feed.prefillDone)/\(feed.prefillTotal) tok", .hit)
            } else {
                right.add("starting", .hit)
            }
        case .decode, .done:
            let glyph =
                feed.phase == .done ? "✓" : spinnerGlyphs[input.spinner % spinnerGlyphs.count]
            right.add(glyph + " ", .hit)
            right.add(
                feed.maxNew > 0 ? "\(feed.tokens)/\(feed.maxNew) tok  " : "\(feed.tokens) tok  ")
            let rate =
                input.model.flatMap { LiveTraceModel.recent($0.tokensPerSecond) }
                ?? feed.tokensPerSecond
            right.add(rate.map { format1($0) + " tok/s" } ?? "-- tok/s", .hit)
        }
        var left = LiveTraceLine(depth: input.depth)
        left.add("emone", .accent)
        var detail = "  " + input.shape.modelName
        if let bits = input.shape.routedExpertBits { detail += " · \(bits)-bit" }
        let room = max(0, width - left.width - right.width - 1)
        left.add(LiveTraceText.truncate(detail, width: room), .dim)
        left.pad(to: width - right.width)
        left.add(right)
        return left
    }

    /// The server's header: the resident model, the client's model id for the
    /// request on screen, and on the right the queue and what the engine is
    /// doing. The right side is kept whole; the left gives way to it, the
    /// client's id first.
    static func serverHeader(
        _ input: LiveTraceFrameInput, server: LiveTraceServerStatus, width: Int
    ) -> LiveTraceLine {
        let feed = input.feed
        var right = LiveTraceLine(depth: input.depth)
        if server.waiting > 0 { right.add("queue \(server.waiting)  ", .accent) }
        if server.running > 1 { right.add("\(server.running) running  ", .accent) }
        if server.isIdle {
            right.add("idle", .dim)
            if let rate = server.last?.tokensPerSecond {
                right.add("  last " + format1(rate) + " tok/s", .hit)
            }
            if let hit = server.last?.cacheHit {
                right.add("  cache \(Int((hit * 100).rounded()))%", .hit)
            }
        } else {
            switch feed.phase {
            case .starting, .prefill:
                right.add("prefill ", .plain)
                if feed.prefillTotal > 0 {
                    right.add("\(feed.prefillDone)/\(feed.prefillTotal) tok", .hit)
                } else {
                    right.add("starting", .hit)
                }
            case .decode, .done:
                right.add(spinnerGlyphs[input.spinner % spinnerGlyphs.count] + " ", .hit)
                right.add("\(feed.tokens) tok  ")
                let rate =
                    input.model.flatMap { LiveTraceModel.recent($0.tokensPerSecond) }
                    ?? feed.tokensPerSecond
                right.add(rate.map { format1($0) + " tok/s" } ?? "-- tok/s", .hit)
            }
        }
        var left = LiveTraceLine(depth: input.depth)
        left.add("emone", .accent)
        let room = max(0, width - left.width - right.width - 1)
        let title = "  " + server.title
        let titleWidth = LiveTraceText.width(of: title)
        var clientText = ""
        if !server.isIdle, let client = server.client, !client.isEmpty {
            clientText = " ← " + client
        }
        let clientWidth = LiveTraceText.width(of: clientText)
        if titleWidth + clientWidth <= room {
            left.add(title, .dim)
            left.add(clientText, .plain)
        } else if titleWidth <= room {
            // The model stays whole; the client gets what is left, or nothing
            // when that is too little to be legible.
            left.add(title, .dim)
            let clientRoom = room - titleWidth
            if clientWidth > 0, clientRoom >= 8 {
                left.add(truncated(clientText, width: clientRoom), .plain)
            }
        } else {
            left.add(truncated(title, width: room), .dim)
        }
        left.pad(to: max(left.width, width - right.width))
        left.add(right)
        return left
    }

    /// `text` cut to `width` cells with an ellipsis when it did not fit.
    private static func truncated(_ text: String, width: Int) -> String {
        guard LiveTraceText.width(of: text) > width, width > 1 else {
            return LiveTraceText.truncate(text, width: width)
        }
        return LiveTraceText.truncate(text, width: width - 1) + "…"
    }

    // MARK: grid and router panel

    static func gridAndPanel(
        _ input: LiveTraceFrameInput, model: LiveTraceModel, width: Int
    ) -> [LiveTraceLine] {
        let grid = gridLines(input, model: model)
        let panel = panelLines(input, model: model)
        let count = LiveTraceViewPlan.gridBlockHeight(grid: model.grid, topK: model.shape.topK)
        var out: [LiveTraceLine] = []
        for row in 0..<count {
            var line = LiveTraceLine(depth: input.depth)
            if row < grid.count { line.add(grid[row]) }
            line.pad(to: gridColumnWidth + gap)
            if row < panel.count { line.add(panel[row]) }
            out.append(line)
        }
        return out
    }

    static func gridLines(_ input: LiveTraceFrameInput, model: LiveTraceModel) -> [LiveTraceLine] {
        let layout = model.grid
        let depth = input.depth
        let cells = model.cells(layer: model.layer)
        let bodyWidth = layout.bodyWidth
        var title =
            " L\(pad2(model.layer + 1)) \(kindName(model.shape, layer: model.layer, long: false))"
        title += " · experts 0..\(model.shape.numExperts - 1) "
        title = LiveTraceText.truncate(title, width: bodyWidth)
        var top = LiveTraceLine(depth: depth)
        top.add("┌", .dim)
        top.add(title)
        top.add(String(repeating: "─", count: bodyWidth - title.count) + "┐", .dim)
        var out = [top]
        for row in 0..<layout.rows {
            var line = LiveTraceLine(depth: depth)
            line.add("│", .dim)
            for col in 0..<layout.cols {
                addCell(&line, cells[row * layout.cols + col], width: layout.cellWidth)
            }
            line.pad(to: bodyWidth + 1)
            line.add("│", .dim)
            out.append(line)
        }
        var bottom = LiveTraceLine(depth: depth)
        bottom.add("└" + String(repeating: "─", count: bodyWidth) + "┘", .dim)
        out.append(bottom)
        return out
    }

    /// The glyph for a pick, by cell width: a hit is a centred block that
    /// leaves the heat visible on either side, a miss fills the cell.
    static func pickGlyph(_ pick: ExpertCell.Pick, width: Int) -> String {
        switch pick {
        case .hit: return width >= 2 ? "▐▌" : "●"
        case .miss: return String(repeating: "█", count: width)
        case .none: return ""
        }
    }

    private static func addCell(_ line: inout LiveTraceLine, _ cell: ExpertCell, width: Int) {
        if cell.padding {
            line.add(String(repeating: " ", count: width))
            return
        }
        switch cell.pick {
        case .hit:
            line.addCell(pickGlyph(.hit, width: width), foreground: .hit, heat: cell.heat)
        case .miss:
            line.addCell(pickGlyph(.miss, width: width), foreground: .miss, heat: 0)
        case .none:
            if line.depth == .none {
                // No background to carry heat: a block whose height is the heat.
                if cell.cached && cell.heat >= LiveTraceTheme.heatFloor {
                    let glyph = blocks[max(1, min(8, Int((cell.heat * 8).rounded())))]
                    line.add(String(repeating: glyph, count: width))
                } else {
                    line.add("·" + String(repeating: " ", count: width - 1))
                }
            } else {
                let dot = cell.cached ? " " : "·"
                line.addCell(
                    dot + String(repeating: " ", count: width - 1), foreground: .veryDim,
                    heat: cell.heat)
            }
        }
    }

    static func panelLines(_ input: LiveTraceFrameInput, model: LiveTraceModel) -> [LiveTraceLine] {
        let depth = input.depth
        let panelWidth = (input.plan.width ?? 78) - gridColumnWidth - gap
        var out: [LiveTraceLine] = []
        var layerLine = labelled("layer", depth: depth)
        if model.hasEvents {
            layerLine.add("L\(model.layer + 1) / \(model.shape.layers)  ")
            layerLine.add(kindName(model.shape, layer: model.layer, long: true) + " + MoE", .dim)
        } else {
            layerLine.add(input.server?.isIdle == true ? "idle" : "waiting for decode", .dim)
        }
        out.append(layerLine)
        out.append(blank(input))
        var picksTitle = LiveTraceLine(depth: depth)
        picksTitle.add("router picks", .dim)
        out.append(picksTitle)
        out.append(contentsOf: pickRows(input, model: model, panelWidth: panelWidth))
        out.append(blank(input))
        // Between requests the bar and the figure keep the last request's.
        let idle = input.server?.isIdle == true
        let recentHit = model.hitRate ?? (idle ? model.lastHitRate : nil)
        var cache = labelled("cache", depth: depth)
        cache.add(bar(model.layerHitFraction ?? recentHit ?? 0, width: 14, depth: depth))
        cache.add(" " + (recentHit.map { "\(Int(($0 * 100).rounded()))% hit" } ?? "-- hit"))
        out.append(cache)
        var tps = labelled("tok/s", depth: depth)
        tps.add(sparkline(model.tokensPerSecond, width: 20, depth: depth))
        let rate = LiveTraceModel.recent(model.tokensPerSecond) ?? input.feed.tokensPerSecond
        tps.add(" " + (rate.map(format1) ?? "--"), .hit)
        out.append(tps)
        var ssd = labelled("ssd", depth: depth)
        ssd.add(sparkline(model.ssdGBPerSecond, width: 20, depth: depth))
        ssd.add(
            " " + (LiveTraceModel.recent(model.ssdGBPerSecond).map(format1) ?? "--") + " GB/s", .hit
        )
        out.append(ssd)
        out.append(blank(input))
        out.append(contentsOf: legend(depth: depth))
        return out
    }

    private static func pickRows(
        _ input: LiveTraceFrameInput, model: LiveTraceModel, panelWidth: Int
    ) -> [LiveTraceLine] {
        let rows = LiveTraceViewPlan.pickRows(topK: model.shape.topK)
        let idWidth = max(3, String(model.shape.numExperts - 1).count)
        let column = min(idWidth + 15, panelWidth / 2)
        let capacity = rows * 2
        let overflow = max(0, model.picks.count - capacity)
        var out: [LiveTraceLine] = []
        for row in 0..<rows {
            var line = LiveTraceLine(depth: input.depth)
            for side in 0..<2 {
                let index = row * 2 + side
                var item = LiveTraceLine(depth: input.depth)
                if overflow > 0, index == capacity - 1 {
                    item.add("+\(overflow + 1) more", .dim)
                } else if index < model.picks.count {
                    let pick = model.picks[index]
                    let id = String(pick.expert)
                    item.add("e" + String(repeating: " ", count: idWidth - id.count) + id + " ")
                    item.add(pick.miss ? "MISS" : "hit ", pick.miss ? .miss : .hit)
                }
                item.pad(to: column)
                line.add(item)
            }
            out.append(line)
        }
        return out
    }

    /// The cell layers, explained: the heat swatches, then the pick glyphs.
    private static func legend(depth: LiveTraceColorDepth) -> [LiveTraceLine] {
        var heat = LiveTraceLine(depth: depth)
        heat.add("recent heat ", .dim)
        for step in [0.1, 0.3, 0.5, 0.75, 1.0] {
            if depth == .none {
                heat.add(
                    blocks[max(1, Int((step * 8).rounded()))]
                        + blocks[max(1, Int((step * 8).rounded()))])
            } else {
                heat.addCell("  ", foreground: .plain, heat: step)
            }
        }
        heat.add(" cold→hot", .dim)
        var picks = LiveTraceLine(depth: depth)
        picks.addCell("▐▌", foreground: .hit, heat: 0.3)
        picks.add(" hit  ")
        picks.addCell("██", foreground: .miss, heat: 0)
        picks.add(" miss (SSD)  ")
        picks.add("·", .veryDim)
        picks.add(" not cached")
        return [heat, picks]
    }

    // MARK: ribbon

    static func ribbon(_ input: LiveTraceFrameInput, model: LiveTraceModel, width: Int)
        -> LiveTraceLine
    {
        let layers = model.shape.layers
        var line = LiveTraceLine(depth: input.depth)
        line.add("L" + pad2(min(model.layer + 1, layers)) + " ")
        let cells = max(1, min(layers, width - line.width - 1))
        let group = (layers + cells - 1) / cells
        for cell in 0..<((layers + group - 1) / group) {
            let range = (cell * group)..<min(layers, cell * group + group)
            if range.contains(model.layer) && model.hasEvents {
                line.add(" ", .cursor)
                continue
            }
            let misses = range.map { Int(model.layerMisses[$0]) }
            if misses.allSatisfy({ $0 < 0 }) || (range.lowerBound > model.layer && model.hasEvents)
            {
                line.add(blocks[0], .veryDim)
            } else {
                addMissCell(&line, misses: misses.max() ?? 0, topK: model.shape.topK)
            }
        }
        return line
    }

    private static func addMissCell(_ line: inout LiveTraceLine, misses: Int, topK: Int) {
        if misses <= 0 {
            line.add("▪", .dim)
        } else {
            let level = max(1, min(8, (misses * 8 + topK - 1) / max(1, topK)))
            line.add(blocks[level], .missLevel(level))
        }
    }

    // MARK: output and log

    static func outputLines(_ input: LiveTraceFrameInput, width: Int) -> [LiveTraceLine] {
        let feed = input.feed
        let depth = input.depth
        var out: [LiveTraceLine] = []
        if let server = input.server, server.isIdle {
            return idleOutputLines(input, server: server, width: width)
        }
        if feed.phase == .starting || feed.phase == .prefill {
            var line = LiveTraceLine(depth: depth)
            line.add("prefilling ")
            let fraction =
                feed.prefillTotal > 0 ? Double(feed.prefillDone) / Double(feed.prefillTotal) : 0
            line.add(bar(fraction, width: 30, depth: depth))
            if feed.prefillTotal > 0 { line.add("  \(feed.prefillTotal) prompt tokens", .dim) }
            out = [line, blank(input)]
        } else {
            let wrapped = LiveTraceText.wrap(feed.textTail, width: width - 1)
            var tail = Array(wrapped.suffix(LiveTraceViewPlan.outputLines))
            while tail.count < LiveTraceViewPlan.outputLines { tail.append("") }
            for (index, text) in tail.enumerated() {
                var line = LiveTraceLine(depth: depth)
                line.add(text)
                if index == tail.count - 1 && feed.phase == .decode { line.add(" ", .cursor) }
                out.append(line)
            }
        }
        return out
    }

    /// The output area between requests: the tail of the last reply, dimmed, or
    /// a line saying what the server is waiting for.
    private static func idleOutputLines(
        _ input: LiveTraceFrameInput, server: LiveTraceServerStatus, width: Int
    ) -> [LiveTraceLine] {
        let depth = input.depth
        let wrapped = LiveTraceText.wrap(input.feed.textTail, width: width - 1)
        var tail = Array(wrapped.suffix(LiveTraceViewPlan.outputLines))
        if input.feed.textTail.isEmpty {
            switch server.residency {
            case .loading: tail = ["loading the model for the next request"]
            case .none: tail = ["no model resident: the next request loads one"]
            case .loaded:
                tail = [server.waiting > 0 ? "request waiting for the model" : "waiting for a request"]
            }
        }
        while tail.count < LiveTraceViewPlan.outputLines { tail.append("") }
        return tail.map { text in
            var line = LiveTraceLine(depth: depth)
            line.add(text, .dim)
            return line
        }
    }

    /// Alert colouring for the log: a line mentioning error, warn or fail.
    static func isAlert(_ line: String) -> Bool {
        let lower = line.lowercased()
        return lower.contains("error") || lower.contains("warn") || lower.contains("fail")
    }

    /// The tail of the log, `LiveTraceViewPlan.logLines` rows or, when the
    /// newest line wraps to more, as many as it takes (up to `extraLogRows`
    /// more). A newest line longer even than that shows its start.
    static func logLines(_ input: LiveTraceFrameInput, width: Int) -> [LiveTraceLine] {
        var rows: [(String, Bool)] = []
        var newestRows = 0
        for raw in input.log {
            let line = LiveTraceText.sanitize(raw)
            let alert = isAlert(line)
            let wrapped = LiveTraceText.hardWrap(line, width: width)
            newestRows = wrapped.count
            for row in wrapped { rows.append((row, alert)) }
        }
        let shown = max(
            LiveTraceViewPlan.logLines,
            min(newestRows, LiveTraceViewPlan.logLines + max(0, input.extraLogRows)))
        let visible =
            newestRows > shown
            ? Array(rows.suffix(newestRows).prefix(shown)) : Array(rows.suffix(shown))
        var out: [LiveTraceLine] = []
        if rows.isEmpty {
            var line = LiveTraceLine(depth: input.depth)
            line.add("(nothing logged)", .veryDim)
            out.append(line)
        }
        for (text, alert) in visible {
            var line = LiveTraceLine(depth: input.depth)
            line.add(text, alert ? .miss : .plain)
            out.append(line)
        }
        while out.count < LiveTraceViewPlan.logLines { out.append(blank(input)) }
        return out
    }

    // MARK: pieces

    private static func labelled(_ label: String, depth: LiveTraceColorDepth) -> LiveTraceLine {
        var line = LiveTraceLine(depth: depth)
        line.add(label + String(repeating: " ", count: max(0, 9 - label.count)), .dim)
        return line
    }

    static func bar(_ fraction: Double, width: Int, depth: LiveTraceColorDepth) -> LiveTraceLine {
        let filled = Int((max(0, min(1, fraction)) * Double(width)).rounded())
        var line = LiveTraceLine(depth: depth)
        line.add(String(repeating: "█", count: filled), .hit)
        line.add(String(repeating: "░", count: width - filled), .dim)
        return line
    }

    static func sparkline(_ values: [Double], width: Int, depth: LiveTraceColorDepth)
        -> LiveTraceLine
    {
        var line = LiveTraceLine(depth: depth)
        let tail = Array(values.suffix(width))
        line.add(String(repeating: " ", count: width - tail.count))
        let peak = max(tail.max() ?? 0, 1e-9)
        for value in tail {
            let level = max(0, min(8, Int((value / peak * 8).rounded())))
            if level == 0 {
                line.add(blocks[0], .dim)
            } else {
                line.add(blocks[level], .heat(level))
            }
        }
        return line
    }

    static func kindName(_ shape: ExpertTraceShape, layer: Int, long: Bool) -> String {
        let kind = layer < shape.layerKinds.count ? shape.layerKinds[layer] : 0
        switch kind {
        case 2: return "GDN"
        case 1: return long ? "full attention" : "attn"
        default: return long ? "sliding window" : "swa"
        }
    }

    static func pad2(_ value: Int) -> String { value < 10 ? "0\(value)" : "\(value)" }

    static func format1(_ value: Double) -> String {
        let tenths = Int((value * 10).rounded())
        return "\(tenths / 10).\(abs(tenths % 10))"
    }
}
