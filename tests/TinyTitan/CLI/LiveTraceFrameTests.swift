import Foundation
import Testing
import TinyTitan

@testable import TinyTitanCLICore

@Suite struct LiveTraceFrameTests {
    static let shape = ExpertTraceShape(
        modelName: "qwen3.8-flash-next", routedExpertBits: 4, layers: 48, numExperts: 512,
        topK: 10, layerKinds: (0..<48).map { $0 % 4 == 3 ? 1 : 2 }, expertBytes: 7_000_000)

    /// A fixed synthetic run: 12 decoded tokens routed through 48 layers, then
    /// the first 9 layers of token 12. Picks are a deterministic function of
    /// (token, layer), and a layer misses when (token + layer) % 5 == 0.
    static func synthetic(tokens: Int = 12, layersOfLast: Int = 9) throws -> LiveTraceModel {
        var model = try #require(LiveTraceModel(shape: shape, slotsPerLayer: 96))
        var trace = SyntheticTrace()
        for token in 0...tokens {
            let layers = token == tokens ? layersOfLast : 48
            for layer in 0..<layers {
                let base = (layer * 37 + (token % 3) * 11) % 480
                let ids = (0..<10).map { base + $0 * 3 }
                let misses = (token + layer) % 5 == 0 ? [1, 4, 7] : [Int]()
                let nanos = UInt64(token) * 80_000_000 + UInt64(layer) * 1_000_000
                trace.add(position: token, layer: layer, ids: ids, misses: misses, atNanos: nanos)
            }
        }
        model.apply(trace.batch)
        return model
    }

    static func feed(phase: LiveTraceFeedSnapshot.Phase = .decode) -> LiveTraceFeedSnapshot {
        var feed = LiveTraceFeedSnapshot()
        feed.phase = phase
        feed.tokens = 12
        feed.maxNew = 64
        feed.textTail =
            "Here is a Swift function that reverses a singly linked list in place.\n"
            + "It walks the list once, flipping each next pointer as it goes, and returns the new head."
        feed.tokensPerSecond = 12.5
        return feed
    }

    static let log = [
        "[prefill 1024 tok]", "warning: expert cache is 96 slots", "[decode expert io] hits 400",
    ]

    /// The frame as plain text, trailing spaces trimmed so a snapshot survives
    /// an editor that strips them.
    static func plain(_ lines: [String]) -> String {
        lines.map { line in
            var text = stripANSI(line)
            while text.hasSuffix(" ") { text.removeLast() }
            return text
        }.joined(separator: "\n")
    }

    static func stripANSI(_ line: String) -> String {
        var out = ""
        var escape = false
        for scalar in line.unicodeScalars {
            if escape {
                if (0x40...0x7E).contains(scalar.value) && scalar != "[" { escape = false }
            } else if scalar == "\u{1B}" {
                escape = true
            } else {
                out.unicodeScalars.append(scalar)
            }
        }
        return out
    }

    static func input(
        model: LiveTraceModel?, plan: LiveTraceViewPlan, feed: LiveTraceFeedSnapshot = feed(),
        depth: LiveTraceColorDepth = .truecolor, shape: ExpertTraceShape = shape,
        log: [String] = log
    ) -> LiveTraceFrameInput {
        LiveTraceFrameInput(
            shape: shape, model: model, feed: feed, log: log, plan: plan, spinner: 3, depth: depth)
    }

    static let esc = "\u{1B}["

    @Test func fullFrameIsAFixedSizeAndMatchesTheSnapshot() throws {
        let model = try Self.synthetic()
        let lines = LiveTraceFrame.compose(
            Self.input(model: model, plan: .full(width: 78, height: 30)))
        #expect(lines.count == 30)
        for line in lines {
            #expect(LiveTraceText.width(of: Self.stripANSI(line)) <= 78)
        }
        #expect(Self.plain(lines) == Self.fullSnapshot)
    }

    @Test(arguments: [LiveTraceColorDepth.none, .ansi16, .ansi256, .truecolor])
    func everyColourDepthGivesTheSameTextAtTheSameSize(depth: LiveTraceColorDepth) throws {
        let model = try Self.synthetic()
        let lines = LiveTraceFrame.compose(
            Self.input(model: model, plan: .full(width: 78, height: 30), depth: depth))
        #expect(lines.count == 30)
        #expect(lines.allSatisfy { LiveTraceText.width(of: Self.stripANSI($0)) <= 78 })
        // Without colour the heat becomes block heights, so the grid differs;
        // every colour depth that can draw a background gives the same text.
        #expect(Self.plain(lines) == (depth == .none ? Self.noColourSnapshot : Self.fullSnapshot))
    }

    @Test func gridCellsCarryHeatInTheBackgroundAndPicksInTheForeground() throws {
        var model = try #require(LiveTraceModel(shape: Self.shape, slotsPerLayer: 96))
        var trace = SyntheticTrace()
        trace.add(position: 0, layer: 0, ids: [0, 1, 40], misses: [1], atNanos: 0)
        trace.add(position: 1, layer: 0, ids: [0, 1, 40], misses: [1], atNanos: 1)
        model.apply(trace.batch)
        let plan = LiveTraceViewPlan.full(width: 78, height: 30)
        // Grid row 0 is line 3: header, blank, border, then the rows.
        let row = LiveTraceFrame.compose(Self.input(model: model, plan: plan, depth: .truecolor))[3]
        let cells = model.cells(layer: 0)
        let rgb = LiveTraceTheme.heatRGB(cells[0].heat)
        let heatBG = "\(Self.esc)48;2;\(rgb.r);\(rgb.g);\(rgb.b)m"
        // Expert 0: a hit over its heat. Expert 1: a miss, solid red, no heat shown.
        #expect(row.contains(heatBG + "\(Self.esc)38;5;84m●\(Self.esc)0m"))
        #expect(row.contains("\(Self.esc)38;5;203m█\(Self.esc)0m"))
        // An untouched cell has no background sequence at all.
        #expect(row.contains("\(Self.esc)38;5;238m·\(Self.esc)0m"))
        #expect(!row.contains("\(Self.esc)48;5;"))
    }

    @Test func twoCharacterCellsUseACentredBlockForAHit() throws {
        let shape = ExpertGridLayoutTests.shape(experts: 256, topK: 8)
        var model = try #require(LiveTraceModel(shape: shape, slotsPerLayer: 96))
        var trace = SyntheticTrace()
        trace.add(position: 0, layer: 0, ids: [0, 1], misses: [1], atNanos: 0)
        model.apply(trace.batch)
        let row = LiveTraceFrame.compose(
            Self.input(
                model: model, plan: .full(width: 78, height: 30), depth: .ansi256, shape: shape))[3]
        #expect(row.contains("▐▌"))
        #expect(row.contains("██"))
        #expect(Self.stripANSI(row).hasPrefix("│▐▌██"))
    }

    @Test func sixteenColourOutputUsesNoExtendedSequences() throws {
        let model = try Self.synthetic()
        let text = LiveTraceFrame.compose(
            Self.input(model: model, plan: .full(width: 78, height: 30), depth: .ansi16)
        ).joined()
        #expect(!text.contains("38;5;") && !text.contains("48;5;") && !text.contains(";2;"))
        #expect(text.contains("\(Self.esc)100m") || text.contains("\(Self.esc)41m"))
    }

    @Test func noColourDropsEveryColourButKeepsTheCursor() throws {
        let model = try Self.synthetic()
        let text = LiveTraceFrame.compose(
            Self.input(model: model, plan: .full(width: 78, height: 30), depth: .none)
        ).joined()
        // Only reverse video (the output and ribbon cursors) and its reset.
        let sequences = text.components(separatedBy: Self.esc).dropFirst().map { part in
            String(part.prefix(while: { $0 != "m" })) + "m"
        }
        #expect(Set(sequences) == ["7m", "0m"])
        #expect(text.contains("▐▌") && text.contains("██"))
    }

    @Test func errorWarnAndFailLinesPrintInTheAlertColour() throws {
        let model = try Self.synthetic()
        let lines = LiveTraceFrame.compose(
            Self.input(
                model: model, plan: .full(width: 78, height: 30),
                log: ["all fine", "warning: low memory", "load FAILED", "an Error occurred"]))
        let log = Array(lines.suffix(4))
        let alert = "\(Self.esc)38;5;203m"
        #expect(!log[0].contains(alert) && !log[0].contains("\u{1B}"))
        #expect(log[1].hasPrefix(alert + "warning"))
        #expect(log[2].hasPrefix(alert + "load FAILED"))
        #expect(log[3].hasPrefix(alert + "an Error"))
    }

    @Test func aDenseModelGetsAHeaderOutputAndLogOnly() {
        let dense = ExpertGridLayoutTests.shape(experts: 0, topK: 0, layers: 32)
        let lines = LiveTraceFrame.compose(
            Self.input(
                model: nil, plan: .compact(width: 78, height: 9), depth: .ansi256, shape: dense))
        #expect(lines.count == 9)
        #expect(Self.plain(lines) == Self.denseSnapshot)
    }

    @Test func prefillShowsProgressInsteadOfOutput() throws {
        var feed = LiveTraceFeedSnapshot()
        feed.phase = .prefill
        feed.prefillDone = 512
        feed.prefillTotal = 2048
        let model = try #require(LiveTraceModel(shape: Self.shape, slotsPerLayer: 96))
        let lines = LiveTraceFrame.compose(
            Self.input(model: model, plan: .full(width: 78, height: 30), feed: feed, log: []))
        let text = Self.plain(lines)
        #expect(text.contains("prefill 512/2048 tok"))
        #expect(text.contains("prefilling ████████░░░░░░░░░░░░░░░░░░░░░░  2048 prompt tokens"))
        #expect(text.contains("waiting for decode"))
        #expect(text.contains("(nothing logged)"))
    }

    @Test(arguments: [(256, 8), (512, 10), (128, 6), (1024, 4), (64, 6), (60, 4), (300, 12)])
    func everyModelShapeComposesToItsPlannedSize(experts: Int, topK: Int) throws {
        let shape = ExpertGridLayoutTests.shape(experts: experts, topK: topK, layers: 40)
        let height = try #require(LiveTraceViewPlan.fullHeight(shape: shape))
        let plan = LiveTraceViewPlan.plan(shape: shape, cols: 80, rows: height + 1)
        #expect(plan == .full(width: 78, height: height))
        var model = try #require(LiveTraceModel(shape: shape, slotsPerLayer: 32))
        var trace = SyntheticTrace()
        for token in 0..<3 {
            for layer in 0..<40 {
                let ids = (0..<topK).map { ($0 * 7 + layer + token) % experts }
                trace.add(
                    position: token, layer: layer, ids: ids, misses: [0, topK - 1],
                    atNanos: UInt64(token * 40 + layer) * 1_000_000)
            }
        }
        model.apply(trace.batch)
        for depth in [LiveTraceColorDepth.none, .ansi16, .ansi256, .truecolor] {
            let lines = LiveTraceFrame.compose(
                Self.input(model: model, plan: plan, depth: depth, shape: shape))
            #expect(lines.count == height)
            #expect(lines.allSatisfy { LiveTraceText.width(of: Self.stripANSI($0)) <= 78 })
            // The grid is a border, rows of cells and a border; whatever the model,
            // the router panel starts at column 38.
            let grid = try #require(ExpertGridLayout.make(numExperts: experts))
            let top = Array(Self.stripANSI(lines[2]))
            #expect(top[grid.bodyWidth + 1] == "┐")
            #expect(String(top[38...]).hasPrefix("layer"))
            let bottom = Array(Self.stripANSI(lines[2 + grid.rows + 1]))
            #expect(bottom[grid.bodyWidth + 1] == "┘")
        }
    }

    // MARK: colour

    @Test func colourDepthFollowsTheEnvironment() {
        typealias Depth = LiveTraceColorDepth
        #expect(
            Depth.detect(term: "xterm-256color", colorTerm: "truecolor", noColor: false)
                == .truecolor)
        #expect(Depth.detect(term: "xterm", colorTerm: "24bit", noColor: false) == .truecolor)
        #expect(Depth.detect(term: "xterm-direct", colorTerm: nil, noColor: false) == .truecolor)
        #expect(Depth.detect(term: "xterm-256color", colorTerm: nil, noColor: false) == .ansi256)
        #expect(Depth.detect(term: "screen-256color", colorTerm: "", noColor: false) == .ansi256)
        #expect(Depth.detect(term: "xterm", colorTerm: nil, noColor: false) == .ansi16)
        #expect(Depth.detect(term: "linux", colorTerm: nil, noColor: false) == .ansi16)
        #expect(
            Depth.detect(term: "xterm-256color", colorTerm: "truecolor", noColor: true) == .none)
        #expect(Depth.detect(term: "dumb", colorTerm: nil, noColor: false) == .none)
        #expect(Depth.detect(term: nil, colorTerm: "truecolor", noColor: false) == .none)
    }

    @Test func heatBackgroundsPerDepth() {
        #expect(LiveTraceTheme.heatBackground(0.0, depth: .truecolor) == "")
        #expect(LiveTraceTheme.heatBackground(0.03, depth: .ansi256) == "")
        #expect(LiveTraceTheme.heatBackground(1.0, depth: .truecolor) == "\u{1B}[48;2;196;128;36m")
        #expect(LiveTraceTheme.heatBackground(0.55, depth: .truecolor) == "\u{1B}[48;2;112;52;60m")
        #expect(LiveTraceTheme.heatBackground(1.0, depth: .none) == "")
        #expect(LiveTraceTheme.heatBackground(0.2, depth: .ansi16) == "\u{1B}[100m")
        #expect(LiveTraceTheme.heatBackground(0.5, depth: .ansi16) == "\u{1B}[41m")
        #expect(LiveTraceTheme.heatBackground(0.9, depth: .ansi16) == "\u{1B}[43m")
        // 256: the nearest cube colour to (196,128,36) is (215,135,0) = 16+36*4+6*2+0.
        #expect(LiveTraceTheme.heatBackground(1.0, depth: .ansi256) == "\u{1B}[48;5;172m")
    }

    @Test func theRampRisesMonotonicallyInBrightness() {
        var last = -1
        for step in stride(from: 0.0, through: 1.0, by: 0.05) {
            let c = LiveTraceTheme.heatRGB(step)
            let brightness = c.r + c.g + c.b
            #expect(brightness >= last)
            last = brightness
        }
    }

    static let fullSnapshot = """
        emone  qwen3.8-flash-next · 4-bit                      ⠸ 12/64 tok  12.5 tok/s

        ┌ L09 GDN · experts 0..511 ──────┐    layer    L9 / 48  GDN + MoE
        │································│
        │································│    router picks
        │································│    e296 hit          e299 MISS
        │································│    e302 hit          e305 hit
        │································│    e308 MISS         e311 hit
        │································│    e314 hit          e317 MISS
        │································│    e320 hit          e323 hit
        │································│
        │································│    cache    ██████████░░░░ 94% hit
        │········●··█··●··●· █· ●· ●· █  │    tok/s            ████████████ 12.5
        │●  ●  ·  ·  ·  · ·· ·· ·· ······│    ssd              █▇▇███▇▇███▇ 2.5 GB/s
        │································│
        │································│    recent heat            cold→hot
        │································│    ▐▌ hit  ██ miss (SSD)  · not cached
        │································│
        │································│
        └────────────────────────────────┘

        L09 ▪▪▪▃▪▪▪▪ ·······································
        ─ output ─────────────────────────────────────────────────────────────────────
        It walks the list once, flipping each next pointer as it goes, and returns
        the new head.
        ─ log ────────────────────────────────────────────────────────────────────────
        [prefill 1024 tok]
        warning: expert cache is 96 slots
        [decode expert io] hits 400

        """

    static let noColourSnapshot = """
        emone  qwen3.8-flash-next · 4-bit                      ⠸ 12/64 tok  12.5 tok/s

        ┌ L09 GDN · experts 0..511 ──────┐    layer    L9 / 48  GDN + MoE
        │································│
        │································│    router picks
        │································│    e296 hit          e299 MISS
        │································│    e302 hit          e305 hit
        │································│    e308 MISS         e311 hit
        │································│    e314 hit          e317 MISS
        │································│    e320 hit          e323 hit
        │································│
        │································│    cache    ██████████░░░░ 94% hit
        │········●··█··●··●·▃█·▃●·▃●·▃█▃▃│    tok/s            ████████████ 12.5
        │●▃▃●▃▃·▃▃·▃▃·▃▃·▃··▃··▃··▃······│    ssd              █▇▇███▇▇███▇ 2.5 GB/s
        │································│
        │································│    recent heat ▁▁▂▂▄▄▆▆██ cold→hot
        │································│    ▐▌ hit  ██ miss (SSD)  · not cached
        │································│
        │································│
        └────────────────────────────────┘

        L09 ▪▪▪▃▪▪▪▪ ·······································
        ─ output ─────────────────────────────────────────────────────────────────────
        It walks the list once, flipping each next pointer as it goes, and returns
        the new head.
        ─ log ────────────────────────────────────────────────────────────────────────
        [prefill 1024 tok]
        warning: expert cache is 96 slots
        [decode expert io] hits 400

        """

    static let denseSnapshot = """
        emone  m · 4-bit                                       ⠸ 12/64 tok  12.5 tok/s
        ─ output ─────────────────────────────────────────────────────────────────────
        It walks the list once, flipping each next pointer as it goes, and returns
        the new head.
        ─ log ────────────────────────────────────────────────────────────────────────
        [prefill 1024 tok]
        warning: expert cache is 96 slots
        [decode expert io] hits 400

        """
}
