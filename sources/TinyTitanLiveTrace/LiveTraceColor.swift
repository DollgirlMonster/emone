/// How many colours the terminal can show.
enum LiveTraceColorDepth: Int, Comparable, Sendable {
    case none = 0
    case ansi16
    case ansi256
    case truecolor

    static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }

    /// From the environment, conservatively: `NO_COLOR` wins, `COLORTERM` can
    /// promise 24-bit, a `TERM` ending in 256color promises 256, and anything
    /// else that is a real terminal gets the 16 every one of them has.
    static func detect(term: String?, colorTerm: String?, noColor: Bool) -> Self {
        if noColor { return .none }
        guard let term = term?.lowercased(), !term.isEmpty, term != "dumb" else { return .none }
        let colorTerm = colorTerm?.lowercased() ?? ""
        if colorTerm.contains("truecolor") || colorTerm.contains("24bit") || term.contains("direct")
        {
            return .truecolor
        }
        if term.contains("256color") { return .ansi256 }
        return .ansi16
    }
}

/// A colour role. `LiveTraceTheme` turns it into an escape sequence.
enum LiveTraceStyle: Equatable, Sendable {
    case plain, dim, veryDim, accent, hit, miss, cursor
    /// A rising green, 1...8, for sparklines.
    case heat(Int)
    /// Misses in a layer, 1...8: green through red.
    case missLevel(Int)
}

enum LiveTraceTheme {
    static let reset = "\u{1B}[0m"
    private static let greens256 = [22, 28, 34, 35, 41, 77, 114, 157]
    private static let missColors256 = [71, 107, 143, 179, 214, 208, 202, 196]

    /// Foreground sequence for a role, empty when there is no colour.
    static func foreground(_ style: LiveTraceStyle, depth: LiveTraceColorDepth) -> String {
        guard depth > .none else { return style == .cursor ? "\u{1B}[7m" : "" }
        if depth == .ansi16 {
            switch style {
            case .plain: return ""
            case .dim, .veryDim: return "\u{1B}[90m"
            case .accent: return "\u{1B}[36m"
            case .hit: return "\u{1B}[92m"
            case .miss: return "\u{1B}[91m"
            case .cursor: return "\u{1B}[7m"
            case .heat(let level): return level >= 5 ? "\u{1B}[92m" : "\u{1B}[32m"
            case .missLevel(let level):
                return level <= 2 ? "\u{1B}[32m" : level <= 5 ? "\u{1B}[33m" : "\u{1B}[91m"
            }
        }
        switch style {
        case .plain: return ""
        case .dim: return "\u{1B}[38;5;244m"
        case .veryDim: return "\u{1B}[38;5;238m"
        case .accent: return "\u{1B}[38;5;80m"
        case .hit: return "\u{1B}[38;5;84m"
        case .miss: return "\u{1B}[38;5;203m"
        case .cursor: return "\u{1B}[7m"
        case .heat(let level): return "\u{1B}[38;5;\(greens256[max(1, min(8, level)) - 1])m"
        case .missLevel(let level):
            return "\u{1B}[38;5;\(missColors256[max(1, min(8, level)) - 1])m"
        }
    }

    /// The heat ramp: dim blue-grey, through dim purple and crimson, to amber.
    private static let heatStops: [(t: Double, r: Double, g: Double, b: Double)] = [
        (0.00, 22, 24, 30), (0.30, 60, 44, 66), (0.55, 112, 52, 60), (0.80, 160, 84, 40),
        (1.00, 196, 128, 36),
    ]

    /// Below this a cell shows the terminal's own background.
    static let heatFloor = 0.04

    static func heatRGB(_ heat: Double) -> (r: Int, g: Int, b: Int) {
        let t = max(0, min(1, heat))
        for index in 1..<heatStops.count where t <= heatStops[index].t {
            let low = heatStops[index - 1]
            let high = heatStops[index]
            let f = (t - low.t) / (high.t - low.t)
            return (
                Int((low.r + (high.r - low.r) * f).rounded()),
                Int((low.g + (high.g - low.g) * f).rounded()),
                Int((low.b + (high.b - low.b) * f).rounded())
            )
        }
        let last = heatStops[heatStops.count - 1]
        return (Int(last.r), Int(last.g), Int(last.b))
    }

    /// Nearest entry of the xterm 6x6x6 colour cube.
    static func xterm256(r: Int, g: Int, b: Int) -> Int {
        func step(_ value: Int) -> Int {
            let levels = [0, 95, 135, 175, 215, 255]
            var best = 0
            for (index, level) in levels.enumerated()
            where abs(level - value) < abs(levels[best] - value) { best = index }
            return best
        }
        return 16 + 36 * step(r) + 6 * step(g) + step(b)
    }

    /// Background sequence for a heat in 0...1, empty below the floor and when
    /// the terminal has no colour.
    static func heatBackground(_ heat: Double, depth: LiveTraceColorDepth) -> String {
        guard depth > .none, heat >= heatFloor else { return "" }
        switch depth {
        case .none:
            return ""
        case .ansi16:
            return heat < 0.35 ? "\u{1B}[100m" : heat < 0.75 ? "\u{1B}[41m" : "\u{1B}[43m"
        case .ansi256:
            let c = heatRGB(heat)
            return "\u{1B}[48;5;\(xterm256(r: c.r, g: c.g, b: c.b))m"
        case .truecolor:
            let c = heatRGB(heat)
            return "\u{1B}[48;2;\(c.r);\(c.g);\(c.b)m"
        }
    }
}
