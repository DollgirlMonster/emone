import TinyTitan

/// How the experts of one layer are laid out as a grid of cells.
///
/// The grid always occupies at most `maxGridChars` x `maxRows` terminal
/// characters, so the view's shape stays put whatever the model. What follows
/// the model is how that budget is spent:
///
/// - up to 256 experts: 16 columns of 2-character cells, one expert per cell
///   (a 256-expert layer is 16 x 16);
/// - up to 512 experts: 32 columns of 1-character cells, one expert per cell
///   (a 512-expert layer is 32 x 16);
/// - more than 512: still 32 x 16 cells, each standing for `bin` consecutive
///   experts and showing the strongest state among them.
struct ExpertGridLayout: Equatable, Sendable {
    static let maxGridChars = 32
    static let maxRows = 16

    let numExperts: Int
    let cols: Int
    let rows: Int
    /// Terminal characters per cell, 1 or 2.
    let cellWidth: Int
    /// Experts represented by one cell.
    let bin: Int

    /// Width of the grid body in terminal characters, borders excluded.
    var bodyWidth: Int { cols * cellWidth }

    /// Number of cells that stand for at least one expert.
    var usedCells: Int { (numExperts + bin - 1) / bin }

    /// The cell an expert falls in.
    func cell(of expert: Int) -> Int { expert / bin }

    /// The experts a cell stands for, or nil for padding past the last expert.
    func experts(inCell cell: Int) -> Range<Int>? {
        let lower = cell * bin
        guard cell >= 0, lower < numExperts else { return nil }
        return lower..<min(numExperts, lower + bin)
    }

    /// nil when the model has no routed experts.
    static func make(numExperts: Int) -> ExpertGridLayout? {
        guard numExperts > 0 else { return nil }
        let cellWidth = numExperts <= 256 ? 2 : 1
        let fullCols = maxGridChars / cellWidth
        let cols = min(fullCols, numExperts)
        let maxCells = fullCols * maxRows
        let bin = (numExperts + maxCells - 1) / maxCells
        let used = (numExperts + bin - 1) / bin
        let rows = (used + cols - 1) / cols
        return ExpertGridLayout(
            numExperts: numExperts, cols: cols, rows: rows, cellWidth: cellWidth, bin: bin)
    }
}

/// Which of the view's two shapes fits, or why neither does.
enum LiveTraceViewPlan: Equatable, Sendable {
    /// Header, expert grid, router panel, layer ribbon, 2-line output, log.
    case full(width: Int, height: Int)
    /// Header, 2-line output, log. Dense models, and terminals too short for
    /// the grid.
    case compact(width: Int, height: Int)
    case unavailable(reason: String)

    static let fullWidth = 78
    static let minimumWidth = 40
    static let outputLines = 2
    static let logLines = 4
    /// Router-panel rows spent on picks, two per row.
    static let maxPickRows = 8

    var height: Int? {
        switch self {
        case .full(_, let h), .compact(_, let h): return h
        case .unavailable: return nil
        }
    }

    var width: Int? {
        switch self {
        case .full(let w, _), .compact(let w, _): return w
        case .unavailable: return nil
        }
    }

    /// Height of the compact view: header, output rule, output, log rule, log.
    static var compactHeight: Int { 1 + 1 + outputLines + 1 + logLines }

    /// Height of the full view for a routed model.
    static func fullHeight(shape: ExpertTraceShape) -> Int? {
        guard shape.isRouted, let grid = ExpertGridLayout.make(numExperts: shape.numExperts)
        else { return nil }
        // header, blank, [grid | panel], blank, ribbon, output rule + 2, log rule + 4
        return 1 + 1 + gridBlockHeight(grid: grid, topK: shape.topK) + 1 + 1 + 1
            + outputLines + 1 + logLines
    }

    /// Lines the grid and the router panel share. The panel needs the layer
    /// line, the picks, the cache/rate rows and the legend; the grid its rows
    /// plus a border above and below. Whichever is taller sets the block.
    static func gridBlockHeight(grid: ExpertGridLayout, topK: Int) -> Int {
        max(grid.rows + 2, panelHeight(topK: topK))
    }

    /// layer, blank, "router picks", picks, blank, cache, tok/s, ssd, blank, 2 legend.
    static func panelHeight(topK: Int) -> Int { 10 + pickRows(topK: topK) }

    static func pickRows(topK: Int) -> Int { min(maxPickRows, max(1, (topK + 1) / 2)) }

    /// Choose a view for this model and terminal. One row below the view is
    /// kept free: the in-place redraw parks the cursor under it.
    static func plan(shape: ExpertTraceShape, cols: Int, rows: Int) -> LiveTraceViewPlan {
        guard cols >= minimumWidth else {
            return .unavailable(
                reason:
                    "the terminal is \(cols) columns wide; the view needs at least \(minimumWidth)")
        }
        let width = min(cols, fullWidth)
        if cols >= fullWidth, let full = fullHeight(shape: shape), rows >= full + 1 {
            return .full(width: width, height: full)
        }
        guard rows >= compactHeight + 1 else {
            return .unavailable(
                reason:
                    "the terminal is \(rows) rows tall; the view needs at least \(compactHeight + 1)"
            )
        }
        return .compact(width: width, height: compactHeight)
    }
}
