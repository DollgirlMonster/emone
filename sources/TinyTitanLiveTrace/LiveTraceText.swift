/// Text handling for the live trace view: everything here is pure so the
/// wrapping and width rules can be tested without a terminal.
enum LiveTraceText {
    /// Terminal cells a character occupies: 2 for East Asian wide characters and
    /// emoji presentation, 1 otherwise. A coarse table, not full UAX #11, but it
    /// keeps the usual CJK and emoji output from overflowing a column.
    static func width(of character: Character) -> Int {
        guard let scalar = character.unicodeScalars.first else { return 0 }
        switch scalar.value {
        case 0x1100...0x115F, 0x2E80...0xA4CF, 0xAC00...0xD7A3, 0xF900...0xFAFF,
            0xFE30...0xFE6F, 0xFF00...0xFF60, 0xFFE0...0xFFE6, 0x1F300...0x1F64F,
            0x1F900...0x1F9FF, 0x20000...0x3FFFD:
            return 2
        default:
            return 1
        }
    }

    static func width(of string: String) -> Int {
        string.reduce(0) { $0 + width(of: $1) }
    }

    /// One line of untrusted text (a stderr line) made safe to print inside the
    /// view: control characters and escape sequences (CSI, OSC and two-byte
    /// escapes) removed, tabs widened.
    static func sanitize(_ line: String) -> String {
        enum State { case text, escape, csi, osc, oscEscape }
        var state = State.text
        var out = ""
        for scalar in line.unicodeScalars {
            switch state {
            case .escape:
                if scalar == "[" {
                    state = .csi
                } else if scalar == "]" {
                    state = .osc
                } else {
                    state = .text
                }
            case .csi:
                if (0x40...0x7E).contains(scalar.value) { state = .text }
            case .osc:
                if scalar.value == 0x07 {
                    state = .text
                } else if scalar.value == 0x1B {
                    state = .oscEscape
                }
            case .oscEscape:
                state = .text
            case .text:
                switch scalar.value {
                case 0x1B: state = .escape
                case 0x09: out += "    "
                case 0x00...0x1F, 0x7F, 0x80...0x9F: continue
                default: out.unicodeScalars.append(scalar)
                }
            }
        }
        return out
    }

    /// Cut a string to at most `width` cells.
    static func truncate(_ string: String, width: Int) -> String {
        var used = 0
        var out = ""
        for character in string {
            let w = Self.width(of: character)
            if used + w > width { break }
            out.append(character)
            used += w
        }
        return out
    }

    /// Wrap generated text to `width` cells: hard breaks at newlines, soft
    /// breaks at the last space that fits, and a hard cut inside a word longer
    /// than a line. Always returns at least one line.
    static func wrap(_ text: String, width: Int) -> [String] {
        let width = max(1, width)
        var lines: [String] = []
        // "\r\n" is one Character in Swift and would not match "\n": drop CRs first.
        let normalized = String(String.UnicodeScalarView(text.unicodeScalars.filter { $0 != "\r" }))
        for paragraph in normalized.split(separator: "\n", omittingEmptySubsequences: false) {
            lines.append(contentsOf: wrapParagraph(String(paragraph), width: width))
        }
        return lines.isEmpty ? [""] : lines
    }

    /// Cut a line into rows of at most `width` cells with no regard for words.
    static func hardWrap(_ string: String, width: Int) -> [String] {
        let width = max(1, width)
        var rows: [String] = []
        var current = ""
        var used = 0
        for character in string {
            let w = Self.width(of: character)
            if used + w > width {
                rows.append(current)
                current = ""
                used = 0
            }
            current.append(character)
            used += w
        }
        rows.append(current)
        return rows
    }

    private static func wrapParagraph(_ paragraph: String, width: Int) -> [String] {
        var lines: [String] = []
        var current: [Character] = []
        var currentWidth = 0
        for raw in paragraph {
            let character: Character = raw == "\t" ? " " : raw
            if let scalar = character.unicodeScalars.first,
                scalar.value < 0x20 || scalar.value == 0x7F
            {
                continue
            }
            let w = Self.width(of: character)
            if currentWidth + w > width {
                if let space = current.lastIndex(of: " "), space > 0 {
                    lines.append(String(current[..<space]))
                    current = Array(current[(space + 1)...])
                } else {
                    lines.append(String(current))
                    current = []
                }
                currentWidth = current.reduce(0) { $0 + Self.width(of: $1) }
                if character == " " && current.isEmpty { continue }
            }
            current.append(character)
            currentWidth += w
        }
        lines.append(String(current))
        return lines
    }
}
