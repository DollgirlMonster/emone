import Darwin
import Foundation

/// The terminal, as far as the dashboard needs one: raw mode, a size, keys, and
/// enough escape sequences to redraw in place.
///
/// Deliberately dependency-free — the alternative is ncurses through a system
/// library target, and none of this needs more than `termios`, `ioctl` and
/// `poll`. Everything terminal-shaped lives here so the view stays pure.
public final class FleetTerminal {
    private var original = termios()
    private var raw = false

    /// The terminal size in cells.
    public struct Size: Sendable, Equatable {
        public let columns: Int
        public let rows: Int
        public init(columns: Int, rows: Int) {
            self.columns = columns
            self.rows = rows
        }
    }

    /// Open the terminal in raw mode on the alternate screen.
    /// - Returns: `nil` when stdin is not a terminal, so a piped run cannot wedge.
    public init?() {
        guard isatty(STDIN_FILENO) == 1, isatty(STDOUT_FILENO) == 1 else { return nil }
        guard tcgetattr(STDIN_FILENO, &original) == 0 else { return nil }
        var settings = original
        // No echo (we draw the input line), no line buffering (keys arrive one at
        // a time), no signals (Ctrl-C comes through as a byte we can honour), no
        // CR→NL translation, no flow control.
        settings.c_lflag &= ~UInt(ECHO | ICANON | ISIG | IEXTEN)
        settings.c_iflag &= ~UInt(IXON | ICRNL | BRKINT | INPCK | ISTRIP)
        settings.c_cc.0 = 1  // VMIN: one byte is enough to return
        settings.c_cc.1 = 0  // VTIME: no timeout, `poll` decides
        guard tcsetattr(STDIN_FILENO, TCSANOW, &settings) == 0 else { return nil }
        raw = true
        write("\u{1B}[?1049h\u{1B}[?25l\u{1B}[2J")
    }

    deinit { restore() }

    /// Leave the alternate screen and put the terminal back as it was.
    public func restore() {
        guard raw else { return }
        write("\u{1B}[?25h\u{1B}[?1049l")
        _ = tcsetattr(STDIN_FILENO, TCSANOW, &original)
        raw = false
    }

    /// The current window size, re-read every frame so a resize is picked up
    /// without a signal handler.
    public func size() -> Size {
        var window = winsize()
        if ioctl(STDOUT_FILENO, TIOCGWINSZ, &window) == 0, window.ws_col > 0, window.ws_row > 0 {
            return Size(columns: Int(window.ws_col), rows: Int(window.ws_row))
        }
        return Size(columns: 100, rows: 30)
    }

    /// Wait up to `timeoutMs` for input and decode whatever arrived.
    /// - Parameter timeoutMs: how long to block with nothing to read.
    /// - Returns: keys in arrival order, empty on timeout.
    public func readKeys(timeoutMs: Int = 250) -> [FleetKey] {
        var descriptor = pollfd(fd: STDIN_FILENO, events: Int16(POLLIN), revents: 0)
        let ready = poll(&descriptor, 1, Int32(timeoutMs))
        guard ready > 0 else { return [] }
        var buffer = [UInt8](repeating: 0, count: 64)
        let count = read(STDIN_FILENO, &buffer, buffer.count)
        guard count > 0 else { return [] }
        return Self.decode(Array(buffer.prefix(count)))
    }

    /// Decode a byte run into keys, including the arrow escape sequences.
    static func decode(_ bytes: [UInt8]) -> [FleetKey] {
        var keys: [FleetKey] = []
        var index = 0
        while index < bytes.count {
            let byte = bytes[index]
            switch byte {
            case 0x1B:
                // Escape alone is a quit; `ESC [ X` is an arrow.
                if index + 2 < bytes.count, bytes[index + 1] == 0x5B {
                    switch bytes[index + 2] {
                    case 0x41: keys.append(.up)
                    case 0x42: keys.append(.down)
                    case 0x43: keys.append(.right)
                    case 0x44: keys.append(.left)
                    default: break
                    }
                    index += 3
                    continue
                }
                keys.append(.escape)
            case 0x0D, 0x0A:
                keys.append(.enter)
            case 0x7F, 0x08:
                keys.append(.backspace)
            case 0x03, 0x04:  // Ctrl-C, Ctrl-D
                keys.append(.escape)
            default:
                if let scalar = Self.scalar(at: index, in: bytes) {
                    keys.append(.character(String(scalar.value)))
                    index += scalar.length
                    continue
                }
            }
            index += 1
        }
        return keys
    }

    /// Read one UTF-8 scalar starting at `index`, reporting how many bytes it took.
    private static func scalar(at index: Int, in bytes: [UInt8]) -> (
        value: Unicode.Scalar, length: Int
    )? {
        let first = bytes[index]
        let length: Int
        var value: UInt32
        switch first {
        case 0x00...0x7F:
            return Unicode.Scalar(UInt32(first)).map { ($0, 1) }
        case 0xC0...0xDF:
            length = 2
            value = UInt32(first & 0x1F)
        case 0xE0...0xEF:
            length = 3
            value = UInt32(first & 0x0F)
        case 0xF0...0xF7:
            length = 4
            value = UInt32(first & 0x07)
        default:
            return nil
        }
        guard index + length <= bytes.count else { return nil }
        for offset in 1..<length {
            let byte = bytes[index + offset]
            guard byte & 0xC0 == 0x80 else { return nil }
            value = (value << 6) | UInt32(byte & 0x3F)
        }
        return Unicode.Scalar(value).map { ($0, length) }
    }

    /// Draw a whole frame: home the cursor, then one line at a time, erasing the
    /// rest of each so a narrower frame leaves nothing behind.
    public func draw(_ frame: FleetFrame) {
        var output = "\u{1B}[H"
        for (index, line) in frame.lines.enumerated() {
            if index == frame.selectedLine {
                output += "\u{1B}[7m" + line + "\u{1B}[0m"
            } else {
                output += line
            }
            output += "\u{1B}[K\r\n"
        }
        output += "\u{1B}[J"
        write(output)
    }

    private func write(_ text: String) {
        FileHandle.standardOutput.write(Data(text.utf8))
    }
}
