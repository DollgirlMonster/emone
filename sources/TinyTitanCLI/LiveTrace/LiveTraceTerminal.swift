import Dispatch
import Foundation
import Synchronization
import TinyTitan

/// Write every byte, retrying short writes and EINTR. A blocked terminal
/// blocks the caller, which is why only the render thread uses it.
func liveTraceWriteAll(_ fd: Int32, _ bytes: [UInt8]) {
    var offset = 0
    while offset < bytes.count {
        let written = bytes.withUnsafeBytes { raw -> Int in
            guard let base = raw.baseAddress else { return 0 }
            return write(fd, base + offset, bytes.count - offset)
        }
        if written < 0 {
            if errno == EINTR { continue }
            return
        }
        if written == 0 { return }
        offset += written
    }
}

/// What the process's terminal looks like, read once at start.
struct TerminalProbe: Equatable, Sendable {
    var stdoutIsTTY: Bool
    var term: String?
    var colorTerm: String?
    var noColor: Bool
    var cols: Int
    var rows: Int

    static func current(
        stdoutFD: Int32 = STDOUT_FILENO,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> TerminalProbe {
        var size = winsize()
        let sized = ioctl(stdoutFD, UInt(TIOCGWINSZ), &size) == 0
        return TerminalProbe(
            stdoutIsTTY: isatty(stdoutFD) != 0,
            term: environment["TERM"],
            colorTerm: environment["COLORTERM"],
            noColor: !(environment["NO_COLOR"] ?? "").isEmpty,
            cols: sized ? Int(size.ws_col) : 0,
            rows: sized ? Int(size.ws_row) : 0)
    }
}

enum LiveTraceDecision: Equatable, Sendable {
    /// `--live-trace` was not given.
    case off
    /// Asked for, but this terminal cannot show it; the run goes on without.
    case disabled(reason: String)
    case run(plan: LiveTraceViewPlan, depth: LiveTraceColorDepth)
}

enum LiveTraceGate {
    /// The one place that decides whether anything is drawn. ANSI output
    /// happens only when stdout is a terminal that understands it.
    static func decide(
        requested: Bool, probe: TerminalProbe, shape: ExpertTraceShape
    ) -> LiveTraceDecision {
        guard requested else { return .off }
        guard probe.stdoutIsTTY else {
            return .disabled(reason: "stdout is not a terminal")
        }
        guard let term = probe.term, !term.isEmpty, term != "dumb" else {
            return .disabled(reason: "TERM is unset or dumb")
        }
        let plan = LiveTraceViewPlan.plan(shape: shape, cols: probe.cols, rows: probe.rows)
        if case .unavailable(let reason) = plan { return .disabled(reason: reason) }
        return .run(
            plan: plan,
            depth: .detect(term: term, colorTerm: probe.colorTerm, noColor: probe.noColor))
    }
}

/// A block of terminal lines redrawn in place, below whatever is already on
/// screen, so scrollback keeps what came before and the run's output follows.
///
/// Between frames the cursor rests on the region's first line. Autowrap is off
/// while it is open, so an over-wide line is cut rather than shoving the rest
/// of the frame down.
final class LiveTraceRegion: Sendable {
    let fd: Int32
    let height: Int

    init(fd: Int32, height: Int) {
        self.fd = fd
        self.height = height
    }

    static let hideCursorNoWrap = "\u{1B}[?25l\u{1B}[?7l"
    static let showCursorWrap = "\u{1B}[?7h\u{1B}[?25h"

    func openSequence() -> String {
        Self.hideCursorNoWrap + String(repeating: "\n", count: height) + "\u{1B}[\(height)A\r"
    }

    func frameSequence(_ lines: [String]) -> String {
        var out = ""
        for index in 0..<height {
            out += "\r\u{1B}[2K" + (index < lines.count ? lines[index] : "") + "\u{1B}[0m\n"
        }
        return out + "\u{1B}[\(height)A"
    }

    func closeSequence() -> String {
        "\u{1B}[\(height)B\r\u{1B}[0m" + Self.showCursorWrap
    }

    func open() { liveTraceWriteAll(fd, Array(openSequence().utf8)) }
    func draw(_ lines: [String]) { liveTraceWriteAll(fd, Array(frameSequence(lines).utf8)) }
    func close() { liveTraceWriteAll(fd, Array(closeSequence().utf8)) }
}

/// The last few stderr lines, for the log region.
final class LiveTraceLogTail: Sendable {
    static let capacity = 64
    private let lines = Mutex<[String]>([])

    func append(_ line: String) {
        lines.withLock { lines in
            lines.append(line)
            if lines.count > Self.capacity { lines.removeFirst(lines.count - Self.capacity) }
        }
    }

    func recent() -> [String] { lines.withLock { $0 } }
}

/// Bytes waiting to be handed back to the real stderr, bounded so a chatty run
/// cannot grow it without limit.
struct LiveTraceByteQueue: Sendable {
    private(set) var chunks: [[UInt8]] = []
    private(set) var bytes = 0
    private(set) var droppedChunks = 0
    let limit: Int

    init(limit: Int) { self.limit = limit }

    mutating func push(_ chunk: [UInt8]) {
        chunks.append(chunk)
        bytes += chunk.count
        while bytes > limit, chunks.count > 1 {
            bytes -= chunks.removeFirst().count
            droppedChunks += 1
        }
    }

    mutating func takeAll() -> [[UInt8]] {
        let all = chunks
        chunks.removeAll()
        bytes = 0
        return all
    }
}

/// Redirects a file descriptor (stderr) into a pipe for the life of the view.
///
/// A reader thread splits the stream into lines for the log region. The real
/// stderr never loses anything: when it is not a terminal (a `2>run.log`
/// redirect) the bytes are passed straight through, unchanged, by a separate
/// writer thread, so a slow sink can stall that thread but never the reader or
/// decode. When it is a terminal they are held, bounded, and written back when
/// the view closes, because writing them live would tear the display.
///
/// unchecked-invariant: `savedFD` and `readFD` are written once in `start` and
/// read afterwards; every other piece of shared state is a `Mutex`, an
/// `Atomic` or a semaphore.
final class LiveTraceStderrCapture: @unchecked Sendable {
    private let targetFD: Int32
    private var savedFD: Int32 = -1
    private var readFD: Int32 = -1
    private let log: LiveTraceLogTail
    private let passthrough: Bool
    private let queue: Mutex<LiveTraceByteQueue>
    private let writerWake = DispatchSemaphore(value: 0)
    private let readerDone = DispatchSemaphore(value: 0)
    private let writerDone = DispatchSemaphore(value: 0)
    private let closing = Atomic<Bool>(false)

    private init(targetFD: Int32, log: LiveTraceLogTail, passthrough: Bool, limit: Int) {
        self.targetFD = targetFD
        self.log = log
        self.passthrough = passthrough
        self.queue = Mutex(LiveTraceByteQueue(limit: limit))
    }

    /// nil when the pipe or the descriptor duplication fails; stderr is then
    /// left exactly as it was.
    static func start(
        targetFD: Int32 = STDERR_FILENO, log: LiveTraceLogTail
    ) -> LiveTraceStderrCapture? {
        let saved = dup(targetFD)
        guard saved >= 0 else { return nil }
        var fds: [Int32] = [0, 0]
        guard pipe(&fds) == 0 else {
            close(saved)
            return nil
        }
        let passthrough = isatty(saved) == 0
        let capture = LiveTraceStderrCapture(
            targetFD: targetFD, log: log, passthrough: passthrough,
            limit: passthrough ? 8 << 20 : 1 << 20)
        capture.savedFD = saved
        capture.readFD = fds[0]
        guard dup2(fds[1], targetFD) >= 0 else {
            close(saved)
            close(fds[0])
            close(fds[1])
            return nil
        }
        close(fds[1])
        let reader = Thread { capture.readLoop() }
        reader.qualityOfService = .utility
        reader.start()
        if passthrough {
            let writer = Thread { capture.writeLoop() }
            writer.qualityOfService = .utility
            writer.start()
        } else {
            capture.writerDone.signal()
        }
        return capture
    }

    private func readLoop() {
        var buffer = [UInt8](repeating: 0, count: 4096)
        var partial: [UInt8] = []
        while true {
            let count = read(readFD, &buffer, buffer.count)
            if count < 0 && errno == EINTR { continue }
            if count <= 0 { break }
            let chunk = Array(buffer[0..<count])
            queue.withLock { $0.push(chunk) }
            if passthrough { writerWake.signal() }
            for byte in chunk {
                if byte == 10 || byte == 13 {
                    emit(partial)
                    partial.removeAll(keepingCapacity: true)
                } else {
                    partial.append(byte)
                }
            }
        }
        emit(partial)
        readerDone.signal()
    }

    private func emit(_ bytes: [UInt8]) {
        guard !bytes.isEmpty else { return }
        log.append(LiveTraceText.sanitize(String(decoding: bytes, as: UTF8.self)))
    }

    private func writeLoop() {
        while true {
            writerWake.wait()
            let chunks = queue.withLock { $0.takeAll() }
            for chunk in chunks { liveTraceWriteAll(savedFD, chunk) }
            if closing.load(ordering: .acquiring) && queue.withLock({ $0.chunks.isEmpty }) {
                break
            }
        }
        writerDone.signal()
    }

    /// Put stderr back, let the reader and writer finish, and return what was
    /// held for a terminal stderr (empty when it was passed through).
    func stop() -> (held: [[UInt8]], droppedChunks: Int) {
        // Replacing the target closes the pipe's last write end, so the reader
        // sees end-of-file once it has consumed everything written.
        dup2(savedFD, targetFD)
        readerDone.wait()
        closing.store(true, ordering: .releasing)
        if passthrough { writerWake.signal() }
        writerDone.wait()
        close(readFD)
        close(savedFD)
        return queue.withLock { queue in
            let dropped = queue.droppedChunks
            return (passthrough ? [] : queue.takeAll(), dropped)
        }
    }
}
