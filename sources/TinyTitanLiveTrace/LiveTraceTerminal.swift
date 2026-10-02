import Dispatch
import Foundation
import Synchronization
import TinyTitan

/// Write the bytes without ever blocking for long on the terminal.
///
/// The view's fd is the process's own stdout (or a dup sharing its open file
/// description), so it cannot be switched to O_NONBLOCK without changing every
/// other writer's stdout. Instead each small chunk waits at most
/// `liveTraceWriteTimeoutMs` for the terminal to accept it; a terminal that
/// stops reading (Ctrl-S, a stalled ssh session, a pty nobody drains) loses the
/// rest of this frame -- the next frame redraws the region whole -- rather than
/// stalling the render thread and, through it, shutdown. Returns false when it
/// gave up.
let liveTraceWriteTimeoutMs: Int32 = 50
let liveTraceWriteChunk = 512

@discardableResult
func liveTraceWriteAll(_ fd: Int32, _ bytes: [UInt8]) -> Bool {
    var offset = 0
    while offset < bytes.count {
        var pfd = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
        let ready = poll(&pfd, 1, liveTraceWriteTimeoutMs)
        if ready < 0 {
            if errno == EINTR { continue }
            return false
        }
        if ready == 0 || pfd.revents & Int16(POLLOUT) == 0 { return false }
        let length = min(liveTraceWriteChunk, bytes.count - offset)
        let written = bytes.withUnsafeBytes { raw -> Int in
            guard let base = raw.baseAddress else { return 0 }
            return write(fd, base + offset, length)
        }
        if written < 0 {
            if errno == EINTR || errno == EAGAIN { continue }
            return false
        }
        if written == 0 { return false }
        offset += written
    }
    return true
}

/// What the process's terminal looks like, read once at start.
public struct TerminalProbe: Equatable, Sendable {
    public var stdoutIsTTY: Bool
    public var term: String?
    public var colorTerm: String?
    public var noColor: Bool
    public var cols: Int
    public var rows: Int

    public init(
        stdoutIsTTY: Bool, term: String?, colorTerm: String?, noColor: Bool, cols: Int, rows: Int
    ) {
        self.stdoutIsTTY = stdoutIsTTY
        self.term = term
        self.colorTerm = colorTerm
        self.noColor = noColor
        self.cols = cols
        self.rows = rows
    }

    public static func current(
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

extension LiveTraceGate {
    /// A server's gate, before any model is known: whether this terminal can
    /// show a view at all. A shape with no routed experts asks for the smallest
    /// view, so this passes exactly when the compact view fits, and the
    /// per-model choice between that and the grid is made when a model loads.
    static func decideForServer(requested: Bool, probe: TerminalProbe) -> LiveTraceDecision {
        decide(requested: requested, probe: probe, shape: LiveTraceModelInfo.placeholderShape(name: ""))
    }
}

/// A block of terminal lines redrawn in place, below whatever is already on
/// screen, so scrollback keeps what came before and the run's output follows.
///
/// Between frames the cursor rests on the region's first line. Autowrap is off
/// while it is open, so an over-wide line is cut rather than shoving the rest
/// of the frame down.
///
/// The height can change while the region is open (`relayout`), which is how a
/// server's view follows a model switch to a different shape.
///
/// unchecked-invariant: `height` is written by `relayout`, and read by the other
/// methods, on one thread at a time: the render thread while it runs, the
/// caller of `open` and `close` before it starts and after it has been joined.
final class LiveTraceRegion: @unchecked Sendable {
    let fd: Int32
    private(set) var height: Int

    init(fd: Int32, height: Int) {
        self.fd = fd
        self.height = height
    }

    static let hideCursorNoWrap = "\u{1B}[?25l\u{1B}[?7l"
    static let showCursorWrap = "\u{1B}[?7h\u{1B}[?25h"

    func openSequence() -> String {
        Self.hideCursorNoWrap + Self.reserve(height)
    }

    /// Make `height` lines of room below the cursor and come back to the first.
    static func reserve(_ height: Int) -> String {
        String(repeating: "\n", count: height) + "\u{1B}[\(height)A\r"
    }

    func frameSequence(_ lines: [String]) -> String {
        var out = ""
        for index in 0..<height {
            out += "\r\u{1B}[2K" + (index < lines.count ? lines[index] : "") + "\u{1B}[0m\n"
        }
        return out + "\u{1B}[\(height)A"
    }

    func closeSequence() -> String { Self.closeSequence(height: height) }

    /// Below the region, colours reset, cursor and autowrap back.
    static func closeSequence(height: Int) -> String {
        "\u{1B}[\(height)B\r\u{1B}[0m" + showCursorWrap
    }

    /// Erase the region and reserve `newHeight` lines from the same first line,
    /// for a view whose shape changed. The cursor is on the first line, so
    /// clearing to the end of the screen removes every old line and nothing
    /// above; scrollback keeps no stale frame.
    func relayoutSequence(to newHeight: Int) -> String {
        "\r\u{1B}[J" + Self.reserve(newHeight)
    }

    func open() {
        liveTraceWriteAll(fd, Array(openSequence().utf8))
        LiveTraceTerminalGuard.arm(fd: fd, height: height)
    }

    func draw(_ lines: [String]) {
        guard !LiveTraceTerminalGuard.wasRestored else { return }
        liveTraceWriteAll(fd, Array(frameSequence(lines).utf8))
    }

    func relayout(to newHeight: Int) {
        guard newHeight != height, !LiveTraceTerminalGuard.wasRestored else { return }
        liveTraceWriteAll(fd, Array(relayoutSequence(to: newHeight).utf8))
        height = newHeight
        LiveTraceTerminalGuard.update(height: newHeight)
    }

    func close() {
        liveTraceWriteAll(fd, Array(closeSequence().utf8))
        LiveTraceTerminalGuard.disarm()
    }
}

/// Puts the terminal back when the process leaves without closing the view: a
/// second termination signal that ends it with `exit`, or any other `exit`
/// while a view is open. A view hides the cursor and turns autowrap off, and a
/// shell left that way is unusable until `reset`.
///
/// This is what a signal or an `exit` can still do; a crash (SIGSEGV, SIGKILL)
/// runs nothing, and the CLI has never tried to survive one.
///
/// One view at a time, process-wide, which is how both front ends use it.
enum LiveTraceTerminalGuard {
    private static let fd = Atomic<Int32>(-1)
    private static let height = Atomic<Int>(0)
    private static let registered = Atomic<Bool>(false)
    /// Set once `restore` has run, so a render thread that is mid-frame when a
    /// forced exit restores the terminal does not draw over the shell's prompt.
    private static let restored = Atomic<Bool>(false)

    static func arm(fd newFD: Int32, height newHeight: Int) {
        height.store(newHeight, ordering: .relaxed)
        restored.store(false, ordering: .relaxed)
        fd.store(newFD, ordering: .releasing)
        if registered.compareExchange(
            expected: false, desired: true, ordering: .acquiringAndReleasing
        ).exchanged {
            atexit { LiveTraceTerminalGuard.restore() }
        }
    }

    static func update(height newHeight: Int) { height.store(newHeight, ordering: .relaxed) }

    static func disarm() { fd.store(-1, ordering: .releasing) }

    static var isArmed: Bool { fd.load(ordering: .acquiring) >= 0 }

    /// True after `restore` has put the terminal back and until the next `arm`.
    static var wasRestored: Bool { restored.load(ordering: .relaxed) }

    /// Write the closing sequence once, if a view is open. Safe to call from
    /// anywhere, any number of times: the first call disarms.
    static func restore() {
        let target = fd.exchange(-1, ordering: .acquiringAndReleasing)
        guard target >= 0 else { return }
        restored.store(true, ordering: .relaxed)
        let sequence = LiveTraceRegion.closeSequence(height: height.load(ordering: .relaxed))
        liveTraceWriteAll(target, Array(sequence.utf8))
    }
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

/// Redirects a file descriptor (stderr, or stdout for a server) into a pipe for
/// the life of the view.
///
/// A reader thread splits the stream into lines for the log region. Nothing is
/// lost from the real destination:
///
/// - when it is not a terminal (a `2>run.log` redirect) the bytes are passed
///   straight through, unchanged, by a separate writer thread;
/// - when a `fileSinkFD` is given (the server's log file) they are appended to it
///   by that same writer thread;
/// - otherwise (a terminal, no file) they are held, bounded, and written back
///   when the view closes, because writing them live would tear the display.
///
/// The writer thread is why a slow sink can stall that thread but never the
/// reader, the pipe, or whoever is logging: the reader only queues.
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
    /// Where the writer thread sends bytes: the log file, or the original
    /// descriptor. Meaningful only when `passthrough`.
    private var sinkFD: Int32 = -1
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

    /// The descriptor the target pointed at before the redirect: for a view that
    /// redirects stdout, the terminal it has to keep drawing on. Valid until
    /// `stop`.
    var originalFD: Int32 { savedFD }

    /// nil when the pipe or the descriptor duplication fails; the target is then
    /// left exactly as it was.
    static func start(
        targetFD: Int32 = STDERR_FILENO, log: LiveTraceLogTail, fileSinkFD: Int32? = nil
    ) -> LiveTraceStderrCapture? {
        let saved = dup(targetFD)
        guard saved >= 0 else { return nil }
        var fds: [Int32] = [0, 0]
        guard pipe(&fds) == 0 else {
            close(saved)
            return nil
        }
        let passthrough = fileSinkFD != nil || isatty(saved) == 0
        let capture = LiveTraceStderrCapture(
            targetFD: targetFD, log: log, passthrough: passthrough,
            limit: passthrough ? 8 << 20 : 1 << 20)
        capture.savedFD = saved
        capture.readFD = fds[0]
        capture.sinkFD = fileSinkFD ?? saved
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
            for chunk in chunks { liveTraceWriteAll(sinkFD, chunk) }
            if closing.load(ordering: .acquiring) && queue.withLock({ $0.chunks.isEmpty }) {
                break
            }
        }
        writerDone.signal()
    }

    /// Put the target back, let the reader and writer finish, and return what
    /// was held for a terminal with no file sink (empty otherwise).
    ///
    /// The reader ends when every write end of the pipe is closed. A process the
    /// server started that still holds one would keep it open, so the wait is
    /// bounded: shutdown must not hang on a stray descriptor.
    func stop() -> (held: [[UInt8]], droppedChunks: Int) {
        // Anything the process buffered in stdio before the swap goes down the
        // pipe now, not after the target is back.
        fflush(nil)
        // Replacing the target closes the pipe's last write end, so the reader
        // sees end-of-file once it has consumed everything written.
        dup2(savedFD, targetFD)
        _ = readerDone.wait(timeout: .now() + 2)
        closing.store(true, ordering: .releasing)
        if passthrough { writerWake.signal() }
        _ = writerDone.wait(timeout: .now() + 2)
        close(readFD)
        close(savedFD)
        return queue.withLock { queue in
            let dropped = queue.droppedChunks
            return (passthrough ? [] : queue.takeAll(), dropped)
        }
    }
}
