import Foundation

/// The full copy of what a view captured, for a terminal that has nowhere else
/// to put it.
///
/// A CLI run is short, so it holds what the view hid and prints it on exit. A
/// server runs for days and logs a line or more per request; holding that in
/// memory until exit would either grow without bound or drop most of it. With
/// the view on, the server's own log (stderr) and anything it prints to stdout
/// are appended here instead of to the terminal the view is drawing on.
///
/// Append-only, owner-readable, one descriptor shared by the capture threads
/// (every `write` to an `O_APPEND` file lands whole at the end).
final class LiveTraceLogFile: Sendable {
    let path: String
    let fd: Int32

    private init(path: String, fd: Int32) {
        self.path = path
        self.fd = fd
    }

    /// `~/Library/Logs/TinyTitan/server-live-trace.log`: where macOS tools keep
    /// per-user logs, and a place the server's operator can find again.
    static func defaultPath(
        home: String = FileManager.default.homeDirectoryForCurrentUser.path
    ) -> String {
        home + "/Library/Logs/TinyTitan/server-live-trace.log"
    }

    /// nil when the directory cannot be made or the file cannot be opened; the
    /// caller then falls back to holding the log until the view closes.
    static func open(path: String, now: Date = Date()) -> LiveTraceLogFile? {
        let directory = (path as NSString).deletingLastPathComponent
        if !directory.isEmpty {
            try? FileManager.default.createDirectory(
                atPath: directory, withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
        }
        let fd = Darwin.open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0o600)
        guard fd >= 0 else { return nil }
        let file = LiveTraceLogFile(path: path, fd: fd)
        let header = "--- live-trace view opened \(now.formatted(.iso8601)) pid \(getpid()) ---\n"
        liveTraceWriteAll(fd, Array(header.utf8))
        return file
    }

    func close() {
        let footer = "--- live-trace view closed \(Date().formatted(.iso8601)) ---\n"
        liveTraceWriteAll(fd, Array(footer.utf8))
        Darwin.close(fd)
    }
}
