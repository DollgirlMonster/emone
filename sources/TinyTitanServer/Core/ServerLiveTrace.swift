import Foundation
import Synchronization
import TinyTitan
import TinyTitanLiveTrace

/// The server's side of `--live-trace`: whether the live view exists, and the
/// one place the rest of the server finds it.
///
/// With the flag off nothing is ever installed, and every hook in the server is
/// a single read of `current` that finds nil. Hooks are read once per request,
/// never per token: a request's decode loop holds the `LiveTraceGeneration` it
/// was given and the loop's cost is one `nil` check per token.
///
/// The view itself (the renderer, the terminal handling, the model of what the
/// router did) is the shared `TinyTitanLiveTrace` target that the CLI's
/// `--live-trace` uses too. What lives here is only what is the server's: when
/// to build it, and what it is told.
public enum ServerLiveTrace {
    private static let installed = Mutex<LiveTraceServerView?>(nil)

    /// The view, when one is installed.
    public static var current: LiveTraceServerView? { installed.withLock { $0 } }

    /// Build the view if `--live-trace` was given and this terminal can show it.
    /// The view is installed but draws nothing until `open`, so what loads at
    /// startup is recorded and the startup banner is still printed normally.
    ///
    /// nil when the flag is off (silently) or the terminal cannot show it (one
    /// line on `note`, stderr by default).
    @discardableResult
    public static func install(
        arguments: ServerArguments,
        nameOverride: String? = nil,
        probe: TerminalProbe? = nil,
        note: @escaping @Sendable (String) -> Void = LiveTraceServerView.noteToStderr
    ) -> LiveTraceServerView? {
        guard arguments.liveTrace else { return nil }
        let view = LiveTraceServerView.prepare(
            options: LiveTraceServerView.Options(
                logPath: arguments.liveTraceLogPath, nameOverride: nameOverride),
            probe: probe, note: note)
        installed.withLock { $0 = view }
        return view
    }

    /// Install a view built by the caller; for tests that need a terminal of
    /// their own choosing.
    static func install(_ view: LiveTraceServerView?) {
        installed.withLock { $0 = view }
    }

    /// Close the view, putting the terminal, stdout and stderr back, and remove
    /// it. Safe to call when nothing is installed, and more than once.
    public static func shutdown() {
        let view = installed.withLock { view -> LiveTraceServerView? in
            let taken = view
            view = nil
            return taken
        }
        view?.close()
    }
}
