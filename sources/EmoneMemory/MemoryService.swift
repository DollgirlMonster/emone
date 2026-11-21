import ContinuityCore
import Foundation

/// What the serving engine talks to.
///
/// It owns the choice of backend, the fallback when durable storage cannot
/// be written, and the session lifecycle. The engine calls four things: `beginSession`,
/// `instructions`, `execute` and `endSession`. Everything else stays here.
///
/// The service never throws at the engine. Memory is optional by design, so
/// a failure produces a logged event and a degraded mode, not a failed
/// completion. The one thing it will not do is report a write as successful
/// when it was not.
public actor MemoryService {
    public private(set) var configuration: MemoryConfiguration
    /// A store supplied by the caller, used for every scope. Nil in normal
    /// operation; this is how the tests drive the service.
    private let injectedStore: (any MemoryStore)?
    private let injectedJournal: (any SessionJournal)?
    /// The resident side-engine, when one is wired. Nil in every test and in a
    /// deployment that has not asked for one, which is what keeps the
    /// callers' no-decision fallback the ordinary path rather than a special
    /// case.
    private let sideEngine: (any MemorySideEngine)?
    /// The background T7 pass and the ranking hints it leaves. Nil exactly
    /// when no side-engine is wired, so the deterministic search path is the
    /// only path a deployment without an engine ever takes.
    private let retrievalHinter: MemoryRetrievalHinter?
    /// How many questions one consolidation may put to the side-engine, in
    /// total.
    ///
    /// A judgement is a full generation on a CPU model, not a lookup:
    /// measured over the wired cases, 15.2 s per case on the 4B and 29.8 s on
    /// the 9B (`docs/side-engine-tasks.md`). A per-fact candidate loop would
    /// therefore cost minutes, so the loop is bounded by a budget for the
    /// whole consolidation instead. Six questions is about a minute and a half
    /// on the 4B, which is the pause consolidation already runs in.
    static let maximumSideEngineQuestions = 6
    /// And how many of them one fact may use — durability, then a duplicate
    /// and a contradiction against one candidate — so the first fact cannot
    /// spend the whole budget.
    static let maximumQuestionsPerFact = 3
    /// One engine, one journal file and one workspace lock per scope.
    ///
    /// Not one engine for the whole service: a request that names another
    /// workspace would otherwise have its facts written into the default
    /// workspace's file, so deleting one project's memory would delete
    /// another's, and the "one file per workspace" the documentation promises
    /// would be false.
    private var workspaces: [MemoryScope: Workspace] = [:]
    /// When each workspace was last used, for deciding which to let go of.
    private var lastUsed: [MemoryScope: Date] = [:]
    private let localStore: InMemoryStore
    /// The engine-authored journal. Separate store, separate key space,
    /// separate trim policy: a busy week of sessions must never evict the
    /// facts the model wrote deliberately.
    private let journalFilter: JournalFilter
    /// Set once a durable operation has failed, so the session prompt can say
    /// memory is not persisting instead of the model assuming it is.
    private var isDegraded = false
    /// Workspaces whose journal failure has been logged, so a disk that stays
    /// full produces one line rather than one per tool call.
    private var reportedJournalFailures: Set<MemoryScope> = []
    private var log: @Sendable (MemoryLogEvent) -> Void

    /// Everything one scope needs, created on first use.
    private struct Workspace {
        let store: any MemoryStore
        let journal: (any SessionJournal)?
        /// Nil when the caller injected its own store, in which case this
        /// service owns no engine to close.
        let engine: ContinuityEngine?
        /// False when the engine has no journal behind it, so writes last
        /// only as long as the process.
        let persists: Bool
    }

    public init(
        configuration: MemoryConfiguration,
        durableStore: (any MemoryStore)? = nil,
        journal: (any SessionJournal)? = nil,
        sideEngine: (any MemorySideEngine)? = nil,
        isIdle: (@Sendable () -> Bool)? = nil,
        log: @escaping @Sendable (MemoryLogEvent) -> Void = { _ in }
    ) {
        self.configuration = configuration
        self.localStore = InMemoryStore(limits: configuration.limits)
        self.journalFilter = configuration.journalLimits.filter
        self.log = log
        self.injectedStore = durableStore
        self.injectedJournal = journal
        self.sideEngine = sideEngine
        // A caller with no idle signal — a benchmark, a test — reads as
        // always idle, which is what makes the pass finish deterministically
        // off the server.
        self.retrievalHinter = sideEngine.map {
            MemoryRetrievalHinter(engine: $0, isIdle: isIdle ?? { true }, log: log)
        }
    }

    /// The engine, store and journal for a scope, built on first use.
    ///
    /// Deferred rather than done in `init` because opening a journal is I/O
    /// that can fail, and an initializer cannot report that to the caller who
    /// will actually be affected by it.
    private func workspace(for scope: MemoryScope) async -> Workspace? {
        if let existing = workspaces[scope] {
            lastUsed[scope] = Date()
            return existing
        }
        guard configuration.isEnabled else { return nil }

        if let injectedStore {
            let workspace = Workspace(
                store: injectedStore, journal: injectedJournal,
                engine: nil, persists: true)
            workspaces[scope] = workspace
            return workspace
        }

        let (engine, persists) = Self.makeEngine(
            configuration: configuration,
            scope: scope, log: log)
        do {
            try await engine.start()
        } catch {
            isDegraded = true
            log(.degraded(operation: "start", detail: "\(error)"))
        }
        let store = ContinuityStore(engine: engine, limits: configuration.limits)
        var journal: (any SessionJournal)?
        if let injectedJournal {
            journal = injectedJournal
        } else if configuration.journalEnabled {
            journal = ContinuityJournalStore(
                engine: engine, store: store,
                limits: configuration.journalLimits)
        }
        let workspace = Workspace(
            store: store, journal: journal, engine: engine,
            persists: persists)
        workspaces[scope] = workspace
        lastUsed[scope] = Date()
        await enforceResidencyBudget(keeping: scope)
        // A new project file may be the one that pushes the count past the
        // cap; the sweep skips every workspace this process holds open.
        await sweepStaleWorkspaces()
        return workspace
    }

    /// Keeps the whole subsystem inside the ceiling, when one is set.
    ///
    /// A ceiling is a total, not a per-workspace allowance: per-workspace
    /// limits alone would multiply it by the number of workspaces a session
    /// has touched. With no ceiling, the default, this does nothing.
    ///
    /// Over the ceiling, the least recently used workspace is closed. Nothing
    /// is lost: everything it held is in its journal, and touching that
    /// workspace again replays it. The workspace in use is never closed.
    private func enforceResidencyBudget(keeping scope: MemoryScope) async {
        guard let ceiling = configuration.storage.maximumMemoryBytes, ceiling > 0 else { return }
        while workspaces.count > 1, await residentBytes() > ceiling {
            let candidates =
                lastUsed
                .filter { $0.key != scope && workspaces[$0.key] != nil }
                .sorted { $0.value < $1.value }
            guard let oldest = candidates.first?.key else { return }
            await workspaces[oldest]?.engine?.shutDown()
            workspaces[oldest] = nil
            lastUsed[oldest] = nil
            // Reopening replays the file into a fresh engine, so a failure
            // there later is a new one and worth its own line.
            reportedJournalFailures.remove(oldest)
            log(
                .degraded(
                    operation: "residency",
                    detail: "closed workspace \(oldest.workspace) to stay inside "
                        + "\(ceiling >> 20) MiB"))
        }
    }

    /// Bytes memory is holding in this process, across every open workspace.
    public func residentBytes() async -> Int {
        var total = 0
        for workspace in workspaces.values {
            total += await workspace.engine?.residentBytes() ?? 0
        }
        return total
    }

    /// Builds the engine, with a journal file when one can be opened.
    ///
    /// A directory that cannot be written is not fatal: the engine still runs
    /// in memory for the session. It is logged, and `isDurable` reports false,
    /// so the prompt tells the model its writes will not outlive the session
    /// rather than letting it assume they will.
    ///
    /// - Returns: the engine, and whether it is actually writing to a file.
    ///   The flag is not cosmetic: without it a session whose journal could
    ///   not be opened would tell the model its writes persist, which is the
    ///   one thing memory must never get wrong.
    private static func makeEngine(
        configuration: MemoryConfiguration,
        scope: MemoryScope,
        log: @Sendable (MemoryLogEvent) -> Void
    ) -> (engine: ContinuityEngine, persists: Bool) {
        let budget = configuration.storage.budget
        let limits = ContinuityCore.MemoryLimits(
            maxValueBytes: configuration.limits.maximumValueBytes,
            maxBytesPerTask: budget.factBytes)
        let engineConfiguration = ContinuityConfiguration(
            memoryLimits: limits,
            sessionLogOptions: SessionLogOptions(maxBytesPerTask: budget.logBytes),
            journalsSessionContent: configuration.journalEnabled)
        do {
            let journal = try FileJournal(
                url: configuration.storage.journalURL(for: scope),
                synchronizesEveryWrite: configuration.storage.synchronizesEveryWrite)
            return (ContinuityEngine(configuration: engineConfiguration, journal: journal), true)
        } catch {
            // A journal held by another server is the expected case here, not
            // a broken install. Either way the session runs without
            // persistence and says so rather than writing into a file someone
            // else is also writing.
            log(.degraded(operation: "openJournal", detail: "\(error)"))
            return (ContinuityEngine(configuration: engineConfiguration), false)
        }
    }

    /// Records a completed turn. Content is filtered to substance here, so no
    /// caller can accidentally journal a tool result or a file dump.
    public func recordTurn(
        session: MemorySessionContext,
        index: Int,
        prompt: String,
        reply: String,
        model: String?,
        promptTokens: Int,
        completionTokens: Int,
        latencyMilliseconds: Int,
        stopReason: String?
    ) async {
        guard let journal = await workspace(for: session.scope)?.journal else { return }
        let filteredPrompt = journalFilter.filter(prompt)
        let filteredReply = journalFilter.filter(reply)
        let turn = JournalTurn(
            session: session.session.id,
            workspace: session.scope.workspace,
            index: index,
            prompt: filteredPrompt.kept,
            reply: filteredReply.kept,
            model: model,
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            latencyMilliseconds: latencyMilliseconds,
            stopReason: stopReason,
            droppedBytes: filteredPrompt.dropped + filteredReply.dropped)
        await journal.record(turn, in: session.scope)
        // Never fails the turn: the reply has already been given. A journal
        // that refused it stops the workspace reporting itself durable.
        _ = await journalFailed(in: session.scope)
        log(.journaled(session: session.session.id, index: index, bytes: turn.byteCount))
        await enforceResidencyBudget(keeping: session.scope)
    }

    /// Open the configured workspace now, so its journal is replayed at boot
    /// rather than on the first request.
    ///
    /// Replay is the one bulk read the store ever does. Paying it at start,
    /// while nothing is being generated, keeps it off the same disk the
    /// expert streamer is about to saturate and off the first user's
    /// latency. Safe to call more than once and safe with memory disabled.
    public func warmUp() async {
        await sweepStaleWorkspaces()
        guard let scope = configuration.scope() else { return }
        _ = await workspace(for: scope)
    }

    /// Keeps project files from piling up, without losing what they know.
    ///
    /// One journal per project means one per directory a client ever ran
    /// from, and nothing else removes them. Two rules. A file untouched for
    /// `retentionDays` has its session log expired -- the transcript, which
    /// is the bulk of it -- and keeps its facts, because a novel paused for
    /// six weeks must not come back without its bible. Beyond
    /// `maximumWorkspaces`, the oldest by last write are deleted outright;
    /// that cap is the only thing that removes facts. A workspace this
    /// process has open is never touched -- its lock is held, and it was
    /// written moments ago in any case. Runs at start and whenever a new
    /// project file is created.
    public func sweepStaleWorkspaces(now: Date = Date()) async {
        let storage = configuration.storage
        guard storage.retentionDays > 0 || storage.maximumWorkspaces > 0 else { return }
        let manager = FileManager.default
        let open = Set(workspaces.keys.map { storage.journalURL(for: $0).standardizedFileURL.path })
        let candidates = Self.projectFiles(under: storage.directory).filter {
            !open.contains($0.url.standardizedFileURL.path)
                && $0.url.deletingPathExtension().lastPathComponent
                    != MemoryConfiguration.sharedWorkspace
        }
        // The cap deletes; it is the only rule that removes facts.
        var doomed: [URL] = []
        if storage.maximumWorkspaces > 0 {
            let ordered = candidates.sorted { $0.modified > $1.modified }
            // The open workspaces count against the cap too.
            let keep = max(0, storage.maximumWorkspaces - open.count)
            doomed = ordered.dropFirst(keep).map(\.url)
        }
        if !doomed.isEmpty {
            var removed: [String] = []
            for url in doomed {
                // Never remove a workspace another process is inside. The cap is
                // the one rule here that deletes facts, and `open` only knows
                // *this* process's workspaces.
                guard !Self.isLockHeld(at: url) else { continue }
                try? manager.removeItem(at: url)
                try? manager.removeItem(at: url.appendingPathExtension("lock"))
                try? manager.removeItem(at: url.appendingPathExtension("compacting"))
                removed.append(url.lastPathComponent)
            }
            log(.swept(removed: removed, reason: "more than \(storage.maximumWorkspaces) projects"))
        }
        // Retention expires the session log and keeps the facts.
        guard storage.retentionDays > 0 else { return }
        let cutoff = now.addingTimeInterval(-Double(storage.retentionDays) * 86_400)
        let deleted = Set(doomed.map(\.path))
        var expired: [String] = []
        for candidate in candidates
        where candidate.modified < cutoff && !deleted.contains(candidate.url.path) {
            if await Self.expireSessionLog(at: candidate.url) {
                expired.append(candidate.url.lastPathComponent)
            }
        }
        if !expired.isEmpty { log(.expired(files: expired)) }
    }

    /// Whether another process holds this project's workspace lock.
    ///
    /// `FileJournal` takes `flock(LOCK_EX | LOCK_NB)` on `<journal>.lock`, so the
    /// probe is the same call: it succeeds only when nobody holds the workspace.
    /// Deleting a file another process is appending to takes its `.lock` with it
    /// -- the file that process's `flock` is attached to -- and leaves it
    /// writing to a deleted inode, which its next compaction then rewrites into
    /// nothing. The retention pass already behaves this way by opening the file
    /// through the ordinary engine; this is the same rule for the cap.
    ///
    /// Everything unreadable, and every unexpected `flock` failure, counts as
    /// held: skipping a deletable file costs disk, deleting a live one costs
    /// data.
    static func isLockHeld(at journalURL: URL) -> Bool {
        let descriptor = open(
            journalURL.appendingPathExtension("lock").path,
            O_RDWR | O_CLOEXEC)
        if descriptor < 0 {
            // No lock file at all means no journal has opened this workspace, so
            // there is nothing that could be holding it. Any other errno is not
            // ours to read in favour of deleting.
            return errno != ENOENT
        }
        defer { close(descriptor) }
        if flock(descriptor, LOCK_EX | LOCK_NB) == 0 {
            _ = flock(descriptor, LOCK_UN)
            return false
        }
        return true
    }

    /// Every project file under the directory with its last-write time.
    /// Synchronous on purpose: a directory enumerator cannot be iterated
    /// from an async context.
    private static func projectFiles(under directory: URL) -> [(url: URL, modified: Date)] {
        guard
            let walker = FileManager.default.enumerator(
                at: directory, includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles])
        else { return [] }
        var files: [(url: URL, modified: Date)] = []
        for case let url as URL in walker where url.pathExtension == "ndjson" {
            let modified =
                (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? .distantPast
            files.append((url, modified))
        }
        return files
    }

    /// Rewrites a project file as a checkpoint of its facts alone.
    ///
    /// Opens it with the ordinary engine, which takes the workspace lock, so
    /// a file another server holds is left alone. Every session is pruned
    /// and the journal compacted; the facts, their history and the task
    /// survive. Returns false when the file could not be opened.
    private static func expireSessionLog(at url: URL) async -> Bool {
        guard let journal = try? FileJournal(url: url) else { return false }
        let engine = ContinuityEngine(journal: journal)
        do { try await engine.start() } catch {
            await engine.shutDown()
            return false
        }
        for task in await engine.tasks() {
            await engine.pruneSessions(taskID: task.id, keeping: 0)
        }
        try? await engine.compactJournal()
        await engine.shutDown()
        return true
    }

    /// Close every workspace, flushing and releasing the workspace locks.
    ///
    /// A workspace's journal holds an exclusive lock for as long as it is
    /// open, so a process that is finished with a workspace has to say so.
    /// Leaving it to deallocation would make the moment another server can
    /// take over depend on when ARC happens to release an actor.
    public func shutDown() async {
        // The background sweep goes first: it drives the same engine, and a
        // judgement must not be in flight while the weights are released.
        await retrievalHinter?.shutdown()
        for workspace in workspaces.values {
            await workspace.engine?.shutDown()
        }
        workspaces.removeAll()
        lastUsed.removeAll()
        reportedJournalFailures.removeAll()
        // The side-engine is a second resident model, so it is released on the
        // same shutdown that releases the stores rather than at process exit.
        await sideEngine?.shutdown()
    }

    /// The journal, for a caller that wants to read it back. Never used to
    /// build a prompt.
    public func journalStore(for scope: MemoryScope? = nil) async -> (any SessionJournal)? {
        guard let resolved = scope ?? configuration.scope() else { return nil }
        return await workspace(for: resolved)?.journal
    }

    public var isEnabled: Bool { configuration.isEnabled }

    /// Whether writes reach durable storage in a scope. False once a durable
    /// operation has failed, false when the journal could not be opened at
    /// all, and false once the journal has refused a write.
    public func isDurable(in scope: MemoryScope) async -> Bool {
        guard !isDegraded, let workspace = await workspace(for: scope),
            workspace.persists
        else { return false }
        return !(await journalFailed(in: scope))
    }

    /// Whether a workspace's journal has refused a write, logged the first
    /// time it is seen.
    ///
    /// Read from the engine rather than inferred from a tool result: a turn's
    /// prompt and reply are journaled on a path where no caller sees the
    /// write fail.
    private func journalFailed(in scope: MemoryScope) async -> Bool {
        guard let store = workspaces[scope]?.store as? ContinuityStore,
            let failure = await store.journalFailure
        else { return false }
        if reportedJournalFailures.insert(scope).inserted {
            log(.degraded(operation: "journal", detail: failure))
        }
        return true
    }

    /// Whether the configuration's own scope is persisting.
    public var isDurable: Bool {
        get async {
            guard let scope = configuration.scope() else { return false }
            return await isDurable(in: scope)
        }
    }

    /// Starts a session and returns what the engine needs to install.
    ///
    /// A failure here degrades rather than propagates: the session continues
    /// with local memory when that is allowed, and with none when it is not.
    /// - Parameter tag: what the session is about, when the caller could
    ///   tell. Recorded on the session, shown in the log; not a scope.
    public func beginSession(
        id: String,
        workspaceOverride: String? = nil,
        modelID: String? = nil,
        tag: String? = nil,
        focus: String? = nil
    ) async -> MemorySessionContext? {
        guard configuration.isEnabled else { return nil }
        guard let scope = configuration.scope(workspaceOverride: workspaceOverride) else {
            log(.rejectedScope(workspaceOverride ?? configuration.workspace))
            return nil
        }
        let session = MemorySession(id: id, modelID: modelID, tag: tag, focus: focus)
        var bootstrap = MemoryBootstrap.empty
        if let workspace = await workspace(for: scope) {
            do {
                bootstrap = try await workspace.store.sessionInit(session, in: scope)
                isDegraded = false
            } catch {
                isDegraded = true
                log(.degraded(operation: "sessionInit", detail: "\(error)"))
                guard configuration.degradesToLocalStore else { return nil }
                bootstrap = (try? await localStore.sessionInit(session, in: scope)) ?? .empty
            }
        } else {
            bootstrap = (try? await localStore.sessionInit(session, in: scope)) ?? .empty
        }
        // The person's own facts, from the shared workspace, ride along on
        // every project's bootstrap. Bounded small: they are preferences,
        // not state, and there should be a handful.
        if let sharedScope = configuration.sharedScope, scope != sharedScope,
            await workspace(for: sharedScope) != nil
        {
            let shared = await recordedFacts(in: sharedScope, limit: 12)
            if !shared.isEmpty { bootstrap = bootstrap.withShared(shared) }
        }
        let durable = await isDurable(in: scope)
        log(
            .sessionStarted(
                session: session.id, scope: scope,
                bootstrapRecords: bootstrap.records.count,
                bootstrapBytes: bootstrap.totalBytes))
        return MemorySessionContext(
            session: session,
            scope: scope,
            bootstrap: bootstrap,
            isDurable: durable)
    }

    /// The system-prompt fragment for a session.
    public func instructions(for context: MemorySessionContext) -> String {
        MemoryPrompt.instructions(
            scope: context.scope,
            session: context.session,
            bootstrap: context.bootstrap,
            isDurable: context.isDurable,
            tools: toolDefinitions().map(\.name))
    }

    /// The tool definitions to advertise, or none when tools are off.
    public func toolDefinitions() -> [MemoryToolDefinition] {
        guard configuration.isEnabled else { return [] }
        return MemoryTools.definitions(surface: configuration.toolSurface)
    }

    /// Runs one memory tool call in a session's scope.
    ///
    /// The scope comes from the session context, never from the call, so a
    /// model cannot reach another workspace by naming one.
    public func execute(
        name: String,
        arguments: [String: MemoryToolValue],
        in context: MemorySessionContext
    ) async -> MemoryToolResult {
        guard configuration.isEnabled else { return .failure("memory is disabled") }
        let store = await activeStore(for: context.scope)
        let (hint, onSearch) = await retrievalContext(
            name: name, arguments: arguments,
            store: store, scope: context.scope)
        let result = await MemoryTools.execute(
            name: name,
            arguments: arguments,
            store: store,
            scope: context.scope,
            session: context.session,
            limits: configuration.limits,
            guarding: configuration.guardsUserFacts,
            retrievalHint: hint,
            onSearch: onSearch)
        // Checked whatever the outcome: a call whose own write landed can
        // still have had a session event refused.
        let journalLost = await journalFailed(in: context.scope)
        if case .failure(let message) = result {
            log(.toolFailed(tool: name, detail: message))
            // A durable backend that failed sends later work to the local
            // store, and marks the session as no longer persisting. A failed
            // journal does not: the engine still holds every fact, and a
            // local retry would answer "stored" for a write that ends with
            // the process.
            if !isDegraded, !journalLost,
                message.contains("unavailable")
                    || message.contains("timed out")
            {
                isDegraded = true
                log(.degraded(operation: name, detail: message))
                if configuration.degradesToLocalStore {
                    return await MemoryTools.execute(
                        name: name,
                        arguments: arguments,
                        store: localStore,
                        scope: context.scope,
                        session: context.session,
                        limits: configuration.limits,
                        guarding: configuration.guardsUserFacts,
                        retrievalHint: hint,
                        onSearch: onSearch)
                }
            }
        } else {
            log(.toolSucceeded(tool: name))
            // Checked after the write, not only when a workspace is opened.
            // A ceiling that only holds while the set of workspaces is
            // changing is not a ceiling.
            await enforceResidencyBudget(keeping: context.scope)
        }
        return result
    }

    /// Ends a session. With consolidation off this only logs; the hook for
    /// asking the model what to keep lives in the engine, which owns
    /// generation.
    public func endSession(_ context: MemorySessionContext) async {
        log(.sessionEnded(session: context.session.id, scope: context.scope))
    }

    /// The ranking hint this call may read, and the closure that schedules the
    /// background pass the next search will read from.
    ///
    /// The closure queues the question on the hinter before the search returns,
    /// so the question is registered by the time the answer is; it still never
    /// waits on a judgement, because `register` only enqueues. That ordering is
    /// why an observer that awaits the hint right after a search cannot race the
    /// registration. Both are empty unless this really is a text search and
    /// a side-engine is wired, which is what leaves the deterministic path
    /// byte-for-byte what it was.
    private func retrievalContext(
        name: String,
        arguments: [String: MemoryToolValue],
        store: any MemoryStore,
        scope: MemoryScope
    )
        async -> (MemoryRetrievalHint, (@Sendable (MemoryQuery) async -> Void)?)
    {
        guard name == "memory_search", let hinter = retrievalHinter,
            let text = arguments["query"]?.stringValue,
            !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return (.none, nil)
        }
        let hint = await hinter.hint(question: text, in: scope)
        let schedule: @Sendable (MemoryQuery) async -> Void = { query in
            guard let asked = query.text,
                !asked.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                return
            }
            await hinter.register(question: asked, in: scope, store: store)
        }
        return (hint, schedule)
    }

    /// Awaits the background T7 sweep, so a test can make a hint observable.
    /// Nothing on the request path calls this.
    func waitForRetrievalHints() async {
        await retrievalHinter?.waitForBackgroundWork()
    }

    /// Facts already in a scope, most important first, so a consolidation
    /// can update an address instead of inventing a near-duplicate beside
    /// it -- and can see the value it would be replacing.
    ///
    /// Values, not only keys. Shown keys alone, a model re-derived every one
    /// of them from a session that said nothing about them, and wrote "not
    /// specified" over a character's eye colour.
    public func recordedFacts(in scope: MemoryScope, limit: Int = 60) async -> [MemoryRecord] {
        let store = await activeStore(for: scope)
        return (try? await store.search(MemoryQuery(limit: limit), in: scope)) ?? []
    }

    /// Stores a consolidation the engine produced at session end.
    public func storeConsolidation(
        _ records: [MemoryRecord],
        in context: MemorySessionContext
    ) async -> Int {
        let store = await activeStore(for: context.scope)
        var written = 0
        var held = 0
        var unchanged = 0
        var duplicates = 0
        var conflicts = 0
        var dropped = 0
        var ruleConflicts = 0
        // Read once, and only when an engine is wired: the deterministic path
        // pays nothing for the check it cannot make.
        let candidates: [MemoryRecord]
        if sideEngine != nil {
            candidates =
                (try? await store.search(
                    MemoryQuery(limit: 400),
                    in: context.scope)) ?? []
        } else {
            candidates = []
        }
        // The shared workspace's facts, read the first time one is needed.
        var sharedCandidates: [MemoryRecord]?
        var questionsLeft = Self.maximumSideEngineQuestions
        for record in records {
            // A fact about the person rather than the project goes to the
            // shared workspace, where every project's bootstrap reads it. Its
            // conventions and preferences are the most user-asserted category
            // in the store, so the guard reaches there too.
            let scope: MemoryScope
            let destination: any MemoryStore
            let isShared: Bool
            if record.isGlobal, let sharedScope = configuration.sharedScope,
                context.scope != sharedScope
            {
                scope = sharedScope
                destination = await activeStore(for: sharedScope)
                isShared = true
            } else {
                scope = context.scope
                destination = store
                isShared = false
            }
            // The extraction is told to write only what changed and still
            // restates unchanged facts: every eye colour in a novel got a v2
            // and a v3 with the identical value. A write that changes nothing
            // is version churn and completion tokens for no fact.
            let current = try? await destination.get(record.key, in: scope)
            if let current, Self.fold(current.value) == Self.fold(record.value) {
                unchanged += 1
                continue
            }
            // Read once per scope, and only with an engine wired: the
            // deterministic path pays nothing for a check it cannot make.
            var pool = candidates
            if isShared, sideEngine != nil {
                if sharedCandidates == nil {
                    sharedCandidates =
                        (try? await destination.search(
                            MemoryQuery(limit: 400), in: scope)) ?? []
                }
                pool = sharedCandidates ?? []
            }
            // T2, T4, T5 and T3 over one budget for the whole consolidation.
            // Durability comes first, then the rule that fixes a value the
            // same key already holds, then the comparison against other keys.
            // The contradiction is advisory; the rule conflict is not, because
            // a rule says the new value cannot be right.
            if let sideEngine, questionsLeft > 0 {
                let outcome = await inspectForWrite(
                    record, current: current, pool: pool, using: sideEngine,
                    budget: min(questionsLeft, Self.maximumQuestionsPerFact))
                questionsLeft -= outcome.asked
                switch outcome.inspection {
                case .dropped:
                    log(.notDurableStopped(key: record.key.rawValue))
                    dropped += 1
                    continue
                case .ruleConflict:
                    // The stored rule fixes this value and the rule wins: the
                    // old value stays and the change is not written.
                    log(.ruleConflictStopped(key: record.key.rawValue))
                    ruleConflicts += 1
                    continue
                case .duplicate(let kept):
                    log(.nearDuplicateStopped(key: record.key.rawValue, kept: kept))
                    duplicates += 1
                    continue
                case .conflict(let conflictsWith):
                    log(
                        .contradictionFound(
                            key: record.key.rawValue,
                            conflictsWith: conflictsWith))
                    conflicts += 1
                case .none:
                    break
                }
            }
            switch await write(
                record, to: destination, scope: scope,
                session: context.session.id,
                flaggingReversions: !isShared)
            {
            case .stored:
                written += 1
                if isShared { log(.sharedFactWritten(key: record.key.rawValue)) }
            case .reverted:
                written += 1
                log(.reversionFlagged(key: record.key.rawValue))
            case .held:
                // Not written: the person said otherwise and the model did
                // not. The address is disputed, so the next session is shown
                // both rather than one of them. No value is logged, ever.
                log(.guardHeld(key: record.key.rawValue))
                held += 1
            case .failed:
                break
            }
        }
        logConsolidationSummary(
            session: context.session.id, written: written,
            unchanged: unchanged, duplicates: duplicates,
            dropped: dropped, ruleConflicts: ruleConflicts,
            conflicts: conflicts)
        return written
    }

    /// One line per counter that fired, then the total.
    private func logConsolidationSummary(
        session: String, written: Int, unchanged: Int,
        duplicates: Int, dropped: Int,
        ruleConflicts: Int, conflicts: Int
    ) {
        if unchanged > 0 { log(.unchangedSkipped(session: session, count: unchanged)) }
        if duplicates > 0 { log(.nearDuplicatesStopped(session: session, count: duplicates)) }
        if dropped > 0 { log(.notDurablesStopped(session: session, count: dropped)) }
        if ruleConflicts > 0 {
            log(.ruleConflictsStopped(session: session, count: ruleConflicts))
        }
        if conflicts > 0 { log(.contradictionsFound(session: session, count: conflicts)) }
        log(.consolidated(session: session, records: written))
    }

    /// Where one record ended up, so the caller keeps the counting and the
    /// logging in one place instead of at each destination.
    enum WriteOutcome {
        case stored
        case reverted
        case held
        case failed
    }

    /// One fact into one store.
    ///
    /// `flaggingReversions` is the consolidation heuristic, not the protocol's
    /// rule: it is on for the project's own store and off for the shared
    /// workspace and for every deliberate tool call.
    private func write(
        _ record: MemoryRecord,
        to store: any MemoryStore,
        scope: MemoryScope,
        session: String,
        flaggingReversions: Bool
    ) async -> WriteOutcome {
        var stamped = record
        stamped.sourceSession = session
        do {
            if let continuity = store as? ContinuityStore {
                switch try await continuity.set(
                    stamped, in: scope,
                    guarding: configuration.guardsUserFacts,
                    flaggingReversions: flaggingReversions)
                {
                case .stored: return .stored
                case .reverted: return .reverted
                case .heldByGuard: return .held
                }
            }
            // Degraded to process-local storage: there is no provenance to
            // enforce precedence with, and the protocol says so rather than
            // pretending.
            _ = try await store.set(
                stamped, in: scope,
                guarding: configuration.guardsUserFacts)
            return .stored
        } catch {
            log(.toolFailed(tool: "consolidation", detail: "\(error)"))
            return .failed
        }
    }

    /// What the side-engine said about one new fact, and what it cost.
    struct SideEngineVerdict {
        /// `false` means the fact is not worth keeping; `nil` is no answer.
        let durable: Bool?
        /// `.conflict` means a stored rule fixes this value; `nil` is no
        /// answer, and `.update` leaves the write alone.
        let supersession: MemorySupersession?
        let duplicate: String?
        let conflict: String?
        let asked: Int
    }

    /// T2, T4, T5 and T3 over one new fact, stopping as soon as the budget is
    /// gone.
    ///
    /// Durability comes first and ends the check when the answer is no — a fact
    /// that is not worth keeping needs no comparison. The rule check needs both
    /// the stored value (`current`) and a rule the caller found. Candidates are
    /// facts in the same leading segment with a different key; the duplicate is
    /// looked for before the contradiction, and finding one ends the search
    /// because there is nothing to add about a fact already stored.
    ///
    /// `isModelDerived` is false for a fact the person asserted, which is not
    /// the engine's to discard or to hold back behind a rule.
    private func inspect(
        _ record: MemoryRecord,
        current: MemoryRecord?,
        rule: String?,
        among candidates: [MemoryRecord],
        using engine: any MemorySideEngine,
        budget: Int,
        isModelDerived: Bool
    ) async -> SideEngineVerdict {
        let fact = MemoryFact(key: record.key.rawValue, value: record.value)
        var asked = 0
        var durable: Bool?
        if isModelDerived, budget > 0 {
            asked += 1
            durable = await engine.isDurable(fact)
            if durable == false {
                return SideEngineVerdict(
                    durable: durable, supersession: nil,
                    duplicate: nil, conflict: nil, asked: asked)
            }
        }
        var supersession: MemorySupersession?
        if isModelDerived, let current, let rule, asked < budget {
            asked += 1
            supersession = await engine.supersedes(
                MemoryFact(key: current.key.rawValue, value: current.value),
                fact, rule: rule)
        }
        var conflict: String?
        for candidate in candidates {
            guard candidate.key != record.key,
                candidate.key.category == record.key.category
            else { continue }
            guard asked < budget else { break }
            let existing = MemoryFact(
                key: candidate.key.rawValue,
                value: candidate.value)
            asked += 1
            // Stored first: the prompts answer YES in that order and NO
            // reversed, so the order is part of the contract.
            if await engine.duplicates(existing, fact) == true {
                return SideEngineVerdict(
                    durable: durable, supersession: supersession,
                    duplicate: candidate.key.rawValue,
                    conflict: nil, asked: asked)
            }
            guard asked < budget else { break }
            asked += 1
            if conflict == nil, await engine.contradicts(existing, fact) == true {
                conflict = candidate.key.rawValue
            }
        }
        return SideEngineVerdict(
            durable: durable, supersession: supersession,
            duplicate: nil, conflict: conflict, asked: asked)
    }

    /// What the engine's answers mean for this write.
    private enum Inspection {
        case none
        case dropped
        case ruleConflict
        case duplicate(String)
        case conflict(String)
    }

    /// Runs the questions and reduces them to one outcome, so the write path
    /// reads as one decision rather than four.
    private func inspectForWrite(
        _ record: MemoryRecord,
        current: MemoryRecord?,
        pool: [MemoryRecord],
        using engine: any MemorySideEngine,
        budget: Int
    ) async -> (inspection: Inspection, asked: Int) {
        let rule = MemoryRuleLookup.rule(for: record.key, among: pool)
        let verdict = await inspect(
            record, current: current, rule: rule, among: pool,
            using: engine, budget: budget,
            isModelDerived: !record.isUserAsserted)
        let inspection: Inspection
        if verdict.durable == false {
            inspection = .dropped
        } else if verdict.supersession == .conflict {
            inspection = .ruleConflict
        } else if let kept = verdict.duplicate {
            inspection = .duplicate(kept)
        } else if let conflictsWith = verdict.conflict {
            inspection = .conflict(conflictsWith)
        } else {
            inspection = .none
        }
        return (inspection, verdict.asked)
    }

    static func fold(_ value: String) -> String {
        value.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: ".\"'"))
    }

    private func activeStore(for scope: MemoryScope) async -> any MemoryStore {
        guard !isDegraded, let workspace = await workspace(for: scope) else { return localStore }
        return workspace.store
    }
}

/// Everything a session needs to carry once memory has started.
public struct MemorySessionContext: Sendable, Equatable {
    public let session: MemorySession
    public let scope: MemoryScope
    public let bootstrap: MemoryBootstrap
    /// False when the session is running on the local fallback, which the
    /// prompt tells the model so it does not promise persistence.
    public let isDurable: Bool

    public init(
        session: MemorySession,
        scope: MemoryScope,
        bootstrap: MemoryBootstrap,
        isDurable: Bool
    ) {
        self.session = session
        self.scope = scope
        self.bootstrap = bootstrap
        self.isDurable = isDurable
    }
}

/// Observable memory events. The engine maps these onto its own log; keeping
/// them as values means this module prints nothing itself and stays testable.
public enum MemoryLogEvent: Sendable, Equatable {
    case sessionStarted(
        session: String, scope: MemoryScope, bootstrapRecords: Int,
        bootstrapBytes: Int)
    case sessionEnded(session: String, scope: MemoryScope)
    case toolSucceeded(tool: String)
    case toolFailed(tool: String, detail: String)
    case degraded(operation: String, detail: String)
    case rejectedScope(String)
    case consolidated(session: String, records: Int)
    case journaled(session: String, index: Int, bytes: Int)
    /// Project files removed by retention, and why.
    case swept(removed: [String], reason: String)
    /// A consolidation wrote a value the key had before; it is now disputed.
    case reversionFlagged(key: String)
    /// The guard refused a model-derived write over what the person asserted.
    /// The person's value stays active and the address is disputed.
    case guardHeld(key: String)
    /// Facts a consolidation returned that already held the same value.
    case unchangedSkipped(session: String, count: Int)
    /// A fact the side-engine judged not worth keeping. It is not stored at
    /// all, and only the key is logged.
    case notDurableStopped(key: String)
    /// How many facts the side-engine's durability check dropped from one
    /// consolidation.
    case notDurablesStopped(session: String, count: Int)
    /// A change to a value a stored rule fixes. The old value stays and the
    /// change is not written; only the key is logged, never the rule or either
    /// value.
    case ruleConflictStopped(key: String)
    /// How many rule conflicts stopped one consolidation's writes.
    case ruleConflictsStopped(session: String, count: Int)
    /// A new key whose content an existing key already carried. The store
    /// keeps one address instead of two, and both keys are named — never a
    /// value.
    case nearDuplicateStopped(key: String, kept: String)
    /// How many facts the side-engine's near-duplicate check stopped in one
    /// consolidation.
    case nearDuplicatesStopped(session: String, count: Int)
    /// A new key the side-engine judged unable to be true at the same time as
    /// an existing one. Advisory: the write is not changed, because
    /// disagreement is not supersession and T4, which would tell them apart,
    /// is not ready.
    case contradictionFound(key: String, conflictsWith: String)
    /// How many possible contradictions one consolidation recorded.
    case contradictionsFound(session: String, count: Int)
    /// A consolidation wrote a fact about the person to the shared workspace.
    case sharedFactWritten(key: String)
    /// One background T7 sweep finished. Neither the question nor a value is
    /// logged — a question is the person's words and memory holds anything the
    /// model wrote — only how much work was done.
    case retrievalHints(judged: Int, answered: Int)
    /// Project files whose session log was expired by retention; facts kept.
    case expired(files: [String])

    /// One log line. Never contains a memory's contents or a credential: the
    /// log is operational, and memory can hold anything the model wrote.
    public var message: String {
        switch self {
        case .sessionStarted(let session, let scope, let records, let bytes):
            return "memory session=\(session) scope=\(scope.namespace)/\(scope.user)/"
                + "\(scope.workspace) bootstrap=\(records) records \(bytes)B"
        case .sessionEnded(let session, _):
            return "memory session=\(session) ended"
        case .toolSucceeded(let tool):
            return "memory tool=\(tool) ok"
        case .toolFailed(let tool, let detail):
            return "memory tool=\(tool) failed: \(detail)"
        case .degraded(let operation, let detail):
            return "memory degraded during \(operation): \(detail)"
        case .rejectedScope(let workspace):
            return "memory disabled for this session: unusable workspace '\(workspace)'"
        case .consolidated(let session, let records):
            return "memory session=\(session) consolidated \(records) records"
        case .journaled(let session, let index, let bytes):
            return "journal session=\(session) turn=\(index) \(bytes)B"
        case .reversionFlagged(let key):
            return "memory reversion flagged as disputed: \(key)"
        case .guardHeld(let key):
            return "memory guard kept the user's fact, marked disputed: \(key)"
        case .sharedFactWritten(let key):
            return "memory shared fact written for every project: \(key)"
        case .retrievalHints(let judged, let answered):
            return "memory retrieval hints: judged \(judged) fact(s), "
                + "\(answered) could answer"
        case .unchangedSkipped(let session, let count):
            return "memory session=\(session) consolidation skipped \(count) unchanged fact(s)"
        case .nearDuplicateStopped(let key, let kept):
            return "memory near-duplicate stopped: \(key) is already \(kept)"
        case .notDurableStopped(let key):
            return "memory not worth keeping, not stored: \(key)"
        case .notDurablesStopped(let session, let count):
            return "memory session=\(session) consolidation dropped \(count) fact(s) not "
                + "worth keeping"
        case .ruleConflictStopped(let key):
            return "memory rule conflict, change not stored: \(key)"
        case .ruleConflictsStopped(let session, let count):
            return "memory session=\(session) consolidation stopped \(count) change(s) a "
                + "rule fixes"
        case .nearDuplicatesStopped(let session, let count):
            return "memory session=\(session) consolidation stopped \(count) near-duplicate(s)"
        case .contradictionFound(let key, let conflictsWith):
            return "memory possible conflict: \(key) may disagree with \(conflictsWith)"
        case .contradictionsFound(let session, let count):
            return "memory session=\(session) consolidation recorded \(count) possible conflict(s)"
        case .expired(let files):
            return "memory expired the session log of \(files.count) project file(s), facts kept: "
                + files.joined(separator: ", ")
        case .swept(let removed, let reason):
            return "memory swept \(removed.count) project file(s) (\(reason)): "
                + removed.joined(separator: ", ")
        }
    }
}
