import Foundation
import TinyTitan

public struct ServerArguments: Equatable, Sendable {
    public let model: String
    public let mtpModel: String?
    public let mtpMemoryMiB: Int
    public let port: Int
    /// Explicit --model-id value; nil derives the API ID from the installed
    /// manifest (for example qwen3.6-35b-a3b or ornith-1.5-35b-a3b).
    public let modelIDOverride: String?
    public let maxContext: Int
    public let queueLimit: Int
    /// Generations that may run at once through the batched engine. One is the
    /// historical single-generation server and the default; the excess still
    /// queues. Above one, each sequence holds its own KV cache and scratch, and
    /// the prompt cache is switched off (it is session-wide, so a prefix could
    /// otherwise be restored into the wrong sequence).
    public let maxConcurrentSequences: Int

    /// The width a session's runner and scratch may actually be built with.
    /// MTP is single-sequence, so it pins the width to one; every plan factory
    /// reads this rather than re-deriving it, so the single-model path and the
    /// catalog loader cannot disagree.
    public var sessionSlots: Int {
        mtpModel == nil ? maxConcurrentSequences : 1
    }
    public let promptCacheMode: ServerPromptCacheMode
    public let promptCacheMaximumEntries: Int
    public let promptCacheMemoryMiB: Int
    public let promptCacheDiskDirectory: String?
    public let promptCacheDiskMiB: Int
    /// Idle seconds before a RAM snapshot is released to its SSD copy; 0 = never.
    public let promptCacheMemoryTTLSeconds: Int
    public let prefillChunkTokens: Int?
    public let kvCachePrecision: KVCachePrecision
    public let ropeScalingMode: RuntimeRoPEScalingMode
    public let thinkingMode: ModelThinkingMode
    /// Reasoning-effort override for effort-aware model families; nil keeps
    /// the template default. Family support is validated at startup against
    /// the installed manifest.
    public let reasoningEffort: ModelReasoningEffort?
    /// Serve on the CPU instead of the GPU, from an affine snapshot rather
    /// than an install.
    ///
    /// For models small enough that a GPU is not the point: a 2B reads about
    /// 1.9 GB per token, which the CPU does at twenty tokens a second on
    /// four performance cores. It leaves the GPU entirely free, which is why
    /// the same engine can run a side model beside a 35B.
    public let cpu: Bool
    /// Keep the whole snapshot faulted into memory rather than leaving
    /// residency to the page cache. On by default with `--cpu`, because a
    /// model that gets evicted between requests pays to fault itself back in.
    public let cpuResident: Bool
    public let expertCacheSlots: Int?
    /// Bytes the routed-expert cache may use. Slots are derived from it and the
    /// model's own expert stride, so this is the knob and the slot count is the
    /// outcome. `--expert-cache-slots` still wins if both are given.
    public let expertCacheBudgetBytes: Int?
    /// Defer the model load to the first inference request.
    public let lazyLoad: Bool
    /// Release the weights after this many idle seconds; 0 disables unloading.
    public let idleUnloadSeconds: Int
    /// Serve every model under this directory, one resident at a time, with
    /// `model` naming the one loaded first. Nil keeps the single-model server
    /// exactly as it was, which the benchmark harness depends on.
    public let modelsDirectory: String?
    /// Print the catalog of `modelsDirectory` as JSON and exit.
    public let catalogOnly: Bool
    /// The explicit `--reasoning` level; nil when the older flags were used.
    public let reasoningLevel: ReasoningLevel?

    /// The server-wide level, however it was spelled: `--reasoning` wins,
    /// otherwise `--thinking` and `--reasoning-effort` say the same thing.
    public var requestedReasoningLevel: ReasoningLevel {
        if let reasoningLevel { return reasoningLevel }
        guard thinkingMode.isEnabled else { return .off }
        return reasoningEffort.flatMap { ReasoningLevel(rawValue: $0.rawValue) } ?? .on
    }

    /// Unloading implies deferring the first load — loading at boot only to
    /// drop it moments later is incoherent. Derived rather than folded into
    /// `lazyLoad` so the struct stays a faithful record of what was typed.
    public var managesResidency: Bool { lazyLoad || idleUnloadSeconds > 0 }

    /// Idle unloading discards the in-memory prefix cache with the session.
    /// With a disk cache configured the entries rehydrate on reload; without
    /// one, every unload costs a full cold prefill on the next request.
    public var unloadDiscardsWarmCache: Bool {
        idleUnloadSeconds > 0
            && promptCacheMode != .off
            && promptCacheDiskDirectory == nil
    }

    /// The accepted slot counts, spelled from the validator that enforces them.
    ///
    /// The help text used to list them by hand and had already drifted: it
    /// stopped at 128 while `RuntimeConfiguration.allowedExpertCacheSlots`
    /// accepts 40, 48, 112, 160, 192 and 256 as well. Spelled from the source
    /// of truth so it cannot drift again.
    static var expertCacheSlotsHelp: String {
        RuntimeConfiguration.allowedExpertCacheSlots
            .map(String.init).joined(separator: ", ")
    }

    public static let usage = """
        usage: TinyTitanServer --model <completed .gturbo directory> [options]
               TinyTitanServer --models-dir <dir> --model <id or dir> [options]
               TinyTitanServer --catalog --models-dir <dir>

          --model <dir>          Required model directory. With --models-dir, the
                                 model loaded first: a catalog id or a directory.
          --models-dir <dir>     Serve every model under dir -- GPU installs and
                                 CPU snapshots alike -- keeping one resident. A
                                 request naming another catalog model waits for
                                 in-flight generations, unloads the resident model
                                 and loads the named one. /v1/models lists them.
          --catalog              With --models-dir: print the catalog as JSON and
                                 exit without loading anything.
          --reasoning <level>    Server-wide reasoning level: off, on, minimal, low,
                                 medium, high, xhigh or max, applied to whichever
                                 model is loaded. A model without that level gets
                                 the closest it has: an effort on an on/off model
                                 is on, on for an effort model is its template's
                                 default effort (extra high for Qwen3.8), off is
                                 always off. Replaces --thinking and
                                 --reasoning-effort, which keep working.
          --mtp-model <dir>      Optional native Qwen/Ornith MTP sidecar directory.
          --mtp-memory-mib <MiB> Strict incremental MTP budget, 256...512
                                 (default 384).
          --port <1...65535>     Loopback port (default 8080).
          --model-id <id>        API model identifier (default derived from the
                                 installed model manifest).
          --max-context <tokens> Native: 4096...262144 (default 262144).
                                 With YaRN: 524288 or 1048576 (default 1048576).
          --rope-scaling <mode>  Context scaling: none or yarn (default none).
          --queue-limit <count>  Maximum queued requests (default 4).
          --max-concurrent-sequences <count>
                                 Generations served at once: a power of two from 1
                                 to 256 (default 1). Requests beyond this plus
                                 --queue-limit are shed with 429. Above 1 each
                                 sequence holds its own KV cache (so memory use
                                 rises) and answers take longer, because one GPU is
                                 shared; the prompt cache is off above 1. The width
                                 actually built is clamped to what memory allows,
                                 and the log says so when it is.
          --prompt-cache-mode <off|single-prefix|multi-prefix>
                                 Prompt KV reuse mode (default multi-prefix).
          --prompt-cache-entries <count>
                                 Maximum retained prefixes, 1...64 (default 4).
          --prompt-cache-memory-mib <MiB>
                                 RAM snapshot budget, 0...4096 (default 256).
          --prompt-cache-disk <dir>
                                 Optional persistent SSD cache directory.
          --prompt-cache-disk-mib <MiB>
                                 SSD snapshot budget, 0...65536 (default 8192).
          --prompt-cache-memory-ttl-seconds <n>
                                 Release a RAM snapshot after n idle seconds,
                                 keeping its SSD copy (0...86400, default 0,
                                 disabled). Needs --prompt-cache-disk; without
                                 one nothing expires. A later request restores
                                 it from SSD.
          --prefill-chunk <tokens>
                                 Prefill chunk size: 32, 64, 128, 256, 512,
                                 1024, 2048, 4096, 8192 or 16384 (default: the
                                 model's profile row, 4096 for the 35B-A3B text
                                 models and 16384 for Qwen3.8 4-bit, lowered to
                                 what --max-context allows). A larger chunk
                                 reads the streamed experts fewer times.
          --kv-bits <4|8|16>     KV-cache storage precision (default 8).
          --thinking <off|on>    Ornith/Qwen reasoning mode (default off, or
                                 TINYTITAN_THINKING_MODE). The model does not expose
                                 low/medium/high effort levels.
          --reasoning-effort <low|medium|xhigh>
                                 Reasoning-effort level (default unset, or
                                 TINYTITAN_REASONING_EFFORT). Requires --thinking on
                                 and a model family whose chat template defines
                                 effort levels (Qwen3.8-Flash-Next); Ornith 1.5
                                 and Qwen 3.6 reject it.
          --expert-cache-slots <count>
                                 Routed-expert cache slots per layer:
                                 \(ServerArguments.expertCacheSlotsHelp).
                                 The default is derived from the model
                                 profile's tuned budget, not fixed.
                                 Environment override:
                                 TINYTITAN_EXPERT_CACHE_SLOTS.
          --ram-budget <size>    Resident-memory target for the whole server, e.g.
                                 4G, 8G, 16G. Minimum 4G. The routed-expert cache
                                 gets what is left after the resident weights and
                                 the runtime (about 3.7G on a Qwen3.8 4-bit
                                 install), and the slot count is the largest
                                 supported rung that fits, so the process stays
                                 under the number given. Below 4G the target cannot
                                 be honoured at all -- the weights plus the 8-slot
                                 minimum cache are already about 4.7G -- so it is
                                 refused rather than silently overshot.
                                 With no flag the install's profile names the *cache*
                                 budget instead -- the measured optimum, which is not
                                 a process target. --expert-cache-slots overrides
                                 both. Expert reads bypass the page cache and have
                                 no fallback, so a smaller cache is markedly slower.
          --lazy-load            Bind the port immediately and defer the model load
                                 to the first inference request (default off).
          --idle-unload-seconds <n>
                                 Release the model weights after n seconds with no
                                 requests, 0...86400 (default 0, disabled). The
                                 next request reloads transparently. Implies
                                 --lazy-load. Pair with --prompt-cache-disk, since
                                 unloading discards the in-memory prefix cache.
          --cpu                  Serve on the CPU from an affine snapshot instead
                                 of the GPU from an install. --model then points at
                                 a snapshot directory. For models small enough that
                                 a GPU is not the point: a 2B runs at about twenty
                                 tokens a second on the performance cores and
                                 leaves the GPU entirely free.
          --no-cpu-resident      With --cpu, leave residency to the page cache
                                 instead of faulting the snapshot in at startup.
                                 Only worth it on a machine too small to hold the
                                 model, where the alternative is swapping.
          --help                 Show this help.
        """

    /// lint:allow-long a flag table: one `case` per option plus its
    /// validation. Splitting it into per-group parsers would hide the
    /// exhaustive switch that makes an unhandled flag a compile-visible gap.
    public static func parse(
        _ input: [String],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> ServerArguments {
        var model: String?
        var mtpModel: String?
        var mtpMemoryMiB = StreamingMTPMemoryPlan.defaultBudgetMiB
        var port = 8080
        var modelIDOverride: String?
        var maxContext = 262_144
        var maxContextWasSet = false
        var queueLimit = 4
        // One generation at a time unless the operator asks for more: the
        // batched path holds one KV cache and scratch buffer per sequence, so
        // the memory bill and the per-answer slowdown are opt-in, not a
        // surprise on a default launch.
        var maxConcurrentSequences = 1
        var promptCacheMode: ServerPromptCacheMode = .multiPrefix
        var promptCacheMaximumEntries = 4
        var promptCacheMemoryMiB = 256
        var promptCacheDiskDirectory: String?
        var promptCacheDiskMiB = 8_192
        var promptCacheMemoryTTLSeconds = 0
        var prefillChunkTokens: Int?
        var kvCachePrecision: KVCachePrecision = .int8
        var ropeScalingMode: RuntimeRoPEScalingMode = .none
        var thinkingMode = ModelThinkingMode.resolved(environment: environment)
        var reasoningEffort = ModelReasoningEffort.resolved(environment: environment)
        var expertCacheSlots: Int?
        var cpu = false
        var cpuResident = true
        var expertCacheBudgetBytes: Int?
        var lazyLoad = false
        var idleUnloadSeconds = 0
        var modelsDirectory: String?
        var catalogOnly = false
        var reasoningLevel: ReasoningLevel?
        // Tracked apart from the values, which the environment can also set:
        // only flags typed next to --reasoning conflict with it.
        var thinkingWasSet = false
        var effortWasSet = false
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            if flag == "--catalog" {
                catalogOnly = true
                index += 1
                continue
            }
            // Valueless flags are consumed before the "requires a value" guard
            // below; otherwise `--lazy-load --port 9999` would swallow --port
            // as this flag's value and then reject it as unknown.
            if flag == "--lazy-load" {
                lazyLoad = true
                index += 1
                continue
            }
            if flag == "--cpu" {
                cpu = true
                index += 1
                continue
            }
            if flag == "--no-cpu-resident" {
                cpuResident = false
                index += 1
                continue
            }
            guard index + 1 < input.count else {
                throw ServerArgumentError.invalid("\(flag) requires a value")
            }
            let value = input[index + 1]
            index += 2
            switch flag {
            case "--model":
                model = value
            case "--mtp-model":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--mtp-model must not be empty")
                }
                mtpModel = value
            case "--mtp-memory-mib":
                guard let parsed = Int(value),
                    StreamingMTPMemoryPlan.allowedBudgetMiB.contains(parsed)
                else {
                    throw ServerArgumentError.invalid(
                        "--mtp-memory-mib must be between 256 and 512")
                }
                mtpMemoryMiB = parsed
            case "--port":
                guard let parsed = Int(value), (1...65_535).contains(parsed) else {
                    throw ServerArgumentError.invalid("--port must be between 1 and 65535")
                }
                port = parsed
            case "--model-id":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--model-id must not be empty")
                }
                modelIDOverride = value
            case "--max-context":
                guard let parsed = Int(value),
                    (1...RuntimeConfiguration.maximumContextTokens).contains(parsed)
                else {
                    throw ServerArgumentError.invalid("--max-context is not supported")
                }
                maxContext = parsed
                maxContextWasSet = true
            case "--rope-scaling":
                guard let parsed = RuntimeRoPEScalingMode(rawValue: value) else {
                    throw ServerArgumentError.invalid("--rope-scaling must be none or yarn")
                }
                ropeScalingMode = parsed
            case "--queue-limit":
                guard let parsed = Int(value), (1...64).contains(parsed) else {
                    throw ServerArgumentError.invalid("--queue-limit must be between 1 and 64")
                }
                queueLimit = parsed
            case "--max-concurrent-sequences":
                // A power of two, up to the engine's slot ceiling. The old bound
                // of 4 was a policy limit, not an engineering one: the batched
                // stores are sized per slot and `BatchedMemoryBudget` clamps the
                // width to what the machine can actually hold at load, logging
                // what it built. So the argument may ask for more than a given
                // Mac will run — that is the operator's call, and the clamp is
                // what keeps it honest.
                guard let parsed = Int(value), parsed >= 1,
                    parsed <= KVCacheManager.maximumSlots,
                    parsed & (parsed - 1) == 0
                else {
                    throw ServerArgumentError.invalid(
                        "--max-concurrent-sequences must be a power of two between 1 and "
                            + "\(KVCacheManager.maximumSlots)")
                }
                maxConcurrentSequences = parsed
            case "--prompt-cache-mode":
                guard let parsed = ServerPromptCacheMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-mode must be off, single-prefix, or multi-prefix")
                }
                promptCacheMode = parsed
            case "--prompt-cache-entries":
                guard let parsed = Int(value), (1...64).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-entries must be between 1 and 64")
                }
                promptCacheMaximumEntries = parsed
            case "--prompt-cache-memory-mib":
                guard let parsed = Int(value), (0...4_096).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-memory-mib must be between 0 and 4096")
                }
                promptCacheMemoryMiB = parsed
            case "--prompt-cache-disk":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-disk must not be empty")
                }
                promptCacheDiskDirectory = value
            case "--prompt-cache-disk-mib":
                guard let parsed = Int(value), (0...65_536).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-disk-mib must be between 0 and 65536")
                }
                promptCacheDiskMiB = parsed
            case "--prompt-cache-memory-ttl-seconds":
                guard let parsed = Int(value), (0...86_400).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-memory-ttl-seconds must be between 0 and 86400")
                }
                promptCacheMemoryTTLSeconds = parsed
            case "--prefill-chunk":
                guard let parsed = Int(value),
                    RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed)
                else {
                    throw ServerArgumentError.invalid("--prefill-chunk is not supported")
                }
                prefillChunkTokens = parsed
            case "--kv-bits":
                guard let bits = Int(value),
                    let parsed = KVCachePrecision(rawValue: bits)
                else {
                    throw ServerArgumentError.invalid("--kv-bits must be 4, 8, or 16")
                }
                kvCachePrecision = parsed
            case "--thinking":
                guard let parsed = ModelThinkingMode(rawValue: value) else {
                    throw ServerArgumentError.invalid("--thinking must be off or on")
                }
                thinkingMode = parsed
                thinkingWasSet = true
            case "--reasoning-effort":
                guard let parsed = ModelReasoningEffort(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--reasoning-effort must be low, medium, or xhigh")
                }
                reasoningEffort = parsed
                effortWasSet = true
            case "--reasoning":
                guard let parsed = ReasoningLevel(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--reasoning must be one of "
                            + ReasoningLevel.allCases.map(\.rawValue).joined(separator: ", "))
                }
                reasoningLevel = parsed
            case "--models-dir":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--models-dir must not be empty")
                }
                modelsDirectory = value
            case "--expert-cache-slots":
                guard let parsed = Int(value),
                    RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed)
                else {
                    throw ServerArgumentError.invalid(
                        "--expert-cache-slots must be one of \(RuntimeConfiguration.allowedExpertCacheSlots)"
                    )
                }
                expertCacheSlots = parsed
            case "--ram-budget":
                guard let parsed = RuntimeConfiguration.parseBudgetBytes(value) else {
                    throw ServerArgumentError.invalid(
                        "--ram-budget must be a positive size such as 4G, 512M or a byte count")
                }
                // Below 4 GiB the process cannot honour the number: the resident
                // weights plus the 8-slot minimum cache are already ~4.7 GiB on a
                // Qwen3.8 4-bit install. Refuse rather than accept a target that
                // silently overshoots by 2x.
                guard parsed >= RuntimeConfiguration.minimumProcessTargetBytes else {
                    throw ServerArgumentError.invalid(
                        "--ram-budget must be at least "
                            + "\(RuntimeConfiguration.minimumProcessTargetBytes >> 30)G: a "
                            + "streaming install holds its weights plus a minimum expert cache "
                            + "beside them, which is about 4.7G for Qwen3.8 4-bit")
                }
                expertCacheBudgetBytes = parsed
            case "--idle-unload-seconds":
                guard let parsed = Int(value), (0...86_400).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--idle-unload-seconds must be between 0 and 86400")
                }
                idleUnloadSeconds = parsed
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        if catalogOnly, modelsDirectory == nil {
            throw ServerArgumentError.invalid("--catalog requires --models-dir")
        }
        // Listing the catalog loads nothing, so it needs no initial model.
        guard let model = model ?? (catalogOnly ? "" : nil) else {
            throw ServerArgumentError.invalid("--model is required")
        }
        if reasoningLevel != nil, thinkingWasSet || effortWasSet {
            throw ServerArgumentError.invalid(
                "--reasoning replaces --thinking and --reasoning-effort; pass one or the other")
        }
        if modelsDirectory != nil, !catalogOnly {
            // Each of these configures one model; with a catalog there is no
            // single model for it to mean.
            let conflicts: [(present: Bool, why: String)] = [
                (modelIDOverride != nil, "--model-id: every catalog model keeps its own id"),
                (mtpModel != nil, "--mtp-model: a draft head belongs to one model"),
                (cpu, "--cpu: the catalog knows which engine each model uses"),
                (
                    idleUnloadSeconds > 0,
                    "--idle-unload-seconds: POST /v1/models/unload releases the resident model"
                ),
            ]
            if let conflict = conflicts.first(where: \.present) {
                throw ServerArgumentError.invalid(
                    "--models-dir cannot be combined with \(conflict.why)")
            }
        }
        if ropeScalingMode == .yarn {
            if !maxContextWasSet {
                maxContext = RuntimeConfiguration.defaultYaRNContextTokens
            }
            guard RuntimeConfiguration.supportedYaRNContextTokens.contains(maxContext) else {
                throw ServerArgumentError.invalid(
                    "YaRN --max-context must be 524288 or 1048576")
            }
        } else {
            guard RuntimeConfiguration.supportedContextTokens.contains(maxContext) else {
                throw ServerArgumentError.invalid("--max-context is not supported")
            }
        }
        if ropeScalingMode == .yarn, mtpModel != nil {
            throw ServerArgumentError.invalid(
                "--mtp-model cannot be combined with --rope-scaling yarn")
        }
        if let effort = reasoningEffort, !thinkingMode.isEnabled {
            throw ServerArgumentError.invalid(
                "--reasoning-effort \(effort.rawValue) requires --thinking on; "
                    + "the chat template ignores effort while thinking is off")
        }
        return ServerArguments(
            model: model,
            mtpModel: mtpModel,
            mtpMemoryMiB: mtpMemoryMiB,
            port: port,
            modelIDOverride: modelIDOverride,
            maxContext: maxContext,
            queueLimit: queueLimit,
            maxConcurrentSequences: maxConcurrentSequences,
            promptCacheMode: promptCacheMode,
            promptCacheMaximumEntries: promptCacheMaximumEntries,
            promptCacheMemoryMiB: promptCacheMemoryMiB,
            promptCacheDiskDirectory: promptCacheDiskDirectory,
            promptCacheDiskMiB: promptCacheDiskMiB,
            promptCacheMemoryTTLSeconds: promptCacheMemoryTTLSeconds,
            prefillChunkTokens: prefillChunkTokens,
            kvCachePrecision: kvCachePrecision,
            ropeScalingMode: ropeScalingMode,
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort,
            cpu: cpu,
            cpuResident: cpuResident,
            expertCacheSlots: expertCacheSlots,
            expertCacheBudgetBytes: expertCacheBudgetBytes,
            lazyLoad: lazyLoad,
            idleUnloadSeconds: idleUnloadSeconds,
            modelsDirectory: modelsDirectory,
            catalogOnly: catalogOnly,
            reasoningLevel: reasoningLevel)
    }
}

public enum ServerArgumentError: Error, Equatable, CustomStringConvertible {
    case help
    case invalid(String)

    public var description: String {
        switch self {
        case .help: "help"
        case .invalid(let message): message
        }
    }
}
