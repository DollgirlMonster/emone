import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanServerCore

@Suite("Server prompt cache")
struct ServerPromptCacheTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "model",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 16_384,
        kvStorage: "fp16",
        fp16RingEnabled: true,
        templateSHA256: "template")

    @Test func textContinuationUsesActualGeneratedHistoryAndOnlyPrefillsSuffix() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first")
        ])
        let initialPrompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let generated = tokenizer.encode("answer", addBOS: false)
        let kvBacked = initialPrompt + generated
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        let continuation = request(
            messages: initial.messages + [
                GFTokenizer.Message(role: .assistant, content: "answer"),
                GFTokenizer.Message(role: .user, content: "second"),
            ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(continuation.messages),
            addBOS: false)
        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected text continuation hit")
            return
        }
        let bridge = tokenizer.encodeTextContinuation(userContent: "second")
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
        #expect(effective[cached] == tokenizer.endOfTurnID)
    }

    @Test func mismatchedLineageDomainAndUnsafeStopsMiss() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first")
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var cache = ServerPromptCache()

        for reason in [StopReason.stopString, .eos] {
            let publication = cache.publish(
                domain: domain,
                request: initial,
                content: "answer",
                calls: [],
                result: rawResult(
                    prompt: prompt,
                    kvBacked: prompt,
                    boundary: tokenizer.eosID,
                    reason: reason))
            // Unsafe stop reasons must not publish a cacheable entry
            // (S17). `publish` returns nil when the entry is rejected.
            #expect(publication == nil)
        }

        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt + tokenizer.encode("answer", addBOS: false),
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let changed = request(messages: [
            GFTokenizer.Message(role: .user, content: "changed"),
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(changed.messages),
            addBOS: false)
        #expect(
            cache.match(
                domain: domain,
                request: changed,
                renderedPromptIDs: rendered,
                tokenizer: tokenizer) == .miss)
    }

    @Test func tailCompletedStopStringDoesNotPublishPrefix() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first")
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var matcher = StreamingStopMatcher(stops: ["🌳stop"])
        #expect(matcher.push("answer 🌳") == "answer ")
        #expect(matcher.push("stop") == "")
        #expect(matcher.isStopped)

        var cache = ServerPromptCache()
        let publication = cache.publish(
            domain: domain,
            request: initial,
            content: "answer ",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn),
            stopStringFiltered: matcher.isStopped)
        // A stop-string-filtered tail must not be published as a cacheable
        // entry: the cached KV would resume from a partial stop string (S17).
        #expect(publication == nil)
    }

    @Test func multiPrefixChoosesLongestExactPrefixAndUsesLRUEviction() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first")
        ])
        var cache = ServerPromptCache(maximumEntries: 2)
        let shortPublication = cache.publish(
            domain: domain,
            request: initial,
            content: "short",
            calls: [],
            result: rawResult(
                prompt: [1, 2],
                kvBacked: [1, 2],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let short = try #require(shortPublication)
        let longPublication = cache.publish(
            domain: domain,
            request: initial,
            content: "long",
            calls: [],
            result: rawResult(
                prompt: [1, 2, 3],
                kvBacked: [1, 2, 3],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let long = try #require(longPublication)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: [1, 2, 3, 4],
            tokenizer: tokenizer)
        #expect(
            match
                == .hit(
                    entryID: long.entry.id,
                    effectivePromptIDs: [1, 2, 3, 4],
                    cachedPromptTokens: 3))

        // The longer entry extends the shorter one, so it superseded it.
        #expect(long.evictedEntryIDs == [short.entry.id])
        #expect(cache.entries.map(\.id) == [long.entry.id])

        let otherPublication = cache.publish(
            domain: domain,
            request: initial,
            content: "other",
            calls: [],
            result: rawResult(
                prompt: [8],
                kvBacked: [8],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let other = try #require(otherPublication)
        #expect(other.evictedEntryIDs.isEmpty)

        let newestPublication = cache.publish(
            domain: domain,
            request: initial,
            content: "newest",
            calls: [],
            result: rawResult(
                prompt: [9],
                kvBacked: [9],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let newest = try #require(newestPublication)
        #expect(newest.evictedEntryIDs == [long.entry.id])
        #expect(cache.entries.map(\.id) == [other.entry.id, newest.entry.id])
    }

    /// Two conversations interleaved: each turn supersedes its own
    /// conversation's previous entry, so the short one's turns can no longer
    /// push the long one's only entry out of a small cache.
    @Test func interleavedConversationsKeepOneEntryEach() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first")
        ])
        var cache = ServerPromptCache(maximumEntries: 2)
        func publish(_ tokens: [Int32]) throws -> ServerPromptCachePublication {
            let publication = cache.publish(
                domain: domain,
                request: initial,
                content: "x",
                calls: [],
                result: rawResult(
                    prompt: tokens,
                    kvBacked: tokens,
                    boundary: tokenizer.endOfTurnID,
                    reason: .endOfTurn))
            return try #require(publication)
        }
        let long = Array(Int32(1)...Int32(40))
        let shortBase: [Int32] = [1, 2, 3, 500]
        let longTurn = try publish(long)
        var shortTurn = try publish(shortBase)
        for extra in Int32(600)..<Int32(603) {
            shortTurn = try publish(shortTurn.entry.kvBackedTokenIDs + [extra])
        }
        #expect(Set(cache.entries.map(\.id)) == [longTurn.entry.id, shortTurn.entry.id])
        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: long + [41],
            tokenizer: tokenizer)
        #expect(
            match
                == .hit(
                    entryID: longTurn.entry.id,
                    effectivePromptIDs: long + [41],
                    cachedPromptTokens: long.count))
    }

    /// An identical-prompt replay of a published entry reports the ENTIRE
    /// prompt as cached (`cachedPromptTokens == rendered.count`): at the
    /// session layer (S12) that leaves nothing to prefill, which is what
    /// "a prompt-cache hit short-circuits prefill" means. The end-to-end
    /// resume-path half of that contract lives inside `ServerModelSession`
    /// (its private init and hardcoded production-arch load make it
    /// untestable with the synthetic toy); this pins the cache layer's half.
    @Test func identicalReplayReportsEntirePromptAsCached() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first")
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var cache = ServerPromptCache()
        let publication = cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let entry = try #require(publication)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: prompt,
            tokenizer: tokenizer)
        #expect(
            match
                == .hit(
                    entryID: entry.entry.id,
                    effectivePromptIDs: prompt,
                    cachedPromptTokens: prompt.count))
    }

    /// A direct prefix hit is decided by the rendered tokens, which already
    /// carry the tool definitions, so a request whose tool list differs from
    /// the entry's still reuses it when its render begins with the entry's
    /// KV-backed tokens. A message-shaped continuation, whose bridge is
    /// rendered with the request's tools, still needs the same list.
    @Test func directPrefixHitIgnoresTheToolListButContinuationsDoNot() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first")
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let kvBacked = prompt + tokenizer.encode("answer", addBOS: false)
        var cache = ServerPromptCache()
        let publication = cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let entry = try #require(publication)
        let tool = GFTokenizer.FunctionDefinition(
            name: "skill_tool", description: "d", parameters: .object([:]))
        let messages =
            initial.messages + [
                GFTokenizer.Message(role: .assistant, content: "answer"),
                GFTokenizer.Message(role: .user, content: "second"),
            ]
        let extended = kvBacked + tokenizer.encode("more", addBOS: false)

        #expect(
            cache.match(
                domain: domain,
                request: request(messages: messages, tools: [tool]),
                renderedPromptIDs: extended,
                tokenizer: tokenizer)
                == .hit(
                    entryID: entry.entry.id,
                    effectivePromptIDs: extended,
                    cachedPromptTokens: kvBacked.count))

        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages),
            addBOS: false)
        #expect(
            cache.match(
                domain: domain,
                request: request(messages: messages, tools: [tool]),
                renderedPromptIDs: rendered,
                tokenizer: tokenizer) == .miss)
    }

    /// Regression: the cache keys on the post-strip view of a request, so a
    /// "<model>-fast" continuation re-renders its tail through CLIStrip too.
    /// Keying on the raw request instead produced a bridge that still carried
    /// the CLI's <system-reminder> scaffolding — an unstripped tail spliced
    /// onto a stripped prefix, so the cached turn silently lost the alias's
    /// strip and stopped reproducing a fresh prefill of the same request.
    @Test func strippedContinuationBridgeDropsReminderScaffolding() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let bloat = GFTokenizer.Message(role: .system, content: "you are an agent")
        let rawFirst = GFTokenizer.Message(
            role: .user,
            content: "first<system-reminder>\ncwd is /tmp\n</system-reminder>")
        let rawSecond = GFTokenizer.Message(
            role: .user,
            content: "second<system-reminder>\nfile changed\n</system-reminder>")

        // Turn 1, exactly as ServerInference composes it: strip, then key the
        // cache on the filtered view that was actually encoded.
        let firstStrip = CLIStrip.filter(messages: [bloat, rawFirst], tools: [])
        let initial = request(messages: [bloat, rawFirst])
            .replacingMessages(firstStrip.messages, tools: firstStrip.tools)
        #expect(initial.messages.map(\.content) == ["first"])

        let initialPrompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let kvBacked = initialPrompt + tokenizer.encode("answer", addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        // Turn 2 arrives with the bloat and both reminder blocks intact.
        let rawContinuation = [
            bloat,
            rawFirst,
            GFTokenizer.Message(role: .assistant, content: "answer"),
            rawSecond,
        ]
        let secondStrip = CLIStrip.filter(messages: rawContinuation, tools: [])
        let continuation = request(messages: rawContinuation)
            .replacingMessages(secondStrip.messages, tools: secondStrip.tools)
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(continuation.messages),
            addBOS: false)
        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected text continuation hit on the stripped view")
            return
        }
        // The bridge is the *stripped* user turn; the reminder block never
        // reaches the model, and the raw turn would have produced a longer one.
        #expect(cached == kvBacked.count)
        #expect(
            effective == kvBacked
                + tokenizer.encodeTextContinuation(userContent: "second"))
        #expect(
            effective != kvBacked
                + tokenizer.encodeTextContinuation(userContent: rawSecond.content ?? ""))

        // And the shape of the defect this guards: an entry keyed on the raw
        // messages still describes a KV range prefilled from the *stripped*
        // ones, so its continuation bridge carries the reminder block — an
        // unstripped tail on a stripped prefix.
        var rawKeyed = ServerPromptCache()
        rawKeyed.publish(
            domain: domain,
            request: request(messages: [bloat, rawFirst]),
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let rawMatch = rawKeyed.match(
            domain: domain,
            request: request(messages: rawContinuation),
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)
        guard case .hit(_, let rawEffective, _) = rawMatch else {
            Issue.record("expected the raw-keyed cache to still hit")
            return
        }
        #expect(
            rawEffective == kvBacked
                + tokenizer.encodeTextContinuation(userContent: rawSecond.content ?? ""))
        #expect(rawEffective != effective)
    }

    /// A user message sent while the agent is mid tool loop lands after the
    /// tool results. It only extends the prompt past the cached turn, so the
    /// request must resume from the cache, not re-prefill the whole session.
    @Test func toolResultsFollowedByUserMessageHit() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "add a recipe toolbelt")
        ])
        let initialPrompt = try tokenizer.encodeToolChat(messages: initial.messages, tools: [])
        // Generated text the render will not reproduce, so the hit has to come
        // from the message-shaped match rather than the direct token prefix.
        let kvBacked = initialPrompt + tokenizer.encode("generated call", addBOS: false)
        let call = ParsedToolCall(
            id: "call_1", name: "create_file",
            arguments: .object(["path": .string("recipe.py")]),
            argumentsJSON: #"{"path":"recipe.py"}"#)
        let assistant = GFTokenizer.Message(
            role: .assistant, content: nil,
            toolCalls: [.init(id: call.id, name: call.name, arguments: call.arguments)])
        let result = GFTokenizer.Message(
            role: .tool, content: "Error: File already exists",
            toolCallID: "call_1", name: "create_file")
        let interjection = GFTokenizer.Message(role: .user, content: "how's it coming?")

        func cache() -> ServerPromptCache {
            var cache = ServerPromptCache()
            cache.publish(
                domain: domain,
                request: initial,
                content: "",
                calls: [call],
                result: rawResult(
                    prompt: initialPrompt,
                    kvBacked: kvBacked,
                    boundary: tokenizer.endOfTurnID,
                    reason: .toolCalls))
            return cache
        }
        func match(_ tail: [GFTokenizer.Message]) throws -> ServerPromptCacheMatch {
            let messages = initial.messages + [assistant] + tail
            var cache = cache()
            return cache.match(
                domain: domain,
                request: request(messages: messages),
                renderedPromptIDs: try tokenizer.encodeToolChat(messages: messages, tools: []),
                tokenizer: tokenizer)
        }

        for tail in [[result], [result, interjection]] {
            guard case .hit(_, let effective, let cached) = try match(tail) else {
                Issue.record("expected a hit for \(tail.map(\.role))")
                continue
            }
            let bridge = try tokenizer.encodeToolResultContinuation(
                cachedMessages: initial.messages,
                assistant: assistant,
                incomingMessages: initial.messages + [assistant] + tail,
                tools: [])
            #expect(cached == kvBacked.count)
            #expect(effective == kvBacked + bridge)
        }

        // Shapes that are not a continuation of the cached turn still miss: a
        // call left unanswered, a tail that does not end on a user turn, and a
        // result for a call the cached turn never made.
        #expect(try match([interjection]) == .miss)
        #expect(
            try match([result, GFTokenizer.Message(role: .assistant, content: "done")]) == .miss)
        #expect(
            try match([
                result,
                GFTokenizer.Message(
                    role: .tool, content: "x", toolCallID: "call_9", name: "create_file"),
                interjection,
            ]) == .miss)
    }

    /// The post-strip view swaps only the messages and tools; every other
    /// validated field must survive, or the cached turn would silently change
    /// sampling or streaming behavior.
    @Test func replacingMessagesPreservesTheRestOfTheRequest() {
        let original = request(
            messages: [GFTokenizer.Message(role: .user, content: "hi")],
            tools: [])
        let replaced = original.replacingMessages(
            [GFTokenizer.Message(role: .user, content: "stripped")],
            tools: [])

        #expect(replaced.messages.map(\.content) == ["stripped"])
        #expect(replaced.stream == original.stream)
        #expect(replaced.includeUsage == original.includeUsage)
        #expect(replaced.maximumCompletionTokens == original.maximumCompletionTokens)
        #expect(replaced.stripCLIPrompt == original.stripCLIPrompt)
        #expect(
            replaced.generationConfig.maxNewTokens
                == original.generationConfig.maxNewTokens)
    }

    private func request(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition] = []
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages,
            tools: tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
    }

    private func rawResult(
        prompt: [Int32],
        kvBacked: [Int32],
        boundary: Int32,
        reason: StopReason
    ) -> RawDecodeResult {
        RawDecodeResult(
            prefillTokens: prompt.count,
            cachedPromptTokens: 0,
            computedPrefillTokens: prompt.count,
            prefillSeconds: 0,
            newTokens: 1,
            decodeSeconds: 0,
            reason: reason,
            kvPosition: kvBacked.count,
            kvBackedTokenIDs: kvBacked,
            uncommittedBoundaryTokenIDs: [boundary])
    }
}
