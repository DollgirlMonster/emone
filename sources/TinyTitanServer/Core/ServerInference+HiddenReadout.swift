import Foundation
import TinyTitan

extension ServerModelSession {
    /// Answers an `x_hidden_states` request: prefill the rendered prompt, read
    /// the last token's residual after the chosen layers, return it.
    ///
    /// The prompt is rendered exactly as an ordinary request's would be (the
    /// chat template, including the generation prompt, so the token read is
    /// the last one the model would have conditioned its answer on).
    ///
    /// Cache behaviour is a *bypass*, modelled on the reasoning-switch
    /// precedent in `resolveCacheStart`: the readout always prefills from
    /// scratch, so it neither resumes a cached prefix nor restores a snapshot,
    /// and it publishes nothing -- no message-shaped entry, no frontier
    /// observation, no frontier checkpoint, no verbatim-render memory, no
    /// prefix-hash record. It does overwrite the live KV, so the live sequence
    /// is gone afterwards (`activePromptCacheEntryID` is cleared and the runner
    /// is reset); every cache entry and its snapshot survives, and the next
    /// ordinary request restores from its entry's snapshot or re-prefills.
    func generateHiddenReadout(
        _ request: ValidatedChatRequest,
        plan: HiddenReadoutPlan
    ) async throws -> ServerCompletion {
        // `generate` has already run the refusals (`refuseHiddenCaptureIfUnavailable`),
        // so nothing below the slot is touched on a refused request.
        let slot = try await acquireSlot()
        defer { releaseSlot(slot) }
        let renderTokenizer = try await resolvedTokenizer(for: request.reasoning)
        let prepared = try preparePrompt(request, renderTokenizer: renderTokenizer)
        let promptIDs = prepared.promptIDs

        // The live KV is about to be overwritten and then reset: nothing may
        // resume on it. Multi-prefix entries keep their snapshots.
        activePromptCacheEntryID = nil
        let progress = PrefillProgressMonitor.begin(total: promptIDs.count, cached: 0)
        defer { PrefillProgressMonitor.end(generation: progress) }
        let result: HiddenReadoutResult
        do {
            result = try await runner.readHiddenStates(
                promptIDs: promptIDs, plan: plan, prefillConfig: prefillConfig
            ) { done in
                PrefillProgressMonitor.prefill(
                    done: done, total: promptIDs.count, generation: progress)
            }
        } catch let error as HiddenReadoutError {
            mtpDecoder?.reset()
            switch error {
            case .invalidLayers(let reason), .unsupported(let reason):
                throw ServerRequestError.invalid(
                    message: reason, param: "x_hidden_states", code: "unsupported_value")
            case .internalInconsistency:
                throw error
            }
        } catch {
            mtpDecoder?.reset()
            throw error
        }
        // The readout reset the target runner; a draft stream riding on it is
        // stale with it.
        mtpDecoder?.reset()
        return ServerCompletion(
            content: "",
            toolCalls: [],
            finishReason: "hidden_states",
            usage: OpenAIUsage(
                promptTokens: promptIDs.count, completionTokens: 0,
                totalTokens: promptIDs.count, cachedTokens: 0, reasoningTokens: 0),
            hiddenStates: HiddenStatesPayload(result))
    }

    /// The refusals that depend on the session, checked before any state moves.
    /// Shared by both modes: a probe and a capture during a generation need the
    /// same model, runner and prefill; the probe adds its cache rule.
    func refuseHiddenCaptureIfUnavailable(_ plan: HiddenReadoutPlan) throws {
        func refuse(_ message: String, _ code: String = "unsupported_value") -> ServerRequestError {
            .invalid(message: message, param: "x_hidden_states", code: code)
        }
        do {
            try plan.validate(numLayers: model.config.numLayers)
        } catch let error as HiddenReadoutError {
            throw refuse(error.description, "invalid_value")
        }
        // A probe resets the runner, which discards the live conversation.
        // Single-prefix keeps no snapshots to restore it from, so serving one
        // would silently cost the live conversation its entry. A capture during
        // a generation leaves the cache exactly as any request would.
        guard !plan.prefillOnly || promptCacheMode != .singlePrefix else {
            throw refuse(
                "x_hidden_states with prefill_only would discard the live conversation, "
                    + "which --prompt-cache-mode single-prefix cannot restore; use "
                    + "multi-prefix or off, or prefill_only=false")
        }
        if let reason = runner.hiddenReadoutRefusal { throw refuse(reason) }
        guard prefillConfig.mode == .chunked else {
            throw refuse("x_hidden_states needs chunked prefill, which is switched off")
        }
    }
}
