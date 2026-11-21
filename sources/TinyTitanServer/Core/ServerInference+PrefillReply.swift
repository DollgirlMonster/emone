import Foundation
import TinyTitan

extension ServerModelSession {
    /// The tokens that stand for a prefilled assistant reply after the
    /// generation prompt (`x_prefill_reply`).
    ///
    /// The reply is committed as the answer of a turn that thought nothing: when
    /// the generation prompt leaves a thought open (thinking on, the prompt ends
    /// `<think>\\n`) the reply closes it first, which renders exactly as the chat
    /// template renders an earlier assistant turn in a replayed history. Any
    /// thinking the reply carries inline is dropped; the prompt cache matches a
    /// turn by its answer alone (`ServerPromptCache.answer`).
    ///
    /// The published entry is the render's KV rows plus these tokens, with the
    /// end-of-turn token as its one uncommitted boundary, so the next request
    /// that replays this reply followed by a user turn resumes from it.
    func prefillReplyTokens(
        _ reply: GFTokenizer.Message,
        generationPrompt: [Int32],
        renderTokenizer: GFTokenizer,
        maxContext: Int
    ) throws -> [Int32] {
        let answer = GFTokenizer.splitThinking(reply.content ?? "").answer
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !answer.isEmpty else {
            throw ServerRequestError.invalid(
                message: "the prefilled reply has no answer text outside its thinking",
                param: "messages", code: "invalid_value")
        }
        let opensThought = StructuredAssistantDecoder.promptLeavesThoughtOpen(
            generationPrompt, tokenizer: renderTokenizer)
        let tokens = renderTokenizer.encode(
            (opensThought ? "\n</think>\n\n" : "") + answer, addBOS: false)
        // One more row for the token the runner samples after the reply.
        guard generationPrompt.count + tokens.count + 1 < maxContext else {
            throw ServerRequestError.invalid(
                message: "prompt and prefilled reply exceed the configured context",
                param: "messages", code: "context_length_exceeded")
        }
        return tokens
    }
}
