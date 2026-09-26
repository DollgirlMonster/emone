import Foundation
import Metal

/// Teacher-forced surprisal of a known continuation after a prefilled context.
///
/// This is the instrument for a *prefill* numerics change (such as
/// `TINYTITAN_PREFILL_MPP_WIDE`): the context goes through chunked prefill,
/// which the change affects, and the continuation goes through decode one token
/// at a time, which it does not. Two runs over the same tokens therefore differ
/// only in what the prefill left in the KV cache and recurrent state, and the
/// per-token difference in negative log-likelihood is a paired measure of it.
public struct ContinuationScore: Sendable, Equatable {
    public let contextTokens: Int
    /// Natural-log negative log-likelihood of each continuation token.
    public let nlls: [Double]
    /// FNV-1a over every token scored or conditioned on, so two runs can be
    /// shown to have read the same text.
    public let tokenHash: UInt64

    public var meanNLL: Double { nlls.isEmpty ? 0 : nlls.reduce(0, +) / Double(nlls.count) }
    public var perplexity: Double { exp(meanNLL) }

    public static func hash(_ tokens: [Int32]) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for token in tokens {
            h = (h ^ UInt64(UInt32(bitPattern: token))) &* 0x0000_0100_0000_01b3
        }
        return h
    }

    /// -log softmax(logits)[target], in Double, with the model's final
    /// soft-cap applied first when it has one (the sampler applies it the same
    /// way; the head writes raw logits).
    public static func negativeLogLikelihood(
        _ logits: UnsafeBufferPointer<Float16>, target: Int, softcap: Float
    ) -> Double {
        func value(_ i: Int) -> Double {
            let raw = Double(logits[i])
            guard softcap > 0 else { return raw }
            let cap = Double(softcap)
            return cap * tanh(raw / cap)
        }
        var peak = -Double.infinity
        for i in 0..<logits.count { peak = max(peak, value(i)) }
        var sum = 0.0
        for i in 0..<logits.count { sum += exp(value(i) - peak) }
        return peak + log(sum) - value(target)
    }
}

public enum ContinuationScoreError: Error, CustomStringConvertible, Equatable {
    case emptyText(context: Int, continuation: Int)
    case tokenOutOfVocabulary(Int32)

    public var description: String {
        switch self {
        case .emptyText(let context, let continuation):
            return "scoring needs a context and a continuation; got \(context) and \(continuation) tokens"
        case .tokenOutOfVocabulary(let token):
            return "token \(token) is outside the model's vocabulary"
        }
    }
}

/// Prefills `context` the way a completion does, then scores `continuation`
/// by teacher forcing through decode. The producer must write real logits
/// (a runtime built with `forceLogitsHead`), not the fused greedy head.
public func scoreContinuation(
    producer: any LogitProducer,
    context: [Int32],
    continuation: [Int32],
    prefillConfig: PrefillRuntimeConfig,
    scratch: RawCompletionScratch,
    logitSoftcap: Float
) async throws -> ContinuationScore {
    guard !context.isEmpty, !continuation.isEmpty else {
        throw ContinuationScoreError.emptyText(
            context: context.count, continuation: continuation.count)
    }
    let vocab = scratch.sampler.vocab
    if let bad = continuation.first(where: { $0 < 0 || Int($0) >= vocab }) {
        throw ContinuationScoreError.tokenOutOfVocabulary(bad)
    }
    await producer.resetSequence(slot: 0)

    var position = 0
    if prefillConfig.mode == .chunked, let chunked = producer as? any ChunkedPrefillRunner {
        let result = try await chunked.prefillChunked(
            tokens: context[...], startPosition: 0, outputMode: .logits,
            config: prefillConfig, into: scratch.logits
        ) { _ in }
        guard result.seed == .logitsWritten else {
            throw PrefillError.unsupportedPrefillSeed(
                "scoring needs prefill logits but the producer returned \(result.seed)")
        }
        position = result.newPosition
    } else {
        for token in context {
            try await producer.produce(token: token, position: position, into: scratch.logits)
            position += 1
        }
    }

    var nlls: [Double] = []
    nlls.reserveCapacity(continuation.count)
    for (index, token) in continuation.enumerated() {
        let row = UnsafeBufferPointer(
            start: scratch.logits.contents().bindMemory(to: Float16.self, capacity: vocab),
            count: vocab)
        nlls.append(
            ContinuationScore.negativeLogLikelihood(
                row, target: Int(token), softcap: logitSoftcap))
        // The last token's own prediction is never scored, so it is not run.
        guard index + 1 < continuation.count else { break }
        try await producer.produce(token: token, position: position, into: scratch.logits)
        position += 1
    }
    return ContinuationScore(
        contextTokens: context.count, nlls: nlls,
        tokenHash: ContinuationScore.hash(context + continuation))
}
