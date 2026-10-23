import Foundation
import Metal

/// Does the MTP verify pass pick the tokens scalar decode picks?
///
/// Every token MTP emits is an argmax of `verifyGreedyPair`, which runs its
/// two rows through the prefill kernels and a fused pair head; scalar greedy
/// decode runs the decode kernels. This walks the same tokens through both
/// and reports, position by position, what each path predicts next and by
/// how much its winner led the runner-up. It needs no MTP sidecar: the verify
/// pass and its rollback run on the target alone.
///
/// Scalar decode first generates `count` greedy tokens after the prefilled
/// context. The sequence is then reset, the context prefilled again, and each
/// generated token is fed as the confirmed row of a verify pair (the next
/// generated token as the draft) and rolled back to that row, exactly as a
/// rejected draft is. So both paths condition on identical tokens, and a
/// position where they disagree is one where MTP's output would split from
/// scalar decode's.
public struct VerifyPairProbe: Sendable {
    public struct Row: Sendable {
        /// The token scalar decode chose here (the teacher-forced next token).
        public let decodeToken: Int32
        /// Top-1 minus top-2 logit of scalar decode's row.
        public let decodeMargin: Float
        /// The token the verify pass's confirmed row predicts.
        public let verifyToken: Int32
        public let verifyMargin: Float
    }

    public let contextTokens: Int
    public let rows: [Row]

    public var mismatches: [Int] {
        rows.indices.filter { rows[$0].decodeToken != rows[$0].verifyToken }
    }

    /// Winner and its lead over the runner-up, in raw logits.
    static func top2(_ logits: UnsafeBufferPointer<Float16>) -> (token: Int32, margin: Float) {
        var best = 0
        var bestValue = -Float.infinity
        var second = -Float.infinity
        for index in 0..<logits.count {
            let value = Float(logits[index])
            if value > bestValue {
                second = bestValue
                bestValue = value
                best = index
            } else if value > second {
                second = value
            }
        }
        return (Int32(best), bestValue - second)
    }
}

extension RealForwardRunner {
    /// See `VerifyPairProbe`. The runner must write real logits
    /// (`forceLogitsHead`), not the fused greedy head.
    public func probeVerifyPairAgreement(
        context: [Int32],
        count: Int,
        prefillConfig: PrefillRuntimeConfig,
        scratch: RawCompletionScratch
    ) async throws -> VerifyPairProbe {
        guard !context.isEmpty, count >= 2 else {
            throw ContinuationScoreError.emptyText(context: context.count, continuation: count)
        }
        let vocab = scratch.sampler.vocab
        func row(_ buffer: MTLBuffer, offset: Int = 0) -> UnsafeBufferPointer<Float16> {
            UnsafeBufferPointer(
                start: buffer.contents().bindMemory(to: Float16.self, capacity: offset + vocab)
                    + offset,
                count: vocab)
        }
        func prefillContext() async throws -> Int {
            await resetSequence(slot: 0)
            let result = try await prefillChunked(
                tokens: context[...], startPosition: 0, outputMode: .logits,
                config: prefillConfig, into: scratch.logits
            ) { _ in }
            guard result.seed == .logitsWritten else {
                throw PrefillError.unsupportedPrefillSeed(
                    "the probe needs prefill logits but the runner returned \(result.seed)")
            }
            return result.newPosition
        }

        // Scalar greedy decode, one token at a time.
        var position = try await prefillContext()
        var tokens = [VerifyPairProbe.top2(row(scratch.logits)).token]
        var decodeMargins: [Float] = []
        while tokens.count < count {
            try await produce(
                token: tokens[tokens.count - 1], position: position, into: scratch.logits)
            position += 1
            let pick = VerifyPairProbe.top2(row(scratch.logits))
            tokens.append(pick.token)
            decodeMargins.append(pick.margin)
        }

        // The same tokens through the verify pass, rolled back to the
        // confirmed row each time as a rejected draft is.
        position = try await prefillContext()
        var rows: [VerifyPairProbe.Row] = []
        rows.reserveCapacity(count - 1)
        for index in 0..<(count - 1) {
            let checkpoint = try captureSpeculativeCheckpoint(maximumBytes: .max)
            _ = try await verifyGreedyPair(
                [tokens[index], tokens[index + 1]], startPosition: position)
            let pick = VerifyPairProbe.top2(row(verificationLogits))
            try rollbackSpeculativeCheckpoint(checkpoint)
            position += 1
            rows.append(
                VerifyPairProbe.Row(
                    decodeToken: tokens[index + 1], decodeMargin: decodeMargins[index],
                    verifyToken: pick.token, verifyMargin: pick.margin))
        }
        return VerifyPairProbe(contextTokens: context.count, rows: rows)
    }
}
