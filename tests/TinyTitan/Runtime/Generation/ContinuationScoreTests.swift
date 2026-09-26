import Foundation
import Metal
import Testing

@testable import TinyTitan

/// `scoreContinuation` is the instrument for a prefill numerics change, so the
/// two things that must hold are the arithmetic (each NLL is the real
/// -log softmax of the target) and the teacher forcing (the context goes through
/// chunked prefill once; each continuation token but the last is fed through
/// decode at its own position, and scored against the logits before it).
@Suite struct ContinuationScoreTests {
    /// Predicts `input + 1` (mod vocab) with logit 4, everything else 0, from
    /// whichever token it was fed last -- so a wrongly fed or skipped token
    /// shows up as a wrong NLL, not just a wrong call count.
    final class NextTokenProducer: LogitProducer, ChunkedPrefillRunner, @unchecked Sendable {
        let vocab: Int
        private(set) var prefills: [(tokens: [Int32], start: Int)] = []
        private(set) var produced: [(token: Int32, position: Int)] = []

        init(vocab: Int) { self.vocab = vocab }

        func reset() {
            prefills.removeAll()
            produced.removeAll()
        }

        private func write(after token: Int32, into logits: MTLBuffer) {
            let row = logits.contents().bindMemory(to: Float16.self, capacity: vocab)
            for i in 0..<vocab { row[i] = 0 }
            row[(Int(token) + 1) % vocab] = 4
        }

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            produced.append((token, position))
            write(after: token, into: logits)
        }

        func prefillChunked(
            tokens: ArraySlice<Int32>, startPosition: Int, outputMode: PrefillOutputMode,
            config: PrefillRuntimeConfig, into logits: MTLBuffer, onProgress: (Int) -> Void
        ) async throws -> PrefillResult {
            prefills.append((Array(tokens), startPosition))
            if let last = tokens.last { write(after: last, into: logits) }
            return PrefillResult(newPosition: startPosition + tokens.count, seed: .logitsWritten)
        }
    }

    private static func nll(_ logits: [Float16], _ target: Int, softcap: Float = 0) -> Double {
        logits.withUnsafeBufferPointer {
            ContinuationScore.negativeLogLikelihood($0, target: target, softcap: softcap)
        }
    }

    @Test func theNLLIsMinusLogSoftmaxOfTheTarget() {
        #expect(abs(Self.nll([Float16](repeating: 0, count: 16), 3) - log(16.0)) < 1e-9)
        let hit = -log(exp(4.0) / (exp(4.0) + 15))
        let miss = -log(1 / (exp(4.0) + 15))
        var peaked = [Float16](repeating: 0, count: 16)
        peaked[5] = 4
        #expect(abs(Self.nll(peaked, 5) - hit) < 1e-6)
        #expect(abs(Self.nll(peaked, 6) - miss) < 1e-6)
        // A soft-cap of 2 squashes the 4 to 2 * tanh(2) before the softmax.
        let capped = 2 * tanh(2.0)
        let cappedHit = -log(exp(capped) / (exp(capped) + 15))
        #expect(abs(Self.nll(peaked, 5, softcap: 2) - cappedHit) < 1e-6)
    }

    @Test func theContextIsPrefilledOnceAndTheContinuationTeacherForced() async throws {
        let vocab = 16
        let producer = NextTokenProducer(vocab: vocab)
        let scratch = try RawCompletionScratch(context: try MetalContext(), vocab: vocab)
        let context: [Int32] = [5, 6, 7]
        let continuation: [Int32] = [8, 9, 3]
        let score = try await scoreContinuation(
            producer: producer, context: context, continuation: continuation,
            prefillConfig: .production(chunkTokens: 128), scratch: scratch, logitSoftcap: 0)

        #expect(producer.prefills.count == 1)
        #expect(producer.prefills.first?.tokens == context)
        #expect(producer.prefills.first?.start == 0)
        // Every continuation token but the last is fed, each at its own position.
        #expect(producer.produced.map(\.token) == [8, 9])
        #expect(producer.produced.map(\.position) == [3, 4])

        // 8 follows 7 and 9 follows 8 (predicted); 3 does not follow 9.
        let hit = -log(exp(4.0) / (exp(4.0) + 15))
        let miss = -log(1 / (exp(4.0) + 15))
        #expect(score.nlls.count == 3)
        #expect(abs(score.nlls[0] - hit) < 1e-6)
        #expect(abs(score.nlls[1] - hit) < 1e-6)
        #expect(abs(score.nlls[2] - miss) < 1e-6)
        #expect(score.contextTokens == 3)
        #expect(score.tokenHash == ContinuationScore.hash(context + continuation))
    }

    @Test func anEmptySideIsRefused() async throws {
        let producer = NextTokenProducer(vocab: 16)
        let scratch = try RawCompletionScratch(context: try MetalContext(), vocab: 16)
        await #expect(throws: ContinuationScoreError.self) {
            _ = try await scoreContinuation(
                producer: producer, context: [], continuation: [1],
                prefillConfig: .production(chunkTokens: 128), scratch: scratch, logitSoftcap: 0)
        }
        await #expect(throws: ContinuationScoreError.self) {
            _ = try await scoreContinuation(
                producer: producer, context: [1], continuation: [99],
                prefillConfig: .production(chunkTokens: 128), scratch: scratch, logitSoftcap: 0)
        }
    }
}
