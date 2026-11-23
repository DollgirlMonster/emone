import Foundation

/// Whether a run fills the routed-expert cache at load (`--warm`).
public enum WarmAtLoadMode: String, Sendable, CaseIterable {
    /// Warm when `WarmAtLoad.decide` predicts it pays for itself.
    case auto
    case on
    case off
}

/// What the load-time warm decision reads. Everything is known before the
/// first token: the cache's capacity, the payload's expert stride, the
/// machine's free memory, and the request's prompt length and token limit.
public struct WarmAtLoadInputs: Sendable, Equatable {
    /// The cache has a slot for every expert of every layer.
    public var cacheHoldsEveryExpert: Bool
    /// Experts the warm would read (all of them on a fresh load).
    public var missingExperts: Int
    /// Bytes of one routed expert (`manifest.expertStride`).
    public var expertBytes: Int
    /// Physical memory, and how much of it is free or reclaimable now.
    public var physicalMemoryBytes: UInt64
    public var availableMemoryBytes: UInt64
    public var promptTokens: Int
    public var maxNewTokens: Int

    public init(
        cacheHoldsEveryExpert: Bool, missingExperts: Int, expertBytes: Int,
        physicalMemoryBytes: UInt64, availableMemoryBytes: UInt64,
        promptTokens: Int, maxNewTokens: Int
    ) {
        self.cacheHoldsEveryExpert = cacheHoldsEveryExpert
        self.missingExperts = missingExperts
        self.expertBytes = expertBytes
        self.physicalMemoryBytes = physicalMemoryBytes
        self.availableMemoryBytes = availableMemoryBytes
        self.promptTokens = promptTokens
        self.maxNewTokens = maxNewTokens
    }
}

public struct WarmAtLoadDecision: Sendable, Equatable {
    public let warm: Bool
    /// One short phrase for the footer: why it warmed or did not.
    public let reason: String
    /// Predicted warm time and predicted time it saves, in seconds.
    public let warmSeconds: Double
    public let savedSeconds: Double
}

/// The load-time warm rule: warm when the cache can hold the whole expert set,
/// the warm leaves enough memory free, and the time it is predicted to save
/// (prefill and decode on a warm cache instead of a cold one) exceeds the time
/// the warm takes. Pure, so the rule is tested without a model.
public enum WarmAtLoad {
    /// Warm read rate, bytes per second: 10,240 experts of 1,769,472 B in a
    /// median 10.25 s (15 warmed runs, M1 Max, external Thunderbolt drive,
    /// 2026-11-23).
    public static let warmBytesPerSecond = 1.77e9

    /// Memory that must stay free after the warm pins the expert set.
    public static let minimumFreeFractionAfterWarm = 0.15

    /// Cold minus warm milliseconds per decoded token, and cold minus warm
    /// prefill seconds, by prompt length. Measured on the M1 Max, Qwen 3.6
    /// 35B-A3B 4-bit, fresh CLI processes, 96 decoded tokens, median of 3
    /// interleaved rounds each way (2026-11-22/23): decode tok/s cold -> warm
    /// 12.76 -> 58.78 (13 tokens), 34.18 -> 48.53 (1k), 45.28 -> 49.08 (8k),
    /// 36.86 -> 36.15 (35k), 27.62 -> 29.07 (74k); prefill tok/s 4.3 -> 46.4,
    /// 136.5 -> 634.2, 889.8 -> 931.8, 622.6 -> 667.0, 427.6 -> 446.2. The 35k
    /// decode difference is inside the noise and taken as 0. Interpolated
    /// between rows, the end rows held beyond them.
    public struct SavingRow: Sendable, Equatable {
        public let promptTokens: Int
        public let decodeMsPerToken: Double
        public let prefillSeconds: Double
    }

    public static let savings: [SavingRow] = [
        SavingRow(promptTokens: 13, decodeMsPerToken: 61.4, prefillSeconds: 2.74),
        SavingRow(promptTokens: 1021, decodeMsPerToken: 8.65, prefillSeconds: 5.87),
        SavingRow(promptTokens: 8088, decodeMsPerToken: 1.71, prefillSeconds: 0.41),
        SavingRow(promptTokens: 34882, decodeMsPerToken: 0.0, prefillSeconds: 3.73),
        SavingRow(promptTokens: 73833, decodeMsPerToken: 1.80, prefillSeconds: 7.21),
    ]

    public static func decide(mode: WarmAtLoadMode, inputs: WarmAtLoadInputs)
        -> WarmAtLoadDecision
    {
        let bytes = Double(max(0, inputs.missingExperts)) * Double(max(0, inputs.expertBytes))
        let warmSeconds = bytes / warmBytesPerSecond
        let saved = predictedSavingSeconds(
            promptTokens: inputs.promptTokens, maxNewTokens: inputs.maxNewTokens)
        func result(_ warm: Bool, _ reason: String) -> WarmAtLoadDecision {
            WarmAtLoadDecision(
                warm: warm, reason: reason, warmSeconds: warmSeconds, savedSeconds: saved)
        }
        switch mode {
        case .off: return result(false, "off")
        case .on:
            return inputs.cacheHoldsEveryExpert
                ? result(true, "on") : result(false, "on, but the cache cannot hold every expert")
        case .auto:
            guard inputs.cacheHoldsEveryExpert else {
                return result(false, "the cache cannot hold every expert")
            }
            guard inputs.missingExperts > 0 else { return result(false, "already warm") }
            let total = Double(inputs.physicalMemoryBytes)
            let freeAfter = Double(inputs.availableMemoryBytes) - bytes
            guard total > 0, freeAfter / total >= minimumFreeFractionAfterWarm else {
                return result(false, "not enough free memory")
            }
            return saved > warmSeconds
                ? result(true, "predicted to save time")
                : result(false, "predicted not to pay off")
        }
    }

    /// Cold minus warm seconds for a run of `promptTokens` and `maxNewTokens`.
    public static func predictedSavingSeconds(promptTokens: Int, maxNewTokens: Int) -> Double {
        let row = interpolated(promptTokens: promptTokens)
        return row.prefillSeconds + Double(max(0, maxNewTokens)) * row.decodeMsPerToken / 1000
    }

    static func interpolated(promptTokens n: Int) -> (
        decodeMsPerToken: Double, prefillSeconds: Double
    ) {
        guard let first = savings.first, let last = savings.last else { return (0, 0) }
        if n <= first.promptTokens { return (first.decodeMsPerToken, first.prefillSeconds) }
        if n >= last.promptTokens { return (last.decodeMsPerToken, last.prefillSeconds) }
        for (a, b) in zip(savings, savings.dropFirst()) where n <= b.promptTokens {
            let f = Double(n - a.promptTokens) / Double(b.promptTokens - a.promptTokens)
            return (
                a.decodeMsPerToken + f * (b.decodeMsPerToken - a.decodeMsPerToken),
                a.prefillSeconds + f * (b.prefillSeconds - a.prefillSeconds)
            )
        }
        return (last.decodeMsPerToken, last.prefillSeconds)
    }
}
