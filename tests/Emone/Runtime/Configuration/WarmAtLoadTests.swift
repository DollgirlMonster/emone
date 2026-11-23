import Testing

@testable import Emone

/// The load-time warm rule: pure, so every branch is checked without a model.
@Suite struct WarmAtLoadTests {
    /// Qwen 3.6 35B-A3B 4-bit on the 64 GB M1 Max, fresh load.
    func inputs(
        prompt: Int = 13, maxNew: Int = 96, fits: Bool = true,
        missing: Int = 10_240, available: UInt64 = 48 << 30
    ) -> WarmAtLoadInputs {
        WarmAtLoadInputs(
            cacheHoldsEveryExpert: fits, missingExperts: missing, expertBytes: 1_769_472,
            physicalMemoryBytes: 64 << 30, availableMemoryBytes: available,
            promptTokens: prompt, maxNewTokens: maxNew)
    }

    @Test func offNeverWarmsAndOnAlwaysDoesWhenItFits() {
        #expect(!WarmAtLoad.decide(mode: .off, inputs: inputs(maxNew: 100_000)).warm)
        #expect(WarmAtLoad.decide(mode: .on, inputs: inputs(maxNew: 1)).warm)
        #expect(!WarmAtLoad.decide(mode: .on, inputs: inputs(fits: false)).warm)
    }

    @Test func autoRefusesWhatCannotFitOrIsAlreadyWarm() {
        #expect(!WarmAtLoad.decide(mode: .auto, inputs: inputs(maxNew: 10_000, fits: false)).warm)
        let warm = WarmAtLoad.decide(mode: .auto, inputs: inputs(maxNew: 10_000, missing: 0))
        #expect(!warm.warm)
        #expect(warm.reason == "already warm")
    }

    @Test func autoKeepsMemoryFree() {
        // 18.1 GB to pin with 20 GB available leaves under 15 % of 64 GB.
        let tight = WarmAtLoad.decide(
            mode: .auto, inputs: inputs(maxNew: 10_000, available: 20 << 30))
        #expect(!tight.warm)
        #expect(tight.reason == "not enough free memory")
    }

    @Test func autoWarmsALongGenerationAndNotAShortOne() {
        let warmSeconds = 10_240 * 1_769_472 / WarmAtLoad.warmBytesPerSecond
        #expect(
            abs(WarmAtLoad.decide(mode: .auto, inputs: inputs()).warmSeconds - warmSeconds) < 1e-9)
        #expect(!WarmAtLoad.decide(mode: .auto, inputs: inputs(maxNew: 8)).warm)
        #expect(WarmAtLoad.decide(mode: .auto, inputs: inputs(maxNew: 2_000)).warm)
    }

    @Test func theBreakEvenIsWhereSavingMeetsTheWarm() {
        let short = WarmAtLoad.savings[0]
        let warmSeconds = 10_240 * 1_769_472 / WarmAtLoad.warmBytesPerSecond
        let breakEven = Int(
            ((warmSeconds - short.prefillSeconds) * 1000 / short.decodeMsPerToken).rounded(.up))
        #expect(!WarmAtLoad.decide(mode: .auto, inputs: inputs(maxNew: breakEven - 1)).warm)
        #expect(WarmAtLoad.decide(mode: .auto, inputs: inputs(maxNew: breakEven + 1)).warm)
    }

    @Test func savingsInterpolateBetweenRowsAndHoldAtTheEnds() {
        let rows = WarmAtLoad.savings
        #expect(
            WarmAtLoad.predictedSavingSeconds(promptTokens: 0, maxNewTokens: 0)
                == rows[0].prefillSeconds)
        let last = rows[rows.count - 1]
        #expect(
            WarmAtLoad.predictedSavingSeconds(promptTokens: 1_000_000, maxNewTokens: 100)
                == last.prefillSeconds + 100 * last.decodeMsPerToken / 1000)
        let mid = (rows[0].promptTokens + rows[1].promptTokens) / 2
        let value = WarmAtLoad.predictedSavingSeconds(promptTokens: mid, maxNewTokens: 0)
        #expect(value <= max(rows[0].prefillSeconds, rows[1].prefillSeconds))
        #expect(value >= min(rows[0].prefillSeconds, rows[1].prefillSeconds))
    }

    @Test func everyModeParsesFromItsFlagValue() {
        for mode in WarmAtLoadMode.allCases {
            #expect(WarmAtLoadMode(rawValue: mode.rawValue) == mode)
        }
        #expect(WarmAtLoadMode(rawValue: "yes") == nil)
    }
}
