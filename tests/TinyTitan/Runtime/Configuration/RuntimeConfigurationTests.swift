import Testing

@testable import TinyTitan

@Suite struct RuntimeConfigurationTests {
    @Test func publicContextChoicesReachQwenMaximum() {
        #expect(
            RuntimeConfiguration.supportedContextTokens
                == [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144])
        #expect(
            RuntimeConfiguration.supportedContextTokens.last
                == RuntimeConfiguration.nativeMaximumContextTokens)
        #expect(RuntimeConfiguration.supportedYaRNContextTokens == [524_288, 1_048_576])
    }

    @Test func productionDefaultsAreStable() throws {
        let runtime = try RuntimeConfiguration(
            expertCacheSlots: 16,
            expertCachePolicy: .lfu,
            rdadvisePolicy: .off,
            prefillEnabled: true,
            prefillChunkTokens: 128,
            prefillAttentionPath: .fullTensorOps2DPreferred,
            forceLogitsHead: false)
        #expect(runtime.fp16RingEnabled)
        #expect(runtime.expertCacheSlots == 16)
        #expect(runtime.expertCachePolicy == .lfu)
        #expect(runtime.rdadvisePolicy == .off)
        #expect(!runtime.rdadviseEnabled)
        #expect(runtime.prefillPolicy == .chunked)
        #expect(runtime.prefillChunkTokens == 128)
        #expect(runtime.prefillAttentionPath == .fullTensorOps2DPreferred)
        #expect(runtime.headPath == .fusedRows)
        #expect(runtime.decodeExpertExecution == .hitFixup)
        #expect(runtime.kvCachePrecision == .int8)
        #expect(runtime.ropeScalingMode == .none)
    }

    @Test func contextScalingValidationIsFailClosed() throws {
        let native = try RuntimeConfiguration()
        try native.validate(maxContext: 262_144)
        #expect(throws: RuntimeConfigurationError.self) {
            try native.validate(maxContext: 524_288)
        }
        let yarn = try RuntimeConfiguration(
            ropeScalingMode: .yarn,
            yarnContextTokens: 524_288)
        try yarn.validate(maxContext: 524_288)
        #expect(throws: RuntimeConfigurationError.self) {
            try yarn.validate(maxContext: 1_048_576)
        }
    }

    @Test func retainedControlsReachTypedRuntime() throws {
        let runtime = try RuntimeConfiguration(
            expertCacheSlots: 32,
            expertCachePolicy: .lru,
            rdadvisePolicy: .adaptive,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            prefillAttentionPath: .causalTiled,
            forceLogitsHead: true,
            decodeExpertExecution: .barrier)
        #expect(runtime.expertCacheSlots == 32)
        #expect(runtime.modelExpertCachePolicy == .lru)
        #expect(runtime.rdadviseEnabled)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.prefillAttentionPath == .causalTiled)
        #expect(runtime.headPath == .logits)
        #expect(runtime.decodeExpertExecution == .barrier)
    }

    @Test func decodeExpertExecutionEnvironmentIsFailClosed() throws {
        #expect(try RuntimeDecodeExpertExecution.environmentValue([:]) == .hitFixup)
        #expect(
            try RuntimeDecodeExpertExecution.environmentValue([
                "TINYTITAN_DECODE_EXPERT_EXECUTION": "barrier"
            ]) == .barrier)
        #expect(
            try RuntimeDecodeExpertExecution.environmentValue([
                "TINYTITAN_DECODE_EXPERT_EXECUTION": "gpu-residency"
            ]) == .gpuResidency)
        #expect(throws: RuntimeConfigurationError.self) {
            try RuntimeDecodeExpertExecution.environmentValue([
                "TINYTITAN_DECODE_EXPERT_EXECUTION": "typo"
            ])
        }
    }

    @Test func expertIOSynchronizationEnvironmentIsFailClosed() throws {
        #expect(try RuntimeExpertIOSynchronization.environmentValue([:]) == .host)
        #expect(
            try RuntimeExpertIOSynchronization.environmentValue([
                "TINYTITAN_EXPERT_IO_SYNC": "event"
            ]) == .event)
        #expect(throws: RuntimeConfigurationError.self) {
            try RuntimeExpertIOSynchronization.environmentValue([
                "TINYTITAN_EXPERT_IO_SYNC": "typo"
            ])
        }
    }

    @Test func expertIOSubmissionEnvironmentIsFailClosed() throws {
        #expect(try RuntimeExpertIOSubmission.environmentValue([:]) == .deferred)
        #expect(
            try RuntimeExpertIOSubmission.environmentValue([
                "TINYTITAN_EXPERT_IO_SUBMISSION": "immediate"
            ]) == .immediate)
        #expect(throws: RuntimeConfigurationError.self) {
            try RuntimeExpertIOSubmission.environmentValue([
                "TINYTITAN_EXPERT_IO_SUBMISSION": "typo"
            ])
        }
    }

    @Test(arguments: [32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384])
    func productionPrefillSupportsPublicChunkSizes(_ chunkTokens: Int) throws {
        let runtime = try RuntimeConfiguration(prefillChunkTokens: chunkTokens)
        #expect(runtime.prefillConfig.mode == .chunked)
        #expect(runtime.prefillConfig.chunkTokens == chunkTokens)
    }

    /// The key selection is indexed `row * visibleKeys + key` in 32 bits, so
    /// chunk x context may reach 2^32 but not pass it. Every native context
    /// takes 16K; the YaRN contexts cap the chunk, and the 4K chunk those
    /// contexts already ran with still fits.
    @Test func aChunkMustFitTheContextsSelectionIndex() throws {
        #expect(RuntimeConfiguration.prefillChunkFits(chunk: 16_384, maxContext: 262_144))
        #expect(!RuntimeConfiguration.prefillChunkFits(chunk: 16_384, maxContext: 524_288))
        #expect(RuntimeConfiguration.prefillChunkFits(chunk: 4_096, maxContext: 1_048_576))
        #expect(!RuntimeConfiguration.prefillChunkFits(chunk: 8_192, maxContext: 1_048_576))
        #expect(!RuntimeConfiguration.prefillChunkFits(chunk: Int.max, maxContext: 2))
        #expect(RuntimeConfiguration.largestPrefillChunk(forContext: 262_144) == 16_384)
        #expect(RuntimeConfiguration.largestPrefillChunk(forContext: 524_288) == 8_192)
        #expect(RuntimeConfiguration.largestPrefillChunk(forContext: 1_048_576) == 4_096)

        let yarn = try RuntimeConfiguration(
            prefillChunkTokens: 16_384, ropeScalingMode: .yarn, yarnContextTokens: 1_048_576)
        #expect(
            throws: RuntimeConfigurationError.prefillChunkTooLargeForContext(
                chunk: 16_384, maxContext: 1_048_576)
        ) {
            try yarn.validate(maxContext: 1_048_576)
        }
        try RuntimeConfiguration(prefillChunkTokens: 16_384).validate(maxContext: 262_144)
        try RuntimeConfiguration(
            prefillChunkTokens: 4_096, ropeScalingMode: .yarn, yarnContextTokens: 1_048_576
        ).validate(maxContext: 1_048_576)
    }

    /// A profile row's chunk is lowered to what the context allows, never
    /// raised, so the Qwen3.8 row's 16K still loads under YaRN.
    @Test func aProfileChunkIsCappedToTheContext() throws {
        #expect(RuntimeConfiguration.profilePrefillChunk(16_384, forContext: 262_144) == 16_384)
        #expect(RuntimeConfiguration.profilePrefillChunk(16_384, forContext: 524_288) == 8_192)
        #expect(RuntimeConfiguration.profilePrefillChunk(16_384, forContext: 1_048_576) == 4_096)
        #expect(RuntimeConfiguration.profilePrefillChunk(4_096, forContext: 1_048_576) == 4_096)
        #expect(RuntimeConfiguration.profilePrefillChunk(2_048, forContext: 65_536) == 2_048)
        for context in [262_144, 524_288, 1_048_576] {
            let chunk = RuntimeConfiguration.profilePrefillChunk(16_384, forContext: context)
            let yarn = context > 262_144
            try RuntimeConfiguration(
                prefillChunkTokens: chunk, ropeScalingMode: yarn ? .yarn : .none,
                yarnContextTokens: yarn ? context : RuntimeConfiguration.defaultYaRNContextTokens
            ).validate(maxContext: context)
        }
    }
}
