import Darwin
import Testing

@testable import TinyTitan

/// One profile per (model, width), resolved family -> table -> environment.
@Suite struct ModelProfileTests {
    static let shipped: [(String, ModelFamily)] = [
        ("qwen3.6-35b-a3b", .qwen36), ("ornith-1.5-35b-a3b", .qwen36),
        ("qwen-agentworld", .qwen36), ("kat-coder-v2.5", .qwen36),
        ("qwen3.8-flash-next", .qwen38flash),
    ]

    @Test func everyShippedInstallHasItsOwnRow() {
        for (id, family) in Self.shipped {
            for bits in [4, 8] {
                let p = ModelProfile.resolve(
                    modelID: id, family: family, weightBits: bits, environment: [:])
                #expect(p.isTabled, "\(id) \(bits)-bit falls back to its family")
                #expect(p.key == ModelProfile.Key(id, bits))
            }
        }
        #expect(ModelProfile.table.count == 10)
    }

    @Test func modelsSharingAFamilyResolveIndependently() {
        // Same family, different keys: editing one row cannot move the other.
        let a = ModelProfile.resolve(
            modelID: "qwen-agentworld", family: .qwen36, weightBits: 4, environment: [:])
        let q = ModelProfile.resolve(
            modelID: "qwen3.6-35b-a3b", family: .qwen36, weightBits: 4, environment: [:])
        #expect(a.key != q.key)
        #expect(ModelProfile.table[a.key] != nil && ModelProfile.table[q.key] != nil)
    }

    @Test func tabledValuesMatchWhatWasMeasured() {
        let q38 = ModelProfile.resolve(
            modelID: "qwen3.8-flash-next", family: .qwen38flash, weightBits: 4, environment: [:])
        #expect(q38.expertCacheBudgetBytes == 12 << 30)
        // Depth 1 since 2026-09-21: the ring was re-measured and wins at both
        // prompt lengths now (7-token +15.7%, ~500-token +14.6%), which
        // supersedes the 2026-09-05 decision to leave it off.
        #expect(q38.prefetchDepth == 1)
        #expect(q38.keepExpertCacheWired)
        let q38b = ModelProfile.resolve(
            modelID: "qwen3.8-flash-next", family: .qwen38flash, weightBits: 8, environment: [:])
        #expect(q38b.expertCacheBudgetBytes == Int(9.5 * Double(1 << 30)))
        // Inferred from the 4-bit A/B (see the 8-bit row comment), not measured.
        #expect(q38b.prefetchDepth == 1)
        #expect(q38.sampling.temperature == 1.0 && q38.sampling.topP == 0.95)
        #expect(!q38.hcFused && !q38.qsaGPUSelect)
        // Prefill, spike 10 and its surprisal A/B (docs/m1-prefill-spike.md).
        #expect(q38.prefillChunkTokens == 16_384)
        #expect(q38.prefillWideMPP && q38.prefillRoutedMPP)
        // Not carried to 8-bit: no surprisal check has seen those weights.
        #expect(q38b.prefillChunkTokens == 4_096)
        #expect(!q38b.prefillWideMPP && !q38b.prefillRoutedMPP)
        let q36 = ModelProfile.resolve(
            modelID: "qwen3.6-35b-a3b", family: .qwen36, weightBits: 8, environment: [:])
        #expect(q36.expertCacheBudgetBytes == 12 << 30)
        #expect(q36.keepExpertCacheWired)
        let q36four = ModelProfile.resolve(
            modelID: "qwen3.6-35b-a3b", family: .qwen36, weightBits: 4, environment: [:])
        #expect(q36four.expertCacheBudgetBytes == 10 << 30)
        #expect(q36.prefetchDepth == 1)
        #expect(q36.prefillChunkTokens == 4_096)
        #expect(q36.sampling == GenerationDefaults.house)
        #expect(!q36.prefillWideMPP && !q36.prefillRoutedMPP)
    }

    /// The prefill switches ship on one row only; every other row, and every
    /// family fallback, keeps the default kernels.
    @Test func prefillSwitchesShipOnlyOnTheQwen38FourBitRow() {
        for (key, row) in ModelProfile.table {
            let expected = key == ModelProfile.Key("qwen3.8-flash-next", 4)
            #expect(row.wideMPP == expected, Comment(rawValue: key.modelID))
            #expect(row.routedMPP == expected, Comment(rawValue: key.modelID))
        }
        let fallback = ModelProfile.resolve(
            modelID: "unknown", family: .qwen36, weightBits: 4, environment: [:])
        #expect(!fallback.prefillWideMPP && !fallback.prefillRoutedMPP)
    }

    @Test func samplingRowsFollowTheirSeries() {
        let qwen36Series = GenerationDefaults.Sampling(
            temperature: 0.6, topK: GenerationDefaults.topK, topP: 0.95)
        let qwen38Series = GenerationDefaults.Sampling(
            temperature: 1.0, topK: GenerationDefaults.topK, topP: 0.95)
        let expected: [(String, ModelFamily, GenerationDefaults.Sampling)] = [
            ("qwen3.6-35b-a3b", .qwen36, qwen36Series),
            ("qwen-agentworld", .qwen36, qwen36Series),
            // A fine-tune does not inherit its base's sampling: KAT-Coder's own
            // generation_config.json asks for 1.0, so its rows must not be
            // swept along by the 0.6 the rest of the qwen36 series uses.
            ("kat-coder-v2.5", .qwen36, qwen38Series),
            ("qwen3.8-flash-next", .qwen38flash, qwen38Series),
            // Not a Qwen-named series: it keeps the house values until its own
            // card is checked.
            ("ornith-1.5-35b-a3b", .qwen36, GenerationDefaults.house),
        ]
        for (id, family, sampling) in expected {
            for bits in [4, 8] {
                let p = ModelProfile.resolve(
                    modelID: id, family: family, weightBits: bits, environment: [:])
                #expect(p.sampling == sampling, "\(id) \(bits)-bit")
            }
        }
    }

    /// The ids the installer writes (`SupportedModelSource`, which this test
    /// target cannot import) carry the width; they must still find their row.
    @Test func installerIDsWithAWidthSuffixFindTheirRow() {
        let installed: [(String, ModelFamily, Int, String)] = [
            ("qwen3.6-35b-a3b-4bit", .qwen36, 4, "qwen3.6-35b-a3b"),
            ("qwen3.6-35b-a3b-8bit", .qwen36, 8, "qwen3.6-35b-a3b"),
            ("ornith-1.5-35b-a3b-4bit", .qwen36, 4, "ornith-1.5-35b-a3b"),
            ("ornith-1.5-35b-a3b-8bit", .qwen36, 8, "ornith-1.5-35b-a3b"),
            ("qwen3.8-flash-next-4bit", .qwen38flash, 4, "qwen3.8-flash-next"),
        ]
        for (id, family, bits, row) in installed {
            let p = ModelProfile.resolve(
                modelID: id, family: family, weightBits: bits, environment: [:])
            #expect(p.isTabled, "\(id) falls back to its family")
            #expect(p.key == ModelProfile.Key(row, bits))
        }
        let q38 = ModelProfile.resolve(
            modelID: "qwen3.8-flash-next-4bit", family: .qwen38flash, weightBits: 4,
            environment: [:])
        #expect(q38.keepExpertCacheWired)
        // Only a trailing width is stripped.
        #expect(ModelProfile.tableModelID("qwen3.6-35b-a3b-mtp-4bit") == "qwen3.6-35b-a3b-mtp")
        #expect(ModelProfile.tableModelID("some-4bit-model") == "some-4bit-model")
    }

    @Test func unknownModelFallsBackToItsFamily() {
        let p = ModelProfile.resolve(
            modelID: "qwen3.6-35b-a3b-mtp-4bit", family: .qwen36MTP, weightBits: 4, environment: [:]
        )
        #expect(!p.isTabled)
        let f = RuntimeConfiguration.decodeTuning(family: .qwen36MTP, weightBits: 4)
        #expect(p.expertCacheBudgetBytes == f.expertCacheBudgetBytes)
        #expect(p.prefetchDepth == f.prefetchDepth)
        #expect(p.prefillChunkTokens == nil)
        #expect(p.sampling == GenerationDefaults.forFamily(.qwen36MTP))
    }

    @Test func environmentOverridesTheTable() {
        let env = [
            "TINYTITAN_ROUTER_TOPK_SIMD": "0", "TINYTITAN_HC_FUSED": "1",
            "TINYTITAN_PREDICTIVE_PREFETCH": "1",
            "TINYTITAN_QSA_GPU_SELECT": "verify",
        ]
        let p = ModelProfile.resolve(
            modelID: "qwen3.6-35b-a3b", family: .qwen36, weightBits: 4, environment: env)
        #expect(!p.routerTopKSimd)
        #expect(p.hcFused)
        #expect(p.qsaGPUSelect)
        let off = ModelProfile.resolve(
            modelID: "qwen3.8-flash-next", family: .qwen38flash, weightBits: 4,
            environment: ["TINYTITAN_PREDICTIVE_PREFETCH": "0"])
        #expect(off.prefetchDepth == 0)

        // The prefill switches go both ways: off where a row ships them on,
        // on where it does not.
        let q38Off = ModelProfile.resolve(
            modelID: "qwen3.8-flash-next", family: .qwen38flash, weightBits: 4,
            environment: ["TINYTITAN_PREFILL_MPP_WIDE": "0", "TINYTITAN_PREFILL_ROUTED_MPP": "0"])
        #expect(!q38Off.prefillWideMPP && !q38Off.prefillRoutedMPP)
        let q36On = ModelProfile.resolve(
            modelID: "qwen3.6-35b-a3b", family: .qwen36, weightBits: 4,
            environment: ["TINYTITAN_PREFILL_MPP_WIDE": "1", "TINYTITAN_PREFILL_ROUTED_MPP": "1"])
        #expect(q36On.prefillWideMPP && q36On.prefillRoutedMPP)
    }

    @Test func theRowDecidesWhetherTheCacheStaysWired() {
        // The tri-state `TINYTITAN_KEEP_WIRED` override is gone: it measured a
        // wash on decode (-0.37%) and the row's own value is the decision, so
        // every streaming row keeps its cache wired and a dense row does not.
        let streaming = ModelProfile.resolve(
            modelID: "qwen3.8-flash-next", family: .qwen38flash,
            weightBits: 4, environment: [:])
        #expect(streaming.keepExpertCacheWired, "the row wires it")
        let dense = ModelProfile.resolve(
            modelID: "qwen3.5-4b", family: .qwen35Dense,
            weightBits: 4, environment: [:])
        #expect(!dense.keepExpertCacheWired, "a dense install has no routed-expert cache")
    }

    @Test func summaryNamesTheKeyAndEveryKnob() {
        let p = ModelProfile.resolve(
            modelID: "qwen-agentworld", family: .qwen36, weightBits: 8, environment: [:])
        for needle in [
            "model=qwen-agentworld", "bits=8", "tabled", "budget=", "prefetch=1",
            "chunk=4096", "topk_simd=true", "hc_fused=false", "keep_wired=true",
            "mpp_wide=false", "routed_mpp=false",
        ] {
            #expect(p.summary.contains(needle), Comment(rawValue: needle))
        }
    }
}
