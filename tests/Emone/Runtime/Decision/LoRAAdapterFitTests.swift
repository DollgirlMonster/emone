import Foundation
import Metal
import Testing

@testable import Emone

/// Which projections a runtime LoRA may target per family, and at what shape
/// (Emone-lmw.14). Pure arithmetic on the shipped configs: no model is
/// loaded. The MoE shapes below are pinned by hand from the configs so a
/// changed formula cannot agree with itself.
@Suite struct LoRAAdapterFitTests {
    private static func shape(
        _ t: LoRATarget, _ layer: Int, _ cfg: ArchConfig
    ) -> [Int]? {
        LoRAAdapterFit.expectedShape(target: t, layer: layer, config: cfg).map {
            [$0.rows, $0.columns]
        }
    }

    /// Qwen 3.6 35B-A3B: layer 0 is Gated DeltaNet, layer 3 full attention.
    /// GDN: qkv = 2*16*128 + 32*128 = 8192, value = 32*128 = 4096, a/b = 32.
    /// Full: q = 2 (output gate) * 16 * 256 = 8192, k/v = 2*256 = 512,
    /// o = 2048 x 16*256.
    @Test func qwen36ShapesAreTheCheckpointProjections() {
        let cfg = ArchConfig.qwen36_35B_A3B
        #expect(Self.shape(.gdnQKV, 0, cfg) == [8192, 2048])
        #expect(Self.shape(.gdnZ, 0, cfg) == [4096, 2048])
        #expect(Self.shape(.gdnA, 0, cfg) == [32, 2048])
        #expect(Self.shape(.gdnB, 0, cfg) == [32, 2048])
        #expect(Self.shape(.gdnOut, 0, cfg) == [2048, 4096])
        #expect(Self.shape(.q, 3, cfg) == [8192, 2048])
        #expect(Self.shape(.k, 3, cfg) == [512, 2048])
        #expect(Self.shape(.v, 3, cfg) == [512, 2048])
        #expect(Self.shape(.o, 3, cfg) == [2048, 4096])
        // A layer has one kind of attention, never both.
        #expect(Self.shape(.q, 0, cfg) == nil)
        #expect(Self.shape(.gdnQKV, 3, cfg) == nil)
        // The MoE MLP: the shared expert (512 wide) on every layer, under its
        // own names; the dense names do not exist here.
        for layer in [0, 3] {
            #expect(Self.shape(.sharedGate, layer, cfg) == [512, 2048])
            #expect(Self.shape(.sharedUp, layer, cfg) == [512, 2048])
            #expect(Self.shape(.sharedDown, layer, cfg) == [2048, 512])
            for t in [LoRATarget.gate, .up, .down] { #expect(Self.shape(t, layer, cfg) == nil) }
        }
        #expect(Self.shape(.q, 40, cfg) == nil)
    }

    /// The 125B: hidden 2560, 24 heads of 256 with the output gate, GDN
    /// 16 K-heads / 48 V-heads of 128: qkv = 2*16*128 + 48*128 = 10240.
    @Test func qwen38ShapesFollowItsConfig() throws {
        let cfg = ArchConfig.qwen38FlashNext
        let linear = try #require((0..<cfg.numLayers).first { cfg.layerIsLinear($0) })
        let attention = try #require((0..<cfg.numLayers).first { !cfg.layerIsLinear($0) })
        #expect(Self.shape(.gdnQKV, linear, cfg) == [10240, 2560])
        #expect(Self.shape(.gdnZ, linear, cfg) == [6144, 2560])
        #expect(Self.shape(.gdnA, linear, cfg) == [48, 2560])
        #expect(Self.shape(.gdnOut, linear, cfg) == [2560, 6144])
        #expect(Self.shape(.q, attention, cfg) == [12288, 2560])
        #expect(Self.shape(.k, attention, cfg) == [512, 2560])
        #expect(Self.shape(.o, attention, cfg) == [2560, 6144])
        #expect(Self.shape(.down, linear, cfg) == nil)
        #expect(Self.shape(.sharedUp, linear, cfg) == [640, 2560])
        #expect(Self.shape(.sharedDown, attention, cfg) == [2560, 640])
    }

    @Test func denseFamilyKeepsItsMLPTargets() {
        let dense = Self.dense()
        #expect(Self.shape(.gate, 0, dense) == [96, 32])
        #expect(Self.shape(.up, 1, dense) == [96, 32])
        #expect(Self.shape(.down, 1, dense) == [32, 96])
        for t in [LoRATarget.sharedGate, .sharedUp, .sharedDown] {
            #expect(Self.shape(t, 0, dense) == nil)
        }
    }

    @Test func refusesWhatAnMoEModelCannotCarry() {
        let cfg = ArchConfig.qwen36_35B_A3B
        let fits = [
            LoRAAdapterFit.Declared(layer: 0, target: .gdnQKV, rows: 8192, columns: 2048),
            LoRAAdapterFit.Declared(layer: 3, target: .q, rows: 8192, columns: 2048),
            LoRAAdapterFit.Declared(layer: 0, target: .sharedUp, rows: 512, columns: 2048),
        ]
        #expect(LoRAAdapterFit.refusal(entries: fits, config: cfg) == nil)
        #expect(LoRAAdapterFit.refusal(entries: [], config: cfg) == nil)

        let mlp = LoRAAdapterFit.refusal(
            entries: fits + [.init(layer: 0, target: .up, rows: 512, columns: 2048)], config: cfg)
        #expect(mlp?.contains("mlp.up_proj") == true)
        #expect(mlp?.contains("shared_expert") == true)

        let wrongKind = LoRAAdapterFit.refusal(
            entries: [.init(layer: 0, target: .q, rows: 8192, columns: 2048)], config: cfg)
        #expect(wrongKind?.contains("Gated-DeltaNet layer") == true)

        let wrongShape = LoRAAdapterFit.refusal(
            entries: [.init(layer: 3, target: .k, rows: 256, columns: 2048)], config: cfg)
        #expect(wrongShape?.contains("256x2048") == true)
        #expect(wrongShape?.contains("512x2048") == true)

        let wrongLayer = LoRAAdapterFit.refusal(
            entries: [.init(layer: 40, target: .gdnZ, rows: 4096, columns: 2048)], config: cfg)
        #expect(wrongLayer?.contains("layer 40") == true)
    }

    /// The synthetic package covers every projection the fit allows, loads
    /// through the real reader, and fits the model it was written for.
    @Test func syntheticPackageFitsItsModel() throws {
        let cfg = ArchConfig.qwen36_35B_A3B
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("lora-fit-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try SyntheticLoRAPackage.write(to: dir, config: cfg, baseModelID: "qwen36-test", rank: 4)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let adapter = try LoRAAdapter(
            directory: dir, baseModelID: "qwen36-test", device: device, maxTokens: 8)
        // 30 GDN layers x 5 + 10 attention layers x 4 + 40 shared experts x 3.
        #expect(adapter.targetCount == 30 * 5 + 10 * 4 + 40 * 3)
        #expect(adapter.rank == 4 && adapter.scale == 2)
        #expect(LoRAAdapterFit.refusal(entries: adapter.declared, config: cfg) == nil)
    }

    @Test func zeroBPackageHasOnlyZeroB() throws {
        let cfg = Self.dense()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("lora-zero-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try SyntheticLoRAPackage.write(
            to: dir, config: cfg, baseModelID: "dense-test", rank: 2, zeroB: true)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let adapter = try LoRAAdapter(
            directory: dir, baseModelID: "dense-test", device: device, maxTokens: 8)
        let floats = adapter.buffer.contents().bindMemory(
            to: Float.self, capacity: adapter.buffer.length / 4)
        var nonzeroA = false
        for d in adapter.declared {
            let entry = try #require(adapter.entry(layer: d.layer, target: d.target))
            let b = floats + entry.bOffset / 4
            #expect((0..<(d.rows * 2)).allSatisfy { b[$0] == 0 })
            let a = floats + entry.aOffset / 4
            if (0..<(d.columns * 2)).contains(where: { a[$0] != 0 }) { nonzeroA = true }
        }
        #expect(nonzeroA, "A stays random, so only B makes the delta zero")
    }

    /// A small dense Qwen 3.5-shaped config: one GDN and one attention layer.
    private static func dense() -> ArchConfig {
        ArchConfig(
            hiddenSize: 32,
            intermediateSize: 96,
            moeIntermediateSize: 96,
            numHeads: 2,
            numKVHeads: 1,
            numFullKVHeads: 1,
            headDim: 16,
            fullHeadDim: 16,
            vocabSize: 64,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 2,
            numExperts: 0,
            topKExperts: 0,
            tieWordEmbeddings: true,
            attentionKEqV: false,
            fullAttentionLayerMask: [2, 1],
            hiddenActivation: "silu",
            family: .qwen35Dense,
            attnOutputGate: true,
            linearAttention: LinearAttentionConfig(
                numKHeads: 2, numVHeads: 2, keyHeadDim: 8, valueHeadDim: 8, convKernelSize: 4))
    }
}
