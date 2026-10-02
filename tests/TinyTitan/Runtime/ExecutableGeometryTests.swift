import Foundation
import Testing

@testable import TinyTitan

/// `validateExecutableGeometry` is the load-time refusal for a model wider
/// than the kernels' threadgroup tiles. These pin where the bound sits for
/// each kernel family, and that the Swift bound is the one the Metal source
/// actually compiles with.
@Suite struct ExecutableGeometryTests {
    private static func config(hiddenSize: Int, numExperts: Int) -> ArchConfig {
        ArchConfig(
            hiddenSize: hiddenSize,
            intermediateSize: 128,
            moeIntermediateSize: 128,
            numHeads: 24,
            numKVHeads: 4,
            numFullKVHeads: 4,
            headDim: 256,
            fullHeadDim: 256,
            vocabSize: 1024,
            slidingWindow: 1024,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 2,
            numExperts: numExperts,
            topKExperts: numExperts == 0 ? 0 : 8,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [2, 1],
            hiddenActivation: "silu",
            family: numExperts == 0 ? .qwen35Dense : .qwen36,
            attnOutputGate: true,
            attentionScale: 0.0625,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: numExperts != 0,
            ropeNeoxSubdim: true,
            linearAttention: .none)
    }

    private static func refused(_ config: ArchConfig) -> Bool {
        do {
            try Model.validateExecutableGeometry(config)
            return false
        } catch ModelError.unsupportedArchitecture(detail: _) {
            return true
        } catch {
            return false
        }
    }

    /// Qwen3.8-27B: dense, 5120 wide, 24 query heads of 256. Before the
    /// Gated-DeltaNet tile was widened it was refused on its first request.
    @Test func qwen38_27bGeometryIsServed() {
        #expect(!Self.refused(Self.config(hiddenSize: 5120, numExperts: 0)))
    }

    @Test func denseWiderThanTheGDNTileIsRefused() {
        let width = Model.maximumDenseThreadgroupTileWidth
        #expect(!Self.refused(Self.config(hiddenSize: width, numExperts: 0)))
        #expect(Self.refused(Self.config(hiddenSize: width + 1, numExperts: 0)))
    }

    /// The MoE tile did not widen with the GDN one.
    @Test func moeKeepsTheNarrowerTile() {
        #expect(!Self.refused(Self.config(hiddenSize: 2816, numExperts: 256)))
        #expect(Self.refused(Self.config(hiddenSize: 2817, numExperts: 256)))
        #expect(Self.refused(Self.config(hiddenSize: 5120, numExperts: 256)))
    }

    /// The guard is only as good as its agreement with the kernels: a Swift
    /// bound wider than the Metal tile lets a model write past it silently.
    @Test func swiftBoundsMatchTheMetalTiles() throws {
        let metal = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Runtime
            .deletingLastPathComponent()  // TinyTitan
            .deletingLastPathComponent()  // tests
            .deletingLastPathComponent()  // repository root
            .appendingPathComponent("sources/TinyTitan/Metal")
        func constant(_ name: String, in file: String) throws -> Int? {
            let text = try String(
                contentsOf: metal.appendingPathComponent(file), encoding: .utf8)
            let line = text.split(separator: "\n").first {
                $0.contains("constexpr uint \(name) =")
            }
            return line?.split(separator: "=").last
                .map { $0.trimmingCharacters(in: .whitespaces.union(.init(charactersIn: ";u"))) }
                .flatMap { Int($0) }
        }
        #expect(
            try constant("kGDNActivationMaxD", in: "GDN/gdn.metal")
                == Model.maximumDenseThreadgroupTileWidth)
        #expect(
            try constant("kMoEXMaxD", in: "MoE/moe.metal")
                == Model.maximumThreadgroupTileWidth)
    }
}
