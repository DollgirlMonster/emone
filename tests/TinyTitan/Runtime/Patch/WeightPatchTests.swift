import CryptoKit
import Foundation
import Metal
import Testing

@testable import TinyTitan

/// The weight-patch package reader (TinyTitan-a37): what it accepts and what
/// it refuses -- another base, a tensor file changed after the manifest
/// pinned it, a decision adapter's files, and a target version 1 cannot
/// apply in decode. The delta math is `LoRADeltaTests`.
@Suite struct WeightPatchTests {
    private static func sha(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func write(
        _ dir: URL, tensors: [(String, [Int])], rank: Int = 1, base: String = "qwen3.5-0.8b",
        format: String = WeightPatch.format
    ) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var header: [String: Any] = ["__metadata__": ["format": format]]
        var body = Data()
        for (name, shape) in tensors {
            let values = (0..<shape.reduce(1, *)).map(Float.init)
            let start = body.count
            values.withUnsafeBytes { body.append(contentsOf: $0) }
            header[name] = ["dtype": "F32", "shape": shape, "data_offsets": [start, body.count]]
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while json.count % 8 != 0 { json.append(0x20) }
        var file = Data()
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { file.append(contentsOf: $0) }
        file.append(json)
        file.append(body)
        try file.write(to: dir.appendingPathComponent(WeightPatch.tensorFile))
        let manifest: [String: Any] = [
            "format": format, "rank": rank, "base": ["model_id": base],
            "description": "test patch",
            "files": [WeightPatch.tensorFile: sha(file)],
        ]
        try JSONSerialization.data(withJSONObject: manifest).write(
            to: dir.appendingPathComponent(WeightPatch.manifestFile))
    }

    private static func scratch() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("patch-\(UUID().uuidString)")
    }

    private static let downPair: [(String, [Int])] = [
        ("layers.15.mlp.down_proj.lora_A", [1, 12]), ("layers.15.mlp.down_proj.lora_B", [8, 1]),
        ("layers.19.mlp.down_proj.lora_A", [1, 12]), ("layers.19.mlp.down_proj.lora_B", [8, 1]),
    ]

    @Test func readsDownProjectionDeltasWithUnitScale() throws {
        let dir = Self.scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.write(dir, tensors: Self.downPair)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let patch = try WeightPatch(
            directory: dir, baseModelID: "qwen3.5-0.8b", device: device, maxTokens: 16)
        #expect(patch.targetCount == 2 && patch.rank == 1)
        #expect(patch.layers == [15, 19])
        #expect(patch.summary == "test patch")
        let down = try #require(patch.entry(layer: 15, target: .down))
        #expect(down.rows == 8 && down.columns == 12)
        #expect(patch.entry(layer: 0, target: .down) == nil)
    }

    @Test func transposesBForTheUpKernel() throws {
        let dir = Self.scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.write(
            dir,
            tensors: [
                ("layers.0.mlp.down_proj.lora_A", [2, 4]),
                ("layers.0.mlp.down_proj.lora_B", [6, 2]),
            ], rank: 2)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let patch = try WeightPatch(
            directory: dir, baseModelID: "qwen3.5-0.8b", device: device, maxTokens: 4)
        let down = try #require(patch.entry(layer: 0, target: .down))
        let floats = patch.buffer.contents().bindMemory(
            to: Float.self, capacity: patch.buffer.length / 4)
        let b = floats + down.bOffset / 4
        for i in 0..<6 {
            for j in 0..<2 { #expect(b[j * 6 + i] == Float(i * 2 + j)) }
        }
    }

    @Test func refusesAnotherBaseAChangedFileAndAnAdapter() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let other = Self.scratch()
        defer { try? FileManager.default.removeItem(at: other) }
        try Self.write(other, tensors: Self.downPair, base: "qwen3.5-4b")
        #expect(throws: WeightPatch.LoadError.self) {
            try WeightPatch(
                directory: other, baseModelID: "qwen3.5-0.8b", device: device, maxTokens: 16)
        }
        let changed = Self.scratch()
        defer { try? FileManager.default.removeItem(at: changed) }
        try Self.write(changed, tensors: Self.downPair)
        let url = changed.appendingPathComponent(WeightPatch.tensorFile)
        var bytes = try Data(contentsOf: url)
        bytes[bytes.count - 1] ^= 0xFF
        try bytes.write(to: url)
        #expect(throws: WeightPatch.LoadError.self) {
            try WeightPatch(
                directory: changed, baseModelID: "qwen3.5-0.8b", device: device, maxTokens: 16)
        }
        let adapter = Self.scratch()
        defer { try? FileManager.default.removeItem(at: adapter) }
        try Self.write(adapter, tensors: Self.downPair, format: "tinytitan-lora-v1")
        #expect(throws: WeightPatch.LoadError.self) {
            try WeightPatch(
                directory: adapter, baseModelID: "qwen3.5-0.8b", device: device, maxTokens: 16)
        }
    }

    @Test func readsHeadScaleVectorsAloneOrBesideDeltas() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let alone = Self.scratch()
        defer { try? FileManager.default.removeItem(at: alone) }
        try Self.write(alone, tensors: [("layers.7.self_attn.head_scale", [8])])
        let scales = try WeightPatch(
            directory: alone, baseModelID: "qwen3.5-0.8b", device: device, maxTokens: 16)
        #expect(scales.targetCount == 0 && scales.headScaleLayers == [7])
        let seven = try #require(scales.headScale(layer: 7))
        #expect(seven.heads == 8)
        let floats = scales.buffer.contents().bindMemory(
            to: Float.self, capacity: scales.buffer.length / 4)
        #expect((0..<8).allSatisfy { floats[seven.offset / 4 + $0] == Float($0) })
        let both = Self.scratch()
        defer { try? FileManager.default.removeItem(at: both) }
        try Self.write(both, tensors: Self.downPair + [("layers.11.self_attn.head_scale", [8])])
        let mixed = try WeightPatch(
            directory: both, baseModelID: "qwen3.5-0.8b", device: device, maxTokens: 16)
        #expect(mixed.targetCount == 2 && mixed.headScaleLayers == [11])
    }

    @Test func readsDeltaNetHeadScalesAndRefusesTwoOnOneLayer() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let dir = Self.scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.write(
            dir,
            tensors: [
                ("layers.2.linear_attn.head_scale", [16]), ("layers.7.self_attn.head_scale", [8]),
            ])
        let patch = try WeightPatch(
            directory: dir, baseModelID: "qwen3.5-0.8b", device: device, maxTokens: 16)
        #expect(patch.headScaleLayers == [2, 7])
        let two = try #require(patch.headScale(layer: 2))
        let seven = try #require(patch.headScale(layer: 7))
        #expect(two.linear && two.heads == 16)
        #expect(!seven.linear && seven.heads == 8)
        let both = Self.scratch()
        defer { try? FileManager.default.removeItem(at: both) }
        try Self.write(
            both,
            tensors: [
                ("layers.2.linear_attn.head_scale", [16]), ("layers.2.self_attn.head_scale", [8]),
            ])
        #expect(throws: WeightPatch.LoadError.self) {
            try WeightPatch(
                directory: both, baseModelID: "qwen3.5-0.8b", device: device, maxTokens: 16)
        }
    }

    @Test func refusesAHeadScaleThatIsNotAVector() throws {
        let dir = Self.scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.write(dir, tensors: [("layers.7.self_attn.head_scale", [2, 4])])
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(throws: WeightPatch.LoadError.self) {
            try WeightPatch(
                directory: dir, baseModelID: "qwen3.5-0.8b", device: device, maxTokens: 16)
        }
    }

    @Test func refusesTargetsDecodeCannotApply() throws {
        let dir = Self.scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.write(
            dir,
            tensors: [
                ("layers.0.mlp.up_proj.lora_A", [1, 8]), ("layers.0.mlp.up_proj.lora_B", [12, 1]),
            ])
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(throws: WeightPatch.LoadError.self) {
            try WeightPatch(
                directory: dir, baseModelID: "qwen3.5-0.8b", device: device, maxTokens: 16)
        }
    }
}
