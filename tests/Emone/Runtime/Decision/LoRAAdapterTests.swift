import CryptoKit
import Foundation
import Metal
import Testing

@testable import Emone

/// The adapter package reader: what it accepts and, mostly, what it refuses --
/// a package for another base, a file changed after the manifest pinned it,
/// and a pair that is not rank-r A/B of a known target. The delta itself is
/// `LoRADeltaTests`; parity on Rev is `EmoneCLI decide --adapter`.
@Suite struct LoRAAdapterTests {
    private static func sha(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// A package with `tensors` (name -> shape) of zeros-but-one values.
    private static func write(
        _ dir: URL, tensors: [(String, [Int])], rank: Int = 2, base: String = "qwen3.5-0.8b"
    ) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var header: [String: Any] = ["__metadata__": ["format": "emone-lora-v1"]]
        var body = Data()
        for (name, shape) in tensors {
            let count = shape.reduce(1, *)
            // Distinct values, so a layout change is visible: value = flat index.
            let values = (0..<count).map(Float.init)
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
        try file.write(to: dir.appendingPathComponent(LoRAAdapter.adapterFile))
        let head = Data("head".utf8)
        try head.write(to: dir.appendingPathComponent(LoRAAdapter.headFile))
        let manifest: [String: Any] = [
            "format": "emone-lora-v1", "model_id": "rev-0.8b", "rank": rank, "alpha": 2 * rank,
            "base": ["model_id": base],
            "files": [LoRAAdapter.adapterFile: sha(file), LoRAAdapter.headFile: sha(head)],
        ]
        try JSONSerialization.data(withJSONObject: manifest).write(
            to: dir.appendingPathComponent(LoRAAdapter.manifestFile))
    }

    private static func scratch() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("lora-\(UUID().uuidString)")
        return dir
    }

    @Test func readsEveryPairWithItsShapeAndScale() throws {
        let dir = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.write(
            dir,
            tensors: [
                ("layers.0.mlp.up_proj.lora_A", [2, 8]), ("layers.0.mlp.up_proj.lora_B", [12, 2]),
                ("layers.3.linear_attn.in_proj_a.lora_A", [2, 8]),
                ("layers.3.linear_attn.in_proj_a.lora_B", [4, 2]),
            ])
        let device = try #require(MTLCreateSystemDefaultDevice())
        let adapter = try LoRAAdapter(
            directory: dir, baseModelID: "qwen3.5-0.8b", device: device, maxTokens: 16)
        #expect(adapter.targetCount == 2)
        #expect(adapter.rank == 2 && adapter.scale == 2)
        let up = try #require(adapter.entry(layer: 0, target: .up))
        #expect(up.rows == 12 && up.columns == 8)
        #expect(adapter.entry(layer: 3, target: .gdnA)?.rows == 4)
        #expect(adapter.entry(layer: 1, target: .up) == nil)
        // B is laid out transposed for `lora_up_fp16`: [r x out], element
        // (j, i) = B[i][j] = i * r + j; A is left as written.
        let floats = adapter.buffer.contents().bindMemory(
            to: Float.self, capacity: adapter.buffer.length / 4)
        let b = floats + up.bOffset / 4
        for i in 0..<12 {
            for j in 0..<2 { #expect(b[j * 12 + i] == Float(i * 2 + j)) }
        }
        let a = floats + up.aOffset / 4
        #expect((0..<16).allSatisfy { a[$0] == Float($0) })
    }

    @Test func refusesAPackageForAnotherBase() throws {
        let dir = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.write(
            dir,
            tensors: [
                ("layers.0.mlp.up_proj.lora_A", [2, 8]), ("layers.0.mlp.up_proj.lora_B", [12, 2]),
            ])
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(throws: LoRAAdapter.LoadError.self) {
            try LoRAAdapter(
                directory: dir, baseModelID: "qwen3.5-4b", device: device, maxTokens: 16)
        }
    }

    @Test func refusesAFileChangedAfterTheManifestPinnedIt() throws {
        let dir = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.write(
            dir,
            tensors: [
                ("layers.0.mlp.up_proj.lora_A", [2, 8]), ("layers.0.mlp.up_proj.lora_B", [12, 2]),
            ])
        try Data("retrained".utf8).write(to: dir.appendingPathComponent(LoRAAdapter.headFile))
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(throws: LoRAAdapter.LoadError.self) {
            try LoRAAdapter(
                directory: dir, baseModelID: "qwen3.5-0.8b", device: device, maxTokens: 16)
        }
    }

    @Test func refusesAPairOfTheWrongRankOrAnUnknownTarget() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        for tensors in [
            [("layers.0.mlp.up_proj.lora_A", [3, 8]), ("layers.0.mlp.up_proj.lora_B", [12, 3])],
            [
                ("layers.0.mlp.side_proj.lora_A", [2, 8]),
                ("layers.0.mlp.side_proj.lora_B", [12, 2]),
            ],
            [("layers.0.mlp.up_proj.lora_A", [2, 8])],
        ] {
            let dir = try Self.scratch()
            defer { try? FileManager.default.removeItem(at: dir) }
            try Self.write(dir, tensors: tensors)
            #expect(throws: LoRAAdapter.LoadError.self) {
                try LoRAAdapter(
                    directory: dir, baseModelID: "qwen3.5-0.8b", device: device, maxTokens: 16)
            }
        }
    }
}
