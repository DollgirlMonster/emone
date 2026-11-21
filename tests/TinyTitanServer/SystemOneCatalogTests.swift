import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanServerCore

/// Which decision adapter `/v1/systemone` serves: a package in the models
/// folder whose base install is in the catalog, chosen or turned off by
/// `TINYTITAN_SYSTEM_ONE`.
@Suite struct SystemOneCatalogTests {
    private func folder(packages: [(String, String, String)]) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("s1-catalog-\(UUID().uuidString)")
        for (name, modelID, base) in packages {
            let dir = root.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let manifest: [String: Any] = ["model_id": modelID, "base": ["model_id": base]]
            try JSONSerialization.data(withJSONObject: manifest)
                .write(to: dir.appendingPathComponent("decision_adapter.json"))
        }
        return root
    }

    private func catalog(_ root: URL) -> ModelCatalog {
        ModelCatalog(
            directory: root,
            entries: [
                ModelCatalog.Entry(
                    id: "qwen3.5-0.8b_8-Bit", name: "Qwen 3.5 0.8B", kind: .gpu(.qwen35Dense),
                    quant: 8,
                    path: root.appendingPathComponent("qwen3.5_0.8B_8Bit"),
                    sampling: GenerationDefaults.house)
            ])
    }

    @Test func servesThePackageWhoseBaseIsInstalled() throws {
        let root = try folder(packages: [
            ("a-orphan_lora", "orphan", "qwen3.5-4b"),
            ("rev-0.8b_lora", "rev-0.8b", "qwen3.5-0.8b"),
        ])
        defer { try? FileManager.default.removeItem(at: root) }
        let package = try #require(catalog(root).systemOnePackage(environment: [:]))
        #expect(package.modelID == "rev-0.8b")
        #expect(package.adapter.lastPathComponent == "rev-0.8b_lora")
        #expect(package.base.id == "qwen3.5-0.8b_8-Bit")
    }

    @Test func theEnvironmentPicksOneOrTurnsItOff() throws {
        let root = try folder(packages: [
            ("rev-0.8b_lora", "rev-0.8b", "qwen3.5-0.8b"), ("zed_lora", "zed", "qwen3.5-0.8b"),
        ])
        defer { try? FileManager.default.removeItem(at: root) }
        let c = catalog(root)
        #expect(c.systemOnePackage(environment: ["TINYTITAN_SYSTEM_ONE": "zed"])?.modelID == "zed")
        #expect(
            c.systemOnePackage(environment: ["TINYTITAN_SYSTEM_ONE": "zed_lora"])?.modelID == "zed")
        #expect(c.systemOnePackage(environment: ["TINYTITAN_SYSTEM_ONE": "off"]) == nil)
        #expect(c.systemOnePackage(environment: ["TINYTITAN_SYSTEM_ONE": "missing"]) == nil)
    }

    @Test func noPackageMeansNoSystemOne() throws {
        let root = try folder(packages: [])
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(catalog(root).systemOnePackage(environment: [:]) == nil)
    }
}
