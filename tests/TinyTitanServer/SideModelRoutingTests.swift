import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanServerCore

/// Side models: small models (session titles, one-word answers) that sit
/// beside the primary instead of being swapped in for it.
@Suite("Side model routing")
struct SideModelRoutingTests {
    private typealias Fixture = RoutingFixture

    /// The point of a side model: naming it neither evicts the primary nor
    /// waits for the primary's generation to finish.
    @Test func aSideModelRunsBesideTheResidentPrimary() async throws {
        let log = RoutingEventLog()
        let gate = RoutingGate()
        let router = try Fixture.router(
            log: log, gates: [Fixture.alpha.id: gate], sideModelIDs: [Fixture.dense.id])
        try await router.preload()
        let primary = Task { try await router.generate(Fixture.request(Fixture.alpha.id)) { _ in } }
        await Fixture.eventually("alpha to start generating") { await gate.isWaiting }

        let title = try await router.generate(Fixture.request(Fixture.dense.id)) { _ in }
        #expect(title.content == Fixture.dense.id)
        #expect(await router.residentModelID == Fixture.alpha.id)
        #expect(await router.residentSideModelIDs() == [Fixture.dense.id])
        #expect(log.maxConcurrentGenerations == 2)

        await gate.open()
        #expect(try await primary.value.content == Fixture.alpha.id)
        #expect(log.loads == ["load alpha_4-Bit", "load dense-2b_4-Bit"])
    }

    @Test func aSideModelLoadsOnceAndIsReusedByItsAliases() async throws {
        let log = RoutingEventLog()
        let router = try Fixture.router(log: log, sideModelIDs: [Fixture.dense.id])
        #expect(router.isSideModel(Fixture.dense.id))
        #expect(router.isSideModel(Fixture.dense.id + "@cpu"))
        #expect(!router.isSideModel(Fixture.alpha.id))
        for _ in 0..<3 { _ = try await router.generate(Fixture.request(Fixture.dense.id)) { _ in } }
        #expect(log.loads == ["load dense-2b_4-Bit"])
        // Nothing asked for the primary, so nothing loaded it.
        #expect(await router.residentModelID == nil)
    }

    /// Started *as* the primary, a would-be side model is the primary.
    @Test func theInitialModelIsNeverASideModel() throws {
        let router = try Fixture.router(
            initial: Fixture.dense.id, log: RoutingEventLog(), sideModelIDs: [Fixture.dense.id])
        #expect(!router.isSideModel(Fixture.dense.id))
    }

    /// A count must not load a side model, as it must not switch the primary.
    @Test func countingForAnUnloadedSideModelUsesItsTokenizerOnly() async throws {
        let log = RoutingEventLog()
        let router = try Fixture.router(log: log, sideModelIDs: [Fixture.dense.id])
        #expect(try await router.countPromptTokens(Fixture.request(Fixture.dense.id)) == 5)
        #expect(log.loads.isEmpty)
        _ = try await router.generate(Fixture.request(Fixture.dense.id)) { _ in }
        #expect(try await router.countPromptTokens(Fixture.request(Fixture.dense.id)) == 7)
    }

    @Test func aManualUnloadReleasesSideModelsToo() async throws {
        let log = RoutingEventLog()
        let router = try Fixture.router(log: log, sideModelIDs: [Fixture.dense.id])
        try await router.preload()
        _ = try await router.generate(Fixture.request(Fixture.dense.id)) { _ in }
        #expect(await router.unload())
        #expect(await router.residentSideModelIDs().isEmpty)
        #expect(await router.residentModelID == nil)
    }

    @Test func theCatalogPicksSideModelsByBaseIdOrEnvironment() {
        let catalog = ModelCatalog(
            directory: URL(fileURLWithPath: "/models"),
            entries: [
                Fixture.alpha,
                ModelCatalog.Entry(
                    id: "qwen3.5-0.8b_8-Bit", name: "Qwen 3.5 0.8B", kind: .gpu(.qwen35Dense),
                    quant: 8, path: URL(fileURLWithPath: "/models/qwen3.5_0.8B_8Bit"),
                    sampling: GenerationDefaults.house),
            ])
        #expect(catalog.sideModelIDs(environment: [:]) == ["qwen3.5-0.8b_8-Bit"])
        #expect(catalog.sideModelIDs(environment: ["TINYTITAN_SIDE_MODELS": "off"]).isEmpty)
        #expect(
            catalog.sideModelIDs(environment: ["TINYTITAN_SIDE_MODELS": "alpha_4-Bit"])
                == ["alpha_4-Bit"])
        #expect(
            catalog.sideModelIDs(environment: ["TINYTITAN_SIDE_MODELS": " alpha , qwen3.5-0.8b"])
                == ["alpha_4-Bit", "qwen3.5-0.8b_8-Bit"])
    }
}
