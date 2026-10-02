import Foundation
import Metal
import Testing

@testable import TinyTitan

/// The hidden-state readout through a real `RealForwardRunner` on the
/// synthetic qwen toy (4 layers, 64-wide, one residual stream): the layer-loop
/// hooks, the blit readback and the reset. A toy fixture is not a model; the
/// real thing (hyper-connection streams, 48 layers) needs the live check.
@Suite struct HiddenReadoutRunnerTests {
    private func makeRunner(slots: Int = 1) throws -> (URL, MetalContext, RealForwardRunner) {
        let dir = try QwenToySynthetic.write(weightBits: 4)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: dir, device: ctx.device, expecting: .qwenToy())
        let runtime = try RuntimeConfiguration(forceLogitsHead: true)
        let runner = try RealForwardRunner(
            model: model, context: ctx, maxContext: 128, slots: slots,
            runtimeConfiguration: runtime)
        return (dir, ctx, runner)
    }

    private func logitsBuffer(_ ctx: MetalContext) throws -> MTLBuffer {
        guard
            let buffer = ctx.device.makeBuffer(
                length: 1024 * MemoryLayout<Float16>.stride, options: .storageModeShared)
        else { throw ModelError.residentBufferWrapFailed }
        return buffer
    }

    private func logits(_ buffer: MTLBuffer) -> [Float16] {
        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: 1024)
        return Array(UnsafeBufferPointer(start: pointer, count: 1024))
    }

    private let prompt: [Int32] = [5, 9, 13, 21, 34, 55, 89, 144, 233, 377]

    private func readout(
        _ runner: RealForwardRunner, layers: [Int], chunk: Int = 32,
        mode: HiddenReadoutStreamMode = .residual, prompt: [Int32]? = nil
    ) async throws -> HiddenReadoutResult {
        try await runner.readHiddenStates(
            promptIDs: prompt ?? self.prompt,
            plan: try HiddenReadoutPlan(layers: layers, streamMode: mode),
            prefillConfig: .production(chunkTokens: chunk))
    }

    @Test func readsOneRowPerRequestedLayerAtTheLastPosition() async throws {
        let (dir, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let result = try await readout(runner, layers: [3, 0, 2, 2])
        #expect(result.position == prompt.count - 1)
        #expect(result.layers.map(\.layer) == [0, 2, 3])
        for layer in result.layers {
            #expect(layer.values.count == 64)
            #expect(layer.values.allSatisfy { $0.isFinite })
            #expect(layer.values.contains { $0 != 0 })
        }
        // The layers differ: each block changes the residual.
        #expect(result.layers[0].values != result.layers[1].values)
    }

    @Test func stoppingEarlyDoesNotChangeTheLayersBeforeTheStop() async throws {
        let (dir, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let shallow = try await readout(runner, layers: [1])
        let deep = try await readout(runner, layers: [1, 3])
        #expect(shallow.layers[0].values == deep.layers[0].values)
        // The same across chunks: earlier chunks also stop at the deepest layer.
        let long = (0..<70).map { Int32(($0 * 37 + 11) % 1000) }
        let shallowLong = try await readout(runner, layers: [1], chunk: 32, prompt: long)
        let deepLong = try await readout(runner, layers: [1, 3], chunk: 32, prompt: long)
        #expect(shallowLong.layers[0].values == deepLong.layers[0].values)
    }

    @Test func aOneStreamFamilysMeanIsTheResidual() async throws {
        let (dir, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let residual = try await readout(runner, layers: [2])
        let mean = try await readout(runner, layers: [2], mode: .meanStreams)
        #expect(residual.layers[0].values == mean.layers[0].values)
    }

    @Test func aMultiChunkPromptReadsTheLastRowOfTheLastChunk() async throws {
        let (dir, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        // 70 tokens in chunks of 32 (32 + 32 + 6) against one chunk of 128.
        let long = (0..<70).map { Int32(($0 * 37 + 11) % 1000) }
        let chunked = try await readout(runner, layers: [3], chunk: 32, prompt: long)
        let whole = try await readout(runner, layers: [3], chunk: 128, prompt: long)
        #expect(chunked.position == 69)
        #expect(whole.position == 69)
        let a = chunked.layers[0].values
        let b = whole.layers[0].values
        let scale = max(b.map { abs($0) }.max() ?? 1, 1e-3)
        let worst = zip(a, b).map { abs($0 - $1) }.max() ?? 0
        #expect(worst <= 0.05 * scale, "chunked and single-chunk rows disagree: \(worst) vs \(scale)")
    }

    @Test func theRunnerIsResetAndAnOrdinaryPrefillIsUnchangedAfterwards() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let head = try logitsBuffer(ctx)
        func ordinaryPrefill() async throws -> [Float16] {
            _ = try await runner.prefillChunked(
                tokens: prompt[...], startPosition: 0, outputMode: .logits,
                config: .production(chunkTokens: 32), into: head, onProgress: { _ in })
            return logits(head)
        }
        let before = try await ordinaryPrefill()
        #expect(runner.continuationPosition == prompt.count)

        // A shallow readout leaves layers above it with stale state; nothing may
        // survive it.
        _ = try await readout(runner, layers: [0])
        #expect(runner.continuationPosition == 0)
        #expect(runner.hiddenReadout?.plan == nil)

        let after = try await ordinaryPrefill()
        #expect(after == before)
        #expect(runner.continuationPosition == prompt.count)
    }

    @Test func aReadoutNeverWritesTheHeadOrAdvancesTheSequence() async throws {
        let (dir, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try await readout(runner, layers: [3])
        #expect(runner.continuationPosition == 0)
        let capture = try #require(runner.hiddenReadout)
        #expect(capture.unusedHead.contents().assumingMemoryBound(to: UInt8.self).pointee == 0)
    }

    @Test func refusesLayersTheModelDoesNotHaveAndDisarms() async throws {
        let (dir, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        await #expect(throws: HiddenReadoutError.self) {
            _ = try await readout(runner, layers: [4])
        }
        #expect(runner.hiddenReadout?.plan == nil)
        // And the runner still serves a valid readout.
        let result = try await readout(runner, layers: [3])
        #expect(result.layers.count == 1)
    }

    @Test func refusesAnEmptyPromptAndABatchedRunner() async throws {
        let (dir, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        await #expect(throws: GeneratorError.self) {
            _ = try await runner.readHiddenStates(
                promptIDs: [], plan: try HiddenReadoutPlan(layers: [0]),
                prefillConfig: .production(chunkTokens: 32))
        }
        #expect(runner.hiddenReadoutRefusal == nil)

        let (dir2, _, batched) = try makeRunner(slots: 2)
        defer { try? FileManager.default.removeItem(at: dir2) }
        #expect(batched.hiddenReadoutRefusal != nil)
        await #expect(throws: HiddenReadoutError.self) {
            _ = try await readout(batched, layers: [0])
        }
    }
}

/// Capture riding an ordinary prefill (`prefill_only: false`): the same hook,
/// no early stop, the head and the sequence left as an ordinary prefill leaves
/// them.
@Suite struct HiddenCaptureDuringPrefillTests {
    private func makeRunner() throws -> (URL, MetalContext, RealForwardRunner) {
        let dir = try QwenToySynthetic.write(weightBits: 4)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: dir, device: ctx.device, expecting: .qwenToy())
        let runner = try RealForwardRunner(
            model: model, context: ctx, maxContext: 128, slots: 1,
            runtimeConfiguration: try RuntimeConfiguration(forceLogitsHead: true))
        return (dir, ctx, runner)
    }

    private func head(_ ctx: MetalContext) throws -> MTLBuffer {
        guard
            let buffer = ctx.device.makeBuffer(
                length: 1024 * MemoryLayout<Float16>.stride, options: .storageModeShared)
        else { throw ModelError.residentBufferWrapFailed }
        return buffer
    }

    private func logits(_ buffer: MTLBuffer) -> [Float16] {
        Array(
            UnsafeBufferPointer(
                start: buffer.contents().bindMemory(to: Float16.self, capacity: 1024),
                count: 1024))
    }

    private let prompt: [Int32] = (0..<70).map { Int32(($0 * 37 + 11) % 1000) }
    private let config = PrefillRuntimeConfig.production(chunkTokens: 32)

    @Test func capturingDoesNotChangeTheLogitsAndMatchesThePrefillOnlyReadout() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let buffer = try head(ctx)
        func ordinary(arm plan: HiddenReadoutPlan?) async throws -> [Float16] {
            runner.reset()
            if let plan {
                try runner.armHiddenCapture(
                    plan: plan, promptEnd: prompt.count, prefillConfig: config)
            }
            _ = try await runner.prefillChunked(
                tokens: prompt[...], startPosition: 0, outputMode: .logits,
                config: config, into: buffer, onProgress: { _ in })
            return logits(buffer)
        }
        let plain = try await ordinary(arm: nil)

        let capturePlan = try HiddenReadoutPlan(layers: [0, 2, 3], prefillOnly: false)
        let captured = try await ordinary(arm: capturePlan)
        #expect(captured == plain, "capturing changed the head's logits")
        // The sequence is intact: an ordinary request would now decode on it.
        #expect(runner.continuationPosition == prompt.count)
        let result = try runner.finishHiddenCapture()
        #expect(result.position == prompt.count - 1)
        #expect(result.path == HiddenCapturePath(usedANE: false, earlyStop: false))
        #expect(runner.hiddenReadout?.plan == nil)

        // The same rows as the prefill-only readout of the same prompt.
        let probe = try await runner.readHiddenStates(
            promptIDs: prompt, plan: try HiddenReadoutPlan(layers: [0, 2, 3]),
            prefillConfig: config)
        #expect(probe.layers == result.layers)
        #expect(probe.position == result.position)
    }

    @Test func aPrefillSplitAtAFrontierBoundaryCapturesOnlyInTheFinalCall() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let buffer = try head(ctx)
        try runner.armHiddenCapture(
            plan: try HiddenReadoutPlan(layers: [3], prefillOnly: false),
            promptEnd: prompt.count, prefillConfig: config)
        // `runRawCompletion` cuts the prefill at a checkpoint boundary and runs
        // a call per piece; the first piece ends in a "last chunk" of its own.
        _ = try await runner.prefillChunked(
            tokens: prompt[0..<32], startPosition: 0, outputMode: .logits,
            config: config, into: buffer, onProgress: { _ in })
        _ = try await runner.prefillChunked(
            tokens: prompt[32...], startPosition: 32, outputMode: .logits,
            config: config, into: buffer, onProgress: { _ in })
        let split = try runner.finishHiddenCapture()
        #expect(split.position == prompt.count - 1)

        let whole = try await runner.readHiddenStates(
            promptIDs: prompt, plan: try HiddenReadoutPlan(layers: [3]), prefillConfig: config)
        #expect(split.layers == whole.layers)
    }

    @Test func aCaptureThatNeverReachesItsPromptEndIsAnInconsistencyNotARow() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let buffer = try head(ctx)
        try runner.armHiddenCapture(
            plan: try HiddenReadoutPlan(layers: [1], prefillOnly: false),
            promptEnd: prompt.count + 5, prefillConfig: config)
        _ = try await runner.prefillChunked(
            tokens: prompt[...], startPosition: 0, outputMode: .logits,
            config: config, into: buffer, onProgress: { _ in })
        #expect(throws: HiddenReadoutError.self) { _ = try runner.finishHiddenCapture() }
        #expect(runner.hiddenReadout?.plan == nil)
    }

    @Test func decodeContinuesOnACapturedPrefillAndTheCaptureCostsLittle() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let buffer = try head(ctx)
        func prefillSeconds(arm: Bool) async throws -> (Double, HiddenReadoutResult?) {
            runner.reset()
            if arm {
                try runner.armHiddenCapture(
                    plan: try HiddenReadoutPlan(layers: [0, 1, 2, 3], prefillOnly: false),
                    promptEnd: prompt.count, prefillConfig: config)
            }
            let started = ContinuousClock.now
            _ = try await runner.prefillChunked(
                tokens: prompt[...], startPosition: 0, outputMode: .logits,
                config: config, into: buffer, onProgress: { _ in })
            let elapsed = started.duration(to: .now)
            let result = arm ? try runner.finishHiddenCapture() : nil
            let seconds =
                Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
            return (seconds, result)
        }
        _ = try await prefillSeconds(arm: false)  // warm up pipelines
        var plain: [Double] = []
        var captured: [Double] = []
        var nanos: [UInt64] = []
        for _ in 0..<5 {
            plain.append(try await prefillSeconds(arm: false).0)
            let (seconds, result) = try await prefillSeconds(arm: true)
            captured.append(seconds)
            nanos.append(try #require(result).captureNanos)
        }
        let medianCapture = nanos.sorted()[nanos.count / 2]
        let line =
            String(
                format: "hidden capture, toy, 4 layers, 70 tokens: capture %.3f ms (median of 5); "
                    + "prefill %.2f ms without, %.2f ms with (medians)",
                Double(medianCapture) / 1e6,
                plain.sorted()[2] * 1e3, captured.sorted()[2] * 1e3)
        print(line)
        // Host-side cost of four blits and the wait: far below a prefill.
        #expect(medianCapture < 50_000_000)

        // Decode carries on from the captured prefill.
        runner.reset()
        try runner.armHiddenCapture(
            plan: try HiddenReadoutPlan(layers: [3], prefillOnly: false),
            promptEnd: prompt.count, prefillConfig: config)
        _ = try await runner.prefillChunked(
            tokens: prompt[...], startPosition: 0, outputMode: .logits,
            config: config, into: buffer, onProgress: { _ in })
        _ = try runner.finishHiddenCapture()
        try await runner.produce(token: 9, position: prompt.count, into: buffer)
        #expect(runner.continuationPosition == prompt.count + 1)
    }
}
