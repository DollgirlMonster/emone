import Foundation
import Metal

/// Hidden-state readout: the last prompt token's residual after chosen layers,
/// for fitting linear probes on the model's activations.
///
/// The residual lives in `PrefillChunkScratchBuffers.hidden`, which is
/// `.storageModePrivate`, so it is read by blitting the one row into a shared
/// buffer (the same pattern as `dumpActivationPrivate`, minus the per-dump
/// allocation and wait). The layer loop in `executePrefillChunk` has three
/// hooks into this file: the per-chunk plan (how many layers to run), the
/// capture after each requested layer, and the skip of the final head.
///
/// Two ways in. `readHiddenStates` is a prefill whose sequence is thrown away:
/// it stops after the deepest requested layer, so layers above it hold stale
/// KV / GDN / indexer state, and it resets the runner before returning --
/// nothing about the prompt survives it. `armHiddenCapture` /
/// `finishHiddenCapture` instead ride an ordinary prefill that the caller runs
/// itself (`runRawCompletion`): no early stop, the head and sampling follow,
/// and the runner is left exactly as an ordinary request leaves it.

/// Persistent readback state: allocated on the first readout and kept, so a
/// client replaying many prompts pays for the buffers once.
final class HiddenReadoutCapture {
    /// `numLayers` rows of `residualWidth` fp16 values, shared so the host can
    /// read it after the blits complete. Row `i` holds `plan.layers[i]`.
    let readback: MTLBuffer
    /// Handed to prefill as its logits destination; the head is skipped, so it
    /// is never written. Its own buffer, so a bug that did write it could not
    /// land on the readback.
    let unusedHead: MTLBuffer
    let rowBytes: Int

    /// Set for the duration of one readout; nil the rest of the time, which is
    /// what keeps every ordinary prefill on its unchanged path.
    var plan: HiddenReadoutPlan?
    /// One blit command buffer per captured layer, committed as the layer
    /// finishes (the next layer overwrites `hidden`) and awaited together once
    /// prefill returns.
    var pending: [MTLCommandBuffer] = []
    var captured: Set<Int> = []
    /// The absolute position one past the prompt's last token. The chunk whose
    /// end equals it is the last chunk of the prefill; a prefill the caller
    /// splits into several `prefillChunked` calls (frontier checkpoints) must
    /// capture only in the final one, which `writeFinalHead` cannot tell
    /// apart.
    var promptEnd = 0
    var capturedPosition: Int?
    var usedANE = false
    var captureNanos: UInt64 = 0

    init(device: MTLDevice, numLayers: Int, rowBytes: Int) throws {
        guard
            let readback = device.makeBuffer(
                length: numLayers * rowBytes, options: .storageModeShared),
            let unusedHead = device.makeBuffer(length: 64, options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        readback.label = "hiddenReadout.readback"
        unusedHead.label = "hiddenReadout.unusedHead"
        self.readback = readback
        self.unusedHead = unusedHead
        self.rowBytes = rowBytes
    }

    func arm(_ plan: HiddenReadoutPlan, promptEnd: Int) {
        self.plan = plan
        self.promptEnd = promptEnd
        pending.removeAll(keepingCapacity: true)
        captured.removeAll(keepingCapacity: true)
        capturedPosition = nil
        usedANE = false
        captureNanos = 0
    }

    func disarm() {
        plan = nil
        pending.removeAll(keepingCapacity: true)
        captured.removeAll(keepingCapacity: true)
        capturedPosition = nil
    }
}

extension RealForwardRunner {
    /// Why this runner cannot serve a readout, written for the client; nil when
    /// it can.
    public var hiddenReadoutRefusal: String? {
        if slots != 1 {
            return "hidden-state readout needs a single-sequence server "
                + "(this one runs \(slots) slots)"
        }
        if cfg.hyperConnections.enabled && Self.sequentialHyperConnectionPrefill {
            return "hidden-state readout is not available while "
                + "TINYTITAN_SEQUENTIAL_HC_PREFILL=1 prefills one token at a time"
        }
        return nil
    }

    /// Residual widths: streams * hiddenSize elements per token row.
    private var hiddenReadoutRowBytes: Int {
        residualWidth * MemoryLayout<Float16>.stride
    }

    private func ensureHiddenReadoutCapture() throws -> HiddenReadoutCapture {
        if let existing = hiddenReadout { return existing }
        let made = try HiddenReadoutCapture(
            device: ctx.device, numLayers: cfg.numLayers, rowBytes: hiddenReadoutRowBytes)
        hiddenReadout = made
        return made
    }

    /// Prefills `promptIDs` from an empty sequence and returns the last
    /// token's residual after each layer in `plan`.
    ///
    /// Always starts from scratch and always leaves the runner reset: the
    /// layers above `plan.deepestLayer` never ran, so the KV, GDN, PLE and
    /// indexer state describes no sequence at all. It runs on the GPU path
    /// even where the ANE would take a chunk (an early stop cannot use it), so
    /// the residual is the one the reference path makes.
    ///
    /// - Parameter onProgress: tokens prefilled so far, as `prefillChunked`
    ///   reports them.
    public func readHiddenStates(
        promptIDs: [Int32],
        plan: HiddenReadoutPlan,
        prefillConfig: PrefillRuntimeConfig,
        onProgress: (Int) -> Void = { _ in }
    ) async throws -> HiddenReadoutResult {
        guard !promptIDs.isEmpty else { throw GeneratorError.emptyPrompt }
        guard plan.prefillOnly else {
            throw HiddenReadoutError.unsupported(
                "readHiddenStates is the prefill-only readout; arm the capture instead")
        }
        try armHiddenCapture(
            plan: plan, promptEnd: promptIDs.count, prefillConfig: prefillConfig)
        defer {
            disarmHiddenCapture()
            reset()
        }
        // A readout never continues a live sequence: begin from empty so the
        // rows are those of a fresh prefill of exactly this prompt.
        await resetSequence(slot: 0)
        let capture = try ensureHiddenReadoutCapture()
        _ = try await prefillChunked(
            tokens: promptIDs[...], startPosition: 0, slot: 0,
            outputMode: .logits, config: prefillConfig,
            into: capture.unusedHead, onProgress: onProgress)
        return try finishHiddenCapture()
    }

    /// Arms a capture for the prefill the caller is about to run, ending at
    /// absolute position `promptEnd`. The capture rides that prefill with no
    /// early stop when `plan.prefillOnly` is false; `finishHiddenCapture`
    /// collects it. Pair every call with `finishHiddenCapture` or
    /// `disarmHiddenCapture`.
    public func armHiddenCapture(
        plan: HiddenReadoutPlan, promptEnd: Int, prefillConfig: PrefillRuntimeConfig
    ) throws {
        try plan.validate(numLayers: cfg.numLayers)
        if let refusal = hiddenReadoutRefusal { throw HiddenReadoutError.unsupported(refusal) }
        guard prefillConfig.mode == .chunked else {
            throw HiddenReadoutError.unsupported(
                "hidden-state readout needs chunked prefill, which is switched off")
        }
        guard promptEnd > 0 else { throw GeneratorError.emptyPrompt }
        try ensureHiddenReadoutCapture().arm(plan, promptEnd: promptEnd)
    }

    /// Drops an armed capture without collecting it (a failed request).
    public func disarmHiddenCapture() {
        hiddenReadout?.disarm()
    }

    /// Waits for the queued blits and returns the rows, disarming the capture.
    /// Throws when the prefill did not reach every requested layer on its last
    /// chunk, which would mean a prefill path the hook does not cover.
    public func finishHiddenCapture() throws -> HiddenReadoutResult {
        guard let capture = hiddenReadout, let plan = capture.plan else {
            throw HiddenReadoutError.internalInconsistency("no hidden-state capture is armed")
        }
        defer { capture.disarm() }
        guard capture.captured == Set(plan.layers), let position = capture.capturedPosition
        else {
            throw HiddenReadoutError.internalInconsistency(
                "prefill finished but captured layers \(capture.captured.sorted()) "
                    + "instead of \(plan.layers)")
        }
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // One wait for every blit: they were committed as each layer finished.
        for cb in capture.pending { try waitForCompletion(cb) }
        let elements = residualWidth
        let base = capture.readback.contents().bindMemory(
            to: Float16.self, capacity: cfg.numLayers * elements)
        let layers = plan.layers.enumerated().map { index, layer in
            HiddenReadoutLayer(
                layer: layer,
                values: HiddenReadoutMath.reduce(
                    UnsafeBufferPointer(start: base + index * elements, count: elements),
                    streams: residualStreamCount, mode: plan.streamMode))
        }
        let nanos = capture.captureNanos &+ (clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start)
        return HiddenReadoutResult(
            position: position, layers: layers,
            path: HiddenCapturePath(usedANE: capture.usedANE, earlyStop: plan.prefillOnly),
            captureNanos: nanos)
    }

    /// The chunk's plan while a readout is armed, nil otherwise. The one call
    /// `executePrefillChunk` makes per chunk.
    func hiddenReadoutChunkPlan(startPosition: Int, tokenCount: Int)
        -> HiddenReadoutPlan.ChunkPlan?
    {
        guard let capture = hiddenReadout, let plan = capture.plan else { return nil }
        let isLast = startPosition + tokenCount == capture.promptEnd
        if isLast { capture.capturedPosition = startPosition + tokenCount - 1 }
        return plan.chunkPlan(isLastChunk: isLast, numLayers: cfg.numLayers)
    }

    /// Notes that the ANE took a chunk of an armed capture's prefill.
    func noteHiddenReadoutUsedANE() {
        hiddenReadout?.usedANE = true
    }

    /// Queues the blit of the chunk's last row after `layer`. Called once the
    /// layer's tail command buffer has been awaited, so `hidden` is complete;
    /// the blit is committed now (the next layer overwrites `hidden`) but not
    /// awaited -- command buffers on one queue run in commit order and
    /// `hidden` is hazard-tracked.
    func captureHiddenReadoutRow(
        layer: Int, hidden: MTLBuffer, tokenCount t: Int
    ) throws {
        guard let capture = hiddenReadout, let plan = capture.plan,
            let index = plan.readbackIndex(forLayer: layer)
        else { return }
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let rowBytes = capture.rowBytes
        guard let cb = ctx.queue.makeCommandBuffer(),
            let blit = cb.makeBlitCommandEncoder()
        else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.copy(
            from: hidden, sourceOffset: (t - 1) * rowBytes,
            to: capture.readback, destinationOffset: index * rowBytes,
            size: rowBytes)
        blit.endEncoding()
        cb.commit()
        capture.pending.append(cb)
        capture.captured.insert(layer)
        capture.captureNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
    }
}
