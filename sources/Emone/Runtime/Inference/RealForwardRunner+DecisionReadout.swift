import Foundation
import Metal

/// Decision readout: the *final-normed* hidden state at chosen prompt
/// positions, for a System One decision model (Kev) whose pointer head scores
/// options from those rows instead of sampling from the LM head.
///
/// Different from the hidden-state readout (`+HiddenCapture`) in three ways,
/// each because the pointer head needs it: the rows are normed with the
/// model's final norm (what `AutoModel.last_hidden_state` returns), they are
/// at any positions of the prefill rather than only its last token, and every
/// layer runs -- the readout rides an ordinary prefill, so the KV and GDN
/// state it leaves is valid to branch from. Only the LM head is skipped.
///
/// The hook in `executePrefillChunk` runs once per chunk after the last layer:
/// one multi-row RMSNorm over the chunk into `scratch.normed`, then one blit
/// of the requested rows into a shared buffer, in one command buffer that is
/// committed and not awaited (the next chunk overwrites `hidden`, and one
/// queue runs in commit order). `finishDecisionReadout` waits once.
final class DecisionReadoutCapture {
    private(set) var readback: MTLBuffer
    let rowBytes: Int
    /// Absolute positions, in the order the caller asked for them.
    var positions: [Int] = []
    var pending: [MTLCommandBuffer] = []
    var filled: Set<Int> = []
    var armed = false
    /// Layers whose raw residual rows (before the final norm) are also read
    /// at `positions`, for heads fitted on middle blocks (Emone-lmw.15).
    /// Row `l * positions.count + i` of `blockReadback` is layer
    /// `blockLayers[l]` at position `i`.
    var blockLayers: [Int] = []
    private(set) var blockReadback: MTLBuffer?
    var blockFilled: Set<Int> = []
    /// Run the LM head as usual, so the same prefill also yields the last
    /// position's logits (a zero-shot letter readout beside the rows).
    var keepHead = false

    init(device: MTLDevice, rowBytes: Int, capacityRows: Int) throws {
        guard
            let buffer = device.makeBuffer(
                length: max(1, capacityRows) * rowBytes, options: .storageModeShared)
        else { throw ModelError.residentBufferWrapFailed }
        buffer.label = "decisionReadout.readback"
        self.readback = buffer
        self.rowBytes = rowBytes
    }

    func ensureCapacity(rows: Int, device: MTLDevice) throws {
        guard rows * rowBytes > readback.length else { return }
        guard
            let buffer = device.makeBuffer(
                length: rows * rowBytes, options: .storageModeShared)
        else { throw ModelError.residentBufferWrapFailed }
        buffer.label = "decisionReadout.readback"
        readback = buffer
    }

    func ensureBlockCapacity(rows: Int, device: MTLDevice) throws {
        guard rows > 0 else { return }
        if let blockReadback, blockReadback.length >= rows * rowBytes { return }
        guard
            let buffer = device.makeBuffer(
                length: rows * rowBytes, options: .storageModeShared)
        else { throw ModelError.residentBufferWrapFailed }
        buffer.label = "decisionReadout.blockReadback"
        blockReadback = buffer
    }
}

/// The rows a decision readout returns: final-normed rows at each position,
/// and, when blocks were requested, the raw residual after each of them.
public struct DecisionReadoutRows: Sendable {
    /// `[position]` rows, final-normed.
    public let final: [[Float]]
    /// `[layer][position]` rows, in the order the layers were requested.
    public let blocks: [[[Float]]]
}

/// A point in a sequence that later prefills can return to: the KV cursor and
/// a GPU-side copy of every Gated-DeltaNet state and conv tail. Returning is a
/// cursor move plus one blit (about 19 MiB for a 24-layer 0.8B), not the host
/// round trip `restoreInferenceState` makes, because attention KV rows are
/// position-addressed and append-only: rows written past the point are simply
/// overwritten by the next branch.
///
/// unchecked-invariant: the buffers are written once by the runner that made
/// the point (inside its forward-step gate) and only read afterwards, by that
/// same runner, also inside the gate; every stored property is a `let`.
public final class DecisionBranchPoint: @unchecked Sendable {
    public let position: Int
    let runnerID: ObjectIdentifier
    let states: [MTLBuffer?]
    let tails: [MTLBuffer?]

    init(position: Int, runnerID: ObjectIdentifier, states: [MTLBuffer?], tails: [MTLBuffer?]) {
        self.position = position
        self.runnerID = runnerID
        self.states = states
        self.tails = tails
    }
}

public enum DecisionReadoutError: Error, CustomStringConvertible, Equatable {
    case unsupported(String)
    case internalInconsistency(String)

    public var description: String {
        switch self {
        case .unsupported(let reason), .internalInconsistency(let reason): reason
        }
    }
}

extension RealForwardRunner {
    /// Why this runner cannot serve a decision readout; nil when it can. The
    /// branch point covers KV and Gated-DeltaNet state only, so any other
    /// per-sequence state (PLE window, QSA indexer, the ANE's shadow tokens)
    /// would silently desync on a rewind.
    public var decisionReadoutRefusal: String? {
        if slots != 1 { return "decision readout needs a single-slot runner" }
        if kv == nil { return "decision readout needs a KV cache" }
        if residualStreamCount != 1 { return "decision readout reads one residual stream" }
        if pleBlock != nil { return "decision readout cannot branch a PLE window" }
        if qsaIndexer != nil { return "decision readout cannot branch a QSA indexer" }
        if anePrefill != nil { return "decision readout cannot branch ANE prefill state" }
        return nil
    }

    private var decisionRowBytes: Int { cfg.hiddenSize * MemoryLayout<Float16>.stride }

    /// Arms a readout of the final-normed rows at `positions` (absolute,
    /// within the prefill the caller is about to run). Pair with
    /// `finishDecisionReadout` or `disarmDecisionReadout`.
    ///
    /// `blockLayers` also reads the raw residual after each of those layers
    /// at the same positions (`finishDecisionReadoutRows`). `keepHead` runs
    /// the LM head as an ordinary prefill would, so the caller's logits
    /// buffer receives the last position's logits too.
    public func armDecisionReadout(
        positions: [Int], blockLayers: [Int] = [], keepHead: Bool = false
    ) throws {
        if let refusal = decisionReadoutRefusal { throw DecisionReadoutError.unsupported(refusal) }
        guard !positions.isEmpty, positions.allSatisfy({ $0 >= 0 }) else {
            throw DecisionReadoutError.unsupported("no readout positions")
        }
        guard blockLayers.allSatisfy({ (0..<cfg.numLayers).contains($0) }),
            Set(blockLayers).count == blockLayers.count
        else {
            throw DecisionReadoutError.unsupported(
                "block layers must be distinct and in 0..<\(cfg.numLayers): \(blockLayers)")
        }
        let capture: DecisionReadoutCapture
        if let existing = decisionReadout {
            capture = existing
            try capture.ensureCapacity(rows: positions.count, device: ctx.device)
        } else {
            capture = try DecisionReadoutCapture(
                device: ctx.device, rowBytes: decisionRowBytes,
                capacityRows: max(positions.count, 64))
            decisionReadout = capture
        }
        try capture.ensureBlockCapacity(
            rows: blockLayers.count * positions.count, device: ctx.device)
        capture.positions = positions
        capture.blockLayers = blockLayers
        capture.keepHead = keepHead
        capture.pending.removeAll(keepingCapacity: true)
        capture.filled.removeAll(keepingCapacity: true)
        capture.blockFilled.removeAll(keepingCapacity: true)
        capture.armed = true
    }

    public func disarmDecisionReadout() {
        decisionReadout?.armed = false
        decisionReadout?.pending.removeAll(keepingCapacity: true)
    }

    /// True while a readout is armed.
    var decisionReadoutArmed: Bool { decisionReadout?.armed == true }

    /// True while an armed readout stands in for the LM head, which
    /// `executePrefillChunk` then skips.
    var decisionReadoutSkipsHead: Bool {
        decisionReadout?.armed == true && decisionReadout?.keepHead != true
    }

    /// The per-layer hook, after layer `layer` has finished (its stage awaits
    /// its own command buffers, so `hidden` is complete): blits the requested
    /// positions' residual rows out, committed and not awaited, as the hidden
    /// readout does.
    func captureDecisionBlockRows(
        layer: Int, hidden: MTLBuffer, startPosition: Int, tokenCount t: Int
    ) throws {
        guard let capture = decisionReadout, capture.armed,
            let slot = capture.blockLayers.firstIndex(of: layer),
            let readback = capture.blockReadback
        else { return }
        let wanted = capture.positions.enumerated().filter {
            $0.element >= startPosition && $0.element < startPosition + t
        }
        guard !wanted.isEmpty else { return }
        let rowBytes = capture.rowBytes
        guard let cb = ctx.queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder()
        else { throw MetalError.commandEncoderFailed }
        cb.label = "decisionReadout.block"
        for (index, position) in wanted {
            let row = slot * capture.positions.count + index
            blit.copy(
                from: hidden, sourceOffset: (position - startPosition) * rowBytes,
                to: readback, destinationOffset: row * rowBytes, size: rowBytes)
            capture.blockFilled.insert(row)
        }
        blit.endEncoding()
        cb.commit()
        capture.pending.append(cb)
    }

    /// The per-chunk hook, after the last layer. Norms the whole chunk (one
    /// dispatch; per-row dispatches cost more than the rows a 32-token branch
    /// does not need) and blits the requested rows out.
    func captureDecisionRows(
        scratch: PrefillChunkScratchBuffers, startPosition: Int, tokenCount t: Int, eps: Float
    ) throws {
        guard let capture = decisionReadout, capture.armed else { return }
        let wanted = capture.positions.enumerated().filter {
            $0.element >= startPosition && $0.element < startPosition + t
        }
        guard !wanted.isEmpty else { return }
        let norm = try model.finalNorm()
        let rowBytes = capture.rowBytes
        guard let cb = ctx.queue.makeCommandBuffer() else { throw MetalError.commandEncoderFailed }
        cb.label = "decisionReadout.norm"
        try prefillRMS.encodeBF16W(
            commandBuffer: cb, x: scratch.hidden, weight: norm.buffer,
            weightOffset: Int(norm.offset), out: scratch.normed,
            t: UInt32(t), d: UInt32(cfg.hiddenSize), eps: eps)
        guard let blit = cb.makeBlitCommandEncoder() else { throw MetalError.commandEncoderFailed }
        for (index, position) in wanted {
            blit.copy(
                from: scratch.normed, sourceOffset: (position - startPosition) * rowBytes,
                to: capture.readback, destinationOffset: index * rowBytes, size: rowBytes)
            capture.filled.insert(index)
        }
        blit.endEncoding()
        cb.commit()
        capture.pending.append(cb)
    }

    /// Waits for the queued readout and returns one row per requested
    /// position, widened to Float, disarming the capture.
    public func finishDecisionReadout() throws -> [[Float]] {
        try finishDecisionReadoutRows().final
    }

    /// `finishDecisionReadout`, plus the block rows the readout was armed for.
    public func finishDecisionReadoutRows() throws -> DecisionReadoutRows {
        guard let capture = decisionReadout, capture.armed else {
            throw DecisionReadoutError.internalInconsistency("no decision readout is armed")
        }
        defer { disarmDecisionReadout() }
        for cb in capture.pending { try waitForCompletion(cb) }
        let count = capture.positions.count
        guard capture.filled.count == count,
            capture.blockFilled.count == capture.blockLayers.count * count
        else {
            let missing = capture.positions.indices.filter { !capture.filled.contains($0) }
            throw DecisionReadoutError.internalInconsistency(
                "prefill did not reach readout positions \(missing.map { capture.positions[$0] })"
                    + " (block rows \(capture.blockFilled.count) of "
                    + "\(capture.blockLayers.count * count))")
        }
        let width = cfg.hiddenSize
        func rows(_ buffer: MTLBuffer, first: Int) -> [[Float]] {
            let base = buffer.contents().bindMemory(
                to: Float16.self, capacity: (first + count) * width)
            return (0..<count).map { index in
                (0..<width).map { Float(base[(first + index) * width + $0]) }
            }
        }
        let blocks: [[[Float]]] =
            capture.blockReadback.map { buffer in
                capture.blockLayers.indices.map { rows(buffer, first: $0 * count) }
            } ?? []
        return DecisionReadoutRows(final: rows(capture.readback, first: 0), blocks: blocks)
    }

    // MARK: - Branch points

    /// Records the current end of the sequence as a point to return to. The
    /// copy is a GPU blit, awaited here so the point is complete when returned.
    public func makeDecisionBranchPoint() async throws -> DecisionBranchPoint {
        if let refusal = decisionReadoutRefusal { throw DecisionReadoutError.unsupported(refusal) }
        try await forwardStepGate.acquire()
        do {
            let point = try encodeBranchCopy(saving: true, into: nil)
            await forwardStepGate.release()
            return point
        } catch {
            await forwardStepGate.release()
            throw error
        }
    }

    /// Returns the sequence to `point`: KV cursor rewound, GDN state and conv
    /// tails copied back. Afterwards the runner continues from
    /// `point.position` exactly as it did when the point was made.
    public func restoreDecisionBranchPoint(_ point: DecisionBranchPoint) async throws {
        guard point.runnerID == ObjectIdentifier(self) else {
            throw DecisionReadoutError.unsupported("branch point belongs to another runner")
        }
        try await forwardStepGate.acquire()
        do {
            guard let kv else { throw DecisionReadoutError.unsupported("no KV cache") }
            _ = try encodeBranchCopy(saving: false, into: point)
            try kv.rewind(slot: 0, to: point.position)
            resetTransientState()
            await forwardStepGate.release()
        } catch {
            await forwardStepGate.release()
            throw error
        }
    }

    /// One blit pass between the live GDN buffers and a branch point's copies,
    /// in either direction. Saving allocates the copies (private storage: only
    /// the GPU reads them).
    private func encodeBranchCopy(
        saving: Bool, into existing: DecisionBranchPoint?
    ) throws -> DecisionBranchPoint {
        guard let kv else { throw DecisionReadoutError.unsupported("no KV cache") }
        var states = existing?.states ?? []
        var tails = existing?.tails ?? []
        guard let cb = ctx.queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder()
        else { throw MetalError.commandEncoderFailed }
        cb.label = saving ? "decisionBranch.save" : "decisionBranch.restore"
        if let gdnState {
            for layer in 0..<cfg.numLayers {
                guard cfg.layerIsLinear(layer) else {
                    if saving {
                        states.append(nil)
                        tails.append(nil)
                    }
                    continue
                }
                let liveState = gdnState.stateBuffer(layer: layer)
                let liveTail = gdnState.convTailBuffer(layer: layer)
                if saving {
                    guard
                        let state = ctx.device.makeBuffer(
                            length: gdnState.stateBytesPerLayer, options: .storageModePrivate),
                        let tail = ctx.device.makeBuffer(
                            length: gdnState.convTailBytesPerLayer, options: .storageModePrivate)
                    else { throw ModelError.residentBufferWrapFailed }
                    blit.copy(
                        from: liveState, sourceOffset: 0, to: state, destinationOffset: 0,
                        size: gdnState.stateBytesPerLayer)
                    blit.copy(
                        from: liveTail, sourceOffset: 0, to: tail, destinationOffset: 0,
                        size: gdnState.convTailBytesPerLayer)
                    states.append(state)
                    tails.append(tail)
                } else if let state = states[layer], let tail = tails[layer] {
                    blit.copy(
                        from: state, sourceOffset: 0, to: liveState, destinationOffset: 0,
                        size: gdnState.stateBytesPerLayer)
                    blit.copy(
                        from: tail, sourceOffset: 0, to: liveTail, destinationOffset: 0,
                        size: gdnState.convTailBytesPerLayer)
                }
            }
        }
        blit.endEncoding()
        cb.commit()
        try waitForCompletion(cb)
        return existing
            ?? DecisionBranchPoint(
                position: kv.position(slot: 0), runnerID: ObjectIdentifier(self),
                states: states, tails: tails)
    }
}
