import Foundation
import Metal

/// Decode without the per-layer router readback, for a model whose expert
/// cache holds every routed expert.
///
/// The ordinary routed layer stops the pipeline once: the CPU waits for the
/// router, plans the cache, binds the picked slots and encodes the expert
/// kernels. With a slot per expert and all of them resident that plan decides
/// nothing -- every pick is a hit and nothing can be evicted -- yet the wait
/// leaves the GPU idle between ~80 command buffers a token (measured
/// 2026-11-10 on Qwen 3.6 4-bit, M1 Max: 11.7 ms of a 34.2 ms token).
///
/// Here `moe_resolve_resident_experts` turns the router's ids into the routed
/// argument buffer on the GPU, through a per-layer table of expert addresses,
/// so every layer of the token is encoded and committed without a wait. The
/// kernels that compute are the same ones in the same order, so the output is
/// the ordinary path's, bit for bit. The next-layer prefetch predictor is not
/// encoded: with nothing left to fetch it has nothing to aim at.
///
/// A token takes this path only when every layer is fully resident and no
/// diagnostic needs the picks mid-token; otherwise it takes the ordinary one.
/// `EMONE_DECODE_RESIDENT=0` turns it off.
extension RealForwardRunner {
    struct ResidentLayer {
        /// The cache map's epoch this entry was built from.
        var epoch: UInt64 = 0
        /// `[numExperts]` UInt64 device addresses; nil while the layer is not
        /// fully resident.
        var addresses: MTLBuffer?
        var buffers: [MTLBuffer] = []
    }

    static let residentDecodeFlag =
        ProcessInfo.processInfo.environment["EMONE_DECODE_RESIDENT"] != "0"
    static let layerTraceFlag =
        ProcessInfo.processInfo.environment["EMONE_LAYER_TRACE"] != nil

    /// `EMONE_DECODE_CB_LAYERS=N` (default 4, read once; 0 turns it off): a
    /// resident token puts N whole layers into one command buffer --
    /// attention, tail, shared and routed stages -- instead of two buffers per
    /// layer (~80 a token on Qwen 3.6). Same kernels, same encode order, same
    /// queue, and the buffers' resources are hazard-tracked, so every kernel
    /// still sees the writes of the ones before it; only the buffer boundaries
    /// go. A value at or above the layer count makes the whole stack one buffer.
    ///
    /// Measured 2026-11-25 (M1 Max 64 GB, Qwen 3.6 35B-A3B 4-bit, warm, 256
    /// tokens, median of 5 interleaved rounds, output byte-identical): off /
    /// 4 / 10 / 40 gave 64.3 / 72.1 / 69.0 / 56.3 tok/s at short context,
    /// 57.8 / 63.1 / 60.3 / 50.0 at 8k and 44.2 / 47.0 / 45.7 / 38.8 at 32k.
    /// In-token GPU idle fell from 2.04 to 0.29 ms at 4. One buffer per token
    /// is slower than none grouped -- probably (not yet tested) because the
    /// GPU then cannot start until the whole token has been encoded.
    static let decodeLayersPerCommandBuffer: Int? = parseDecodeLayersPerCommandBuffer(
        ProcessInfo.processInfo.environment["EMONE_DECODE_CB_LAYERS"])

    static let defaultDecodeLayersPerCommandBuffer = 4

    /// Unset or unparseable: the default. 0 or negative: off (per-layer
    /// buffers, as before the switch existed).
    static func parseDecodeLayersPerCommandBuffer(_ raw: String?) -> Int? {
        guard let raw, let n = Int(raw.trimmingCharacters(in: .whitespaces))
        else { return defaultDecodeLayersPerCommandBuffer }
        return n > 0 ? n : nil
    }

    /// Layers per command buffer for a resident token, or nil for the
    /// per-layer buffers. A sparse-attention entry may commit and replace the
    /// attention buffer mid-layer, so that family keeps per-layer buffers.
    var groupedDecodeLayersPerCommandBuffer: Int? {
        guard let n = Self.decodeLayersPerCommandBuffer, qsaIndexer == nil else { return nil }
        return n
    }

    /// Why no token can take the resident path under this runner's settings,
    /// whatever the cache holds; nil when the settings allow it. The cache's
    /// own state (every expert resident or not) is `prepareResidentDecode`'s.
    var residentPathSettingsRefusal: String? {
        if !Self.residentDecodeFlag { return "EMONE_DECODE_RESIDENT=0" }
        if cfg.numExperts == 0 { return "dense model (no routed experts)" }
        if cfg.topKExperts > moe.maxStreamedExperts {
            return "top-k \(cfg.topKExperts) exceeds \(moe.maxStreamedExperts)"
        }
        if decodeExpertExecution != .hitFixup {
            return "decode expert execution is \(decodeExpertExecution), not hitFixup"
        }
        if expertIOSynchronization != .host {
            return "expert I/O synchronization is \(expertIOSynchronization), not host"
        }
        if splitKernelTiming { return "EMONE_KERNEL_SPLIT timing" }
        if activationDumpDirectory != nil { return "activation dump" }
        if prefetchTraceFD >= 0 { return "EMONE_PREFETCH_TRACE" }
        if Self.probe2TraceEnabled { return "probe-2 trace" }
        if Self.layerTraceFlag { return "EMONE_LAYER_TRACE" }
        if MLPCapture.shared.active { return "MLP capture" }
        return nil
    }

    /// Whether this token can take the resident path; refreshes the address
    /// tables of any layer whose cache map changed.
    func prepareResidentDecode(warm: Bool = true) throws -> Bool {
        residentTokenBuffers.removeAll(keepingCapacity: true)
        try reapResidentWarm(blocking: false)
        guard residentPathSettingsRefusal == nil else { return false }
        if residentLayers.count != cfg.numLayers {
            residentLayers = [ResidentLayer](repeating: ResidentLayer(), count: cfg.numLayers)
        }
        for L in 0..<cfg.numLayers {
            switch try model.routedExpertFullResidency(
                layer: L, knownEpoch: residentLayers[L].epoch)
            {
            case .unchanged:
                if residentLayers[L].addresses == nil { return false }
            case .notFull(let epoch):
                residentLayers[L] = ResidentLayer(epoch: epoch)
                if warm { try warmTowardFullResidency(from: L) }
                return false
            case .full(let epoch, let addresses, let buffers):
                let length = addresses.count * MemoryLayout<UInt64>.stride
                guard
                    let table = residentLayers[L].addresses
                        ?? ctx.device.makeBuffer(length: length, options: .storageModeShared)
                else { throw MetalError.bufferAllocationFailed("decode.residentAddresses") }
                addresses.withUnsafeBytes { raw in
                    if let base = raw.baseAddress { memcpy(table.contents(), base, length) }
                }
                residentLayers[L] = ResidentLayer(
                    epoch: epoch, addresses: table, buffers: buffers)
            }
        }
        if residentArgBuffer == nil {
            // The shader's `RoutedBlobs` holds 16 pointers whatever top-k is.
            residentArgBuffer = ctx.device.makeBuffer(
                length: 16 * MoE.routedArgumentPointerStride, options: .storageModeShared)
            residentRouteLog = ctx.device.makeBuffer(
                length: cfg.numLayers * cfg.topKExperts * MemoryLayout<UInt32>.stride,
                options: .storageModeShared)
        }
        return residentArgBuffer != nil && residentRouteLog != nil
    }

    /// Reads toward a complete cache, for a model whose cache has a slot per
    /// expert (any other is left alone).
    ///
    /// When few experts are missing -- after a prefill, the handful per layer
    /// it never routed to -- they are read here and the next token is
    /// resident. A cold cache is filled in the background instead: a window
    /// of reads is kept in flight across tokens, topped up at each token's
    /// start and never waited on, so the drive stays busy while the tokens
    /// decode on the ordinary path. (Waiting for two plans at each token's
    /// end filled 32 experts a token, about a quarter of what the drive
    /// delivers; reading on the token's own time cost 11% of cold decode.)
    static let residentWarmInline = 512
    /// Plans in flight (16 experts each). 6, 12 and 24 measured alike on Qwen
    /// 3.6 4-bit (40.1 / 40.7 / 40.2 tok/s over a cold 1,024 tokens, 2 at
    /// 38.7): past the drive's rate a wider window only pins more slots.
    static let residentWarmWindow = 6

    func warmTowardFullResidency(from firstLayer: Int) throws {
        guard let slots = model.routedExpertCacheSlotCount(), slots >= cfg.numExperts else {
            return
        }
        var missing = 0
        for layer in firstLayer..<cfg.numLayers {
            missing += try model.missingRoutedExpertCount(layer: layer)
            if missing > Self.residentWarmInline { break }
        }
        var layer = firstLayer
        if missing <= Self.residentWarmInline {
            try reapResidentWarm(blocking: true)
            while layer < cfg.numLayers {
                let loaded = try model.warmRoutedExperts(layer: layer, limit: 16)
                if loaded == 0 { layer += 1 }
                totalResidentWarmedExperts &+= UInt64(loaded)
            }
            return
        }
        while residentWarmOperations.count < Self.residentWarmWindow,
            layer < cfg.numLayers
        {
            if let warm = try model.beginWarmRoutedExperts(layer: layer, limit: 16) {
                residentWarmOperations.append(warm)
                totalResidentWarmedExperts &+= UInt64(warm.operation.plan.misses.count)
            } else {
                layer += 1
            }
        }
    }

    /// Unpins the slots of warm reads that have finished, and surfaces their
    /// errors. `blocking` waits for the ones still in flight as well, which
    /// leaves no slot loading.
    func reapResidentWarm(blocking: Bool) throws {
        guard !residentWarmOperations.isEmpty else { return }
        let operations = residentWarmOperations
        residentWarmOperations.removeAll(keepingCapacity: true)
        var failure: Error?
        for warm in operations {
            let state = warm.operation.state
            if !blocking, state == .submitted || state == .inFlight {
                residentWarmOperations.append(warm)
                continue
            }
            do { _ = try warm.operation.wait() } catch { failure = failure ?? error }
            warm.lease.release()
        }
        if let failure { throw failure }
    }

    func drainResidentWarm() throws { try reapResidentWarm(blocking: true) }

    static let residentIdleWarmPlans = 12

    /// Fills the routed-expert cache while no request is running, so the
    /// first reply after a load decodes resident instead of paying for the
    /// read (on Qwen 3.6 4-bit about 14 GB, ~7 s of a cold reply). A no-op
    /// for a model whose cache cannot hold every expert.
    ///
    /// The step gate is held a batch at a time -- twelve plans, a fraction of
    /// a second -- so a request that arrives waits for one batch at most.
    /// Once a prefill or decode step has run, the warm stops: decode fills
    /// the rest itself (`warmTowardFullResidency`), and the caller may start
    /// this again when the request is over.
    public func warmResidentExpertsWhileIdle() async {
        guard Self.residentDecodeFlag, cfg.numExperts > 0,
            let slots = model.routedExpertCacheSlotCount(), slots >= cfg.numExperts
        else { return }
        var activityAtStart: UInt64?
        var layer = 0
        while !Task.isCancelled {
            do { try await forwardStepGate.acquire() } catch { return }
            var finished = activityAtStart.map { $0 != forwardActivityCount } ?? false
            activityAtStart = forwardActivityCount
            if !finished {
                // A failed read ends the warm quietly: the token that needs
                // the expert reads it again and reports the error there.
                finished = (try? warmIdleBatch(from: &layer)) ?? true
            }
            await forwardStepGate.release()
            if finished { return }
            await Task.yield()
        }
    }

    /// One batch of the idle warm, read concurrently and waited on here.
    /// Returns whether every layer is complete. Call under the step gate.
    func warmIdleBatch(from layer: inout Int) throws -> Bool {
        try reapResidentWarm(blocking: true)
        while residentWarmOperations.count < Self.residentIdleWarmPlans, layer < cfg.numLayers {
            if let warm = try model.beginWarmRoutedExperts(layer: layer, limit: 16) {
                residentWarmOperations.append(warm)
                totalResidentWarmedExperts &+= UInt64(warm.operation.plan.misses.count)
            } else {
                layer += 1
            }
        }
        try reapResidentWarm(blocking: true)
        return layer >= cfg.numLayers
    }

    /// The routed stage of one layer with no readback: resolve, phase 1,
    /// phase 2 and the residual exit, encoded into the layer's tail buffer
    /// (behind its router and shared MLP), which is committed here. The
    /// layer's buffers are kept for the end-of-token check.
    func encodeResidentRoutedMoE(
        layer L: Int,
        attnCB: MTLCommandBuffer?, softmaxCB: MTLCommandBuffer?,
        tailCB routedCB: MTLCommandBuffer,
        commit: Bool = true
    ) throws {
        guard L < residentLayers.count, let addresses = residentLayers[L].addresses,
            let argBuffer = residentArgBuffer, let routeLog = residentRouteLog
        else { throw ModelError.residentBufferWrapFailed }
        let D = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let topK = UInt32(cfg.topKExperts)
        let slots = residentLayers[L].buffers
        let routedOffsets = try model.routedExpertOffsets(layer: L)
        try moe.encodeResolveResidentExperts(
            commandBuffer: routedCB,
            topKIndices: outIndices, expertAddresses: addresses,
            routedArgBuffer: argBuffer, routeLog: routeLog,
            routeLogOffset: L * cfg.topKExperts * MemoryLayout<UInt32>.stride,
            topK: topK, numExperts: UInt32(cfg.numExperts))
        try moe.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: routedCB,
            routedArgBuffer: argBuffer, routedBlobs: [], routedOffsets: routedOffsets,
            x: routedX, acts: moeActs, d: D, f: FmoE, topK: topK,
            allResident: slots)
        try moe.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: routedCB,
            routedArgBuffer: argBuffer, routedBlobs: [], routedOffsets: routedOffsets,
            acts: moeActs, routingWeights: outWeights,
            residual: h1Buf, y: h2Buf, d: D, f: FmoE, topK: topK,
            allResident: slots)
        try encodeResidualExitDecode(
            commandBuffer: routedCB, hidden: hidden, delta: h2Buf, sublayer: .mlp, layer: L)
        // Grouped (`EMONE_DECODE_CB_LAYERS`): the buffer holds several whole
        // layers and is committed, and later checked, once, after its last.
        guard commit else { return }
        routedCB.commitEncoded()
        if let attnCB { residentTokenBuffers.append(("attn_norm_qkv", attnCB)) }
        if let softmaxCB { residentTokenBuffers.append(("attn_softmax", softmaxCB)) }
        residentTokenBuffers.append(
            (attnCB == nil ? "decode_layer_group" : "tail_router_shared_routed", routedCB))
    }

    /// Once the token's last buffer has landed: surface any command error,
    /// record the GPU spans, and do from the id log what the cache plan and
    /// the traces did per layer.
    func finishResidentToken(position: Int) throws {
        guard let last = residentTokenBuffers.last else { return }
        let waitStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try waitForCompletion(last.cb)
        totalWaitNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - waitStart
        defer { residentTokenBuffers.removeAll(keepingCapacity: true) }
        for (role, cb) in residentTokenBuffers {
            if let error = cb.error {
                throw ModelError.commandBufferFailed(detail: "\(role) command buffer: \(error)")
            }
            recordKernelGPU(role: role, cb)
        }
        if CommandBufferGapTrace.enabled {
            CommandBufferGapTrace.record(residentTokenBuffers.map(\.cb))
        }
        guard let routeLog = residentRouteLog else { return }
        let k = cfg.topKExperts
        let ids = routeLog.contents().bindMemory(to: UInt32.self, capacity: cfg.numLayers * k)
        for L in 0..<cfg.numLayers {
            let experts = (0..<k).map { Int(ids[L * k + $0]) }
            try model.recordRoutedResidentUse(layer: L, experts: experts)
            recordRouteTrace(layer: L, position: position, experts: experts)
            #if !EMONE_NO_EXPERT_TRACE
                expertTrace?.record(
                    layer: L, position: position, experts: experts, missIndices: [])
            #endif
        }
        totalResidentTokens &+= 1
    }
}

/// `EMONE_CB_GAP_TRACE=1` (default off, read once): for every resident decode
/// token, the GPU idle between its layer command buffers, and how much of
/// that idle is the GPU waiting for the CPU to hand it the next buffer. One
/// line goes to stderr at exit.
///
/// For consecutive buffers a then b, the gap is `b.gpuStartTime -
/// a.gpuEndTime`. The CPU-late part is how far `b.kernelStartTime` (when the
/// host began scheduling b, i.e. just after its commit) lies past
/// `a.gpuEndTime`, capped at the gap: idle the GPU spent with nothing queued.
/// The rest is the cost of crossing a buffer boundary with b already queued.
enum CommandBufferGapTrace {
    static let enabled: Bool = {
        let on = ProcessInfo.processInfo.environment["EMONE_CB_GAP_TRACE"] == "1"
        if on { atexit { CommandBufferGapTrace.report() } }
        return on
    }()

    nonisolated(unsafe) static var tokens = 0
    nonisolated(unsafe) static var buffers = 0
    nonisolated(unsafe) static var gapSeconds = 0.0
    nonisolated(unsafe) static var cpuLateSeconds = 0.0
    nonisolated(unsafe) static var busySeconds = 0.0
    nonisolated(unsafe) static var spanSeconds = 0.0

    /// The token's buffers in commit order; a buffer listed twice counts once.
    static func record(_ list: [MTLCommandBuffer]) {
        var seen = Set<ObjectIdentifier>()
        let cbs = list.filter { seen.insert(ObjectIdentifier($0)).inserted }
        guard let first = cbs.first, let last = cbs.last, first.gpuEndTime > 0 else { return }
        let summary = summarize(
            cbs.map { (schedule: $0.kernelStartTime, start: $0.gpuStartTime, end: $0.gpuEndTime) })
        tokens += 1
        buffers += cbs.count
        gapSeconds += summary.gap
        cpuLateSeconds += summary.cpuLate
        busySeconds += summary.busy
        spanSeconds += last.gpuEndTime - first.gpuStartTime
    }

    /// Pure arithmetic over (scheduled, start, end) host times, for tests.
    static func summarize(
        _ spans: [(schedule: Double, start: Double, end: Double)]
    ) -> (gap: Double, cpuLate: Double, busy: Double) {
        var gap = 0.0
        var late = 0.0
        var busy = 0.0
        for (i, s) in spans.enumerated() {
            busy += max(0, s.end - s.start)
            guard i > 0 else { continue }
            let g = max(0, s.start - spans[i - 1].end)
            gap += g
            late += min(g, max(0, s.schedule - spans[i - 1].end))
        }
        return (gap, late, busy)
    }

    static func report() {
        guard tokens > 0 else {
            FileHandle.standardError.write(Data("cb gaps: no resident tokens\n".utf8))
            return
        }
        let t = Double(tokens)
        let line = String(
            format: "cb gaps: %d tokens, %.1f buffers/token, layer span %.3f ms/token, "
                + "busy %.3f, idle %.3f (cpu-late %.3f, boundary %.3f) ms/token\n",
            tokens, Double(buffers) / t, spanSeconds / t * 1e3, busySeconds / t * 1e3,
            gapSeconds / t * 1e3, cpuLateSeconds / t * 1e3,
            (gapSeconds - cpuLateSeconds) / t * 1e3)
        FileHandle.standardError.write(Data(line.utf8))
    }
}
