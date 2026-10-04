import Foundation
import Metal

struct PrefillAttentionParams: Sendable, Equatable {
    var startPosition: UInt32
    var queryCount: UInt32
    var headDim: UInt32
    var numQHeads: UInt32
    var numKVHeads: UInt32
    var kvValidCount: UInt32
    var slidingWindow: UInt32
    var kvTokenStrideElements: UInt32
    var qTokenStrideElements: UInt32
    var oTokenStrideElements: UInt32
    var scale: Float
    var kvBits: UInt32
    var kvTokenStrideBytes: UInt32
    var kvValueBytes: UInt32
    var kvGroupSize: UInt32

    init(
        startPosition: UInt32,
        queryCount: UInt32,
        headDim: UInt32,
        numQHeads: UInt32,
        numKVHeads: UInt32,
        kvValidCount: UInt32,
        slidingWindow: UInt32,
        kvTokenStrideElements: UInt32,
        qTokenStrideElements: UInt32,
        oTokenStrideElements: UInt32,
        scale: Float,
        kvBits: UInt32 = 16,
        kvTokenStrideBytes: UInt32 = 0,
        kvValueBytes: UInt32 = 0,
        kvGroupSize: UInt32 = UInt32(KVCacheManager.quantizationGroupSize)
    ) {
        self.startPosition = startPosition
        self.queryCount = queryCount
        self.headDim = headDim
        self.numQHeads = numQHeads
        self.numKVHeads = numKVHeads
        self.kvValidCount = kvValidCount
        self.slidingWindow = slidingWindow
        self.kvTokenStrideElements = kvTokenStrideElements
        self.qTokenStrideElements = qTokenStrideElements
        self.oTokenStrideElements = oTokenStrideElements
        self.scale = scale
        self.kvBits = kvBits
        self.kvTokenStrideBytes = kvTokenStrideBytes
        self.kvValueBytes = kvValueBytes
        self.kvGroupSize = kvGroupSize
    }
}

enum PrefillAttentionError: Error, CustomStringConvertible {
    case tensorOpsUnavailable(reason: String)
    case commandEncoderFailed

    public var description: String {
        switch self {
        case .tensorOpsUnavailable(let reason):
            return "TensorOps 2D prefill attention requested but unavailable: \(reason)"
        case .commandEncoderFailed:
            return "Failed to create Metal compute command encoder"
        }
    }
}

final class PrefillAttention {
    private let context: MetalContext
    private let psoCausalTiled: MTLComputePipelineState
    private let psoCausalQSATiled: MTLComputePipelineState?
    /// `attention_prefill_causal_qsa_gqa`: the QSA kernel with one threadgroup
    /// per (token, KV head) serving all of that KV head's query heads, so each
    /// selected K/V row is read once rather than once per query head.
    private let psoCausalQSAGQA: MTLComputePipelineState?
    /// The grouped QSA kernel on the simdgroup matrix units
    /// (`attention_prefill_causal_qsa_gqa_mma`), head dim 256 only.
    private let psoCausalQSAGQAMMA: MTLComputePipelineState?

    /// `attention_prefill_qsa_masked_flash` and its tile-flag pass: QSA as
    /// dense flash tiles under the selection mask, head dim 256 only.
    private let psoQSAMaskedFlash: MTLComputePipelineState?
    private let psoQSAFlashTileFlags: MTLComputePipelineState?
    /// Packed mode (`FC_QSA_FLASH_PACKED`) and its per-row-tile block lists.
    private let psoQSAFlashPacked: MTLComputePipelineState?
    private let psoQSAGroupBlocks: MTLComputePipelineState?
    private var packedBlockLists: MTLBuffer?
    private var packedBlockCounts: MTLBuffer?
    /// Walk each row tile's selected 4-key blocks, four to a tile, instead of
    /// whole 16-key tiles. Set from `ModelProfile.prefillQSAPacked`.
    var packedFlash = false
    /// One byte per (8-row tile, 16-key tile) of a chunk; grown on demand.
    private var flashTileFlags: MTLBuffer?
    static let flashRows = 8
    static let flashKeys = 16
    /// Run a selection through the masked flash kernel instead of the
    /// gathered grouped kernels. Set from `ModelProfile.prefillQSAFlash`.
    var maskedFlash = false
    /// Run causal attention with no selection through the same flash kernel
    /// (every key up to the row's position kept, no tile skipped) instead of
    /// `attention_prefill_causal_tiled`. Set from `ModelProfile.prefillDenseFlash`.
    var denseFlash = false

    /// Whether the matrix-unit QSA kernel compiled on this device.
    var hasQSAMatrixUnitKernel: Bool { psoCausalQSAGQAMMA != nil }

    /// The grouped QSA kernel's products on the matrix units, on by default;
    /// `TINYTITAN_PREFILL_QSA_MMA=0` restores the scalar grouped kernel. It sums
    /// in a different order (and rescales per 16-key tile rather than 128), so
    /// the output changes by rounding; the surprisal A/B found no measurable
    /// change (docs/m1-prefill-spike.md, spike 10), and it cut the attention
    /// layers' GPU time from 72 s to 45.5 s on a ~17K-token M1 Max prefill.
    static let qsaMatrixUnits =
        ProcessInfo.processInfo.environment["TINYTITAN_PREFILL_QSA_MMA"] != "0"

    /// The grouped QSA kernel, on by default. It keeps the per-head kernel's
    /// arithmetic, so the output is byte-identical
    /// (`PrefillAttentionQSAGroupedTests`, and Qwen3.8 4-bit end to end on an
    /// M1 Max, three rounds), and it halved `attn_core` there (54.2 -> 28.7 s
    /// per 7.9K-token prefill). `TINYTITAN_PREFILL_QSA_GQA=0` restores the
    /// per-head kernel for an A/B on one build.
    static let qsaGroupedQueryHeads =
        ProcessInfo.processInfo.environment["TINYTITAN_PREFILL_QSA_GQA"] != "0"
    static let qsaGQAMaxGroup = 16
    static let qsaGQAMaxHeadDim = 256
    /// One byte, bound whenever no selection is in play; `useKeep` is what
    /// actually turns the mask off.
    private let emptyKeepMask: MTLBuffer
    private let psoFullTensorOps2DValidityV2: MTLComputePipelineState?
    /// K7: recorded once at init so an explicit TensorOps path request can
    /// throw the real reason instead of a bare `preconditionFailure`.
    private let tensorOpsUnavailableReason: String

    init(context: MetalContext) throws {
        self.context = context
        self.psoCausalTiled = try context.pipeline("attention_prefill_causal_tiled")
        // Tile-synchronised variant: one barrier per tile of keys instead of
        // one per key. TINYTITAN_QSA_TILED=0 falls back for A/B on one build.
        self.psoCausalQSATiled =
            (try? context.pipeline(
                "attention_prefill_causal_qsa_tiled"))
        self.psoCausalQSAGQA =
            (try? context.pipeline(
                "attention_prefill_causal_qsa_gqa"))
        self.psoCausalQSAGQAMMA =
            (try? context.pipeline(
                "attention_prefill_causal_qsa_gqa_mma"))
        self.psoQSAMaskedFlash =
            (try? context.pipeline("attention_prefill_qsa_masked_flash"))
        self.psoQSAFlashTileFlags =
            (try? context.pipeline("qsa_flash_tile_flags"))
        self.psoQSAFlashPacked = try? context.pipeline(
            "attention_prefill_qsa_masked_flash",
            constants: [MetalFunctionConstant(index: 111, value: .bool(true))])
        self.psoQSAGroupBlocks = try? context.pipeline("qsa_flash_group_blocks")
        // Four bytes, not one: the kernels declare `keepIdx`/`keepIndices` as
        // `device const uint*`, and Metal's own validation aborts a binding
        // whose length is shorter than the argument it is bound to ("space for
        // 1 bytes, but argument has a length(4)"). The value is never read when
        // `useKeep` is 0, so zero is also the honest placeholder: a stray read
        // keeps nothing rather than whatever the allocator left there.
        guard
            let empty = context.device.makeBuffer(
                length: MemoryLayout<UInt32>.size, options: .storageModeShared)
        else {
            throw PrefillAttentionError.commandEncoderFailed
        }
        empty.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0
        empty.label = "prefillAttention.keepMask.unused"
        self.emptyKeepMask = empty
        if context.device.supportsFamily(.apple10) {
            do {
                self.psoFullTensorOps2DValidityV2 = try context.pipeline(
                    "attention_prefill_full_tensorops_2d_validity_v2")
                self.tensorOpsUnavailableReason = ""
            } catch {
                self.psoFullTensorOps2DValidityV2 = nil
                self.tensorOpsUnavailableReason = "\(error)"
            }
        } else {
            self.psoFullTensorOps2DValidityV2 = nil
            self.tensorOpsUnavailableReason =
                "device does not support Apple10 MPP tensor operations"
        }
    }

    func encodeCausal(
        commandBuffer: MTLCommandBuffer,
        q: MTLBuffer, qOffset: Int = 0,
        k: MTLBuffer, kOffset: Int = 0,
        v: MTLBuffer, vOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        params: PrefillAttentionParams,
        kvRingCapacity: UInt32 = 0,
        keepMask: MTLBuffer? = nil,
        keepStride: Int = 0,
        keepIndices: MTLBuffer? = nil,
        keepIndexStride: Int = 0,
        keepCounts: MTLBuffer? = nil,
        keepRowOffset: Int = 0,
        path: RuntimePrefillAttentionPath = .causalTiled,
        groupedQueryHeads: Bool = PrefillAttention.qsaGroupedQueryHeads,
        matrixUnits: Bool = PrefillAttention.qsaMatrixUnits
    ) throws {
        validate(params)

        // The flash kernel serves a selection (masked) and, with `denseFlash`,
        // plain causal attention (no selection) too.
        if maskedFlash && keepMask != nil || denseFlash && keepMask == nil,
            kvRingCapacity == 0, params.headDim == 256,
            // Full-attention layers pass the whole visible range as their
            // window, which limits nothing; a real window would.
            params.slidingWindow == 0 || params.slidingWindow >= params.kvValidCount,
            keepRowOffset % Self.flashRows == 0,
            params.numQHeads % params.numKVHeads == 0,
            params.numQHeads / params.numKVHeads <= 16,
            path != .fullTensorOps2DValidityV2,
            let flash = psoQSAMaskedFlash, let tileFlags = psoQSAFlashTileFlags
        {
            try encodeMaskedFlash(
                commandBuffer: commandBuffer, pipeline: flash, flagsPipeline: tileFlags,
                q: q, qOffset: qOffset, k: k, kOffset: kOffset, v: v, vOffset: vOffset,
                out: out, outOffset: outOffset, params: params,
                keepMask: keepMask, keepStride: keepStride, keepRowOffset: keepRowOffset)
            return
        }

        let (pipeline, useTensorOps, usesGroupedQSA, fixedThreads) = try selectPipeline(
            params: params, kvRingCapacity: kvRingCapacity, keepMask: keepMask,
            path: path, groupedQueryHeads: groupedQueryHeads, matrixUnits: matrixUnits)
        let headDim = Int(params.headDim)
        let threadWidth = max(1, pipeline.threadExecutionWidth)
        let threadCount =
            fixedThreads
            ?? (useTensorOps
                ? 128
                : roundUp(max(threadWidth, headDim), toMultipleOf: threadWidth))
        precondition(
            threadCount <= pipeline.maxTotalThreadsPerThreadgroup,
            "tiled prefill attention requires headDim <= maxTotalThreadsPerThreadgroup")

        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw PrefillAttentionError.commandEncoderFailed
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var p = params
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
        // Only the tiled kernel reads the selection; the TensorOps path would
        // ignore it, which is the wrong kind of quiet for a mask.
        precondition(
            keepMask == nil || !useTensorOps,
            "sparse key selection is not implemented for the "
                + "TensorOps prefill path")
        // useKeep 2 means "the selection arrived compacted": loop over the
        // index list instead of scanning every visible key for a mask byte.
        // TINYTITAN_QSA_COMPACT=0 keeps the mask scan, so the two forms can be
        // compared on one build. They must agree token for token: the
        // compacted list is the same selection, only enumerated.
        let compactionAllowed =
            ProcessInfo.processInfo
            .environment["TINYTITAN_QSA_COMPACT"] != "0"
        let compacted =
            compactionAllowed
            && keepMask != nil && keepIndices != nil && keepCounts != nil
        var useKeep = UInt32(keepMask == nil ? 0 : (compacted ? 2 : 1))
        var stride = UInt32(keepStride)
        var indexStride = UInt32(keepIndexStride)
        // A row tile (see encodeCausalTiled) starts `keepRowOffset` rows into
        // the chunk's selection: the kernels index it by dispatch-local row.
        enc.setBuffer(
            keepMask ?? emptyKeepMask,
            offset: keepMask == nil ? 0 : keepRowOffset * keepStride, index: 5)
        enc.setBytes(&useKeep, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 7)
        let u32 = MemoryLayout<UInt32>.stride
        enc.setBuffer(
            keepIndices ?? emptyKeepMask,
            offset: keepIndices == nil ? 0 : keepRowOffset * keepIndexStride * u32, index: 8)
        enc.setBuffer(
            keepCounts ?? emptyKeepMask,
            offset: keepCounts == nil ? 0 : keepRowOffset * u32, index: 9)
        enc.setBytes(&indexStride, length: MemoryLayout<UInt32>.size, index: 10)
        let groups =
            useTensorOps
            ? MTLSize(
                width: Int(params.queryCount),
                height: Int(params.numQHeads) / 8,
                depth: 1)
            : MTLSize(
                width: Int(params.queryCount),
                // The grouped kernel covers a KV head's query heads per group.
                height: Int(usesGroupedQSA ? params.numKVHeads : params.numQHeads),
                depth: 1)
        enc.dispatchThreadgroups(
            groups,
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// The selection as masked dense flash tiles: a pass marking which
    /// (8-row, 16-key) tiles any row keeps, then the attention over only
    /// those tiles, one threadgroup per (8 rows, KV head) with a simdgroup per
    /// query head. `keepRowOffset` (a multiple of 8) places this dispatch's
    /// rows in the chunk's mask and its flags in the chunk's flag buffer.
    private func encodeMaskedFlash(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        flagsPipeline: MTLComputePipelineState,
        q: MTLBuffer, qOffset: Int,
        k: MTLBuffer, kOffset: Int,
        v: MTLBuffer, vOffset: Int,
        out: MTLBuffer, outOffset: Int,
        params: PrefillAttentionParams,
        keepMask: MTLBuffer?, keepStride: Int, keepRowOffset: Int
    ) throws {
        let rows = Int(params.queryCount)
        let rowTiles = (rows + Self.flashRows - 1) / Self.flashRows
        // Without a selection the kernel masks by position: keys up to the
        // visible count, no flags pass.
        let keepStride = keepMask == nil ? Int(params.kvValidCount) : keepStride
        let keyTiles = (keepStride + Self.flashKeys - 1) / Self.flashKeys
        var causalOnly = UInt32(keepMask == nil ? 1 : 0)
        let flagOffset = keepMask == nil ? 0 : keepRowOffset / Self.flashRows * keyTiles
        let needed = keepMask == nil ? 1 : flagOffset + rowTiles * keyTiles
        if (flashTileFlags?.length ?? 0) < needed {
            guard
                let made = context.device.makeBuffer(
                    length: max(needed, 1), options: .storageModePrivate)
            else { throw PrefillAttentionError.commandEncoderFailed }
            made.label = "prefillAttention.qsaFlash.tileFlags"
            flashTileFlags = made
        }
        guard let flags = flashTileFlags else { throw PrefillAttentionError.commandEncoderFailed }
        var stride = UInt32(keepStride)
        var rowCount = UInt32(rows)
        var tiles = UInt32(keyTiles)
        let maskOffset = keepMask == nil ? 0 : keepRowOffset * keepStride
        // Packed: the row tiles' block lists stand in for the tile flags.
        var listBinding: (MTLBuffer, Int)?
        var countsBinding: (MTLBuffer, Int)?
        var listStride = UInt32(0)
        var attentionPipeline = pipeline
        if let keepMask, packedFlash, let packedPSO = psoQSAFlashPacked,
            let listPSO = psoQSAGroupBlocks
        {
            let blocksPerRow = (keepStride + 3) / 4
            let firstTile = keepRowOffset / Self.flashRows
            let listBytes = (firstTile + rowTiles) * blocksPerRow * 4
            let countBytes = (firstTile + rowTiles) * 4
            if (packedBlockLists?.length ?? 0) < listBytes {
                packedBlockLists = context.device.makeBuffer(
                    length: listBytes, options: .storageModePrivate)
                packedBlockLists?.label = "prefillAttention.qsaFlash.blockLists"
            }
            if (packedBlockCounts?.length ?? 0) < countBytes {
                packedBlockCounts = context.device.makeBuffer(
                    length: countBytes, options: .storageModePrivate)
                packedBlockCounts?.label = "prefillAttention.qsaFlash.blockCounts"
            }
            guard let lists = packedBlockLists, let counts = packedBlockCounts,
                let listEnc = commandBuffer.makeComputeCommandEncoder()
            else { throw PrefillAttentionError.commandEncoderFailed }
            listStride = UInt32(blocksPerRow)
            let listOffset = firstTile * blocksPerRow * 4
            let countOffset = firstTile * 4
            listEnc.setComputePipelineState(listPSO)
            listEnc.setBuffer(keepMask, offset: maskOffset, index: 0)
            listEnc.setBytes(&stride, length: 4, index: 1)
            listEnc.setBytes(&rowCount, length: 4, index: 2)
            listEnc.setBuffer(lists, offset: listOffset, index: 3)
            listEnc.setBytes(&listStride, length: 4, index: 4)
            listEnc.setBuffer(counts, offset: countOffset, index: 5)
            listEnc.dispatchThreadgroups(
                MTLSize(width: rowTiles, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            listEnc.endEncoding()
            listBinding = (lists, listOffset)
            countsBinding = (counts, countOffset)
            attentionPipeline = packedPSO
        } else if let keepMask {
            guard let flagEnc = commandBuffer.makeComputeCommandEncoder() else {
                throw PrefillAttentionError.commandEncoderFailed
            }
            flagEnc.setComputePipelineState(flagsPipeline)
            flagEnc.setBuffer(keepMask, offset: maskOffset, index: 0)
            flagEnc.setBytes(&stride, length: 4, index: 1)
            flagEnc.setBytes(&rowCount, length: 4, index: 2)
            flagEnc.setBuffer(flags, offset: flagOffset, index: 3)
            flagEnc.setBytes(&tiles, length: 4, index: 4)
            flagEnc.dispatchThreads(
                MTLSize(width: keyTiles, height: rowTiles, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            flagEnc.endEncoding()
        }

        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw PrefillAttentionError.commandEncoderFailed
        }
        enc.setComputePipelineState(attentionPipeline)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var p = params
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
        enc.setBuffer(keepMask ?? emptyKeepMask, offset: maskOffset, index: 5)
        enc.setBytes(&stride, length: 4, index: 6)
        if let listBinding {
            enc.setBuffer(listBinding.0, offset: listBinding.1, index: 7)
            enc.setBytes(&listStride, length: 4, index: 8)
        } else {
            enc.setBuffer(flags, offset: flagOffset, index: 7)
            enc.setBytes(&tiles, length: 4, index: 8)
        }
        enc.setBytes(&causalOnly, length: 4, index: 9)
        enc.setBuffer(countsBinding?.0 ?? emptyKeepMask, offset: countsBinding?.1 ?? 0, index: 10)
        let group = Int(params.numQHeads / params.numKVHeads)
        enc.dispatchThreadgroups(
            MTLSize(width: rowTiles, height: Int(params.numKVHeads), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * max(group, 4), height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Query rows per command buffer for a chunk whose last query sees
    /// `visibleEnd` keys, or nil to keep the whole chunk in one.
    ///
    /// macOS kills a command buffer that holds the GPU long enough to stall
    /// the display (kIOGPUCommandBufferCallbackErrorImpactingInteractivity).
    /// Measured on an M1 Max, Qwen3.8 4-bit: a 16,384-row attention core at
    /// ~74K context held the GPU 3.26 s in one buffer, and a request at ~70K
    /// failed that way. Splitting rows is exact -- every kernel here indexes
    /// by dispatch-local row and derives the query's position from
    /// `startPosition + row` -- and costs one commit per tile, with no wait.
    /// ~3.6K rows at 74K and 1K at 262K keep a tile well under a second.
    /// `TINYTITAN_PREFILL_ATTN_TILE_ROWS` overrides the row count; 0 disables.
    static func causalTileRows(queryCount: Int, visibleEnd: Int) -> Int? {
        let override = ProcessInfo.processInfo.environment["TINYTITAN_PREFILL_ATTN_TILE_ROWS"]
            .flatMap(Int.init)
        let rows: Int
        if let override {
            guard override > 0 else { return nil }
            rows = override
        } else {
            let scaled = 4_096 * 65_536 / max(visibleEnd, 65_536)
            rows = max(256, scaled / 128 * 128)
        }
        return rows < queryCount ? rows : nil
    }

    /// `encodeCausal` over the chunk in row tiles, each tile its own command
    /// buffer, committed without waiting so the GPU queue stays full. Returns
    /// the command buffer the caller continues in. One tile is exactly
    /// `encodeCausal`. `onCommit` sees each buffer this commits (not the
    /// returned one), so a caller can time them once they complete.
    func encodeCausalTiled(
        commandBuffer: MTLCommandBuffer,
        queue: MTLCommandQueue,
        q: MTLBuffer,
        k: MTLBuffer, kOffset: Int = 0,
        v: MTLBuffer, vOffset: Int = 0,
        out: MTLBuffer,
        params: PrefillAttentionParams,
        kvRingCapacity: UInt32 = 0,
        keepMask: MTLBuffer? = nil,
        keepStride: Int = 0,
        keepIndices: MTLBuffer? = nil,
        keepIndexStride: Int = 0,
        keepCounts: MTLBuffer? = nil,
        path: RuntimePrefillAttentionPath = .causalTiled,
        tileRows: Int? = nil,
        onCommit: ((MTLCommandBuffer) -> Void)? = nil
    ) throws -> MTLCommandBuffer {
        let total = Int(params.queryCount)
        let rows =
            tileRows
            ?? Self.causalTileRows(
                queryCount: total,
                visibleEnd: Int(params.startPosition) + total)
            ?? total
        var cb = commandBuffer
        var first = 0
        while first < total {
            let count = min(rows, total - first)
            var tile = params
            tile.startPosition = params.startPosition + UInt32(first)
            tile.queryCount = UInt32(count)
            try encodeCausal(
                commandBuffer: cb,
                q: q, qOffset: first * Int(params.qTokenStrideElements) * 2,
                k: k, kOffset: kOffset,
                v: v, vOffset: vOffset,
                out: out, outOffset: first * Int(params.oTokenStrideElements) * 2,
                params: tile,
                kvRingCapacity: kvRingCapacity,
                keepMask: keepMask, keepStride: keepStride,
                keepIndices: keepIndices, keepIndexStride: keepIndexStride,
                keepCounts: keepCounts, keepRowOffset: first,
                path: path)
            first += count
            if first < total {
                cb.commit()
                onCommit?(cb)
                guard let next = queue.makeCommandBuffer() else {
                    throw PrefillAttentionError.commandEncoderFailed
                }
                cb = next
            }
        }
        return cb
    }

    /// Which kernel serves this call: TensorOps for the one shape it covers,
    /// the grouped or per-head QSA kernel when a selection is in play, else the
    /// plain tiled kernel.
    private func selectPipeline(
        params: PrefillAttentionParams,
        kvRingCapacity: UInt32,
        keepMask: MTLBuffer?,
        path: RuntimePrefillAttentionPath,
        groupedQueryHeads: Bool,
        matrixUnits: Bool
    ) throws -> (
        pipeline: MTLComputePipelineState, tensorOps: Bool, groupedQSA: Bool, threads: Int?
    ) {
        let requestsTensorOps =
            path == .fullTensorOps2DPreferred
            || path == .fullTensorOps2DValidityV2
        // The pinned model uses 512/16/2 only for full attention; its
        // sliding-window layers use 256/16/8. A future model that reuses this
        // shape for sliding attention must add a full-visibility check here.
        let tensorOpsShape =
            requestsTensorOps
            && params.kvBits == 16
            && kvRingCapacity == 0
            && params.headDim == 512
            && params.numQHeads == 16
            && params.numKVHeads == 2
            && params.scale == 1.0
        let tensorOpsPipeline = tensorOpsShape ? psoFullTensorOps2DValidityV2 : nil
        let useTensorOps = tensorOpsPipeline != nil
        let pipeline: MTLComputePipelineState
        var usesGroupedQSA = false
        var fixedThreads: Int?
        if let tensorOpsPipeline {
            pipeline = tensorOpsPipeline
        } else if tensorOpsShape && path == .fullTensorOps2DValidityV2 {
            // K7: the caller explicitly requested the TensorOps path — fail
            // loudly with the recorded reason instead of crashing or silently
            // running a different kernel. Only auto-selected paths fall back.
            throw PrefillAttentionError.tensorOpsUnavailable(
                reason: tensorOpsUnavailableReason.isEmpty
                    ? "TensorOps pipeline failed to compile"
                    : tensorOpsUnavailableReason)
        } else {
            // Explicit mode also falls back for incompatible shapes. Benchmark
            // fixtures must use 512/16/2 to prove that TensorOps ran.
            // The tiled QSA kernel is only better when there is a selection
            // to iterate: with none, `iterations` is the whole visible range
            // and its per-tile bookkeeping buys nothing.
            let wantQSATiled =
                keepMask != nil
                && ProcessInfo.processInfo.environment["TINYTITAN_QSA_TILED"] != "0"
            if wantQSATiled, kvRingCapacity == 0, matrixUnits, params.headDim == 256,
                params.numQHeads / params.numKVHeads <= 16,
                let mma = psoCausalQSAGQAMMA,
                groupedQSAPipeline(params: params, requested: groupedQueryHeads) != nil
            {
                // The kernel pads the heads to 16 matrix rows.
                // Same grid as the grouped kernel, a fixed four simdgroups.
                pipeline = mma
                usesGroupedQSA = true
                fixedThreads = 128
            } else if wantQSATiled, kvRingCapacity == 0,
                let gqa = groupedQSAPipeline(params: params, requested: groupedQueryHeads)
            {
                pipeline = gqa
                usesGroupedQSA = true
            } else if wantQSATiled, let qsa = psoCausalQSATiled, kvRingCapacity == 0 {
                pipeline = qsa
            } else {
                pipeline = causalTiledPipeline(kvRingCapacity: kvRingCapacity)
            }
        }
        return (pipeline, useTensorOps, usesGroupedQSA, fixedThreads)
    }

    /// The grouped QSA pipeline when it was asked for, compiled, and fits this
    /// shape: at most `qsaGQAMaxGroup` query heads per KV head (its per-head
    /// state is sized for that), a head no wider than `qsaGQAMaxHeadDim`, and a
    /// threadgroup of at least one thread per query head.
    private func groupedQSAPipeline(
        params: PrefillAttentionParams, requested: Bool
    ) -> MTLComputePipelineState? {
        guard requested, let gqa = psoCausalQSAGQA,
            params.numKVHeads > 0, params.numQHeads % params.numKVHeads == 0
        else { return nil }
        let group = Int(params.numQHeads / params.numKVHeads)
        let headDim = Int(params.headDim)
        let width = max(1, gqa.threadExecutionWidth)
        let threads = roundUp(max(width, headDim), toMultipleOf: width)
        guard group <= Self.qsaGQAMaxGroup, headDim <= Self.qsaGQAMaxHeadDim,
            threads >= group, threads <= gqa.maxTotalThreadsPerThreadgroup
        else { return nil }
        return gqa
    }

    private func validate(_ params: PrefillAttentionParams) {
        precondition(params.headDim > 0, "headDim must be positive")
        precondition(params.queryCount > 0, "queryCount must be positive")
        precondition(params.numQHeads > 0, "numQHeads must be positive")
        precondition(params.numKVHeads > 0, "numKVHeads must be positive")
        precondition(
            params.numQHeads % params.numKVHeads == 0,
            "numQHeads must be divisible by numKVHeads")
        precondition(
            params.qTokenStrideElements >= params.numQHeads * params.headDim,
            "q token stride is too small")
        precondition(
            params.oTokenStrideElements >= params.numQHeads * params.headDim,
            "output token stride is too small")
        if params.kvBits == 16 {
            precondition(
                params.kvTokenStrideElements >= params.numKVHeads * params.headDim,
                "KV token stride is too small")
        } else {
            precondition(
                params.kvBits == 4 || params.kvBits == 8,
                "KV bits must be 4, 8, or 16")
            precondition(
                params.kvTokenStrideBytes > 0,
                "quantized KV token stride must be positive")
        }
        precondition(
            params.startPosition + params.queryCount <= params.kvValidCount,
            "kvValidCount must include all in-flight query rows")
    }

    private func roundUp(_ value: Int, toMultipleOf multiple: Int) -> Int {
        ((value + multiple - 1) / multiple) * multiple
    }

    private func causalTiledPipeline(kvRingCapacity: UInt32) -> MTLComputePipelineState {
        guard kvRingCapacity > 0 else { return psoCausalTiled }
        do {
            return try context.pipeline(
                "attention_prefill_causal_tiled",
                constants: [MetalFunctionConstant(index: 76, value: .uint32(kvRingCapacity))])
        } catch {
            preconditionFailure("failed to build KV ring prefill attention pipeline: \(error)")
        }
    }
}
