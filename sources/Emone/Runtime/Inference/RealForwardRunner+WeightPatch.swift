import Foundation
import Metal

/// Weight patches (Emone-a37): a hand-written low-rank edit laid over this
/// runner's forward passes. Prefill goes through the same seams as the
/// decision adapter (`RealForwardRunner+LoRA.swift`); decode adds the
/// `mlp.down_proj` delta to the dense MLP's output before the residual add.
extension RealForwardRunner {
    /// Lays `patch` over this runner, or removes it with nil. Refused for
    /// anything but the dense Qwen 3.5 family, and when the patch's shapes or
    /// layers do not fit this model.
    public func setWeightPatch(_ patch: WeightPatch?) throws {
        guard let patch else {
            weightPatch = nil
            return
        }
        guard cfg.family == .qwen35Dense else {
            throw ModelError.internalInconsistency(
                detail:
                    "weight patches cover the dense Qwen 3.5 family; this is \(cfg.family.rawValue)"
            )
        }
        // Only the down_proj deltas ride the batched dense FFN; head scales sit
        // after the attention output gate on every prefill path.
        guard patch.targetCount == 0 || prefillBatchedSharedExpert else {
            throw ModelError.internalInconsistency(
                detail:
                    "a weight patch needs the batched dense FFN (the profile's dense QMM switch)")
        }
        for layer in patch.layers {
            guard layer >= 0, layer < cfg.numLayers else {
                throw ModelError.internalInconsistency(
                    detail: "weight patch names layer \(layer); this model has \(cfg.numLayers)")
            }
            if let down = patch.entry(layer: layer, target: .down),
                down.rows != cfg.hiddenSize || down.columns != cfg.intermediateSize
            {
                throw ModelError.internalInconsistency(
                    detail:
                        "weight patch down_proj of layer \(layer) is \(down.rows)x\(down.columns); "
                        + "the projection is \(cfg.hiddenSize)x\(cfg.intermediateSize)")
            }
        }
        for layer in patch.headScaleLayers {
            guard let scale = patch.headScale(layer: layer), layer >= 0, layer < cfg.numLayers,
                scale.linear == cfg.layerIsLinear(layer), scale.linear || cfg.attnOutputGate
            else {
                throw ModelError.internalInconsistency(
                    detail:
                        "weight patch head_scale of layer \(layer) names the wrong kind of attention "
                        + "(gated full attention takes self_attn, Gated DeltaNet linear_attn)")
            }
            let heads = scale.linear ? cfg.linearAttention.numVHeads : cfg.numHeads
            guard scale.heads == heads else {
                throw ModelError.internalInconsistency(
                    detail:
                        "weight patch head_scale of layer \(layer) has \(scale.heads) factors; "
                        + "the layer has \(heads) heads")
            }
        }
        if loraDeltaKernel == nil { loraDeltaKernel = try LoRADelta(context: ctx) }
        if !patch.headScaleLayers.isEmpty, headScaleKernel == nil {
            headScaleKernel = try HeadScale(context: ctx)
        }
        weightPatch = patch
    }

    /// The head-scale seam, decode and prefill alike: `attention[t, h, :] *=
    /// scale[h]` over `tokens` rows, after the output gate (full attention) or
    /// the gated norm (DeltaNet) and before the output projection.
    func encodeWeightPatchHeadScale(
        commandBuffer: MTLCommandBuffer, layer L: Int, attention: MTLBuffer, tokens: Int
    ) throws {
        guard let patch = weightPatch, let scale = patch.headScale(layer: L) else { return }
        guard let kernel = headScaleKernel else {
            throw ModelError.internalInconsistency(detail: "head scales set without their kernel")
        }
        try kernel.encode(
            commandBuffer: commandBuffer, x: attention, scale: patch.buffer,
            scaleOffset: scale.offset, heads: scale.heads,
            dim: scale.linear ? cfg.linearAttention.valueHeadDim : cfg.fullHeadDim, tokens: tokens)
    }

    /// Decode's seam: `h1 += (act A^T) B^T` for one token, after the dense
    /// MLP's down projection and before the residual add reads `h1Buf`.
    func encodeWeightPatchDecodeDown(commandBuffer: MTLCommandBuffer, layer L: Int) throws {
        guard let patch = weightPatch, let entry = patch.entry(layer: L, target: .down) else {
            return
        }
        guard let kernel = loraDeltaKernel else {
            throw ModelError.internalInconsistency(detail: "weight patch set without its kernel")
        }
        try kernel.encode(
            commandBuffer: commandBuffer,
            x: denseScratchAct, xStrideElements: entry.columns,
            y: h1Buf, yStrideElements: entry.rows,
            a: patch.buffer, aOffset: entry.aOffset, b: patch.buffer, bOffset: entry.bOffset,
            scratch: patch.scratch,
            rank: patch.rank, rows: entry.rows, columns: entry.columns, tokens: 1, scale: 1)
    }
}
