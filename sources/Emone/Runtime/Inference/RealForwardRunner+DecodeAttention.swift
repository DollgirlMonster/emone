import Foundation
import Metal

/// Decode-time attention encoding, split out of `RealForwardRunner+Decode.swift`.
///
/// Gated-DeltaNet (linear attention) and gated full attention for a single
/// decoded token, plus the packed query/gate projection they share. Pure code
/// motion: the same encoder bodies, moved so one file holds one concern and no
/// file in this directory passes 1,600 lines.

extension RealForwardRunner {
    func encodeLinearAttentionDecode(
        _ cb: inout MTLCommandBuffer, layer L: Int,
        slot: Int = 0
    ) throws {
        guard let gdn, let gdnState, let gdnQKVRaw, let gdnConvOut,
            let gdnZ, let gdnA, let gdnB, let gdnY, let gdnOut
        else {
            throw ModelError.internalInconsistency(
                detail:
                    "linear-attention layer \(L) without GDN kernels (arch mask misconfiguration)")
        }
        let la = cfg.linearAttention
        let D = UInt32(cfg.hiddenSize)
        let qkvW = try model.linearInProjQKV(layer: L)
        let zW = try model.linearInProjZ(layer: L)
        let aW = try model.linearInProjA(layer: L)
        let bW = try model.linearInProjB(layer: L)
        let outW = try model.linearOutProj(layer: L)
        let convW = try model.linearConv1d(layer: L)
        let aLog = try model.linearALog(layer: L)
        let dtBias = try model.linearDtBias(layer: L)
        let gatedNormW = try model.linearNorm(layer: L)

        // One dispatch over the concatenated qkv/z/a/b row space instead of four
        // separate GEMVs (a and b were 4 threadgroups each). The fused kernel
        // reads all four as packed 4-bit nibbles, so it is only usable when all
        // four *are* affine: the dense Qwen 3.5 installs keep `a` and `b` at
        // bf16, and the per-projection GEMV below routes bf16, 4- and 8-bit.
        // Reading bf16 bytes as nibbles is silent nonsense, not an error.
        let fusedIsUsable =
            model.attentionWeightBits == 4
            && qkvW.dtype == 0 && zW.dtype == 0 && aW.dtype == 0 && bW.dtype == 0
        if fusedIsUsable {
            if !ablated("inproj") {
                try gdn.encodeInputProjections(
                    commandBuffer: cb,
                    x: normed,
                    qkv: qkvW, qkvOut: gdnQKVRaw,
                    z: zW, zOut: gdnZ,
                    a: aW, aOut: gdnA,
                    b: bW, bOut: gdnB,
                    hiddenSize: cfg.hiddenSize)
            }
        } else {
            try encodeRoleGEMV(
                commandBuffer: cb, projection: qkvW,
                weightBits: model.gdnProjectionWeightBits,
                x: normed, y: gdnQKVRaw,
                m: UInt32(la.qkvDim), n: D)
            try encodeRoleGEMV(
                commandBuffer: cb, projection: zW,
                weightBits: model.gdnProjectionWeightBits,
                x: normed, y: gdnZ,
                m: UInt32(la.valueDim), n: D)
            try encodeRoleGEMV(
                commandBuffer: cb, projection: aW,
                weightBits: model.gdnProjectionWeightBits,
                x: normed, y: gdnA,
                m: UInt32(la.numVHeads), n: D)
            try encodeRoleGEMV(
                commandBuffer: cb, projection: bW,
                weightBits: model.gdnProjectionWeightBits,
                x: normed, y: gdnB,
                m: UInt32(la.numVHeads), n: D)
        }
        try rotate(&cb, role: "gdn.inproj")

        if !ablated("conv") {
            let tail = gdnState.convTailSlot(layer: L, slot: slot)
            try gdn.encodeConvDecode(
                commandBuffer: cb,
                tail: tail.buffer, tailOffset: tail.offset,
                qkv: gdnQKVRaw,
                convWeight: convW.buffer,
                convWeightOffset: Int(convW.offset),
                out: gdnConvOut)
        }
        try rotate(&cb, role: "gdn.conv")
        if !ablated("qknorm") {
            try gdn.encodeQKNorm(commandBuffer: cb, convOut: gdnConvOut)
        }
        try rotate(&cb, role: "gdn.qknorm")
        if !ablated("delta") {
            let state = gdnState.stateSlot(layer: L, slot: slot)
            try gdn.encodeDeltaStepDecode(
                commandBuffer: cb,
                convOut: gdnConvOut,
                aProj: gdnA,
                bProj: gdnB,
                aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
                state: state.buffer, stateOffset: state.offset,
                y: gdnY)
        }
        try rotate(&cb, role: "gdn.delta")
        if !ablated("gatednorm") {
            try gdn.encodeGatedNorm(
                commandBuffer: cb,
                y: gdnY,
                z: gdnZ,
                weight: gatedNormW.buffer,
                weightOffset: Int(gatedNormW.offset),
                out: gdnOut)
        }
        try rotate(&cb, role: "gdn.gatednorm")
        if !ablated("outproj") {
            try encodePrimaryGEMV(
                commandBuffer: cb,
                weights: outW.buffer, weightsOffset: Int(outW.offset),
                scales: outW.buffer, scalesOffset: Int(outW.scaleOffset),
                biases: outW.buffer, biasesOffset: Int(outW.biasOffset),
                x: gdnOut, y: oOut, m: D, n: UInt32(la.valueDim))
        }
        try rotate(&cb, role: "gdn.outproj")
    }

    /// Qwen full attention (attn_output_gate), one decode step: packed
    /// [query ; gate] q_proj split per head, weighted per-head q/k norms
    /// (no V norm), NeoX sub-dim RoPE, full attention with the configured
    /// scale, sigmoid output gate, then o_proj into `oOut`.
    func encodeGatedFullQKVProjection(
        _ cb: MTLCommandBuffer,
        layer: Int,
        qOutput: MTLBuffer,
        kOutput: (buffer: MTLBuffer, offset: Int),
        vOutput: (buffer: MTLBuffer, offset: Int),
        qDimension: UInt32,
        kvDimension: UInt32
    ) throws {
        let q = try model.qProj(layer: layer)
        let k = try model.kProj(layer: layer)
        let v = try model.vProj(layer: layer)
        let hiddenDimension = UInt32(cfg.hiddenSize)
        if model.qoProjectionWeightBits == 4 && model.kvProjectionWeightBits == 4 {
            try fusedQKVGEMV.encode(
                commandBuffer: cb,
                qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                vWeights: v.buffer, vWeightsOffset: Int(v.offset),
                vScales: v.buffer, vScalesOffset: Int(v.scaleOffset),
                vBiases: v.buffer, vBiasesOffset: Int(v.biasOffset),
                x: normed,
                qOut: qOutput,
                kOut: kOutput.buffer, kOutOffset: kOutput.offset,
                vOut: vOutput.buffer, vOutOffset: vOutput.offset,
                qRows: 2 * qDimension,
                kvRows: kvDimension,
                n: hiddenDimension)
        } else {
            try encodePrimaryGEMV(
                commandBuffer: cb, projection: q,
                x: normed, y: qOutput,
                m: 2 * qDimension, n: hiddenDimension)
            try encodePrimaryGEMV(
                commandBuffer: cb, projection: k,
                x: normed, y: kOutput.buffer,
                yOffset: kOutput.offset,
                m: kvDimension, n: hiddenDimension)
            try encodePrimaryGEMV(
                commandBuffer: cb, projection: v,
                x: normed, y: vOutput.buffer,
                yOffset: vOutput.offset,
                m: kvDimension, n: hiddenDimension)
        }
    }

    func encodeGatedFullAttentionDecode(
        _ cb: inout MTLCommandBuffer,
        layer L: Int,
        position: Int,
        slot: Int = 0,
        seqLen: UInt32,
        keepMask: MTLBuffer? = nil
    ) throws {
        guard let elementwise, let rope, let qPackedScratch, let attnGateScratch else {
            throw ModelError.internalInconsistency(
                detail:
                    "attn_output_gate layer \(L) without gate kernels (arch mask misconfiguration)")
        }
        guard let kv else {
            throw ModelError.internalInconsistency(
                detail: "full attention requires a KV cache")
        }
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = 1e-6
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = UInt32(cfg.numHeads * headDim)
        let kvDim = UInt32(numKV * headDim)
        let kSlot = kv.kSlot(layer: L, position: position, slot: slot)
        let vSlot = kv.vSlot(layer: L, position: position, slot: slot)
        let quantizedKV = kv.precision.isQuantized
        let kWrite = quantizedKV ? (buffer: kStage, offset: 0) : kSlot
        let vWrite = quantizedKV ? (buffer: vStage, offset: 0) : vSlot
        let o = try model.oProj(layer: L)
        let qNormW = try model.qNorm(layer: L)
        let kNormW = try model.kNorm(layer: L)
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)

        try encodeGatedFullQKVProjection(
            cb, layer: L, qOutput: qPackedScratch,
            kOutput: kWrite, vOutput: vWrite,
            qDimension: qDim, kvDimension: kvDim)
        try rotate(&cb, role: "qsa.qkv")
        try elementwise.encodeSplitQGate(
            commandBuffer: cb,
            packed: qPackedScratch,
            q: qScratch,
            gate: attnGateScratch,
            heads: cfg.numHeads,
            dim: headDim)
        try rms.encodeBF16WPerHead(
            commandBuffer: cb,
            x: qScratch,
            weight: qNormW.buffer,
            weightOffset: Int(qNormW.offset),
            out: qScratch,
            headDim: UInt32(headDim),
            numHeads: cfg.numHeads,
            eps: eps)
        try rms.encodeBF16WPerHead(
            commandBuffer: cb,
            x: kWrite.buffer, xOffset: kWrite.offset,
            weight: kNormW.buffer,
            weightOffset: Int(kNormW.offset),
            out: kWrite.buffer, outOffset: kWrite.offset,
            headDim: UInt32(headDim),
            numHeads: numKV,
            eps: eps)
        try rope.encodeNeoxSubdim(
            commandBuffer: cb,
            data: qScratch,
            position: UInt32(position),
            headDim: UInt32(headDim),
            numHeads: UInt32(cfg.numHeads),
            rotaryDim: rotaryDim,
            theta: Float(cfg.fullRopeTheta))
        try rotate(&cb, role: "qsa.gate_norm_rope1")
        try rope.encodeNeoxSubdim(
            commandBuffer: cb,
            data: kWrite.buffer,
            dataOffset: kWrite.offset,
            position: UInt32(position),
            headDim: UInt32(headDim),
            numHeads: UInt32(numKV),
            rotaryDim: rotaryDim,
            theta: Float(cfg.fullRopeTheta))
        if quantizedKV {
            try encodeQuantizedKV(
                commandBuffer: cb, kv: kv, layer: L,
                position: position, slot: slot, keySource: kStage,
                valueSource: vStage, elementCount: Int(kvDim))
        }
        let keyView = kv.keyView(layer: L, slot: slot, validTokenCount: Int(seqLen))
        let valueView = kv.valueView(layer: L, slot: slot, validTokenCount: Int(seqLen))
        try attention.encodeFull(
            commandBuffer: cb,
            q: qScratch,
            k: keyView.buffer, kOffset: keyView.offset,
            v: valueView.buffer, vOffset: valueView.offset,
            out: attnOut,
            headDim: UInt32(headDim),
            numQHeads: UInt32(cfg.numHeads),
            numKVHeads: UInt32(numKV),
            seqLen: seqLen,
            scale: Float(cfg.attentionScale),
            kvFormat: keyView,
            keepMask: keepMask)
        try rotate(&cb, role: "qsa.attention")
        try elementwise.encodeSigmoidGateMul(
            commandBuffer: cb,
            out: attnOut,
            gate: attnGateScratch,
            count: Int(qDim))
        try rotate(&cb, role: "qsa.gatemul")
        try encodePrimaryGEMV(
            commandBuffer: cb,
            weights: o.buffer, weightsOffset: Int(o.offset),
            scales: o.buffer, scalesOffset: Int(o.scaleOffset),
            biases: o.buffer, biasesOffset: Int(o.biasOffset),
            x: attnOut, y: oOut, m: D, n: qDim)
        try rotate(&cb, role: "qsa.oproj")
    }
}
