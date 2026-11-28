import Foundation

/// Which projections a runtime-LoRA adapter may target on a model, and the
/// shape each must have (Emone-lmw.14).
///
/// Every family runs its prefill attention through the same seams
/// (`encodeLinearAttentionPrefill` / `encodeFullAttentionPrefill`), so the
/// attention and Gated-DeltaNet projections can be adapted on the dense Qwen
/// 3.5, the Qwen 3.6 MoE and the 125B alike. The MLP differs by family: the
/// dense family's is `mlp.{gate,up,down}_proj`; an MoE layer's is the routed
/// experts plus `mlp.shared_expert.{gate,up,down}_proj`. Both of those run as
/// the same batched FFN with a seam before its activation, so the dense MLP
/// and the MoE shared expert are adapted. The routed experts are not, and an
/// adapter naming the other family's MLP is refused rather than run with that
/// part silently dropped.
///
/// The check is pure (config in, refusal out), so it runs before any GPU
/// work: a mismatch found mid-prefill would leave a half-written branch.
public enum LoRAAdapterFit {
    /// `(rows, columns)` of `target` in `layer`, or nil when that layer of
    /// this model has no such adaptable projection.
    public static func expectedShape(
        target: LoRATarget, layer: Int, config cfg: ArchConfig
    ) -> (rows: Int, columns: Int)? {
        guard layer >= 0, layer < cfg.numLayers else { return nil }
        let d = cfg.hiddenSize
        let linear = cfg.layerIsLinear(layer)
        let la = cfg.linearAttention
        let full = cfg.fullAttentionLayerMask[layer] == 1
        let headDim = full ? cfg.fullHeadDim : cfg.headDim
        let kvHeads = full ? cfg.numFullKVHeads : cfg.numKVHeads
        let qDim = cfg.numHeads * headDim
        let kvDim = kvHeads * headDim
        let dense = cfg.family == .qwen35Dense
        switch target {
        case .q: return linear ? nil : (cfg.attnOutputGate ? 2 * qDim : qDim, d)
        case .k: return linear ? nil : (kvDim, d)
        case .v: return linear || cfg.attentionKEqV ? nil : (kvDim, d)
        case .o: return linear ? nil : (d, qDim)
        case .gdnQKV: return linear ? (la.qkvDim, d) : nil
        case .gdnZ: return linear ? (la.valueDim, d) : nil
        case .gdnA, .gdnB: return linear ? (la.numVHeads, d) : nil
        case .gdnOut: return linear ? (d, la.valueDim) : nil
        case .gate, .up:
            return dense ? (cfg.intermediateSize, d) : nil
        case .down:
            return dense ? (d, cfg.intermediateSize) : nil
        case .sharedGate, .sharedUp:
            return dense ? nil : (cfg.intermediateSize, d)
        case .sharedDown:
            return dense ? nil : (d, cfg.intermediateSize)
        }
    }

    /// One adapted projection as a package declares it.
    public struct Declared: Sendable, Equatable {
        public let layer: Int
        public let target: LoRATarget
        public let rows: Int
        public let columns: Int

        public init(layer: Int, target: LoRATarget, rows: Int, columns: Int) {
            self.layer = layer
            self.target = target
            self.rows = rows
            self.columns = columns
        }
    }

    /// Why `entries` cannot run on a model of `config`, or nil when every one
    /// names a projection that exists there, at its shape. The first problem
    /// in (layer, target) order is reported, so the message is stable.
    public static func refusal(entries: [Declared], config cfg: ArchConfig) -> String? {
        let ordered = entries.sorted {
            ($0.layer, $0.target.rawValue) < ($1.layer, $1.target.rawValue)
        }
        for entry in ordered {
            guard entry.layer >= 0, entry.layer < cfg.numLayers else {
                return "adapter names layer \(entry.layer); this model has \(cfg.numLayers)"
            }
            guard let shape = expectedShape(target: entry.target, layer: entry.layer, config: cfg)
            else {
                return "adapter targets \(entry.target.rawValue) of layer \(entry.layer), "
                    + "which \(cfg.family.rawValue) \(whyMissing(entry, cfg))"
            }
            guard shape.rows == entry.rows, shape.columns == entry.columns else {
                return "adapter \(entry.target.rawValue) of layer \(entry.layer) is "
                    + "\(entry.rows)x\(entry.columns); the projection is "
                    + "\(shape.rows)x\(shape.columns)"
            }
        }
        return nil
    }

    private static func whyMissing(_ entry: Declared, _ cfg: ArchConfig) -> String {
        switch entry.target {
        case .gate, .up, .down:
            return "does not have (an MoE layer's MLP is routed experts plus "
                + "mlp.shared_expert.*; only the shared expert is adapted)"
        case .sharedGate, .sharedUp, .sharedDown:
            return "does not have (a dense layer's MLP is mlp.{gate,up,down}_proj)"
        case .gdnQKV, .gdnZ, .gdnA, .gdnB, .gdnOut:
            return "has no Gated-DeltaNet projection on (it is an attention layer)"
        case .q, .k, .v, .o:
            return cfg.layerIsLinear(entry.layer)
                ? "has no softmax-attention projection on (it is a Gated-DeltaNet layer)"
                : "does not carry (keys and values share one projection)"
        }
    }
}
