import Foundation

/// How a layer's residual row is returned.
public enum HiddenReadoutStreamMode: String, Sendable, Equatable, CaseIterable {
    /// Every residual stream, widened to float32: `streams * hiddenSize`
    /// values, stream-major (`[s0 d0..dD-1, s1 d0..dD-1, ...]`). A pre-norm
    /// family has one stream, so this is the plain hidden state.
    case residual
    /// The unweighted mean over the streams: `hiddenSize` values. Identity on
    /// a one-stream family.
    case meanStreams = "mean_streams"
}

public enum HiddenReadoutError: Error, CustomStringConvertible, Equatable {
    case invalidLayers(String)
    /// The runner is in a configuration the readout cannot serve; the reason
    /// is written for the client that asked.
    case unsupported(String)
    case internalInconsistency(String)

    public var description: String {
        switch self {
        case .invalidLayers(let reason), .unsupported(let reason),
            .internalInconsistency(let reason):
            return reason
        }
    }
}

/// What to read back from a prefill: the layers, and how to reduce each row.
///
/// Layers are 0-based block indices. The row read for layer `L` is the
/// residual after block `L`'s *full* update -- attention branch and the
/// MLP/MoE tail both applied -- of the last prompt token. There is no
/// per-layer norm on the residual, so nothing is normalised on the way out.
///
/// Two modes. With `prefillOnly` (the default) the plan stops after the
/// deepest requested layer: the readout is a prefill whose sequence is
/// discarded, so layers above it are not run. Without it the capture rides an
/// ordinary prefill: every layer runs, the head and sampling follow, and the
/// row is the one from the forward pass that produced the reply.
public struct HiddenReadoutPlan: Sendable, Equatable {
    /// Ascending and unique, whatever order the caller named them in.
    public let layers: [Int]
    public let streamMode: HiddenReadoutStreamMode
    /// True: stop after the deepest layer, skip the head, discard the sequence.
    /// False: capture during an ordinary prefill, no early stop.
    public let prefillOnly: Bool

    /// The most layers one request may ask for. Each costs one 20 KB row on
    /// the wire as base64; the cap is a bound on the response, not on the GPU.
    public static let maximumLayers = 16

    /// Sorts and de-duplicates `layers`. Throws on an empty, negative or
    /// over-long list; the upper bound depends on the model and is checked by
    /// `validate(numLayers:)`.
    public init(
        layers: [Int], streamMode: HiddenReadoutStreamMode = .residual,
        prefillOnly: Bool = true
    ) throws {
        let unique = Array(Set(layers)).sorted()
        guard !unique.isEmpty else {
            throw HiddenReadoutError.invalidLayers("layers must name at least one layer")
        }
        guard unique[0] >= 0 else {
            throw HiddenReadoutError.invalidLayers("layers must be non-negative block indices")
        }
        guard unique.count <= Self.maximumLayers else {
            throw HiddenReadoutError.invalidLayers(
                "at most \(Self.maximumLayers) distinct layers may be read in one request")
        }
        self.layers = unique
        self.streamMode = streamMode
        self.prefillOnly = prefillOnly
    }

    public var deepestLayer: Int { layers[layers.count - 1] }

    /// Throws unless every layer exists in a model with `numLayers` blocks.
    public func validate(numLayers: Int) throws {
        guard deepestLayer < numLayers else {
            throw HiddenReadoutError.invalidLayers(
                "layer \(deepestLayer) is out of range; this model has \(numLayers) layers "
                    + "(valid indices 0...\(numLayers - 1))")
        }
    }

    /// What one prefill chunk does under this plan.
    public struct ChunkPlan: Sendable, Equatable {
        /// Layers run on the chunk: `0..<layerLimit`.
        public let layerLimit: Int
        /// Layers whose last-row residual is read after they finish. Empty on
        /// every chunk but the last, because the last prompt token is a row of
        /// the last chunk.
        public let captureLayers: [Int]
        /// True when the chunk stops early: the final head is skipped and the
        /// ANE (whose chunk bookkeeping cannot represent a stop) is not used.
        /// False for a capture riding an ordinary prefill, whose path is
        /// otherwise unchanged.
        public let earlyStop: Bool
    }

    /// A `prefillOnly` plan stops every chunk after the deepest layer (the
    /// layers above it are dead work for a discarded sequence); otherwise
    /// every chunk runs the whole stack. Either way only the last chunk
    /// captures.
    public func chunkPlan(isLastChunk: Bool, numLayers: Int) -> ChunkPlan {
        ChunkPlan(
            layerLimit: prefillOnly ? min(numLayers, deepestLayer + 1) : numLayers,
            captureLayers: isLastChunk ? layers : [],
            earlyStop: prefillOnly)
    }

    /// Row `index` of the readback buffer holds this layer, or nil if the plan
    /// does not read it.
    public func readbackIndex(forLayer layer: Int) -> Int? {
        layers.firstIndex(of: layer)
    }
}

/// One layer's row as returned to the caller.
public struct HiddenReadoutLayer: Sendable, Equatable {
    public let layer: Int
    public let values: [Float]

    public init(layer: Int, values: [Float]) {
        self.layer = layer
        self.values = values
    }
}

/// Which prefill produced the rows, so a client knows what it measured.
public struct HiddenCapturePath: Sendable, Equatable {
    /// The Neural Engine took at least one chunk's full-attention layers
    /// (only possible for a capture riding an ordinary prefill, and only where
    /// an ANE sidecar is installed); otherwise the GPU did.
    public let usedANE: Bool
    /// The capture stopped after the deepest layer.
    public let earlyStop: Bool

    public init(usedANE: Bool = false, earlyStop: Bool = true) {
        self.usedANE = usedANE
        self.earlyStop = earlyStop
    }
}

public struct HiddenReadoutResult: Sendable, Equatable {
    /// Absolute index, in the sequence, of the token whose row was read.
    public let position: Int
    public let layers: [HiddenReadoutLayer]
    public let path: HiddenCapturePath
    /// Host time the capture itself cost: encoding and committing the blits
    /// during prefill, the wait for them, and the reduction. Excludes the GPU
    /// time of the blits, which is a 20 KB copy per layer.
    public let captureNanos: UInt64

    public init(
        position: Int, layers: [HiddenReadoutLayer],
        path: HiddenCapturePath = HiddenCapturePath(), captureNanos: UInt64 = 0
    ) {
        self.position = position
        self.layers = layers
        self.path = path
        self.captureNanos = captureNanos
    }
}

/// The reductions from a raw fp16 residual row, kept pure so they are tested
/// without a GPU.
public enum HiddenReadoutMath {
    /// Widens a row of fp16 residual values to float32.
    public static func widen(_ row: UnsafeBufferPointer<Float16>) -> [Float] {
        row.map { Float($0) }
    }

    /// Mean over `streams` stream-major blocks of `row`: element `d` of the
    /// result is `(row[d] + row[D + d] + ...) / streams`, summed in float32 in
    /// ascending stream order. `row.count` must be a multiple of `streams`.
    public static func meanOfStreams(_ row: [Float], streams: Int) -> [Float] {
        precondition(streams > 0 && row.count % streams == 0, "row is not whole streams")
        guard streams > 1 else { return row }
        let dim = row.count / streams
        var out = [Float](repeating: 0, count: dim)
        for s in 0..<streams {
            let base = s * dim
            for d in 0..<dim { out[d] += row[base + d] }
        }
        let scale = Float(streams)
        for d in 0..<dim { out[d] /= scale }
        return out
    }

    /// The row a request asked for, from the raw fp16 residual row.
    public static func reduce(
        _ row: UnsafeBufferPointer<Float16>, streams: Int, mode: HiddenReadoutStreamMode
    ) -> [Float] {
        let widened = widen(row)
        switch mode {
        case .residual: return widened
        case .meanStreams: return meanOfStreams(widened, streams: streams)
        }
    }
}
