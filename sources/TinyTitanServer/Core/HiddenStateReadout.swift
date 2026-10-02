import Foundation
import TinyTitan

/// Hidden-state readout on `POST /v1/chat/completions`: an opt-in request
/// extension that returns the residual stream of the last prompt token after
/// chosen layers, for fitting linear probes. It is a request extension, not a
/// new endpoint, so a client that does not send `x_hidden_states` sees nothing
/// of it.
///
/// ```json
/// "x_hidden_states": {"layers": [12, 24, 36, 47], "stream_mode": "residual",
///                     "prefill_only": true, "cache": "bypass",
///                     "encoding": "base64_f32"}
/// ```
///
/// Two modes. `prefill_only: true` (the default, with `cache: "bypass"`) is a
/// probe: prefill, read, discard. `prefill_only: false` (with `cache: "reuse"`,
/// its default) captures during an ordinary generation, which streams, resumes
/// and publishes its prompt cache like any other request. The other fields
/// have one supported value each, still named so that a later mode is an
/// addition rather than a change of meaning. `docs/server-api.md` is the
/// reference.
public struct OpenAIHiddenStatesRequest: Codable, Equatable, Sendable {
    /// 0-based block indices.
    public let layers: [Int]?
    /// `residual` (default) or `mean_streams`.
    public let streamMode: String?
    /// `true` (default): stop after the deepest layer and answer with the
    /// readout alone. `false`: capture during an ordinary generation.
    public let prefillOnly: Bool?
    /// `bypass` (default with `prefill_only: true`) or `reuse` (default with
    /// `prefill_only: false`); the other pairing is refused.
    public let cache: String?
    /// Only `base64_f32` is supported; omitted means `base64_f32`.
    public let encoding: String?
    /// Only `last` is supported; omitted means `last`.
    public let positions: String?

    public init(
        layers: [Int]? = nil, streamMode: String? = nil, prefillOnly: Bool? = nil,
        cache: String? = nil, encoding: String? = nil, positions: String? = nil
    ) {
        self.layers = layers
        self.streamMode = streamMode
        self.prefillOnly = prefillOnly
        self.cache = cache
        self.encoding = encoding
        self.positions = positions
    }

    enum CodingKeys: String, CodingKey {
        case layers, cache, encoding, positions
        case streamMode = "stream_mode"
        case prefillOnly = "prefill_only"
    }
}

extension OpenAIRequestValidator {
    /// Checks one `x_hidden_states` object and turns it into the engine's plan.
    ///
    /// The upper bound on a layer is the served model's block count where the
    /// family has one fixed geometry (Qwen3.8 Flash: 48). A family whose
    /// geometry comes from the install's manifest (the dense Qwen 3.5 models)
    /// is checked again, against the loaded model, when the request runs.
    static func validateHiddenStates(
        _ request: OpenAIHiddenStatesRequest,
        stream: Bool,
        family: ModelFamily
    ) throws -> HiddenReadoutPlan {
        func refuse(_ message: String, _ field: String, _ code: String = "unsupported_value")
            -> ServerRequestError
        {
            .invalid(message: message, param: "x_hidden_states.\(field)", code: code)
        }
        guard let layers = request.layers, !layers.isEmpty else {
            throw refuse("layers must name at least one layer", "layers", "invalid_value")
        }
        let streamMode: HiddenReadoutStreamMode
        if let raw = request.streamMode {
            guard let mode = HiddenReadoutStreamMode(rawValue: raw) else {
                throw refuse(
                    "stream_mode must be \"residual\" or \"mean_streams\"", "stream_mode")
            }
            streamMode = mode
        } else {
            streamMode = .residual
        }
        let prefillOnly = request.prefillOnly ?? true
        // A probe discards the live sequence, so it cannot reuse the cache; a
        // capture rides an ordinary generation, so it cannot bypass it.
        let cache = request.cache ?? (prefillOnly ? "bypass" : "reuse")
        guard cache == "bypass" || cache == "reuse" else {
            throw refuse("cache must be \"bypass\" or \"reuse\"", "cache")
        }
        guard cache == (prefillOnly ? "bypass" : "reuse") else {
            throw refuse(
                prefillOnly
                    ? "prefill_only=true answers from a throwaway prefill and always "
                        + "bypasses the cache; use prefill_only=false with cache=\"reuse\" "
                        + "to capture during a generation"
                    : "cache=\"bypass\" needs prefill_only=true; a capture during a "
                        + "generation uses the cache like any request",
                "cache")
        }
        // A probe is one JSON object with a vector per layer, not a stream of
        // deltas; a capture during a generation streams like any request, with
        // the vectors on the last data chunk.
        guard !(stream && prefillOnly) else {
            throw ServerRequestError.invalid(
                message: "x_hidden_states with prefill_only=true cannot be combined with "
                    + "stream=true; use prefill_only=false to capture while streaming",
                param: "stream", code: "unsupported_value")
        }
        guard (request.encoding ?? "base64_f32") == "base64_f32" else {
            throw refuse("only encoding=\"base64_f32\" is supported", "encoding")
        }
        guard (request.positions ?? "last") == "last" else {
            throw refuse("only positions=\"last\" is supported", "positions")
        }
        do {
            let plan = try HiddenReadoutPlan(
                layers: layers, streamMode: streamMode, prefillOnly: prefillOnly)
            if let numLayers = ArchConfig.knownArchitectures[family]?.numLayers {
                try plan.validate(numLayers: numLayers)
            }
            return plan
        } catch let error as HiddenReadoutError {
            throw refuse(error.description, "layers", "invalid_value")
        }
    }
}

/// What a readout returns, ready to be put in the response.
///
/// `hidden_states` is an object keyed by layer index (as a string) whose
/// values describe one `[1, dim]` float32 row each, plus `position`, the
/// absolute index of the token that was read:
///
/// ```json
/// "hidden_states": {"12": {"dim": 10240, "dtype": "float32", "shape": [1, 10240],
///                          "data": "<base64 little-endian float32>"},
///                   "position": 41,
///                   "capture_path": {"runner": "plain", "prefill": "gpu",
///                                    "early_stop": false},
///                   "capture_ms": 0.4}
/// ```
public struct HiddenStatesPayload: Equatable, Sendable {
    public let result: HiddenReadoutResult

    public init(_ result: HiddenReadoutResult) {
        self.result = result
    }

    /// Float32 values as base64 of their little-endian bytes.
    public static func base64Float32(_ values: [Float]) -> String {
        var data = Data(capacity: values.count * MemoryLayout<UInt32>.size)
        for value in values {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        return data.base64EncodedString()
    }

    /// The `hidden_states` response object.
    public func jsonObject() -> [String: Any] {
        var object: [String: Any] = [:]
        for layer in result.layers {
            object[String(layer.layer)] = [
                "dim": layer.values.count,
                "dtype": "float32",
                "shape": [1, layer.values.count],
                "data": Self.base64Float32(layer.values),
            ] as [String: Any]
        }
        object["position"] = result.position
        object["capture_path"] = [
            "runner": "plain",
            "prefill": result.path.usedANE ? "ane" : "gpu",
            "early_stop": result.path.earlyStop,
        ] as [String: Any]
        object["capture_ms"] = Double(result.captureNanos) / 1e6
        return object
    }
}
