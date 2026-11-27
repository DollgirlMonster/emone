import CryptoKit
import Foundation

/// Writes an `emone-lora-v1` adapter package of seeded random weights over
/// every projection `LoRAAdapterFit` allows on a model (Emone-lmw.14). No
/// trained System One adapter exists for the MoE models yet, so this is what
/// the branch-adapter checks run on: it means nothing, but it has the shapes
/// a trained one will have and changes the branch's numbers visibly.
///
/// `zeroB` writes every B as zeros, so the delta is exactly +0 and a branch
/// run through it must match the unadapted branch bit for bit. The head file
/// is a placeholder: the package format pins one, nothing here reads it.
public enum SyntheticLoRAPackage {
    public static func write(
        to directory: URL, config cfg: ArchConfig, baseModelID: String,
        rank: Int = 16, seed: UInt64 = 0x5EED, zeroB: Bool = false
    ) throws {
        precondition(rank > 0 && rank <= LoRADelta.maximumRank)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var rng = LoRASeedStream(seed: seed)
        var header: [String: Any] = ["__metadata__": ["format": "emone-lora-v1"]]
        var body = Data()
        func append(_ name: String, rows: Int, columns: Int, bound: Float) {
            var values = [Float](repeating: 0, count: rows * columns)
            if bound > 0 {
                for i in values.indices { values[i] = rng.uniform(bound) }
            }
            let start = body.count
            values.withUnsafeBytes { body.append(contentsOf: $0) }
            header[name] = [
                "dtype": "F32", "shape": [rows, columns], "data_offsets": [start, body.count],
            ]
        }
        for layer in 0..<cfg.numLayers {
            for target in LoRATarget.allCases {
                guard
                    let shape = LoRAAdapterFit.expectedShape(
                        target: target, layer: layer, config: cfg)
                else { continue }
                let stem = "layers.\(layer).\(target.rawValue)"
                // A ~ U(+-1/sqrt(in)) keeps x A^T near unit scale; B small, so
                // the delta moves the branch without swamping the base.
                append(
                    "\(stem).lora_A", rows: rank, columns: shape.columns,
                    bound: 1 / Float(shape.columns).squareRoot())
                append(
                    "\(stem).lora_B", rows: shape.rows, columns: rank,
                    bound: zeroB ? 0 : 0.02)
            }
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while json.count % 8 != 0 { json.append(0x20) }
        var file = Data()
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { file.append(contentsOf: $0) }
        file.append(json)
        file.append(body)
        try file.write(to: directory.appendingPathComponent(LoRAAdapter.adapterFile))
        let head = Data("synthetic placeholder head".utf8)
        try head.write(to: directory.appendingPathComponent(LoRAAdapter.headFile))
        let manifest: [String: Any] = [
            "format": "emone-lora-v1",
            "model_id": zeroB ? "synthetic-zero-b" : "synthetic-random",
            "rank": rank, "alpha": 2 * rank,
            "base": ["model_id": baseModelID],
            "files": [LoRAAdapter.adapterFile: sha256(file), LoRAAdapter.headFile: sha256(head)],
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys]).write(
            to: directory.appendingPathComponent(LoRAAdapter.manifestFile))
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

/// Deterministic, platform-independent: the same seed writes the same package.
private struct LoRASeedStream {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [-bound, bound).
    mutating func uniform(_ bound: Float) -> Float {
        let unit = Float(next() >> 40) / Float(1 << 24)
        return (2 * unit - 1) * bound
    }
}
