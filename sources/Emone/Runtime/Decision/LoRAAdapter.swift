import CryptoKit
import Foundation
import Metal

/// A projection a runtime-LoRA adapter can target, by its checkpoint path
/// inside a decoder layer.
public enum LoRATarget: String, CaseIterable, Sendable {
    case q = "self_attn.q_proj"
    case k = "self_attn.k_proj"
    case v = "self_attn.v_proj"
    case o = "self_attn.o_proj"
    case gdnQKV = "linear_attn.in_proj_qkv"
    case gdnZ = "linear_attn.in_proj_z"
    case gdnA = "linear_attn.in_proj_a"
    case gdnB = "linear_attn.in_proj_b"
    case gdnOut = "linear_attn.out_proj"
    case gate = "mlp.gate_proj"
    case up = "mlp.up_proj"
    case down = "mlp.down_proj"
    /// An MoE layer's shared expert: the one MLP every token runs, beside the
    /// routed experts (Emone-lmw.14). The routed experts are not targets.
    case sharedGate = "mlp.shared_expert.gate_proj"
    case sharedUp = "mlp.shared_expert.up_proj"
    case sharedDown = "mlp.shared_expert.down_proj"
}

/// A decision adapter applied at request time over a base install's weights
/// (Emone-lmw.11): every A `[r x in]` and B `[out x r]` resident once at
/// fp32, laid over the base projections' outputs by `LoRADelta`. Read from a
/// package `tools/prepare_qwen35.py --adapter-only` writes: the adapter, the
/// pointer head and a manifest naming the base and pinning both files.
///
/// unchecked-invariant: every stored property is set in `init` and never
/// mutated; the adapter buffer is written once there and only read by the
/// GPU afterwards, and the scratch is used by the one runner it is set on.
public final class LoRAAdapter: @unchecked Sendable {

    public enum LoadError: Error, CustomStringConvertible {
        case notAnAdapter(String)
        case wrongBase(String)
        case corrupt(String)

        public var description: String {
            switch self {
            case .notAnAdapter(let s): return "not a decision adapter package: \(s)"
            case .wrongBase(let s): return "adapter does not belong to this base: \(s)"
            case .corrupt(let s): return "adapter package is damaged: \(s)"
            }
        }
    }

    public static let adapterFile = "decision_adapter.safetensors"
    public static let manifestFile = "decision_adapter.json"
    public static let headFile = "decision_head.safetensors"

    struct Entry {
        let aOffset: Int
        let bOffset: Int
        let rows: Int
        let columns: Int
    }

    public let directory: URL
    public let modelID: String
    public let baseModelID: String
    public let rank: Int
    public let scale: Float
    let buffer: MTLBuffer
    let scratch: MTLBuffer
    let maxTokens: Int
    private let entries: [Int: [LoRATarget: Entry]]

    public var targetCount: Int { entries.values.reduce(0) { $0 + $1.count } }

    func entry(layer: Int, target: LoRATarget) -> Entry? { entries[layer]?[target] }

    /// Every adapted projection with its shape, for `LoRAAdapterFit`.
    public var declared: [LoRAAdapterFit.Declared] {
        entries.flatMap { layer, targets in
            targets.map { target, entry in
                LoRAAdapterFit.Declared(
                    layer: layer, target: target, rows: entry.rows, columns: entry.columns)
            }
        }
    }

    /// Reads and checks a package. `baseModelID` is the install the adapter
    /// is to run over; a package naming another base is refused.
    public init(
        directory: URL, baseModelID: String, device: MTLDevice, maxTokens: Int
    ) throws {
        let manifestURL = directory.appendingPathComponent(Self.manifestFile)
        guard let manifestData = try? Data(contentsOf: manifestURL),
            let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
            manifest["format"] as? String == "emone-lora-v1",
            let base = manifest["base"] as? [String: Any],
            let baseID = base["model_id"] as? String,
            let modelID = manifest["model_id"] as? String,
            let rank = manifest["rank"] as? Int, rank > 0, rank <= LoRADelta.maximumRank,
            let alpha = manifest["alpha"] as? Int,
            let files = manifest["files"] as? [String: String]
        else { throw LoadError.notAnAdapter("\(manifestURL.path) is missing or unreadable") }
        guard baseID == baseModelID else {
            throw LoadError.wrongBase(
                "\(directory.lastPathComponent) is for \(baseID), not \(baseModelID)")
        }
        for name in [Self.adapterFile, Self.headFile] {
            guard let pinned = files[name] else {
                throw LoadError.notAnAdapter("the manifest pins no digest for \(name)")
            }
            let got = try Self.sha256(of: directory.appendingPathComponent(name))
            guard got == pinned.lowercased() else {
                throw LoadError.corrupt("\(name) is sha256 \(got), the manifest pins \(pinned)")
            }
        }
        let data = try Data(contentsOf: directory.appendingPathComponent(Self.adapterFile))
        let (header, payloadStart) = try Self.header(data)
        let payloadCount = data.count - payloadStart
        guard
            let made = data.withUnsafeBytes({ raw -> MTLBuffer? in
                guard let base = raw.baseAddress else { return nil }
                return device.makeBuffer(
                    bytes: base + payloadStart, length: max(payloadCount, 4),
                    options: .storageModeShared)
            }),
            let scratch = device.makeBuffer(
                // A grouped delta keeps every member's u at once.
                length: LoRADelta.maximumGroup * max(maxTokens, 1) * rank
                    * MemoryLayout<Float>.stride,
                options: .storageModePrivate)
        else { throw MetalError.bufferAllocationFailed("LoRA adapter") }
        var entries: [Int: [LoRATarget: Entry]] = [:]
        let floats = made.contents().bindMemory(to: Float.self, capacity: payloadCount / 4)
        for (layer, target, a, b) in try Self.pairs(header, rank: rank, payloadCount: payloadCount)
        {
            // B is stored [out x r]; `lora_up_fp16` reads it transposed,
            // [r x out], so a simdgroup's lanes read adjacent output rows.
            let rows = b.shape[0]
            let base = floats + b.offset / 4
            let copy = Array(UnsafeBufferPointer(start: base, count: rows * rank))
            for i in 0..<rows {
                for j in 0..<rank { base[j * rows + i] = copy[i * rank + j] }
            }
            entries[layer, default: [:]][target] = Entry(
                aOffset: a.offset, bOffset: b.offset, rows: rows, columns: a.shape[1])
        }
        self.directory = directory
        self.modelID = modelID
        self.baseModelID = baseID
        self.rank = rank
        self.scale = Float(alpha) / Float(rank)
        self.buffer = made
        self.scratch = scratch
        self.maxTokens = maxTokens
        self.entries = entries
    }

    private struct Tensor {
        let offset: Int
        let shape: [Int]
    }

    private static func header(_ data: Data) throws -> ([String: Any], Int) {
        guard data.count >= 8 else { throw LoadError.corrupt("adapter file too short") }
        let length = data.withUnsafeBytes {
            Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)))
        }
        guard length > 0, 8 + length <= data.count,
            let header = try JSONSerialization.jsonObject(with: data[8..<(8 + length)])
                as? [String: Any],
            (header["__metadata__"] as? [String: String])?["format"] == "emone-lora-v1"
        else { throw LoadError.corrupt("unreadable safetensors header") }
        return (header, 8 + length)
    }

    /// Every `layers.<L>.<target>.lora_A` / `lora_B` pair, checked: F32, A
    /// `[rank x in]`, B `[out x rank]`, inside the payload, and nothing else.
    private static func pairs(
        _ header: [String: Any], rank: Int, payloadCount: Int
    ) throws -> [(Int, LoRATarget, Tensor, Tensor)] {
        var parts: [String: [String: Tensor]] = [:]
        for (name, value) in header where name != "__metadata__" {
            guard let entry = value as? [String: Any], entry["dtype"] as? String == "F32",
                let shape = entry["shape"] as? [Int], shape.count == 2,
                let offsets = entry["data_offsets"] as? [Int], offsets.count == 2,
                offsets[1] - offsets[0] == shape[0] * shape[1] * 4, offsets[1] <= payloadCount,
                offsets[0] % 4 == 0
            else { throw LoadError.corrupt("\(name) is not an in-bounds F32 matrix") }
            for part in ["A", "B"] where name.hasSuffix(".lora_\(part)") {
                parts[String(name.dropLast(7)), default: [:]][part] = Tensor(
                    offset: offsets[0], shape: shape)
            }
        }
        return try parts.map { stem, pair in
            let fields = stem.split(separator: ".", maxSplits: 2)
            guard fields.count == 3, fields[0] == "layers", let layer = Int(fields[1]),
                let target = LoRATarget(rawValue: String(fields[2])),
                let a = pair["A"], let b = pair["B"],
                a.shape[0] == rank, b.shape[1] == rank
            else {
                throw LoadError.corrupt("\(stem) is not a rank-\(rank) A/B pair of a known target")
            }
            return (layer, target, a, b)
        }
    }

    private static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let block = try handle.read(upToCount: 1 << 20), !block.isEmpty {
            hasher.update(data: block)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
