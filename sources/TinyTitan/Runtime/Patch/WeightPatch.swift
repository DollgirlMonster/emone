import CryptoKit
import Foundation
import Metal

/// A hand-written weight edit laid over a base install at runtime
/// (TinyTitan-a37): low-rank deltas `y += (x A^T) B^T` on named projections,
/// applied in prefill **and** decode, so the base `.gturbo` is never rewritten
/// and the edit stays a small file someone can audit.
///
/// It shares `LoRADelta` and the runner's delta seams with the decision
/// adapter but is its own package: a trained adapter carries a pointer head
/// and is scaled by alpha/r, a patch carries neither, and a different format
/// id stops either loader from taking the other's files. Version 1 patches
/// `mlp.down_proj` only -- the one dense projection whose decode output the
/// runner can adjust after the fact; gate and up sit inside the fused shared
/// MLP kernel before its activation.
///
/// A patch may also scale whole attention heads (TinyTitan-bbm.3): an F32
/// `layers.<L>.self_attn.head_scale` vector, one factor per query head,
/// multiplies each head's output before `o_proj` -- the same as scaling that
/// head's o_proj columns, which no rank-32 delta can express. On a Gated
/// DeltaNet layer, `layers.<L>.linear_attn.head_scale` does the same per value
/// head, after the gated norm and before `out_proj`.
///
/// Package layout: `weight_patch.json` (format, base model id, rank, a
/// description of what the edit does and how it was found, and the sha256 of
/// the tensor file) and `weight_patch.safetensors` holding F32
/// `layers.<L>.<target>.lora_A` `[rank x in]` / `lora_B` `[out x rank]` pairs
/// and `layers.<L>.self_attn.head_scale` `[heads]` vectors.
///
/// unchecked-invariant: every stored property is set in `init` and never
/// mutated; the buffer is written once there and only read by the GPU
/// afterwards, and the scratch is used by the one runner it is set on.
public final class WeightPatch: @unchecked Sendable {

    public enum LoadError: Error, CustomStringConvertible {
        case notAPatch(String)
        case wrongBase(String)
        case corrupt(String)
        case unsupported(String)

        public var description: String {
            switch self {
            case .notAPatch(let s): return "not a weight patch: \(s)"
            case .wrongBase(let s): return "weight patch does not belong to this base: \(s)"
            case .corrupt(let s): return "weight patch is damaged: \(s)"
            case .unsupported(let s): return "weight patch is not supported: \(s)"
            }
        }
    }

    public static let format = "tinytitan-patch-v1"
    public static let manifestFile = "weight_patch.json"
    public static let tensorFile = "weight_patch.safetensors"
    /// Projections a version-1 patch may touch (see the type comment).
    public static let supportedTargets: Set<LoRATarget> = [.down]
    static let headScaleSuffixes = [".self_attn.head_scale", ".linear_attn.head_scale"]

    struct Entry {
        let aOffset: Int
        let bOffset: Int
        let rows: Int
        let columns: Int
    }

    public let directory: URL
    public let baseModelID: String
    public let summary: String
    public let rank: Int
    let buffer: MTLBuffer
    let scratch: MTLBuffer
    let maxTokens: Int
    private let entries: [Int: [LoRATarget: Entry]]
    /// Per layer: the byte offset of its head-scale vector, its length, and
    /// whether it names DeltaNet value heads (`linear_attn`).
    private let headScales: [Int: (offset: Int, heads: Int, linear: Bool)]

    public var targetCount: Int { entries.values.reduce(0) { $0 + $1.count } }
    public var layers: [Int] { entries.keys.sorted() }
    public var headScaleLayers: [Int] { headScales.keys.sorted() }

    func entry(layer: Int, target: LoRATarget) -> Entry? { entries[layer]?[target] }
    func headScale(layer: Int) -> (offset: Int, heads: Int, linear: Bool)? { headScales[layer] }

    /// Reads and checks a package against the install it is to run over.
    /// `maxTokens` is the longest prefill chunk the scratch must hold.
    public init(
        directory: URL, baseModelID: String, device: MTLDevice, maxTokens: Int
    ) throws {
        let manifestURL = directory.appendingPathComponent(Self.manifestFile)
        guard let manifestData = try? Data(contentsOf: manifestURL),
            let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
            manifest["format"] as? String == Self.format,
            let base = manifest["base"] as? [String: Any],
            let baseID = base["model_id"] as? String,
            let rank = manifest["rank"] as? Int, rank > 0, rank <= LoRADelta.maximumRank,
            let files = manifest["files"] as? [String: String],
            let pinned = files[Self.tensorFile]
        else { throw LoadError.notAPatch("\(manifestURL.path) is missing or unreadable") }
        guard baseID == baseModelID else {
            throw LoadError.wrongBase(
                "\(directory.lastPathComponent) is for \(baseID), not \(baseModelID)")
        }
        let tensorURL = directory.appendingPathComponent(Self.tensorFile)
        let data = try Data(contentsOf: tensorURL)
        let got = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard got == pinned.lowercased() else {
            throw LoadError.corrupt(
                "\(Self.tensorFile) is sha256 \(got), the manifest pins \(pinned)")
        }
        let (header, payloadStart) = try Self.header(data)
        let payloadCount = data.count - payloadStart
        guard
            let made = data.withUnsafeBytes({ raw -> MTLBuffer? in
                guard let start = raw.baseAddress else { return nil }
                return device.makeBuffer(
                    bytes: start + payloadStart, length: max(payloadCount, 4),
                    options: .storageModeShared)
            }),
            let scratch = device.makeBuffer(
                length: LoRADelta.maximumGroup * max(maxTokens, 1) * rank
                    * MemoryLayout<Float>.stride,
                options: .storageModePrivate)
        else { throw MetalError.bufferAllocationFailed("weight patch") }
        let (pairHeader, scales) = try Self.headScales(header, payloadCount: payloadCount)
        var entries: [Int: [LoRATarget: Entry]] = [:]
        let floats = made.contents().bindMemory(to: Float.self, capacity: payloadCount / 4)
        for (layer, target, a, b) in try Self.pairs(
            pairHeader, rank: rank, payloadCount: payloadCount)
        {
            // Stored [out x r]; `lora_up_fp16` reads B transposed, [r x out].
            let rows = b.shape[0]
            let start = floats + b.offset / 4
            let copy = Array(UnsafeBufferPointer(start: start, count: rows * rank))
            for i in 0..<rows {
                for j in 0..<rank { start[j * rows + i] = copy[i * rank + j] }
            }
            entries[layer, default: [:]][target] = Entry(
                aOffset: a.offset, bOffset: b.offset, rows: rows, columns: a.shape[1])
        }
        for (layer, scale) in scales {
            let values = UnsafeBufferPointer(start: floats + scale.offset / 4, count: scale.heads)
            guard values.allSatisfy({ $0.isFinite }) else {
                throw LoadError.corrupt("layer \(layer) head_scale holds a non-finite factor")
            }
        }
        guard !entries.isEmpty || !scales.isEmpty else {
            throw LoadError.corrupt("the patch holds no deltas and no head scales")
        }
        self.directory = directory
        self.baseModelID = baseID
        self.summary = manifest["description"] as? String ?? ""
        self.rank = rank
        self.buffer = made
        self.scratch = scratch
        self.maxTokens = maxTokens
        self.entries = entries
        self.headScales = scales
    }

    /// Takes the `head_scale` vectors out of the header: F32, one dimension,
    /// in bounds, finite. Returns the rest of the header for `pairs`.
    private static func headScales(
        _ header: [String: Any], payloadCount: Int
    ) throws -> ([String: Any], [Int: (offset: Int, heads: Int, linear: Bool)]) {
        var rest = header
        var scales: [Int: (offset: Int, heads: Int, linear: Bool)] = [:]
        for (name, value) in header {
            guard let suffix = headScaleSuffixes.first(where: { name.hasSuffix($0) }) else {
                continue
            }
            let fields = name.dropLast(suffix.count).split(separator: ".")
            guard fields.count == 2, fields[0] == "layers", let layer = Int(fields[1]),
                let entry = value as? [String: Any], entry["dtype"] as? String == "F32",
                let shape = entry["shape"] as? [Int], shape.count == 1, shape[0] > 0,
                let offsets = entry["data_offsets"] as? [Int], offsets.count == 2,
                offsets[1] - offsets[0] == shape[0] * 4, offsets[1] <= payloadCount,
                offsets[0] % 4 == 0
            else { throw LoadError.corrupt("\(name) is not an in-bounds F32 vector") }
            guard scales[layer] == nil else {
                throw LoadError.corrupt("layer \(layer) has two head_scale vectors")
            }
            scales[layer] = (offsets[0], shape[0], suffix.hasPrefix(".linear_attn"))
            rest.removeValue(forKey: name)
        }
        return (rest, scales)
    }

    private struct Tensor {
        let offset: Int
        let shape: [Int]
    }

    private static func header(_ data: Data) throws -> ([String: Any], Int) {
        guard data.count >= 8 else { throw LoadError.corrupt("tensor file too short") }
        let length = data.withUnsafeBytes {
            Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)))
        }
        guard length > 0, 8 + length <= data.count,
            let header = try JSONSerialization.jsonObject(with: data[8..<(8 + length)])
                as? [String: Any],
            (header["__metadata__"] as? [String: String])?["format"] == Self.format
        else { throw LoadError.corrupt("unreadable safetensors header") }
        return (header, 8 + length)
    }

    /// Every `layers.<L>.<target>.lora_A` / `lora_B` pair: F32, A
    /// `[rank x in]`, B `[out x rank]`, in bounds, on a supported target.
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
            var matched = false
            for part in ["A", "B"] where name.hasSuffix(".lora_\(part)") {
                parts[String(name.dropLast(7)), default: [:]][part] = Tensor(
                    offset: offsets[0], shape: shape)
                matched = true
            }
            guard matched else { throw LoadError.corrupt("\(name) is neither lora_A nor lora_B") }
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
            guard supportedTargets.contains(target) else {
                throw LoadError.unsupported(
                    "\(stem): version 1 patches \(supportedTargets.map(\.rawValue).sorted()) only")
            }
            return (layer, target, a, b)
        }
    }
}
