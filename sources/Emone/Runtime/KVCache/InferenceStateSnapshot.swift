import Foundation

/// Serializable description of every persistent inference-state buffer needed
/// to resume a prompt without replaying its cached tokens.
///
/// Version 2 adds Qwen3.8-Flash-Next's per-sequence state beyond K/V and GDN:
/// the sparse indexer's raw and pooled keys (`qsaLayers`,
/// `qsaSegmentLengths`) and the n-gram block's convolution window and hashed
/// token context (`pleSegmentLengths`, `pleContext`). Without them a restore
/// would rank blocks from the previous conversation's keys, so version 1 is
/// still accepted only by a runtime that has neither subsystem -- which keeps
/// the MoE families' saved caches valid across the upgrade.
public struct InferenceStateSnapshotDescriptor: Codable, Equatable, Sendable {
    public static let currentVersion = 2
    public static let supportedVersions: Set<Int> = [1, 2]

    public let version: Int
    public let position: Int
    public let kvSegmentLengths: [Int]
    public let gdnSegmentLengths: [Int]
    /// Indexer layers in payload order; two segments each: raw keys, then
    /// pooled blocks.
    public let qsaLayers: [Int]
    public let qsaSegmentLengths: [Int]
    /// The n-gram block's carried convolution window (one segment when present).
    public let pleSegmentLengths: [Int]
    /// The n-gram block's hashed token context, newest first.
    public let pleContext: [Int32]
    public let payloadBytes: Int

    public init(
        version: Int = currentVersion,
        position: Int,
        kvSegmentLengths: [Int],
        gdnSegmentLengths: [Int],
        qsaLayers: [Int] = [],
        qsaSegmentLengths: [Int] = [],
        pleSegmentLengths: [Int] = [],
        pleContext: [Int32] = [],
        payloadBytes: Int
    ) {
        self.version = version
        self.position = position
        self.kvSegmentLengths = kvSegmentLengths
        self.gdnSegmentLengths = gdnSegmentLengths
        self.qsaLayers = qsaLayers
        self.qsaSegmentLengths = qsaSegmentLengths
        self.pleSegmentLengths = pleSegmentLengths
        self.pleContext = pleContext
        self.payloadBytes = payloadBytes
    }

    private enum CodingKeys: String, CodingKey {
        case version, position, kvSegmentLengths, gdnSegmentLengths
        case qsaLayers, qsaSegmentLengths, pleSegmentLengths, pleContext, payloadBytes
    }

    /// Version 1 metadata on disk has no indexer or n-gram fields; they decode
    /// as empty, and `RealForwardRunner.restoreInferenceState` refuses such a
    /// snapshot on a runtime that needs them.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        version = try c.decode(Int.self, forKey: .version)
        position = try c.decode(Int.self, forKey: .position)
        kvSegmentLengths = try c.decode([Int].self, forKey: .kvSegmentLengths)
        gdnSegmentLengths = try c.decode([Int].self, forKey: .gdnSegmentLengths)
        qsaLayers = try c.decodeIfPresent([Int].self, forKey: .qsaLayers) ?? []
        qsaSegmentLengths = try c.decodeIfPresent([Int].self, forKey: .qsaSegmentLengths) ?? []
        pleSegmentLengths = try c.decodeIfPresent([Int].self, forKey: .pleSegmentLengths) ?? []
        pleContext = try c.decodeIfPresent([Int32].self, forKey: .pleContext) ?? []
        payloadBytes = try c.decode(Int.self, forKey: .payloadBytes)
    }
}

/// In-memory inference-state checkpoint. The payload is a concatenation of the
/// K/V segments, the Qwen gated-DeltaNet recurrent-state segments, the sparse
/// indexer's segments and the n-gram block's window, in that order.
public struct InferenceStateSnapshot: Equatable, Sendable {
    public let descriptor: InferenceStateSnapshotDescriptor
    public let payload: Data

    public init(descriptor: InferenceStateSnapshotDescriptor, payload: Data) {
        self.descriptor = descriptor
        self.payload = payload
    }
}

public enum InferenceStateSnapshotError: Error, Equatable, CustomStringConvertible {
    case unsupportedVersion(Int)
    case invalidPosition(Int)
    case invalidLayout
    case invalidPayloadSize(expected: Int, actual: Int)
    case exceedsLimit(bytes: Int, limit: Int)
    case integerOverflow
    /// A live subsystem's state is not carried by the snapshot, so a restore
    /// would leave the previous conversation's buffers mixed into this prefix.
    case stateNotInSnapshot(String)

    public var description: String {
        switch self {
        case .unsupportedVersion(let version):
            "unsupported inference-state snapshot version: \(version)"
        case .invalidPosition(let position):
            "invalid inference-state snapshot position: \(position)"
        case .invalidLayout:
            "inference-state snapshot layout does not match the loaded runtime"
        case .invalidPayloadSize(let expected, let actual):
            "inference-state snapshot payload is \(actual) bytes; expected \(expected)"
        case .exceedsLimit(let bytes, let limit):
            "inference-state snapshot requires \(bytes) bytes; cache limit is \(limit)"
        case .integerOverflow:
            "inference-state snapshot size overflow"
        case .stateNotInSnapshot(let feature):
            "inference-state snapshot does not carry \(feature) state, and restoring "
                + "would leave the previous conversation's buffers in place"
        }
    }
}

extension InferenceStateSnapshotDescriptor {
    public func validatedPayloadBytes() throws -> Int {
        var total = 0
        guard qsaSegmentLengths.count == 2 * qsaLayers.count, pleSegmentLengths.count <= 1 else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        for length in kvSegmentLengths + gdnSegmentLengths + qsaSegmentLengths + pleSegmentLengths {
            guard length >= 0 else { throw InferenceStateSnapshotError.invalidLayout }
            let (next, overflow) = total.addingReportingOverflow(length)
            guard !overflow else { throw InferenceStateSnapshotError.integerOverflow }
            total = next
        }
        guard total == payloadBytes else {
            throw InferenceStateSnapshotError.invalidPayloadSize(
                expected: total,
                actual: payloadBytes)
        }
        return total
    }
}
