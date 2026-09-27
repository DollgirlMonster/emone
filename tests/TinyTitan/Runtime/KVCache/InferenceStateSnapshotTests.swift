import Foundation
import Metal
import Testing

@testable import TinyTitan

/// Qwen3.8-Flash-Next's per-sequence state beyond K/V and GDN -- the sparse
/// indexer's keys and the n-gram block's window -- travels in a version 2
/// snapshot. Without it a restore was refused outright, so every cache miss
/// on that family was a full re-prefill.
@Suite("Inference-state snapshot v2")
struct InferenceStateSnapshotTests {
    @Test("Version 1 metadata still decodes, with no indexer or n-gram state")
    func version1Decodes() throws {
        let json =
            #"{"version":1,"position":8,"kvSegmentLengths":[16],"#
            + #""gdnSegmentLengths":[4],"payloadBytes":20}"#
        let d = try JSONDecoder().decode(
            InferenceStateSnapshotDescriptor.self, from: Data(json.utf8))
        #expect(d.version == 1 && d.qsaLayers.isEmpty && d.pleSegmentLengths.isEmpty)
        #expect(try d.validatedPayloadBytes() == 20)
        #expect(InferenceStateSnapshotDescriptor.supportedVersions.contains(1))
    }

    @Test("Version 2 round-trips and counts every segment")
    func version2RoundTrips() throws {
        let d = InferenceStateSnapshotDescriptor(
            position: 8, kvSegmentLengths: [16], gdnSegmentLengths: [4],
            qsaLayers: [3, 7], qsaSegmentLengths: [10, 2, 10, 2],
            pleSegmentLengths: [6], pleContext: [5, 4, 3], payloadBytes: 50)
        let back = try JSONDecoder().decode(
            InferenceStateSnapshotDescriptor.self, from: JSONEncoder().encode(d))
        #expect(back == d && back.version == 2)
        #expect(try back.validatedPayloadBytes() == 50)
    }

    @Test("Indexer segments must pair with indexer layers")
    func mismatchedLayoutIsRefused() {
        let d = InferenceStateSnapshotDescriptor(
            position: 8, kvSegmentLengths: [16], gdnSegmentLengths: [],
            qsaLayers: [3], qsaSegmentLengths: [10], payloadBytes: 26)
        #expect(throws: InferenceStateSnapshotError.invalidLayout) {
            try d.validatedPayloadBytes()
        }
    }

    private static let config = SparseIndexerConfig(
        numHeads: 4, numKVHeads: 1, headDim: 128, budget: 64, compressRatio: 4)

    /// Deterministic bytes, so a round trip that drops or shifts one shows.
    private static func pattern(_ count: Int, seed: UInt8) -> [UInt8] {
        (0..<count).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ Int(seed)) }
    }

    @Test("Indexer state restores byte for byte, including a partial tail block")
    func indexerRoundTrip() throws {
        let ctx = try MetalContext()
        let indexer = try QSAIndexer(
            context: ctx, config: Self.config, ropeTheta: 10_000_000, capacity: 256)
        let position = 37  // 9 full blocks and a tail of 1
        let raw = position * 128 * 2
        let pooled = 10 * 128 * 2
        let layers = [3, 11]
        let lengths = [raw, pooled, raw, pooled]
        let bytes = lengths.enumerated().flatMap { Self.pattern($1, seed: UInt8($0)) }
        var offset = 0
        try bytes.withUnsafeBytes {
            try indexer.restoreSnapshot(
                layers: layers, segmentLengths: lengths, position: position,
                bytes: $0, offset: &offset)
        }
        #expect(offset == bytes.count)
        #expect(indexer.snapshotLayers == layers)
        #expect(indexer.snapshotSegmentLengths(position: position) == lengths)
        var captured = Data()
        indexer.appendSnapshotPayload(to: &captured, position: position)
        #expect([UInt8](captured) == bytes)
    }

    @Test("An indexer snapshot of the wrong size is refused")
    func indexerRejectsWrongLengths() throws {
        let indexer = try QSAIndexer(
            context: try MetalContext(), config: Self.config,
            ropeTheta: 10_000_000, capacity: 256)
        let bytes = [UInt8](repeating: 1, count: 64)
        var offset = 0
        #expect(throws: InferenceStateSnapshotError.invalidLayout) {
            try bytes.withUnsafeBytes {
                try indexer.restoreSnapshot(
                    layers: [0], segmentLengths: [32, 32], position: 8,
                    bytes: $0, offset: &offset)
            }
        }
    }

    @Test("The n-gram block's convolution window restores byte for byte")
    func pleWindowRoundTrip() throws {
        let block = try PLEBlock(
            context: try MetalContext(), dim: 8, streams: 2, embedDim: 16,
            kernelSize: 4, dilation: 2)
        let bytes = Self.pattern(block.windowBytes, seed: 9)
        var offset = 0
        try bytes.withUnsafeBytes { try block.restoreWindow(from: $0, offset: &offset) }
        #expect(offset == block.windowBytes)
        var captured = Data()
        block.appendWindow(to: &captured)
        #expect([UInt8](captured) == bytes)
    }
}
