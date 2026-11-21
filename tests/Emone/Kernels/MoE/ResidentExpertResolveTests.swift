import Metal
import Testing

@testable import Emone

/// The pieces resident decode (`RealForwardRunner+ResidentDecode`) rests on:
/// the cache's full-residency snapshot, the kernel that resolves a router's
/// picks through it, and the argument-buffer layout that kernel writes.
@Suite struct ResidentExpertResolveTests {
    private static let experts = PreadExpertStreamerTests.numExperts

    private func makeStreamer(
        _ context: MetalContext, url: URL, slots: Int
    ) throws -> PreadExpertStreamer {
        try PreadExpertStreamer(
            layout: PreadExpertStreamerTests.makeLayout(path: url.path),
            device: context.device,
            slotCount: slots)
    }

    @Test func fullResidencyAppearsOnlyOnceEveryExpertIsLoaded() throws {
        let url = try PreadExpertStreamerTests.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let context = try MetalContext()
        let streamer = try makeStreamer(context, url: url, slots: Self.experts)

        guard case .notFull(let emptyEpoch) = streamer.fullResidency(knownEpoch: nil) else {
            Issue.record("an empty cache reported full residency")
            return
        }
        #expect(streamer.missingExperts(limit: 2) == [0, 1])
        guard case .unchanged = streamer.fullResidency(knownEpoch: emptyEpoch) else {
            Issue.record("an untouched map did not read as unchanged")
            return
        }

        // Out of id order, so a table indexed by slot would be wrong.
        let order = Array((0..<Self.experts).reversed())
        let loaded = try streamer.loadExpertsCached(experts: order)
        #expect(streamer.missingExperts(limit: Self.experts).isEmpty)
        guard
            case .full(let epoch, let addresses, let buffers) = streamer.fullResidency(
                knownEpoch: emptyEpoch)
        else {
            Issue.record("a fully loaded cache did not report full residency")
            return
        }
        #expect(epoch != emptyEpoch)
        #expect(buffers.count == Self.experts)
        for (index, expert) in order.enumerated() {
            #expect(addresses[expert] == loaded[index].buffer.gpuAddress + loaded[index].offset)
        }
        guard case .unchanged = streamer.fullResidency(knownEpoch: epoch) else {
            Issue.record("a stable full map did not read as unchanged")
            return
        }
        // An all-hit plan moves nothing, so the table a caller holds stays valid.
        _ = try streamer.loadExpertsCached(experts: [1, 2])
        guard case .unchanged = streamer.fullResidency(knownEpoch: epoch) else {
            Issue.record("an all-hit plan changed the map's epoch")
            return
        }
    }

    @Test func aCacheSmallerThanTheLayerIsNeverFullAndNeverWarmed() throws {
        let url = try PreadExpertStreamerTests.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let context = try MetalContext()
        let streamer = try makeStreamer(context, url: url, slots: Self.experts - 1)
        _ = try streamer.loadExpertsCached(experts: Array(0..<(Self.experts - 1)))
        guard case .notFull = streamer.fullResidency(knownEpoch: nil) else {
            Issue.record("a cache without a slot per expert reported full residency")
            return
        }
        #expect(streamer.missingExperts(limit: Self.experts).isEmpty)
    }

    @Test func residentUseCountsAsHits() throws {
        let url = try PreadExpertStreamerTests.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let context = try MetalContext()
        let streamer = try makeStreamer(context, url: url, slots: Self.experts)
        _ = try streamer.loadExpertsCached(experts: Array(0..<Self.experts))
        let before = streamer.statistics()
        streamer.recordResidentUse(experts: [0, 0, 1])
        let after = streamer.statistics()
        #expect(after.plans == before.plans + 1)
        #expect(after.requestedExperts == before.requestedExperts + 3)
        #expect(after.hits == before.hits + 3)
        #expect(after.misses == before.misses)
    }

    @Test func resolveKernelWritesThePickedAddressesAndLogsTheIDs() throws {
        let url = try PreadExpertStreamerTests.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let context = try MetalContext()
        let streamer = try makeStreamer(context, url: url, slots: Self.experts)
        _ = try streamer.loadExpertsCached(experts: Array((0..<Self.experts).reversed()))
        guard case .full(_, let addresses, _) = streamer.fullResidency(knownEpoch: nil) else {
            Issue.record("cache not full")
            return
        }
        let moe = try MoE(
            context: context, siluActivation: true,
            specializedD: 2048, specializedF: 512,
            specializedNumExperts: UInt32(Self.experts))
        let topK = moe.maxStreamedExperts
        let last = UInt32(Self.experts - 1)
        // One id past the range: the kernel clamps it as the CPU path does.
        let picks: [UInt32] = (0..<topK).map { $0 == 2 ? last + 5 : UInt32($0 % Self.experts) }
        let clamped = picks.map { min($0, last) }
        func buffer<T>(_ values: [T]) throws -> MTLBuffer {
            try values.withUnsafeBytes { bytes in
                let base = try #require(bytes.baseAddress)
                return try #require(
                    context.device.makeBuffer(
                        bytes: base, length: bytes.count, options: .storageModeShared))
            }
        }
        let sentinel = UInt32.max
        let log = try buffer([UInt32](repeating: sentinel, count: 2 * topK))
        let args = try buffer([UInt64](repeating: 0, count: 16))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try moe.encodeResolveResidentExperts(
            commandBuffer: commandBuffer,
            topKIndices: try buffer(picks), expertAddresses: try buffer(addresses),
            routedArgBuffer: args, routeLog: log,
            routeLogOffset: topK * MemoryLayout<UInt32>.stride,
            topK: UInt32(topK), numExperts: UInt32(Self.experts))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let written = args.contents().bindMemory(to: UInt64.self, capacity: 16)
        let logged = log.contents().bindMemory(to: UInt32.self, capacity: 2 * topK)
        for position in 0..<topK {
            #expect(written[position] == addresses[Int(clamped[position])])
            #expect(logged[topK + position] == clamped[position])
            #expect(logged[position] == sentinel)
        }
    }

    /// The resolve kernel writes raw device addresses where the ordinary path
    /// uses `MTLArgumentEncoder`. The routed kernels read the same struct
    /// either way, so the two encodings have to be the same bytes.
    @Test func argumentEncoderAndRawAddressesAgree() throws {
        let context = try MetalContext()
        let moe = try MoE(context: context, siluActivation: true)
        let topK = moe.maxStreamedExperts
        let blobs = try (0..<topK).map { _ in
            try #require(context.device.makeBuffer(length: 4096, options: .storageModeShared))
        }
        let offsets = (0..<topK).map { $0 * 64 }
        let encoded = try #require(
            moe.makeRoutedArgumentBuffer(
                routedBlobs: blobs, topK: UInt32(topK), routedBufferOffsets: offsets))
        #expect(encoded.length >= topK * MoE.routedArgumentPointerStride)
        let pointers = encoded.contents().bindMemory(to: UInt64.self, capacity: topK)
        for index in 0..<topK {
            #expect(pointers[index] == blobs[index].gpuAddress + UInt64(offsets[index]))
        }
    }
}
