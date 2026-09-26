import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

private let mppAvailableForRoutedGEMM: Bool = {
    guard let context = try? MetalContext() else { return false }
    return MPPPrefillInt4QMM(context: context).isAvailable
}()

/// `PrefillRoutedExpertGEMM` runs a streamed tile as grouped MPP GEMMs. It has to
/// land every (token, rank) row where the tile kernels do and hold the CPU
/// reference's values to MPP rounding (weights rounded to half): experts with
/// several pairs each so the GEMMs have real rows, a partial MPP row tile, and
/// expert blobs at non-zero offsets inside their buffers.
extension PrefillGroupedRoutedMoETests {
    @Test(
        .enabled(if: mppAvailableForRoutedGEMM, "Requires runtime MPP TensorOps support"),
        arguments: [4, 8])
    func groupedGEMMTileMatchesTheReference(weightBits: Int) throws {
        let (d, f, rows, topK) = (64, 64, 40, 2)
        var pairs: [PrefillTokenExpertPair] = []
        for token in 0..<rows {
            pairs.append(Self.pair(token: UInt32(token), expert: UInt32(token % 5), rank: 0))
            pairs.append(Self.pair(token: UInt32(token), expert: UInt32(5 + token % 3), rank: 1))
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: rows, topK: topK, numExperts: 16, tileExpertCount: 16)
        let pool = Self.makeSyntheticExpertPool(numExperts: 16, d: d, f: f, weightBits: weightBits)
        let hidden = (0..<(rows * d)).map { i in Float16(Float((i % 17) - 8) * 0.25) }
        let expected = Self.cpuSyntheticRoutePartials(
            routes: routes, hidden: hidden, hiddenStride: d, pool: pool,
            topK: topK, d: d, f: f)

        let ctx = try MetalContext()
        let gemm = try #require(
            try PrefillRoutedExpertGEMM(context: ctx, weightBits: weightBits, siluActivation: false))
        guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
            let pairBuffer = ctx.device.makeBuffer(
                bytes: routes.sortedPairs,
                length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
                options: .storageModeShared),
            let outputBuffer = Fp16Buffer.make(
                ctx.device, halves: [Float16](repeating: -77, count: rows * topK * d)),
            let commandBuffer = ctx.queue.makeCommandBuffer()
        else {
            Issue.record("allocation failed")
            return
        }
        for (index, tile) in routes.tiles.enumerated() {
            let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: index, routes: routes)
            let binding = try PrefillStreamedTileBinding(
                expertIDs: expertIDs,
                views: Self.streamedViewsWithNonzeroOffsets(
                    device: ctx.device, pool: pool, expertIDs: expertIDs))
            let start = Int(tile.groupStart)
            let took = try gemm.encodeTile(
                commandBuffer: commandBuffer,
                hidden: hiddenBuffer, hiddenStrideElements: d,
                sortedPairs: pairBuffer, routePartials: outputBuffer,
                tile: tile, groups: routes.groups[start..<(start + Int(tile.groupCount))],
                binding: binding, offsets: pool.offsets, d: d, f: f, topK: topK)
            #expect(took)
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let actual = Fp16Buffer.readHalf(outputBuffer, count: rows * topK * d)
        #expect(!actual.contains(-77), "a (token, rank) row was never written")
        let maxRef = expected.reduce(Float(0)) { max($0, abs(Float($1))) }
        let maxDiff = zip(actual, expected).reduce(Float(0)) {
            max($0, abs(Float($1.0) - Float($1.1)))
        }
        // The synthetic pool's weights are 0.001-0.03, so real outputs are
        // ~0.01; this only guards against an all-zero reference.
        #expect(maxRef > 1e-3, "the reference is too small to prove anything (max \(maxRef))")
        #expect(
            maxDiff <= 0.02 * maxRef,
            "weightBits=\(weightBits): max |diff| \(maxDiff) against max |ref| \(maxRef)")
    }
}
