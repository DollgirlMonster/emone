import Metal
import Testing

@testable import TinyTitan

@Suite struct PrefillChunkScratchTests {
    @Test func qwen36T32LayoutMatchesScratchContract() {
        let layout = PrefillChunkScratchLayout(config: .qwen36_35B_A3B, chunkTokens: 32)

        #expect(layout.chunkTokens == 32)
        #expect(layout.hiddenElements == 32 * 2048)
        #expect(layout.normedElements == 32 * 2048)
        // q_proj emits per-head [query ; gate] halves: 2 * 16 heads * 256.
        #expect(layout.qElements == 32 * 8192)
        #expect(layout.kStageElements == 32 * 512)
        #expect(layout.vStageElements == 32 * 512)
        #expect(layout.attentionOutputElements == 32 * 4096)
        #expect(layout.denseXElements == 32 * 2048)
        #expect(layout.routedXElements == 32 * 2048)
        #expect(layout.routerXElements == 32 * 2048)
        #expect(layout.h1Elements == 32 * 2048)
        #expect(layout.h2Elements == 32 * 2048)
        #expect(layout.routePartialElements == 32 * 8 * 2048)
        #expect(layout.routeIDElements == 32 * 8)
        #expect(layout.routeWeightElements == 32 * 8)
        #expect(layout.sharedExpertScratchElements == 512)
        #expect(layout.sharedExpertBatchElements == 32 * 512)
        #expect(layout.routedPairMicrobatchRows == 32)
        #expect(layout.routedGateUpActElements == 3 * 32 * 512)
        #expect(layout.routedDownOutputElements == 32 * 2048)
        // Qwen 3.6 extensions: split q/gate halves, gated-DeltaNet bundle,
        // and the shared-expert scalar gate.
        #expect(layout.attnGateElementsPerToken == 4096)
        #expect(layout.gdnQKVDim == 8192)
        #expect(layout.gdnValueDim == 4096)
        #expect(layout.gdnVHeads == 32)
        #expect(layout.sharedScalarGateElements == 1)
        #expect(layout.gdnConvOutElements == 32 * 8192)
        #expect(layout.gdnZElements == 32 * 4096)
        #expect(layout.gdnAElements == 32 * 32)
        #expect(layout.sharedScalarGateBufferElements == 32)

        let worksheetT32UpperBound = Int(4.5 * 1_048_576.0)
        #expect(layout.totalPersistentBytes <= worksheetT32UpperBound)
    }

    @Test func layoutClampsChunkSizeToRuntimeBounds() {
        #expect(PrefillChunkScratchLayout(config: .qwen36_35B_A3B, chunkTokens: 0).chunkTokens == 1)
        #expect(
            PrefillChunkScratchLayout(config: .qwen36_35B_A3B, chunkTokens: 8_192).chunkTokens
                == 8_192)
        #expect(
            PrefillChunkScratchLayout(config: .qwen36_35B_A3B, chunkTokens: 32_768).chunkTokens
                == 16_384)
    }

    /// Scratch is linear in the chunk: on Qwen3.8 about 0.81 GiB at 4K and
    /// 3.22 GiB at 16K (the ten-way route partials are the largest part). That
    /// is the memory price of the 16K chunk, and why it is an option rather
    /// than a default on smaller machines.
    @Test func qwen38ScratchAt16KIsLinearAndBounded() {
        let at4K = PrefillChunkScratchLayout(config: .qwen38FlashNext, chunkTokens: 4_096)
        let at16K = PrefillChunkScratchLayout(config: .qwen38FlashNext, chunkTokens: 16_384)
        #expect(at16K.chunkTokens == 16_384)
        // Only the routed microbatch and the one-row shared scratch are fixed.
        #expect(at16K.devicePrivateBytes < 4 * at4K.devicePrivateBytes)
        #expect(at4K.totalPersistentBytes < 850 * 1_048_576)
        #expect(at16K.totalPersistentBytes < 3_350 * 1_048_576)
    }

    @Test func qwenLongChunkScratchRemainsBounded() {
        let layout = PrefillChunkScratchLayout(
            config: .qwen36_35B_A3B,
            chunkTokens: 4_096)
        #expect(layout.chunkTokens == 4_096)
        #expect(layout.totalPersistentBytes < 1_024 * 1_048_576)
    }

    @Test func allocationUsesPrivateScratchAndSharedRouteMetadata() throws {
        let ctx = try MetalContext()
        let toy = ArchConfig.qwenToy()
        let layout = PrefillChunkScratchLayout(config: toy, chunkTokens: 4)

        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)

        #expect(scratch.layout == layout)
        #expect(scratch.hidden.length == layout.hiddenElements * MemoryLayout<Float16>.stride)
        #expect(scratch.denseX.length == layout.denseXElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routedX.length == layout.routedXElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routerX.length == layout.routerXElements * MemoryLayout<Float16>.stride)
        #expect(
            scratch.routePartials.length == layout.routePartialElements
                * MemoryLayout<Float16>.stride)
        #expect(scratch.routeIDs.length == layout.routeIDElements * MemoryLayout<UInt32>.stride)
        #expect(
            scratch.routeWeights.length == layout.routeWeightElements * MemoryLayout<Float16>.stride
        )
        #expect(
            scratch.routedGateUpActScratch.length == layout.routedGateUpActElements
                * MemoryLayout<Float16>.stride)
        #expect(
            scratch.routedDownScratch.length == layout.routedDownOutputElements
                * MemoryLayout<Float16>.stride)
        #expect(scratch.hidden.storageMode == MTLStorageMode.private)
        #expect(scratch.denseX.storageMode == MTLStorageMode.private)
        #expect(scratch.routedX.storageMode == MTLStorageMode.private)
        #expect(scratch.routerX.storageMode == MTLStorageMode.private)
        #expect(scratch.routedGateUpActScratch.storageMode == MTLStorageMode.private)
        #expect(scratch.routedDownScratch.storageMode == MTLStorageMode.private)
        #expect(scratch.routeIDs.storageMode == MTLStorageMode.shared)
        #expect(scratch.routeWeights.storageMode == MTLStorageMode.shared)
    }
}
