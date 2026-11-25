import Testing

@testable import Emone

/// `EMONE_DECODE_CB_LAYERS` (layers per decode command buffer) and the
/// `EMONE_CB_GAP_TRACE` arithmetic. The grouped path itself needs a GPU and a
/// model: it is checked on the M1 Max with the golden baseline, not here.
@Suite struct DecodeCommandBufferGroupingTests {
    @Test func layersPerBufferIsOffUnlessAPositiveCount() {
        #expect(RealForwardRunner.parseDecodeLayersPerCommandBuffer(nil) == nil)
        #expect(RealForwardRunner.parseDecodeLayersPerCommandBuffer("") == nil)
        #expect(RealForwardRunner.parseDecodeLayersPerCommandBuffer("0") == nil)
        #expect(RealForwardRunner.parseDecodeLayersPerCommandBuffer("-4") == nil)
        #expect(RealForwardRunner.parseDecodeLayersPerCommandBuffer("four") == nil)
        #expect(RealForwardRunner.parseDecodeLayersPerCommandBuffer("1") == 1)
        #expect(RealForwardRunner.parseDecodeLayersPerCommandBuffer(" 10 ") == 10)
        #expect(RealForwardRunner.parseDecodeLayersPerCommandBuffer("40") == 40)
    }

    @Test func gapSplitsIntoCPULateAndBoundary() {
        // a: 0-10. b queued at 5 (before a ended), runs 12-20: 2 of boundary.
        // c queued at 25 (5 after b ended), runs 26-30: 6 idle, 5 of it the
        // GPU waiting for the CPU.
        let s = CommandBufferGapTrace.summarize([
            (schedule: 0, start: 0, end: 10),
            (schedule: 5, start: 12, end: 20),
            (schedule: 25, start: 26, end: 30),
        ])
        #expect(s.gap == 8)
        #expect(s.cpuLate == 5)
        #expect(s.busy == 22)
    }

    @Test func overlappingBuffersAndOneBufferHaveNoGap() {
        let one = CommandBufferGapTrace.summarize([(schedule: 0, start: 1, end: 4)])
        #expect(one.gap == 0 && one.cpuLate == 0 && one.busy == 3)
        // A buffer that starts before the previous one ends counts no gap,
        // and a late schedule never counts more than the gap.
        let overlap = CommandBufferGapTrace.summarize([
            (schedule: 0, start: 0, end: 10),
            (schedule: 50, start: 9, end: 12),
            (schedule: 50, start: 13, end: 14),
        ])
        #expect(overlap.gap == 1)
        #expect(overlap.cpuLate == 1)
    }
}
