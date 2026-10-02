import Testing

@testable import TinyTitan

/// The hidden-state readout's pure parts: the plan, the per-chunk capture
/// schedule and the row reductions. None of it needs a GPU or a model.
@Suite struct HiddenReadoutPlanTests {
    @Test func layersAreSortedAndDeduplicated() throws {
        let plan = try HiddenReadoutPlan(layers: [36, 12, 47, 12, 24, 36])
        #expect(plan.layers == [12, 24, 36, 47])
        #expect(plan.deepestLayer == 47)
        #expect(plan.streamMode == .residual)
    }

    @Test func refusesEmptyNegativeAndTooManyLayers() {
        #expect(throws: HiddenReadoutError.self) { try HiddenReadoutPlan(layers: []) }
        #expect(throws: HiddenReadoutError.self) { try HiddenReadoutPlan(layers: [3, -1]) }
        #expect(throws: HiddenReadoutError.self) {
            try HiddenReadoutPlan(layers: Array(0..<(HiddenReadoutPlan.maximumLayers + 1)))
        }
    }

    @Test func capCountsDistinctLayersNotRequestedOnes() throws {
        // Seventeen entries, sixteen distinct: within the cap once folded.
        let layers = Array(0..<HiddenReadoutPlan.maximumLayers) + [0]
        #expect(try HiddenReadoutPlan(layers: layers).layers.count == 16)
    }

    @Test func validateChecksTheDeepestLayerAgainstTheModel() throws {
        let plan = try HiddenReadoutPlan(layers: [0, 47])
        try plan.validate(numLayers: 48)
        #expect(throws: HiddenReadoutError.self) { try plan.validate(numLayers: 47) }
    }

    @Test func everyChunkStopsAfterTheDeepestLayerButOnlyTheLastCaptures() throws {
        let plan = try HiddenReadoutPlan(layers: [12, 24])
        let early = plan.chunkPlan(isLastChunk: false, numLayers: 48)
        #expect(early.layerLimit == 25)
        #expect(early.captureLayers.isEmpty)
        let last = plan.chunkPlan(isLastChunk: true, numLayers: 48)
        #expect(last.layerLimit == 25)
        #expect(last.captureLayers == [12, 24])
    }

    @Test func aCaptureDuringAGenerationNeverStopsEarly() throws {
        let plan = try HiddenReadoutPlan(layers: [12, 24], prefillOnly: false)
        let early = plan.chunkPlan(isLastChunk: false, numLayers: 48)
        #expect(early == .init(layerLimit: 48, captureLayers: [], earlyStop: false))
        let last = plan.chunkPlan(isLastChunk: true, numLayers: 48)
        #expect(last == .init(layerLimit: 48, captureLayers: [12, 24], earlyStop: false))
        // A prefill-only plan of the same layers is a different plan.
        #expect(try HiddenReadoutPlan(layers: [12, 24]) != plan)
        #expect(
            try HiddenReadoutPlan(layers: [12, 24]).chunkPlan(isLastChunk: true, numLayers: 48)
                .earlyStop)
    }

    @Test func theDeepestLayerOfTheModelRunsTheWholeStack() throws {
        let plan = try HiddenReadoutPlan(layers: [47])
        #expect(plan.chunkPlan(isLastChunk: true, numLayers: 48).layerLimit == 48)
        // A plan deeper than the stack (not validated) never runs past it.
        let deep = try HiddenReadoutPlan(layers: [99])
        #expect(deep.chunkPlan(isLastChunk: true, numLayers: 48).layerLimit == 48)
    }

    @Test func planAppliedToRealChunkSpansCapturesOnceInTheLastChunk() throws {
        let plan = try HiddenReadoutPlan(layers: [5, 9])
        // 10 tokens in chunks of 4: spans of 4, 4 and 2 tokens.
        let spans = PrefillChunkPlanner.spans(
            tokenCount: 10, startPosition: 0, chunkTokens: 4)
        let perChunk = spans.indices.map {
            plan.chunkPlan(isLastChunk: $0 == spans.count - 1, numLayers: 48)
        }
        #expect(perChunk.map(\.layerLimit) == [10, 10, 10])
        #expect(perChunk.map(\.captureLayers) == [[], [], [5, 9]])
        // The last prompt token is the last row of the last chunk.
        let last = try #require(spans.last)
        #expect(last.startPosition + last.tokenCount - 1 == 9)
    }

    @Test func readbackRowsFollowTheSortedLayers() throws {
        let plan = try HiddenReadoutPlan(layers: [30, 2, 17])
        #expect(plan.readbackIndex(forLayer: 2) == 0)
        #expect(plan.readbackIndex(forLayer: 17) == 1)
        #expect(plan.readbackIndex(forLayer: 30) == 2)
        #expect(plan.readbackIndex(forLayer: 3) == nil)
    }

    @Test func meanOfStreamsIsStreamMajor() {
        // Four streams of three values: stream s holds s*10 + d.
        let row: [Float] = (0..<4).flatMap { s in (0..<3).map { Float(s * 10 + $0) } }
        let mean = HiddenReadoutMath.meanOfStreams(row, streams: 4)
        #expect(mean == [15, 16, 17])
    }

    @Test func meanOfOneStreamIsTheRow() {
        let row: [Float] = [1.5, -2, 3]
        #expect(HiddenReadoutMath.meanOfStreams(row, streams: 1) == row)
    }

    @Test func reduceWidensFP16AndAppliesTheStreamMode() {
        let values: [Float16] = [1, 2, 3, 4, 5, 6, 7, 8]  // 2 streams of 4
        values.withUnsafeBufferPointer { buffer in
            let residual = HiddenReadoutMath.reduce(buffer, streams: 2, mode: .residual)
            #expect(residual == [1, 2, 3, 4, 5, 6, 7, 8])
            let mean = HiddenReadoutMath.reduce(buffer, streams: 2, mode: .meanStreams)
            #expect(mean == [3, 4, 5, 6])
        }
    }

    @Test func fp16RoundingSurvivesTheWideningExactly() {
        // 0.1 is not representable in fp16; the float32 returned is the fp16
        // value, not the nearest float32 to 0.1.
        let values: [Float16] = [0.1]
        values.withUnsafeBufferPointer { buffer in
            let widened = HiddenReadoutMath.reduce(buffer, streams: 1, mode: .residual)
            #expect(widened == [Float(Float16(0.1))])
            #expect(widened[0] != Float(0.1))
        }
    }
}
