import Foundation
import Testing

@testable import TinyTitan

@Suite struct ExpertTraceRingTests {
    static func shape(experts: Int = 512, topK: Int = 10, layers: Int = 48) -> ExpertTraceShape {
        ExpertTraceShape(
            modelName: "test", routedExpertBits: 4, layers: layers, numExperts: experts,
            topK: topK, layerKinds: (0..<layers).map { $0 % 4 == 3 ? 1 : 2 },
            expertBytes: 1_000_000)
    }

    @Test func recordsAndDrainsInOrderWithMissMask() {
        let ring = ExpertTraceRing(shape: Self.shape(), capacity: 16)
        ring.record(layer: 3, position: 7, experts: [5, 9, 300, 511], missIndices: [1, 3])
        ring.record(layer: 4, position: 7, experts: [1, 2], missIndices: [])
        var batch = ExpertTraceBatch()
        ring.drain(into: &batch)
        #expect(batch.events.count == 2)
        let first = batch.events[0]
        #expect(first.layer == 3 && first.position == 7 && first.count == 4)
        #expect(Array(batch.ids(of: first)) == [5, 9, 300, 511])
        #expect(first.missMask == 0b1010)
        #expect(batch.events[1].missMask == 0)
        #expect(Array(batch.ids(of: batch.events[1])) == [1, 2])
        #expect(batch.events[1].ticks >= first.ticks)
        ring.drain(into: &batch)
        #expect(batch.events.isEmpty)
    }

    @Test func noCachePlanMeansEveryPickWasRead() {
        let ring = ExpertTraceRing(shape: Self.shape(), capacity: 4)
        ring.record(layer: 0, position: 0, experts: [1, 2, 3], missIndices: nil)
        var batch = ExpertTraceBatch()
        ring.drain(into: &batch)
        #expect(batch.events[0].missMask == 0b111)
    }

    @Test func missIndicesOutsideThePicksAreIgnored() {
        let ring = ExpertTraceRing(shape: Self.shape(topK: 4), capacity: 4)
        ring.record(layer: 0, position: 0, experts: [1, 2], missIndices: [0, 5, -1])
        var batch = ExpertTraceBatch()
        ring.drain(into: &batch)
        #expect(batch.events[0].missMask == 0b1)
    }

    @Test func picksBeyondTheShapeAreTruncated() {
        let ring = ExpertTraceRing(shape: Self.shape(topK: 3), capacity: 4)
        ring.record(layer: 0, position: 0, experts: [1, 2, 3, 4, 5], missIndices: [4])
        var batch = ExpertTraceBatch()
        ring.drain(into: &batch)
        #expect(Array(batch.ids(of: batch.events[0])) == [1, 2, 3])
        #expect(batch.events[0].missMask == 0)
    }

    @Test func wrapsAroundAcrossManyDrains() {
        let ring = ExpertTraceRing(shape: Self.shape(topK: 2), capacity: 5)
        var batch = ExpertTraceBatch()
        var next = 0
        for round in 0..<20 {
            let count = round % 4 + 1
            for _ in 0..<count {
                ring.record(
                    layer: next % 48, position: next, experts: [next, next + 1], missIndices: [0])
                next += 1
            }
            ring.drain(into: &batch)
            #expect(batch.events.count == count)
            for event in batch.events {
                #expect(
                    Array(batch.ids(of: event)) == [
                        UInt32(event.position), UInt32(event.position) + 1,
                    ])
                #expect(event.missMask == 1)
            }
            #expect(batch.dropped == 0)
        }
    }

    @Test func aFullRingDropsNewEventsAndCountsThem() {
        let ring = ExpertTraceRing(shape: Self.shape(topK: 1), capacity: 3)
        for index in 0..<5 {
            ring.record(layer: 0, position: index, experts: [index], missIndices: [])
        }
        #expect(ring.pending == 3)
        var batch = ExpertTraceBatch()
        ring.drain(into: &batch)
        // The oldest are kept: decode never overwrites what the reader has not read.
        #expect(batch.events.map(\.position) == [0, 1, 2])
        #expect(batch.dropped == 2)
        ring.record(layer: 0, position: 9, experts: [9], missIndices: [])
        ring.drain(into: &batch)
        #expect(batch.events.map(\.position) == [9])
    }

    @Test func producerAndConsumerOnTwoThreadsLoseNothingUnaccounted() {
        let total = 200_000
        let ring = ExpertTraceRing(shape: Self.shape(topK: 4), capacity: 64)
        let producer = Thread {
            for index in 0..<total {
                let value = index % 1000
                ring.record(
                    layer: value % 48, position: index,
                    experts: [value, value + 1, value + 2, value + 3],
                    missIndices: value % 2 == 0 ? [1] : [])
            }
        }
        producer.start()
        var batch = ExpertTraceBatch()
        var seen = 0
        var last = -1
        var corrupt = 0
        let deadline = Date().addingTimeInterval(60)
        while Date() < deadline {
            ring.drain(into: &batch)
            for event in batch.events {
                seen += 1
                if Int(event.position) <= last { corrupt += 1 }
                last = Int(event.position)
                let value = Int(event.position) % 1000
                let expected = [value, value + 1, value + 2, value + 3].map { UInt32($0) }
                if Array(batch.ids(of: event)) != expected { corrupt += 1 }
                if event.layer != Int32(value % 48) { corrupt += 1 }
                if event.missMask != (value % 2 == 0 ? 0b10 : 0) { corrupt += 1 }
            }
            // Every event is either read or counted as dropped.
            if seen + Int(batch.dropped) >= total { break }
            Thread.sleep(forTimeInterval: 0.0005)
        }
        #expect(corrupt == 0)
        #expect(seen + Int(batch.dropped) == total)
        #expect(seen > 0)
    }

    @Test func denseShapeIsNotRouted() {
        #expect(!Self.shape(experts: 0, topK: 0).isRouted)
        #expect(Self.shape().isRouted)
    }

    @Test func tickConversionIsMonotonicAndNonZero() {
        let one = ExpertTraceRing.nanoseconds(fromTicks: 1_000_000)
        let two = ExpertTraceRing.nanoseconds(fromTicks: 2_000_000)
        #expect(one > 0)
        #expect(two >= one * 2 - 2 && two <= one * 2 + 2)
    }
}
