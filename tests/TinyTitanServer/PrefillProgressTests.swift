import Foundation
import Testing

@testable import TinyTitanServerCore

/// `PrefillProgressMonitor` is process-wide, so these run one at a time.
@Suite("Prefill progress monitor", .serialized)
struct PrefillProgressTests {
    @Test func aGenerationWalksPrefillDecodeIdle() {
        let generation = PrefillProgressMonitor.begin(total: 100, cached: 40)
        var reading = PrefillProgressMonitor.snapshot
        #expect(reading.phase == .prefill)
        #expect(reading.done == 40)
        #expect(reading.total == 100)
        #expect(reading.generation == generation)

        PrefillProgressMonitor.prefill(done: 72, total: 100, generation: generation)
        #expect(PrefillProgressMonitor.snapshot.done == 72)

        PrefillProgressMonitor.decoding(generation: generation)
        reading = PrefillProgressMonitor.snapshot
        #expect(reading.phase == .decode)
        #expect(reading.done == 100)

        PrefillProgressMonitor.end(generation: generation)
        reading = PrefillProgressMonitor.snapshot
        #expect(reading.phase == .idle)
        #expect(reading.generation == generation)
    }

    @Test func anOlderGenerationCannotClobberANewerOne() {
        let older = PrefillProgressMonitor.begin(total: 10, cached: 0)
        let newer = PrefillProgressMonitor.begin(total: 50, cached: 5)
        #expect(newer == older &+ 1)
        PrefillProgressMonitor.prefill(done: 9, total: 10, generation: older)
        PrefillProgressMonitor.end(generation: older)
        let reading = PrefillProgressMonitor.snapshot
        #expect(reading.phase == .prefill)
        #expect(reading.done == 5)
        #expect(reading.total == 50)
        PrefillProgressMonitor.end(generation: newer)
    }

    @Test func theCachedPrefixNeverExceedsTheTotal() {
        let generation = PrefillProgressMonitor.begin(total: 8, cached: 20)
        #expect(PrefillProgressMonitor.snapshot.done == 8)
        PrefillProgressMonitor.end(generation: generation)
    }
}
