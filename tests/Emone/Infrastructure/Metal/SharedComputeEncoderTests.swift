import Foundation
import Metal
import Testing

@testable import Emone

/// `EMONE_SHARED_ENCODER=1` folds every kernel of a command buffer into one
/// serial compute encoder. What could go wrong silently is the bookkeeping: a
/// kernel ending the shared encoder under the next one, two buffers sharing an
/// encoder, or a buffer reaching commit with the encoder still open (Metal
/// asserts on that). The flag itself is read once at launch, so these tests
/// drive the bookkeeping directly with `sharing: true`.
@Suite struct SharedComputeEncoderTests {
    @Test func flagIsOffUnlessSetAndOffUnderTimestamps() {
        #expect(!SharedComputeEncoder.resolve([:]))
        #expect(!SharedComputeEncoder.resolve(["EMONE_SHARED_ENCODER": "0"]))
        #expect(SharedComputeEncoder.resolve(["EMONE_SHARED_ENCODER": "1"]))
        #expect(
            !SharedComputeEncoder.resolve([
                "EMONE_SHARED_ENCODER": "1", "EMONE_ENCODER_TIMESTAMPS": "/tmp/t.csv",
            ]))
    }

    @Test func oneOpenEncoderPerCommandBufferUntilClosed() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
            let queue = device.makeCommandQueue(),
            let a = queue.makeCommandBuffer(),
            let b = queue.makeCommandBuffer()
        else { return }

        #expect(a.openSharedEncoder == nil)
        let first = try #require(a.sharedComputeEncoder())
        #expect(first.isSharedEncoder)
        // A kernel releasing the shared encoder leaves it open for the next.
        first.releaseEncoder(sharing: true)
        let second = try #require(a.sharedComputeEncoder())
        #expect(first === second)
        #expect(a.openSharedEncoder === first)

        // Another command buffer gets its own encoder.
        let other = try #require(b.sharedComputeEncoder())
        #expect(other !== first)

        // Closing really ends it and drops the tag; the next kernel opens a
        // fresh one (Metal may hand back the same recycled object).
        a.endSharedEncoder()
        #expect(a.openSharedEncoder == nil)
        #expect(!first.isSharedEncoder)
        let third = try #require(a.sharedComputeEncoder())
        #expect(third.isSharedEncoder)
        a.endSharedEncoder()
        a.endSharedEncoder()  // idempotent

        // A plain encoder after the close -- possibly the recycled object --
        // must really be ended by release, or commit asserts.
        let plain = try #require(a.makeComputeCommandEncoder(dispatchType: .concurrent))
        #expect(!plain.isSharedEncoder)
        plain.releaseEncoder(sharing: true)

        // A blit after the close is legal, and both buffers commit cleanly.
        let blit = try #require(a.makeBlitCommandEncoder())
        blit.endEncoding()
        b.endSharedEncoder()
        a.commit()
        b.commit()
        a.waitUntilCompleted()
        b.waitUntilCompleted()
        #expect(a.status == .completed)
        #expect(b.status == .completed)
    }

    @Test func aPlainEncoderIsEndedByRelease() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
            let queue = device.makeCommandQueue(),
            let cb = queue.makeCommandBuffer(),
            let plain = cb.makeComputeCommandEncoder(dispatchType: .concurrent)
        else { return }
        #expect(!plain.isSharedEncoder)
        // Not the shared encoder, so release ends it even with sharing on;
        // commit would assert otherwise.
        plain.releaseEncoder(sharing: true)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)
    }
}
