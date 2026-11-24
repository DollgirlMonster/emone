import Foundation
import Metal
import ObjectiveC

/// `EMONE_SHARED_ENCODER=1` (default off, read once): every
/// `timedComputeEncoder()` on a command buffer returns that buffer's one open
/// *serial* compute encoder instead of a new encoder per kernel, so a warm
/// decode token encodes a handful of encoders rather than ~900.
///
/// A serial compute encoder runs its dispatches in order, each seeing the
/// writes of the ones before it, so folding consecutive kernels of one command
/// buffer into one serial encoder changes no values. Every kernel sets its own
/// pipeline, buffers, bytes and threadgroup memory; bindings left over from an
/// earlier kernel are only visible to a kernel that reads an index it never
/// set, which would already be a bug.
///
/// Bookkeeping: the open encoder hangs off its command buffer as an associated
/// object, and the encoder itself is tagged so `releaseEncoder()` (what every
/// call site does instead of `endEncoding()`) can leave it open. The shared
/// encoder is really ended by `closeSharedEncoder()`, which runs before
/// anything else that needs the buffer free of an open encoder: a blit, an
/// event wait, a concurrent compute encoder, and `commitEncoded()`.
///
/// Off, `timedComputeEncoder()` makes a new encoder and `releaseEncoder()` is
/// `endEncoding()`, exactly as before. `EMONE_ENCODER_TIMESTAMPS` needs one
/// encoder per kernel, so it turns sharing off (with a warning).
public enum SharedComputeEncoder {
    public static let enabled: Bool = {
        let on = resolve(ProcessInfo.processInfo.environment)
        if on {
            atexit {
                FileHandle.standardError.write(
                    Data("shared encoder: \(SharedComputeEncoder.opened) encoders opened\n".utf8))
            }
        }
        return on
    }()

    static func resolve(_ env: [String: String]) -> Bool {
        guard env["EMONE_SHARED_ENCODER"] == "1" else { return false }
        if env["EMONE_ENCODER_TIMESTAMPS"] != nil {
            FileHandle.standardError.write(
                Data(
                    "EMONE_SHARED_ENCODER ignored: EMONE_ENCODER_TIMESTAMPS needs one encoder per kernel\n"
                        .utf8))
            return false
        }
        return true
    }

    // Associated-object keys: only their addresses matter.
    nonisolated(unsafe) static var openKey: UInt8 = 0
    nonisolated(unsafe) static var sharedTagKey: UInt8 = 0

    /// Shared encoders opened since launch, printed to stderr at exit when on.
    nonisolated(unsafe) static var opened = 0
}

public extension MTLCommandBuffer {
    /// The open shared encoder, when sharing is on and one is open.
    var openSharedEncoder: MTLComputeCommandEncoder? {
        withUnsafePointer(to: &SharedComputeEncoder.openKey) {
            objc_getAssociatedObject(self, $0) as? MTLComputeCommandEncoder
        }
    }

    /// Shared encoder for this buffer, opening one when none is open.
    func sharedComputeEncoder() -> MTLComputeCommandEncoder? {
        if let open = openSharedEncoder { return open }
        guard let encoder = makeComputeCommandEncoder(dispatchType: .serial) else { return nil }
        withUnsafePointer(to: &SharedComputeEncoder.sharedTagKey) {
            objc_setAssociatedObject(encoder, $0, true, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        }
        withUnsafePointer(to: &SharedComputeEncoder.openKey) {
            objc_setAssociatedObject(self, $0, encoder, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        }
        SharedComputeEncoder.opened &+= 1
        return encoder
    }

    /// Ends the shared encoder if one is open. A no-op when sharing is off.
    func closeSharedEncoder() {
        guard SharedComputeEncoder.enabled else { return }
        endSharedEncoder()
    }

    /// Ends the shared encoder if one is open, whatever the flag says.
    func endSharedEncoder() {
        guard let open = openSharedEncoder else { return }
        // Metal recycles encoder objects: clear the tag, or a later plain
        // encoder that reuses this object would be left open by
        // `releaseEncoder()` (measured: the next encoder here is the same object).
        withUnsafePointer(to: &SharedComputeEncoder.sharedTagKey) {
            objc_setAssociatedObject(open, $0, nil, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        }
        withUnsafePointer(to: &SharedComputeEncoder.openKey) {
            objc_setAssociatedObject(self, $0, nil, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        }
        open.endEncoding()
    }

    /// `commit()`, closing the shared encoder first.
    func commitEncoded() {
        closeSharedEncoder()
        commit()
    }

    /// `makeBlitCommandEncoder()`, closing the shared encoder first.
    func blitEncoder() -> MTLBlitCommandEncoder? {
        closeSharedEncoder()
        return makeBlitCommandEncoder()
    }

    /// `encodeWaitForEvent`, closing the shared encoder first.
    func waitForEvent(_ event: MTLEvent, value: UInt64) {
        closeSharedEncoder()
        encodeWaitForEvent(event, value: value)
    }
}

public extension MTLComputeCommandEncoder {
    /// True for an encoder handed out by `sharedComputeEncoder()`.
    var isSharedEncoder: Bool {
        withUnsafePointer(to: &SharedComputeEncoder.sharedTagKey) {
            objc_getAssociatedObject(self, $0) as? Bool ?? false
        }
    }

    /// What a kernel does when it has finished encoding: `endEncoding()`,
    /// except on the shared encoder, which stays open for the next kernel.
    func releaseEncoder(sharing: Bool = SharedComputeEncoder.enabled) {
        if sharing, isSharedEncoder { return }
        endEncoding()
    }
}
