import Foundation
import Metal

/// Opt-in decode capture of the dense MLP for weight-space work
/// (TinyTitan-a37): with `TINYTITAN_MLP_CAPTURE=<file>` and
/// `TINYTITAN_MLP_CAPTURE_LAYERS=15,19`, every decode position appends, for
/// each listed layer, one record of little-endian int32 position, int32
/// layer, int32 input token, then the MLP output (`hidden` fp16) and the
/// down projection's input (`intermediate` fp16). Off unless both are set; a
/// captured layer waits for its MLP, so it is a diagnostic, never a default.
/// Alongside it, `<file>.resid` gets the residual stream after the layer
/// (position, layer, token, then `hidden` fp16) -- the layer's output. On a
/// routed (MoE) layer, `<file>.moe` gets position, layer, the top-k expert
/// ids (int32 each), then the experts' input (`hidden` fp16).
///
/// unchecked-invariant: the handle and layer set are fixed at first use and
/// every write holds `lock`.
final class MLPCapture: @unchecked Sendable {
    static let shared = MLPCapture()

    let layers: Set<Int>
    private let handle: FileHandle?
    private let residHandle: FileHandle?
    private let moeHandle: FileHandle?
    private let lock = NSLock()

    private init() {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["TINYTITAN_MLP_CAPTURE"],
            let list = env["TINYTITAN_MLP_CAPTURE_LAYERS"]
        else {
            layers = []
            handle = nil
            residHandle = nil
            moeHandle = nil
            return
        }
        layers = Set(list.split(separator: ",").compactMap { Int($0) })
        FileManager.default.createFile(atPath: path, contents: nil)
        handle = FileHandle(forWritingAtPath: path)
        FileManager.default.createFile(atPath: path + ".resid", contents: nil)
        residHandle = FileHandle(forWritingAtPath: path + ".resid")
        FileManager.default.createFile(atPath: path + ".moe", contents: nil)
        moeHandle = FileHandle(forWritingAtPath: path + ".moe")
    }

    var active: Bool { handle != nil && !layers.isEmpty }

    func record(
        position: Int, layer: Int, token: Int32,
        output: MTLBuffer, hidden: Int, act: MTLBuffer, intermediate: Int
    ) {
        guard let handle else { return }
        var head = [Int32(position), Int32(layer), token]
        var data = Data(bytes: &head, count: 12)
        data.append(Data(bytes: output.contents(), count: hidden * 2))
        data.append(Data(bytes: act.contents(), count: intermediate * 2))
        lock.lock()
        defer { lock.unlock() }
        handle.write(data)
    }

    func recordRouted(position: Int, layer: Int, experts: [Int], input: MTLBuffer, width: Int) {
        guard let moeHandle else { return }
        var head = [Int32(position), Int32(layer)] + experts.map { Int32($0) }
        var data = Data(bytes: &head, count: head.count * 4)
        data.append(Data(bytes: input.contents(), count: width * 2))
        lock.lock()
        defer { lock.unlock() }
        moeHandle.write(data)
    }

    func recordResidual(position: Int, layer: Int, token: Int32, hidden: MTLBuffer, width: Int) {
        guard let residHandle else { return }
        var head = [Int32(position), Int32(layer), token]
        var data = Data(bytes: &head, count: 12)
        data.append(Data(bytes: hidden.contents(), count: width * 2))
        lock.lock()
        defer { lock.unlock() }
        residHandle.write(data)
    }
}

extension RealForwardRunner {
    /// Waits for layer `L`'s dense MLP and records it, when the capture
    /// names this layer. `h1Buf` and `denseScratchAct` are shared buffers.
    func captureDenseMLP(
        sharedCB: MTLCommandBuffer, layer L: Int, position: Int, token: Int32
    ) throws {
        let capture = MLPCapture.shared
        guard capture.active, capture.layers.contains(L) else { return }
        try waitForCompletion(sharedCB)
        capture.record(
            position: position, layer: L, token: token,
            output: h1Buf, hidden: cfg.hiddenSize,
            act: denseScratchAct, intermediate: cfg.intermediateSize)
    }

    /// Records a routed layer's expert input and its top-k choice. Called
    /// after the router readback, when `routedX` is settled (shared buffer).
    func captureRoutedInput(layer L: Int, position: Int, experts: [Int]) {
        let capture = MLPCapture.shared
        guard capture.active, capture.layers.contains(L) else { return }
        capture.recordRouted(
            position: position, layer: L, experts: experts, input: routedX, width: cfg.hiddenSize)
    }

    /// Waits for layer `L`'s residual add and records the residual stream
    /// after the layer, when the capture names this layer.
    func captureResidual(exitCB: MTLCommandBuffer, layer L: Int, position: Int, token: Int32) throws
    {
        let capture = MLPCapture.shared
        guard capture.active, capture.layers.contains(L) else { return }
        try waitForCompletion(exitCB)
        capture.recordResidual(
            position: position, layer: L, token: token, hidden: hidden, width: cfg.hiddenSize)
    }
}
