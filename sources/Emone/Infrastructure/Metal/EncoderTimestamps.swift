import Foundation
import Metal

/// DIAG (local only): `EMONE_ENCODER_TIMESTAMPS=<file>` samples the GPU
/// timestamp at the start and end of every compute encoder (stage-boundary
/// counter sampling, the only kind an M1 Max offers) and, at process exit,
/// writes one `label,start,end` line per encoder in GPU ticks converted to
/// seconds on the CPU clock. Off: `timedComputeEncoder()` is
/// `makeComputeCommandEncoder()`, nothing else changes.
// unchecked-invariant: every field is read and written under `lock`.
final class EncoderTimestamps: @unchecked Sendable {
    static let shared = EncoderTimestamps()
    static let path = ProcessInfo.processInfo.environment["EMONE_ENCODER_TIMESTAMPS"]

    private let lock = NSLock()
    private var device: MTLDevice?
    private var buffers: [MTLCounterSampleBuffer] = []
    private var used = 0
    private var labels: [String] = []
    private var perBuffer =
        Int(
            ProcessInfo.processInfo.environment["EMONE_ENCODER_TIMESTAMP_SAMPLES"] ?? "") ?? 4_096
    private var reported = false
    private var calibration: (cpu0: MTLTimestamp, gpu0: MTLTimestamp)?

    func attachment(for device: MTLDevice, label: String) -> (MTLCounterSampleBuffer, Int)? {
        lock.lock()
        defer { lock.unlock() }
        if self.device == nil {
            self.device = device
            let (cpu, gpu) = device.sampleTimestamps()
            calibration = (cpu, gpu)
            atexit { EncoderTimestamps.shared.write() }
        }
        if buffers.isEmpty || used + 2 > perBuffer {
            guard
                let set = device.counterSets?.first(where: {
                    $0.name == MTLCommonCounterSet.timestamp.rawValue
                })
            else { return nil }
            let d = MTLCounterSampleBufferDescriptor()
            d.counterSet = set
            d.storageMode = .shared
            d.sampleCount = perBuffer
            let b: MTLCounterSampleBuffer
            do { b = try device.makeCounterSampleBuffer(descriptor: d) } catch {
                if !reported {
                    reported = true
                    FileHandle.standardError.write(Data("counter buffer failed: \(error)\n".utf8))
                }
                return nil
            }
            buffers.append(b)
            used = 0
        }
        let index = used
        used += 2
        labels.append(label)
        return (buffers[buffers.count - 1], index)
    }

    func write() {
        lock.lock()
        defer { lock.unlock() }
        guard let path = Self.path, let device, let calibration else { return }
        let (cpu1, gpu1) = device.sampleTimestamps()
        let scale =
            gpu1 > calibration.gpu0
            ? Double(cpu1 - calibration.cpu0) / Double(gpu1 - calibration.gpu0) : 1
        var tb = mach_timebase_info_data_t()
        mach_timebase_info(&tb)
        let secondsPerTick = Double(tb.numer) / Double(tb.denom) / 1e9
        var out = ""
        var label = 0
        for (b, buffer) in buffers.enumerated() {
            let count = b == buffers.count - 1 ? used : perBuffer
            guard count > 0 else { continue }
            let resolved: Data?
            do { resolved = try buffer.resolveCounterRange(0..<count) } catch {
                FileHandle.standardError.write(Data("resolve failed: \(error)\n".utf8)); continue
            }
            guard let data = resolved else {
                FileHandle.standardError.write(
                    Data("resolve returned nil for \(count) samples\n".utf8));
                continue
            }
            data.withUnsafeBytes { raw in
                let ts = raw.bindMemory(to: MTLCounterResultTimestamp.self)
                var i = 0
                while i + 1 < ts.count, label < labels.count {
                    let s = Double(ts[i].timestamp) - Double(calibration.gpu0)
                    let e = Double(ts[i + 1].timestamp) - Double(calibration.gpu0)
                    let cs = (Double(calibration.cpu0) + s * scale) * secondsPerTick
                    let ce = (Double(calibration.cpu0) + e * scale) * secondsPerTick
                    out += "\(labels[label]),\(cs),\(ce)\n"
                    label += 1
                    i += 2
                }
            }
        }
        FileHandle.standardError.write(
            Data(
                "encoder timestamps: \(labels.count) encoders, \(buffers.count) sample buffers\n"
                    .utf8))
        try? out.write(toFile: path, atomically: true, encoding: .utf8)
    }
}

extension MTLCommandBuffer {
    /// `makeComputeCommandEncoder()`, timestamped under EMONE_ENCODER_TIMESTAMPS,
    /// or the buffer's shared serial encoder under EMONE_SHARED_ENCODER=1.
    /// Callers finish with `releaseEncoder()`, never `endEncoding()`.
    func timedComputeEncoder(
        dispatchType: MTLDispatchType = .serial, file: String = #fileID, line: Int = #line
    ) -> MTLComputeCommandEncoder? {
        if SharedComputeEncoder.enabled {
            // A concurrent encoder is its own pass: close the shared one first.
            if dispatchType == .serial { return sharedComputeEncoder() }
            endSharedEncoder()
            return makeComputeCommandEncoder(dispatchType: dispatchType)
        }
        guard EncoderTimestamps.path != nil,
            let (buffer, index) = EncoderTimestamps.shared.attachment(
                for: device, label: "\(file.split(separator: "/").last ?? ""):\(line)")
        else { return makeComputeCommandEncoder(dispatchType: dispatchType) }
        let d = MTLComputePassDescriptor()
        d.dispatchType = dispatchType
        d.sampleBufferAttachments[0].sampleBuffer = buffer
        d.sampleBufferAttachments[0].startOfEncoderSampleIndex = index
        d.sampleBufferAttachments[0].endOfEncoderSampleIndex = index + 1
        return makeComputeCommandEncoder(descriptor: d)
    }
}
