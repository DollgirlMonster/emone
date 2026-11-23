import Darwin
import Foundation
import Emone

/// `--warm`: decide whether to fill the routed-expert cache before the prompt,
/// do it when the answer is yes, and print one line with the choice and why.
func warmAtLoad(
    args: Args, model: Model, runner: RealForwardRunner, promptIds: [Int32], stderr: FileHandle
) async throws {
    guard args.warm != .off, model.config.numExperts > 0 else { return }
    let layers = model.config.numLayers
    var missing = 0
    for layer in 0..<layers { missing += try model.missingRoutedExpertCount(layer: layer) }
    let slots = model.routedExpertCacheSlotCount() ?? 0
    let inputs = WarmAtLoadInputs(
        cacheHoldsEveryExpert: slots >= model.config.numExperts,
        missingExperts: missing,
        expertBytes: try model.routedExpertByteStride(layer: 0),
        physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
        availableMemoryBytes: availableMemoryBytes(),
        promptTokens: promptIds.count,
        maxNewTokens: args.maxNew)
    let decision = WarmAtLoad.decide(mode: args.warm, inputs: inputs)
    var line = String(
        format: "[warm=%@ mode=%@ reason=%@ predicted_warm_s=%.1f predicted_saved_s=%.1f",
        decision.warm ? "on" : "off", args.warm.rawValue, decision.reason,
        decision.warmSeconds, decision.savedSeconds)
    if decision.warm {
        let started = Date()
        let before = runner.totalResidentWarmedExperts
        await runner.warmResidentExpertsWhileIdle()
        line += String(
            format: " warm_s=%.2f warmed=%llu", Date().timeIntervalSince(started),
            runner.totalResidentWarmedExperts &- before)
    }
    try? stderr.write(contentsOf: Data((line + "]\n").utf8))
}

/// Free, inactive and purgeable pages: memory the warm can take without
/// pushing the machine into compression or swap.
func availableMemoryBytes() -> UInt64 {
    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(
        MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride)
    let result = withUnsafeMutablePointer(to: &stats) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
        }
    }
    guard result == KERN_SUCCESS else { return 0 }
    let pages =
        UInt64(stats.free_count) + UInt64(stats.inactive_count) + UInt64(stats.purgeable_count)
    return pages * UInt64(max(0, sysconf(_SC_PAGESIZE)))
}
