import TinyTitan
import TinyTitanLiveTrace

extension ServerModelSession {
    /// Tell the `--live-trace` view, if there is one, that this model is now
    /// resident and what shape it is. Nothing is computed when there is none.
    nonisolated func reportLiveTraceLoad() {
        guard let live = ServerLiveTrace.current else { return }
        // The grid is fed from the plain decode path, one sequence at a time:
        // batched serving shares the runner between sequences, and MTP's draft
        // and verify passes do not go through the traced router readback. The
        // view says so and shows its compact layout rather than an empty grid.
        let unavailable: String? =
            slots > 1
            ? "batched serving (--max-concurrent-sequences above 1) does not record routing"
            : mtpDecoder != nil ? "MTP decode does not record routing" : nil
        live.modelLoaded(
            token: liveTraceToken,
            info: LiveTraceModelInfo(
                name: defaultModelID, engine: "gpu", shape: runner.expertTraceShape(),
                slotsPerLayer: expertCacheSlots, gridUnavailable: unavailable))
    }
}

extension CPUModelBackend {
    /// The CPU engine has no runner to trace, so the view gets a model with no
    /// shape and shows its compact layout.
    nonisolated func reportLiveTraceLoad(name: String) {
        ServerLiveTrace.current?.modelLoaded(
            token: liveTraceToken,
            info: LiveTraceModelInfo(name: name, engine: "cpu", shape: nil, slotsPerLayer: nil))
    }
}
