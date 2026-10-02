import Foundation

extension RealForwardRunner {
    /// The shape a live trace view should be drawn for, from this model alone.
    public func expertTraceShape() -> ExpertTraceShape {
        ExpertTraceShape(
            modelName: model.modelID,
            routedExpertBits: model.manifest.quant?.routedExpert.weightBits,
            layers: cfg.numLayers,
            numExperts: cfg.numExperts,
            topK: cfg.topKExperts,
            layerKinds: cfg.fullAttentionLayerMask,
            expertBytes: Int(model.packedExpertsLayout.expertStride))
    }

    /// Start recording per-layer routing decisions during decode, for a live
    /// trace view. Off until this is called. Returns nil for a dense model,
    /// which has no routed experts to record, and leaves the hook disabled.
    ///
    /// Call before generation starts and from the task that drives the runner;
    /// the runner is single-flight and `expertTrace` is read on every layer.
    /// This changes nothing the model computes: the hook only reads values
    /// decode has already read back.
    @discardableResult
    public func enableExpertTrace(
        capacity: Int = ExpertTraceRing.defaultCapacity
    ) -> ExpertTraceRing? {
        let shape = expertTraceShape()
        guard shape.isRouted else {
            expertTrace = nil
            return nil
        }
        let ring = ExpertTraceRing(shape: shape, capacity: capacity)
        expertTrace = ring
        return ring
    }

    public func disableExpertTrace() {
        expertTrace = nil
    }
}
