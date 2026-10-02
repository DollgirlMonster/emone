import Foundation
import Synchronization

/// What a live trace view needs to know about the model to size itself.
///
/// Everything here follows the model: the grid is drawn for `numExperts`, the
/// pick list for `topK`, the layer ribbon for `layers`. Nothing is assumed to be
/// 256 experts or top-8.
public struct ExpertTraceShape: Sendable, Equatable {
    public let modelName: String
    /// Width of the routed experts, when the manifest declares one.
    public let routedExpertBits: Int?
    public let layers: Int
    public let numExperts: Int
    public let topK: Int
    /// `ArchConfig.fullAttentionLayerMask`: 0 = sliding window, 1 = full
    /// attention, 2 = linear (GDN).
    public let layerKinds: [UInt8]
    /// Bytes a routed expert read moves, for the SSD rate.
    public let expertBytes: Int

    public init(
        modelName: String, routedExpertBits: Int?, layers: Int, numExperts: Int,
        topK: Int, layerKinds: [UInt8], expertBytes: Int
    ) {
        self.modelName = modelName
        self.routedExpertBits = routedExpertBits
        self.layers = layers
        self.numExperts = numExperts
        self.topK = topK
        self.layerKinds = layerKinds
        self.expertBytes = expertBytes
    }

    /// A dense model has no routed experts: nothing to put in a grid.
    public var isRouted: Bool { numExperts > 0 && topK > 0 }
}

/// One layer's routing decision for one decoded token.
public struct ExpertTraceEvent: Sendable, Equatable {
    /// `mach_absolute_time()` ticks when the layer's picks were recorded; see
    /// `ExpertTraceRing.nanoseconds(fromTicks:)`.
    public var ticks: UInt64
    public var position: Int32
    public var layer: Int32
    public var count: Int32
    /// Where this event's expert ids start in `ExpertTraceBatch.ids`.
    public var idStart: Int32
    /// Bit i set: pick i (in router order) was an SSD read, not a cache hit.
    public var missMask: UInt64

    public init(
        ticks: UInt64, position: Int32, layer: Int32, count: Int32, idStart: Int32,
        missMask: UInt64
    ) {
        self.ticks = ticks
        self.position = position
        self.layer = layer
        self.count = count
        self.idStart = idStart
        self.missMask = missMask
    }
}

/// Events drained from a ring, in order, with their ids in one flat array.
/// Reused across drains so the render thread allocates nothing in steady state.
public struct ExpertTraceBatch: Sendable {
    public var events: [ExpertTraceEvent] = []
    public var ids: [UInt32] = []
    /// Events the decode thread discarded because the ring was full (the
    /// renderer fell behind), cumulative.
    public var dropped: UInt64 = 0

    public init() {}

    public mutating func removeAll() {
        events.removeAll(keepingCapacity: true)
        ids.removeAll(keepingCapacity: true)
    }

    public func ids(of event: ExpertTraceEvent) -> ArraySlice<UInt32> {
        let start = Int(event.idStart)
        return ids[start..<(start + Int(event.count))]
    }
}

/// A fixed-capacity, lock-free, single-producer single-consumer ring of routing
/// events: the decode thread writes, the render thread reads.
///
/// The decode thread calls `record` once per MoE layer per token. It copies at
/// most `topK` ids and a miss mask into preallocated storage, reads the cycle
/// counter, and publishes with one release store. There is no lock, no
/// read-modify-write atomic, no allocation, and it never waits: when the ring
/// is full the *new* event is dropped and counted, so a stalled renderer costs
/// the view some history and costs decode nothing.
///
/// This is deliberately not `TINYTITAN_ROUTE_TRACE` (a file write per layer)
/// and does not touch `routedExpertResidentIDs`.
///
/// unchecked-invariant: exactly one thread calls `record` (it owns `produced`,
/// `droppedLocal` and every slot between the consumer's tail and its own head)
/// and exactly one thread calls `drain` (it owns `consumed`). The slot memory is
/// handed across with a release store of `head` and an acquire load of it on
/// the other side, and back with the same pair on `tail`.
public final class ExpertTraceRing: @unchecked Sendable {
    public static let defaultCapacity = 4096
    /// A miss mask is one `UInt64`; a router with more picks than this is
    /// recorded truncated to its first 64.
    public static let maxRecordedPicks = 64

    public let shape: ExpertTraceShape
    public let capacity: Int
    private let stride: Int

    private let ids: UnsafeMutablePointer<UInt32>
    private let ticks: UnsafeMutablePointer<UInt64>
    private let meta: UnsafeMutablePointer<Int32>  // position, layer, count per slot
    private let masks: UnsafeMutablePointer<UInt64>

    private let head = Atomic<UInt64>(0)
    private let tail = Atomic<UInt64>(0)
    private let dropped = Atomic<UInt64>(0)
    // Producer-private.
    private var produced: UInt64 = 0
    private var droppedLocal: UInt64 = 0
    // Consumer-private.
    private var consumed: UInt64 = 0

    public init(shape: ExpertTraceShape, capacity: Int = ExpertTraceRing.defaultCapacity) {
        let cap = max(1, capacity)
        let stride = max(1, min(shape.topK, Self.maxRecordedPicks))
        self.shape = shape
        self.capacity = cap
        self.stride = stride
        ids = .allocate(capacity: cap * stride)
        ids.initialize(repeating: 0, count: cap * stride)
        ticks = .allocate(capacity: cap)
        ticks.initialize(repeating: 0, count: cap)
        meta = .allocate(capacity: cap * 3)
        meta.initialize(repeating: 0, count: cap * 3)
        masks = .allocate(capacity: cap)
        masks.initialize(repeating: 0, count: cap)
    }

    deinit {
        ids.deallocate()
        ticks.deallocate()
        meta.deallocate()
        masks.deallocate()
    }

    /// Record one layer's picks. Called on the decode thread only.
    ///
    /// - Parameters:
    ///   - experts: the router's top-k expert ids, in router order.
    ///   - missIndices: positions within `experts` that were SSD reads
    ///     (`RoutedExpertFetchPlan.misses`), or nil when no cache plan was made
    ///     and every pick is read from SSD.
    public func record(
        layer: Int, position: Int, experts: [Int], missIndices: [Int]?
    ) {
        let cap = UInt64(capacity)
        let h = produced
        if h &- tail.load(ordering: .acquiring) >= cap {
            droppedLocal &+= 1
            dropped.store(droppedLocal, ordering: .relaxed)
            return
        }
        let n = min(experts.count, stride)
        var mask: UInt64 = 0
        if let missIndices {
            for index in missIndices where index >= 0 && index < n {
                mask |= 1 << UInt64(index)
            }
        } else if n > 0 {
            mask = n >= 64 ? UInt64.max : (1 << UInt64(n)) - 1
        }
        let slot = Int(h % cap)
        let base = ids + slot * stride
        for i in 0..<n { base[i] = UInt32(truncatingIfNeeded: experts[i]) }
        ticks[slot] = mach_absolute_time()
        meta[slot * 3] = Int32(truncatingIfNeeded: position)
        meta[slot * 3 + 1] = Int32(truncatingIfNeeded: layer)
        meta[slot * 3 + 2] = Int32(n)
        masks[slot] = mask
        produced = h &+ 1
        head.store(h &+ 1, ordering: .releasing)
    }

    /// Move everything recorded since the last drain into `batch`. Called on
    /// the render thread only.
    public func drain(into batch: inout ExpertTraceBatch) {
        batch.removeAll()
        let cap = UInt64(capacity)
        let available = head.load(ordering: .acquiring)
        var sequence = consumed
        while sequence < available {
            let slot = Int(sequence % cap)
            let count = Int(meta[slot * 3 + 2])
            let start = batch.ids.count
            let base = ids + slot * stride
            for i in 0..<count { batch.ids.append(base[i]) }
            batch.events.append(
                ExpertTraceEvent(
                    ticks: ticks[slot], position: meta[slot * 3],
                    layer: meta[slot * 3 + 1], count: Int32(count),
                    idStart: Int32(start), missMask: masks[slot]))
            sequence &+= 1
        }
        consumed = available
        tail.store(available, ordering: .releasing)
        batch.dropped = dropped.load(ordering: .relaxed)
    }

    /// Events recorded and not yet drained.
    public var pending: Int {
        Int(head.load(ordering: .acquiring) &- tail.load(ordering: .acquiring))
    }

    private static let timebase: mach_timebase_info_data_t = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return info
    }()

    /// Convert two tick values' difference, or an absolute tick count, to
    /// nanoseconds.
    public static func nanoseconds(fromTicks ticks: UInt64) -> UInt64 {
        let info = timebase
        guard info.denom != 0 else { return ticks }
        return ticks / UInt64(info.denom) * UInt64(info.numer)
            + ticks % UInt64(info.denom) * UInt64(info.numer) / UInt64(info.denom)
    }
}
