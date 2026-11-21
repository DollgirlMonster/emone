import Foundation
import Metal

/// The runtime-LoRA delta (TinyTitan-lmw.11): `y += scale * (x A^T) B^T` laid
/// over a base projection's fp16 output, so a decision adapter runs on the
/// resident weights instead of a merged copy. Two dispatches in one encoder
/// (serial, so the second reads what the first wrote): `lora_down_fp16` into
/// a fp32 `[tokens x rank]` scratch, then `lora_up_fp16` into y.
final class LoRADelta {
    private let downPSO: MTLComputePipelineState
    private let upPSO: MTLComputePipelineState
    private let downGroupPSO: MTLComputePipelineState
    private let upGroupPSO: MTLComputePipelineState

    /// The rank the down kernel can take: one simdgroup per rank row, at most
    /// 32 to a threadgroup of 1,024.
    static let maximumRank = 32
    /// Tokens per `lora_up_fp16` thread (`LORA_UP_TILE` in the shader).
    static let upTile = 8
    /// Tokens per `lora_down_fp16` threadgroup (`LORA_DOWN_TILE` in the shader).
    static let downTile = 2

    init(context: MetalContext) throws {
        downPSO = try context.pipeline("lora_down_fp16")
        upPSO = try context.pipeline("lora_up_fp16")
        downGroupPSO = try context.pipeline("lora_down_group_fp16")
        upGroupPSO = try context.pipeline("lora_up_group_fp16")
    }

    /// `x` is `[tokens x columns]` fp16 at `xStrideElements`, `y` is
    /// `[tokens x rows]` fp16 at `yStrideElements` and is accumulated into;
    /// `a` is `[rank x columns]` and `b` B transposed, `[rank x rows]`, both fp32
    /// row-major;
    /// `scratch` holds at least `tokens * rank` floats.
    func encode(
        commandBuffer: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int = 0, xStrideElements: Int,
        y: MTLBuffer, yOffset: Int = 0, yStrideElements: Int,
        a: MTLBuffer, aOffset: Int, b: MTLBuffer, bOffset: Int,
        scratch: MTLBuffer,
        rank: Int, rows: Int, columns: Int, tokens: Int, scale: Float
    ) throws {
        guard tokens > 0 else { return }
        guard rank > 0, rank <= Self.maximumRank,
            scratch.length >= tokens * rank * MemoryLayout<Float>.stride
        else {
            throw MetalError.invalidState(
                "LoRA delta: rank \(rank) or scratch \(scratch.length) B "
                    + "does not fit \(tokens) tokens")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        try encode(
            encoder: encoder, x: x, xOffset: xOffset, xStrideElements: xStrideElements,
            y: y, yOffset: yOffset, yStrideElements: yStrideElements,
            a: a, aOffset: aOffset, b: b, bOffset: bOffset, scratch: scratch,
            rank: rank, rows: rows, columns: columns, tokens: tokens, scale: scale)
        encoder.endEncoding()
    }

    /// The same delta into an open encoder, so the deltas of projections that
    /// share an input run in one encoder. The encoder is serial: each delta's
    /// up reads the scratch its own down just wrote, and the next delta's down
    /// overwrites it only after.
    func encode(
        encoder: MTLComputeCommandEncoder,
        x: MTLBuffer, xOffset: Int = 0, xStrideElements: Int,
        y: MTLBuffer, yOffset: Int = 0, yStrideElements: Int,
        a: MTLBuffer, aOffset: Int, b: MTLBuffer, bOffset: Int,
        scratch: MTLBuffer,
        rank: Int, rows: Int, columns: Int, tokens: Int, scale: Float
    ) throws {
        guard tokens > 0 else { return }
        guard rank > 0, rank <= Self.maximumRank,
            scratch.length >= tokens * rank * MemoryLayout<Float>.stride
        else {
            throw MetalError.invalidState(
                "LoRA delta: rank \(rank) or scratch \(scratch.length) B "
                    + "does not fit \(tokens) tokens")
        }
        var k = UInt32(columns)
        var r = UInt32(rank)
        var xs = UInt32(xStrideElements)
        encoder.setComputePipelineState(downPSO)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(a, offset: aOffset, index: 1)
        encoder.setBuffer(scratch, offset: 0, index: 2)
        encoder.setBytes(&k, length: 4, index: 3)
        encoder.setBytes(&r, length: 4, index: 4)
        encoder.setBytes(&xs, length: 4, index: 5)
        var tokenCount = UInt32(tokens)
        encoder.setBytes(&tokenCount, length: 4, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (tokens + Self.downTile - 1) / Self.downTile, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * rank, height: 1, depth: 1))

        var n = UInt32(rows)
        var ys = UInt32(yStrideElements)
        var s = scale
        var t = UInt32(tokens)
        encoder.setComputePipelineState(upPSO)
        encoder.setBuffer(y, offset: yOffset, index: 0)
        encoder.setBuffer(b, offset: bOffset, index: 1)
        encoder.setBuffer(scratch, offset: 0, index: 2)
        encoder.setBytes(&n, length: 4, index: 3)
        encoder.setBytes(&r, length: 4, index: 4)
        encoder.setBytes(&ys, length: 4, index: 5)
        encoder.setBytes(&s, length: 4, index: 6)
        encoder.setBytes(&t, length: 4, index: 7)
        let width = min(upPSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: rows, height: (tokens + Self.upTile - 1) / Self.upTile, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    /// Most members `encodeGroup` takes (the shader's fixed arrays).
    static let maximumGroup = 4

    /// One member of a grouped delta: its output and its A / B^T, as byte
    /// offsets into the adapter buffer.
    struct Member {
        let y: MTLBuffer
        let yStrideElements: Int
        let rows: Int
        let aOffset: Int
        let bOffset: Int
    }

    /// The deltas of up to four projections that read the same `x`
    /// (`columns` wide) in two dispatches over (tokens x members), into an open
    /// encoder. `scratch` holds `members * tokens * rank` floats.
    func encodeGroup(
        encoder: MTLComputeCommandEncoder,
        x: MTLBuffer, xStrideElements: Int, columns: Int,
        members: [Member], adapter: MTLBuffer, scratch: MTLBuffer,
        rank: Int, tokens: Int, scale: Float
    ) throws {
        guard tokens > 0, !members.isEmpty else { return }
        guard members.count <= Self.maximumGroup, rank > 0, rank <= Self.maximumRank,
            scratch.length >= members.count * tokens * rank * MemoryLayout<Float>.stride
        else {
            throw MetalError.invalidState(
                "LoRA group: \(members.count) members of rank \(rank) do not fit the scratch")
        }
        func four(_ values: [Int]) -> (UInt32, UInt32, UInt32, UInt32) {
            let v = values.map(UInt32.init) + Array(repeating: 0, count: 4 - values.count)
            return (v[0], v[1], v[2], v[3])
        }
        let uOffsets = members.indices.map { $0 * tokens * rank }
        var down = LoRAGroupDownParams(
            k: UInt32(columns), rank: UInt32(rank), xStride: UInt32(xStrideElements),
            members: UInt32(members.count), aOffset: four(members.map { $0.aOffset / 4 }),
            uOffset: four(uOffsets))
        encoder.setComputePipelineState(downGroupPSO)
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(adapter, offset: 0, index: 1)
        encoder.setBuffer(scratch, offset: 0, index: 2)
        encoder.setBytes(&down, length: MemoryLayout<LoRAGroupDownParams>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: tokens, height: members.count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * rank, height: 1, depth: 1))

        var up = LoRAGroupUpParams(
            rank: UInt32(rank), tokens: UInt32(tokens), members: UInt32(members.count),
            scale: scale, rows: four(members.map(\.rows)),
            yStride: four(members.map(\.yStrideElements)),
            bOffset: four(members.map { $0.bOffset / 4 }), uOffset: four(uOffsets))
        encoder.setComputePipelineState(upGroupPSO)
        for slot in 0..<4 {
            encoder.setBuffer(members[min(slot, members.count - 1)].y, offset: 0, index: slot)
        }
        encoder.setBuffer(adapter, offset: 0, index: 4)
        encoder.setBuffer(scratch, offset: 0, index: 5)
        encoder.setBytes(&up, length: MemoryLayout<LoRAGroupUpParams>.stride, index: 6)
        let widest = members.map(\.rows).max() ?? 0
        let width = min(upGroupPSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(
                width: widest, height: (tokens + Self.upTile - 1) / Self.upTile,
                depth: members.count),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}

/// `LoRAGroupDown` in utility.metal: field for field, four-wide arrays.
struct LoRAGroupDownParams {
    var k: UInt32
    var rank: UInt32
    var xStride: UInt32
    var members: UInt32
    var aOffset: (UInt32, UInt32, UInt32, UInt32)
    var uOffset: (UInt32, UInt32, UInt32, UInt32)
}

/// `LoRAGroupUp` in utility.metal.
struct LoRAGroupUpParams {
    var rank: UInt32
    var tokens: UInt32
    var members: UInt32
    var scale: Float
    var rows: (UInt32, UInt32, UInt32, UInt32)
    var yStride: (UInt32, UInt32, UInt32, UInt32)
    var bOffset: (UInt32, UInt32, UInt32, UInt32)
    var uOffset: (UInt32, UInt32, UInt32, UInt32)
}
