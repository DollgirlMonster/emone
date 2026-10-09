#include <metal_stdlib>
using namespace metal;

// Unquantized GEMV over a bf16 weight matrix.
//
// The quantized kernels beside this one exist because weights are normally
// stored packed with per-group scales and biases. A tensor kept at the
// checkpoint's own bf16 has none of that: no packing to unpick, no group to
// rescale, one multiply per element. It is here so a build can leave chosen
// tensors unquantized -- the 8-bit build promotes the families whose error
// does not average away -- without a separate code path at the call site.
//
// Layout matches the quantized kernels exactly: one simdgroup per row, each
// lane taking two adjacent columns, so a caller can swap between them on the
// tensor's dtype and change nothing else.
kernel void bf16_gemv_simd(
    device const bfloat* weights [[buffer(0)]],
    device const half*   x       [[buffer(1)]],
    device half*         y       [[buffer(2)]],
    constant uint&       rows    [[buffer(3)]],
    constant uint&       columns [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint lanes = 32;
    const uint row = tg_idx * rows_per_threadgroup + sg_idx;
    if (row >= rows) return;

    device const bfloat* row_weights = weights + row * columns;
    float accumulator = 0.0f;
    // Columns are a multiple of 64 for every tensor this serves, which is the
    // same guarantee the affine kernels rely on for their group stride.
    for (uint base = 0; base < columns; base += lanes * 2u) {
        const uint first = base + lane * 2u;
        accumulator = fma(float(row_weights[first]), float(x[first]),
                          accumulator);
        accumulator = fma(float(row_weights[first + 1u]), float(x[first + 1u]),
                          accumulator);
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) y[row] = half(accumulator);
}

// `bf16_gemv_simd` for a one-row weight over many independent inputs: one
// simdgroup per input row instead of one dispatch per input row. A prefill
// chunk's shared-expert scalar gate (2560 -> 1) used to issue one dispatch per
// token -- 16,931 per layer on a 16.9K-token prompt, ~6 s of GPU across the
// stack on an M1 Max. Each simdgroup runs exactly `bf16_gemv_simd`'s loop and
// reduction for its row, so every output is the same bits.
kernel void bf16_gemv_simd_one_row_many_x(
    device const bfloat* weights  [[buffer(0)]],
    device const half*   x        [[buffer(1)]],
    device half*         y        [[buffer(2)]],
    constant uint&       count    [[buffer(3)]],
    constant uint&       columns  [[buffer(4)]],
    constant uint&       xStride  [[buffer(5)]],
    constant uint&       yStride  [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint lanes = 32;
    const uint token = tg_idx * rows_per_threadgroup + sg_idx;
    if (token >= count) return;
    device const half* xr = x + token * xStride;
    float accumulator = 0.0f;
    for (uint base = 0; base < columns; base += lanes * 2u) {
        const uint first = base + lane * 2u;
        accumulator = fma(float(weights[first]), float(xr[first]), accumulator);
        accumulator = fma(float(weights[first + 1u]), float(xr[first + 1u]),
                          accumulator);
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) y[token * yStride] = half(accumulator);
}
