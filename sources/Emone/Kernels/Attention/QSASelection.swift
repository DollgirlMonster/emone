import Metal

/// A QSA chunk's key selection in both forms.
///
/// `mask` is one byte per visible key per query: the form the attention kernel
/// originally consumed, and still what the activation dump compares. `indices`
/// is the same selection compacted to ascending key ids with `counts[query]`
/// valid entries -- what the kernel loops over now.
///
/// The two are kept together deliberately. The mask is how the selection is
/// built and how it is checked; the indices are how it is executed. Deriving
/// one from the other at the point of use is what made attention O(context)
/// when the selection is O(budget).
///
/// A selection made on the GPU (`QSAIndexer.encodeSelectPrefill`) has only the
/// mask, and it is complete only once its command buffer has run: the masked
/// flash attention reads nothing else, and a consumer that needs the indices
/// on the host (the ANE, the activation dump) takes the host selection.
struct QSASelection {
    let mask: MTLBuffer
    let maskStride: Int
    let indices: MTLBuffer?
    let indexStride: Int
    let counts: MTLBuffer?
}
