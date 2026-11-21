/// Deterministic 64-bit PRNG. Same algorithm as the inline copies in the
/// pre-refactor tests — but conforms to `RandomNumberGenerator` so it
/// composes with stdlib random APIs.
public struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    public init(seed: UInt64) {
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z &>> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z &>> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z &>> 31)
    }

    /// Uniform float in [lo, hi). 24-bit mantissa precision.
    public mutating func uniform(_ lo: Float, _ hi: Float) -> Float {
        let u = Float(next() >> 40) / Float(1 << 24)
        return lo + (hi - lo) * u
    }

}
