//! Per-step Helmholtz weight `1 / J₀(k r)` (proposal §3).
//! For the Laplace checkpoint set every step weight to 1.

/// Bessel J₀ via `libm` (matches CPU reference and shader LUT later).
pub fn bessel_j0(x: f32) -> f32 {
    libm::j0f(x)
}

/// Step weight for Helmholtz. Returns `None` if J₀ is near a root (unstable).
pub fn helmholtz_step_weight(k: f32, radius: f32) -> Option<f32> {
    let j = bessel_j0(k * radius);
    if j.abs() < 1e-6 {
        None
    } else {
        Some(1.0 / j)
    }
}

/// Laplace / empty-tank checkpoint: every step weight is 1.
pub fn laplace_step_weight() -> f32 {
    1.0
}

/// Product of step weights along a walk.
pub fn accumulate_weights(weights: &[f32]) -> f32 {
    weights.iter().copied().product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn j0_at_zero_is_one() {
        assert!((bessel_j0(0.0) - 1.0).abs() < 1e-6);
        assert!((helmholtz_step_weight(1.0, 0.0).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_near_root() {
        // First positive root of J₀ is ≈ 2.4048
        assert!(helmholtz_step_weight(1.0, 2.4048255).is_none());
    }
}
