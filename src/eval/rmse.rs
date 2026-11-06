//! Relative RMSE helpers for complex / real fields (proposal §4, §6).

use crate::eval::fdm::Complex32;

pub fn relative_rmse(approx: &[f32], reference: &[f32]) -> f32 {
    assert_eq!(approx.len(), reference.len());
    assert!(!reference.is_empty());
    let mut err = 0.0f32;
    let mut ref_norm = 0.0f32;
    for (a, r) in approx.iter().zip(reference.iter()) {
        let d = a - r;
        err += d * d;
        ref_norm += r * r;
    }
    if ref_norm < 1e-20 {
        return err.sqrt();
    }
    (err / ref_norm).sqrt()
}

pub fn relative_rmse_complex(approx: &[Complex32], reference: &[Complex32]) -> f32 {
    assert_eq!(approx.len(), reference.len());
    assert!(!reference.is_empty());
    let mut err = 0.0f32;
    let mut ref_norm = 0.0f32;
    for (a, r) in approx.iter().zip(reference.iter()) {
        let d = *a - *r;
        err += d.norm_sq();
        ref_norm += r.norm_sq();
    }
    if ref_norm < 1e-20 {
        return err.sqrt();
    }
    (err / ref_norm).sqrt()
}

/// Mean |∂n u| on obstacle samples, normalized by range of u (proposal §4).
pub fn mean_normal_derivative_over_range(
    normal_derivs: &[f32],
    field_min: f32,
    field_max: f32,
) -> f32 {
    assert!(!normal_derivs.is_empty());
    let mean = normal_derivs.iter().map(|v| v.abs()).sum::<f32>() / normal_derivs.len() as f32;
    let range = (field_max - field_min).abs().max(1e-12);
    mean / range
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_fields_zero_rmse() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(relative_rmse(&a, &a) < 1e-7);
    }
}
