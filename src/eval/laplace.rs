//! Empty-tank Laplace checkpoint (proposal §6).
//!
//! Boundary: left=1, right=0, top/bottom = 1−x.
//! Analytical solution: u(x,y) = 1 − x.

/// Dirichlet value on the outer tank boundary for the proposal BC set.
pub fn dirichlet_on_boundary(x: f32, _y: f32, tank_width: f32) -> f32 {
    1.0 - (x / tank_width)
}

/// Exact solution of Laplace on the empty unit (or width×height) rectangle
/// with the proposal Dirichlet data.
pub fn analytical_laplace_empty_tank(x: f32, _y: f32, tank_width: f32) -> f32 {
    1.0 - (x / tank_width)
}

/// Sample analytical field on a regular grid (row-major, y slowest or x fastest).
pub fn sample_grid(nx: usize, ny: usize, tank_size: [f32; 2]) -> Vec<f32> {
    let [w, h] = tank_size;
    let mut out = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        let y = (j as f32 + 0.5) / ny as f32 * h;
        for i in 0..nx {
            let x = (i as f32 + 0.5) / nx as f32 * w;
            out.push(analytical_laplace_empty_tank(x, y, w));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytical_matches_boundaries() {
        let w = 1.0;
        assert!((analytical_laplace_empty_tank(0.0, 0.3, w) - 1.0).abs() < 1e-6);
        assert!((analytical_laplace_empty_tank(1.0, 0.7, w) - 0.0).abs() < 1e-6);
        assert!((dirichlet_on_boundary(0.25, 0.0, w) - 0.75).abs() < 1e-6);
        assert!((dirichlet_on_boundary(0.25, 1.0, w) - 0.75).abs() < 1e-6);
    }
}
