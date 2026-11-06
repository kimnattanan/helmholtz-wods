//! CPU finite-difference reference for Helmholtz (stub for Week 2).
//!
//! Real BC → standing wave; complex Dirichlet → traveling crest (proposal §2).

use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Debug, Default)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
    pub fn norm_sq(self) -> f32 {
        self.re * self.re + self.im * self.im
    }
}

impl Add for Complex32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex32 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul<f32> for Complex32 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FdmGrid {
    pub nx: usize,
    pub ny: usize,
    pub tank_size: [f32; 2],
    pub k: f32,
    pub values: Vec<Complex32>,
}

impl FdmGrid {
    pub fn zeros(nx: usize, ny: usize, tank_size: [f32; 2], k: f32) -> Self {
        Self {
            nx,
            ny,
            tank_size,
            k,
            values: vec![Complex32::ZERO; nx * ny],
        }
    }

    pub fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }
}

/// Placeholder: Week 2 fills a Gauss–Seidel / sparse solve for
/// (Δ + k²) u = 0 with Dirichlet outer BC and zero-Neumann obstacles.
pub fn solve_helmholtz_fdm_stub(grid: &mut FdmGrid) {
    let w = grid.tank_size[0];
    let nx = grid.nx;
    for j in 0..grid.ny {
        for i in 0..nx {
            let x = (i as f32 + 0.5) / nx as f32 * w;
            let idx = j * nx + i;
            // Standing-wave-compatible seed from Laplace analytical (k→0 limit).
            grid.values[idx] = Complex32::new(1.0 - x / w, 0.0);
        }
    }
}
