//! Helmholtz wave height via Walk on Decomposed Subdomains (WoDS).
//!
//! Pipeline (proposal §5):
//! SDF → tiles/co-edges → Pass A (WoS × 1/J₀) → Pass B (solve I−Q) →
//! Pass C (assemble complex field) → Pass D (phase × height).

pub mod algo;
pub mod eval;
pub mod gpu;

pub use algo::{sdf, tiles, walk_on_stars, weight};
