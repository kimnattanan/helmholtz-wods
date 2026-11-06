//! One Walk-on-Stars step on a zero-Neumann / Dirichlet SDF scene.
//!
//! On Neumann hit: reflect from the closest obstacle point.
//! On Dirichlet / tile boundary: terminate and deposit weight.

use crate::algo::sdf::{BoundaryKind, SceneSdf};
use crate::algo::weight::{helmholtz_step_weight, laplace_step_weight};

#[derive(Clone, Copy, Debug)]
pub struct WalkConfig {
    pub k: f32,
    pub max_steps: u32,
    pub eps: f32,
    /// If true, use Helmholtz `1/J₀(kr)`; else Laplace weight 1.
    pub helmholtz: bool,
    /// Optional cap on star radius (e.g. tile half-diagonal).
    pub radius_cap: f32,
}

impl Default for WalkConfig {
    fn default() -> Self {
        Self {
            k: 0.0,
            max_steps: 64,
            eps: 1e-4,
            helmholtz: false,
            radius_cap: f32::INFINITY,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WalkStep {
    pub from: [f32; 2],
    pub to: [f32; 2],
    pub radius: f32,
    pub weight: f32,
    pub hit_boundary: Option<BoundaryKind>,
}

#[derive(Clone, Debug)]
pub struct WalkResult {
    pub steps: Vec<WalkStep>,
    pub endpoint: [f32; 2],
    pub total_weight: f32,
    pub terminated_on: Option<BoundaryKind>,
}

/// Uniform direction on the unit circle from a tiny LCG (GPU will use a better RNG).
pub fn sample_direction(seed: &mut u32) -> [f32; 2] {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    let u = (*seed >> 8) as f32 / (1u32 << 24) as f32;
    let theta = std::f32::consts::TAU * u;
    [theta.cos(), theta.sin()]
}

fn step_weight(cfg: &WalkConfig, radius: f32) -> Option<f32> {
    if cfg.helmholtz {
        helmholtz_step_weight(cfg.k, radius)
    } else {
        Some(laplace_step_weight())
    }
}

/// Single WoS step from `p`. Shrinks radius by SDF; reflects on zero-Neumann.
pub fn walk_step(
    scene: &SceneSdf,
    p: [f32; 2],
    cfg: &WalkConfig,
    seed: &mut u32,
) -> Option<WalkStep> {
    let q = scene.query(p, cfg.eps);
    if q.distance <= cfg.eps {
        return Some(WalkStep {
            from: p,
            to: p,
            radius: 0.0,
            weight: 1.0,
            hit_boundary: Some(q.boundary),
        });
    }

    let radius = q.distance.min(cfg.radius_cap);
    let weight = step_weight(cfg, radius)?;
    let dir = sample_direction(seed);
    let mut to = [p[0] + dir[0] * radius, p[1] + dir[1] * radius];

    // If the star exits the water domain, pull back to closest boundary.
    let q_to = scene.query(to, cfg.eps);
    let hit = if q_to.distance < 0.0 || q_to.distance <= cfg.eps {
        let (closest, kind) = scene.closest_boundary_point(to);
        to = closest;
        Some(kind)
    } else {
        None
    };

    // Zero-Neumann: reflect by projecting onto the tangent after hitting.
    let to = if let Some(BoundaryKind::ZeroNeumann) = hit {
        let (c, _) = scene.closest_boundary_point(to);
        let n = scene.query(c, cfg.eps).gradient;
        // Place slightly outside solid along outward domain normal.
        [c[0] + n[0] * cfg.eps * 2.0, c[1] + n[1] * cfg.eps * 2.0]
    } else {
        to
    };

    Some(WalkStep {
        from: p,
        to,
        radius,
        weight,
        hit_boundary: hit,
    })
}

/// Run a short walk until Dirichlet / max steps (used for Laplace checks & Pass C stub).
pub fn walk_until_boundary(
    scene: &SceneSdf,
    mut p: [f32; 2],
    cfg: &WalkConfig,
    seed: &mut u32,
) -> WalkResult {
    let mut steps = Vec::new();
    let mut total_weight = 1.0;
    let mut terminated_on = None;

    for _ in 0..cfg.max_steps {
        let Some(step) = walk_step(scene, p, cfg, seed) else {
            break;
        };
        total_weight *= step.weight;
        p = step.to;
        let hit = step.hit_boundary;
        steps.push(step);
        if let Some(BoundaryKind::Dirichlet) = hit {
            terminated_on = hit;
            break;
        }
        if hit == Some(BoundaryKind::ZeroNeumann) {
            // Reflect and continue (Neumann does not terminate).
            continue;
        }
        // Optional: stop if we left the tank numerically.
        if scene.tank_wall_distance(p) <= cfg.eps {
            terminated_on = Some(BoundaryKind::Dirichlet);
            break;
        }
    }

    WalkResult {
        steps,
        endpoint: p,
        total_weight,
        terminated_on,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tank_walk_reaches_wall() {
        let scene = SceneSdf::empty_tank(1.0, 1.0);
        let cfg = WalkConfig {
            max_steps: 32,
            radius_cap: 0.5,
            ..WalkConfig::default()
        };
        let mut seed = 1;
        let result = walk_until_boundary(&scene, [0.5, 0.5], &cfg, &mut seed);
        assert!(!result.steps.is_empty());
        assert!(scene.tank_wall_distance(result.endpoint) < 0.05);
    }
}
