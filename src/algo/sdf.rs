//! Scene SDF as a CSG tree of circle / box / rounded-box nodes.
//! Obstacles are zero-Neumann; tank walls are Dirichlet (handled separately).

use std::f32::consts::SQRT_2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryKind {
    /// Outer tank walls (Dirichlet).
    Dirichlet,
    /// Obstacle surfaces (zero-Neumann reflection).
    ZeroNeumann,
}

#[derive(Clone, Copy, Debug)]
pub enum SdfPrimitive {
    Circle { radius: f32 },
    Box { half_extents: [f32; 2] },
    RoundedBox { half_extents: [f32; 2], radius: f32 },
}

#[derive(Clone, Copy, Debug)]
pub enum SdfOp {
    Union,
    Intersection,
    Difference,
}

#[derive(Clone, Debug)]
pub struct SdfNode {
    pub primitive: SdfPrimitive,
    /// Column-major 2×3 affine: columns are x-basis, y-basis, translation.
    pub transform: [[f32; 2]; 3],
    pub op_with_next: Option<SdfOp>,
}

impl SdfNode {
    pub fn circle(center: [f32; 2], radius: f32) -> Self {
        Self {
            primitive: SdfPrimitive::Circle { radius },
            transform: [[1.0, 0.0], [0.0, 1.0], center],
            op_with_next: None,
        }
    }

    pub fn box_centered(center: [f32; 2], half_extents: [f32; 2]) -> Self {
        Self {
            primitive: SdfPrimitive::Box { half_extents },
            transform: [[1.0, 0.0], [0.0, 1.0], center],
            op_with_next: None,
        }
    }

    fn world_to_local(&self, p: [f32; 2]) -> [f32; 2] {
        let t = self.transform[2];
        let q = [p[0] - t[0], p[1] - t[1]];
        let x = self.transform[0];
        let y = self.transform[1];
        // Inverse of orthonormal-ish basis (exact for pure translation+rotation+uniform scale later).
        let det = x[0] * y[1] - x[1] * y[0];
        let inv_det = if det.abs() < 1e-12 { 1.0 } else { 1.0 / det };
        [
            (q[0] * y[1] - q[1] * y[0]) * inv_det,
            (-q[0] * x[1] + q[1] * x[0]) * inv_det,
        ]
    }

    pub fn signed_distance(&self, p: [f32; 2]) -> f32 {
        let q = self.world_to_local(p);
        match self.primitive {
            SdfPrimitive::Circle { radius } => (q[0] * q[0] + q[1] * q[1]).sqrt() - radius,
            SdfPrimitive::Box { half_extents } => sd_box(q, half_extents),
            SdfPrimitive::RoundedBox {
                half_extents,
                radius,
            } => sd_box(q, half_extents) - radius,
        }
    }
}

fn sd_box(p: [f32; 2], b: [f32; 2]) -> f32 {
    let d = [(p[0].abs() - b[0]), (p[1].abs() - b[1])];
    let outside = [d[0].max(0.0), d[1].max(0.0)];
    outside[0].hypot(outside[1]) + d[0].min(0.0).max(d[1].min(0.0))
}

/// Tank [0,w]×[0,h] water domain minus obstacle CSG.
#[derive(Clone, Debug)]
pub struct SceneSdf {
    pub tank_size: [f32; 2],
    pub obstacles: Vec<SdfNode>,
}

#[derive(Clone, Copy, Debug)]
pub struct SdfQuery {
    pub distance: f32,
    pub gradient: [f32; 2],
    pub boundary: BoundaryKind,
}

impl SceneSdf {
    pub fn empty_tank(width: f32, height: f32) -> Self {
        Self {
            tank_size: [width, height],
            obstacles: Vec::new(),
        }
    }

    /// Convention: water domain SDF > 0 in water, ≤ 0 inside solids / outside tank.
    /// Obstacle nodes use solid SDF (negative inside). Water = min(wall, obstacle_sd).
    pub fn domain_distance(&self, p: [f32; 2]) -> f32 {
        let to_wall = self.tank_wall_distance(p);
        if self.obstacles.is_empty() {
            return to_wall;
        }
        let to_obstacle = self.obstacle_distance(p);
        to_wall.min(to_obstacle)
    }

    pub fn tank_wall_distance(&self, p: [f32; 2]) -> f32 {
        let [w, h] = self.tank_size;
        (p[0]).min(w - p[0]).min(p[1]).min(h - p[1])
    }

    pub fn obstacle_distance(&self, p: [f32; 2]) -> f32 {
        combine_nodes(&self.obstacles, p).unwrap_or(f32::INFINITY)
    }

    pub fn query(&self, p: [f32; 2], eps: f32) -> SdfQuery {
        let d_wall = self.tank_wall_distance(p);
        let d_obs = if self.obstacles.is_empty() {
            f32::INFINITY
        } else {
            self.obstacle_distance(p)
        };

        // Nearest boundary is whichever SDF is smaller (both positive in water).
        let (distance, boundary) = if d_obs < d_wall {
            (d_obs, BoundaryKind::ZeroNeumann)
        } else {
            (d_wall, BoundaryKind::Dirichlet)
        };

        let g = [
            self.domain_distance([p[0] + eps, p[1]]) - self.domain_distance([p[0] - eps, p[1]]),
            self.domain_distance([p[0], p[1] + eps]) - self.domain_distance([p[0], p[1] - eps]),
        ];
        let len = (g[0] * g[0] + g[1] * g[1]).sqrt().max(1e-12);
        SdfQuery {
            distance,
            gradient: [g[0] / len, g[1] / len],
            boundary,
        }
    }

    /// Closest point estimate on the active boundary using SDF + gradient.
    pub fn closest_boundary_point(&self, p: [f32; 2]) -> ([f32; 2], BoundaryKind) {
        let q = self.query(p, 1e-4);
        (
            [
                p[0] - q.gradient[0] * q.distance,
                p[1] - q.gradient[1] * q.distance,
            ],
            q.boundary,
        )
    }
}

fn combine_nodes(nodes: &[SdfNode], p: [f32; 2]) -> Option<f32> {
    let mut iter = nodes.iter();
    let first = iter.next()?;
    let mut acc = first.signed_distance(p);
    let mut pending_op = first.op_with_next;
    for node in iter {
        let d = node.signed_distance(p);
        acc = match pending_op.unwrap_or(SdfOp::Union) {
            SdfOp::Union => acc.min(d),
            SdfOp::Intersection => acc.max(d),
            SdfOp::Difference => acc.max(-d),
        };
        pending_op = node.op_with_next;
    }
    Some(acc)
}

/// Safe star radius: distance to domain boundary, capped by tile half-diagonal.
pub fn star_radius(domain_dist: f32, tile_half_diag: f32) -> f32 {
    domain_dist.max(0.0).min(tile_half_diag).min(SQRT_2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_distance_at_center() {
        let node = SdfNode::circle([0.5, 0.5], 0.1);
        assert!((node.signed_distance([0.5, 0.5]) + 0.1).abs() < 1e-5);
        assert!((node.signed_distance([0.5, 0.7]) - 0.1).abs() < 1e-5);
    }

    #[test]
    fn empty_tank_center_is_positive() {
        let scene = SceneSdf::empty_tank(1.0, 1.0);
        assert!(scene.domain_distance([0.5, 0.5]) > 0.4);
        assert!(scene.domain_distance([-0.1, 0.5]) < 0.0);
    }

    #[test]
    fn obstacle_carves_water() {
        let mut scene = SceneSdf::empty_tank(1.0, 1.0);
        scene.obstacles.push(SdfNode::circle([0.5, 0.5], 0.1));
        assert!(scene.domain_distance([0.5, 0.5]) < 0.0);
        assert!(scene.domain_distance([0.5, 0.8]) > 0.0);
    }
}
