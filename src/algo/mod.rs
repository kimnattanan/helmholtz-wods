//! CPU-side algorithm building blocks shared with GPU shaders later.

pub mod sdf;
pub mod tiles;
pub mod walk_on_stars;
pub mod weight;

pub use sdf::{BoundaryKind, SceneSdf, SdfNode, SdfOp, SdfPrimitive};
pub use tiles::{CoEdge, TileGrid, TileId};
pub use walk_on_stars::{WalkConfig, WalkStep, walk_step};
pub use weight::helmholtz_step_weight;
