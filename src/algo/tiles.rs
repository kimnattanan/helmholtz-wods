//! Regular rectangular subdomain tiling and shared co-edges.
//! Empty tiles (no obstacle hit) share one operator; only obstacle-touching
//! co-edges allocate their own histogram slots (proposal §5).

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoEdgeKind {
    InteriorVertical,
    InteriorHorizontal,
    /// Shared empty-tile operator (no per-edge histogram).
    SharedEmpty,
}

#[derive(Clone, Debug)]
pub struct CoEdge {
    pub id: u32,
    pub kind: CoEdgeKind,
    pub left_or_bottom: TileId,
    pub right_or_top: TileId,
    /// Axis-aligned segment in tank coordinates: (x0,y0)→(x1,y1).
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub touches_obstacle: bool,
    pub histogram_slot: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct Tile {
    pub id: TileId,
    pub min: [f32; 2],
    pub max: [f32; 2],
    pub touches_obstacle: bool,
}

impl Tile {
    pub fn center(&self) -> [f32; 2] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
        ]
    }

    pub fn half_diagonal(&self) -> f32 {
        let hx = 0.5 * (self.max[0] - self.min[0]);
        let hy = 0.5 * (self.max[1] - self.min[1]);
        hx.hypot(hy)
    }

    pub fn contains(&self, p: [f32; 2]) -> bool {
        p[0] >= self.min[0]
            && p[0] < self.max[0]
            && p[1] >= self.min[1]
            && p[1] < self.max[1]
    }
}

#[derive(Clone, Debug)]
pub struct TileGrid {
    pub tank_size: [f32; 2],
    pub nx: u32,
    pub ny: u32,
    pub tiles: Vec<Tile>,
    pub co_edges: Vec<CoEdge>,
}

impl TileGrid {
    /// Build an `nx × ny` grid over `[0,w]×[0,h]`.
    /// `obstacle_hit(tile_min, tile_max)` marks tiles/edges that need local WoS.
    pub fn build(
        tank_size: [f32; 2],
        nx: u32,
        ny: u32,
        mut obstacle_hit: impl FnMut([f32; 2], [f32; 2]) -> bool,
    ) -> Self {
        assert!(nx >= 1 && ny >= 1);
        let [w, h] = tank_size;
        let dx = w / nx as f32;
        let dy = h / ny as f32;

        let mut tiles = Vec::with_capacity((nx * ny) as usize);
        for j in 0..ny {
            for i in 0..nx {
                let min = [i as f32 * dx, j as f32 * dy];
                let max = [(i + 1) as f32 * dx, (j + 1) as f32 * dy];
                let touches = obstacle_hit(min, max);
                tiles.push(Tile {
                    id: TileId(j * nx + i),
                    min,
                    max,
                    touches_obstacle: touches,
                });
            }
        }

        let mut co_edges = Vec::new();
        let mut hist_slot = 0u32;
        let mut edge_id = 0u32;

        // Vertical interfaces between (i,j) and (i+1,j).
        for j in 0..ny {
            for i in 0..nx - 1 {
                let left = &tiles[(j * nx + i) as usize];
                let right = &tiles[(j * nx + i + 1) as usize];
                let touches = left.touches_obstacle || right.touches_obstacle;
                let x = right.min[0];
                let a = [x, left.min[1]];
                let b = [x, left.max[1]];
                let (kind, histogram_slot) = if touches {
                    let slot = Some(hist_slot);
                    hist_slot += 1;
                    (CoEdgeKind::InteriorVertical, slot)
                } else {
                    (CoEdgeKind::SharedEmpty, None)
                };
                co_edges.push(CoEdge {
                    id: edge_id,
                    kind,
                    left_or_bottom: left.id,
                    right_or_top: right.id,
                    a,
                    b,
                    touches_obstacle: touches,
                    histogram_slot,
                });
                edge_id += 1;
            }
        }

        // Horizontal interfaces between (i,j) and (i,j+1).
        for j in 0..ny - 1 {
            for i in 0..nx {
                let bottom = &tiles[(j * nx + i) as usize];
                let top = &tiles[((j + 1) * nx + i) as usize];
                let touches = bottom.touches_obstacle || top.touches_obstacle;
                let y = top.min[1];
                let a = [bottom.min[0], y];
                let b = [bottom.max[0], y];
                let (kind, histogram_slot) = if touches {
                    let slot = Some(hist_slot);
                    hist_slot += 1;
                    (CoEdgeKind::InteriorHorizontal, slot)
                } else {
                    (CoEdgeKind::SharedEmpty, None)
                };
                co_edges.push(CoEdge {
                    id: edge_id,
                    kind,
                    left_or_bottom: bottom.id,
                    right_or_top: top.id,
                    a,
                    b,
                    touches_obstacle: touches,
                    histogram_slot,
                });
                edge_id += 1;
            }
        }

        Self {
            tank_size,
            nx,
            ny,
            tiles,
            co_edges,
        }
    }

    pub fn tile_index(&self, p: [f32; 2]) -> Option<TileId> {
        let [w, h] = self.tank_size;
        if p[0] < 0.0 || p[1] < 0.0 || p[0] >= w || p[1] >= h {
            return None;
        }
        let i = ((p[0] / w) * self.nx as f32).floor() as u32;
        let j = ((p[1] / h) * self.ny as f32).floor() as u32;
        let i = i.min(self.nx - 1);
        let j = j.min(self.ny - 1);
        Some(TileId(j * self.nx + i))
    }

    pub fn histogram_slots(&self) -> u32 {
        self.co_edges
            .iter()
            .filter_map(|e| e.histogram_slot)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_grid_shares_operator() {
        let grid = TileGrid::build([1.0, 1.0], 4, 4, |_, _| false);
        assert_eq!(grid.tiles.len(), 16);
        assert!(grid.co_edges.iter().all(|e| e.histogram_slot.is_none()));
        assert_eq!(grid.histogram_slots(), 0);
    }

    #[test]
    fn obstacle_tile_gets_histogram() {
        let grid = TileGrid::build([1.0, 1.0], 4, 4, |min, max| {
            // Mark center tile (1,1) roughly around [0.25,0.5)²
            min[0] >= 0.25 && max[0] <= 0.5 && min[1] >= 0.25 && max[1] <= 0.5
        });
        assert!(grid.histogram_slots() > 0);
        assert!(grid.co_edges.iter().any(|e| e.touches_obstacle));
    }
}
