#include "hwods/algo/tiles.hpp"

#include <cassert>
#include <cmath>

namespace hwods {

std::array<float, 2> Tile::center() const {
  return {0.5f * (min[0] + max[0]), 0.5f * (min[1] + max[1])};
}

float Tile::half_diagonal() const {
  const float hx = 0.5f * (max[0] - min[0]);
  const float hy = 0.5f * (max[1] - min[1]);
  return std::hypot(hx, hy);
}

bool Tile::contains(std::array<float, 2> p) const {
  return p[0] >= min[0] && p[0] < max[0] && p[1] >= min[1] && p[1] < max[1];
}

TileGrid TileGrid::build(std::array<float, 2> tank_size, uint32_t nx, uint32_t ny,
                         ObstacleHitFn obstacle_hit) {
  assert(nx >= 1 && ny >= 1);
  TileGrid grid;
  grid.tank_size = tank_size;
  grid.nx = nx;
  grid.ny = ny;

  const float w = tank_size[0];
  const float h = tank_size[1];
  const float dx = w / static_cast<float>(nx);
  const float dy = h / static_cast<float>(ny);

  grid.tiles.reserve(static_cast<std::size_t>(nx * ny));
  for (uint32_t j = 0; j < ny; ++j) {
    for (uint32_t i = 0; i < nx; ++i) {
      Tile t;
      t.min = {static_cast<float>(i) * dx, static_cast<float>(j) * dy};
      t.max = {static_cast<float>(i + 1) * dx, static_cast<float>(j + 1) * dy};
      t.id = TileId{j * nx + i};
      t.touches_obstacle = obstacle_hit(t.min, t.max);
      grid.tiles.push_back(t);
    }
  }

  uint32_t edge_id = 0;

  for (uint32_t j = 0; j < ny; ++j) {
    for (uint32_t i = 0; i + 1 < nx; ++i) {
      const Tile& left = grid.tiles[j * nx + i];
      const Tile& right = grid.tiles[j * nx + i + 1];
      const bool touches = left.touches_obstacle || right.touches_obstacle;
      const float x = right.min[0];
      CoEdge e;
      e.id = edge_id++;
      e.left_or_bottom = left.id;
      e.right_or_top = right.id;
      e.a = {x, left.min[1]};
      e.b = {x, left.max[1]};
      e.touches_obstacle = touches;
      if (touches) {
        e.kind = CoEdgeKind::InteriorVertical;
        e.histogram_slot = grid.histogram_slots++;
      } else {
        e.kind = CoEdgeKind::SharedEmpty;
      }
      grid.co_edges.push_back(e);
    }
  }

  for (uint32_t j = 0; j + 1 < ny; ++j) {
    for (uint32_t i = 0; i < nx; ++i) {
      const Tile& bottom = grid.tiles[j * nx + i];
      const Tile& top = grid.tiles[(j + 1) * nx + i];
      const bool touches = bottom.touches_obstacle || top.touches_obstacle;
      const float y = top.min[1];
      CoEdge e;
      e.id = edge_id++;
      e.left_or_bottom = bottom.id;
      e.right_or_top = top.id;
      e.a = {bottom.min[0], y};
      e.b = {bottom.max[0], y};
      e.touches_obstacle = touches;
      if (touches) {
        e.kind = CoEdgeKind::InteriorHorizontal;
        e.histogram_slot = grid.histogram_slots++;
      } else {
        e.kind = CoEdgeKind::SharedEmpty;
      }
      grid.co_edges.push_back(e);
    }
  }

  return grid;
}

std::optional<TileId> TileGrid::tile_index(std::array<float, 2> p) const {
  const float w = tank_size[0];
  const float h = tank_size[1];
  if (p[0] < 0.f || p[1] < 0.f || p[0] >= w || p[1] >= h) return std::nullopt;
  uint32_t i = static_cast<uint32_t>(std::floor((p[0] / w) * static_cast<float>(nx)));
  uint32_t j = static_cast<uint32_t>(std::floor((p[1] / h) * static_cast<float>(ny)));
  i = std::min(i, nx - 1);
  j = std::min(j, ny - 1);
  return TileId{j * nx + i};
}

}  // namespace hwods
