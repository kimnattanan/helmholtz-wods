#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace hwods {

struct TileId {
  uint32_t value = 0;
  bool operator==(TileId o) const { return value == o.value; }
};

enum class CoEdgeKind { InteriorVertical, InteriorHorizontal, SharedEmpty };

struct CoEdge {
  uint32_t id = 0;
  CoEdgeKind kind = CoEdgeKind::SharedEmpty;
  TileId left_or_bottom{};
  TileId right_or_top{};
  std::array<float, 2> a{0, 0};
  std::array<float, 2> b{0, 0};
  bool touches_obstacle = false;
  std::optional<uint32_t> histogram_slot;
};

struct Tile {
  TileId id{};
  std::array<float, 2> min{0, 0};
  std::array<float, 2> max{0, 0};
  bool touches_obstacle = false;

  std::array<float, 2> center() const;
  float half_diagonal() const;
  bool contains(std::array<float, 2> p) const;
};

struct TileGrid {
  std::array<float, 2> tank_size{1.f, 1.f};
  uint32_t nx = 1;
  uint32_t ny = 1;
  std::vector<Tile> tiles;
  std::vector<CoEdge> co_edges;

  using ObstacleHitFn =
      std::function<bool(std::array<float, 2> min, std::array<float, 2> max)>;

  static TileGrid build(std::array<float, 2> tank_size, uint32_t nx, uint32_t ny,
                        ObstacleHitFn obstacle_hit);

  std::optional<TileId> tile_index(std::array<float, 2> p) const;
  uint32_t histogram_slots() const;
};

}  // namespace hwods
