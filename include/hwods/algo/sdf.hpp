#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace hwods {

enum class BoundaryKind { Dirichlet, ZeroNeumann };

enum class SdfPrimitiveKind { Circle, Box, RoundedBox };

enum class SdfOp { Union, Intersection, Difference };

struct SdfNode {
  SdfPrimitiveKind kind = SdfPrimitiveKind::Circle;
  float radius = 0.f;                       // circle / rounded
  std::array<float, 2> half_extents{0, 0};  // box / rounded
  /// Column-major 2×3: x-basis, y-basis, translation.
  std::array<std::array<float, 2>, 3> transform{{{{1, 0}}, {{0, 1}}, {{0, 0}}}};
  std::optional<SdfOp> op_with_next;

  static SdfNode circle(std::array<float, 2> center, float r);
  static SdfNode box_centered(std::array<float, 2> center, std::array<float, 2> he);
  float signed_distance(std::array<float, 2> p) const;

 private:
  std::array<float, 2> world_to_local(std::array<float, 2> p) const;
};

struct SdfQuery {
  float distance = 0.f;
  std::array<float, 2> gradient{0, 0};
  BoundaryKind boundary = BoundaryKind::Dirichlet;
};

struct SceneSdf {
  std::array<float, 2> tank_size{1.f, 1.f};
  std::vector<SdfNode> obstacles;

  static SceneSdf empty_tank(float width, float height);
  float domain_distance(std::array<float, 2> p) const;
  float tank_wall_distance(std::array<float, 2> p) const;
  float obstacle_distance(std::array<float, 2> p) const;
  SdfQuery query(std::array<float, 2> p, float eps) const;
  std::pair<std::array<float, 2>, BoundaryKind> closest_boundary_point(
      std::array<float, 2> p) const;
};

float star_radius(float domain_dist, float tile_half_diag);

}  // namespace hwods
