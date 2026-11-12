#include "hwods/algo/sdf.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace hwods {
namespace {

float sd_box(std::array<float, 2> p, std::array<float, 2> b) {
  const float dx = std::fabs(p[0]) - b[0];
  const float dy = std::fabs(p[1]) - b[1];
  const float ox = std::max(dx, 0.f);
  const float oy = std::max(dy, 0.f);
  return std::hypot(ox, oy) + std::min(std::max(dx, dy), 0.f);
}

std::optional<float> combine_nodes(const std::vector<SdfNode>& nodes,
                                   std::array<float, 2> p) {
  if (nodes.empty()) return std::nullopt;
  float acc = nodes[0].signed_distance(p);
  std::optional<SdfOp> pending = nodes[0].op_with_next;
  for (std::size_t i = 1; i < nodes.size(); ++i) {
    const float d = nodes[i].signed_distance(p);
    const SdfOp op = pending.value_or(SdfOp::Union);
    switch (op) {
      case SdfOp::Union:
        acc = std::min(acc, d);
        break;
      case SdfOp::Intersection:
        acc = std::max(acc, d);
        break;
      case SdfOp::Difference:
        acc = std::max(acc, -d);
        break;
    }
    pending = nodes[i].op_with_next;
  }
  return acc;
}

}  // namespace

SdfNode SdfNode::circle(std::array<float, 2> center, float r) {
  SdfNode n;
  n.kind = SdfPrimitiveKind::Circle;
  n.radius = r;
  n.transform[2] = center;
  return n;
}

SdfNode SdfNode::box_centered(std::array<float, 2> center, std::array<float, 2> he) {
  SdfNode n;
  n.kind = SdfPrimitiveKind::Box;
  n.half_extents = he;
  n.transform[2] = center;
  return n;
}

std::array<float, 2> SdfNode::world_to_local(std::array<float, 2> p) const {
  const auto& t = transform[2];
  const auto& x = transform[0];
  const auto& y = transform[1];
  const float qx = p[0] - t[0];
  const float qy = p[1] - t[1];
  const float det = x[0] * y[1] - x[1] * y[0];
  const float inv = std::fabs(det) < 1e-12f ? 1.f : 1.f / det;
  return {(qx * y[1] - qy * y[0]) * inv, (-qx * x[1] + qy * x[0]) * inv};
}

float SdfNode::signed_distance(std::array<float, 2> p) const {
  const auto q = world_to_local(p);
  switch (kind) {
    case SdfPrimitiveKind::Circle:
      return std::hypot(q[0], q[1]) - radius;
    case SdfPrimitiveKind::Box:
      return sd_box(q, half_extents);
    case SdfPrimitiveKind::RoundedBox:
      return sd_box(q, half_extents) - radius;
  }
  return 0.f;
}

SceneSdf SceneSdf::empty_tank(float width, float height) {
  return SceneSdf{.tank_size = {width, height}, .obstacles = {}};
}

float SceneSdf::tank_wall_distance(std::array<float, 2> p) const {
  const float w = tank_size[0];
  const float h = tank_size[1];
  return std::min({p[0], w - p[0], p[1], h - p[1]});
}

float SceneSdf::obstacle_distance(std::array<float, 2> p) const {
  return combine_nodes(obstacles, p).value_or(std::numeric_limits<float>::infinity());
}

float SceneSdf::domain_distance(std::array<float, 2> p) const {
  const float to_wall = tank_wall_distance(p);
  if (obstacles.empty()) return to_wall;
  return std::min(to_wall, obstacle_distance(p));
}

SdfQuery SceneSdf::query(std::array<float, 2> p, float eps) const {
  const float d_wall = tank_wall_distance(p);
  const float d_obs =
      obstacles.empty() ? std::numeric_limits<float>::infinity() : obstacle_distance(p);

  SdfQuery q;
  if (d_obs < d_wall) {
    q.distance = d_obs;
    q.boundary = BoundaryKind::ZeroNeumann;
  } else {
    q.distance = d_wall;
    q.boundary = BoundaryKind::Dirichlet;
  }

  const float gx =
      domain_distance({p[0] + eps, p[1]}) - domain_distance({p[0] - eps, p[1]});
  const float gy =
      domain_distance({p[0], p[1] + eps}) - domain_distance({p[0], p[1] - eps});
  const float len = std::max(std::hypot(gx, gy), 1e-12f);
  q.gradient = {gx / len, gy / len};
  return q;
}

std::pair<std::array<float, 2>, BoundaryKind> SceneSdf::closest_boundary_point(
    std::array<float, 2> p) const {
  const SdfQuery q = query(p, 1e-4f);
  return {{p[0] - q.gradient[0] * q.distance, p[1] - q.gradient[1] * q.distance},
          q.boundary};
}

float star_radius(float domain_dist, float tile_half_diag) {
  constexpr float kSqrt2 = 1.41421356237f;
  return std::min({std::max(domain_dist, 0.f), tile_half_diag, kSqrt2});
}

}  // namespace hwods
