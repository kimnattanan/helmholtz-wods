#include "hwods/algo/walk_on_stars.hpp"

#include "hwods/algo/sdf.hpp"
#include "hwods/algo/weight.hpp"
#include "hwods/eval/laplace.hpp"

#include <cmath>
#include <numbers>

namespace hwods {
namespace {

std::optional<float> step_weight(const WalkConfig& cfg, float radius) {
  if (cfg.helmholtz) return helmholtz_step_weight(cfg.k, radius);
  return laplace_step_weight();
}

}  // namespace

std::array<float, 2> sample_direction(uint32_t& seed) {
  seed = seed * 1664525u + 1013904223u;
  const float u = static_cast<float>(seed >> 8) / static_cast<float>(1u << 24);
  const float theta = std::numbers::pi_v<float> * 2.f * u;
  return {std::cos(theta), std::sin(theta)};
}

std::optional<WalkStep> walk_step(const SceneSdf& scene, std::array<float, 2> p,
                                  const WalkConfig& cfg, uint32_t& seed) {
  const SdfQuery q = scene.query(p, cfg.eps);
  if (q.distance <= cfg.eps) {
    return WalkStep{p, p, 0.f, 1.f, q.boundary};
  }

  float radius = std::min(q.distance, cfg.radius_cap);
  auto weight = step_weight(cfg, radius);

  if (!weight) return std::nullopt;

  const auto dir = sample_direction(seed);
  std::array<float, 2> to{p[0] + dir[0] * radius, p[1] + dir[1] * radius};

  const SdfQuery q_to = scene.query(to, cfg.eps);
  std::optional<BoundaryKind> hit;
  if (q_to.distance <= cfg.eps) {
    to = {to[0] - q_to.gradient[0] * q_to.distance,
          to[1] - q_to.gradient[1] * q_to.distance};
    hit = q_to.boundary;
    if (hit == BoundaryKind::ZeroNeumann) {
      const auto n = q_to.gradient;
      to = {to[0] + n[0] * cfg.eps * 2.f, to[1] + n[1] * cfg.eps * 2.f};
    }
  }

  return WalkStep{p, to, radius, *weight, hit};
}

WalkResult walk_until_boundary(const SceneSdf& scene, std::array<float, 2> p,
                               const WalkConfig& cfg, uint32_t& seed) {
  WalkResult result;
  result.endpoint = p;
  result.total_weight = 1.f;

  for (uint32_t i = 0; i < cfg.max_steps; ++i) {
    const auto step = walk_step(scene, p, cfg, seed);
    if (!step) break;
    result.total_weight *= step->weight;
    p = step->to;
    const auto hit = step->hit_boundary;
    result.steps.push_back(*step);

    if (hit == BoundaryKind::Dirichlet) {
      result.terminated_on = hit;
      break;
    }
    if (hit == BoundaryKind::ZeroNeumann) {
      continue;
    }
    if (scene.tank_wall_distance(p) <= cfg.eps) {
      result.terminated_on = BoundaryKind::Dirichlet;
      break;
    }
  }

  result.endpoint = p;
  return result;
}

WosEstimate estimate_wos_at_point(const SceneSdf& scene, std::array<float, 2> p,
                                  const WalkConfig& cfg, uint32_t walks,
                                  uint32_t& seed) {
  WosEstimate est;
  if (walks == 0) return est;

  if (scene.obstacle_distance(p) <= 0.f) {
    return est;
  }

  float acc = 0.f;
  uint32_t valid_walks = 0;

  for (uint32_t w = 0; w < walks; ++w) {
    const auto walk = walk_until_boundary(scene, p, cfg, seed);
    if (walk.terminated_on == BoundaryKind::Dirichlet ||
        scene.tank_wall_distance(walk.endpoint) <= cfg.eps * 2.f) {
      const float u_b =
          dirichlet_on_boundary(walk.endpoint[0], walk.endpoint[1], scene.tank_size[0]);
      acc += walk.total_weight * u_b;
      valid_walks++;
    }
  }

  est.walks_completed = valid_walks;
  if (valid_walks > 0) {
    est.value = Complex32{acc / static_cast<float>(valid_walks), 0.f};
  }
  return est;
}

std::vector<Complex32> estimate_wos_field(const SceneSdf& scene,
                                         std::size_t nx, std::size_t ny,
                                         const WalkConfig& cfg,
                                         uint32_t walks_per_point,
                                         uint32_t seed) {
  std::vector<Complex32> field;
  field.reserve(nx * ny);
  const float w = scene.tank_size[0];
  const float h = scene.tank_size[1];
  const float hx = w / static_cast<float>(nx);
  const float hy = h / static_cast<float>(ny);

  for (std::size_t j = 0; j < ny; ++j) {
    const float y = (static_cast<float>(j) + 0.5f) * hy;
    for (std::size_t i = 0; i < nx; ++i) {
      const float x = (static_cast<float>(i) + 0.5f) * hx;
      const auto est = estimate_wos_at_point(scene, {x, y}, cfg, walks_per_point, seed);
      field.push_back(est.value);
    }
  }
  return field;
}

}  // namespace hwods
