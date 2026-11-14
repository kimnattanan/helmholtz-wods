#include "hwods/algo/walk_on_stars.hpp"

#include "hwods/algo/weight.hpp"

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

  const float radius = std::min(q.distance, cfg.radius_cap);
  const auto weight = step_weight(cfg, radius);
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
    auto step = walk_step(scene, p, cfg, seed);
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

}  // namespace hwods
