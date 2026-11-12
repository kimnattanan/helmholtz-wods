#pragma once

#include "hwods/algo/sdf.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace hwods {

struct WalkConfig {
  float k = 0.f;
  uint32_t max_steps = 64;
  float eps = 1e-4f;
  bool helmholtz = false;
  float radius_cap = std::numeric_limits<float>::infinity();
};

struct WalkStep {
  std::array<float, 2> from{0, 0};
  std::array<float, 2> to{0, 0};
  float radius = 0.f;
  float weight = 1.f;
  std::optional<BoundaryKind> hit_boundary;
};

struct WalkResult {
  std::vector<WalkStep> steps;
  std::array<float, 2> endpoint{0, 0};
  float total_weight = 1.f;
  std::optional<BoundaryKind> terminated_on;
};

std::array<float, 2> sample_direction(uint32_t& seed);

std::optional<WalkStep> walk_step(const SceneSdf& scene, std::array<float, 2> p,
                                  const WalkConfig& cfg, uint32_t& seed);

WalkResult walk_until_boundary(const SceneSdf& scene, std::array<float, 2> p,
                               const WalkConfig& cfg, uint32_t& seed);

}  // namespace hwods
