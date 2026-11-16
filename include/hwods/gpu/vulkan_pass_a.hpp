#pragma once

#include "hwods/algo/sdf.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace hwods {

/// Host copy of `Frame` in shaders/pass_a.comp (std140).
struct PassAParams {
  std::array<float, 2> tank_size{1.f, 1.f};
  float k = 0.f;
  float eps = 1e-4f;
  uint32_t max_steps = 64;
  uint32_t walks_per_bucket = 64;
  uint32_t seed = 1;
};

/// One `vec2` histogram bin. The Pass A stub writes
/// (domain distance at the tank center, node count) into bin 0.
struct PassABin {
  float re = 0.f;
  float im = 0.f;
};

/// Headless Pass A: uniform Frame + SDF SSBO + histogram SSBO.
/// No swapchain — pass_a.comp does not write an image.
class VulkanPassA {
 public:
  VulkanPassA();
  ~VulkanPassA();

  VulkanPassA(const VulkanPassA&) = delete;
  VulkanPassA& operator=(const VulkanPassA&) = delete;

  /// Upload `params` and `nodes`, dispatch pass_a.comp, read `histogram_bins` back.
  std::vector<PassABin> dispatch(const PassAParams& params, const std::vector<SdfNode>& nodes,
                                 uint32_t histogram_bins);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Circle, axis-aligned box, and rounded box inside a unit tank.
std::vector<SdfNode> demo_pass_a_nodes();

/// Dispatch the demo scene once. Throws if bin 0 disagrees with the CPU SDF.
PassABin run_pass_a_demo();

}  // namespace hwods
