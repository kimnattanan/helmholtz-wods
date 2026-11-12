#include "hwods/algo/sdf.hpp"
#include "hwods/algo/tiles.hpp"
#include "hwods/algo/walk_on_stars.hpp"
#include "hwods/algo/weight.hpp"
#include "hwods/eval/laplace.hpp"
#include "hwods/eval/rmse.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

int fails = 0;

void expect(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    ++fails;
  }
}

}  // namespace

int main() {
  {
    const auto node = hwods::SdfNode::circle({0.5f, 0.5f}, 0.1f);
    expect(std::fabs(node.signed_distance({0.5f, 0.5f}) + 0.1f) < 1e-5f, "circle center");
    expect(std::fabs(node.signed_distance({0.5f, 0.7f}) - 0.1f) < 1e-5f, "circle rim");
  }
  {
    const auto scene = hwods::SceneSdf::empty_tank(1.f, 1.f);
    expect(scene.domain_distance({0.5f, 0.5f}) > 0.4f, "empty tank center");
    expect(scene.domain_distance({-0.1f, 0.5f}) < 0.f, "outside tank");
  }
  {
    auto scene = hwods::SceneSdf::empty_tank(1.f, 1.f);
    scene.obstacles.push_back(hwods::SdfNode::circle({0.5f, 0.5f}, 0.1f));
    expect(scene.domain_distance({0.5f, 0.5f}) < 0.f, "obstacle carves water");
    expect(scene.domain_distance({0.5f, 0.8f}) > 0.f, "water outside obstacle");
  }
  {
    const auto grid =
        hwods::TileGrid::build({1.f, 1.f}, 4, 4, [](auto, auto) { return false; });
    expect(grid.tiles.size() == 16, "tile count");
    expect(grid.histogram_slots() == 0, "shared empty hist");
  }
  {
    const auto grid = hwods::TileGrid::build({1.f, 1.f}, 4, 4, [](auto min, auto max) {
      return min[0] >= 0.25f && max[0] <= 0.5f && min[1] >= 0.25f && max[1] <= 0.5f;
    });
    expect(grid.histogram_slots() > 0, "obstacle hist slots");
  }
  {
    expect(std::fabs(hwods::bessel_j0(0.f) - 1.f) < 1e-6f, "J0(0)");
    expect(hwods::helmholtz_step_weight(1.f, 0.f).value_or(0.f) == 1.f, "weight at 0");
    expect(!hwods::helmholtz_step_weight(1.f, 2.4048255f).has_value(), "reject J0 root");
  }
  {
    const auto scene = hwods::SceneSdf::empty_tank(1.f, 1.f);
    hwods::WalkConfig cfg;
    cfg.max_steps = 32;
    cfg.radius_cap = 0.5f;
    uint32_t seed = 1;
    const auto result = hwods::walk_until_boundary(scene, {0.5f, 0.5f}, cfg, seed);
    expect(!result.steps.empty(), "walk has steps");
    expect(scene.tank_wall_distance(result.endpoint) < 0.05f, "walk reaches wall");
  }
  {
    expect(std::fabs(hwods::analytical_laplace_empty_tank(0.f, 0.3f, 1.f) - 1.f) < 1e-6f,
           "laplace left");
    expect(std::fabs(hwods::analytical_laplace_empty_tank(1.f, 0.7f, 1.f)) < 1e-6f,
           "laplace right");
  }
  {
    const std::vector<float> a{1.f, 2.f, 3.f};
    expect(hwods::relative_rmse(a, a) < 1e-7f, "rmse identical");
  }

  if (fails) {
    std::cerr << fails << " test(s) failed\n";
    return 1;
  }
  std::cout << "All algo tests passed\n";
  return 0;
}
