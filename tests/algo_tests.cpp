#include "hwods/algo/sdf.hpp"
#include "hwods/algo/tiles.hpp"
#include "hwods/algo/walk_on_stars.hpp"
#include "hwods/algo/weight.hpp"
#include "hwods/eval/fdm.hpp"
#include "hwods/eval/laplace.hpp"
#include "hwods/eval/rmse.hpp"

#include <algorithm>
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
    expect(grid.histogram_slots == 0, "shared empty hist");
  }
  {
    const auto grid = hwods::TileGrid::build({1.f, 1.f}, 4, 4, [](auto min, auto max) {
      return min[0] >= 0.25f && max[0] <= 0.5f && min[1] >= 0.25f && max[1] <= 0.5f;
    });
    expect(grid.histogram_slots > 0, "obstacle hist slots");
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
  {
    auto grid = hwods::FdmGrid::zeros(32, 32, {1.f, 1.f}, 0.f);
    hwods::solve_helmholtz_fdm(grid);
    std::vector<float> approx;
    std::vector<float> ref;
    approx.reserve(grid.nx * grid.ny);
    ref.reserve(grid.nx * grid.ny);
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        const float x = (static_cast<float>(i) + 0.5f) / static_cast<float>(grid.nx);
        const float y = (static_cast<float>(j) + 0.5f) / static_cast<float>(grid.ny);
        approx.push_back(grid.values[grid.idx(i, j)].re);
        ref.push_back(hwods::analytical_laplace_empty_tank(x, y, 1.f));
      }
    }
    const float rmse = hwods::relative_rmse(approx, ref);
    expect(rmse < 0.01f, "FDM k=0 matches analytical laplace <= 1%");
  }
  {
    const auto scene = hwods::make_example_scene(1.f, 1.f);
    auto grid = hwods::FdmGrid::zeros(32, 32, {1.f, 1.f}, 2.f);
    hwods::solve_helmholtz_fdm(grid, &scene);
    float max_abs = 0.f;
    for (const auto& v : grid.values) {
      max_abs = std::max(max_abs, std::fabs(v.re));
      max_abs = std::max(max_abs, std::fabs(v.im));
    }
    expect(max_abs < 10.f, "FDM example scene stays bounded");
    const float hx = 1.f / 32.f;
    const float x = (9.f + 0.5f) * hx;
    const float y = (17.f + 0.5f) * hx;
    expect(scene.obstacle_distance({x, y}) > 0.f, "example hole is water");
    expect(std::fabs(grid.values[grid.idx(9, 17)].re) < 1e-4f, "FDM hole cell is 0");
  }
  {
    const auto scene = hwods::make_example_scene(1.f, 1.f);

    hwods::WalkConfig cfg;
    cfg.k = 1.0f;
    cfg.helmholtz = true;
    cfg.max_steps = 32;
    cfg.radius_cap = 0.2f;
    uint32_t seed = 42;

    const auto est_inside = hwods::estimate_wos_at_point(scene, {0.25f, 0.55f}, cfg, 64, seed);
    expect(est_inside.value.re == 0.f, "WoS inside obstacle is 0");

    const auto est_water = hwods::estimate_wos_at_point(scene, {0.10f, 0.50f}, cfg, 64, seed);
    expect(est_water.value.re > 0.5f && est_water.value.re < 1.1f, "WoS near left wall ~0.9");
  }

  if (fails) {
    std::cerr << fails << " test(s) failed\n";
    return 1;
  }
  std::cout << "All algo tests passed\n";
  return 0;
}
