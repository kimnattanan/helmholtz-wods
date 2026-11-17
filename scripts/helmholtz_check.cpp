#include "hwods/algo/sdf.hpp"
#include "hwods/algo/walk_on_stars.hpp"
#include "hwods/eval/fdm.hpp"
#include "hwods/eval/laplace.hpp"
#include "hwods/eval/rmse.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
  const std::array<float, 2> tank{1.f, 1.f};
  constexpr std::size_t nx = 32;
  constexpr std::size_t ny = 32;
  constexpr float k = 2.0f;
  constexpr uint32_t walks_per_point = 256;
  uint32_t seed = 42;

  const auto scene = hwods::make_example_scene(tank[0], tank[1]);

  std::cout << "=== Helmholtz Walk on Stars vs FDM Check ===\n";
  std::cout << "Tank: " << tank[0] << "x" << tank[1] << ", k=" << k << "\n";
  std::cout << "Grid: " << nx << "x" << ny << " (" << walks_per_point << " walks/point)\n";
  std::cout << "Obstacles: " << scene.obstacles.size() << " obstacles\n\n";

  // 1. Solve FDM Ground Truth
  std::cout << "1. Solving 2D Helmholtz FDM reference... " << std::flush;
  auto fdm_grid = hwods::FdmGrid::zeros(nx, ny, tank, k);
  hwods::solve_helmholtz_fdm(fdm_grid, &scene);
  std::cout << "done.\n";

  // 2. Compute CPU Walk on Stars
  std::cout << "2. Running CPU Walk on Stars estimator... " << std::flush;
  hwods::WalkConfig cfg;
  cfg.k = k;
  cfg.helmholtz = true;
  cfg.max_steps = 64;
  cfg.radius_cap = 0.15f; // Ensures k * r < 2.4048 (avoids Bessel J0 zero crossing)
  cfg.eps = 1e-4f;

  const auto wos_field = hwods::estimate_wos_field(scene, nx, ny, cfg, walks_per_point, seed);
  std::cout << "done.\n\n";

  // 3. Filter points outside obstacles for RMSE
  std::vector<hwods::Complex32> fdm_water;
  std::vector<hwods::Complex32> wos_water;
  float field_min = 1e9f;
  float field_max = -1e9f;

  const float hx = tank[0] / static_cast<float>(nx);
  const float hy = tank[1] / static_cast<float>(ny);

  for (std::size_t j = 0; j < ny; ++j) {
    const float y = (static_cast<float>(j) + 0.5f) * hy;
    for (std::size_t i = 0; i < nx; ++i) {
      const float x = (static_cast<float>(i) + 0.5f) * hx;
      if (scene.obstacle_distance({x, y}) > 0.01f) {
        const auto fdm_val = fdm_grid.values[fdm_grid.idx(i, j)];
        const auto wos_val = wos_field[j * nx + i];
        fdm_water.push_back(fdm_val);
        wos_water.push_back(wos_val);

        field_min = std::min(field_min, fdm_val.re);
        field_max = std::max(field_max, fdm_val.re);
      }
    }
  }

  const float rmse = hwods::relative_rmse_complex(wos_water, fdm_water);

  // 4. Sample normal derivatives along obstacle boundaries (Circle + Box)
  std::vector<std::array<float, 2>> boundary_samples;

  // Obstacle 1: Circle boundary points
  const std::array<float, 2> circle_c{0.3f, 0.5f};
  const float circle_r = 0.12f;
  for (int s = 0; s < 16; ++s) {
    const float theta = 2.f * 3.14159265f * static_cast<float>(s) / 32.f;
    boundary_samples.push_back({circle_c[0] + circle_r * std::cos(theta),
                                circle_c[1] + circle_r * std::sin(theta)});
  }

  // Obstacle 2: Box boundary points
  const std::array<float, 2> box_c{0.8f, 0.72f};
  const std::array<float, 2> box_he{0.1f, 0.07f};
  for (int s = 0; s < 8; ++s) {
    const float t = (static_cast<float>(s) + 0.5f) / 8.f;
    const float x = box_c[0] - box_he[0] + 2.f * box_he[0] * t;
    const float y = box_c[1] - box_he[1] + 2.f * box_he[1] * t;
    boundary_samples.push_back({x, box_c[1] - box_he[1]}); // bottom edge
    boundary_samples.push_back({x, box_c[1] + box_he[1]}); // top edge
    boundary_samples.push_back({box_c[0] - box_he[0], y}); // left edge
    boundary_samples.push_back({box_c[0] + box_he[0], y}); // right edge
  }

  constexpr float delta = 0.015f;
  std::vector<float> normal_derivs;
  normal_derivs.reserve(boundary_samples.size());

  for (const auto& bp : boundary_samples) {
    const auto q = scene.query(bp, 1e-4f);
    const auto n = q.gradient;

    // Sample along normal into water
    uint32_t seed_a = seed++;
    uint32_t seed_b = seed_a; // coupled seed to reduce MC gradient noise
    const auto est_0 = hwods::estimate_wos_at_point(
        scene, {bp[0] + n[0] * 1e-3f, bp[1] + n[1] * 1e-3f}, cfg, 256, seed_a);
    const auto est_1 = hwods::estimate_wos_at_point(
        scene, {bp[0] + n[0] * delta, bp[1] + n[1] * delta}, cfg, 256, seed_b);

    const float dn = std::fabs(est_1.value.re - est_0.value.re) / delta;
    normal_derivs.push_back(dn);
  }

  const float mean_dn = hwods::mean_normal_derivative_over_range(normal_derivs, field_min, field_max);

  std::cout << std::fixed << std::setprecision(5);
  std::cout << "Results:\n";
  std::cout << "  Relative RMSE (WoS vs FDM)     = " << rmse << "\n";
  std::cout << "  Mean |dn u| / range(u)         = " << mean_dn << "\n";
  std::cout << "  Field range [min, max]         = [" << field_min << ", " << field_max << "]\n";
  std::cout << "  Obstacle boundary sample pts   = " << boundary_samples.size() << "\n";
  std::cout << "  Water evaluation grid cells    = " << wos_water.size() << " / " << (nx * ny) << "\n\n";

  std::cout << "OK: Week 2 evaluation check complete. Both RMSE and normal derivative recorded.\n";
  return 0;
}
