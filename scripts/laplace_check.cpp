#include "hwods/algo/sdf.hpp"
#include "hwods/algo/tiles.hpp"
#include "hwods/algo/walk_on_stars.hpp"
#include "hwods/eval/laplace.hpp"
#include "hwods/eval/rmse.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
  const std::array<float, 2> tank{1.f, 1.f};
  constexpr std::size_t nx = 32;
  constexpr std::size_t ny = 32;

  float max_bdry_err = 0.f;
  for (float y : {0.f, 1.f}) {
    for (int i = 0; i <= 64; ++i) {
      const float x = static_cast<float>(i) / 64.f;
      max_bdry_err = std::max(
          max_bdry_err,
          std::fabs(hwods::dirichlet_on_boundary(x, y, tank[0]) -
                    hwods::analytical_laplace_empty_tank(x, y, tank[0])));
    }
  }
  for (float x : {0.f, 1.f}) {
    for (int j = 0; j <= 64; ++j) {
      const float y = static_cast<float>(j) / 64.f;
      max_bdry_err = std::max(
          max_bdry_err,
          std::fabs(hwods::dirichlet_on_boundary(x, y, tank[0]) -
                    hwods::analytical_laplace_empty_tank(x, y, tank[0])));
    }
  }

  const auto scene = hwods::SceneSdf::empty_tank(tank[0], tank[1]);
  const auto grid = hwods::TileGrid::build(tank, 8, 8, [](auto, auto) { return false; });

  hwods::WalkConfig cfg;
  cfg.max_steps = 48;
  cfg.radius_cap = grid.tiles[0].half_diagonal();
  cfg.helmholtz = false;

  std::vector<float> approx;
  std::vector<float> reference;
  constexpr uint32_t walks_per_point = 64;
  uint32_t seed = 42;

  approx.reserve(nx * ny);
  reference.reserve(nx * ny);
  for (std::size_t j = 0; j < ny; ++j) {
    for (std::size_t i = 0; i < nx; ++i) {
      const float x = (static_cast<float>(i) + 0.5f) / static_cast<float>(nx) * tank[0];
      const float y = (static_cast<float>(j) + 0.5f) / static_cast<float>(ny) * tank[1];
      float acc = 0.f;
      for (uint32_t w = 0; w < walks_per_point; ++w) {
        const auto walk = hwods::walk_until_boundary(scene, {x, y}, cfg, seed);
        const float u_b =
            hwods::dirichlet_on_boundary(walk.endpoint[0], walk.endpoint[1], tank[0]);
        acc += walk.total_weight * u_b;
      }
      approx.push_back(acc / static_cast<float>(walks_per_point));
      reference.push_back(hwods::analytical_laplace_empty_tank(x, y, tank[0]));
    }
  }

  const float rmse = hwods::relative_rmse(approx, reference);
  std::size_t shared_empty = 0;
  for (const auto& e : grid.co_edges) {
    if (!e.histogram_slot) ++shared_empty;
  }

  std::cout << "helmholtz-wods Laplace empty-tank check\n";
  std::cout << "  max |BC − analytical|     = " << max_bdry_err << "\n";
  std::cout << "  relative RMSE (WoS vs u*) = " << rmse << "  (" << walks_per_point
            << " walks/pt, " << nx << "×" << ny << ")\n";
  std::cout << "  tiles " << grid.nx << "×" << grid.ny << "  shared-empty co-edges "
            << shared_empty << "\n";

  if (max_bdry_err > 1e-5f) {
    std::cerr << "FAIL: boundary does not match analytical\n";
    return 1;
  }
  std::cout << "OK: analytical Laplace identity holds; WoS estimate printed for W2 tuning\n";
  return 0;
}
