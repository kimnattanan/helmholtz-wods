#include "hwods/eval/fdm.hpp"

#include "hwods/algo/sdf.hpp"
#include "hwods/eval/laplace.hpp"

#include <cmath>

namespace hwods {

FdmGrid FdmGrid::zeros(std::size_t nx, std::size_t ny, std::array<float, 2> tank_size,
                       float k) {
  FdmGrid g;
  g.nx = nx;
  g.ny = ny;
  g.tank_size = tank_size;
  g.k = k;
  g.values.assign(nx * ny, Complex32::zero());
  return g;
}

void solve_helmholtz_fdm(FdmGrid& grid, const SceneSdf* scene) {
  const std::size_t nx = grid.nx;
  const std::size_t ny = grid.ny;
  if (nx == 0 || ny == 0) return;

  const float w = grid.tank_size[0];
  const float h = grid.tank_size[1];
  const float hx = w / static_cast<float>(nx);
  const float hy = h / static_cast<float>(ny);
  const float inv_hx2 = 1.f / (hx * hx);
  const float inv_hy2 = 1.f / (hy * hy);
  const float k2 = grid.k * grid.k;

  auto is_obstacle = [&](float px, float py) -> bool {
    if (!scene || scene->obstacles.empty()) return false;
    return scene->obstacle_distance({px, py}) <= 0.f;
  };

  // Initialize with linear ramp for fast convergence
  for (std::size_t j = 0; j < ny; ++j) {
    const float y = (static_cast<float>(j) + 0.5f) * hy;
    for (std::size_t i = 0; i < nx; ++i) {
      const float x = (static_cast<float>(i) + 0.5f) * hx;
      if (is_obstacle(x, y)) {
        grid.values[grid.idx(i, j)] = Complex32::zero();
      } else {
        grid.values[grid.idx(i, j)] = Complex32{dirichlet_on_boundary(x, y, w), 0.f};
      }
    }
  }

  constexpr std::size_t max_iters = 5000;
  constexpr float tol_sq = 1e-12f;
  const float omega = (k2 > 0.f) ? 1.0f : 1.2f;

  for (std::size_t iter = 0; iter < max_iters; ++iter) {
    float max_delta_sq = 0.f;

    for (std::size_t j = 0; j < ny; ++j) {
      const float y = (static_cast<float>(j) + 0.5f) * hy;
      for (std::size_t i = 0; i < nx; ++i) {
        const float x = (static_cast<float>(i) + 0.5f) * hx;
        const std::size_t center_idx = grid.idx(i, j);

        if (is_obstacle(x, y)) {
          grid.values[center_idx] = Complex32::zero();
          continue;
        }

        float coeff = -k2;
        Complex32 rhs = Complex32::zero();

        // Left (-x)
        if (i == 0) {
          coeff += 2.f * inv_hx2;
          rhs = rhs + Complex32{2.f * dirichlet_on_boundary(0.f, y, w) * inv_hx2, 0.f};
        } else if (is_obstacle(x - hx, y)) {
          // Zero-Neumann: no flux across boundary, no coeff addition
        } else {
          coeff += inv_hx2;
          rhs = rhs + grid.values[grid.idx(i - 1, j)] * inv_hx2;
        }

        // Right (+x)
        if (i + 1 == nx) {
          coeff += 2.f * inv_hx2;
          rhs = rhs + Complex32{2.f * dirichlet_on_boundary(w, y, w) * inv_hx2, 0.f};
        } else if (is_obstacle(x + hx, y)) {
          // Zero-Neumann
        } else {
          coeff += inv_hx2;
          rhs = rhs + grid.values[grid.idx(i + 1, j)] * inv_hx2;
        }

        // Bottom (-y)
        if (j == 0) {
          coeff += 2.f * inv_hy2;
          rhs = rhs + Complex32{2.f * dirichlet_on_boundary(x, 0.f, w) * inv_hy2, 0.f};
        } else if (is_obstacle(x, y - hy)) {
          // Zero-Neumann
        } else {
          coeff += inv_hy2;
          rhs = rhs + grid.values[grid.idx(i, j - 1)] * inv_hy2;
        }

        // Top (+y)
        if (j + 1 == ny) {
          coeff += 2.f * inv_hy2;
          rhs = rhs + Complex32{2.f * dirichlet_on_boundary(x, h, w) * inv_hy2, 0.f};
        } else if (is_obstacle(x, y + hy)) {
          // Zero-Neumann
        } else {
          coeff += inv_hy2;
          rhs = rhs + grid.values[grid.idx(i, j + 1)] * inv_hy2;
        }

        if (std::fabs(coeff) < 1e-12f) continue;

        const Complex32 target = rhs / coeff;
        const Complex32 current = grid.values[center_idx];
        const Complex32 updated = current * (1.f - omega) + target * omega;

        const float delta_sq = (updated - current).norm_sq();
        if (delta_sq > max_delta_sq) {
          max_delta_sq = delta_sq;
        }

        grid.values[center_idx] = updated;
      }
    }

    if (max_delta_sq < tol_sq) {
      break;
    }
  }
}

void solve_helmholtz_fdm_stub(FdmGrid& grid) {
  solve_helmholtz_fdm(grid, nullptr);
}

}  // namespace hwods
