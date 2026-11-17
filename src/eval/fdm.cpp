#include "hwods/eval/fdm.hpp"

#include "hwods/algo/sdf.hpp"
#include "hwods/eval/laplace.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace hwods {
namespace {

struct WaterCell {
  std::size_t i = 0;
  std::size_t j = 0;
  std::size_t grid_index = 0;
};

/// Factor A in place and solve A x = re and A y = im. A zero pivot is a free
/// unknown (sealed cavity at k = 0); that component is set to 0.
void solve_dense(std::vector<double>& a, std::size_t n, std::vector<double>& re,
                 std::vector<double>& im) {
  constexpr double kPivotEps = 1e-12;
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot = col;
    double best = std::fabs(a[col * n + col]);
    for (std::size_t row = col + 1; row < n; ++row) {
      const double v = std::fabs(a[row * n + col]);
      if (v > best) {
        best = v;
        pivot = row;
      }
    }
    if (best < kPivotEps) {
      re[col] = 0.0;
      im[col] = 0.0;
      continue;
    }
    if (pivot != col) {
      for (std::size_t c = col; c < n; ++c) std::swap(a[col * n + c], a[pivot * n + c]);
      std::swap(re[col], re[pivot]);
      std::swap(im[col], im[pivot]);
    }
    const double diag = a[col * n + col];
    for (std::size_t row = col + 1; row < n; ++row) {
      const double factor = a[row * n + col] / diag;
      if (factor == 0.0) continue;
      a[row * n + col] = 0.0;
      for (std::size_t c = col + 1; c < n; ++c) a[row * n + c] -= factor * a[col * n + c];
      re[row] -= factor * re[col];
      im[row] -= factor * im[col];
    }
  }

  for (std::size_t row = n; row-- > 0;) {
    const double diag = a[row * n + row];
    if (std::fabs(diag) < kPivotEps) {
      re[row] = 0.0;
      im[row] = 0.0;
      continue;
    }
    double br = re[row];
    double bi = im[row];
    for (std::size_t c = row + 1; c < n; ++c) {
      br -= a[row * n + c] * re[c];
      bi -= a[row * n + c] * im[c];
    }
    re[row] = br / diag;
    im[row] = bi / diag;
  }
}

}  // namespace

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

  std::vector<WaterCell> water;
  water.reserve(nx * ny);
  std::vector<int> column(nx * ny, -1);

  for (std::size_t j = 0; j < ny; ++j) {
    const float y = (static_cast<float>(j) + 0.5f) * hy;
    for (std::size_t i = 0; i < nx; ++i) {
      const float x = (static_cast<float>(i) + 0.5f) * hx;
      const std::size_t id = grid.idx(i, j);
      if (is_obstacle(x, y)) {
        grid.values[id] = Complex32::zero();
        continue;
      }
      column[id] = static_cast<int>(water.size());
      water.push_back(WaterCell{i, j, id});
    }
  }

  const std::size_t n = water.size();
  if (n == 0) return;

  std::vector<double> a(n * n, 0.0);
  std::vector<double> re(n, 0.0);
  std::vector<double> im(n, 0.0);

  for (std::size_t row = 0; row < n; ++row) {
    const std::size_t i = water[row].i;
    const std::size_t j = water[row].j;
    const float x = (static_cast<float>(i) + 0.5f) * hx;
    const float y = (static_cast<float>(j) + 0.5f) * hy;
    double diag = -static_cast<double>(k2);

    auto couple_fluid = [&](std::size_t ni, std::size_t nj, double inv) {
      const int col = column[grid.idx(ni, nj)];
      diag += inv;
      a[row * n + static_cast<std::size_t>(col)] -= inv;
    };

    // Left (-x). Obstacle face: zero flux, no coefficient.
    if (i == 0) {
      diag += 2.0 * inv_hx2;
      re[row] += 2.0 * dirichlet_on_boundary(0.f, y, w) * inv_hx2;
    } else if (!is_obstacle(x - hx, y)) {
      couple_fluid(i - 1, j, inv_hx2);
    }

    // Right (+x)
    if (i + 1 == nx) {
      diag += 2.0 * inv_hx2;
      re[row] += 2.0 * dirichlet_on_boundary(w, y, w) * inv_hx2;
    } else if (!is_obstacle(x + hx, y)) {
      couple_fluid(i + 1, j, inv_hx2);
    }

    // Bottom (-y)
    if (j == 0) {
      diag += 2.0 * inv_hy2;
      re[row] += 2.0 * dirichlet_on_boundary(x, 0.f, w) * inv_hy2;
    } else if (!is_obstacle(x, y - hy)) {
      couple_fluid(i, j - 1, inv_hy2);
    }

    // Top (+y)
    if (j + 1 == ny) {
      diag += 2.0 * inv_hy2;
      re[row] += 2.0 * dirichlet_on_boundary(x, h, w) * inv_hy2;
    } else if (!is_obstacle(x, y + hy)) {
      couple_fluid(i, j + 1, inv_hy2);
    }

    a[row * n + row] += diag;
  }

  solve_dense(a, n, re, im);

  for (std::size_t row = 0; row < n; ++row) {
    grid.values[water[row].grid_index] =
        Complex32{static_cast<float>(re[row]), static_cast<float>(im[row])};
  }
}

}  // namespace hwods
