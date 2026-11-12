#include "hwods/eval/fdm.hpp"

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

void solve_helmholtz_fdm_stub(FdmGrid& grid) {
  const float w = grid.tank_size[0];
  for (std::size_t j = 0; j < grid.ny; ++j) {
    for (std::size_t i = 0; i < grid.nx; ++i) {
      const float x = (static_cast<float>(i) + 0.5f) / static_cast<float>(grid.nx) * w;
      grid.values[grid.idx(i, j)] = Complex32{1.f - x / w, 0.f};
    }
  }
}

}  // namespace hwods
