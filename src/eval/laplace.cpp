#include "hwods/eval/laplace.hpp"

namespace hwods {

float dirichlet_on_boundary(float x, float /*y*/, float tank_width) {
  return 1.f - (x / tank_width);
}

float analytical_laplace_empty_tank(float x, float /*y*/, float tank_width) {
  return 1.f - (x / tank_width);
}

std::vector<float> sample_laplace_grid(std::size_t nx, std::size_t ny,
                                       std::array<float, 2> tank_size) {
  const float w = tank_size[0];
  const float h = tank_size[1];
  std::vector<float> out;
  out.reserve(nx * ny);
  for (std::size_t j = 0; j < ny; ++j) {
    const float y = (static_cast<float>(j) + 0.5f) / static_cast<float>(ny) * h;
    for (std::size_t i = 0; i < nx; ++i) {
      const float x = (static_cast<float>(i) + 0.5f) / static_cast<float>(nx) * w;
      out.push_back(analytical_laplace_empty_tank(x, y, w));
    }
  }
  return out;
}

}  // namespace hwods
