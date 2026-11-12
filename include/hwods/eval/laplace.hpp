#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace hwods {

float dirichlet_on_boundary(float x, float y, float tank_width);
float analytical_laplace_empty_tank(float x, float y, float tank_width);
std::vector<float> sample_laplace_grid(std::size_t nx, std::size_t ny,
                                       std::array<float, 2> tank_size);

}  // namespace hwods
