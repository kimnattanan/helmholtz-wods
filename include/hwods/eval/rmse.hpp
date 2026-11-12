#pragma once

#include "hwods/eval/fdm.hpp"

#include <span>

namespace hwods {

float relative_rmse(std::span<const float> approx, std::span<const float> reference);
float relative_rmse_complex(std::span<const Complex32> approx,
                            std::span<const Complex32> reference);
float mean_normal_derivative_over_range(std::span<const float> normal_derivs,
                                        float field_min, float field_max);

}  // namespace hwods
