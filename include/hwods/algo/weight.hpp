#pragma once

#include <optional>
#include <span>

namespace hwods {

float bessel_j0(float x);
std::optional<float> helmholtz_step_weight(float k, float radius);
float laplace_step_weight();
float accumulate_weights(std::span<const float> weights);

}  // namespace hwods
