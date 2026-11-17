#include "hwods/algo/weight.hpp"

#include <cmath>

namespace hwods {

// float bessel_j0(float x) {
//   // macOS/libm exposes j0(double); cast to float for the project API.
//   return static_cast<float>(::j0(static_cast<double>(x)));
// }

float bessel_j0(float x) {
  const float y2 = (x * 0.5f) * (x * 0.5f);
  float term = 1.f;
  float sum = 1.f;
  for (int m = 1; m <= 6; ++m) {
    term *= -y2 / static_cast<float>(m * m);
    sum += term;
  }
  return sum;
}

std::optional<float> helmholtz_step_weight(float k, float radius) {
  const float j = bessel_j0(k * radius);
  if (std::fabs(j) < 1e-6f) return std::nullopt;
  return 1.f / j;
}

float laplace_step_weight() { return 1.f; }

float accumulate_weights(std::span<const float> weights) {
  float p = 1.f;
  for (float w : weights) p *= w;
  return p;
}

}  // namespace hwods
