#include "hwods/eval/rmse.hpp"

#include <cassert>
#include <cmath>

namespace hwods {

float relative_rmse(std::span<const float> approx, std::span<const float> reference) {
  assert(approx.size() == reference.size());
  assert(!reference.empty());
  float err = 0.f;
  float ref_norm = 0.f;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const float d = approx[i] - reference[i];
    err += d * d;
    ref_norm += reference[i] * reference[i];
  }
  if (ref_norm < 1e-20f) return std::sqrt(err);
  return std::sqrt(err / ref_norm);
}

float relative_rmse_complex(std::span<const Complex32> approx,
                            std::span<const Complex32> reference) {
  assert(approx.size() == reference.size());
  assert(!reference.empty());
  float err = 0.f;
  float ref_norm = 0.f;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    err += (approx[i] - reference[i]).norm_sq();
    ref_norm += reference[i].norm_sq();
  }
  if (ref_norm < 1e-20f) return std::sqrt(err);
  return std::sqrt(err / ref_norm);
}

float mean_normal_derivative_over_range(std::span<const float> normal_derivs,
                                        float field_min, float field_max) {
  assert(!normal_derivs.empty());
  float sum = 0.f;
  for (float v : normal_derivs) sum += std::fabs(v);
  const float mean = sum / static_cast<float>(normal_derivs.size());
  const float range = std::max(std::fabs(field_max - field_min), 1e-12f);
  return mean / range;
}

}  // namespace hwods
