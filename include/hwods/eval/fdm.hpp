#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace hwods {

struct Complex32 {
  float re = 0.f;
  float im = 0.f;
  static constexpr Complex32 zero() { return {}; }
  float norm_sq() const { return re * re + im * im; }
  Complex32 operator+(Complex32 o) const { return {re + o.re, im + o.im}; }
  Complex32 operator-(Complex32 o) const { return {re - o.re, im - o.im}; }
  Complex32 operator*(float s) const { return {re * s, im * s}; }
  Complex32 operator/(float s) const { return {re / s, im / s}; }
};

struct SceneSdf;

struct FdmGrid {
  std::size_t nx = 0;
  std::size_t ny = 0;
  std::array<float, 2> tank_size{1.f, 1.f};
  float k = 0.f;
  std::vector<Complex32> values;

  static FdmGrid zeros(std::size_t nx, std::size_t ny, std::array<float, 2> tank_size,
                       float k);
  std::size_t idx(std::size_t i, std::size_t j) const { return j * nx + i; }
};

/// Solve complex (Δ + k²)u = 0 on rectangular tank with Dirichlet outer walls
/// and optional zero-Neumann obstacles.
void solve_helmholtz_fdm(FdmGrid& grid, const SceneSdf* scene = nullptr);
void solve_helmholtz_fdm_stub(FdmGrid& grid);

}  // namespace hwods
