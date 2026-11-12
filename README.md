# helmholtz-wods (C++)

**Agents / LLMs:** read [`LLM_CONTEXT.md`](./LLM_CONTEXT.md) before any change.

**2D Helmholtz wave height via Walk on Decomposed Subdomains (WoDS)** — hybrid RMSE ≥ **1.5×** better than pure Walk on Stars at equal walk budget.

Paper: Jambon et al., SIGGRAPH 2026 — [doi:10.1145/3811340](https://doi.org/10.1145/3811340)

## Stack

- C++20, CMake ≥ 3.20
- GLFW (window / surface)
- Vulkan 1.2+ via MoltenVK
- GLSL → SPIR-V (`glslangValidator`)

## Build & run

```bash
brew install molten-vk vulkan-loader vulkan-headers glslang glfw

cd term/projects/helmholtz-wods
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(brew --prefix)"
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/laplace-check
VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json ./build/helmholtz-wods
```

## Layout

```
include/hwods/algo|eval|gpu   public headers
src/algo src/eval src/gpu     implementations
shaders/                      fill + Pass A/C/D stubs
scripts/laplace_check.cpp
tests/algo_tests.cpp
```

## Week 1

- `./build/helmholtz-wods` — compute fill window
- `./build/laplace-check` — analytical Laplace `u=1−x` + WoS estimate
- `ctest` — SDF / tiles / J₀ / WoS / RMSE unit checks
