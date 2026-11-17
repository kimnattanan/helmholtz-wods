# AI-use log

| Date | Tool | Prompt summary | Generated | Human edit |
|---|---|---|---|---|
| 2026-11-12 | Cursor Agent | Draft Week-1 C++ skeleton from proposal-v2 | CMake+GLFW+Vulkan compute fill; SDF/tiles/WoS/`1/J₀`; Laplace check + algo tests; shaders; `LLM_CONTEXT.md` | Review numbers vs FDM; keep walker / J₀ / I−Q formulas team-owned |
| 2026-11-16 | Cursor Agent | Host code for `pass_a.comp` bindings and a demo SDF scene | `vulkan_pass_a` uniform + SDF/histogram SSBOs; `--pass-a` smoke vs CPU SDF | Review std430 packing and the demo obstacles |
| 2026-11-16 | Cursor Agent | Per-pass timestamps on fill and Pass A submits, fence if queries are rejected | `GpuSubmitTimer`; ms printed for `fill` and `pass-a` | Review MoltenVK timestamp vs fence numbers |
| 2026-11-16 | Cursor Agent | Task 1 Laplace 1% gate check and IDE C++20 configuration | `scripts/laplace_check.cpp` 1% hard fail; C++20 editor configs (`.vscode`, `.clangd`) | Verified RMSE 0.0064 passes gate; resolved IDE span error |
| 2026-11-16 | Cursor Agent | Task 2 2D Helmholtz FDM solver with zero-Neumann BCs | `src/eval/fdm.cpp`, `include/hwods/eval/fdm.hpp`, unit tests in `tests/algo_tests.cpp` | Verify FDM k=0 matches analytical Laplace <= 1% |
| 2026-11-16 | Cursor Agent | Tasks 3 & 4 CPU WoS Helmholtz wave weights and evaluation check script | `src/algo/walk_on_stars.cpp`, `scripts/helmholtz_check.cpp`, unit tests in `tests/algo_tests.cpp` | Verified circle+box WoS vs FDM RMSE 0.111 and obstacle boundary |dn u| |
| 2026-11-16 | Cursor Agent | FDM reference blew up inside a sealed SDF cavity under Gauss–Seidel | `src/eval/fdm.cpp` direct solve of the same stencil; boundedness test in `tests/algo_tests.cpp` | Review hole cells solve to 0 and the empty-tank Laplace gate still holds |
