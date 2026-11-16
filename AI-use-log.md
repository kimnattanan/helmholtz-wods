# AI-use log

| Date | Tool | Prompt summary | Generated | Human edit |
|---|---|---|---|---|
| 2026-11-12 | Cursor Agent | Draft Week-1 C++ skeleton from proposal-v2 | CMake+GLFW+Vulkan compute fill; SDF/tiles/WoS/`1/J₀`; Laplace check + algo tests; shaders; `LLM_CONTEXT.md` | Review numbers vs FDM; keep walker / J₀ / I−Q formulas team-owned |
| 2026-11-16 | Cursor Agent | Host code for `pass_a.comp` bindings and a demo SDF scene | `vulkan_pass_a` uniform + SDF/histogram SSBOs; `--pass-a` smoke vs CPU SDF | Review std430 packing and the demo obstacles |
| 2026-11-16 | Cursor Agent | Per-pass timestamps on fill and Pass A submits, fence if queries are rejected | `GpuSubmitTimer`; ms printed for `fill` and `pass-a` | Review MoltenVK timestamp vs fence numbers |
