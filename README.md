# helmholtz-wods

**Agents / LLMs:** read [`LLM_CONTEXT.md`](./LLM_CONTEXT.md) before any change. A Cursor rule (`.cursor/rules/read-llm-context.mdc`) enforces this.

**2D Helmholtz wave height in a tank via Walk on Decomposed Subdomains (WoDS)**  
Relative RMSE of the hybrid solver vs pure Walk on Stars ≥ **1.5× better** at equal walk budget.

Paper: Jambon et al., *Walk on Decomposed Subdomains*, ACM TOG / SIGGRAPH 2026  
([doi:10.1145/3811340](https://doi.org/10.1145/3811340))

## Team

| Name | ID | Role |
|---|---|---|
| รวินท์ โกศล | 6832191421 | GPU core, RenderDoc, Week-5 video |
| ณัฐนันท์ จึงธนศิริกูล | 6832067321 | Algorithm, CPU reference, evaluation, report, GitHub Pages |

## Pipeline

```
SDF → tiles / co-edges → Pass A (WoS × 1/J₀) → Pass B (solve I−Q on CPU)
    → Pass C (assemble complex field) → Pass D (phase × height)
```

| Pass | Where | Job |
|---|---|---|
| A | compute | Short WoS on obstacle-touching co-edges; accumulate exit weights |
| B | CPU | Assemble / solve interface system |
| C | compute | Per-pixel short walk to tile edge; read interface values |
| D | graphics | `height = Re(u e^{-iωt})` to swapchain (vsync off) |

## Week 1 skeleton

- `cargo run --release` — winit window, ash compute dispatch, fill screen
- `cargo run --release --bin laplace-check` — analytical Laplace `u=1−x` + WoS estimate
- `cargo test` — SDF, tiles, J₀ weight, RMSE unit tests

## Requirements (Apple Silicon / MoltenVK)

```bash
brew install molten-vk vulkan-loader vulkan-headers glslang
```

The app probes `/opt/homebrew/lib/libvulkan.dylib` automatically. If needed:

```bash
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
export VULKAN_LIB=/opt/homebrew/lib/libvulkan.dylib
```

`build.rs` compiles GLSL → SPIR-V with `glslc` or `glslangValidator`.

## Layout

```
src/
  algo/     SDF CSG, tiles/co-edges, Walk on Stars, 1/J₀ weight
  eval/     Laplace analytical, FDM stub, RMSE
  gpu/      ash + winit (fill → later Pass A/C/D)
shaders/    fill.comp, pass_a.comp, pass_c.comp, pass_d.{vert,frag}
scripts/    laplace_check.rs
AI-use-log.md
```

## Run

```bash
cd term/projects/helmholtz-wods
cargo test
cargo run --release --bin laplace-check
cargo run --release
```

## Hardware / eval target

Apple Silicon **M5**, macOS, Vulkan 1.3 via MoltenVK. Record OS + SDK versions here before Week 5:

- macOS:
- MoltenVK / Vulkan loader:
- RenderDoc:

## License notes

Keep upstream licenses for `ash`, `winit`, and any allocator helpers. Cite WoDS 2026 and Walk on Stars 2023 in the report. Scene SDFs are hand-authored (no game/film assets).
