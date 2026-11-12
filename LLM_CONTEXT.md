# LLM_CONTEXT — read this before any change

**Mandatory.** Before editing, scaffolding, refactoring, or advising on this repo, read this file end-to-end. Prefer it over chat memory. If a request conflicts with this file, follow this file and say so.

| | |
|---|---|
| Project root | `term/projects/helmholtz-wods/` |
| Course | 2110479 Computer Graphics — Term Project 2026 |
| Proposal | `term/docs/proposal-v3.docx` (source of truth for scope; C++ stack) |
| Title | Helmholtz 2D wave height in a tank via **Walk on Decomposed Subdomains (WoDS)** |
| Goal line | Hybrid RMSE better than pure Walk on Stars by **≥ 1.5×** at equal walk budget |
| Paper | Jambon et al., *Walk on Decomposed Subdomains*, ACM TOG / SIGGRAPH 2026 — doi:10.1145/3811340 |
| Also cite | Walk on Stars 2023 |
| Stack | **C++20** + **GLFW** + **Vulkan** (MoltenVK), GLSL → SPIR-V (`glslangValidator` via CMake) |
| Eval HW | Apple Silicon **M5**, macOS |

---

## 1. What we are building

Solve **2D Helmholtz** in a rectangular tank of width 1 (height as set), with **any number / shape of static obstacles**, **without a volume mesh** and **without baking shape/count into shaders**.

- Water domain via **SDF** (circle / box / rounded box + union / intersection / difference).
- Outer tank: **Dirichlet** — left=1, right=0, top/bottom ramp 1→0 in `x` → analytical Laplace on empty tank is **`u(x,y)=1−x`**.
- Every obstacle: **zero-Neumann** (reflect), convex or not.
- Field `u` is **complex**. Display height by **phase rotation** each frame: `height = Re(u · e^{-iωt})` — **do not re-solve** Pass A/B just to animate.
- Real Dirichlet → standing oscillation. Traveling crest needs **complex** Dirichlet in both solver and FDM reference.

Demo must show: height around **several** obstacle shapes, plus an **error graph** of hybrid vs pure WoS at equal walk budget.

---

## 2. In scope vs cut (do not expand)

**Keep:** subdomain tiling + linear `I − Q`; Walk on Stars on SDF Neumann; empty tiles share one operator.

**Cut:** 3D, moving obstacles every frame, city scenes, non-rect tiles, meshes, nonzero Neumann / Robin / volumetric sources, re-estimating operators every frame for phase.

**Interactive:** change Dirichlet → Pass B only; move/add obstacle → re-estimate only touching co-edges.

---

## 3. Pipeline (names are fixed)

```
SDF → tiles / co-edges → Pass A (WoS × 1/J₀) → hist SSBO
    → Pass B (CPU: assemble & solve I−Q) → interface values
    → Pass C (short walks → complex field RG32F)
    → Pass D (phase × height → swapchain, vsync off)
```

**Plan B:** storage image → SSBO+frag; no timestamps → fence; histogram blow-up → obstacle co-edges only, shrink `k`.

---

## 4. Success criteria

Hybrid vs CPU FDM and vs GPU pure WoS (equal walk budget). Rel. RMSE hybrid vs pure WoS **≥ 1.5× lower**. Week-2 gate: empty-tank Laplace WoS within **1%** of `u=1−x` before Helmholtz weights.

Ablations: walks/bucket {64,256,1024}; tile res {4,8,16} at fixed `k`.

All report numbers from **team runs**.

---

## 5. Weekly plan

| Week | Must ship |
|---|---|
| **1** | Skeleton: `./helmholtz-wods` fills screen via **compute**; SDF; WoS; `1/J₀`; tiles; `laplace-check` |
| **2** | SDF SSBO; histograms; timestamps; Laplace ≤1%; FDM; CPU full WoS; RMSE + ∂n |
| **3** | GPU Helmholtz WoS = CPU on circle+box; first hybrid; height; RenderDoc + ms |
| **4** | Shared empty op; ≥3 obstacles / ≥2 kinds / ≥1 non-convex; Pass D phase; local re-estimate; ablations |
| **5** | README clone→build→run; `v1.0`; video; report; Pages; 10-min demo |

**Hard gate:** end of W1 = window that dispatches compute and fills the screen.

---

## 6. Repo map

```
LLM_CONTEXT.md
AI-use-log.md
README.md
CMakeLists.txt              ← GLFW + Vulkan; compiles GLSL→SPIR-V
include/hwods/algo/         ← SDF, tiles, WoS, J₀
include/hwods/eval/         ← Laplace, FDM stub, RMSE
include/hwods/gpu/          ← Vulkan fill API
src/algo/ src/eval/ src/gpu/
src/main.cpp                ← W1 compute fill
scripts/laplace_check.cpp
tests/algo_tests.cpp
shaders/                    ← fill.comp, pass_{a,c}.comp, pass_d.{vert,frag}
```

**Commands**

```bash
cd term/projects/helmholtz-wods
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/laplace-check
./build/helmholtz-wods
```

```bash
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
```

Instance: **portability enumeration**. Device: **portability_subset**.

---

## 7. Team roles

| Person | ID | Own |
|---|---|---|
| รวินท์ โกศล | 6832191421 | GPU core, RenderDoc, W5 video |
| ณัฐนันท์ จึงธนศิริกูล | 6832067321 | Algorithm, CPU reference, evaluation, report, GitHub Pages |

---

## 8. AI-use & ethics

- AI may help: Vulkan/GLFW/CMake boilerplate, draft shaders, report prose.
- Team must explain: WoS, `1/J₀(kr)`, `I−Q`, RMSE.
- Log every AI session in `AI-use-log.md`.
- Hand-authored SDFs only. Cite WoDS 2026 + Walk on Stars 2023. Keep GLFW/Vulkan licenses.

---

## 9. Coding rules for agents

1. Read this file first.
2. Match Pass A/B/C/D vocabulary.
3. Smallest diff; no engine swap (no MoltenVK→Metal rewrite, no wgpu) unless asked.
4. Shaders stay GLSL; CMake compiles them.
5. Keep CPU algo dual with shaders (same weight, SDF query, and BC semantics).
6. Laplace before Helmholtz.
7. Never fabricate timing/RMSE.
8. Prefer Plan B on MoltenVK failures.
9. After edits: `ctest` at minimum; run `./build/helmholtz-wods` when touching GPU/shaders.
10. Refuse out-of-scope features.

---

## 10. Risks → Plan B

Same as proposal: shrink tiles/`k` for J₀ blow-up; SDF shrink+resample for non-convex; empty-tank weight=1 before Neumann/`k`; SSBO+fence on MoltenVK issues; W1 fill-screen is the Vulkan/GLFW/CMake setup firewall.

---

## 11. Self-check

- [ ] Still 2D Helmholtz + WoDS hybrid?
- [ ] Pass names unchanged?
- [ ] `AI-use-log.md` updated?
- [ ] Build/test commands mentioned?
