# LLM_CONTEXT — read this before any change

**Mandatory.** Before editing, scaffolding, refactoring, or advising on this repo, read this file end-to-end. Prefer it over chat memory. If a request conflicts with this file, follow this file and say so.

| | |
|---|---|
| Project root | `term/projects/helmholtz-wods/` |
| Course | 2110479 Computer Graphics — Term Project 2026 |
| Proposal | `term/docs/proposal-v2.docx` (source of truth for scope) |
| Title | Helmholtz 2D wave height in a tank via **Walk on Decomposed Subdomains (WoDS)** |
| Goal line | Hybrid RMSE better than pure Walk on Stars by **≥ 1.5×** at equal walk budget |
| Paper | Jambon et al., *Walk on Decomposed Subdomains*, ACM TOG / SIGGRAPH 2026 — doi:10.1145/3811340 |
| Also cite | Walk on Stars 2023 |
| Stack | Rust + `ash` + `winit`, GLSL → SPIR-V (`glslc` / `glslangValidator`), Vulkan 1.3 via **MoltenVK** |
| Eval HW | Apple Silicon **M5**, macOS |

---

## 1. What we are building

Solve **2D Helmholtz** in a rectangular tank of width 1 (height as set), with **any number / shape of static obstacles**, **without a volume mesh** and **without baking shape/count into shaders**.

- Water domain via **SDF** (circle / box / rounded box + union / intersection / difference; non-closed shapes may use an SDF texture later).
- Outer tank: **Dirichlet** — left=1, right=0, top/bottom ramp 1→0 in `x` → analytical Laplace on empty tank is **`u(x,y)=1−x`**.
- Every obstacle: **zero-Neumann** (reflect), convex or not.
- Field `u` is **complex**. Display height at time `t` by **phase rotation** each frame: `height = Re(u · e^{-iωt})` — **do not re-solve** Pass A/B just to animate.
- Real Dirichlet → standing oscillation. Traveling crest needs **complex** Dirichlet in both solver and FDM reference.

Demo must show: height rising/falling around **several** obstacle shapes in one scene, plus an **error graph** of hybrid vs pure WoS at equal walk budget.

---

## 2. In scope vs cut (do not expand)

**Keep from WoDS 2026**

- Subdomain tiling + assemble with a **linear** interface solve (`I − Q`)
- Walk on Stars on zero-Neumann obstacles via SDF
- Empty tiles share one operator (walk cost grows with obstacles, not tank size)

**Cut — refuse feature creep**

- 3D, moving obstacles every frame, city-scale scenes, microstructure, path planning
- Non-rectangular tiles, triangle meshes
- Nonzero Neumann, Robin, volumetric sources
- Re-estimating the operator every frame for phase animation (phase is Pass D only)

**Interactive allowances (proposal demo)**

- Change Dirichlet → re-run **Pass B only**
- Move/add obstacle → re-estimate **only co-edges that touch** the changed shape

---

## 3. Pipeline (names are fixed)

```
SDF → tiles / co-edges → Pass A (WoS × 1/J₀) → hist SSBO
    → Pass B (CPU: assemble & solve I−Q) → interface values
    → Pass C (short walks → complex field RG32F)
    → Pass D (phase × height → swapchain, vsync off)
```

| Pass | Where | Job |
|---|---|---|
| A | compute | WoS on obstacle-touching co-edges (+ one shared empty op); weight each step by `1/J₀(kr)`; reflect on Neumann |
| B | CPU | Read histogram, build/solve complex interface system |
| C | compute | Per-pixel short walk to tile boundary; read solved interface |
| D | graphics | Phase rotate + draw height |

**GPU resources**

- UBO: tank size, `k`, `ω`, SDF counts, walk caps, seed, outer Dirichlet
- SSBO: SDF nodes, co-edge histograms, interface values; field-in-SSBO is Plan B
- Images: complex field `RG32F`; present to swapchain
- Host-visible buffer for FDM reference when computing RMSE
- Empty (non-obstacle) co-edges do **not** own private histograms

**Plan B**

- Storage image broken on MoltenVK → SSBO + fragment read
- No timestamp queries → fence around submit; report as CPU queue time
- Histogram too large → obstacle-touching co-edges only; shrink `k` before cutting walks

---

## 4. Success criteria (measure these; do not invent others)

Compare hybrid vs **CPU FDM** and vs **GPU pure WoS** (same walk budget, no tiling / no `I−Q`).

| Metric | Minimum | Stretch |
|---|---|---|
| Rel. RMSE hybrid vs FDM @ 256 / 1024 walks per bucket | meet proposal table | stretch column |
| Rel. RMSE hybrid vs pure WoS @ equal steps | **≥ 1.5× lower** | higher |
| Mean steps to tile edge vs full-domain walk | shorter by proposal factor | stretch |
| Mean \|∂n u\| on all obstacle boundaries / range(u) | near 0 (Neumann) | tighter |
| Pass C @ 1080p, 4 walks/px | ms budget in proposal | stretch |
| Pass A (unique co-edges) on M5 | ms budget in proposal | stretch |
| Pass D @ 1080p | ms budget in proposal | stretch |

**Ablations (required ≥ 2 knobs)**

1. Walks per bucket: 64, 256, 1024 → RMSE + Pass A time  
2. Tile resolution at fixed `k`: 4, 8, 16 → step length, max \|weight\|, system size, RMSE  

**Baselines**

- CPU FDM Helmholtz (RMSE + normal derivative reference)
- GPU pure WoS with Helmholtz weights, no tiles
- Week-2 gate: empty-tank **Laplace** WoS within **1%** of `u=1−x` before enabling wave weights

All report numbers must come from **team runs**, not AI guesses.

---

## 5. Weekly plan (do not skip gates)

| Week | Must ship |
|---|---|
| **1** | Proposal + skeleton: `cargo run --release` fills screen via **compute**; SDF list; WoS step; `1/J₀`; tiles/co-edges; Laplace analytical script |
| **2** | SDF SSBO multi-shape; histograms; timestamps; Laplace ≤1%; FDM Helmholtz; CPU full-domain WoS; RMSE + ∂n scripts |
| **3 mid** | GPU Helmholtz WoS matches CPU on circle+box scene; first hybrid (`I−Q` solves); height image; first RenderDoc + ms table |
| **4** | Shared empty operator; ≥3 obstacles, ≥2 kinds, ≥1 non-convex; Pass D phase; Dirichlet→B only; move obstacle→local re-estimate; ablations + before/after bottleneck numbers |
| **5** | README clone→`cargo run --release`; tag `v1.0`; 60–90s video (≥30s phase height); 4–6p report; GitHub Pages; 10-min demo (team can explain walker, `J₀`, linear system, number provenance) |

**Hard risk gate:** end of W1 must have a window that dispatches compute and fills the screen — do not burn W2 on ash setup.

---

## 6. Repo map (where to edit)

```
LLM_CONTEXT.md          ← this file (read first)
AI-use-log.md           ← log every AI-assisted change
README.md               ← human runbook
docs/architecture.md    ← short data-flow notes
Cargo.toml / build.rs   ← deps; GLSL→SPIR-V at build
src/main.rs             ← demo entry (W1: fill window)
src/lib.rs
src/algo/               ← SDF, tiles, WoS, J₀ weight (CPU reference shared with GPU intent)
src/eval/               ← Laplace analytical, FDM stub, RMSE
src/gpu/                ← ash/winit Vulkan path
shaders/                ← fill.comp, pass_a.comp, pass_c.comp, pass_d.{vert,frag}
scripts/laplace_check.rs
```

**Commands**

```bash
cd term/projects/helmholtz-wods
cargo test
cargo run --release --bin laplace-check
cargo run --release
```

MoltenVK: app probes `/opt/homebrew/lib/libvulkan.dylib`. If needed:

```bash
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
export VULKAN_LIB=/opt/homebrew/lib/libvulkan.dylib
```

Instance must enable **portability enumeration**; device needs **portability_subset** on macOS.

---

## 7. Team roles

| Person | ID | Own |
|---|---|---|
| รวินท์ โกศล | 6832191421 | GPU core, RenderDoc, W5 video |
| ณัฐนันท์ จึงธนศิริกูล | 6832067321 | Algorithm, CPU reference, evaluation, report, GitHub Pages |

Do not rewrite the other person's area without a clear handoff note in the reply.

---

## 8. AI-use & ethics (non-negotiable)

- AI may help: ash/`Cargo.toml`/`build.rs` boilerplate, draft shaders, report prose.
- Team must be able to explain: WoS, `1/J₀(kr)`, `I−Q` assembly, RMSE formula.
- **Every** AI session that changes code or docs: append a row to `AI-use-log.md` (date, tool, prompt summary, generated parts, human edits).
- Assets: **hand-authored SDFs only** — no game/film models.
- Preserve crate licenses (`ash`, `winit`, …). Cite WoDS 2026 + Walk on Stars 2023 in the report.

---

## 9. Coding rules for agents

1. **Read this file first.** Then open only the files needed for the task.
2. **Match proposal vocabulary:** Pass A/B/C/D, co-edge, histogram, `I−Q`, WoS, zero-Neumann, phase.
3. **Prefer smallest diff.** No drive-by refactors, no new frameworks, no vulkano/wgpu swap unless explicitly asked.
4. **Shaders stay GLSL** compiled by `build.rs`; do not commit to a different shading language.
5. **CPU algo and GPU shader math must stay dual** — same weight, same SDF query semantics, same BC.
6. **Laplace before Helmholtz.** Do not enable `1/J₀` paths until empty-tank Laplace gate is green.
7. **Numbers:** never fabricate timing/RMSE; wire measurement hooks or leave `TODO(measure)`.
8. **MoltenVK fragility:** prefer Plan B paths if storage images / timestamps fail; document in README.
9. After substantive edits: `cargo test` at minimum; GPU path `cargo run --release` when touching `src/gpu` or shaders.
10. If asked for something out of scope (3D, meshing, moving obstacles every frame, etc.), refuse and point here.

---

## 10. Risks → default Plan B

| Risk | Response |
|---|---|
| `1/J₀` weight blows up | Shrink tiles / lower `k` until longest-step \|J₀\| factor safe; else ship Laplace + document wave-weight limit |
| Star step tunnels non-convex / wrong body | Shrink step by SDF and resample — do **not** collapse scene to one circle |
| `I−Q` singular / wrong | Empty tank, weight=1, match analytical before Neumann and before `k>0` |
| MoltenVK compute / timestamps | SSBO + fence + RenderDoc |
| Pass A bandwidth | Histogram only obstacle-touching co-edges |
| W2 lost to ash setup | W1 fill-screen is the firewall |

Fallback deliverable if Helmholtz hybrid slips: still ship **tiled Helmholtz height** with **scene-wide SDF WoS** (not a random unrelated demo).

---

## 11. Quick self-check before finishing a task

- [ ] Still 2D Helmholtz + WoDS hybrid as above?
- [ ] Pass names / data flow unchanged?
- [ ] No mesh / no out-of-scope BC?
- [ ] `AI-use-log.md` updated if AI generated or heavily edited code?
- [ ] Tests or run command mentioned for what you changed?
