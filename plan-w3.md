# Week 3 plan — CPU core and GPU core

Mid-checkpoint from the proposal and `LLM_CONTEXT.md`.

**Must ship**

1. GPU Helmholtz Walk on Stars on obstacle-touching tiles matches the CPU walker on one scene that contains a circle and a box together.
2. First hybrid: Pass B solves `I − Q`, and the window shows wave height around both obstacles.
3. First RenderDoc capture, plus a milliseconds table for Pass A, Pass C, and Pass D (Pass B timed with a fence).

**Owners**

| Person | Core |
|---|---|
| ณัฐนันท์ จึงธนศิริกูล | CPU walker, histogram reference, `I − Q`, evaluation script |
| รวินท์ โกศล | GPU Pass A / C / D, RenderDoc, timestamps |

Week 4 stays out of this week: shared empty-tile operator, three-or-more obstacles, non-convex shapes, Dirichlet-only Pass B updates, local re-estimate, and the walk-count / tile-resolution ablations.

---

## Where week 2 stopped

CPU already has the full-domain estimator (`estimate_wos_at_point`, `estimate_wos_field`), `1/J₀` via `helmholtz_step_weight`, FDM, Laplace ≤ 1%, and `./build/helmholtz-check` on `make_example_scene`. Tiles already mark obstacle-touching co-edges with `histogram_slot` and leave empty co-edges as `SharedEmpty` with no slot.

GPU already uploads the SDF SSBO and a histogram SSBO (`VulkanPassA`, `shaders/pass_a.comp`) and times submits (`GpuSubmitTimer`). `pass_a.comp` only writes bin 0 as `(domain distance at the tank center, node count)`. `pass_c.comp` stores zeros. `pass_d.frag` already has `height = Re(u · e^{-iφ})` but the window still presents `fill.comp`. There is no co-edge buffer, no Pass B, and no `I − Q` solver.

`make_example_scene` is a CSG stack (rounded box, difference, intersection). It is not the week 3 gate scene.

---

## Freeze on day 1 (shared contract)

Both cores implement this layout. Change it only together.

**Scene.** New `circle_box_scene`: unit tank, one circle and one axis-aligned box, union, both convex, zero-Neumann. Outer Dirichlet stays `u = 1 − x` on the tank wall (`dirichlet_on_boundary`). Do not use `make_example_scene` for the agreement run.

**Grid.** `4×4` tiles. Pass A walks only co-edges with `histogram_slot` set (obstacle-touching). Empty co-edges stay unestimated. Week 4 replaces the empty-tile fallback below with one shared operator.

**Walk.** Same step as `walk_step`: domain distance, LCG direction (`seed = seed * 1664525 + 1013904223`, 24-bit fraction, full turn), Neumann reflect by pushing `2 * eps` along the gradient, stop on Dirichlet. Star radius is `min(domain distance, distance to the source tile boundary, radius_cap)`. A step that lands on another co-edge is an exit, not a reflection.

**Weight.** One `J₀` implementation in C++ and the same polynomial in GLSL. CPU `::j0` and a hand-rolled shader series will not match bit for bit; the agreement build must call the shared series on both sides. `helmholtz_step_weight` stays `1/J₀(kr)`, and a step with `|J₀| < 1e-6` is a failed walk (same as today’s `nullopt`). Keep `k = 2`, `radius_cap = 0.15` so `kr` stays below the first `J₀` zero (`≈ 2.4048`). `eps = 1e-4`, `max_steps = 64`, `walks_per_bucket = 64` for the first picture (256 only after the bins match).

**Histogram SSBO** (`vec2`, real then imag). For each source slot `s`:

- `Q[s, t]` — mean total weight of walks that exit through destination slot `t`
- `b[s]` — mean of `total_weight * dirichlet(exit)` for walks that hit the tank wall

Dirichlet is real this week, so imaginary parts stay at 0. Still store `vec2`. Index order is `histogram_slot` from `TileGrid::build` (vertical co-edges, then horizontal).

**Pass B.** CPU only. Solve `(I − Q) u = b` for complex interface values. Upload `u` as the `InterfaceValues` SSBO Pass C already declares.

**Pass C.** Per pixel, short Helmholtz walk inside that pixel’s tile until a co-edge or the tank wall. Co-edge exit: `weight * u[slot]`. Dirichlet exit: `weight * (1 − x)`. Write `RG32F`.

**Empty tiles (week 3 only).** A pixel whose tile does not touch an obstacle walks straight to the tank Dirichlet with the same weights. Label this fallback in the check output. Week 4 deletes it.

**Pass D.** Read the field image. Phase comes from frame time. Do not dispatch Pass A or Pass B again to animate. Vsync off.

**Seeds.** One `uint32` seed per bucket, derived as `seed0 + slot` (and `+ walk_index` inside the bucket) with the LCG above. CPU and GPU start from the same `seed0`.

---

## CPU core

1. **Shared `J₀`.** Put the series in `weight.cpp` and a comment block the shader copies. Unit-test it against `::j0` on `[0, 2.4]`. Keep `::j0` only as that oracle.
2. **Co-edge walker.** New function next to `walk_until_boundary`: start at uniform samples on co-edge `[a, b]`, radius capped by the source tile, exit on another slotted co-edge or on Dirichlet. Reuse `walk_step` semantics (reflect, weight, failed `J₀`).
3. **Histogram fill.** CPU Pass A: for every slotted co-edge, accumulate `Q` and `b` with `walks_per_bucket` walks. This is the reference the GPU readback must match.
4. **Pass B.** Assemble dense `I − Q` (slot count is small at `4×4`) and solve the complex system. Add a tiny test: empty-tank, `k = 0`, weight `1`, solution consistent with `u = 1 − x` on a single interior sample. If the system is singular, shrink `k` or the tile size before touching the solver.
5. **CPU Pass C.** Same short-walk rule as the shader, including the empty-tile fallback. Returns an `RG` field on a coarse grid (32×32 is enough for the check).
6. **Agreement script** `scripts/hybrid_check.cpp` (or extend `helmholtz_check.cpp` with a flag). Prints:
   - relative RMSE of CPU histogram vs GPU histogram readback
   - relative RMSE of CPU Pass C vs GPU Pass C on water cells (outside both obstacles)
   - relative RMSE of CPU hybrid vs FDM on that circle+box scene (record only; the 1.5× pure-WoS goal is week 4)
   - mean `|∂n u| / range(u)` on the circle and the box, same finite-difference idea as `helmholtz_check.cpp`
   - Pass A / B / C / D milliseconds once the GPU path prints them

Exit: `ctest` still passes, and the script runs headless against `VulkanPassA` once the shader writes real bins.

## GPU core

1. **Co-edge SSBO.** Upload `TileGrid` co-edges the walker needs: segment `a,b`, source tile id, `histogram_slot`, kind. Frame UBO gains `k`, `eps`, `max_steps`, `walks_per_bucket`, `seed`, tile counts. SDF node packing in `pass_a.comp` already matches the host; keep it.
2. **`pass_a.comp` walker.** Replace the bin-0 stub. One workgroup item = one walk of one bucket. SDF, LCG, reflect, and `1/J₀` match the CPU function names’ behavior. Atomic-add the complex weight into `Q` or `b`. Dispatch only slotted co-edges.
3. **Readback check.** Extend `VulkanPassA::dispatch` (or a sibling) to run the circle+box scene and compare bins to the CPU histogram. Fail the process if relative RMSE is above the gate in the next section.
4. **Pass B glue.** After readback, call the CPU solver, upload `InterfaceValues`. Time this stretch with the existing fence helper and print `pass-b` milliseconds. No GPU linear solver.
5. **`pass_c.comp`.** Short walks, write `RG32F`. Empty-tile fallback as frozen above. Bind the interface SSBO that is already declared.
6. **Present Pass D.** Swap the window path off `fill.comp` onto Pass C’s image + `pass_d.frag`. Vsync off. Phase uniform updates every frame; Pass A and Pass B stay idle while the height oscillates. Real Dirichlet ⇒ standing wave (imaginary channel ≈ 0). That is the expected picture.
7. **Timestamps.** `GpuSubmitTimer` already covers fill and Pass A. Add Pass C and Pass D. If MoltenVK rejects timestamp queries, keep the fence fallback and label the log `cpu-queue` so the report does not call it a GPU timestamp.
8. **RenderDoc.** One capture that contains the Pass A dispatch, the Pass C dispatch, and the Pass D draw. Save the capture path in the ms note below. Do not invent the numbers.

Exit: `./build/helmholtz-wods` opens a window whose height moves around the circle and the box without re-solving, and `--pass-a` (or the new check) prints bin error against the CPU.

---

## Joint gate (end of week)

Run on the M5, Release build, MoltenVK ICD set:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/laplace-check
./build/helmholtz-check          # week 2 scene, must still run
./build/hybrid-check             # new circle+box CPU vs GPU
VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json ./build/helmholtz-wods
```

| Check | Gate |
|---|---|
| Histogram, GPU vs CPU, circle+box, same seed | relative RMSE ≤ `1e-2` |
| Pass C field, GPU vs CPU, water cells | relative RMSE ≤ `1e-2` |
| Window | height visible around both obstacles; phase animates with Pass A/B idle |
| Laplace week-2 gate | still ≤ 1% of `u = 1 − x` |
| RenderDoc | capture of Pass A, Pass C, Pass D |
| ms | table filled from that run |

Milliseconds (write the measured values; leave blank until the run):

| Pass | How | ms |
|---|---|---|
| A | timestamp, else fence | |
| B | fence around the CPU solve | |
| C | timestamp, else fence | |
| D | timestamp, else fence | |

RenderDoc capture file: ``

If the histogram gate fails, fix the walker (seed, radius cap, exit test, or `J₀`) before spending time on Pass D shading.

---

## Suggested order

1. Freeze the contract above and add `circle_box_scene` plus the shared `J₀` test.
2. CPU histogram + Pass B unit test (empty tank, weight 1).
3. GPU Pass A until bin RMSE clears the gate. This is the blocking checkpoint.
4. CPU Pass C reference and GPU Pass C against it.
5. Wire Pass D, vsync off, phase only.
6. One RenderDoc capture and the ms table.

CPU can stub Pass C on analytic Dirichlet (`k = 0`, no obstacles) while GPU Pass A is in progress. GPU can upload a CPU-solved interface buffer before Pass B is tidy, then switch to the real `I − Q` solution without changing the shader.

---

## Plan B (only if the gate blocks)

- `|J₀|` blows up or walks die: shrink `radius_cap` or `k`, re-measure, do not drop Helmholtz weights and ship a Laplace picture unless both are still unsafe.
- Histogram too big: it should not be — only obstacle co-edges at `4×4`. Do not allocate slots for `SharedEmpty`.
- Storage image fails on MoltenVK: field in an SSBO, Pass D reads it as a texel buffer. Pass A is already an SSBO.
- Timestamps rejected: fence times, marked as CPU-queue time.
- `I − Q` singular: drop to `k = 0` on the empty tank first and confirm `u = 1 − x`, then re-enable the circle and the box.
