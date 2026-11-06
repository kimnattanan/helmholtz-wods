# Architecture (skeleton)

## Data flow

1. **Scene** — tank size + SDF node list (circle / box / rounded box + CSG ops).
2. **TileGrid** — regular `nx×ny` rectangles; co-edges that touch obstacles own histogram slots; empty tiles share one operator.
3. **Pass A** — GPU WoS on co-edges; each step multiplies `1/J₀(kr)`; Neumann hits reflect via closest SDF point.
4. **Pass B** — CPU reads histogram, builds `I − Q`, solves complex interface values.
5. **Pass C** — GPU short walks to tile boundary; writes `RG32F` field.
6. **Pass D** — rotate phase each frame without re-running A/B; draw height.

## GPU resources (proposal §5)

- **UBO** — tank size, `k`, `ω`, SDF node count, walk caps, seed, outer Dirichlet
- **SSBO** — SDF nodes, co-edge histograms, interface values, optional field fallback
- **Images** — complex field `RG32F`; swapchain present

## Plan B

- Storage image issues on MoltenVK → field in SSBO, fragment samples as texture
- Timestamp queries unavailable → fence around submit; report as CPU queue time
- Histogram blow-up → only obstacle-touching co-edges; shrink `k` before cutting walks
