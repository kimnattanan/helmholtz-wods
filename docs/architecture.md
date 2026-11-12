# Architecture (C++ skeleton)

## Data flow

1. **Scene** — tank size + SDF node list (circle / box / rounded box + CSG).
2. **TileGrid** — regular `nx×ny`; obstacle-touching co-edges own histogram slots.
3. **Pass A** — GPU WoS × `1/J₀(kr)`; Neumann reflects via closest SDF point.
4. **Pass B** — CPU builds/solves `I − Q`.
5. **Pass C** — short walks → `RG32F` field.
6. **Pass D** — phase × height (no A/B re-run).

## Week 1 GPU path

Compute shader writes storage image → blit → swapchain present (GLFW + Vulkan).

## Plan B

- Storage image issues → SSBO + fragment
- No timestamps → fence around submit
- Histogram blow-up → obstacle co-edges only; shrink `k`
