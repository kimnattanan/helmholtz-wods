//! Week-1 / Week-2 Laplace checkpoint on an empty tank.
//!
//! Analytical u(x,y)=1−x must match Dirichlet data on all four walls.
//! Run: `cargo run --release --bin laplace-check`

use helmholtz_wods::algo::sdf::SceneSdf;
use helmholtz_wods::algo::tiles::TileGrid;
use helmholtz_wods::algo::walk_on_stars::{WalkConfig, walk_until_boundary};
use helmholtz_wods::eval::laplace::{analytical_laplace_empty_tank, dirichlet_on_boundary};
use helmholtz_wods::eval::rmse::relative_rmse;

fn main() {
    let tank = [1.0f32, 1.0];
    let nx = 32usize;
    let ny = 32usize;

    // 1) Boundary data ↔ analytical
    let mut max_bdry_err = 0.0f32;
    for &y in &[0.0, 1.0] {
        for i in 0..=64 {
            let x = i as f32 / 64.0;
            let e = (dirichlet_on_boundary(x, y, tank[0])
                - analytical_laplace_empty_tank(x, y, tank[0]))
            .abs();
            max_bdry_err = max_bdry_err.max(e);
        }
    }
    for &x in &[0.0f32, 1.0] {
        for j in 0..=64 {
            let y = j as f32 / 64.0;
            let e = (dirichlet_on_boundary(x, y, tank[0])
                - analytical_laplace_empty_tank(x, y, tank[0]))
            .abs();
            max_bdry_err = max_bdry_err.max(e);
        }
    }

    // 2) Naive Monte Carlo WoS estimate at interior samples (Laplace weights = 1)
    let scene = SceneSdf::empty_tank(tank[0], tank[1]);
    let grid = TileGrid::build(tank, 8, 8, |_, _| false);
    let cfg = WalkConfig {
        max_steps: 48,
        radius_cap: grid.tiles[0].half_diagonal(),
        helmholtz: false,
        ..WalkConfig::default()
    };

    let mut approx = Vec::new();
    let mut reference = Vec::new();
    let walks_per_point = 64u32;
    let mut seed = 42u32;

    for j in 0..ny {
        for i in 0..nx {
            let x = (i as f32 + 0.5) / nx as f32 * tank[0];
            let y = (j as f32 + 0.5) / ny as f32 * tank[1];
            let mut acc = 0.0f32;
            for _ in 0..walks_per_point {
                let walk = walk_until_boundary(&scene, [x, y], &cfg, &mut seed);
                let u_b = dirichlet_on_boundary(walk.endpoint[0], walk.endpoint[1], tank[0]);
                acc += walk.total_weight * u_b;
            }
            approx.push(acc / walks_per_point as f32);
            reference.push(analytical_laplace_empty_tank(x, y, tank[0]));
        }
    }

    let rmse = relative_rmse(&approx, &reference);
    println!("helmholtz-wods Laplace empty-tank check");
    println!("  max |BC − analytical|     = {max_bdry_err:.3e}");
    println!("  relative RMSE (WoS vs u*) = {rmse:.4}  ({walks_per_point} walks/pt, {nx}×{ny})");
    println!(
        "  tiles {}×{}  shared-empty co-edges {}",
        grid.nx,
        grid.ny,
        grid.co_edges
            .iter()
            .filter(|e| e.histogram_slot.is_none())
            .count()
    );

    // Week-2 gate is ≤ 1%; Week-1 just needs the script + analytical identity.
    if max_bdry_err > 1e-5 {
        eprintln!("FAIL: boundary does not match analytical");
        std::process::exit(1);
    }
    println!("OK: analytical Laplace identity holds; WoS estimate printed for W2 tuning");
}
