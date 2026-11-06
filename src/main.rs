//! Helmholtz / WoDS demo entry.
//! Week 1: `cargo run --release` → compute fill window.

use anyhow::Result;

fn main() -> Result<()> {
    println!("helmholtz-wods — Walk on Decomposed Subdomains");
    println!("Week 1: compute dispatch fill (Pass A/C/D come online W2–W4)");
    helmholtz_wods::gpu::run_fill_window()
}
