//! Reference solvers and metrics (FDM, RMSE, Laplace analytical).

pub mod fdm;
pub mod laplace;
pub mod rmse;

pub use laplace::{analytical_laplace_empty_tank, dirichlet_on_boundary};
pub use rmse::{relative_rmse, relative_rmse_complex};
