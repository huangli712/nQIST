// Project : nQIST
// File    : src/lib.rs
// Purpose : crate root and module declarations
//
// Numerical results must match the PPSC/NESSi 2.0.0 reference point by point.
//!
//! nQIST
//!
//! Non-equilibrium DMFT impurity solvers (PPSC NCA/OCA, IPT) and the
//! DMFT self-consistency driver, built on the `KadanoffBaym` contour
//! Green's function crate and mirroring the PPSC programs of NESSi
//! 2.0.0.

#![forbid(unsafe_code)]

pub mod boson;
pub mod dmft;
pub mod ed;
pub mod hamilton;
pub mod hilbert;
pub mod io;
pub mod ipt;
pub mod nca;
pub mod observable;
pub mod oca;
pub mod solver;
pub mod xca;
pub mod xca_rt;

/// Crate version as reported by `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
