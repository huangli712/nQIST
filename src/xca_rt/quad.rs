// Project : nQIST (real-time XCA)
// File    : src/xca_rt/quad.rs
// Purpose : quadrature layer. The reference-family integrators are
//           re-exported from `oca` DIRECTLY (single source of truth -
//           any local copy drifts, as the calibration found the hard
//           way), plus one guarded variant for the chain evaluator,
//           whose short buffers cannot feed the xiw warmup branch.

pub(crate) use crate::oca::{conv2b_scalar, gregory_scalar, lowside_scalar, poly_integral_scalar};

use kadanoffbaym::complex::Complex;
use kadanoffbaym::weights::Integrator;

/// Sub-interval integral for the chain evaluator: identical to the
/// reference `poly_integral_scalar` on long buffers; short buffers
/// (length <= i0 + k, which the xiw branch cannot index) fall back
/// to the Gregory startup rows, clamped to the buffer.
pub(crate) fn poly_integral_chain(
    v: &[Complex],
    i0: usize,
    i1: usize,
    integ: &Integrator,
) -> Complex {
    if i1 <= i0 {
        // zero-width interval: contributes nothing
        return Complex::ZERO;
    }
    let k = integ.k as usize;
    if i1 - i0 > k || v.len() <= i0 + k {
        // Gregory on the sub-interval with the startup rows clamped
        let n = i1 - i0;
        let slice = &v[i0..=i1];
        let jmax = n.max(k).min(slice.len().saturating_sub(1));
        let mut out = Complex::ZERO;
        for j in 0..=jmax {
            let w = integ.giw.at(n, j);
            if w != 0.0 {
                out += Complex::new(w, 0.0) * slice[j];
            }
        }
        out
    } else {
        poly_integral_scalar(v, i0, i1, integ)
    }
}
