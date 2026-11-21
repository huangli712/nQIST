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

/// Quadrature weight on the interval's own nodes. Short intervals use
/// the interpolant through those nodes, without truncating a Gregory row.
pub(crate) fn chain_weight(n: usize, j: usize, integ: &Integrator) -> f64 {
    use kadanoffbaym::weights::PolynomialIntegrationWeights;
    use std::sync::OnceLock;

    if n == 0 {
        return 0.0;
    }
    if n > integ.k as usize {
        return integ.giw.at(n, j);
    }
    static WEIGHTS: OnceLock<[PolynomialIntegrationWeights; 10]> = OnceLock::new();
    let weights = WEIGHTS.get_or_init(|| {
        std::array::from_fn(|index| PolynomialIntegrationWeights::new((index + 1) as i64))
    });
    weights[n - 1].at(0, n, j)
}

/// Sub-interval integral using the full startup stencil when available.
pub(crate) fn poly_integral_chain(
    v: &[Complex],
    i0: usize,
    i1: usize,
    integ: &Integrator,
) -> Complex {
    let n = i1.saturating_sub(i0);
    if n == 0 {
        return Complex::ZERO;
    }
    let k = integ.k as usize;
    if n <= k && v.len() > i0 + k && i1 <= k {
        return poly_integral_scalar(v, i0, i1, integ);
    }
    v[i0..=i1].iter().enumerate().fold(Complex::ZERO, |sum, (j, &value)| {
        let w = chain_weight(n, j, integ);
        if w == 0.0 { sum } else { sum + Complex::new(w, 0.0) * value }
    })
}

/// Entrywise matrix variant of [`poly_integral_chain`]: identical
/// weights applied per matrix entry (the quadrature is
/// entrywise-linear), so the result is exactly the scalar kernel
/// applied to each entry channel of the block-valued array.
pub(crate) fn poly_integral_chain_mat(
    v: &[Vec<Complex>],
    i0: usize,
    i1: usize,
    nel: usize,
    integ: &Integrator,
) -> Vec<Complex> {
    let mut out = vec![Complex::ZERO; nel];
    for q in 0..nel {
        let col: Vec<Complex> = v.iter().map(|b| b[q]).collect();
        out[q] = poly_integral_chain(&col, i0, i1, integ);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_chain_integral_preserves_constant() {
        let integ = Integrator::new(5);
        for n in 1..=5 {
            let values = vec![Complex::ONE; n + 1];
            let got = poly_integral_chain(&values, 0, n, &integ);
            assert!((got - Complex::new(n as f64, 0.0)).abs() < 1e-12, "n={n}: {got:?}");
        }
    }

    #[test]
    fn short_chain_integral_preserves_affine_offset_interval() {
        let integ = Integrator::new(5);
        let values = [Complex::ZERO; 7].map(|_| Complex::new(2.0, 1.0));
        let mut values = values;
        values[6] = Complex::new(4.0, 3.0);
        let got = poly_integral_chain(&values, 5, 6, &integ);
        assert!((got - Complex::new(3.0, 2.0)).abs() < 1e-12, "{got:?}");
    }
}
