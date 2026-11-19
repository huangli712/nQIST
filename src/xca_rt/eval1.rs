// Project : nQIST (real-time XCA)
// File    : src/xca_rt/eval1.rs
// Purpose : order-1 real-time XCA evaluator = NCA, with the
//           hybridization reads replaced by pole reconstructions.
//           Calibration anchor: must reproduce
//           `nca::accumulate_sigma_tstp` to fit precision.
//
//           The volume reads (greater, lesser, tv) go through the
//           pole basis; the equal-time contact combinations keep raw
//           point reads (separation is irrelevant at a point).

use super::fit::{CsoeFit, csoe_coeffs};
use super::{gtr, les_pp, tv_pp};
use crate::nca::{Hybridization, SigmaDiagram};
use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::PseudoGreen;

#[inline]
fn phase(w: f64, a: i64, b: i64, h: f64) -> Complex {
    (-Complex::I * w * h * (a - b) as f64).exp()
}

/// Pole-separated view of the single NCA line: greater from the fit;
/// les/tv projected onto the same basis per anchor.
pub(crate) struct Line1<'a> {
    pub lam: &'a dyn LamRead,
    pub fit: &'a CsoeFit,
    pub h: f64,
}

/// Point reads the evaluator needs from the line object, mirroring
/// the `delta_*` accessors of `nca`.
pub(crate) trait LamRead {
    fn gtr(&self, n: i64, j: i64) -> Complex;
    fn les(&self, j: i64, n: i64) -> Complex;
    fn tv(&self, n: i64, m: i64) -> Complex;
    fn ret(&self, n: i64, j: i64) -> Complex;
}

impl LamRead for kadanoffbaym::types::ContourGreen {
    fn gtr(&self, n: i64, j: i64) -> Complex {
        super::gtr_c(self, n, j)
    }
    fn les(&self, j: i64, n: i64) -> Complex {
        super::les_c(self, j, n)
    }
    fn tv(&self, n: i64, m: i64) -> Complex {
        self.lmix.raw_block_at(n, m)[0]
    }
    fn ret(&self, n: i64, j: i64) -> Complex {
        self.ret.raw_block_at(n, j)[0]
    }
}

impl<'a> Line1<'a> {
    pub fn new(lam: &'a dyn LamRead, fit: &'a CsoeFit, h: f64) -> Self {
        Line1 { lam, fit, h }
    }

    pub fn gtr(&self, n: i64, j: i64) -> Complex {
        let mut acc = Complex::ZERO;
        for (l, &c) in self.fit.coefs.iter().enumerate() {
            acc += c * phase(self.fit.poles[l], n, j, self.h);
        }
        acc
    }

    pub fn les(&self, j: i64, n: i64, nsamp: usize) -> Complex {
        // project the les component (function of n at anchor j)
        let values: Vec<Complex> = (0..=nsamp as i64).map(|t| self.lam.les(j, t)).collect();
        let coefs = csoe_coeffs(&self.fit.poles, self.h, &values);
        let mut acc = Complex::ZERO;
        for (l, &c) in coefs.iter().enumerate() {
            acc += c * phase(self.fit.poles[l], n, 0, self.h);
        }
        acc
    }

    pub fn tv(&self, n: i64, m: i64, nsamp: usize) -> Complex {
        let values: Vec<Complex> = (0..=nsamp as i64).map(|t| self.lam.tv(t, m)).collect();
        let coefs = csoe_coeffs(&self.fit.poles, self.h, &values);
        let mut acc = Complex::ZERO;
        for (l, &c) in coefs.iter().enumerate() {
            acc += c * phase(self.fit.poles[l], n, 0, self.h);
        }
        acc
    }
}

/// Order-1 separated evaluator; mirrors `accumulate_sigma_tstp` with
/// the volume line reads reconstructed from the pole basis.
pub(crate) fn add_sigma_tstp_xca1(
    n: i64,
    d: &SigmaDiagram,
    pp_g: &[PseudoGreen],
    h: f64,
    fit: &CsoeFit,
    pp_sigma: &mut [PseudoGreen],
) {
    let dim_out = pp_sigma[d.out].size1;
    let ntau = pp_g[d.sa].ntau as i64;
    let ga = &pp_g[d.sa];
    let line = Line1::new(d.delta, fit, h);
    let (op1, op2) = (d.op1.get(0, 0), d.op2.get(0, 0));
    // non-equal-time ret (greater) and lesser
    for j in 0..n {
        let l = line.gtr(n, j);
        let prod = op1 * gtr(ga, n, j) * op2;
        let add = d.prefactor * l * prod;
        let idx = pp_sigma[d.out].ret.idx(n, j);
        pp_sigma[d.out].ret.block_mut(idx)[0] += add;
        let l = line.les(j, n, n as usize);
        let prod = op1 * les_pp(ga, j, n) * op2;
        let add = d.prefactor * l * prod;
        pp_sigma[d.out]
            .less
            .block_mut(pp_sigma[d.out].less.idx(j, n))[0] += add;
    }
    // mixed component
    for m in 0..=ntau {
        let l = line.tv(n, m, n as usize);
        let prod = op1 * tv_pp(ga, n, m) * op2;
        let add = d.prefactor * l * prod;
        pp_sigma[d.out].lmix.raw_block_at_mut(n, m)[0] += add;
    }
    // equal-time contacts: raw point reads (lam^< - (lam^R)^dag and
    // -(lam^<)^dag), exactly as the reference
    {
        let l_nn = d.delta.les(n, n);
        let r_nn = d.delta.ret(n, n);
        let lam_et = l_nn - r_nn.conj();
        let prod = op1 * gtr(ga, n, n) * op2;
        pp_sigma[d.out].ret.block_mut(pp_sigma[d.out].ret.idx(n, n))[0] +=
            d.prefactor * lam_et * prod;
        let l = -l_nn.conj();
        let prod = op1 * les_pp(ga, n, n) * op2;
        pp_sigma[d.out]
            .less
            .block_mut(pp_sigma[d.out].less.idx(n, n))[0] += d.prefactor * l * prod;
    }
}

/// Convenience: all channels of `pp_ints` with a shared fit per
/// distinct delta object identity is NOT attempted here; callers
/// supply one fit per diagram (same fit object reused per channel).
#[allow(dead_code)]
pub(crate) fn _channels(_pp_ints: &[Hybridization]) -> usize {
    _pp_ints.len()
}
