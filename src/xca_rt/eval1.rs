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

use super::fit::{CsoeFit, CsoeSolver};
use super::{gtr, les_pp, tv_pp};
use crate::nca::{Hybridization, SigmaDiagram};
use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::PseudoGreen;

#[inline]
fn phase(w: f64, a: i64, b: i64, h: f64) -> Complex {
    (-Complex::I * w * h * (a - b) as f64).exp()
}

/// Pole-separated view of the single NCA line: greater from the fit;
/// les on the reflected poles (one projection per line), tv projected
/// per anchor.
pub(crate) struct Line1<'a> {
    pub lam: &'a dyn LamRead,
    pub fit: &'a CsoeFit,
    pub h: f64,
    /// solvers of the tv projections (fit poles) and of the les
    /// projection (reflected poles), built on first use for the window
    /// of the step
    tv_solver: std::cell::RefCell<Option<CsoeSolver>>,
    les_solver: std::cell::RefCell<Option<CsoeSolver>>,
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
        Line1 {
            lam,
            fit,
            h,
            tv_solver: std::cell::RefCell::new(None),
            les_solver: std::cell::RefCell::new(None),
        }
    }

    pub fn gtr(&self, n: i64, j: i64) -> Complex {
        let mut acc = Complex::ZERO;
        for (l, &c) in self.fit.coefs.iter().enumerate() {
            acc += c * phase(self.fit.poles[l], n, j, self.h);
        }
        acc
    }

    /// Coefficients of the les component on the reflected poles
    /// `-w_l` (see [`reflected_poles`]). The projection reads the
    /// lower-triangle row at anchor 0 over `t = 0..=n`: for the
    /// time-translation invariant baths the real-time route is defined
    /// for, the coefficients do not depend on the anchor, and one
    /// projection per line serves every `j`.
    pub fn les_coefs(&self, n: i64) -> Vec<Complex> {
        let ns = n as usize + 1;
        if self.les_solver.borrow().as_ref().map_or(true, |s| s.ns() != ns) {
            let poles = reflected_poles(&self.fit.poles);
            *self.les_solver.borrow_mut() = Some(CsoeSolver::new(&poles, self.h, ns));
        }
        let values: Vec<Complex> = (0..=n).map(|t| self.lam.les(0, t)).collect();
        self.les_solver.borrow().as_ref().unwrap().apply(&values)
    }

    /// Les at `(j, n)`, `j <= n`, from the coefficients of
    /// [`Self::les_coefs`].
    pub fn les(&self, coefs: &[Complex], j: i64, n: i64) -> Complex {
        let mut acc = Complex::ZERO;
        for (l, &c) in coefs.iter().enumerate() {
            // exp(+i w_l h (n - j))
            acc += c * phase(self.fit.poles[l], j, n, self.h);
        }
        acc
    }

    pub fn tv(&self, n: i64, m: i64, nsamp: usize) -> Complex {
        let ns = nsamp + 1;
        if self.tv_solver.borrow().as_ref().map_or(true, |s| s.ns() != ns) {
            *self.tv_solver.borrow_mut() = Some(CsoeSolver::new(&self.fit.poles, self.h, ns));
        }
        let values: Vec<Complex> = (0..=nsamp as i64).map(|t| self.lam.tv(t, m)).collect();
        let coefs = self.tv_solver.borrow().as_ref().unwrap().apply(&values);
        let mut acc = Complex::ZERO;
        for (l, &c) in coefs.iter().enumerate() {
            acc += c * phase(self.fit.poles[l], n, 0, self.h);
        }
        acc
    }
}

/// Projection basis of the conjugated components (les, vt) of a
/// hybridization line: the reflections `-w_l` of the fit poles. The fit
/// samples the greater `ret - les*` of the bath; the les data itself
/// and the conjugated lmix (`vt_c`) therefore rotate opposite to it on
/// a bath stored by KadanoffBaym (the single-band dimer bath: a 1-pole
/// fit reconstructs les/vt to 1e-11 on the reflected poles, 1e-3 on
/// the fit poles).
pub(crate) fn reflected_poles(poles: &[f64]) -> Vec<f64> {
    poles.iter().map(|w| -w).collect()
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
    // multi-band sectors take the matrix path (the same dispatch
    // rule as the stock oca kernel: ANY involved block > 1x1 - a
    // dim-1 OUTPUT sector with dim-2 intermediates still needs the
    // full op1 . G . op2 sandwich, the (0,0) corner would truncate
    // it); the scalar fast path below is the all-1x1 single-band
    // production route, kept bit-identical
    if pp_g[d.sa].size1 > 1 || pp_sigma[d.out].size1 > 1 {
        add_sigma_tstp_xca1_matrix(n, d, pp_g, h, fit, pp_sigma);
        return;
    }
    let dim_out = pp_sigma[d.out].size1;
    let ntau = pp_g[d.sa].ntau as i64;
    let ga = &pp_g[d.sa];
    let line = Line1::new(d.delta, fit, h);
    let (op1, op2) = (d.op1.get(0, 0), d.op2.get(0, 0));
    // non-equal-time ret (greater) and lesser
    let les_coefs = line.les_coefs(n);
    for j in 0..n {
        let l = line.gtr(n, j);
        let prod = op1 * gtr(ga, n, j) * op2;
        let add = d.prefactor * l * prod;
        let idx = pp_sigma[d.out].ret.idx(n, j);
        pp_sigma[d.out].ret.block_mut(idx)[0] += add;
        let l = line.les(&les_coefs, j, n);
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

/// Multi-band (matrix-block) order-1 path: the SAME kernel
/// structure as `nca::accumulate_sigma_tstp` (op1 · G^sa · op2
/// sandwiches into every output entry) with every hybridization
/// read replaced by the pole-basis reconstruction of [`Line1`] -
/// the independent-implementation cross-check axis of the rt-XCA
/// campaign, now band-generic (anchored by
/// `xca1_matches_nca_realtime_twoband`).
#[allow(clippy::too_many_arguments)]
fn add_sigma_tstp_xca1_matrix(
    n: i64,
    d: &SigmaDiagram,
    pp_g: &[PseudoGreen],
    h: f64,
    fit: &CsoeFit,
    pp_sigma: &mut [PseudoGreen],
) {
    use crate::nca::{pp_greater, pp_les, pp_tv};
    let dim_out = pp_sigma[d.out].size1;
    let ntau = pp_g[d.sa].ntau as i64;
    let ga = &pp_g[d.sa];
    let line = Line1::new(d.delta, fit, h);
    // non-equal-time ret (greater) and lesser
    let les_coefs = line.les_coefs(n);
    for j in 0..n {
        let l = line.gtr(n, j);
        let prod = d.op1.mat_mul(&pp_greater(ga, n, j)).mat_mul(&d.op2);
        let idx = pp_sigma[d.out].ret.idx(n, j);
        for i in 0..dim_out {
            for k in 0..dim_out {
                pp_sigma[d.out].ret.block_mut(idx)[i * dim_out + k] +=
                    d.prefactor * l * prod.get(i, k);
            }
        }
        let l = line.les(&les_coefs, j, n);
        let prod = d.op1.mat_mul(&pp_les(ga, j, n)).mat_mul(&d.op2);
        let idx = pp_sigma[d.out].less.idx(j, n);
        for i in 0..dim_out {
            for k in 0..dim_out {
                pp_sigma[d.out].less.block_mut(idx)[i * dim_out + k] +=
                    d.prefactor * l * prod.get(i, k);
            }
        }
    }
    // mixed component
    for m in 0..=ntau {
        let l = line.tv(n, m, n as usize);
        let prod = d.op1.mat_mul(&pp_tv(ga, n, m)).mat_mul(&d.op2);
        for i in 0..dim_out {
            for k in 0..dim_out {
                pp_sigma[d.out].lmix.raw_block_at_mut(n, m)[i * dim_out + k] +=
                    d.prefactor * l * prod.get(i, k);
            }
        }
    }
    // equal-time contacts: raw point reads, exactly as the scalar
    // path and the reference (the delta data is scalar per channel)
    {
        let l_nn = d.delta.les(n, n);
        let r_nn = d.delta.ret(n, n);
        let lam_et = l_nn - r_nn.conj();
        let prod = d.op1.mat_mul(&pp_greater(ga, n, n)).mat_mul(&d.op2);
        let idx = pp_sigma[d.out].ret.idx(n, n);
        for i in 0..dim_out {
            for k in 0..dim_out {
                pp_sigma[d.out].ret.block_mut(idx)[i * dim_out + k] +=
                    d.prefactor * lam_et * prod.get(i, k);
            }
        }
        let l = -l_nn.conj();
        let prod = d.op1.mat_mul(&pp_les(ga, n, n)).mat_mul(&d.op2);
        let idx = pp_sigma[d.out].less.idx(n, n);
        for i in 0..dim_out {
            for k in 0..dim_out {
                pp_sigma[d.out].less.block_mut(idx)[i * dim_out + k] +=
                    d.prefactor * l * prod.get(i, k);
            }
        }
    }
}
