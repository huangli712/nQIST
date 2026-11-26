// Project : nQIST (real-time XCA)
// File    : src/xca_rt/line.rs
// Purpose : The line abstraction of the order-1 and order-2 real-time
//           XCA kernels. A kernel reads a hybridization line through
//           four component accessors in the conventions of the stock
//           NCA/OCA kernels (`sgtr`, `stv`, `svt`, `sles`) plus the
//           separable terms of its conjugated mixed component. Two
//           implementations provide the reads:
//
//           - the exponential-sum route (`fit.rs`): [`SepLine1`] for
//             the order-1 kernel, [`SepLine`] for the order-2 kernels.
//             It reconstructs every component from one sample column
//             and therefore needs a time-translation invariant bath.
//           - the two-time separable route (`twotime.rs`), valid for
//             any bath.

use super::eval1::Line1;
use super::eval2::LineSep;
use super::fit::CsoeFit;
use super::tv_c;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::ContourGreen;

/// Point reads of one hybridization line on the window `0..=n1` of a
/// real-time step.
pub(crate) trait RtLine {
    /// Greater `sgtr(n, j)`: `ret - les^dag` in the triangle, the
    /// equal-time contact, the Hermitian continuation out of it.
    fn gtr(&self, n: i64, j: i64) -> Complex;
    /// Mixed `stv(n, m)` (the lmix component).
    fn tv(&self, n: i64, m: i64) -> Complex;
    /// Conjugated mixed `svt(tau, t) = -sign lmix(t, ntau - tau)^dag`.
    fn vt(&self, tau: i64, t: i64) -> Complex;
    /// Lesser `sles(j, n)`: `less(j, n)` in the triangle, the
    /// Hermitian continuation out of it.
    fn les(&self, j: i64, n: i64) -> Complex;
    /// Number of separable terms of the conjugated mixed component.
    fn vt_nterms(&self) -> usize;
    /// Term `q` of the conjugated mixed component: `(profile over the
    /// Matsubara axis m = 0..=ntau, factor over t = 0..=n1)` with
    /// `vt(m, t) = sum_q factor[q][t] * profile[q][m]`.
    fn vt_term(&self, q: usize) -> (Vec<Complex>, Vec<Complex>);
}

/// Order-1 exponential-sum line ([`Line1`]) on the window `0..=n1`.
pub(crate) struct SepLine1<'a> {
    line: &'a Line1<'a>,
    n1: usize,
    ntau: i64,
    sign: f64,
    /// lesser coefficients of the anchor `n` (one projection per anchor)
    les: std::cell::RefCell<Option<(i64, Vec<Complex>)>>,
}

impl<'a> SepLine1<'a> {
    pub fn new(line: &'a Line1<'a>, n1: usize, ntau: i64, sign: f64) -> Self {
        SepLine1 {
            line,
            n1,
            ntau,
            sign,
            les: std::cell::RefCell::new(None),
        }
    }
}

impl RtLine for SepLine1<'_> {
    fn gtr(&self, n: i64, j: i64) -> Complex {
        self.line.gtr(n, j)
    }

    fn tv(&self, n: i64, m: i64) -> Complex {
        self.line.tv(n, m, self.n1)
    }

    fn vt(&self, tau: i64, t: i64) -> Complex {
        -Complex::new(self.sign, 0.0) * self.tv(t, self.ntau - tau).conj()
    }

    fn les(&self, j: i64, n: i64) -> Complex {
        let cached = self.les.borrow();
        let coefs = match cached.as_ref() {
            Some((anchor, c)) if *anchor == n => c.clone(),
            _ => {
                drop(cached);
                let c = self.line.les_coefs(n);
                *self.les.borrow_mut() = Some((n, c.clone()));
                c
            }
        };
        self.line.les(&coefs, j, n)
    }

    fn vt_nterms(&self) -> usize {
        0
    }

    fn vt_term(&self, _q: usize) -> (Vec<Complex>, Vec<Complex>) {
        unreachable!("the order-1 kernel has no separable mixed term")
    }
}

/// Order-2 exponential-sum line ([`LineSep`]) on the window `0..=n1`.
pub(crate) struct SepLine<'a> {
    sep: &'a LineSep<'a>,
    fit: &'a CsoeFit,
    lam: &'a ContourGreen,
    h: f64,
    n1: usize,
}

impl<'a> SepLine<'a> {
    pub fn new(sep: &'a LineSep<'a>, fit: &'a CsoeFit, lam: &'a ContourGreen, h: f64, n1: usize) -> Self {
        SepLine {
            sep,
            fit,
            lam,
            h,
            n1,
        }
    }
}

impl RtLine for SepLine<'_> {
    fn gtr(&self, n: i64, j: i64) -> Complex {
        self.sep.gtr(n, j, self.n1)
    }

    fn tv(&self, n: i64, m: i64) -> Complex {
        tv_c(self.lam, n, m)
    }

    fn vt(&self, tau: i64, t: i64) -> Complex {
        self.sep.vt(tau, t, self.n1)
    }

    fn les(&self, j: i64, n: i64) -> Complex {
        self.sep.les(j, n, self.n1)
    }

    fn vt_nterms(&self) -> usize {
        self.fit.poles.len()
    }

    fn vt_term(&self, q: usize) -> (Vec<Complex>, Vec<Complex>) {
        let profile = (0..=self.lam.ntau as i64)
            .map(|m| self.sep.vt_coef_at(m, q, self.n1))
            .collect();
        // vt(m, t) = sum_l c_l[m] e^{+i w_l h t} (the same phase the
        // order-2 kernels apply: exp(-i w h (0 - t)))
        let factor = (0..=self.n1 as i64)
            .map(|t| (-Complex::I * self.fit.poles[q] * self.h * (0 - t) as f64).exp())
            .collect();
        (profile, factor)
    }
}
