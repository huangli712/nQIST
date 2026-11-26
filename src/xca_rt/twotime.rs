// Project : nQIST (real-time XCA)
// File    : src/xca_rt/twotime.rs
// Purpose : Two-time separable expansion of a hybridization channel on
//           the real-time grid: the general-bath route of the order-1
//           and order-2 real-time XCA kernels.
//
//           The exponential-sum route (`fit.rs`) reconstructs every
//           component of a line from one sample column and therefore
//           needs a time-translation invariant bath; self-consistent
//           non-equilibrium DMFT does not provide one (after a quench
//           the Bethe bath deviates from time translation invariance at
//           the percent level). Here each component is factorized
//           where it is defined on the grid,
//
//             gtr(t_i, t_j)   = sum_q a[q][i] b[q][j],    i >= j
//             les(t_j, t_i)   = sum_q a[q][i] b[q][j],    j <= i
//             tv (t_i, tau_m) = sum_q a[q][i] b[q][j=m],  all i, m
//
//           The triangle components are fitted by a cross
//           approximation: the last `r` rows (observed over the whole
//           width) are the skeleton, the factors of the other rows
//           follow by least squares over the observed prefix. A bath
//           built from `p` exact exponentials is fitted exactly at
//           rank `p`; for a smooth bath the rank grows until the
//           maximum residual on the support is within the tolerance.

use super::line::RtLine;
use super::{gtr_c, les_c, tv_c};
use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::ContourGreen;

/// Alternating-least-squares polishing sweeps of a nearly converged
/// fit.
const MAX_SWEEPS: usize = 40;
/// A rank whose residual lies within this factor of the tolerance is
/// polished instead of grown further.
const POLISH_FACTOR: f64 = 100.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Support {
    /// entry (i, j) belongs to the fit if `i >= j` (square data)
    Lower,
    /// every entry belongs to the fit
    Full,
}

/// A masked `rows x cols` view of row-major data: entry (i, j) is
/// `data[i * cols + j]`; only the entries of the support are read.
struct Masked<'a> {
    rows: usize,
    cols: usize,
    data: &'a [Complex],
    support: Support,
}

impl<'a> Masked<'a> {
    fn new(rows: usize, cols: usize, data: &'a [Complex], support: Support) -> Self {
        Masked {
            rows,
            cols,
            data,
            support,
        }
    }

    /// rows of column `j` that belong to the support, as a range
    fn col_rows(&self, j: usize) -> std::ops::Range<usize> {
        match self.support {
            Support::Lower => j..self.rows,
            Support::Full => 0..self.rows,
        }
    }

    /// columns of row `i` that belong to the support, as a range
    fn row_cols(&self, i: usize) -> std::ops::Range<usize> {
        match self.support {
            Support::Lower => 0..i + 1,
            Support::Full => 0..self.cols,
        }
    }

    /// Largest magnitude on the support.
    fn max_abs(&self) -> f64 {
        let mut s = 0.0f64;
        for i in 0..self.rows {
            for j in self.row_cols(i) {
                s = s.max(self.data[i * self.cols + j].abs());
            }
        }
        s
    }

    /// Largest residual magnitude `|data - sum_q a[q][i] b[q][j]|` on
    /// the support.
    fn residual(&self, a: &[Vec<Complex>], b: &[Vec<Complex>]) -> f64 {
        let mut worst = 0.0f64;
        for i in 0..self.rows {
            for j in self.row_cols(i) {
                let mut acc = self.data[i * self.cols + j];
                for (aq, bq) in a.iter().zip(b) {
                    acc -= aq[i] * bq[j];
                }
                worst = worst.max(acc.abs());
            }
        }
        worst
    }
}

/// A separable form `sum_q a[q][i] b[q][j]` and its maximum residual
/// on the support it was fitted to.
#[derive(Clone)]
pub(crate) struct LowRank {
    pub a: Vec<Vec<Complex>>,
    pub b: Vec<Vec<Complex>>,
    pub max_abs: f64,
}

impl LowRank {
    #[inline]
    pub fn rank(&self) -> usize {
        self.a.len()
    }

    #[inline]
    pub fn at(&self, i: usize, j: usize) -> Complex {
        let mut acc = Complex::ZERO;
        for (a, b) in self.a.iter().zip(&self.b) {
            acc += a[i] * b[j];
        }
        acc
    }
}

/// Solve the full-rank `n x n` linear system `g x = rhs` (Gaussian
/// elimination with partial pivoting; the arguments are consumed).
fn solve(mut g: Vec<Complex>, mut rhs: Vec<Complex>, n: usize) -> Vec<Complex> {
    for k in 0..n {
        let mut p = k;
        for i in (k + 1)..n {
            if g[i * n + k].abs_sqr() > g[p * n + k].abs_sqr() {
                p = i;
            }
        }
        if p != k {
            for j in 0..n {
                g.swap(k * n + j, p * n + j);
            }
            rhs.swap(k, p);
        }
        let piv = g[k * n + k];
        if piv.abs_sqr() == 0.0 {
            rhs[k] = Complex::ZERO;
            continue;
        }
        for i in (k + 1)..n {
            let f = g[i * n + k] / piv;
            if f.abs_sqr() == 0.0 {
                continue;
            }
            for j in k..n {
                let pivot = g[k * n + j];
                g[i * n + j] -= f * pivot;
            }
            let b = rhs[k];
            rhs[i] -= f * b;
        }
    }
    let mut x = vec![Complex::ZERO; n];
    for i in (0..n).rev() {
        let mut acc = rhs[i];
        for j in (i + 1)..n {
            acc -= g[i * n + j] * x[j];
        }
        if g[i * n + i].abs_sqr() > 0.0 {
            x[i] = acc / g[i * n + i];
        }
    }
    x
}

/// Least squares `min_w sum_k |sum_q cols[q][k] w_q - rhs[k]|^2`:
/// Householder QR with at least as many equations as unknowns (a
/// deficient column gets the weight 0), otherwise the minimum-norm
/// solution of the ridge normal equations.
fn ls_solve(cols: &[&[Complex]], rhs: &[Complex]) -> Vec<Complex> {
    let (r, m) = (cols.len(), rhs.len());
    if r == 0 {
        return Vec::new();
    }
    if m >= r {
        let mut x: Vec<Vec<Complex>> = cols.iter().map(|c| c.to_vec()).collect();
        let mut d = rhs.to_vec();
        let mut v = vec![Complex::ZERO; m];
        for k in 0..r {
            let mut norm2 = 0.0f64;
            for i in k..m {
                norm2 += x[k][i].abs_sqr();
            }
            if norm2 == 0.0 {
                continue;
            }
            let norm = norm2.sqrt();
            let alpha = x[k][k];
            // x[k][k] <- -sign(alpha) * norm makes v = x - beta e_1
            let beta = if alpha.abs_sqr() == 0.0 {
                -Complex::new(norm, 0.0)
            } else {
                -alpha * (norm / alpha.abs())
            };
            v[k] = alpha - beta;
            for i in (k + 1)..m {
                v[i] = x[k][i];
                x[k][i] = Complex::ZERO;
            }
            let mut vv = 0.0f64;
            for i in k..m {
                vv += v[i].abs_sqr();
            }
            if vv == 0.0 {
                continue;
            }
            for q in k..r {
                let mut s = Complex::ZERO;
                for i in k..m {
                    s += v[i].conj() * x[q][i];
                }
                let f = Complex::new(2.0 / vv, 0.0) * s;
                for i in k..m {
                    x[q][i] -= f * v[i];
                }
            }
            let mut s = Complex::ZERO;
            for i in k..m {
                s += v[i].conj() * d[i];
            }
            let f = Complex::new(2.0 / vv, 0.0) * s;
            for i in k..m {
                d[i] -= f * v[i];
            }
            x[k][k] = beta;
        }
        let mut w = vec![Complex::ZERO; r];
        for k in (0..r).rev() {
            if x[k][k].abs_sqr() == 0.0 {
                continue;
            }
            let mut acc = d[k];
            for j in (k + 1)..r {
                acc -= x[j][k] * w[j];
            }
            w[k] = acc / x[k][k];
        }
        return w;
    }
    // underdetermined: ridge normal equations
    let mut g = vec![Complex::ZERO; r * r];
    let mut rr = vec![Complex::ZERO; r];
    for k in 0..m {
        for p in 0..r {
            let cp = cols[p][k].conj();
            for q in 0..r {
                g[p * r + q] += cp * cols[q][k];
            }
            rr[p] += cp * rhs[k];
        }
    }
    let mut trace = 0.0f64;
    for q in 0..r {
        trace += g[q * r + q].re;
    }
    let ridge = 1.0e-12 * trace / r as f64 + 1.0e-300;
    for q in 0..r {
        g[q * r + q] += Complex::new(ridge, 0.0);
    }
    solve(g, rr, r)
}

/// One alternating-least-squares sweep: every column factor, then
/// every row factor.
fn als_sweep(m: &Masked, a: &mut [Vec<Complex>], b: &mut [Vec<Complex>]) {
    let r = a.len();
    if r == 0 {
        return;
    }
    for j in 0..m.cols {
        let rows = m.col_rows(j);
        let design: Vec<&[Complex]> = a.iter().map(|c| &c[rows.clone()]).collect();
        let rhs: Vec<Complex> = rows.clone().map(|i| m.data[i * m.cols + j]).collect();
        let w = ls_solve(&design, &rhs);
        for q in 0..r {
            b[q][j] = w[q];
        }
    }
    for i in 0..m.rows {
        let cols = m.row_cols(i);
        let design: Vec<&[Complex]> = b.iter().map(|c| &c[cols.clone()]).collect();
        let rhs = m.data[i * m.cols..i * m.cols + cols.end].to_vec();
        let w = ls_solve(&design, &rhs);
        for q in 0..r {
            a[q][i] = w[q];
        }
    }
}

/// Rank-`r` cross approximation of the masked data: the last `r` rows
/// are the skeleton (their `a` factors are the unit matrix), the
/// skeleton entries over the leading columns are the `b` factors
/// there, and the factors of the other rows follow by least squares
/// over their observed prefix. The trailing columns of the triangle
/// (observed on skeleton rows only) take their observed entries as
/// `b` factors.
fn cross_approx(m: &Masked, r: usize) -> (Vec<Vec<Complex>>, Vec<Vec<Complex>>) {
    let (rows, cols) = (m.rows, m.cols);
    let first = rows - r;
    // leading columns covered by the skeleton; a triangle keeps its
    // trailing `r` columns for the second stage below
    let lead = match m.support {
        Support::Lower => cols - r,
        Support::Full => cols,
    };
    let mut b: Vec<Vec<Complex>> = (0..r)
        .map(|q| (0..lead).map(|j| m.data[(first + q) * cols + j]).collect())
        .collect();
    let mut a: Vec<Vec<Complex>> = vec![vec![Complex::ZERO; rows]; r];
    for q in 0..r {
        a[q][first + q] = Complex::ONE;
    }
    for i in 0..first {
        let width = m.row_cols(i).end;
        let design: Vec<&[Complex]> = b.iter().map(|c| &c[..width]).collect();
        let rhs = m.data[i * cols..i * cols + width].to_vec();
        let w = ls_solve(&design, &rhs);
        for q in 0..r {
            a[q][i] = w[q];
        }
    }
    for q in 0..r {
        b[q].resize(cols, Complex::ZERO);
    }
    for j in lead..cols {
        for q in 0..r {
            let i = first + q;
            if m.row_cols(i).contains(&j) {
                b[q][j] = m.data[i * cols + j];
            }
        }
    }
    (a, b)
}

/// Masked low-rank fit: the rank grows from 1 until the maximum
/// residual is within `tol_abs` (up to `max_rank`); a rank whose
/// residual is already within [`POLISH_FACTOR`] of the tolerance is
/// polished by alternating least squares first.
fn factorize(m: &Masked, tol_abs: f64, max_rank: usize) -> LowRank {
    let width = match m.support {
        Support::Lower => m.rows,
        Support::Full => m.cols,
    };
    let rmax = max_rank.min(m.rows).min(width).max(1);
    if m.max_abs() == 0.0 {
        return LowRank {
            a: Vec::new(),
            b: Vec::new(),
            max_abs: 0.0,
        };
    }
    let mut best: Option<LowRank> = None;
    for r in 1..=rmax {
        let (mut a, mut b) = cross_approx(m, r);
        let mut err = m.residual(&a, &b);
        if err > tol_abs && err <= POLISH_FACTOR * tol_abs {
            for _ in 0..MAX_SWEEPS {
                let before = err;
                als_sweep(m, &mut a, &mut b);
                err = m.residual(&a, &b);
                if err <= tol_abs || err >= before {
                    break;
                }
            }
        }
        let fit = LowRank { a, b, max_abs: err };
        if err <= tol_abs || best.as_ref().map_or(true, |o| fit.max_abs < o.max_abs) {
            if err <= tol_abs {
                return fit;
            }
            best = Some(fit);
        }
    }
    best.expect("at least one rank is tried")
}

/// Two-time separable line of a hybridization channel on the window
/// `0..=n` of a real-time step.
pub(crate) struct TwoTimeLine<'a> {
    lam: &'a ContourGreen,
    gtr: LowRank,
    les: LowRank,
    tv: LowRank,
    /// relative residuals `(gtr, les, tv)` of the three fits
    rel: [f64; 3],
    ntau: usize,
    sign: f64,
}

impl<'a> TwoTimeLine<'a> {
    /// Factorize the three components of `lam` on `0..=n` to the
    /// relative tolerance `tol_rel` (of the largest magnitude on each
    /// support), up to the rank `max_rank`.
    pub fn new(lam: &'a ContourGreen, n: usize, tol_rel: f64, max_rank: usize) -> Self {
        let size = n + 1;
        let ntau = lam.ntau as usize + 1;
        let mut gtr = vec![Complex::ZERO; size * size];
        let mut les = vec![Complex::ZERO; size * size];
        for i in 0..size {
            for j in 0..=i {
                gtr[i * size + j] = gtr_c(lam, i as i64, j as i64);
                les[i * size + j] = les_c(lam, j as i64, i as i64);
            }
        }
        let mut tv = vec![Complex::ZERO; size * ntau];
        for i in 0..size {
            for m in 0..ntau {
                tv[i * ntau + m] = tv_c(lam, i as i64, m as i64);
            }
        }
        let fit = |data: &Vec<Complex>, rows: usize, cols: usize, support: Support| {
            let m = Masked::new(rows, cols, data, support);
            let scale = m.max_abs();
            let f = factorize(&m, tol_rel * scale, max_rank);
            let rel = if scale > 0.0 { f.max_abs / scale } else { 0.0 };
            (f, rel)
        };
        let (gtr_fit, rel_gtr) = fit(&gtr, size, size, Support::Lower);
        let (les_fit, rel_les) = fit(&les, size, size, Support::Lower);
        let (tv_fit, rel_tv) = fit(&tv, size, ntau, Support::Full);
        let sign = lam.sign.as_i64() as f64;
        TwoTimeLine {
            lam,
            gtr: gtr_fit,
            les: les_fit,
            tv: tv_fit,
            rel: [rel_gtr, rel_les, rel_tv],
            ntau: ntau - 1,
            sign,
        }
    }

    /// Ranks `(gtr, les, tv)`.
    pub fn ranks(&self) -> (usize, usize, usize) {
        (self.gtr.rank(), self.les.rank(), self.tv.rank())
    }

    /// Relative residuals `(gtr, les, tv)` of the three fits; the
    /// tolerance is met when every entry is at most the `tol_rel` the
    /// line was built with.
    pub fn relative_residuals(&self) -> [f64; 3] {
        self.rel
    }
}

impl RtLine for TwoTimeLine<'_> {
    fn gtr(&self, n: i64, j: i64) -> Complex {
        if j <= n {
            self.gtr.at(n as usize, j as usize)
        } else {
            gtr_c(self.lam, n, j)
        }
    }

    fn tv(&self, n: i64, m: i64) -> Complex {
        self.tv.at(n as usize, m as usize)
    }

    fn vt(&self, tau: i64, t: i64) -> Complex {
        // -sign lmix(t, ntau - tau)^dag
        -Complex::new(self.sign, 0.0) * self.tv(t, self.ntau as i64 - tau).conj()
    }

    fn les(&self, j: i64, n: i64) -> Complex {
        if j <= n {
            self.les.at(n as usize, j as usize)
        } else {
            -self.les.at(j as usize, n as usize).conj()
        }
    }

    fn vt_nterms(&self) -> usize {
        self.tv.rank()
    }

    fn vt_term(&self, q: usize) -> (Vec<Complex>, Vec<Complex>) {
        // vt(m, t) = -sign conj(sum_q tb[q][ntau - m] ta[q][t])
        let ntau = self.ntau;
        let profile = (0..=ntau)
            .map(|m| -Complex::new(self.sign, 0.0) * self.tv.b[q][ntau - m].conj())
            .collect();
        let factor = self.tv.a[q].iter().map(|x| x.conj()).collect();
        (profile, factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kadanoffbaym::types::{Contour, Sign};

    fn fit(data: &[Complex], rows: usize, cols: usize, support: Support, tol: f64) -> LowRank {
        let m = Masked::new(rows, cols, data, support);
        factorize(&m, tol * m.max_abs(), rows.max(cols))
    }

    #[test]
    fn exact_low_rank_is_recovered() {
        let n = 12usize;
        let mut data = vec![Complex::ZERO; n * n];
        for i in 0..n {
            for j in 0..=i {
                let d = 0.1 * (i - j) as f64;
                data[i * n + j] = Complex::new(0.7 * (-0.4 * d).exp(), 0.0)
                    * (-Complex::I * 2.0 * d).exp()
                    + Complex::new(0.3, 0.1) * (-Complex::I * 3.0 * d).exp();
            }
        }
        let f = fit(&data, n, n, Support::Lower, 1e-12);
        assert_eq!(f.rank(), 2, "rank of an exact rank-2 kernel");
        assert!(f.max_abs < 1e-12, "residual {}", f.max_abs);
        let mut full_data = data.clone();
        for i in 0..n {
            for j in (i + 1)..n {
                let d = 0.1 * (i as f64 - j as f64);
                full_data[i * n + j] = Complex::new(0.7 * (-0.4 * d).exp(), 0.0)
                    * (-Complex::I * 2.0 * d).exp()
                    + Complex::new(0.3, 0.1) * (-Complex::I * 3.0 * d).exp();
            }
        }
        let full = fit(&full_data, n, n, Support::Full, 1e-12);
        assert_eq!(full.rank(), 2, "rank of the full matrix");
        assert!(full.max_abs < 1e-12, "full residual {}", full.max_abs);
    }

    #[test]
    fn smooth_kernel_reaches_tolerance() {
        let n = 40usize;
        let mut data = vec![Complex::ZERO; n * n];
        for i in 0..n {
            for j in 0..=i {
                let (ti, tj) = (0.1 * i as f64, 0.1 * j as f64);
                let d = ti - tj;
                // non-translation-invariant smooth kernel
                data[i * n + j] = Complex::new(
                    (-(d / 0.3).powi(2) - (ti + tj) / 4.0).exp(),
                    (-(d / 0.5).powi(2)).exp() * (ti - tj),
                );
            }
        }
        let f = fit(&data, n, n, Support::Lower, 1e-10);
        assert!(f.max_abs <= 1e-10, "residual {}", f.max_abs);
        assert!(f.rank() < n, "rank {} of a smooth kernel", f.rank());
    }

    /// Two-pole time-translation invariant bath on a real-time grid.
    struct Bath {
        nt: usize,
        contour: Contour,
        lam: ContourGreen,
    }

    impl Bath {
        fn new(nt: usize, ntau: usize) -> Self {
            let (h, beta) = (0.05f64, 5.0f64);
            let contour = Contour::new(nt as i64, ntau as i64, 1, 1, h * nt as f64, beta);
            let mut lam = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
            let eps = [-0.3f64, 0.7];
            let v2 = [0.16f64, 0.09];
            let f: Vec<f64> = eps.iter().map(|e| 1.0 / (1.0 + (beta * e).exp())).collect();
            let ret = |tau: f64| {
                Complex::new(0.0, -1.0)
                    * (0..2)
                        .map(|k| Complex::new(v2[k], 0.0) * (-Complex::I * eps[k] * tau).exp())
                        .sum::<Complex>()
            };
            for i in 0..=nt {
                for j in 0..=i {
                    lam.ret.raw_block_at_mut(i as i64, j as i64)[0] = ret(h * (i - j) as f64);
                    lam.less.raw_block_at_mut(j as i64, i as i64)[0] = Complex::new(0.0, 1.0)
                        * (0..2)
                            .map(|k| {
                                Complex::new(v2[k] * f[k], 0.0)
                                    * (-Complex::I * eps[k] * h * (j as f64 - i as f64)).exp()
                            })
                            .sum::<Complex>();
                }
                for m in 0..=ntau {
                    lam.lmix.raw_block_at_mut(i as i64, m as i64)[0] = Complex::new(0.0, 1.0)
                        * (0..2)
                            .map(|k| {
                                Complex::new(v2[k], 0.0)
                                    * (-Complex::I * eps[k] * h * i as f64).exp()
                                    * (-eps[k] * beta * m as f64 / ntau as f64).exp()
                            })
                            .sum::<Complex>();
                }
            }
            Bath {
                nt,
                contour,
                lam,
            }
        }
    }

    #[test]
    fn line_reads_match_the_raw_points() {
        let b = Bath::new(20, 40);
        let n = b.nt;
        let line = TwoTimeLine::new(&b.lam, n, 1e-10, 32);
        assert_eq!(line.ranks(), (2, 2, 2), "ranks of a two-pole bath");
        assert!(line.relative_residuals().iter().all(|r| *r <= 1e-10));
        let mut worst = 0.0f64;
        for i in 0..=n as i64 {
            for j in 0..=i {
                worst = worst.max((line.gtr(i, j) - gtr_c(&b.lam, i, j)).abs());
                worst = worst.max((line.les(j, i) - les_c(&b.lam, j, i)).abs());
                worst = worst.max((line.les(i, j) - les_c(&b.lam, i, j)).abs());
            }
            for m in 0..=40 {
                worst = worst.max((line.tv(i, m) - tv_c(&b.lam, i, m)).abs());
                worst = worst.max((line.vt(m, i) - super::super::vt_c(&b.lam, m, i)).abs());
            }
        }
        assert!(worst < 1e-9, "line read deviation {worst}");
        // separable form of the conjugated mixed component
        let terms = line.vt_nterms();
        for t in 0..=n {
            for m in 0..=40i64 {
                let mut acc = Complex::ZERO;
                for q in 0..terms {
                    let (profile, factor) = line.vt_term(q);
                    acc += factor[t] * profile[m as usize];
                }
                assert!((acc - line.vt(m, t as i64)).abs() < 1e-12, "vt term {t} {m}");
            }
        }
    }
}
