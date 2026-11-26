// Project : nQIST (real-time XCA)
// File    : src/xca_rt/fit.rs
// Purpose : complex-exponential SOE fits of real-time hybridization
//           components (prototype-validated; see mod.rs).

use super::gtr_c;
use crate::xca::soe::{dk_roots, hankel_order, prony_annihilator};
use crate::xca::soe::jacobi_eigh;
use kadanoffbaym::complex::Complex;

/// Fit of `f(t_i) ~= sum_l c_l exp(-i w_l t_i)` on a uniform real-time
/// grid, over REAL pole frequencies (bath levels) with complex
/// coefficients: greedy pole selection + least squares, reusing the
/// Jacobi eigensolver of the Matsubara fitter via a stacked real
/// system.
#[derive(Debug, Clone)]
pub(crate) struct CsoeFit {
    /// Real pole frequencies `w_l`.
    pub poles: Vec<f64>,
    /// Complex coefficients `c_l`.
    pub coefs: Vec<Complex>,
    /// Max abs residual on the sample grid.
    pub resid: f64,
}

/// Per-component pole projections sharing one pole set: the real-time
/// line factors of every contour family are reconstructed from these.
///   Delta^>(t)   = sum_l c_gtr[l] e^{-i w_l t}
///   Delta^<(t)   = sum_l c_les[l] e^{-i w_l t}
///   Delta^tv(t,tau) = sum_l c_tv[l] e^{-i w_l t} e^{-w_l (tau - tau_ref)}
///   Delta(tau)   = sum_l c_mat[l] K(tau, w_l)
#[derive(Debug, Clone)]
pub(crate) struct CsoeProj {
    pub poles: Vec<f64>,
    pub c_gtr: Vec<Complex>,
    pub c_les: Vec<Complex>,
    /// The projection-time row-0 lesser baseline (identical to
    /// `c_les` at construction; a SEPARATE field so amplitude
    /// levers that rescale `c_les` (the eps-scaling gate) leave
    /// this data-derived baseline untouched - the per-row shape in
    /// the (3,2) chord divides by THIS, keeping the eps^m scaling
    /// exact while the row refit carries the row structure)
    pub les_row0: Vec<Complex>,
    /// The DIRECT FIT's greater coefficients (base.coefs) - the
    /// source the reference LineSep::gtr evaluates. The out-of-
    /// triangle read (u1 < n) carries the -conj() hermitian
    /// convention: ls.gtr(u1, n) = -sum_l conj(c_fit[l])
    /// e^{+iw(n-u1)h}. On near-purely-imaginary coefficients
    /// -c* ~ c (the calibration fixtures' historical blind spot);
    /// complex-coefficient fits need the exact form.
    pub c_gtr_fit: Vec<Complex>,
    /// The projection-time ROW-0 raw-greater row fit (the row basis -
    /// NOT the column-0 `c_gtr` projection, which lives on the
    /// opposite pole-sign convention). The (2,5) chord's per-row
    /// shape divides by this, mirroring `les_row0`'s role for the
    /// (3,2) chord. Data-only: never rescaled by amplitude levers
    /// (the scale flows through `c_gtr`).
    pub gtr_row0: Vec<Complex>,
    pub c_tv: Vec<Complex>,
    /// The boson tv REFLECTED mate (2026-11-29 two-mate fix): the
    /// free-boson D^tv carries TWO thermal mates sharing ONE
    /// t-phase - A_l e^{-w_l tau} (normal) and B_l e^{-w_l beta}
    /// e^{+w_l tau} (reflected) - which the single tau_ref=0
    /// column split misassigns (the legacy c_tv e^{w beta} form
    /// phase-flipped the reflected mate at large tau; instrument:
    /// XCA_TVPROBE worst 8.6e-2 at (10, beta)). Solved from the
    /// tau=0 / tau=beta column fits: st = e^{-w beta},
    /// B = (colb - col0 st)/(1 - st^2), A = col0 - B st; this field
    /// stores B_l st_l. ZERO for fermion statistics (their fixture
    /// tv is single-exponential; the fermion path stays
    /// bit-identical)
    pub c_tv_r: Vec<Complex>,
    /// The KMS-complete TWO-MATE tv amplitudes (fermion m >= 3
    /// chords): tv(t,tau) = sum_l e^{-i w_l t h}
    /// (c_tv_a[l] e^{-w_l tau} + c_tv_b[l] e^{+w_l tau}) with
    /// c_tv_b = b_l e^{-w_l beta} (the e^{w(tau-beta)} mate in
    /// mode-7 form). Equals (c_tv, 0) on single-exponential data.
    pub c_tv_a: Vec<Complex>,
    pub c_tv_b: Vec<Complex>,
    pub tau_ref: usize,
    pub c_mat: Vec<Complex>,
}

/// Fermi kernel `K(tau, w) = -e^{-w tau} / (1 + e^{-beta w})`.
#[inline]
fn fermi_kernel(tau: f64, w: f64, beta: f64) -> f64 {
    -(-w * tau).exp() / (1.0 + (-beta * w).exp())
}

thread_local! {
    static COL_VT_FITS: std::cell::RefCell<
        std::collections::HashMap<(usize, u64, i64, i64, u64, u64), Vec<Complex>>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The TAU-COLUMN fit of the vt data at the fixed real time n:
///   vt(delta, tau_m, n) = sum_l cf[l] e^{-w_l (beta - m dtau)}
/// The per-pole assignment ALIGNED with the data's separated
/// form (the (1,5) chord's factorization; the two-sided dump
/// verdict: the c_tv projection mis-assigns near-degenerate
/// poles - the same-channel totals close coincidentally, the
/// mixed combos expose it)
pub(crate) fn vt_col_fit(
    delta: &kadanoffbaym::types::ContourGreen,
    poles: &[f64],
    beta: f64,
    dtau: f64,
    ntau: i64,
    n: i64,
) -> Vec<Complex> {
    // XCA_FIT_FREEZE (diagnostic): keep the column fit at its n = 0
    // coefficients to separate the fit's time dependence from the rest
    let n = if std::env::var_os("XCA_FIT_FREEZE").is_some() { 0 } else { n };
    let key = (
        delta as *const _ as usize,
        poles
            .iter()
            .fold(0u64, |a, w| a.rotate_left(7) ^ w.to_bits()),
        n,
        ntau,
        beta.to_bits(),
        dtau.to_bits(),
    );
    COL_VT_FITS.with(|c| {
        if let Some(v) = c.borrow().get(&key) {
            return v.clone();
        }
        let np = poles.len();
        let mut b = vec![0.0f64; (ntau as usize + 1) * np];
        for j in 0..=ntau {
            for (l, &w) in poles.iter().enumerate() {
                b[j as usize * np + l] = (-w * (beta - dtau * j as f64)).exp();
            }
        }
        let values: Vec<Complex> = (0..=ntau).map(|m| super::vt_c(delta, m, n)).collect();
        let fit = project_real_basis(&b, &values);
        c.borrow_mut().insert(key, fit.clone());
        fit
    })
}

/// Real least squares of complex `values` on a REAL basis matrix
/// `b[j][l]` (solve real and imaginary parts separately through the
/// Jacobi eigensolver with the same equilibration as `csoe_coeffs`).
fn project_real_basis(b: &[f64], values: &[Complex]) -> Vec<Complex> {
    let ns = values.len();
    let p = b.len() / ns.max(1);
    let dim = p;
    let mut m = vec![0.0f64; dim * dim];
    let mut rhs_re = vec![0.0f64; dim];
    let mut rhs_im = vec![0.0f64; dim];
    let mut scale = vec![0.0f64; dim];
    for l in 0..p {
        let mut nrm = 0.0f64;
        for j in 0..ns {
            nrm += b[j * p + l] * b[j * p + l];
        }
        scale[l] = nrm.sqrt().max(1e-300);
    }
    for a in 0..p {
        for bx in 0..p {
            let mut acc = 0.0f64;
            for j in 0..ns {
                acc += b[j * p + a] * b[j * p + bx];
            }
            m[a * dim + bx] = acc / (scale[a] * scale[bx]);
        }
        let mut acre = 0.0f64;
        let mut acim = 0.0f64;
        for j in 0..ns {
            acre += b[j * p + a] * values[j].re;
            acim += b[j * p + a] * values[j].im;
        }
        rhs_re[a] = acre / scale[a];
        rhs_im[a] = acim / scale[a];
    }
    let (w, q) = jacobi_eigh(m, dim);
    let wmax = w.iter().fold(0.0f64, |x, y| x.max(*y)).max(1e-300);
    let ridge = 1.0e-13 * wmax;
    let solve = |rhs: &[f64]| -> Vec<f64> {
        let mut x = vec![0.0f64; dim];
        for k in 0..dim {
            let mut dot = 0.0;
            for a in 0..dim {
                dot += q[a * dim + k] * rhs[a];
            }
            let lam = w[k] + ridge;
            if lam.abs() > 1e-14 * wmax {
                let ck = dot / lam;
                for a in 0..dim {
                    x[a] += q[a * dim + k] * ck;
                }
            }
        }
        x
    };
    let xr = solve(&rhs_re);
    let xi = solve(&rhs_im);
    (0..p)
        .map(|l| Complex::new(xr[l] / scale[l], xi[l] / scale[l]))
        .collect()
}

/// Project every line component of a steady hybridization onto the
/// pole basis of `base` (greater fit). Reads through the SAME
/// component accessors the reference kernel uses.
#[allow(clippy::too_many_arguments)]
pub(crate) fn csoe_project_all(
    g: &kadanoffbaym::types::ContourGreen,
    base: &CsoeFit,
    beta: f64,
    dtau: f64,
    h: f64,
    nt: usize,
    ntau: usize,
) -> CsoeProj {
    // greater: ret - less* at positive differences
    let v_gtr: Vec<Complex> = (0..=nt)
        .map(|i| {
            if i == 0 {
                g.ret.raw_block_at(0, 0)[0] + g.less.raw_block_at(0, 0)[0]
            } else {
                g.ret.raw_block_at(i as i64, 0)[0] - g.less.raw_block_at(0, i as i64)[0].conj()
            }
        })
        .collect();
    // lesser at positive differences
    let v_les: Vec<Complex> = (0..=nt)
        .map(|i| {
            if i == 0 {
                g.less.raw_block_at(0, 0)[0]
            } else {
                g.less.raw_block_at(0, i as i64)[0]
            }
        })
        .collect();
    // tv at a reference Matsubara slice tau_ref = 0 (the kernel
    // factor e^{-w (tau - tau_ref)} carries the tau dependence)
    let tau_ref = 0usize;
    let v_tv: Vec<Complex> = (0..=nt)
        .map(|i| g.lmix.raw_block_at(i as i64, tau_ref as i64)[0])
        .collect();
    let c_gtr = csoe_coeffs(&base.poles, h, &v_gtr);
    let c_les = csoe_coeffs(&base.poles, h, &v_les);
    // tv: the FERMION route keeps the legacy tau_ref = 0 column fit
    // (c_tv) bit-exact - every m=1/m=2 consumer (eval1/eval2/evalm
    // and the stock-anchored m=2 family gates) depends on it. The
    // KMS-complete TWO-MATE values ride the SEPARATE c_tv_a/c_tv_b
    // fields consumed only by the m >= 3 chord arms: the TRUE tv of
    // a production bath is
    //   tv(t,tau) = sum_l e^{-i w_l t h} (a_l e^{-w_l tau} + b_l e^{w_l (tau-beta)})
    // (machine-verified 1e-13 on the dimer deltas); the legacy
    // single-column form was exact only on the fixture's
    // single-exponential fill.
    let (c_tv, c_tv_r, c_tv_a, c_tv_b) = if matches!(g.sign, kadanoffbaym::types::Sign::Bose) {
        let vals0: Vec<Complex> = (0..=nt as i64)
            .map(|t| g.lmix.raw_block_at(t, 0)[0])
            .collect();
        let valsb: Vec<Complex> = (0..=nt as i64)
            .map(|t| g.lmix.raw_block_at(t, ntau as i64)[0])
            .collect();
        let col0 = csoe_coeffs(&base.poles, h, &vals0);
        let colb = csoe_coeffs(&base.poles, h, &valsb);
        let mut a = Vec::with_capacity(base.poles.len());
        let mut br = Vec::with_capacity(base.poles.len());
        for (l, &w) in base.poles.iter().enumerate() {
            let st = (-w * beta).exp();
            let den = 1.0 - st * st;
            if den.abs() < 1e-14 {
                // w ~ 0: the mates are indistinguishable
                a.push(col0[l]);
                br.push(Complex::ZERO);
            } else {
                let b_ = (colb[l] - col0[l] * st) / den;
                let a_ = col0[l] - b_ * st;
                a.push(a_);
                br.push(b_ * st);
            }
        }
        (a.clone(), br.clone(), a, br)
    } else {
        let c_tv = csoe_coeffs(&base.poles, h, &v_tv);
        let c_tv_r = vec![Complex::ZERO; base.poles.len()];
        // the two-mate split on the SAME columns (a = c_tv, b = 0 on
        // single-exponential fixture data)
        let vals0: Vec<Complex> = (0..=nt as i64)
            .map(|t| g.lmix.raw_block_at(t, 0)[0])
            .collect();
        let valsb: Vec<Complex> = (0..=nt as i64)
            .map(|t| g.lmix.raw_block_at(t, ntau as i64)[0])
            .collect();
        let col0 = csoe_coeffs(&base.poles, h, &vals0);
        let colb = csoe_coeffs(&base.poles, h, &valsb);
        let mut a = Vec::with_capacity(base.poles.len());
        let mut br = Vec::with_capacity(base.poles.len());
        for (l, &w) in base.poles.iter().enumerate() {
            let st = (-w * beta).exp();
            let den = 1.0 - st * st;
            if den.abs() < 1e-14 {
                a.push(col0[l]);
                br.push(Complex::ZERO);
            } else {
                let b_ = (colb[l] - col0[l] * st) / den;
                let a_ = col0[l] - b_ * st;
                a.push(a_);
                br.push(b_ * st);
            }
        }
        (c_tv, c_tv_r, a, br)
    };
    // Matsubara: real kernel basis on the data's OWN statistics -
    // the FERMI kernel K for fermion deltas, the BOSON kernel B for
    // boson ones (the consumers evaluate c_mat with the matching
    // kernel: evalfam2 modes 3/4 use k_tau, modes 13/14 use
    // boson_k_tau). The legacy form projected EVERYTHING on K -
    // a boson c_mat then carried the spurious constant factor
    // (1-e^{-bw})/(1+e^{-bw}) (0.9866 at w=1, beta=5; ~1.34% per
    // mat chord, ~2.7% on the all-I cut), the residual behind the
    // j=0 les boson piece's ~6% Re deficit (instrument-anchored
    // 2026-11-29)
    let boson = matches!(g.sign, kadanoffbaym::types::Sign::Bose);
    let mut b = vec![0.0f64; (ntau + 1) * base.poles.len()];
    for j in 0..=ntau {
        for (l, &w) in base.poles.iter().enumerate() {
            b[j * base.poles.len() + l] = if boson {
                super::boson_rt::boson_k_tau(dtau * j as f64, w, beta)
            } else {
                fermi_kernel(dtau * j as f64, w, beta)
            };
        }
    }
    let v_mat: Vec<Complex> = (0..=ntau)
        .map(|j| {
            let raw = g.mat.raw();
            // mat blocks: (ntau+1) points, block index j
            raw[j]
        })
        .collect();
    let c_mat = project_real_basis(&b, &v_mat);
    let les_row0 = c_les.clone();
    let c_gtr_fit = base.coefs.clone();
    // the row-0 raw-greater row fit (see the field doc): values
    // gtr_c(g, 0, t) over the projection window
    let v_gtr_row0: Vec<Complex> = (0..=nt).map(|i| gtr_c(g, 0, i as i64)).collect();
    let gtr_row0 = csoe_coeffs(&base.poles, h, &v_gtr_row0);
    CsoeProj {
        poles: base.poles.clone(),
        c_gtr,
        c_les,
        les_row0,
        c_gtr_fit,
        gtr_row0,
        c_tv,
        c_tv_r,
        c_tv_a,
        c_tv_b,
        tau_ref,
        c_mat,
    }
}

/// Candidate pole grid: `n_w` points in `[-w_max, w_max]`, denser
/// near zero (power spacing).
fn pole_candidates(w_max: f64, n_half: usize) -> Vec<f64> {
    let mut v = Vec::new();
    for j in 0..n_half {
        let s = ((j + 1) as f64) / (n_half as f64);
        let w = w_max * s.powf(1.5);
        v.push(-w);
    }
    for j in 0..n_half {
        let s = ((j + 1) as f64) / (n_half as f64);
        let w = w_max * s.powf(1.5);
        v.push(w);
    }
    v
}

/// Solver of the stacked real least squares on a fixed pole basis and
/// window: `A[i][l] = exp(-i w_l t_i)`, `t_i = h i`, `i = 0..ns-1`,
/// `[Re A; Im A] [c_re; c_im] = [Re v; Im v]`. The design matrix is
/// decomposed once per basis by a one-sided Jacobi SVD; `apply` solves
/// for new sample values through the truncated pseudo-inverse (the
/// small singular values stay accurate, which the normal equations do
/// not give). The per-anchor projections of the evaluators (one solve
/// per Matsubara point) share one solver.
pub(crate) struct CsoeSolver {
    p: usize,
    ns: usize,
    /// columns `2 * ns` rows of the rotated design `A V` (its column
    /// `k` is `sigma_k u_k`), row-major
    ar: Vec<f64>,
    sigma: Vec<f64>,
    /// right singular vectors, column-major
    v: Vec<f64>,
    /// largest singular value
    smax: f64,
}

impl CsoeSolver {
    pub(crate) fn new(poles: &[f64], h: f64, ns: usize) -> Self {
        let p = poles.len();
        let dim = 2 * p;
        // stacked design matrix [Re A, -Im A; Im A, Re A], row-major
        let mut a = vec![0.0f64; 2 * ns * dim];
        for i in 0..ns {
            let t = h * i as f64;
            for (l, &w) in poles.iter().enumerate() {
                let ph = (-Complex::I * w * t).exp();
                a[i * dim + l] = ph.re;
                a[i * dim + p + l] = -ph.im;
                a[(ns + i) * dim + l] = ph.im;
                a[(ns + i) * dim + p + l] = ph.re;
            }
        }
        let (ar, sigma, v) = crate::xca::soe::jacobi_svd(a, 2 * ns, dim);
        let smax = sigma.iter().cloned().fold(0.0f64, f64::max);
        CsoeSolver {
            p,
            ns,
            ar,
            sigma,
            v,
            smax,
        }
    }

    /// Number of samples the solver is built for.
    pub(crate) fn ns(&self) -> usize {
        self.ns
    }

    /// Least-squares coefficients of `values` (length `ns`) on the
    /// basis: `x = V sigma^+ U^T b` with the singular values below
    /// `1e-13 sigma_max` treated as null.
    pub(crate) fn apply(&self, values: &[Complex]) -> Vec<Complex> {
        let (p, ns) = (self.p, self.ns);
        let dim = 2 * p;
        let b: Vec<f64> = values
            .iter()
            .map(|z| z.re)
            .chain(values.iter().map(|z| z.im))
            .collect();
        let mut x = vec![0.0f64; dim];
        for k in 0..dim {
            if self.sigma[k] <= 1.0e-13 * self.smax {
                continue;
            }
            let mut proj = 0.0;
            for r in 0..2 * ns {
                proj += self.ar[r * dim + k] * b[r];
            }
            // (U^T b)_k / sigma_k = (sigma_k u_k . b) / sigma_k^2
            let ck = proj / (self.sigma[k] * self.sigma[k]);
            for a in 0..dim {
                x[a] += self.v[a * dim + k] * ck;
            }
        }
        let mut coefs = Vec::with_capacity(p);
        for l in 0..p {
            coefs.push(Complex::new(x[l], x[l + p]));
        }
        coefs
    }
}

/// Least-squares coefficients of `values` on the pole basis
/// `exp(-i w_l t_i)` (a one-shot `CsoeSolver`); reuse a
/// [`CsoeSolver`] for many value vectors on the same basis and window.
pub(crate) fn csoe_coeffs(poles: &[f64], h: f64, values: &[Complex]) -> Vec<Complex> {
    CsoeSolver::new(poles, h, values.len()).apply(values)
}

/// Fit summary (coefficients plus max residual) over given poles.
fn solve_csoe_lsq(poles: &[f64], h: f64, values: &[Complex]) -> CsoeFit {
    let coefs = csoe_coeffs(poles, h, values);
    let mut resid = 0.0f64;
    for (i, &v) in values.iter().enumerate() {
        let mut acc = Complex::ZERO;
        for (l, &c) in coefs.iter().enumerate() {
            acc += c * (-Complex::I * poles[l] * h * i as f64).exp();
        }
        resid = resid.max((acc - v).abs());
    }
    CsoeFit {
        poles: poles.to_vec(),
        coefs,
        resid,
    }
}

/// Greedy complex-exponential fit with coordinate refinement.
pub(crate) fn csoe_fit(h: f64, values: &[Complex], w_max: f64, max_poles: usize) -> CsoeFit {
    let candidates = pole_candidates(w_max, 24);
    let ncand = candidates.len();
    let base_resid = values.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let mut best = CsoeFit {
        poles: Vec::new(),
        coefs: Vec::new(),
        resid: base_resid,
    };
    for _ in 0..max_poles {
        // try adding each candidate
        let mut best_gain = -1.0f64;
        let mut cand_fit: Option<CsoeFit> = None;
        for c in 0..ncand {
            if best.poles.contains(&candidates[c]) {
                continue;
            }
            let mut trial = best.poles.clone();
            trial.push(candidates[c]);
            let fit = solve_csoe_lsq(&trial, h, values);
            let resid_l2: f64 = {
                // L2 for the gain metric (consistent with the solve)
                let mut s = 0.0;
                for i in 0..values.len() {
                    let mut acc = Complex::ZERO;
                    for l in 0..trial.len() {
                        acc += fit.coefs[l] * (-Complex::I * trial[l] * h * i as f64).exp();
                    }
                    let d = (acc - values[i]).abs();
                    s += d * d;
                }
                s
            };
            let base_l2: f64 = {
                let mut s = 0.0;
                for i in 0..values.len() {
                    let mut acc = Complex::ZERO;
                    for l in 0..best.poles.len() {
                        acc += best.coefs[l] * (-Complex::I * best.poles[l] * h * i as f64).exp();
                    }
                    let d = (acc - values[i]).abs();
                    s += d * d;
                }
                s
            };
            let gain = base_l2 - resid_l2;
            if gain > best_gain {
                best_gain = gain;
                cand_fit = Some(fit);
            }
        }
        let Some(fit) = cand_fit else {
            break;
        };
        best = fit;
        // coordinate refinement of pole positions
        let mut step = 0.25f64;
        let mut sweeps = 0usize;
        while step > 1.0e-6 && sweeps < 60 {
            let mut improved = false;
            sweeps += 1;
            for pi in 0..best.poles.len() {
                for dir in [-1.0f64, 1.0] {
                    let orig = best.poles[pi];
                    let cand = orig * (1.0 + dir * step);
                    if cand.abs() > w_max {
                        continue;
                    }
                    let mut trial = best.poles.clone();
                    trial[pi] = cand;
                    let fit = solve_csoe_lsq(&trial, h, values);
                    if fit.resid < best.resid * (1.0 - 1.0e-6) {
                        best = fit;
                        improved = true;
                    } else {
                        trial[pi] = orig;
                    }
                }
            }
            if !improved {
                step *= 0.5;
            }
        }
        if best.resid < 1.0e-12 {
            break;
        }
    }
    // drop poles with negligible weight (they only cost diagram sums)
    loop {
        let cmax = best.coefs.iter().map(|c| c.abs()).fold(0.0f64, f64::max);
        let mut dropped = false;
        for pi in 0..best.poles.len() {
            if best.coefs[pi].abs() < 1.0e-6 * cmax {
                let mut trial = best.poles.clone();
                trial.remove(pi);
                if trial.is_empty() {
                    continue;
                }
                let fit = solve_csoe_lsq(&trial, h, values);
                if fit.resid <= best.resid * 10.0 {
                    best = fit;
                    dropped = true;
                    break;
                }
            }
        }
        if !dropped {
            break;
        }
    }
    best
}

// ------------------------------------------------------------------
// tolerance-driven fit
// ------------------------------------------------------------------

/// Squared L2 error of `sum_l coefs[l] exp(-i poles[l] t_i)` on the
/// sample grid.
pub(crate) fn csoe_l2(poles: &[f64], coefs: &[Complex], h: f64, values: &[Complex]) -> f64 {
    let mut s = 0.0f64;
    for (i, &v) in values.iter().enumerate() {
        let t = h * i as f64;
        let mut acc = Complex::ZERO;
        for (l, &c) in coefs.iter().enumerate() {
            acc += c * (-Complex::I * poles[l] * t).exp();
        }
        let d = acc - v;
        s += d.re * d.re + d.im * d.im;
    }
    s
}

/// Pole polish by Levenberg-Marquardt on the squared L2 residual,
/// with the residues solved by least squares at every step (variable
/// projection). The Jacobian is the derivative of the projected
/// residual (forward differences, one least-squares solve per pole),
/// so the step is the full Gauss-Newton one.
fn polish_csoe(
    mut poles: Vec<f64>,
    h: f64,
    values: &[Complex],
    w_max: f64,
    target: f64,
) -> CsoeFit {
    let p = poles.len();
    if p == 0 {
        return solve_csoe_lsq(&poles, h, values);
    }
    let ns = values.len();
    // residual of the projected model
    let residual = |poles: &[f64]| -> Vec<Complex> {
        let coefs = csoe_coeffs(poles, h, values);
        (0..ns)
            .map(|i| {
                let t = h * i as f64;
                let mut acc = Complex::ZERO;
                for (l, &c) in coefs.iter().enumerate() {
                    acc += c * (-Complex::I * poles[l] * t).exp();
                }
                acc - values[i]
            })
            .collect()
    };
    let norm2 = |r: &[Complex]| -> f64 {
        r.iter().map(|d| d.re * d.re + d.im * d.im).sum()
    };
    let mut r = residual(&poles);
    let mut f = norm2(&r);
    // stop as soon as the squared residual is below the target
    if f <= target {
        return solve_csoe_lsq(&poles, h, values);
    }
    let mut lambda = 1.0e-3f64;
    for _ in 0..200 {
        // full Jacobian of the projected residual by forward differences
        let mut jac = vec![Complex::ZERO; ns * p];
        for l in 0..p {
            let eps = 1.0e-7 * poles[l].abs().max(1.0);
            let mut up = poles.clone();
            up[l] += eps;
            let r_up = residual(&up);
            for i in 0..ns {
                jac[i * p + l] = (r_up[i] - r[i]) / eps;
            }
        }
        let mut jtj = vec![0.0f64; p * p];
        let mut jtr = vec![0.0f64; p];
        for i in 0..ns {
            for a in 0..p {
                let ja = jac[i * p + a];
                jtr[a] += ja.re * r[i].re + ja.im * r[i].im;
                for b in 0..=a {
                    let jb = jac[i * p + b];
                    jtj[a * p + b] += ja.re * jb.re + ja.im * jb.im;
                }
            }
        }
        for a in 0..p {
            for b in 0..a {
                jtj[b * p + a] = jtj[a * p + b];
            }
        }
        let diag: Vec<f64> = (0..p).map(|a| jtj[a * p + a]).collect();
        let mut accepted = false;
        for _ in 0..40 {
            // (J^T J + lambda diag) step
            let mut m = jtj.clone();
            for a in 0..p {
                m[a * p + a] += lambda * diag[a] + 1.0e-300;
            }
            let (eig, vec) = crate::xca::soe::jacobi_eigh(m, p);
            let emax = eig
                .iter()
                .cloned()
                .fold(0.0f64, f64::max)
                .max(1.0e-300);
            let mut delta = vec![0.0f64; p];
            for k in 0..p {
                if eig[k] > 1.0e-14 * emax {
                    let mut dot = 0.0;
                    for a in 0..p {
                        dot += vec[a * p + k] * jtr[a];
                    }
                    let ck = -dot / eig[k];
                    for a in 0..p {
                        delta[a] += vec[a * p + k] * ck;
                    }
                }
            }
            let trial: Vec<f64> = poles
                .iter()
                .zip(&delta)
                .map(|(x, d)| (x + d).clamp(-w_max, w_max))
                .collect();
            let r_t = residual(&trial);
            let f_t = norm2(&r_t);
            if f_t < f {
                let rel = (f - f_t) / f.max(1.0e-300);
                poles = trial;
                r = r_t;
                f = f_t;
                lambda = (lambda / 3.0).max(1.0e-15);
                accepted = true;
                if f <= target || rel < 1.0e-13 {
                    return solve_csoe_lsq(&poles, h, values);
                }
                break;
            }
            lambda *= 4.0;
            if lambda > 1.0e15 {
                return solve_csoe_lsq(&poles, h, values);
            }
        }
        if !accepted {
            break;
        }
    }
    solve_csoe_lsq(&poles, h, values)
}

/// Drop poles whose residue is below `1e-12` of the largest one (the
/// spurious roots of a Prony filter of an order above the true one)
/// and solve the residues again.
fn prune_negligible_csoe(fit: CsoeFit, h: f64, values: &[Complex]) -> CsoeFit {
    let cmax = fit.coefs.iter().map(|c| c.abs()).fold(0.0f64, f64::max);
    let keep: Vec<usize> = (0..fit.poles.len())
        .filter(|&l| fit.coefs[l].abs() > 1.0e-12 * cmax)
        .collect();
    if keep.is_empty() || keep.len() == fit.poles.len() {
        return fit;
    }
    let poles: Vec<f64> = keep.iter().map(|&l| fit.poles[l]).collect();
    solve_csoe_lsq(&poles, h, values)
}

/// Warm-started refit: residues on `poles`, a pole polish if needed;
/// `Some(fit)` when the tolerance is met (the hybridization of a DMFT
/// iteration differs slightly from the previous one, so the previous
/// poles usually stay within tolerance), `None` otherwise.
pub(crate) fn csoe_fit_warm(
    poles: &[f64],
    h: f64,
    values: &[Complex],
    w_max: f64,
    tol: f64,
) -> Option<CsoeFit> {
    if poles.is_empty() || poles.len() > values.len() {
        return None;
    }
    let fit = solve_csoe_lsq(poles, h, values);
    let target = tol * tol * values.len() as f64;
    if csoe_l2(&fit.poles, &fit.coefs, h, values) <= target {
        return Some(fit);
    }
    let fit = polish_csoe(poles.to_vec(), h, values, w_max, target);
    if csoe_l2(&fit.poles, &fit.coefs, h, values) <= target {
        return Some(fit);
    }
    None
}

/// Tolerance-driven complex-exponential fit, the real-time
/// counterpart of `xca::soe::soe_fit_tol`: the pole count is an output
/// of the tolerance `tol` (root-mean-square error on the sample grid)
/// rather than an input.
///
/// 1. order: the exponential order starts from the numerical rank of
///    the Hankel matrix of the samples and grows until the fit meets
///    `tol` (up to `max_poles`),
/// 2. poles: Prony's method - the angles inside `[-w_max, w_max]` of
///    the roots of the annihilating filter of the current order,
/// 3. residues: linear least squares on the samples, a pole polish
///    (Levenberg-Marquardt on the squared residual) whenever the
///    Prony poles are not exact, and the pruning of negligible poles.
///
/// A window of at most nine samples that Prony's method cannot resolve
/// (the first real-time steps) is interpolated exactly by evenly
/// phased frequencies. If the tolerance is still not reached the best
/// fit is returned.
pub(crate) fn csoe_fit_tol(
    h: f64,
    values: &[Complex],
    w_max: f64,
    max_poles: usize,
    tol: f64,
) -> CsoeFit {
    assert!(!values.is_empty(), "csoe_fit_tol: need at least one sample");
    assert!(w_max > 0.0, "csoe_fit_tol: w_max must be positive");
    assert!(max_poles >= 1, "csoe_fit_tol: max_poles must be >= 1");
    assert!(tol > 0.0, "csoe_fit_tol: tol must be positive");
    // a single sample is a fit of its own (zero frequency)
    if values.len() == 1 {
        return CsoeFit {
            poles: vec![0.0],
            coefs: vec![values[0]],
            resid: 0.0,
        };
    }
    let ns = values.len();
    let n = ns - 1;
    let rms = |l2: f64| (l2 / ns as f64).sqrt();
    let base_l2: f64 = values.iter().map(|v| v.re * v.re + v.im * v.im).sum();
    let mut best = CsoeFit {
        poles: Vec::new(),
        coefs: Vec::new(),
        resid: values.iter().map(|v| v.abs()).fold(0.0f64, f64::max),
    };
    let mut best_l2 = base_l2;
    if rms(base_l2) <= tol {
        return best;
    }
    let pmax = max_poles.min(n.saturating_sub(1)).max(1);
    for p in hankel_order(values, pmax)..=pmax {
        let Some(a) = prony_annihilator(values, p) else {
            break;
        };
        // the roots of an exact exponential sum sit on the unit circle
        // (real frequencies); a continuous spectrum truncates to roots
        // scattered around it - either way the angle is the pole
        let mut poles: Vec<f64> = dk_roots(&a)
            .iter()
            .filter_map(|z| {
                if !(0.5..=2.0).contains(&z.abs()) {
                    return None;
                }
                let w = -z.im.atan2(z.re) / h;
                (w.abs() <= w_max).then_some(w)
            })
            .collect();
        poles.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut merged: Vec<f64> = Vec::with_capacity(poles.len());
        for &w in &poles {
            let dup = merged
                .last()
                .map_or(false, |l| (w - l).abs() <= 1.0e-9 * (1.0 + w.abs()));
            if !dup {
                merged.push(w);
            }
        }
        if merged.is_empty() {
            continue;
        }
        let mut fit = solve_csoe_lsq(&merged, h, values);
        if rms(csoe_l2(&fit.poles, &fit.coefs, h, values)) > 1.0e-14 * rms(base_l2) {
            fit = polish_csoe(fit.poles.clone(), h, values, w_max, tol * tol * ns as f64);
        }
        fit = prune_negligible_csoe(fit, h, values);
        let l2 = csoe_l2(&fit.poles, &fit.coefs, h, values);
        if l2 < best_l2 {
            best_l2 = l2;
            best = fit;
        }
        if rms(best_l2) <= tol {
            return best;
        }
    }
    // Prony's method needs an oscillation resolved by several samples;
    // a short window (the first real-time steps) is interpolated
    // exactly by as many evenly phased frequencies as there are samples
    // (a Vandermonde system in exp(-i w h) with distinct nodes)
    if rms(best_l2) > tol && n <= 8 {
        let p = n + 1;
        let poles: Vec<f64> = (0..p)
            .map(|j| {
                std::f64::consts::PI * ((2 * j + 1) as f64 / p as f64 - 1.0) / h
            })
            .collect();
        let fit = solve_csoe_lsq(&poles, h, values);
        let l2 = csoe_l2(&fit.poles, &fit.coefs, h, values);
        if l2 < best_l2 {
            best = fit;
        }
    }
    best
}

#[cfg(test)]
mod tol_tests {
    use super::{csoe_fit_tol, csoe_l2};
    use kadanoffbaym::complex::Complex;

    fn series(h: f64, n: usize, terms: &[(f64, Complex)]) -> Vec<Complex> {
        (0..=n)
            .map(|i| {
                let t = h * i as f64;
                terms
                    .iter()
                    .fold(Complex::ZERO, |acc, &(w, c)| {
                        acc + c * (-Complex::I * w * t).exp()
                    })
            })
            .collect()
    }

    #[test]
    fn two_frequencies_recovered_exactly() {
        let (h, n) = (0.025, 24usize);
        let terms = [
            (0.5, Complex::new(0.0, -0.16)),
            (-0.3, Complex::new(0.0, -0.1)),
        ];
        let values = series(h, n, &terms);
        let fit = csoe_fit_tol(h, &values, 12.0, 24, 1.0e-9);
        let rms = (csoe_l2(&fit.poles, &fit.coefs, h, &values) / values.len() as f64).sqrt();
        assert_eq!(fit.poles.len(), 2, "{:?}", fit.poles);
        assert!((fit.poles[0] + 0.3).abs() < 1.0e-9, "{:?}", fit.poles);
        assert!((fit.poles[1] - 0.5).abs() < 1.0e-9, "{:?}", fit.poles);
        assert!(rms < 1.0e-13, "rms {rms:.3e}");
    }

    /// The phonon propagator of the Holstein dimer: two frequencies
    /// `+-omega0` with the thermal weights `n_B + 1` and `n_B`.
    #[test]
    fn phonon_propagator_two_poles() {
        let (h, n, w0, beta) = (0.025, 24usize, 0.7f64, 5.0f64);
        let nb = 1.0 / ((beta * w0).exp() - 1.0);
        let terms = [
            (w0, Complex::new(0.0, -(nb + 1.0))),
            (-w0, Complex::new(0.0, -nb)),
        ];
        let values = series(h, n, &terms);
        let fit = csoe_fit_tol(h, &values, 12.0, 24, 1.0e-9);
        let rms = (csoe_l2(&fit.poles, &fit.coefs, h, &values) / values.len() as f64).sqrt();
        assert_eq!(fit.poles.len(), 2, "{:?}", fit.poles);
        assert!((fit.poles[0] + w0).abs() < 1.0e-9, "{:?}", fit.poles);
        assert!((fit.poles[1] - w0).abs() < 1.0e-9, "{:?}", fit.poles);
        assert!((fit.coefs[0].im + nb).abs() < 1.0e-12, "{:?}", fit.coefs);
        assert!(rms < 1.0e-13, "rms {rms:.3e}");
    }

    /// Displaced poles converge to the exact ones under the polish.
    #[test]
    fn polish_recovers_displaced_poles() {
        let (h, n) = (0.1, 40usize);
        let values = series(
            h,
            n,
            &[
                (0.5, Complex::new(0.0, -0.16)),
                (-0.8, Complex::new(0.0, -0.12)),
            ],
        );
        let fit = super::polish_csoe(vec![0.48, -0.75, 1.3], h, &values, 12.0, 1.0e-22);
        let mut got = fit.poles.clone();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((got[0] + 0.8).abs() < 1.0e-9, "{got:?}");
        assert!((got[1] - 0.5).abs() < 1.0e-9, "{got:?}");
        let rms = (csoe_l2(&fit.poles, &fit.coefs, h, &values) / values.len() as f64).sqrt();
        eprintln!("polish: poles {got:?}, rms {rms:.3e}");
        assert!(rms < 1.0e-12, "rms {rms:.3e}");
    }

    /// A continuous spectrum (semicircle, Bethe lattice) is a sum of
    /// infinitely many exponentials: the pole count is an output of
    /// the tolerance.
    #[test]
    fn semicircle_reaches_tolerance_with_moderate_pole_count() {
        let (h, n) = (0.02, 300usize);
        let nq = 512;
        let mut values = vec![Complex::ZERO; n + 1];
        for q in 0..nq {
            let th = std::f64::consts::PI * (q as f64 + 0.5) / nq as f64;
            let x = 2.0 * th.cos();
            let wt = (2.0 / std::f64::consts::PI) * th.sin().powi(2)
                * (std::f64::consts::PI / nq as f64);
            let f = 1.0 / (1.0 + (5.0 * x).exp());
            for (i, v) in values.iter_mut().enumerate() {
                *v += wt * f * (-Complex::I * x * h * i as f64).exp();
            }
        }
        for tol in [1.0e-6, 1.0e-9] {
            let fit = csoe_fit_tol(h, &values, 12.0, 40, tol);
            let rms = (csoe_l2(&fit.poles, &fit.coefs, h, &values) / values.len() as f64).sqrt();
            eprintln!("semicircle tol {tol:.0e}: {} poles, rms {rms:.3e}", fit.poles.len());
            assert!(rms <= tol, "{tol:.0e}: {} poles, rms {rms:.3e}", fit.poles.len());
        }
    }
}

