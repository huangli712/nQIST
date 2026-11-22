// Project : nQIST (real-time XCA)
// File    : src/xca_rt/fit.rs
// Purpose : complex-exponential SOE fits of real-time hybridization
//           components (prototype-validated; see mod.rs).

use super::gtr_c;
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

/// Least-squares coefficients of `values` on the pole basis
/// `exp(-i w_l t_i)` (stacked real system through the Jacobi
/// eigensolver); used both by the greedy fit and by the vt/les
/// projections of `eval2`.
/// Equilibrated least squares over the given real pole positions,
/// solved as a stacked real system through the Jacobi eigensolver.
/// Least-squares coefficients of `values` on the pole basis
/// `exp(-i w_l t_i)` (stacked real system through the Jacobi
/// eigensolver); used by the greedy fit and by the vt/les
/// projections of the evaluators.
pub(crate) fn csoe_coeffs(poles: &[f64], h: f64, values: &[Complex]) -> Vec<Complex> {
    let ns = values.len();
    let p = poles.len();
    // design A[i][l] = exp(-i w_l t_i); stack [Re A; Im A] x [c_re;
    // c_im] = [Re v; Im v]
    let dim = 2 * p;
    let mut m = vec![0.0f64; dim * dim];
    let mut rhs = vec![0.0f64; dim];
    let mut scale = vec![0.0f64; dim];
    for l in 0..p {
        let mut nr = 0.0f64;
        let mut ni = 0.0f64;
        for i in 0..ns {
            let ph = (-Complex::I * poles[l] * h * i as f64).exp();
            nr += ph.re * ph.re;
            ni += ph.im * ph.im;
        }
        let sc = (nr + ni).sqrt().max(1e-300);
        scale[l] = sc;
        scale[l + p] = sc;
    }
    // stacked real system [Ar, -Ai; Ai, Ar] [cr; ci] = [vr; vi]:
    // row factor of index a and column factor of index b
    let row = |a: usize, ph: Complex| -> (f64, f64) {
        if a < p {
            (ph.re, ph.im)
        } else {
            (-ph.im, ph.re)
        }
    };
    let col = |b: usize, ph: Complex| -> (f64, f64) {
        if b < p {
            (ph.re, ph.im)
        } else {
            (-ph.im, ph.re)
        }
    };
    for a in 0..dim {
        for b in 0..dim {
            let mut acc = 0.0;
            for i in 0..ns {
                let pa = (-Complex::I * poles[a % p] * h * i as f64).exp();
                let pb = (-Complex::I * poles[b % p] * h * i as f64).exp();
                let (are, aim) = row(a, pa);
                let (bre, bim) = col(b, pb);
                acc += are * bre + aim * bim;
            }
            m[a * dim + b] = acc / (scale[a] * scale[b]);
        }
    }
    for a in 0..dim {
        let mut acc = 0.0;
        for i in 0..ns {
            let pa = (-Complex::I * poles[a % p] * h * i as f64).exp();
            let (are, aim) = row(a, pa);
            acc += are * values[i].re + aim * values[i].im;
        }
        rhs[a] = acc / scale[a];
    }
    // solve (M + ridge) x = rhs via the Jacobi eigendecomposition
    let (w, q) = crate::xca::soe::jacobi_eigh(m, dim);
    let wmax = w.iter().fold(0.0f64, |x, y| x.max(*y)).max(1e-300);
    let ridge = 1.0e-13 * wmax;
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
    let mut coefs = Vec::with_capacity(p);
    for l in 0..p {
        let cr = x[l] / scale[l];
        let ci = x[l + p] / scale[l + p];
        coefs.push(Complex::new(cr, ci));
    }
    coefs
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
