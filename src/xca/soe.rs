// Project : nQIST
// File    : src/xca/soe.rs
// Purpose : sum-of-exponentials (SOE) fit of the hybridization
//           function on the Matsubara grid for XCA diagram separation
//
// Following Huang, Golez, Strand, Kaye, SciPost Phys. 19, 121 (2025)
// (arXiv:2503.19727), Sec. III: represent Delta(tau) ~ sum_l c_l
// K(tau, w_l) with the fermionic kernel
//   K(tau, w) = -exp(-w tau) / (1 + exp(-beta w)),
// on a fixed symmetric pole grid. Coefficients come from a
// column-equilibrated least-squares fit on the equispaced Matsubara
// grid (TRIQS/xca uses an AAA-based fit; the fixed-grid least
// squares is self-contained and semi-exact for discrete baths).

use kadanoffbaym::complex::Complex;

/// Fermionic imaginary-time kernel `K(tau, w)` for `0 <= tau <= beta`.
#[inline]
pub(crate) fn k_tau(tau: f64, w: f64, beta: f64) -> f64 {
    -(-w * tau).exp() / (1.0 + (-beta * w).exp())
}

/// Symmetric pole grid `[-w_max .. -w_min, w_min .. w_max]` with
/// power-law clustering towards zero (`2 * n_half` poles).
fn pole_grid(w_max: f64, n_half: usize) -> Vec<f64> {
    let mut poles = Vec::with_capacity(2 * n_half);
    for j in 0..n_half {
        let s = ((j + 1) as f64) / (n_half as f64);
        poles.push(-w_max * s.powf(1.5));
    }
    for j in 0..n_half {
        let s = ((n_half - j) as f64) / (n_half as f64);
        poles.push(w_max * s.powf(1.5));
    }
    poles
}

/// Jacobi eigenvalue decomposition of a real symmetric `n x n` matrix
/// (consumes `a`). Returns `(values, vectors)` column-major with
/// `a = V diag(values) V^T`.
pub(crate) fn jacobi_eigh(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..50 * n.max(2) * n.max(2) {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i * n + j] * a[i * n + j];
            }
        }
        if off < 1.0e-16 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1.0e-30 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let values: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (values, v)
}

/// One SOE decomposition of a scalar imaginary-time function sampled
/// on the equispaced Matsubara grid `tau_i = dtau * i`.
#[derive(Clone, Debug)]
pub(crate) struct SoeFit {
    /// Poles `w_l` (dimensional energies).
    pub poles: Vec<f64>,
    /// Coefficients `c_l` with `f(tau) ~ sum_l c_l K(tau, w_l)`.
    pub coefs: Vec<Complex>,
    /// Maximum absolute residual of the fit on the sample grid
    /// (reported quality metric).
    pub resid: f64,
    /// L2 residual (sum of squares) of the fit — the internal
    /// selection criterion, consistent with the least-squares solve.
    pub resid_l2: f64,
    beta: f64,
}

impl SoeFit {
    /// A pole-free fit: order 1 reads the raw hybridization and never
    /// touches the fits.
    pub(crate) fn unused(beta: f64) -> SoeFit {
        SoeFit {
            poles: Vec::new(),
            coefs: Vec::new(),
            resid: 0.0,
            resid_l2: 0.0,
            beta,
        }
    }
}

impl SoeFit {
    /// Number of poles.
    pub fn len(&self) -> usize {
        self.poles.len()
    }

    /// `K^+_l(tau_i) = K(tau_i, w_l)` on grid point `i`.
    #[inline]
    pub fn kp(&self, i: usize, l: usize, dtau: f64) -> f64 {
        k_tau(dtau * i as f64, self.poles[l], self.beta)
    }

    /// `K^-_l(tau_i) = K(tau_i, -w_l)` on grid point `i`.
    #[inline]
    pub fn km(&self, i: usize, l: usize, dtau: f64) -> f64 {
        k_tau(dtau * i as f64, -self.poles[l], self.beta)
    }

    /// `K^+_l(0)`.
    #[inline]
    pub fn kp0(&self, l: usize) -> f64 {
        k_tau(0.0, self.poles[l], self.beta)
    }

    /// `K^-_l(0)`.
    #[inline]
    pub fn km0(&self, l: usize) -> f64 {
        k_tau(0.0, -self.poles[l], self.beta)
    }
}

/// Greedy SOE fit, the fallback of `soe_fit_tol` (non-real sample
/// data, or AAA yielding no usable pole): greedy pole selection over a
/// dense candidate grid followed by a least-squares solve on the
/// selected subset, until the residual drops below `1e-12` or
/// `max_poles` is reached. Pole-budget driven (no error tolerance) and
/// discontinuous in its input, which is why it is not the default.
///
/// Greedy selection keeps the selected poles well separated, which
/// is what the vertex/edge separation of the diagram evaluation
/// needs: near-degenerate pole combinations (the failure mode of a
/// plain fixed-grid least squares) are amplified by the separation
/// identity and destroy the reconstruction.
#[allow(clippy::needless_range_loop)]
fn soe_fit_greedy(
    beta: f64,
    dtau: f64,
    values: &[Complex],
    w_max: f64,
    max_poles: usize,
) -> SoeFit {
    assert!(values.len() >= 2, "soe_fit_greedy: need at least two samples");
    assert!(w_max > 0.0, "soe_fit_greedy: w_max must be positive");
    assert!(max_poles >= 1, "soe_fit_greedy: max_poles must be >= 1");
    let ntau = values.len() - 1;
    let _ = ntau;
    let candidates = pole_grid(w_max, 24);
    let ncand = candidates.len();

    // baseline: the zero fit
    let base_l2: f64 = values.iter().map(|v| v.abs() * v.abs()).sum();
    let base_resid = values.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let mut best = SoeFit {
        poles: Vec::new(),
        coefs: Vec::new(),
        resid: base_resid,
        resid_l2: base_l2,
        beta,
    };
    let tol = 1.0e-12;

    // add one pole at a time from the candidate grid, refining all
    // selected pole positions by coordinate descent after every
    // addition (a simplified bilevel scheme: the per-step refinement
    // prevents the near-degenerate pole pairs that a plain greedy
    // least squares picks and that the diagram separation amplifies)
    let mut selected: Vec<usize> = Vec::new();
    for _ in 0..max_poles {
        if best.resid < tol {
            break;
        }
        let mut best_gain = 0.0f64;
        let mut best_cand = usize::MAX;
        let mut best_fit: Option<SoeFit> = None;
        for c in 0..ncand {
            if selected.contains(&c) {
                continue;
            }
            let mut poles = best.poles.clone();
            poles.push(candidates[c]);
            let fit = solve_lsq(&poles, beta, dtau, values);
            let gain = best.resid_l2 - fit.resid_l2;
            if gain > best_gain {
                best_gain = gain;
                best_cand = c;
                best_fit = Some(fit);
            }
        }
        if best_cand == usize::MAX {
            break;
        }
        selected.push(best_cand);
        best = best_fit.expect("greedy keeps the best fit");

        // coordinate-descent refinement of all pole positions
        // (bounded: meaningful-improvement acceptance plus a sweep
        // cap - near-degenerate pole pairs otherwise wiggle forever
        // with microscopic gains, and every trial step costs a
        // least-squares solve with an eigendecomposition)
        let mut poles = best.poles.clone();
        let mut step = 0.25f64;
        let mut sweeps = 0usize;
        while step > 1.0e-6 && sweeps < 60 {
            let mut improved = false;
            sweeps += 1;
            for pi in 0..poles.len() {
                for dir in [-1.0f64, 1.0] {
                    let orig = poles[pi];
                    let mut cand = orig * (1.0 + dir * step);
                    if cand.abs() < 1.0e-6 {
                        cand = if orig >= 0.0 { 1.0e-6 } else { -1.0e-6 };
                    }
                    if cand.abs() > w_max {
                        continue;
                    }
                    poles[pi] = cand;
                    let fit = solve_lsq(&poles, beta, dtau, values);
                    if fit.resid_l2 < best.resid_l2 * (1.0 - 1.0e-6) {
                        best = fit;
                        improved = true;
                    } else {
                        poles[pi] = orig;
                    }
                }
            }
            if !improved {
                step *= 0.5;
            }
        }
        if best.resid < tol {
            break;
        }
    }

    // backward elimination of negligible poles: the greedy loop often
    // parks tiny weights on extra poles while chasing the residual
    // tolerance, and every surviving pole costs p^(m-1) in the
    // diagram sums. Drop any pole whose removal keeps the fit within
    // an order of magnitude of the achieved L2 residual.
    if !best.poles.is_empty() {
        let keep_thresh = best.resid_l2 * 10.0;
        loop {
            let mut removed = false;
            for pi in 0..best.poles.len() {
                let mut trial: Vec<f64> = best.poles.clone();
                trial.remove(pi);
                if trial.is_empty() {
                    continue;
                }
                let fit = solve_lsq(&trial, beta, dtau, values);
                if fit.resid_l2 <= keep_thresh {
                    best = fit;
                    removed = true;
                    break;
                }
            }
            if !removed {
                break;
            }
        }
    }
    best
}

/// Equilibrated least squares over explicit pole positions.
#[allow(clippy::needless_range_loop)]
fn solve_lsq(poles: &[f64], beta: f64, dtau: f64, values: &[Complex]) -> SoeFit {
    let ntau = values.len() - 1;
    let p = poles.len();
    let mut a = vec![0.0f64; (ntau + 1) * p];
    for i in 0..=ntau {
        for l in 0..p {
            a[i * p + l] = k_tau(dtau * i as f64, poles[l], beta);
        }
    }
    // column equilibration
    let mut scale = vec![0.0f64; p];
    for l in 0..p {
        let mut norm = 0.0;
        for i in 0..=ntau {
            norm += a[i * p + l] * a[i * p + l];
        }
        scale[l] = norm.sqrt().max(1.0e-300);
        for i in 0..=ntau {
            a[i * p + l] /= scale[l];
        }
    }
    // normal equations
    let mut m = vec![0.0f64; p * p];
    for u in 0..p {
        for w in u..p {
            let mut acc = 0.0;
            for i in 0..=ntau {
                acc += a[i * p + u] * a[i * p + w];
            }
            m[u * p + w] = acc;
            m[w * p + u] = acc;
        }
    }
    let mut bre = vec![0.0f64; p];
    let mut bim = vec![0.0f64; p];
    for l in 0..p {
        let mut arre = 0.0;
        let mut arim = 0.0;
        for i in 0..=ntau {
            arre += a[i * p + l] * values[i].re;
            arim += a[i * p + l] * values[i].im;
        }
        bre[l] = arre;
        bim[l] = arim;
    }
    let (eigvals, vecs) = jacobi_eigh(m, p);
    let lambda_max = eigvals.iter().cloned().fold(0.0f64, f64::max).max(1.0e-300);
    let ridge = 1.0e-13 * lambda_max;
    let solve = |b: &[f64]| -> Vec<f64> {
        let mut proj = vec![0.0f64; p];
        for l in 0..p {
            let mut acc = 0.0;
            for i in 0..p {
                acc += vecs[i * p + l] * b[i];
            }
            proj[l] = acc / (eigvals[l] + ridge);
        }
        let mut x = vec![0.0f64; p];
        for i in 0..p {
            let mut acc = 0.0;
            for l in 0..p {
                acc += vecs[i * p + l] * proj[l];
            }
            x[i] = acc;
        }
        x
    };
    let xre = solve(&bre);
    let xim = solve(&bim);
    let mut coefs = Vec::with_capacity(p);
    for l in 0..p {
        coefs.push(Complex::new(xre[l] / scale[l], xim[l] / scale[l]));
    }
    let mut resid = 0.0f64;
    let mut resid_l2 = 0.0f64;
    for i in 0..=ntau {
        let mut acc = Complex::ZERO;
        for l in 0..p {
            acc += coefs[l] * a[i * p + l] * scale[l];
        }
        resid = resid.max((acc - values[i]).abs());
        resid_l2 += {
            let d = (acc - values[i]).abs();
            d * d
        };
    }
    SoeFit {
        poles: poles.to_vec(),
        coefs,
        resid,
        resid_l2,
        beta,
    }
}

/// Linear least squares by one-sided Jacobi SVD of the column-
/// equilibrated design matrix with the singular values below
/// `cutoff * sigma_max` truncated. The dense pole grid of the fine
/// representation is ill-conditioned on purpose and the normal
/// equations cannot resolve it (accuracy `sqrt(eps) * sigma_max`).
fn solve_lsq_svd(
    poles: &[f64],
    beta: f64,
    dtau: f64,
    values: &[Complex],
    cutoff: f64,
) -> SoeFit {
    let ntau = values.len() - 1;
    let rows = ntau + 1;
    let p = poles.len();
    let mut a = vec![0.0f64; rows * p];
    let mut scale = vec![0.0f64; p];
    for l in 0..p {
        let mut norm = 0.0;
        for i in 0..rows {
            let k = k_tau(dtau * i as f64, poles[l], beta);
            a[i * p + l] = k;
            norm += k * k;
        }
        scale[l] = norm.sqrt().max(1.0e-300);
        for i in 0..rows {
            a[i * p + l] /= scale[l];
        }
    }
    let (ar, sigma, v) = jacobi_svd(a, rows, p);
    let smax = sigma.iter().cloned().fold(0.0f64, f64::max);
    // x = V diag(1 / sigma_k) U^T values, with (sigma_k U_k) the
    // columns of the rotated matrix `ar`
    let mut xre = vec![0.0f64; p];
    let mut xim = vec![0.0f64; p];
    for k in 0..p {
        if sigma[k] < cutoff * smax {
            continue;
        }
        let (mut pre, mut pim) = (0.0f64, 0.0f64);
        for i in 0..rows {
            pre += ar[i * p + k] * values[i].re;
            pim += ar[i * p + k] * values[i].im;
        }
        for l in 0..p {
            let f = v[l * p + k] / (sigma[k] * sigma[k]);
            xre[l] += f * pre;
            xim[l] += f * pim;
        }
    }
    let coefs: Vec<Complex> = (0..p)
        .map(|l| Complex::new(xre[l] / scale[l], xim[l] / scale[l]))
        .collect();
    let (mut resid, mut resid_l2) = (0.0f64, 0.0f64);
    for i in 0..rows {
        let mut acc = Complex::ZERO;
        for l in 0..p {
            acc += coefs[l] * k_tau(dtau * i as f64, poles[l], beta);
        }
        let d = (acc - values[i]).abs();
        resid = resid.max(d);
        resid_l2 += d * d;
    }
    SoeFit {
        poles: poles.to_vec(),
        coefs,
        resid,
        resid_l2,
        beta,
    }
}

/// `dK(tau, w) / dw` for `K = -exp(-w tau) / (1 + exp(-beta w))`:
/// with `g(w) = 1 / (1 + exp(beta w))`, `dK/dw = -K (tau - beta g(w))`.
#[inline]
fn dk_dw(tau: f64, w: f64, beta: f64) -> f64 {
    -k_tau(tau, w, beta) * (tau - beta / (1.0 + (beta * w).exp()))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Minimize `objective(x) = (f, grad_f, done)` over the box
/// `[lo, hi]^n` with limited-memory BFGS (two-loop recursion,
/// Armijo backtracking line search, projection onto the box). Returns
/// `(f, done)` at the final `x`.
pub(crate) fn lbfgs_box(
    x: &mut [f64],
    lo: f64,
    hi: f64,
    mem: usize,
    max_iter: usize,
    mut objective: impl FnMut(&[f64]) -> (f64, Vec<f64>, bool),
) -> (f64, bool) {
    let n = x.len();
    let (mut f, mut g, mut done) = objective(x);
    let mut s: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<Vec<f64>> = Vec::new();
    let mut rho: Vec<f64> = Vec::new();
    for _ in 0..max_iter {
        if done {
            return (f, done);
        }
        let gnorm2 = dot(&g, &g);
        if gnorm2 < 1.0e-300 {
            break;
        }
        // two-loop recursion: q = H g
        let mut q = g.clone();
        let mut alpha = vec![0.0f64; s.len()];
        for k in (0..s.len()).rev() {
            alpha[k] = rho[k] * dot(&s[k], &q);
            for i in 0..n {
                q[i] -= alpha[k] * y[k][i];
            }
        }
        if let Some(last) = s.last() {
            let ylast = &y[s.len() - 1];
            let yy = dot(ylast, ylast);
            if yy > 0.0 {
                let gamma = dot(last, ylast) / yy;
                for qi in q.iter_mut() {
                    *qi *= gamma;
                }
            }
        }
        for k in 0..s.len() {
            let beta = rho[k] * dot(&y[k], &q);
            for i in 0..n {
                q[i] += (alpha[k] - beta) * s[k][i];
            }
        }
        let mut d: Vec<f64> = g.iter().zip(&q).map(|(gi, qi)| -qi).collect();
        let mut slope = dot(&g, &d);
        if !(slope < 0.0) {
            // not a descent direction: fall back to steepest descent
            for (di, gi) in d.iter_mut().zip(&g) {
                *di = -gi;
            }
            slope = -gnorm2;
        }
        // projected backtracking line search
        let mut t = 1.0;
        let mut moved = false;
        for _ in 0..40 {
            let x_new: Vec<f64> = x
                .iter()
                .zip(&d)
                .map(|(xi, di)| (xi + t * di).clamp(lo, hi))
                .collect();
            let (f_new, g_new, done_new) = objective(&x_new);
            if f_new <= f + 1.0e-4 * t * slope {
                let sk: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
                let yk: Vec<f64> = g_new.iter().zip(g.iter()).map(|(a, b)| a - b).collect();
                let sy = dot(&sk, &yk);
                if sy > 1.0e-300 {
                    s.push(sk);
                    y.push(yk);
                    rho.push(1.0 / sy);
                    if s.len() > mem {
                        s.remove(0);
                        y.remove(0);
                        rho.remove(0);
                    }
                }
                x.copy_from_slice(&x_new);
                f = f_new;
                g = g_new;
                done = done_new;
                moved = true;
                break;
            }
            t *= 0.5;
        }
        if !moved {
            break;
        }
    }
    (f, done)
}

/// Bilevel refinement of the pole positions at fixed pole count
/// (blueprint: the bilevel optimization of Huang et al., SciPost Phys.
/// 19, 121 (2025), Sec. III.3): the residues are the linear
/// least-squares optimum for the given poles (inner level) and the pole
/// positions minimize the resulting L2 error (outer level, L-BFGS with
/// the variable-projection gradient, valid because the residues are
/// optimal). Stops as soon as the max-abs error is at or below `tol`.
fn refine_poles(
    poles: &[f64],
    beta: f64,
    dtau: f64,
    values: &[Complex],
    w_max: f64,
    tol: f64,
    max_iter: usize,
) -> SoeFit {
    let ntau = values.len() - 1;
    let mut x = poles.to_vec();
    let objective = |p: &[f64]| -> (f64, Vec<f64>, bool) {
        let fit = solve_lsq(p, beta, dtau, values);
        let mut grad = vec![0.0f64; p.len()];
        for i in 0..=ntau {
            let tau = dtau * i as f64;
            let mut re = -values[i].re;
            let mut im = -values[i].im;
            for l in 0..p.len() {
                let k = k_tau(tau, p[l], beta);
                re += fit.coefs[l].re * k;
                im += fit.coefs[l].im * k;
            }
            for l in 0..p.len() {
                let c = &fit.coefs[l];
                grad[l] += 2.0 * (re * c.re + im * c.im) * dk_dw(tau, p[l], beta);
            }
        }
        (fit.resid_l2, grad, fit.resid <= tol)
    };
    lbfgs_box(&mut x, -w_max, w_max, 8, max_iter, objective);
    solve_lsq(&x, beta, dtau, values)
}

/// Merge poles closer than `1e-3 w_max` (their kernels are nearly
/// collinear, which makes the least-squares solve ill-conditioned):
/// a merged pair is replaced by the mean of its two positions. Returns
/// the number of merges.
fn merge_close_poles(poles: &mut Vec<f64>, w_max: f64) -> usize {
    let mut merged = 0;
    loop {
        poles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut at = usize::MAX;
        let mut gap = f64::MAX;
        for i in 0..poles.len().saturating_sub(1) {
            let g = poles[i + 1] - poles[i];
            if g < gap {
                gap = g;
                at = i;
            }
        }
        if at == usize::MAX || gap >= 1.0e-3 * w_max {
            return merged;
        }
        let mean = 0.5 * (poles[at] + poles[at + 1]);
        poles.remove(at + 1);
        poles[at] = mean;
        merged += 1;
    }
}

/// Default tolerance on the root-mean-square error of the SOE fit on
/// the Matsubara grid: the pole count is grown until the fit meets it
/// (see `soe_fit_tol`).
pub(crate) const FIT_TOL_DEFAULT: f64 = 1.0e-9;

/// Positive fermionic Matsubara frequencies of the AAA sample grid
/// (blueprint: adapol `_fermionic_matsubara_frequency_grid`,
/// `|omega| <= 4 w_max`).
fn aaa_sample_frequencies(beta: f64, w_max: f64) -> Vec<f64> {
    let n_max = (4.0 * beta * w_max / std::f64::consts::PI) as usize + 1;
    (0..n_max)
        .map(|n| std::f64::consts::PI * (2 * n + 1) as f64 / beta)
        .collect()
}

pub(crate) fn cdiv(a: Complex, b: Complex) -> Complex {
    let d = b.re * b.re + b.im * b.im;
    Complex::new((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d)
}

/// Half the numerical rank of the stacked real Hankel matrix of the
/// samples: the exponential order estimate the growth loop starts
/// from (the complex rank of an exact order-p sum is p, its stacked
/// real rank 2p).
pub(crate) fn hankel_order(values: &[Complex], pmax: usize) -> usize {
    let n = values.len() - 1;
    let cols = ((n + 1) / 2).min(pmax + 2).max(2);
    if n + 1 < cols + 2 {
        return 1;
    }
    let rows = n + 1 - cols;
    let c2 = 2 * cols;
    let mut a = vec![0.0f64; 2 * rows * c2];
    for m in 0..rows {
        for k in 0..cols {
            let v = values[m + k];
            a[m * c2 + k] = v.re;
            a[m * c2 + cols + k] = -v.im;
            a[(rows + m) * c2 + k] = v.im;
            a[(rows + m) * c2 + cols + k] = v.re;
        }
    }
    let (_, sigma, _) = jacobi_svd(a, 2 * rows, c2);
    let smax = sigma
        .iter()
        .cloned()
        .fold(0.0f64, f64::max)
        .max(1.0e-300);
    let rank = sigma.iter().filter(|s| **s > 1.0e-9 * smax).count();
    ((rank + 1) / 2).clamp(1, pmax)
}

/// Prony annihilator of order `p`: a complex vector `a` with
/// `sum_{k=0..p} a_k v_{m+k} = 0` for every `m` with `m + p <= n`, the
/// null vector of the `(n - p + 1) x (p + 1)` Hankel matrix of the
/// samples. Its null space is the span of `a` and `i a`, so the null
/// vector of the stacked real matrix `[Re H, -Im H; Im H, Re H]` is
/// `a` up to a complex factor, which the roots of `sum_k a_k z^k` do
/// not see. Returns `None` when the sample count cannot support the
/// order.
pub(crate) fn prony_annihilator(values: &[Complex], p: usize) -> Option<Vec<Complex>> {
    let n = values.len() - 1;
    let rows = n.checked_sub(p)?.checked_add(1)?;
    if p == 0 || rows < 2 {
        return None;
    }
    let cols = 2 * (p + 1);
    let mut a = vec![0.0f64; 2 * rows * cols];
    for m in 0..rows {
        for k in 0..=p {
            let v = values[m + k];
            a[m * cols + k] = v.re;
            a[m * cols + p + 1 + k] = -v.im;
            a[(rows + m) * cols + k] = v.im;
            a[(rows + m) * cols + p + 1 + k] = v.re;
        }
    }
    // equilibrate the columns; the null vector transforms back with
    // the inverse scales
    let mut scale = vec![0.0f64; cols];
    for c in 0..cols {
        let norm: f64 = (0..2 * rows)
            .map(|r| {
                let x = a[r * cols + c];
                x * x
            })
            .sum();
        scale[c] = norm.sqrt().max(1.0e-300);
        for r in 0..2 * rows {
            a[r * cols + c] /= scale[c];
        }
    }
    let (_, sigma, v) = jacobi_svd(a, 2 * rows, cols);
    let k = (0..cols).min_by(|&x, &y| sigma[x].partial_cmp(&sigma[y]).unwrap())?;
    Some(
        (0..=p)
            .map(|j| {
                Complex::new(
                    v[j * cols + k] / scale[j],
                    v[(p + 1 + j) * cols + k] / scale[p + 1 + j],
                )
            })
            .collect(),
    )
}

/// All roots of `sum_k a_k z^k`: Durand-Kerner's simultaneous Newton
/// iteration on the monic form. The polynomial is normalized at
/// whichever end has the larger coefficient (a reversed polynomial
/// yields the reciprocal roots), so the roots stay of order one for a
/// filter of oscillatory or decaying exponentials.
pub(crate) fn dk_roots(a: &[Complex]) -> Vec<Complex> {
    let amax = a.iter().map(|c| c.abs()).fold(0.0f64, f64::max);
    if amax == 0.0 {
        return Vec::new();
    }
    let lo = a.iter().position(|c| c.abs() > 1.0e-13 * amax).unwrap_or(0);
    let hi = a
        .iter()
        .rposition(|c| c.abs() > 1.0e-13 * amax)
        .unwrap_or(0);
    if hi <= lo {
        return Vec::new();
    }
    let c = &a[lo..=hi];
    let m = c.len() - 1;
    let reversed = c[0].abs() > c[m].abs();
    let monic: Vec<Complex> = (0..=m)
        .map(|k| {
            if reversed {
                cdiv(c[m - k], c[0])
            } else {
                cdiv(c[k], c[m])
            }
        })
        .collect();
    let eval = |z: Complex| -> Complex {
        monic
            .iter()
            .rev()
            .fold(Complex::ZERO, |acc, &ck| acc * z + ck)
    };
    // classic off-axis starting points on a circle of radius 0.7
    let mut z: Vec<Complex> = Vec::with_capacity(m);
    let mut cur = Complex::new(0.7f64, 0.0);
    let turn = Complex::new(0.4 / 0.9849, 0.9 / 0.9849);
    for _ in 0..m {
        z.push(cur);
        cur = cur * turn;
    }
    for _ in 0..500 {
        let mut step_max = 0.0f64;
        for k in 0..m {
            let mut den = Complex::new(1.0, 0.0);
            for j in 0..m {
                if j != k {
                    den = den * (z[k] - z[j]);
                }
            }
            if den.abs() == 0.0 {
                z[k] = z[k] + Complex::new(1.0e-6 * (k + 1) as f64, 1.0e-6);
                continue;
            }
            let dq = cdiv(eval(z[k]), den);
            if !dq.re.is_finite() || !dq.im.is_finite() {
                z[k] = z[k] + Complex::new(1.0e-6 * (k + 1) as f64, 1.0e-6);
                continue;
            }
            let len = dq.abs();
            let cap = 10.0 * (1.0 + z[k].abs());
            let dq = if len > cap { dq * (cap / len) } else { dq };
            z[k] = z[k] - dq;
            step_max = step_max.max(dq.abs());
        }
        if step_max <= 1.0e-14 {
            break;
        }
    }
    if reversed {
        z.iter()
            .map(|zz| cdiv(Complex::new(1.0, 0.0), *zz))
            .collect()
    } else {
        z
    }
}

/// One-sided Jacobi SVD: orthogonalize the columns of the `rows x n`
/// matrix `a` (row-major, consumed) by right rotations. Returns the
/// rotated matrix `a V` (columns `sigma_k U_k`), the singular values
/// `sigma_k`, and the right singular vectors `V` (column-major). One-
/// sided Jacobi keeps the small singular values accurate to
/// `eps * sigma_max` in absolute terms, which the normal equations
/// `a^T a` do not (there the accuracy is `sqrt(eps) * sigma_max`).
pub(crate) fn jacobi_svd(
    mut a: Vec<f64>,
    rows: usize,
    n: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..60 {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let (mut alpha, mut beta, mut gamma) = (0.0f64, 0.0f64, 0.0f64);
                for r in 0..rows {
                    let ap = a[r * n + p];
                    let aq = a[r * n + q];
                    alpha += ap * ap;
                    beta += aq * aq;
                    gamma += ap * aq;
                }
                if alpha == 0.0
                    || beta == 0.0
                    || gamma.abs() <= 1.0e-15 * (alpha * beta).sqrt()
                {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for r in 0..rows {
                    let ap = a[r * n + p];
                    let aq = a[r * n + q];
                    a[r * n + p] = c * ap - s * aq;
                    a[r * n + q] = s * ap + c * aq;
                }
                for r in 0..n {
                    let vp = v[r * n + p];
                    let vq = v[r * n + q];
                    v[r * n + p] = c * vp - s * vq;
                    v[r * n + q] = s * vp + c * vq;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let sigma: Vec<f64> = (0..n)
        .map(|q| {
            (0..rows)
                .map(|r| a[r * n + q] * a[r * n + q])
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    (a, sigma, v)
}

/// Right singular vector of the smallest singular value of the `rows x
/// n` matrix `a` (row-major, consumed): the AAA weights are the null
/// vector of the Loewner matrix.
fn smallest_right_singular_vector(a: Vec<f64>, rows: usize, n: usize) -> Vec<f64> {
    let (_, sigma, v) = jacobi_svd(a, rows, n);
    let mut at = 0;
    let mut smallest = f64::MAX;
    for (q, s) in sigma.iter().enumerate() {
        if *s < smallest {
            smallest = *s;
            at = q;
        }
    }
    (0..n).map(|r| v[r * n + at]).collect()
}

/// Max residual of the conjugate-symmetric barycentric model
/// `(support, w)` over the samples that are not support points, and
/// the sample where it occurs (blueprint: adapol `aaa.aaa_step`).
fn aaa_worst_sample(
    nu: &[f64],
    fre: &[f64],
    fim: &[f64],
    support: &[usize],
    wre: &[f64],
    wim: &[f64],
) -> (f64, usize) {
    let (mut worst, mut at) = (-1.0f64, 0usize);
    for i in 0..nu.len() {
        if support.contains(&i) {
            continue;
        }
        // r_i = [sum_j w_j f_j / (z_i - z_j) + conj(w_j f_j) / (z_i - conj(z_j))]
        //       / [the same with f -> 1],  z = i nu
        let (mut nr, mut ni, mut dr, mut di) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (j, &sj) in support.iter().enumerate() {
            let a = -1.0 / (nu[i] - nu[sj]); // 1 / (i (nu_i - nu_j)) = a i
            let b = -1.0 / (nu[i] + nu[sj]); // 1 / (i (nu_i + nu_j)) = b i
            let (wr, wi) = (wre[j], wim[j]);
            let (fr, fi) = (fre[sj], fim[sj]);
            let pr = wr * fr - wi * fi;
            let pi = wr * fi + wi * fr;
            nr += a * (-pi) + b * pi;
            ni += a * pr + b * pr;
            dr += a * (-wi) + b * wi;
            di += a * wr + b * wr;
        }
        let (rr, ri) = if dr * dr + di * di > 0.0 {
            let d = dr * dr + di * di;
            ((nr * dr + ni * di) / d, (ni * dr - nr * di) / d)
        } else {
            (0.0, 0.0)
        };
        let res2 = (fre[i] - rr).powi(2) + (fim[i] - ri).powi(2);
        if res2 > worst {
            worst = res2;
            at = i;
        }
    }
    (worst.sqrt(), at)
}

/// One step of the conjugate-pair constrained AAA (blueprint: App. A
/// of Huang et al., SciPost Phys. 19, 121 (2025), and
/// `triqs_xca.adapol_depr.aaa.aaa_matrix_real`): the support pair
/// `(i nu, -i nu)` at the largest residual of the current model is
/// added, and the weights (conjugate on the two partners) are the
/// smallest right singular vector of the symmetrized Loewner matrix.
/// Returns the AAA residual after the update.
fn aaa_step(
    nu: &[f64],
    fre: &[f64],
    fim: &[f64],
    support: &mut Vec<usize>,
    wre: &mut Vec<f64>,
    wim: &mut Vec<f64>,
) -> f64 {
    let (_, at) = aaa_worst_sample(nu, fre, fim, support, wre, wim);
    support.push(at);
    wre.push(0.0);
    wim.push(0.0);
    let s = support.len();
    let rows = 2 * (nu.len() - s);
    if rows == 0 {
        return 0.0;
    }
    // unknowns (u_j) then (v_j) with w_j = u_j + i v_j; real part
    // equations in even rows, imaginary part equations in odd rows
    let mut a = vec![0.0f64; rows * (2 * s)];
    let mut r = 0usize;
    for i in 0..nu.len() {
        if support.contains(&i) {
            continue;
        }
        for (j, &sj) in support.iter().enumerate() {
            let aa = -1.0 / (nu[i] - nu[sj]);
            let bb = -1.0 / (nu[i] + nu[sj]);
            // L = (F_i - f_j) (aa i), R = (F_i - conj f_j) (bb i)
            let (lr, li) = (aa * (fim[sj] - fim[i]), aa * (fre[i] - fre[sj]));
            let (rr, ri) = (bb * (-fim[i] - fim[sj]), bb * (fre[i] - fre[sj]));
            a[r * (2 * s) + j] = lr + rr;
            a[r * (2 * s) + s + j] = -(li - ri);
            a[(r + 1) * (2 * s) + j] = li + ri;
            a[(r + 1) * (2 * s) + s + j] = lr - rr;
        }
        r += 2;
    }
    let w = smallest_right_singular_vector(a, rows, 2 * s);
    for j in 0..s {
        wre[j] = w[j];
        wim[j] = w[s + j];
    }
    aaa_worst_sample(nu, fre, fim, support, wre, wim).0
}

/// Poles of the barycentric model: the real roots of its denominator.
/// Each conjugate support pair contributes
/// `w_j (z + i nu_j) + conj(w_j) (z - i nu_j) = a_j z - b_j` with
/// `a_j = 2 Re w_j` and `b_j = 2 nu_j Im w_j`, so the denominator is
/// `q(z) = z A(z^2) - B(z^2)` with real polynomials `A`, `B` of degree
/// `s - 1`: `A(y) = sum_j a_j prod_{k != j} (y + nu_k^2)` and the same
/// with `b_j` for `B`. The roots come from Durand-Kerner's simultaneous
/// Newton iteration on the monic polynomial in a variable scaled by the
/// geometric mean of the support frequencies (its coefficients then
/// span a modest range). Two structures need care: a leading
/// coefficient negligible against the others marks a degree-deficient
/// model whose last root sits at infinity, and a root far outside the
/// pole window (an irrelevant zero-pole pair of the AAA model) stops
/// the convergence of the others, so such roots are deflated and the
/// remaining polynomial is solved again. The roots with an imaginary
/// part of at most `tol_im` are kept and moved to the real axis
/// (blueprint: adapol `imag_tol = 1e-4`; roots off the real axis are
/// Froissart doublets). Returns the real parts inside `[-x_max, x_max]`.
fn aaa_poles(nu: &[f64], wre: &[f64], wim: &[f64], x_max: f64, tol_im: f64) -> Vec<f64> {
    type C = (f64, f64);
    let s = nu.len();
    if s == 0 {
        return Vec::new();
    }
    let norm2 = |z: C| z.0 * z.0 + z.1 * z.1;
    // scale = geometric mean of the support frequencies
    let scale = (nu.iter().map(|n| n.ln()).sum::<f64>() / s as f64).exp();
    // coefficients of A and -B in y = (z / scale)^2, low to high
    let rho: Vec<f64> = nu.iter().map(|n| (n / scale) * (n / scale)).collect();
    let mut a = vec![0.0f64; s];
    let mut b = vec![0.0f64; s];
    for j in 0..s {
        // prod_{k != j} (y + rho_k), by repeated in-place multiplication
        let mut p = vec![0.0f64; s];
        let mut deg = 0usize;
        p[0] = 1.0;
        for (k, &r) in rho.iter().enumerate() {
            if k != j {
                p[deg + 1] = p[deg];
                for i in (1..=deg).rev() {
                    p[i] = r * p[i] + p[i - 1];
                }
                p[0] *= r;
                deg += 1;
            }
        }
        let aj = 2.0 * wre[j];
        let bj = -2.0 * (nu[j] / scale) * wim[j];
        for i in 0..s {
            a[i] += aj * p[i];
            b[i] += bj * p[i];
        }
    }
    // q(zeta) = zeta A(zeta^2) - B(zeta^2): odd coefficients from A,
    // even coefficients from -B
    let mut monic: Vec<f64> = (0..2 * s)
        .map(|k| if k % 2 == 1 { a[k / 2] } else { b[k / 2] })
        .collect();
    let cmax = monic.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if cmax == 0.0 {
        return Vec::new();
    }
    // degree-deficient model: negligible leading coefficients (their
    // roots sit at infinity)
    let mut m = (0..monic.len())
        .rev()
        .find(|&k| monic[k].abs() > 1.0e-13 * cmax)
        .unwrap();
    monic.truncate(m + 1);
    let lead = monic[m];
    for v in monic.iter_mut() {
        *v /= lead;
    }

    let mut roots: Vec<C> = Vec::new();
    for _ in 0..4 {
        roots = durand_kerner(&monic, m);
        let mut far: Vec<C> = roots
            .iter()
            .copied()
            .filter(|z| norm2(*z) > 1.0e4)
            .collect();
        far.sort_by(|x, y| norm2(*y).partial_cmp(&norm2(*x)).unwrap());
        if far.is_empty() {
            break;
        }
        for z in &far {
            if z.1.abs() <= 1.0e-12 * norm2(*z).sqrt() {
                deflate_linear(&mut monic, m, z.0);
                m -= 1;
            } else if z.1 > 0.0 {
                // deflated together with its conjugate (q has real
                // coefficients)
                deflate_quadratic(&mut monic, m, 2.0 * z.0, norm2(*z));
                m -= 2;
            }
        }
        monic.truncate(m + 1);
    }
    let mut poles: Vec<f64> = roots
        .iter()
        .filter(|z| z.1.abs() <= tol_im && z.0.abs() * scale <= x_max && z.0.is_finite())
        .map(|z| z.0 * scale)
        .collect();
    poles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    poles
}

/// All `m` roots of the monic polynomial with the coefficients `monic`
/// (low to high, `monic[m] = 1`), by Durand-Kerner's simultaneous
/// Newton iteration. A step is damped to `10 (1 + |z|)`: a root far
/// outside the others is reached in a few iterations without throwing
/// the others off.
fn durand_kerner(monic: &[f64], m: usize) -> Vec<(f64, f64)> {
    type C = (f64, f64);
    let cadd = |a: C, b: C| (a.0 + b.0, a.1 + b.1);
    let csub = |a: C, b: C| (a.0 - b.0, a.1 - b.1);
    let cmul = |a: C, b: C| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
    let cdiv = |a: C, b: C| {
        let d = b.0 * b.0 + b.1 * b.1;
        ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
    };
    let norm = |z: C| (z.0 * z.0 + z.1 * z.1).sqrt();
    let p = |z: C| -> C {
        let mut acc = (1.0f64, 0.0f64);
        for &ck in monic[..m].iter().rev() {
            acc = cadd(cmul(acc, z), (ck, 0.0));
        }
        acc
    };
    // classic off-axis starting points on a circle of radius 0.7
    let mut z: Vec<C> = Vec::with_capacity(m);
    let mut cur = (0.7f64, 0.0f64);
    for _ in 0..m {
        z.push(cur);
        cur = cmul(cur, (0.4 / 0.9849, 0.9 / 0.9849));
    }
    for _ in 0..500 {
        let mut step_max = 0.0f64;
        for k in 0..m {
            let mut den = (1.0f64, 0.0f64);
            for j in 0..m {
                if j != k {
                    den = cmul(den, csub(z[k], z[j]));
                }
            }
            let dq = cdiv(p(z[k]), den);
            if !dq.0.is_finite() || !dq.1.is_finite() {
                z[k] = cadd(z[k], (1.0e-6 * (k + 1) as f64, 1.0e-6));
                continue;
            }
            let len = norm(dq);
            let cap = 10.0 * (1.0 + norm(z[k]));
            let dq = if len > cap {
                (dq.0 * cap / len, dq.1 * cap / len)
            } else {
                dq
            };
            z[k] = csub(z[k], dq);
            step_max = step_max.max(norm(dq));
        }
        if step_max <= 1.0e-14 {
            break;
        }
    }
    z
}

/// Divide the monic polynomial `p` (coefficients low to high, degree
/// `m`) in place by `(z - r)`; the quotient takes the first `m`
/// coefficients.
fn deflate_linear(p: &mut [f64], m: usize, r: f64) {
    for k in (0..m).rev() {
        p[k] += r * p[k + 1];
    }
    p.copy_within(1..=m, 0);
}

/// Divide the monic polynomial `p` (coefficients low to high, degree
/// `m`) in place by `(z^2 - s z + t)`; the quotient takes the first
/// `m - 1` coefficients.
fn deflate_quadratic(p: &mut [f64], m: usize, s: f64, t: f64) {
    p[m - 1] += s;
    for k in (0..m - 1).rev() {
        p[k] += s * p[k + 1] - t * p[k + 2];
    }
    p.copy_within(2..=m, 0);
}
/// Drop poles whose largest contribution on the sample grid is below
/// `1e-12` of the largest one (Froissart-like real poles with
/// negligible residues), then refit the residues.
fn prune_negligible_poles(fit: SoeFit, beta: f64, dtau: f64, values: &[Complex]) -> SoeFit {
    let ntau = values.len() - 1;
    let contrib = |l: usize| -> f64 {
        let c = fit.coefs[l].abs();
        (0..=ntau)
            .map(|i| c * k_tau(dtau * i as f64, fit.poles[l], beta).abs())
            .fold(0.0f64, f64::max)
    };
    let largest = (0..fit.len()).map(contrib).fold(0.0f64, f64::max);
    let kept: Vec<usize> = (0..fit.len())
        .filter(|&l| contrib(l) > 1.0e-12 * largest)
        .collect();
    if kept.is_empty() || kept.len() == fit.len() {
        return fit;
    }
    let poles: Vec<f64> = kept.iter().map(|&l| fit.poles[l]).collect();
    solve_lsq(&poles, beta, dtau, values)
}

/// Tolerance-driven SOE fit (blueprint: Huang et al., SciPost Phys. 19,
/// 121 (2025), Sec. III and App. A; the `adapol` pipeline used by
/// TRIQS/xca). The pole count is an output of the tolerance `tol`
/// rather than an input:
///
/// 1. fine pole representation: linear least squares on a fixed dense
///    pole grid (the DLR role of the reference pipeline), solved by SVD
///    with truncation,
/// 2. pole step: conjugate-pair constrained AAA on the analytic
///    Matsubara values of the fine representation, one support pair
///    per step; its real poles are the candidate pole locations,
/// 3. residue step: linear least squares of the residues against the
///    sample grid, with negligible poles pruned, followed by a bilevel
///    polish of the pole positions (L-BFGS) for promising fits,
/// 4. steps 2-3 are repeated until the fit meets `tol`, the pole budget
///    `max_poles` is reached, or no further progress is made.
///
/// `tol` bounds the root-mean-square error on the sample grid (the
/// imaginary-time L2 norm the reference pipeline uses for `hyb_tol`);
/// `resid` reports the max-abs error.
///
/// The dense-grid least squares is accurate relative to the scale of
/// the data it fits, so steps 1-4 run in up to three rounds of residual
/// boosting: each later round fits the residual of the previous best SOE
/// on the dense grid and feeds AAA the analytic Matsubara values of
/// best SOE + residual fit.
///
/// If the best error still exceeds `tol` the best fit is returned and a
/// notice goes to stderr (once per process); non-real sample data (no
/// conjugate symmetry for the constrained AAA) falls back to the greedy
/// fit.
pub(crate) fn soe_fit_tol(
    beta: f64,
    dtau: f64,
    values: &[Complex],
    w_max: f64,
    max_poles: usize,
    tol: f64,
) -> SoeFit {
    assert!(
        values.len() >= 2,
        "soe_fit_tol: need at least two samples"
    );
    assert!(w_max > 0.0, "soe_fit_tol: w_max must be positive");
    assert!(
        max_poles >= 1,
        "soe_fit_tol: max_poles must be >= 1"
    );
    assert!(tol > 0.0, "soe_fit_tol: tol must be positive");
    let ntau = values.len() - 1;
    let base_resid = values.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let base_l2: f64 = values.iter().map(|v| v.abs() * v.abs()).sum();
    let zero = || SoeFit {
        poles: Vec::new(),
        coefs: Vec::new(),
        resid: base_resid,
        resid_l2: base_l2,
        beta,
    };
    let rms = |f: &SoeFit| (f.resid_l2 / (ntau + 1) as f64).sqrt();
    if rms(&zero()) <= tol {
        return zero();
    }
    if values.iter().any(|v| v.im != 0.0) {
        return soe_fit_greedy(beta, dtau, values, w_max, max_poles);
    }

    // analytic Matsubara values F(i omega) = sum_l c_l / (i omega - w_l)
    let eval_iw = |f: &SoeFit, om: f64| -> (f64, f64) {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for l in 0..f.len() {
            let w = f.poles[l];
            let c = &f.coefs[l];
            let d = w * w + om * om;
            re += (c.re * (-w) + c.im * om) / d;
            im += (c.im * (-w) - c.re * om) / d;
        }
        (re, im)
    };
    let grid = pole_grid(w_max, 12);
    let nu = aaa_sample_frequencies(beta, w_max);
    let mut best = zero();

    for _ in 0..6 {
        // the poles of the previous best stay in every fit of the
        // round; the AAA of the residual adds to them
        let base: Vec<f64> = best.poles.clone();
        // residual of the previous best SOE, fitted on the dense grid
        let resid: Vec<Complex> = (0..=ntau)
            .map(|i| {
                let tau = dtau * i as f64;
                let mut acc = 0.0;
                for l in 0..best.len() {
                    acc += best.coefs[l].re * k_tau(tau, best.poles[l], beta);
                }
                Complex::new(values[i].re - acc, 0.0)
            })
            .collect();
        let fine = solve_lsq_svd(&grid, beta, dtau, &resid, 1.0e-13);
        let mut fre = vec![0.0f64; nu.len()];
        let mut fim = vec![0.0f64; nu.len()];
        for (k, &om) in nu.iter().enumerate() {
            let (br, bi) = eval_iw(&best, om);
            let (fr, fi) = eval_iw(&fine, om);
            fre[k] = br + fr;
            fim[k] = bi + fi;
        }

        // grow the AAA support until the fit meets tol
        let round_start = rms(&best);
        let (mut support, mut wre, mut wim) = (Vec::new(), Vec::new(), Vec::new());
        let mut prev_aaa = f64::MAX;
        let mut stall = 0usize;
        for _ in 0..(max_poles + 1) / 2 {
            let aaa_resid = aaa_step(&nu, &fre, &fim, &mut support, &mut wre, &mut wim);
            let nu_s: Vec<f64> = support.iter().map(|&i| nu[i]).collect();
            let found = aaa_poles(&nu_s, &wre, &wim, w_max, 1.0e-4);
            // union with the poles of the previous best
            let mut poles: Vec<f64> = base.clone();
            for &w in &found {
                if !poles.iter().any(|p| (p - w).abs() <= 1.0e-9 * (1.0 + w.abs())) {
                    poles.push(w);
                }
            }
            poles.sort_by(|x, y| x.partial_cmp(y).unwrap());
            if poles.len() > max_poles {
                break;
            }
            if !poles.is_empty() {
                let mut fit = prune_negligible_poles(
                    solve_lsq(&poles, beta, dtau, values),
                    beta,
                    dtau,
                    values,
                );
                // polish only a promising fit: close to the tolerance or
                // to the best so far (no early exit at tol: a fit that
                // meets tol is polished to the least-squares optimum of
                // its pole count)
                let best_rms = rms(&best);
                if rms(&fit) <= 1.0e4 * tol || rms(&fit) <= 10.0 * best_rms {
                    fit = refine_poles(&fit.poles, beta, dtau, values, w_max, 0.0, 100);
                    if merge_close_poles(&mut fit.poles, w_max) > 0 {
                        fit = refine_poles(&fit.poles, beta, dtau, values, w_max, 0.0, 100);
                    }
                }
                if rms(&fit) < best_rms {
                    best = fit;
                    stall = 0;
                } else {
                    stall += 1;
                }
                if rms(&best) <= tol {
                    return best;
                }
            } else {
                stall += 1;
            }
            // the AAA residual stopped improving (this round's data
            // floor) or the fits stopped improving
            if aaa_resid >= prev_aaa || stall >= 3 {
                break;
            }
            prev_aaa = aaa_resid;
        }
        if rms(&best) >= 0.5 * round_start {
            break; // no substantial progress
        }
    }
    // Prony's method on the tau grid - the kernel of a pole is an
    // exponential in the grid index, `K(k dtau, w) = a(w) exp(-w dtau)^k`
    // - as the alternative pole initialization when the AAA growth
    // stalls (coarse grids, where the Matsubara values of the fine
    // representation are too noisy for the AAA): order from the
    // numerical rank of the Hankel matrix, grown until the fit meets
    // `tol`.
    if rms(&best) > tol {
        let pmax = max_poles.min(ntau / 2).max(1);
        for p in hankel_order(values, pmax)..=pmax {
            let Some(a) = prony_annihilator(values, p) else {
                break;
            };
            // real positive roots are the pole rates exp(-w dtau)
            let mut poles: Vec<f64> = dk_roots(&a)
                .iter()
                .filter_map(|z| {
                    if z.re <= 0.0 || z.im.abs() > 1.0e-8 * z.abs() {
                        return None;
                    }
                    let w = -z.re.ln() / dtau;
                    (w.abs() <= w_max).then_some(w)
                })
                .collect();
            poles.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let mut merged: Vec<f64> = Vec::with_capacity(poles.len());
            for &w in &poles {
                let dup = merged
                    .last()
                    .map_or(false, |l| (w - l).abs() <= 1.0e-6 * (1.0 + w.abs()));
                if !dup {
                    merged.push(w);
                }
            }
            if merged.is_empty() {
                continue;
            }
            let mut fit = prune_negligible_poles(
                solve_lsq(&merged, beta, dtau, values),
                beta,
                dtau,
                values,
            );
            let best_rms = rms(&best);
            if rms(&fit) <= 1.0e4 * tol || rms(&fit) <= 10.0 * best_rms {
                fit = refine_poles(&fit.poles, beta, dtau, values, w_max, 0.0, 100);
                if merge_close_poles(&mut fit.poles, w_max) > 0 {
                    fit = refine_poles(&fit.poles, beta, dtau, values, w_max, 0.0, 100);
                }
            }
            if rms(&fit) < best_rms {
                best = fit;
            }
            if rms(&best) <= tol {
                return best;
            }
        }
    }
    // last resort: the fine representation itself (the dense pole
    // grid) when no compact fit meets the tolerance - it costs more
    // poles in the diagram sums, but it is a fit like any other
    if rms(&best) > tol {
        let fine = solve_lsq_svd(&grid, beta, dtau, values, 1.0e-13);
        if rms(&fine) < rms(&best) {
            best = fine;
        }
    }
    if best.poles.is_empty() {
        return soe_fit_greedy(beta, dtau, values, w_max, max_poles);
    }
    if rms(&best) > tol {
        report_unmet_tolerance(&best, ntau, tol);
    }
    best
}

/// Report a fit that did not reach its tolerance. Once per process
/// on stderr unless `NQIST_FIT_DEBUG` is set (then on every call); the
/// best fit is returned either way.
fn report_unmet_tolerance(fit: &SoeFit, ntau: usize, tol: f64) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    let rms = (fit.resid_l2 / (ntau + 1) as f64).sqrt();
    let msg = format!(
        "nQIST: XCA hybridization fit did not reach its tolerance: rms error \
         {rms:.3e} > tol {tol:.3e} ({} poles, max-abs error {:.3e}); returning \
         the best fit. Raise the pole budget, widen the pole window, or \
         check the data for structure an SOE cannot represent.",
        fit.len(),
        fit.resid
    );
    if std::env::var_os("NQIST_FIT_DEBUG").is_some() {
        eprintln!("{msg}");
    } else {
        ONCE.call_once(|| eprintln!("{msg}"));
    }
}

/// SOE fit with the default tolerance `FIT_TOL_DEFAULT`.
pub(crate) fn soe_fit(
    beta: f64,
    dtau: f64,
    values: &[Complex],
    w_max: f64,
    max_poles: usize,
) -> SoeFit {
    soe_fit_tol(beta, dtau, values, w_max, max_poles, FIT_TOL_DEFAULT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_recovers_known_eigenpairs() {
        // [[1.5, 0.5], [0.5, 1.5]] -> eigenvalues {1, 2}
        let a = vec![1.5, 0.5, 0.5, 1.5];
        let (w, v) = jacobi_eigh(a, 2);
        let mut sorted = w.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-12, "{w:?}");
        assert!((sorted[1] - 2.0).abs() < 1e-12, "{w:?}");
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0;
                for k in 0..2 {
                    acc += v[k * 2 + i] * v[k * 2 + j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((acc - want).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn kernel_separation_identity_holds() {
        // K(t - t', w) = K(t, w) K(t', -w) / K(0, -w)  (PRX 14, 031034 eq. 22)
        let beta = 5.0;
        for w in [-2.0f64, -0.3, 0.5, 1.7] {
            let k0m = k_tau(0.0, -w, beta);
            let t1 = 1.3;
            let t2 = 0.6;
            let lhs = k_tau(t1 - t2, w, beta);
            let rhs = k_tau(t1, w, beta) * k_tau(t2, -w, beta) / k0m;
            let rel = ((lhs - rhs) / lhs).abs();
            assert!(rel < 1e-12, "w = {w}: {lhs} vs {rhs}");
        }
    }

    #[test]
    fn soe_fit_recovers_single_exponential() {
        // Delta(tau) = 0.25 K(tau, -0.7): a one-site discrete bath
        let beta = 5.0;
        let ntau = 128;
        let dtau = beta / ntau as f64;
        let values: Vec<Complex> = (0..=ntau)
            .map(|i| Complex::new(0.25 * k_tau(dtau * i as f64, -0.7, beta), 0.0))
            .collect();
        let fit = soe_fit(beta, dtau, &values, 4.0, 12);
        assert!(fit.resid < 5.0e-8, "resid = {}", fit.resid);
    }

    #[test]
    fn soe_fit_two_bath_sites() {
        // Delta(tau) = 0.09 K(tau, -1.2) + 0.16 K(tau, 0.8)
        let beta = 8.0;
        let ntau = 200;
        let dtau = beta / ntau as f64;
        let values: Vec<Complex> = (0..=ntau)
            .map(|i| {
                Complex::new(
                    0.09 * k_tau(dtau * i as f64, -1.2, beta)
                        + 0.16 * k_tau(dtau * i as f64, 0.8, beta),
                    0.0,
                )
            })
            .collect();
        let fit = soe_fit(beta, dtau, &values, 5.0, 12);
        assert!(fit.resid < 1.0e-8, "resid = {}", fit.resid);
    }

    #[test]
    fn dk_dw_matches_finite_difference() {
        let beta = 5.0;
        for w in [-3.0f64, -0.3, 0.5, 2.0] {
            for tau in [0.0f64, 0.4, 2.5, 5.0] {
                let h = 1.0e-6;
                let fd = (k_tau(tau, w + h, beta) - k_tau(tau, w - h, beta)) / (2.0 * h);
                let an = dk_dw(tau, w, beta);
                assert!(
                    (fd - an).abs() <= 1.0e-8 * (1.0 + an.abs()),
                    "w={w} tau={tau}: {fd} vs {an}"
                );
            }
        }
    }

    /// Regression: a degree-deficient AAA model. The atomic-limit
    /// Green function of the half-filled single band is exactly two
    /// poles `+- U / 2`; the second AAA step has a third denominator
    /// root that runs off to infinity, which must not disturb the two
    /// wanted ones (the first iterations of a DMFT loop fit this data).
    #[test]
    fn soe_fit_tol_recovers_exact_poles_of_a_degree_deficient_model() {
        let (beta, ntau) = (5.0, 250);
        let dtau = beta / ntau as f64;
        let values: Vec<Complex> = (0..=ntau)
            .map(|i| {
                let t = dtau * i as f64;
                Complex::new(
                    0.5 * k_tau(t, 1.75, beta) + 0.5 * k_tau(t, -1.75, beta),
                    0.0,
                )
            })
            .collect();
        let fit = soe_fit_tol(beta, dtau, &values, 12.0, 32, 1.0e-9);
        assert_eq!(fit.len(), 2, "poles {:?}", fit.poles);
        assert!((fit.poles[0] + 1.75).abs() < 1e-10, "{:?}", fit.poles);
        assert!((fit.poles[1] - 1.75).abs() < 1e-10, "{:?}", fit.poles);
        assert!(fit.resid < 1e-11, "resid {}", fit.resid);
    }

    /// Weights of a conjugate-pair barycentric model whose denominator
    /// is the given `q(zeta) = zeta A(zeta^2) - B(zeta^2)` (coefficients
    /// of `A` and `B` low to high).
    fn weights_for(nu: &[f64], a_coef: &[f64], b_coef: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let s = nu.len();
        let scale = (nu.iter().map(|n| n.ln()).sum::<f64>() / s as f64).exp();
        let rho: Vec<f64> = nu.iter().map(|n| (n / scale) * (n / scale)).collect();
        // column j: coefficients of prod_{k != j} (y + rho_k)
        let mut col = vec![vec![0.0f64; s]; s];
        for j in 0..s {
            col[j][0] = 1.0;
            let mut deg = 0usize;
            for (k, &r) in rho.iter().enumerate() {
                if k == j {
                    continue;
                }
                deg += 1;
                col[j][deg] = col[j][deg - 1];
                for i in (1..deg).rev() {
                    col[j][i] = r * col[j][i] + col[j][i - 1];
                }
                col[j][0] *= r;
            }
        }
        let solve = |rhs: &[f64]| -> Vec<f64> {
            let mut m: Vec<Vec<f64>> = (0..s)
                .map(|i| {
                    let mut row: Vec<f64> = (0..s).map(|j| col[j][i]).collect();
                    row.push(rhs[i]);
                    row
                })
                .collect();
            for i in 0..s {
                let piv = (i..s)
                    .max_by(|&x, &y| m[x][i].abs().partial_cmp(&m[y][i].abs()).unwrap())
                    .unwrap();
                m.swap(i, piv);
                for r in i + 1..s {
                    let f = m[r][i] / m[i][i];
                    for c in i..=s {
                        m[r][c] -= f * m[i][c];
                    }
                }
            }
            let mut x = vec![0.0f64; s];
            for i in (0..s).rev() {
                x[i] = (m[i][s]
                    - (i + 1..s).map(|c| m[i][c] * x[c]).sum::<f64>())
                    / m[i][i];
            }
            x
        };
        let a = solve(a_coef);
        let bt = solve(b_coef);
        let wre: Vec<f64> = a.iter().map(|v| v / 2.0).collect();
        let wim: Vec<f64> = (0..s)
            .map(|j| bt[j] / (2.0 * nu[j] / scale))
            .collect();
        (wre, wim)
    }

    /// Roots far outside the pole window (an irrelevant zero-pole pair of
    /// the AAA model) are deflated without disturbing the wanted roots:
    /// one far real root, and a far conjugate pair (the wanted roots
    /// `-2, -1, 0.5` in the scaled variable map to `-4, -2, 1` and
    /// `-2, -1, 0.5` times `32^(1/3)`).
    #[test]
    fn aaa_poles_deflates_far_roots() {
        let nu = [1.0, 4.0];
        // q = (z - 0.5)(z + 2)(z - 1e7)
        let (wre, wim) = weights_for(
            &nu,
            &[-1.0 - 1.5e7, 1.0],
            &[-1.0e7, 1.0e7 - 1.5],
        );
        let poles = aaa_poles(&nu, &wre, &wim, 10.0, 1.0e-4);
        assert_eq!(poles.len(), 2, "{poles:?}");
        assert!((poles[0] + 4.0).abs() < 1e-5, "{poles:?}");
        assert!((poles[1] - 1.0).abs() < 1e-5, "{poles:?}");

        let scale = 32.0f64.cbrt();
        let nu = [1.0, 4.0, 8.0];
        // q = (z - 0.5)(z + 2)(z + 1)(z^2 + 1e8)
        let (wre, wim) = weights_for(
            &nu,
            &[5.0e7, 1.0e8 + 0.5, 1.0],
            &[1.0e8, -2.5e8 + 1.0, -2.5],
        );
        let poles = aaa_poles(&nu, &wre, &wim, 10.0, 1.0e-4);
        assert_eq!(poles.len(), 3, "{poles:?}");
        for (got, want) in poles.iter().zip([-2.0, -1.0, 0.5]) {
            assert!((got - want * scale).abs() < 1e-5 * scale, "{poles:?}");
        }
    }

    #[test]
    fn soe_fit_tol_reaches_tolerance_on_semicircle() {
        // Delta(tau) = int rho(w) K(tau, w) dw with the semicircular
        // density rho(w) = (2/pi) sqrt(1 - w^2): a continuous bath
        let beta = 5.0;
        let ntau = 250;
        let dtau = beta / ntau as f64;
        let nq = 512;
        let values: Vec<Complex> = (0..=ntau)
            .map(|i| {
                let tau = dtau * i as f64;
                let mut acc = 0.0;
                for q in 0..nq {
                    let th = std::f64::consts::PI * (q as f64 + 0.5) / nq as f64;
                    // w = cos(th): rho(w) dw = (2/pi) sin^2(th) d(th)
                    acc += (2.0 / std::f64::consts::PI) * th.sin() * th.sin()
                        * k_tau(tau, th.cos(), beta);
                }
                Complex::new(acc * std::f64::consts::PI / nq as f64, 0.0)
            })
            .collect();
        let fit = soe_fit_tol(beta, dtau, &values, 6.0, 40, 1.0e-9);
        let rms = (fit.resid_l2 / (ntau + 1) as f64).sqrt();
        println!(
            "semicircle: {} poles, rms {:.3e}, max-abs err {:.3e}",
            fit.len(),
            rms,
            fit.resid
        );
        assert!(rms <= 1.0e-9, "{} poles, rms {rms:.3e}", fit.len());
    }

    #[test]
    fn smallest_right_singular_vector_finds_null_vector() {
        // columns: c3 = c1 + c2 -> the smallest right singular vector
        // is +-(1, 1, -1) / sqrt(3)
        let a = vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 9.0, //
            2.0, 2.0, 4.0, //
            -1.0, 0.5, -0.5,
        ];
        let v = smallest_right_singular_vector(a, 4, 3);
        let s = 1.0 / 3.0f64.sqrt();
        let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        for (x, want) in v.iter().zip([s, s, -s]) {
            assert!((sign * x - want).abs() < 1.0e-12, "v = {v:?}");
        }
    }

    #[test]
    fn aaa_recovers_poles_of_a_sum_of_poles() {
        // F(z) = 0.2 / (z - 0.3) + 0.5 / (z + 1.1): the Matsubara form of
        // Delta(tau) = 0.2 K(tau, 0.3) + 0.5 K(tau, -1.1)
        let beta = 5.0;
        let want = [0.3f64, -1.1];
        let residues = [0.2f64, 0.5];
        let nu = aaa_sample_frequencies(beta, 4.0);
        let mut fre = vec![0.0f64; nu.len()];
        let mut fim = vec![0.0f64; nu.len()];
        for (k, &om) in nu.iter().enumerate() {
            for (p, r) in want.iter().zip(residues) {
                let d = p * p + om * om;
                fre[k] += r * (-p) / d;
                fim[k] += r * (-om) / d;
            }
        }
        let (mut support, mut wre, mut wim) = (Vec::new(), Vec::new(), Vec::new());
        let mut aaa_resid = f64::MAX;
        for _ in 0..2 {
            aaa_resid = aaa_step(&nu, &fre, &fim, &mut support, &mut wre, &mut wim);
        }
        assert!(aaa_resid < 1.0e-10, "AAA residual {aaa_resid:.3e}");
        let nu_s: Vec<f64> = support.iter().map(|&i| nu[i]).collect();
        let found = aaa_poles(&nu_s, &wre, &wim, 4.0, 1.0e-4);
        assert!(found.len() >= 2, "found {found:?}");
        // each genuine pole is found; extra real poles (Froissart-like
        // pole-zero pairs) are pruned by the pipeline
        for b in want {
            assert!(
                found.iter().any(|a| (a - b).abs() < 1.0e-8),
                "found {found:?}, want {want:?}"
            );
        }
    }

    #[test]
    fn soe_fit_tol_returns_exact_poles_for_discrete_bath() {
        // Delta(tau) = 0.09 K(tau, -1.2) + 0.16 K(tau, 0.8): no extra
        // poles beyond the two bath sites
        let beta = 8.0;
        let ntau = 200;
        let dtau = beta / ntau as f64;
        let values: Vec<Complex> = (0..=ntau)
            .map(|i| {
                Complex::new(
                    0.09 * k_tau(dtau * i as f64, -1.2, beta)
                        + 0.16 * k_tau(dtau * i as f64, 0.8, beta),
                    0.0,
                )
            })
            .collect();
        let fit = soe_fit_tol(beta, dtau, &values, 5.0, 12, 1.0e-9);
        let mut poles = fit.poles.clone();
        poles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "discrete bath: {} poles {poles:?}, resid {:.3e}, coefs {:?}",
            fit.len(),
            fit.resid,
            fit.coefs
        );
        assert_eq!(fit.len(), 2, "poles {poles:?}");
        assert!((poles[0] + 1.2).abs() < 1.0e-6 && (poles[1] - 0.8).abs() < 1.0e-6);
        assert!(fit.resid < 1.0e-12, "resid {}", fit.resid);
    }
}

