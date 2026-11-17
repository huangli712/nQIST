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
fn jacobi_eigh(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
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
        if off < 1.0e-28 {
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

/// Fit `values[i]` (`tau_i = dtau * i`, `i = 0..=ntau`): greedy pole
/// selection over a dense candidate grid followed by a least-squares
/// solve on the selected subset, until the residual drops below
/// `tol` or `max_poles` is reached.
///
/// Greedy selection keeps the selected poles well separated, which
/// is what the vertex/edge separation of the diagram evaluation
/// needs: near-degenerate pole combinations (the failure mode of a
/// plain fixed-grid least squares) are amplified by the separation
/// identity and destroy the reconstruction.
#[allow(clippy::needless_range_loop)]
pub(crate) fn soe_fit(
    beta: f64,
    dtau: f64,
    values: &[Complex],
    w_max: f64,
    max_poles: usize,
) -> SoeFit {
    assert!(values.len() >= 2, "soe_fit: need at least two samples");
    assert!(w_max > 0.0, "soe_fit: w_max must be positive");
    assert!(max_poles >= 1, "soe_fit: max_poles must be >= 1");
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
        let mut poles = best.poles.clone();
        let mut step = 0.25f64;
        while step > 1.0e-6 {
            let mut improved = false;
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
                    if fit.resid_l2 < best.resid_l2 {
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
}
