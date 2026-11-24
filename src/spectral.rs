//! Equilibrium spectral functions.
//!
//! `A(ω) = -(1/π) Im G^R(ω)` from a retarded Green's function sampled
//! on a uniform real-time grid (`τ = 0, h, ..., (n-1)h`), with an
//! exponential (Lorentzian) broadening `η`:
//!
//! ```text
//! A(ω) = -(1/π) Im Σ_k w_k e^{iωτ_k} e^{-ητ_k} G^R(τ_k)
//! ```
//!
//! where `w_k` are the trapezoid weights. The samples must come from a
//! stationary (time-translation-invariant) equilibrium state: a quenched
//! or otherwise non-equilibrium state has no unique `A(ω)`. The finite
//! window `T = (n-1)h` limits the resolution to `~1/T`; keep `ηT ≳ 3`
//! to suppress the truncation ripple.
//!
//! [`spectral_from_gf`] extracts the row `G^R(t_end, t_end - k h)` from
//! a solver [`GreenTimeStep`] (as returned by
//! `ImpuritySolver::get_spgf`), so a converged equilibrium DMFT run can
//! be post-processed without storing the full two-time contour:
//!
//! ```ignore
//! let gf = solver.get_spgf(&verts, nt as i64, &pp_ints);
//! let omega = nqist::spectral::omega_grid(4.0, 321);
//! let a = nqist::spectral::spectral_from_gf(&gf[0], nt as i64, h, &omega, 0.4);
//! ```
//!
//! Exact references for small systems are available through
//! [`crate::ed::EdReference::spectral_function`].

use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::GreenTimeStep;

/// Uniform `ω` grid on `[-omega_max, omega_max]` with `n >= 2` points.
pub fn omega_grid(omega_max: f64, n: usize) -> Vec<f64> {
    assert!(n >= 2, "omega_grid: need at least two points");
    (0..n)
        .map(|k| -omega_max + 2.0 * omega_max * k as f64 / (n - 1) as f64)
        .collect()
}

/// `A(ω)` from `G^R(τ_k)`, `τ_k = k h`, with an exponential damping
/// `e^{-ητ}` (Lorentzian broadening `η`).
pub fn spectral_from_ret(samples: &[Complex], h: f64, omega: &[f64], eta: f64) -> Vec<f64> {
    assert!(h > 0.0, "spectral_from_ret: h must be positive");
    assert!(eta >= 0.0, "spectral_from_ret: eta must be non-negative");
    let n = samples.len();
    let mut out = vec![0.0f64; omega.len()];
    if n == 0 {
        return out;
    }
    for (k, &om) in omega.iter().enumerate() {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (i, g) in samples.iter().enumerate() {
            let tau = i as f64 * h;
            let w = if i == 0 || i + 1 == n { 0.5 } else { 1.0 } * h;
            let damp = (-eta * tau).exp() * w;
            let (s, c) = (om * tau).sin_cos();
            re += (g.re * c - g.im * s) * damp;
            im += (g.re * s + g.im * c) * damp;
        }
        out[k] = -im / std::f64::consts::PI;
    }
    out
}

/// `A(ω)` from the retarded row of a solver [`GreenTimeStep`] at
/// `t_end`: samples `G^R(t_end, t_end - k h)` for `k = 0..=t_end`.
pub fn spectral_from_gf(
    gf: &GreenTimeStep,
    t_end: i64,
    h: f64,
    omega: &[f64],
    eta: f64,
) -> Vec<f64> {
    assert!(t_end >= 0, "spectral_from_gf: t_end must be non-negative");
    let mut samples = Vec::with_capacity(t_end as usize + 1);
    for j in (0..=t_end).rev() {
        samples.push(gf.ret.raw_block_at(t_end, j)[0]);
    }
    spectral_from_ret(&samples, h, omega, eta)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lorentzian(x: f64, eps: f64, gamma: f64) -> f64 {
        gamma / (std::f64::consts::PI * ((x - eps).powi(2) + gamma * gamma))
    }

    fn damped_ret(n: usize, h: f64, eps: f64, gamma: f64) -> Vec<Complex> {
        (0..n)
            .map(|k| {
                let tau = k as f64 * h;
                Complex::new(0.0, -1.0)
                    * Complex::from_polar(1.0, -eps * tau)
                    * (-gamma * tau).exp()
            })
            .collect()
    }

    #[test]
    fn lorentzian_ret_transform() {
        // G^R(tau) = -i exp(-i eps tau - gamma tau)  =>  A(omega) is a
        // Lorentzian of width gamma + eta centred at eps.
        let (eps, gamma, eta, h) = (0.7f64, 0.45f64, 0.05f64, 0.01f64);
        let samples = damped_ret(4000, h, eps, gamma);
        let omega = omega_grid(3.0, 241);
        let a = spectral_from_ret(&samples, h, &omega, eta);
        let mut worst = 0.0f64;
        for (&om, &val) in omega.iter().zip(&a) {
            worst = worst.max((val - lorentzian(om, eps, gamma + eta)).abs());
        }
        println!("lorentzian transform: worst {worst:.3e}");
        assert!(worst < 2e-4, "transform error {worst:.3e}");
    }

    #[test]
    fn spectral_sum_rule_is_one() {
        // The damped transform integrates to the analytic Lorentzian
        // weight inside the omega window, (2/pi) atan(W/width); the
        // tails outside the window are the only deficit.
        let (eps, gamma, eta, h) = (0.2f64, 0.6f64, 0.1f64, 0.01f64);
        let samples = damped_ret(4000, h, eps, gamma);
        let w_max = 12.0;
        let omega = omega_grid(w_max, 4001);
        let a = spectral_from_ret(&samples, h, &omega, eta);
        let dw = omega[1] - omega[0];
        let sum: f64 = a.iter().map(|v| v * dw).sum();
        let want = 2.0 / std::f64::consts::PI * (w_max / (gamma + eta)).atan();
        println!("spectral sum rule: {sum:.6} (analytic window weight {want:.6})");
        assert!((sum - want).abs() < 2e-3, "sum rule {sum:.6} vs {want:.6}");
    }

    #[test]
    fn omega_grid_is_symmetric() {
        assert_eq!(omega_grid(2.0, 5), vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    }
}
