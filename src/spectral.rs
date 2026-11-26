//! Spectral functions from real-time Green's functions.
//!
//! Equilibrium: `A(ω) = -(1/π) Im G^R(ω)` from a retarded Green's
//! function sampled on a uniform real-time grid (`τ = 0, h, ...,\n//! (n-1)h`), with an exponential (Lorentzian) broadening `η` and an
//! optional apodization window ([`Window`]):
//!
//! ```text
//! A(ω) = -(1/π) Im Σ_k w_k H_k e^{iωτ_k} e^{-ητ_k} G^R(τ_k)
//! ```
//!
//! where `w_k` are the trapezoid weights and `H_k` the window. The
//! samples must come from a stationary (time-translation-invariant)
//! state: only then is the transform a function of `ω` alone. The finite
//! window `T = (n-1)h` limits the resolution to `~1/T`; [`auto_eta`]
//! returns the documented `3/T` broadening (`ηT ≳ 3` suppresses the
//! truncation ripple).
//!
//! Occupied / unoccupied: [`spectral_from_less`] and
//! [`spectral_from_greater`] transform the lesser/greater rows,
//! `A^<(ω) = +Im S^</π`, `A^>(ω) = -Im S^>/π`, so that
//! `A = A^< + A^>`.
//!
//! Non-equilibrium: a quenched or driven state has no unique `A(ω)`.
//! [`crate::solver::ImpuritySolver::get_spgf_full`] returns the full
//! two-time Green's function; [`wigner_spectral_from_gf`] evaluates the
//! causal Wigner spectrum
//!
//! ```text
//! A(ω, t) = -(1/π) Im Σ_j w_j H_j e^{iωτ_j} e^{-ητ_j}
//!           G^R(t + τ_j/2, t - τ_j/2),   τ_j = 2 j h >= 0,
//! ```
//!
//! [`wigner_spectral_series`] covers several times and
//! [`wigner_spectral_at`] interpolates linearly between adjacent grid
//! times. [`spectral_from_gf`] at `t_end` is the transient (single-row)
//! variant. The resolution is limited by the available relative-time
//! window and the transform still contains the switch-on transient.
//! No Padé: the spectral route is real-time only.
//!
//! The KB solver storage leaves the retarded equal-time block at zero;
//! the spectral helpers restore the physical `G^R(t,t) = -i` before the
//! transform.
//!
//! Exact references for small systems: [`crate::ed::EdReference::spectral_function`]
//! (equilibrium Lehmann) and `EdEvolution::greens_ret/les/gtr` (the
//! non-equilibrium samples).

use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::GreenTimeStep;

/// Apodization window for the finite-time transform. The weights are
/// causal: index `0` maps to `τ = 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    /// No taper; only the exponential damping `η`.
    Rect,
    /// Raised cosine `0.5 (1 + cos(π k / (n-1)))`: 1 at the first
    /// sample, 0 at the last.
    Hann,
    /// `exp(-(k/σ)^2 / 2)` with `σ = n/4`: 1 at the first sample.
    Gaussian,
}

impl Window {
    fn weight(self, k: usize, n: usize) -> f64 {
        match self {
            Window::Rect => 1.0,
            Window::Hann => {
                if n <= 1 {
                    1.0
                } else {
                    0.5 * (1.0 + (std::f64::consts::PI * k as f64 / (n - 1) as f64).cos())
                }
            }
            Window::Gaussian => {
                let sigma = 0.25 * n as f64;
                (-0.5 * (k as f64 / sigma).powi(2)).exp()
            }
        }
    }
}

/// Uniform `ω` grid on `[-omega_max, omega_max]` with `n >= 2` points.
pub fn omega_grid(omega_max: f64, n: usize) -> Vec<f64> {
    assert!(n >= 2, "omega_grid: need at least two points");
    (0..n)
        .map(|k| -omega_max + 2.0 * omega_max * k as f64 / (n - 1) as f64)
        .collect()
}

/// Broadening satisfying the `ηT ≳ 3` truncation-rejection rule.
pub fn auto_eta(t_max: f64) -> f64 {
    assert!(t_max > 0.0, "auto_eta: t_max must be positive");
    3.0 / t_max
}

/// Trapezoid integral `∫ A(ω) dω` on the given grid.
pub fn integrate_spectrum(a: &[f64], omega: &[f64]) -> f64 {
    assert_eq!(a.len(), omega.len(), "integrate_spectrum: length mismatch");
    if a.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for k in 0..a.len() - 1 {
        sum += 0.5 * (a[k] + a[k + 1]) * (omega[k + 1] - omega[k]);
    }
    sum
}

/// Imaginary part of the windowed/damped Fourier sum at every `ω`.
fn transform_im(
    samples: &[Complex],
    tau_step: f64,
    omega: &[f64],
    eta: f64,
    window: Window,
) -> Vec<f64> {
    assert!(tau_step > 0.0, "spectral: tau_step must be positive");
    assert!(eta >= 0.0, "spectral: eta must be non-negative");
    let n = samples.len();
    let mut out = vec![0.0f64; omega.len()];
    if n < 2 {
        return out;
    }
    for (k, &om) in omega.iter().enumerate() {
        let mut im = 0.0f64;
        for (i, g) in samples.iter().enumerate() {
            let tau = i as f64 * tau_step;
            let w = if i == 0 || i + 1 == n { 0.5 } else { 1.0 } * tau_step;
            let damp = (-eta * tau).exp() * w * window.weight(i, n);
            let (s, c) = (om * tau).sin_cos();
            im += (g.re * s + g.im * c) * damp;
        }
        out[k] = im;
    }
    out
}

/// Retarded row `G^R(t_end, t_end - k h)`, `k = 0..=t_end`. The KB
/// storage leaves the equal-time block zero; the physical
/// `G^R(t,t) = -i` is restored here.
fn ret_row(gf: &GreenTimeStep, t_end: i64) -> Vec<Complex> {
    let mut v: Vec<Complex> = (0..=t_end)
        .map(|k| gf.ret.raw_block_at(t_end, t_end - k)[0])
        .collect();
    v[0] = Complex::new(0.0, -1.0);
    v
}

/// Lesser row in the standard ordering `G^<(t_end, t_end - k h)`:
/// the KB `less(j, n)` stores `G^<(t_j, t_n)` with `t_j < t_n`, and
/// the anti-Hermiticity `G^<(t,t') = -G^<(t',t)*` gives the row with
/// the first time later.
fn less_row(gf: &GreenTimeStep, t_end: i64) -> Vec<Complex> {
    (0..=t_end)
        .map(|k| -gf.less.raw_block_at(t_end - k, t_end)[0].conj())
        .collect()
}

/// Greater row `G^>(t_end, t_end - k h) = G^R + G^<` (for `τ >= 0`).
fn greater_row(gf: &GreenTimeStep, t_end: i64) -> Vec<Complex> {
    ret_row(gf, t_end)
        .into_iter()
        .zip(less_row(gf, t_end))
        .map(|(r, l)| r + l)
        .collect()
}

/// `A(ω)` from `G^R(τ_k)`, `τ_k = k h`, with damping `η` and `window`.
pub fn spectral_from_ret_windowed(
    samples: &[Complex],
    h: f64,
    omega: &[f64],
    eta: f64,
    window: Window,
) -> Vec<f64> {
    transform_im(samples, h, omega, eta, window)
        .into_iter()
        .map(|v| -v / std::f64::consts::PI)
        .collect()
}

/// `A(ω)` from `G^R(τ_k)` with exponential damping only.
pub fn spectral_from_ret(samples: &[Complex], h: f64, omega: &[f64], eta: f64) -> Vec<f64> {
    spectral_from_ret_windowed(samples, h, omega, eta, Window::Rect)
}

/// Spectrum from the retarded row at `t_end` (see module docs).
pub fn spectral_from_gf_windowed(
    gf: &GreenTimeStep,
    t_end: i64,
    h: f64,
    omega: &[f64],
    eta: f64,
    window: Window,
) -> Vec<f64> {
    assert!(t_end >= 0, "spectral_from_gf: t_end must be non-negative");
    spectral_from_ret_windowed(&ret_row(gf, t_end), h, omega, eta, window)
}

/// Spectrum from the retarded row at `t_end` with exponential damping
/// only.
pub fn spectral_from_gf(
    gf: &GreenTimeStep,
    t_end: i64,
    h: f64,
    omega: &[f64],
    eta: f64,
) -> Vec<f64> {
    spectral_from_gf_windowed(gf, t_end, h, omega, eta, Window::Rect)
}

/// Occupied spectrum `A^<(ω) = +Im S^</π` from the lesser row at `t_end`.
pub fn spectral_from_less_windowed(
    gf: &GreenTimeStep,
    t_end: i64,
    h: f64,
    omega: &[f64],
    eta: f64,
    window: Window,
) -> Vec<f64> {
    assert!(t_end >= 0, "spectral_from_less: t_end must be non-negative");
    transform_im(&less_row(gf, t_end), h, omega, eta, window)
        .into_iter()
        .map(|v| v / std::f64::consts::PI)
        .collect()
}

/// Occupied spectrum with exponential damping only.
pub fn spectral_from_less(
    gf: &GreenTimeStep,
    t_end: i64,
    h: f64,
    omega: &[f64],
    eta: f64,
) -> Vec<f64> {
    spectral_from_less_windowed(gf, t_end, h, omega, eta, Window::Rect)
}

/// Unoccupied spectrum `A^>(ω) = -Im S^>/π` from the greater row at
/// `t_end` (`G^> = G^R + G^<` for `τ >= 0`).
pub fn spectral_from_greater_windowed(
    gf: &GreenTimeStep,
    t_end: i64,
    h: f64,
    omega: &[f64],
    eta: f64,
    window: Window,
) -> Vec<f64> {
    assert!(
        t_end >= 0,
        "spectral_from_greater: t_end must be non-negative"
    );
    transform_im(&greater_row(gf, t_end), h, omega, eta, window)
        .into_iter()
        .map(|v| -v / std::f64::consts::PI)
        .collect()
}

/// Unoccupied spectrum with exponential damping only.
pub fn spectral_from_greater(
    gf: &GreenTimeStep,
    t_end: i64,
    h: f64,
    omega: &[f64],
    eta: f64,
) -> Vec<f64> {
    spectral_from_greater_windowed(gf, t_end, h, omega, eta, Window::Rect)
}

/// Largest valid causal Wigner half-window at time `t`.
fn wigner_jmax(t: i64, tau_max: i64, t_max: i64) -> i64 {
    assert!(
        0 <= t && t <= t_max,
        "wigner: need 0 <= t <= t_max"
    );
    assert!(tau_max >= 1, "wigner: tau_max must be >= 1");
    let jmax = tau_max.min(t).min(t_max - t);
    assert!(jmax >= 1, "wigner: no causal window at t = {t}");
    jmax
}

fn wigner_ret_samples(full: &GreenTimeStep, t: i64, jmax: i64) -> Vec<Complex> {
    let mut v: Vec<Complex> = (0..=jmax)
        .map(|j| full.ret.raw_block_at(t + j, t - j)[0])
        .collect();
    v[0] = Complex::new(0.0, -1.0);
    v
}

fn wigner_less_samples(full: &GreenTimeStep, t: i64, jmax: i64) -> Vec<Complex> {
    (0..=jmax)
        .map(|j| -full.less.raw_block_at(t - j, t + j)[0].conj())
        .collect()
}

fn wigner_greater_samples(full: &GreenTimeStep, t: i64, jmax: i64) -> Vec<Complex> {
    wigner_ret_samples(full, t, jmax)
        .into_iter()
        .zip(wigner_less_samples(full, t, jmax))
        .map(|(r, l)| r + l)
        .collect()
}

/// Wigner `A(ω, t)` from causal samples `G^R(t + τ/2, t - τ/2)` at
/// `τ_j = 2 j h`, `j = 0..=J` (slice length `J + 1`). The caller must
/// supply the same `window` to the reference transform.
pub fn wigner_spectral_from_ret(
    samples: &[Complex],
    h: f64,
    omega: &[f64],
    eta: f64,
    window: Window,
) -> Vec<f64> {
    transform_im(samples, 2.0 * h, omega, eta, window)
        .into_iter()
        .map(|v| -v / std::f64::consts::PI)
        .collect()
}

/// Wigner `A^<(ω, t) = +Im S/π` from causal lesser samples.
pub fn wigner_spectral_from_less(
    samples: &[Complex],
    h: f64,
    omega: &[f64],
    eta: f64,
    window: Window,
) -> Vec<f64> {
    transform_im(samples, 2.0 * h, omega, eta, window)
        .into_iter()
        .map(|v| v / std::f64::consts::PI)
        .collect()
}

/// Wigner `A^>(ω, t) = -Im S/π` from causal greater samples.
pub fn wigner_spectral_from_greater(
    samples: &[Complex],
    h: f64,
    omega: &[f64],
    eta: f64,
    window: Window,
) -> Vec<f64> {
    transform_im(samples, 2.0 * h, omega, eta, window)
        .into_iter()
        .map(|v| -v / std::f64::consts::PI)
        .collect()
}

/// Wigner `A(ω, t)` from the full two-time retarded Green's function
/// ([`crate::solver::ImpuritySolver::get_spgf_full`]).
pub fn wigner_spectral_from_gf(
    full: &GreenTimeStep,
    t: i64,
    tau_max: i64,
    t_max: i64,
    h: f64,
    omega: &[f64],
    eta: f64,
    window: Window,
) -> Vec<f64> {
    let jmax = wigner_jmax(t, tau_max, t_max);
    wigner_spectral_from_ret(&wigner_ret_samples(full, t, jmax), h, omega, eta, window)
}

/// Wigner `A^<(ω, t)` from the full two-time Green's function.
pub fn wigner_less_from_gf(
    full: &GreenTimeStep,
    t: i64,
    tau_max: i64,
    t_max: i64,
    h: f64,
    omega: &[f64],
    eta: f64,
    window: Window,
) -> Vec<f64> {
    let jmax = wigner_jmax(t, tau_max, t_max);
    wigner_spectral_from_less(&wigner_less_samples(full, t, jmax), h, omega, eta, window)
}

/// Wigner `A^>(ω, t)` from the full two-time Green's function.
pub fn wigner_greater_from_gf(
    full: &GreenTimeStep,
    t: i64,
    tau_max: i64,
    t_max: i64,
    h: f64,
    omega: &[f64],
    eta: f64,
    window: Window,
) -> Vec<f64> {
    let jmax = wigner_jmax(t, tau_max, t_max);
    wigner_spectral_from_greater(
        &wigner_greater_samples(full, t, jmax),
        h,
        omega,
        eta,
        window,
    )
}

/// Wigner spectra at several grid times.
pub fn wigner_spectral_series(
    full: &GreenTimeStep,
    times: &[i64],
    tau_max: i64,
    t_max: i64,
    h: f64,
    omega: &[f64],
    eta: f64,
    window: Window,
) -> Vec<Vec<f64>> {
    times
        .iter()
        .map(|&t| wigner_spectral_from_gf(full, t, tau_max, t_max, h, omega, eta, window))
        .collect()
}

/// Wigner spectrum at a possibly non-grid time: the spectra at the two
/// adjacent grid times are computed with the common window and
/// interpolated linearly.
pub fn wigner_spectral_at(
    full: &GreenTimeStep,
    t: f64,
    tau_max: i64,
    t_max: i64,
    h: f64,
    omega: &[f64],
    eta: f64,
    window: Window,
) -> Vec<f64> {
    let tc = t.clamp(0.0, t_max as f64);
    let t0 = tc.floor() as i64;
    let t1 = (t0 + 1).min(t_max);
    if t0 == t1 {
        return wigner_spectral_from_gf(full, t0, tau_max, t_max, h, omega, eta, window);
    }
    let jmax = tau_max.min(t0).min(t_max - t1);
    assert!(jmax >= 1, "wigner_spectral_at: no causal window at t = {t}");
    let a0 = wigner_spectral_from_ret(&wigner_ret_samples(full, t0, jmax), h, omega, eta, window);
    let a1 = wigner_spectral_from_ret(&wigner_ret_samples(full, t1, jmax), h, omega, eta, window);
    let w = tc - t0 as f64;
    a0.iter()
        .zip(&a1)
        .map(|(&x, &y)| (1.0 - w) * x + w * y)
        .collect()
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
        let sum = integrate_spectrum(&a, &omega);
        let want = 2.0 / std::f64::consts::PI * (w_max / (gamma + eta)).atan();
        println!("spectral sum rule: {sum:.6} (analytic window weight {want:.6})");
        assert!((sum - want).abs() < 2e-3, "sum rule {sum:.6} vs {want:.6}");
    }

    #[test]
    fn omega_grid_is_symmetric() {
        assert_eq!(omega_grid(2.0, 5), vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn auto_eta_is_three_over_t() {
        assert!((auto_eta(8.0) - 0.375).abs() < 1e-15);
    }

    #[test]
    fn window_shapes() {
        let n = 9usize;
        assert_eq!(Window::Rect.weight(0, n), 1.0);
        assert_eq!(Window::Rect.weight(n - 1, n), 1.0);
        assert_eq!(Window::Hann.weight(0, n), 1.0);
        assert!(Window::Hann.weight(n - 1, n).abs() < 1e-15);
        assert_eq!(Window::Gaussian.weight(0, n), 1.0);
        assert!(Window::Gaussian.weight(n - 1, n) < 1e-2);
        for k in 1..n {
            assert!(Window::Gaussian.weight(k, n) <= Window::Gaussian.weight(k - 1, n));
        }
    }

    #[test]
    fn wigner_delta_weight() {
        // A single -i sample at τ = 0 (the τ spacing is 2h) samples a
        // δ(τ) retarded function -> A(ω) = h/π (the τ = 0 endpoint
        // carries the trapezoid half-weight).
        let jmax = 4usize;
        let mut samples = vec![Complex::ZERO; jmax + 1];
        samples[0] = Complex::new(0.0, -1.0);
        let a = wigner_spectral_from_ret(&samples, 0.01, &omega_grid(2.0, 5), 0.0, Window::Rect);
        for &v in &a {
            assert!((v - 0.01 / std::f64::consts::PI).abs() < 1e-12, "{v}");
        }
    }
}
