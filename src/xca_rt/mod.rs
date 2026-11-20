// Project : nQIST (real-time XCA, formal implementation)
// File    : src/xca_rt/mod.rs
// Purpose : Production module for the real-time (Kadanoff-Baym)
//           extension of the XCA solver. Component conventions and
//           the validated prototype results live in FEASIBILITY.md
//           (parent directory of the crate).
//
//           Layered as:
//             fit.rs   - complex-exponential SOE fits of the
//                        hybridization components,
//             quad.rs  - the reference quadrature helpers
//                        (conv2b / poly sub-interval / Matsubara
//                        lowside) mirrored exactly,
//             eval2.rs - the order-2 five-component-family separated
//                        evaluator, calibrated against
//                        oca::add_sigma_tstp_oca,
//             evalm.rs - the arbitrary-order all-greater family
//                        (chain evaluation; the structure that
//                        generalizes).
//
//           The pseudo-particle ret slot stores the GREATER
//           component; `sgtr = ret - less*` reconstructs the
//           greater of a ContourGreen.

mod boson_rt;
pub(crate) mod eval1;
pub(crate) mod eval2;
pub(crate) mod eval2m;
pub(crate) mod evalfam;
pub(crate) mod evalfam2;
pub(crate) mod evalfam2m;
pub(crate) mod evalm;
pub(crate) mod fit;
pub(crate) mod quad;
#[cfg(test)]
mod tests;

pub(crate) use eval2::add_sigma_tstp_xca2;
pub(crate) use fit::{CsoeFit, csoe_fit};

/// Greater-component fits of every channel on the full real-time
/// grid (the fit window must cover the evaluation window).
pub(crate) fn fit_channels_rt_ext(
    pp_ints: &[crate::nca::Hybridization],
    h: f64,
    nt: usize,
    max_poles: usize,
    w_max: f64,
) -> Vec<CsoeFit> {
    pp_ints
        .iter()
        .map(|x| fit_one_delta_rt_ext(x.delta, h, nt, max_poles, w_max))
        .collect()
}

pub(crate) fn fit_channels_rt(
    pp_ints: &[crate::nca::Hybridization],
    h: f64,
    nt: usize,
    max_poles: usize,
    w_max: f64,
) -> Vec<CsoeFit> {
    pp_ints
        .iter()
        .map(|hyb| fit_one_delta_rt(hyb.delta, h, nt, max_poles, w_max))
        .collect()
}

/// SOE fit of a single delta (fermion or boson). POLE DISCOVERY
/// TARGET: the LESSER component when it carries meaningful weight -
/// for a discrete bath each bath pole contributes exactly ONE
/// exponential there (no f-mixed conjugates), so the greedy
/// separates the physical poles cleanly. The sgtr (ret - less*) of
/// a multi-pole bath is a 2x-component MIXTURE (physical + f-mixed
/// conjugates) whose greedy fit collapses to a near-degenerate
/// compromise with cancelling coefficients - an ill-conditioned
/// reconstruction that amplifies through the I-axis kernels (the
/// historical "2-pole representativity residual" is largely this
/// pathology). Below the amplitude threshold (clean high-epsilon
/// fermion baths, bosons with tiny n_B) the sgtr path is kept -
/// it is well-conditioned there and all validated anchors depend
/// on it.
/// Plain sgtr-only fit (the historical behavior - the per-row
/// LineSeg projections of the eval1/eval2 routes are well
/// conditioned even on compromise poles; the pathology matters
/// only for the pole-separated BLOCK-CUT reconstruction, whose
/// consumers use the _ext variant).
pub(crate) fn fit_one_delta_rt(
    g: &kadanoffbaym::types::ContourGreen,
    h: f64,
    nt: usize,
    max_poles: usize,
    w_max: f64,
) -> CsoeFit {
    let v_gtr: Vec<Complex> = (0..=nt as i64)
        .map(|i| {
            if i == 0 {
                g.ret.raw_block_at(0, 0)[0] + g.less.raw_block_at(0, 0)[0]
            } else {
                g.ret.raw_block_at(i, 0)[0] - g.less.raw_block_at(0, i)[0].conj()
            }
        })
        .collect();
    csoe_fit(h, &v_gtr, w_max, max_poles)
}

pub(crate) fn fit_one_delta_rt_ext(
    g: &kadanoffbaym::types::ContourGreen,
    h: f64,
    nt: usize,
    max_poles: usize,
    w_max: f64,
) -> CsoeFit {
    let v_gtr: Vec<Complex> = (0..=nt as i64)
        .map(|i| {
            if i == 0 {
                g.ret.raw_block_at(0, 0)[0] + g.less.raw_block_at(0, 0)[0]
            } else {
                g.ret.raw_block_at(i, 0)[0] - g.less.raw_block_at(0, i)[0].conj()
            }
        })
        .collect();
    let v_les: Vec<Complex> = (0..=nt as i64)
        .map(|i| g.less.raw_block_at(0, i)[0])
        .collect();
    let fit_gtr = csoe_fit(h, &v_gtr, w_max, max_poles);
    // pathology detector: the sgtr greedy collapsed to a
    // near-degenerate pole pair (the multi-pole mixture compromise)
    let degenerate = fit_gtr.poles.windows(2).any(|w| (w[0] - w[1]).abs() < 0.05);
    if degenerate {
        let fit_les = csoe_fit(h, &v_les, w_max, max_poles);
        // PRUNE THE RESIDUAL-FLOOR CHASERS (the 2026-11-29
        // conditioning fix): the greedy keeps a few extra poles to
        // chase its refinement floor even on clean data (the twin
        // fixture: a 2-bath-pole lesser kept 4 poles). Those
        // spurious poles carry negligible LESSER weight but their
        // negations create near-degenerate pairs in the extended
        // set (0.017 apart observed) whose sgtr projection needs
        // cancelling O(10) coefficients - the ill-conditioning
        // behind the twin anchors' residual. Keep only poles with
        // meaningful lesser amplitude before extending.
        let cmax_l = fit_les.coefs.iter().map(|c| c.abs()).fold(0.0f64, f64::max);
        let poles_les: Vec<f64> = fit_les
            .poles
            .iter()
            .zip(fit_les.coefs.iter())
            .filter(|(_, c)| c.abs() > 1.0e-3 * cmax_l)
            .map(|(&w, _)| w)
            .collect();
        // the physical poles from the lesser cover the lesser/tv
        // structures; the GREATER carries the f-mixed conjugates at
        // the NEGATED poles - extend the set with the conjugates
        // (analytically known) so every component projects
        // faithfully
        let mut poles = poles_les.clone();
        for &w in &poles_les {
            if !poles.iter().any(|&p| (p + w).abs() < 1e-8) {
                poles.push(-w);
            }
        }
        // CONSISTENT GREATER COEFFICIENTS (the 2026-11-29 fit-layer
        // fix): the struct's coefs ARE the greater coefficients by
        // contract (LineSep::gtr_raw in eval1/eval2 and evalm's
        // greater chords consume them directly). Returning
        // fit_les.coefs - the LESSER fit's coefficients, and SHORTER
        // than the extended poles - made gtr_raw a poor
        // reconstruction on every degenerate-branch channel (the
        // twin-fixture FAM class-split root cause: truncated
        // lesser coefficients phased as greater) and left the
        // struct inconsistent (poles.len() != coefs.len()). Project
        // the ACTUAL sgtr column onto the extended set instead -
        // the same LSQ csoe_project_all performs for c_gtr, so the
        // LineSep and block-cut reconstructions agree by
        // construction.
        let coefs = fit::csoe_coeffs(&poles, h, &v_gtr);
        let mut resid = 0.0f64;
        for (i, &val) in v_gtr.iter().enumerate() {
            let mut acc = Complex::ZERO;
            for (l, &c) in coefs.iter().enumerate() {
                acc += c * (-Complex::I * poles[l] * h * i as f64).exp();
            }
            let d = val - acc;
            resid += d.re * d.re + d.im * d.im;
        }
        CsoeFit {
            poles,
            coefs,
            resid: resid.sqrt(),
        }
    } else {
        fit_gtr
    }
}

/// Greater component accessor identical to the reference `psc`
pub(crate) fn gtr(pp: &PseudoGreen, n: i64, j: i64) -> Complex {
    if j <= n {
        pp.ret.raw_block_at(n, j)[0]
    } else {
        -pp.ret.raw_block_at(j, n)[0].conj()
    }
}

/// Greater of a ContourGreen (`sgtr` semantics with the hermitian
/// fallback out of triangle).
pub(crate) fn gtr_c(g: &ContourGreen, n: i64, j: i64) -> Complex {
    if j < n {
        g.ret.raw_block_at(n, j)[0] - g.less.raw_block_at(j, n)[0].conj()
    } else if j == n {
        g.ret.raw_block_at(n, n)[0] + g.less.raw_block_at(n, n)[0]
    } else {
        -g.ret.raw_block_at(j, n)[0].conj() + g.less.raw_block_at(n, j)[0]
    }
}

/// Lesser of a ContourGreen (`sles` semantics).
pub(crate) fn les_c(g: &ContourGreen, j: i64, n: i64) -> Complex {
    if j <= n {
        g.less.raw_block_at(j, n)[0]
    } else {
        -g.less.raw_block_at(n, j)[0].conj()
    }
}

/// Lesser of a PseudoGreen (`pls` semantics).
pub(crate) fn les_pp(pp: &PseudoGreen, j: i64, n: i64) -> Complex {
    if j <= n {
        pp.less.raw_block_at(j, n)[0]
    } else {
        -pp.less.raw_block_at(n, j)[0].conj()
    }
}

/// tv component of a PseudoGreen (`ptv`).
pub(crate) fn tv_pp(pp: &PseudoGreen, n: i64, m: i64) -> Complex {
    pp.lmix.raw_block_at(n, m)[0]
}

/// vt component of a PseudoGreen (`pvt`).
pub(crate) fn vt_pp(pp: &PseudoGreen, tau: i64, t: i64) -> Complex {
    -Complex::new(pp.sign.as_i64() as f64, 0.0)
        * pp.lmix.raw_block_at(t, pp.ntau as i64 - tau)[0].conj()
}

/// tv component of a ContourGreen (`stv`, raw lmix read).
pub(crate) fn tv_c(g: &ContourGreen, n: i64, m: i64) -> Complex {
    g.lmix.raw_block_at(n, m)[0]
}

/// vt component of a ContourGreen (`svt`).
pub(crate) fn vt_c(g: &ContourGreen, tau: i64, t: i64) -> Complex {
    -Complex::new(g.sign.as_i64() as f64, 0.0)
        * g.lmix.raw_block_at(t, g.ntau as i64 - tau)[0].conj()
}

use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{ContourGreen, PseudoGreen};
