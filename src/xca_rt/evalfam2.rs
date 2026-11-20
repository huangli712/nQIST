// Project : nQIST (real-time XCA)
// File    : src/xca_rt/evalfam2.rs
// Purpose : block-cut contour families, v2: the general chain walker
//           with L (lower, descending), I (imaginary) and R (upper)
//           blocks. Validation gates:
//             - JUpper (0,0,nr) == evalm greater chain (machine),
//             - m=2 component-SUM gates vs the reference kernel
//               (per the structural finding: per-family
//               correspondence does not hold; the block cuts sum to
//               the physical Sigma^</Sigma^mix).
//
//           Chain state per phase:
//             L: T(t) for t in [0..j]   (descending from the anchor j)
//             I: T(tau) for tau in [0..ntau]
//             R: T(t) for t in [0..n]
//           Boundaries:
//             anchor-lower -> L : greater-slot read (j, t)
//             anchor-lower -> I : tv read (j, tau)
//             anchor-lower -> R : lesser read (j, t)
//             anchor-tau   -> I : Matsubara mat difference read
//             anchor-tau   -> R : vt read (m, t)
//             L -> I  : tv read (t, tau)
//             L -> R  : lesser read (t_L, t_R)
//             I -> R  : vt read (tau, t)
//             L -> L  : greater-slot read (bigger, smaller)
//             I -> I  : Matsubara mat difference read (Volterra)
//             R -> R  : greater-slot difference read
//           Closing to n: block of the LAST vertex (L: lesser read
//           (t, n); I: vt read (tau, n); R: greater-slot read (t, n)
//           folded as the final chain step).
//
//           Chord factors (pole basis, per flavor):
//             Gtr: c_gtr e^{-i w t2} e^{+i w t1}   (args (t2, t1))
//             Les: c_les e^{-i w t_first} e^{+i w t_second}
//             Tv : c_tv e^{-i w t} e^{-w tau}
//             Vt : conj(c_tv) e^{+i w t} e^{-w (beta - tau)}
//             Mat: c_mat K(tau2 - tau1, w)
//           Endpoint factors apply pointwise in the owning vertex's
//           state; the anchor/n-side constants fold into cprod.

use super::evalm::{ChannelIter, RtChain, build_rt_chain};
use super::fit::{CsoeProj, csoe_coeffs};
use super::quad::poly_integral_chain;
use super::{gtr, gtr_c, les_c};
use crate::hilbert::HilbertSpace;
use crate::nca::Hybridization;
use crate::xca::topology::connected_pairings;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::PseudoGreen;
use kadanoffbaym::weights::Integrator;

#[derive(Debug, Clone, Copy)]
pub(crate) enum Anchor {
    JUpper(i64),
    JLower(i64),
    Tau(i64),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FamCut {
    pub nl: usize,
    pub ni: usize,
    pub nr: usize,
}

pub(crate) fn families(m: usize, anchor: &Anchor) -> Vec<FamCut> {
    let nint = 2 * m - 2;
    let mut out = Vec::new();
    match anchor {
        Anchor::JUpper(_) => out.push(FamCut {
            nl: 0,
            ni: 0,
            nr: nint,
        }),
        Anchor::JLower(_) => {
            for nl in 0..=nint {
                for ni in 0..=(nint - nl) {
                    out.push(FamCut {
                        nl,
                        ni,
                        nr: nint - nl - ni,
                    });
                }
            }
        }
        Anchor::Tau(_) => {
            for ni in 0..=nint {
                out.push(FamCut {
                    nl: 0,
                    ni,
                    nr: nint - ni,
                });
            }
        }
    }
    out
}

/// Mirrored-data detection for the backward-channel mode flips
/// (instrument-anchored 2026-11-29): a time-reflected channel delta
/// (d1c) carries its greater component's fit weight on NEGATED poles
/// (its phases run opposite), which the tau kernels reproduce only
/// with the flipped exponents (tv mode 6 / vt mode 5 / the negated
/// mat pole). A plain delta (e.g. a fixture's shared d1) has the
/// greater weight on positive poles and must NOT flip - keying on
/// the backward flag alone corrupted every cut whose backward
/// channel carried plain data (cutsum cut#1/cut#4 sector asymmetry
/// 1.97x while the reference stayed symmetric).
pub(crate) fn mirrored_line(li: usize, proj: &[CsoeProj], cs: &[usize]) -> bool {
    let p = &proj[cs[li]];
    let (mut wpos, mut wneg) = (0.0f64, 0.0f64);
    for (l, &w) in p.poles.iter().enumerate() {
        let mag = p.c_gtr[l].re * p.c_gtr[l].re + p.c_gtr[l].im * p.c_gtr[l].im;
        if w < 0.0 {
            wneg += mag;
        } else {
            wpos += mag;
        }
    }
    wneg > wpos
}

thread_local! {
    /// Per-row raw-greater fit cache for the vertex-anchored gtr
    /// chord ((2,5)): rows t in [0..=n] of gtr_c(delta, t, s)
    /// projected on the channel's own poles. The fits depend only on
    /// (delta data, poles, n, h) so they are shared across
    /// combos/chains/cuts within a thread; keyed by the delta
    /// address + a poles fingerprint to stay safe against address
    /// reuse across fixtures.
    static ROW_GTR_FITS: std::cell::RefCell<
        std::collections::HashMap<(usize, u64, i64, u64), Vec<Vec<Complex>>>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

thread_local! {
    /// Per-row lesser fit cache for the (3,2) per-row chord: the
    /// row-jj fit of les_c(delta, jj, t) on [0..=n]. Keyed like
    /// ROW_GTR_FITS (delta address + poles fingerprint + row + n + h).
    static ROW_LES_FITS: std::cell::RefCell<
        std::collections::HashMap<(usize, u64, i64, i64, u64), Vec<Complex>>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

pub(crate) fn row_les_fit(
    delta: &kadanoffbaym::types::ContourGreen,
    poles: &[f64],
    h: f64,
    row: i64,
    n: i64,
) -> Vec<Complex> {
    let key = (
        delta as *const _ as usize,
        poles
            .iter()
            .fold(0u64, |a, w| a.rotate_left(7) ^ w.to_bits()),
        row,
        n,
        h.to_bits(),
    );
    ROW_LES_FITS.with(|c| {
        if let Some(v) = c.borrow().get(&key) {
            return v.clone();
        }
        let vals: Vec<Complex> = (0..=n).map(|t| les_c(delta, row, t)).collect();
        let fit = csoe_coeffs(poles, h, &vals);
        c.borrow_mut().insert(key, fit.clone());
        fit
    })
}

thread_local! {
    /// Per-imaginary-index vt fit cache for the (4,2) per-m chord
    /// (the tau anchor's fixed-m coefficient): the csoe_coeffs fit
    /// of the raw vt data vt_c(delta, m, t) over t on the pole
    /// basis - exactly what LineSep::vt computes internally (its
    /// vt_coef). Keyed like ROW_LES_FITS (delta address + poles
    /// fingerprint + m + n + h).
    static ROW_VT_FITS: std::cell::RefCell<
        std::collections::HashMap<(usize, u64, i64, i64, u64), Vec<Complex>>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

pub(crate) fn vt_row_fit(
    delta: &kadanoffbaym::types::ContourGreen,
    poles: &[f64],
    h: f64,
    m: i64,
    n: i64,
) -> Vec<Complex> {
    let key = (
        delta as *const _ as usize,
        poles
            .iter()
            .fold(0u64, |a, w| a.rotate_left(7) ^ w.to_bits()),
        m,
        n,
        h.to_bits(),
    );
    ROW_VT_FITS.with(|c| {
        if let Some(v) = c.borrow().get(&key) {
            return v.clone();
        }
        let vals: Vec<Complex> = (0..=n).map(|t| super::vt_c(delta, m, t)).collect();
        let fit = csoe_coeffs(poles, h, &vals);
        c.borrow_mut().insert(key, fit.clone());
        fit
    })
}

pub(crate) fn row_gtr_fits(
    delta: &kadanoffbaym::types::ContourGreen,
    poles: &[f64],
    h: f64,
    n: i64,
) -> Vec<Vec<Complex>> {
    let key = (
        delta as *const _ as usize,
        poles
            .iter()
            .fold(0u64, |a, w| a.rotate_left(7) ^ w.to_bits()),
        n,
        h.to_bits(),
    );
    ROW_GTR_FITS.with(|c| {
        if let Some(v) = c.borrow().get(&key) {
            return v.clone();
        }
        let fits: Vec<Vec<Complex>> = (0..=n)
            .map(|t| {
                let vals: Vec<Complex> = (0..=n).map(|s| gtr_c(delta, t, s)).collect();
                csoe_coeffs(poles, h, &vals)
            })
            .collect();
        c.borrow_mut().insert(key, fits.clone());
        fits
    })
}

/// The row multiplier at (vertex v, absolute time t): the per-row
/// chord's data-derived shape, 1 when the vertex carries no
/// row-anchored line.
pub(crate) fn rm(rowmult: &[Option<Vec<Complex>>], v: usize, t: i64) -> Complex {
    rowmult[v].as_ref().map_or(Complex::ONE, |a| a[t as usize])
}

/// Audit helper (XCA_INTDUMP instrument): complex as "re,im" with
/// full double precision for point-by-point machine comparison.
fn cz(z: Complex) -> String {
    format!("{:+.15e},{:+.15e}", z.re, z.im)
}

#[inline]
fn les_slot(g: &PseudoGreen, a: i64, b: i64) -> Complex {
    if a <= b {
        g.less.raw_block_at(a, b)[0]
    } else {
        -g.less.raw_block_at(b, a)[0].conj()
    }
}

#[inline]
fn tv_slot(g: &PseudoGreen, t: i64, tau: i64) -> Complex {
    g.lmix.raw_block_at(t, tau)[0]
}

#[inline]
fn vt_fwd(g: &PseudoGreen, tau: i64, t: i64) -> Complex {
    let ntau = g.ntau as i64;
    -Complex::new(g.sign.as_i64() as f64, 0.0) * g.lmix.raw_block_at(t, ntau - tau)[0].conj()
}

#[inline]
fn mat_read(g: &PseudoGreen, d: i64) -> Complex {
    let raw = g.mat.raw();
    if d >= 0 {
        raw[d as usize]
    } else {
        // anti-periodic Matsubara continuation
        let s = g.sign.as_i64() as f64;
        -raw[(-d) as usize].conj() * Complex::new(s, 0.0)
    }
}

/// Fermi kernel with overflow-safe branch.
pub(crate) fn k_tau(tau: f64, w: f64, beta: f64) -> f64 {
    if w >= 0.0 {
        -(-w * tau).exp() / (1.0 + (-beta * w).exp())
    } else {
        -(-w * (tau - beta)).exp() / (1.0 + (beta * w).exp())
    }
}

/// Chord flavor between two block codes (0 = L, 1 = I, 2 = R,
/// 3 = lower anchor, 4 = tau anchor, 5 = n anchor).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Flavor {
    Gtr,
    Les,
    TvRealFirst,
    VtTauFirst,
    Mat,
}

fn chord_flavor(a: u8, b: u8) -> Flavor {
    match (a, b) {
        (2, 2) | (0, 0) | (3, 0) | (2, 5) => Flavor::Gtr,
        (3, 2) | (0, 2) | (0, 5) => Flavor::Les,
        (3, 1) | (0, 1) => Flavor::TvRealFirst,
        (1, 2) | (1, 5) | (4, 2) => Flavor::VtTauFirst,
        (1, 1) | (4, 1) => Flavor::Mat,
        _ => Flavor::Gtr, // unreachable for valid cuts
    }
}

/// The BOSON Matsubara kernel B(tau, omega) = -e^{-w tau} /
/// (1 - e^{-beta w}) - the convolution-semigroup companion of the
/// fermion K, required by the future Holstein rt extension (the
/// boson delta D0's Matsubara component carries this profile).
/// Overflow-safe: for w < 0 the direct denominator underflows to
/// 1 - e^{+beta|w|} -> -inf, so the reflected form
/// -e^{-w(tau-beta)}/(e^{beta w}-1) is used (identical by the
/// e^{beta w} renormalization, and decaying for w < 0).
pub(crate) fn boson_kernel(tau: f64, omega: f64, beta: f64) -> f64 {
    if omega >= 0.0 {
        -(-omega * tau).exp() / (1.0 - (-beta * omega).exp())
    } else {
        -(-omega * (tau - beta)).exp() / ((beta * omega).exp() - 1.0)
    }
}

/// Evaluate ONE (topology, channels, s0, pole-combo, cut): the
/// natural (undecorated) family value.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_cut(
    m: usize,
    cut: &FamCut,
    anchor: &Anchor,
    chain: &RtChain,
    pairs: &[(usize, usize)],
    cs: &[usize],
    combo: &[usize],
    proj: &[CsoeProj],
    pp_ints: &[Hybridization],
    bath: Option<&crate::nca::BosonBath>,
    pp_g: &[PseudoGreen],
    n: i64,
    h: f64,
    beta: f64,
    dtau: f64,
    kt: i64,
    integ: &Integrator,
) -> Complex {
    let ntau = pp_g[0].ntau as i64;
    // block code of each vertex (2m vertices; 0/2m-1 are anchors)
    let block_of = |v: usize| -> u8 {
        if v == 0 {
            match anchor {
                Anchor::JUpper(_) => 2,
                Anchor::JLower(_) => 3,
                Anchor::Tau(_) => 4,
            }
        } else if v == 2 * m - 1 {
            5
        } else if v <= cut.nl {
            0
        } else if v <= cut.nl + cut.ni {
            1
        } else {
            2
        }
    };
    // chord setup: explicit per-case endpoint factors (see the
    // module doc); `wsum[v]` accumulates the +-w of the vertex's
    // real-time phases, `isum[v]` the imaginary-side kernel factors,
    // and anchor/n-side constants fold into `cprod`
    let mut cprod = Complex::ONE;
    let mut wsum = vec![0.0f64; 2 * m];
    // per-entry (pole w, kernel mode, per-entry coefficient): the
    // boson two-mate tv/vt chords push TWO entries per line (the
    // normal and reflected thermal mates share the t-phase but
    // carry different tau kernels AND coefficients - a single
    // cprod scalar cannot carry both)
    let mut isum: Vec<Vec<(f64, u8, Complex)>> = vec![Vec::new(); 2 * m];
    // per-vertex row multipliers (the vertex-anchored per-row chords,
    // currently (2,5)): rowmult[v][t] multiplies vertex v's state at
    // ABSOLUTE time t, replacing the generic wsum phase for that
    // line; None = no row-anchored line at v (multiplier 1)
    let mut rowmult: Vec<Option<Vec<Complex>>> = vec![None; 2 * m];
    let jj = match anchor {
        Anchor::JUpper(jj) | Anchor::JLower(jj) => *jj,
        Anchor::Tau(_) => 0,
    };
    let mm = match anchor {
        Anchor::Tau(mm) => *mm,
        _ => 0,
    };
    // The all-I les cut (ni == 2, no real vertices) pairs its two I
    // vertices REVERSED relative to the forward cycle walk, matching
    // the reference les(4) algebra: the INNER integration variable
    // carries the n-adjacent segment (gc) + the line-(0,2) chord +
    // ops[2], the OUTER carries the j-adjacent segment (ga) + the
    // line-(1,3) chord + ops[1]. Invisible at constant fixtures
    // (all profiles equal - why the calibrations passed) but the
    // only pairing reproducing the reference sum on asymmetric
    // states (anchored by the bisection six-cut-sum comparison).
    // the m=2 (ni == 2) reversal is reference-anchored; the m=3
    // (ni == 4) generalization follows the same pattern (seed hosts
    // the LAST I vertex's factors) and is ED-adjudicated - the
    // XCA_M3_REV env knob selects it for comparison runs
    let rev_i = matches!(anchor, Anchor::JLower(_))
        && cut.nl == 0
        && cut.nr == 0
        && (cut.ni == 2 || (cut.ni == 4 && std::env::var("XCA_M3_REV").is_ok()));
    for li in 0..m {
        let (e, l) = pairs[li];
        let p = &proj[cs[li]];
        let w = p.poles[combo[li]];
        let be = block_of(e);
        let bl = block_of(l);
        // channel statistics via the mixed convention: cs[li] < nf
        // = fermion (pp_ints), nf = boson forward d0, nf+1 = boson
        // backward d0_cc (bosons never carry the fermionic minus)
        let nf = pp_ints.len();
        let ferm_line = cs[li] < nf;
        let bwd_line = if ferm_line {
            pp_ints[cs[li]].backward
        } else {
            bath.is_some() && cs[li] == nf + 1
        };
        match (be, bl) {
            // ---- greater-slot chords ----
            (2, 2) => {
                // evalm-validated convention: the chord value is
                // c e^{+i w (t_l - t_e)} (frame-independent); with
                // absolute vertex phases the anchor endpoint at j
                // contributes e^{-i w j h}
                // JUpper/ret-class anchor: ALL lines read the
                // FORWARD IN-triangle greater (the ret1 family's
                // gtr(n, i1) convention, both statistics) = c
                // e^{-iw(t_l - t_e)h} (cell-verified on d0:
                // gtr_c(d0,10,5) = -0.0912i e^{-i 0.25}); the
                // les/vt-class anchors keep the out-convention
                // e^{+iw(t_l-t_e)} (only reachable at m >= 3 - the
                // m=2 pairings have no internal-internal greater
                // line). Mirrors the evalm phi_w in-read flip (the
                // machine gate JUpper == evalm)
                if matches!(anchor, Anchor::JUpper(_)) {
                    cprod *= p.c_gtr[combo[li]];
                    wsum[l] += w;
                    wsum[e] += -w;
                } else {
                    cprod *= p.c_gtr[combo[li]];
                    wsum[l] += -w;
                    wsum[e] += w;
                }
            }
            (2, 5) | (5, 2) => {
                // n-side constant + the R-vertex phase
                let jjl = match anchor {
                    Anchor::JUpper(jx) | Anchor::JLower(jx) => *jx,
                    Anchor::Tau(_) => 0,
                };
                let rv = if be == 2 { e } else { l };
                // DATA-TRACKING CONVENTION (the 2pole-calib verdict,
                // 2026-11-29): the column-0 projection c_gtr (LSQ on
                // the full pole set of the actual sgtr column) tracks
                // the reference DATA better than LineSep::gtr's own
                // truncated-coef gtr_raw on the ext fits' degenerate
                // branch (fit_les.coefs phased as greater - a poor
                // reconstruction; the stock comparison degraded 50x
                // when mirroring it exactly). The twin L2 class-split
                // residual is the ext-fit pole-set conditioning
                // (memory #2558's pathology), a FIT-LAYER issue - the
                // fix belongs in pole discovery, not the chord. The
                // -conj(c_fit) experiment lives in c_gtr_fit (unused;
                // kept for the record).
                //
                // ANCHOR-CONDITIONAL (2026-11-29 JUpper audit): the
                // JUpper/ret-class family reads the FORWARD IN-
                // triangle greater gtr(n, u) = c e^{-iw(n-u)h} (both
                // statistics; cell-verified on the d0 data), while
                // the JLower/les-class families read the REVERSED
                // out-triangle -conj form = c e^{+iw(n-u)h} (pure-
                // imaginary c) - the shared e^{+} form is validated
                // there (the closed JLower anchor)
                if matches!(anchor, Anchor::JUpper(_)) {
                    cprod *= p.c_gtr[combo[li]] * (-Complex::I * w * h * (n - jjl) as f64).exp();
                    wsum[rv] += -w;
                } else if matches!(anchor, Anchor::Tau(_)) {
                    // MIX per-row semantics (2026-11-20 Tau campaign,
                    // the two-constant audit signature): the stock's
                    // mix families evaluate line2 as LineSep::gtr(i1,
                    // n) - the PER-ROW gtr fit (row_gtr_fits[u1][l] .
                    // e^{-iwnh}, the INTMINE-L2E instrument's exact
                    // form). The global column-0 c_gtr (the
                    // JLower-calibrated choice) differs per pole; the
                    // odd sectors' pole cancellations amplify the
                    // difference into the per-class constant ratios.
                    // The coefficient rides rowmult[rv] (applied at
                    // the R-vertex time by the window's rm())
                    let cgr = row_gtr_fits(pp_ints[cs[li]].delta, &p.poles, h, n);
                    let l0 = combo[li];
                    rowmult[rv] = Some((0..=n).map(|t| cgr[t as usize][l0]).collect());
                    cprod *= (-Complex::I * w * h * n as f64).exp();
                } else {
                    cprod *= p.c_gtr[combo[li]] * (Complex::I * w * h * (n - jjl) as f64).exp();
                    wsum[rv] += w;
                }
            }
            (0, 0) => {
                // Delta^>(t_e, t_l): e = earlier contour = BIGGER time
                cprod *= p.c_gtr[combo[li]];
                wsum[e] += -w;
                wsum[l] += w;
            }
            (3, 0) | (0, 3) => {
                // Delta^>(j, t_L) = c e^{-i w j} e^{+i w t}: the
                // kernel must evaluate e^{+iw t_L} -> wsum -w (the
                // legacy +w contradicted the stated form; fermion
                // lines machine-validated). BOSON lines take the
                // SAME sign now (2026-11-29 per-row session): the
                // d0/d0_cc real-time data is cell-verified
                // IDENTICAL to the fermion convention (les/gtr =
                // c e^{-iw(t2-t1)h} difference form; d0_cc's
                // real-time components are a copy of d0's), so the
                // legacy boson +w evaluated the SUM form
                // c e^{-iw(j+t_L)h} - its error entered through
                // the L window [0..j], the j-growing anchor drift
                // (0.02/0.26/0.44 at j=0/5/10). The old "boson
                // keeps the legacy sign" verdict predates fixes
                // 1-8 (R-close REV, parity, (3,2) time base).
                cprod *= p.c_gtr[combo[li]] * (-Complex::I * w * h * jj as f64).exp();
                let lv = if be == 0 { e } else { l };
                wsum[lv] += -w;
            }
            // ---- lesser chords ----
            (3, 2) | (2, 3) => {
                // Delta^<(j, t) = c e^{-i w j} e^{+i w t}; for a
                // BACKWARD channel the lesser data IS the partner
                // forward channel's greater fill (d1c.less = g), so
                // the coefficient and pole come from the partner
                // NOTE: the per-row projection (the exact LineSep
                // semantics, tested with both sampling ranges [0..n]
                // and [0..n-kt] - hypotheses 7, 7b, 8) is
                // consistently WORSE in this architecture: the
                // correctness of the les factor alone EXPOSES residual
                // mismatches in the other structures (the current
                // best state's global-c partially compensates them).
                // The remaining work is the full architecture-level
                // audit (memory #2575, the 10-hypothesis ledger).
                // The les chord coefficient: the reference les
                // families ALWAYS project the channel delta's LESSER
                // component (LineSeg::les on d.lam1 - the backward
                // channels swap only the OPERATORS and the stat sign,
                // never the Keldysh component; instrument-anchored
                // 2026-11-29: the historical bwd->c_gtr swap broke
                // every cut whose line-0 chain was backward - cut#0
                // (cs[0]=1) at 11%, while cut#3 (cs[0]=2, forward,
                // swap never fired) stayed exact - see FEASIBILITY.md
                // audit)
                //
                // NOTE: the per-row projection (the exact LineSep
                // semantics, tested with both sampling ranges [0..n]
                // and [0..n-kt] - hypotheses 7, 7b, 8) is
                // consistently WORSE in this architecture: the
                // correctness of the les factor alone EXPOSES residual
                // mismatches in the other structures (the current
                // best state's global-c partially compensates them).
                // The remaining work is the full architecture-level
                // audit (memory #2575, the 10-hypothesis ledger).
                let wl = w;
                let rv = if be == 2 { e } else { l };
                // PER-ROW PROJECTION (2026-11-29, the j-tail
                // instrument): the reference LineSeg re-fits each
                // anchor row on its own window; the global-c form's
                // compromise residuals scramble as the relative
                // phase e^{-iw(t-j)h} rotates - 26% at j=40
                // (separation 4.0) while the per-row refit stays
                // exact at the data level. FERMION lines only: the
                // boson channels (cs >= pp_ints.len(), the Holstein
                // bath) keep the global-c form - their walk
                // calibration owns the boson side. The row is FIXED
                // (jj) so one csoe_coeffs per line per call; the
                // kernel time base decides the fold:
                //   row form:        sum_l c_row[l] e^{-iw_l t h}
                //   relative kernel: c' = c_row e^{-iw jj h}
                //   absolute kernel: c' = c_row
                // (TTI degeneracy: c_row ~ c_les e^{+iw j h} so the
                // relative fold reproduces the global-c form; the
                // real win is non-stationary rows - the calib
                // fixture - where NO constant works)
                let ferm_line_l = cs[li] < pp_ints.len();
                let rel_base = block_of(rv.wrapping_sub(1)) == 2;
                let cl = if ferm_line_l {
                    // SCALE-PRESERVING per-row projection: the SHAPE
                    // (row-jj structure relative to row 0) comes from
                    // per-row data refits; the SCALE comes from
                    // p.c_les so amplitude levers (the eps-scaling
                    // gate's coefficient halving) still reach the
                    // line - a pure data refit bypassed them and
                    // degraded the m=3 eps^3 gate to eps^2. TTI
                    // degeneracy: c_j/c_0 = e^{+iw j h} ->
                    // cl = c_les e^{+iw j h} fold, and with
                    // fold = e^{-iw j h} (relative base) the exact
                    // global-c form returns. Marginal row-0 poles
                    // fall back to the TTI shape.
                    let delta_l = pp_ints[cs[li]].delta;
                    let cj = row_les_fit(delta_l, &p.poles, h, jj, n);
                    // the shape baseline is the PROJECTION-TIME row-0
                    // fit (p.les_row0) - bit-identical to the global
                    // c_les until an amplitude lever rescales c_les,
                    // so the unscaled shape reduces to the pure
                    // per-row form exactly while the eps^m lever
                    // stays exact (the lever cannot reach a data
                    // refit)
                    let c0 = &p.les_row0;
                    let c0max = c0
                        .iter()
                        .map(|c| c.re * c.re + c.im * c.im)
                        .fold(0.0f64, f64::max);
                    let shape = if {
                        let m = c0[combo[li]].re * c0[combo[li]].re
                            + c0[combo[li]].im * c0[combo[li]].im;
                        m > 1e-16 * c0max
                    } {
                        cj[combo[li]] / c0[combo[li]]
                    } else {
                        (Complex::I * wl * h * jj as f64).exp()
                    };
                    p.c_les[combo[li]]
                        * shape
                        * if rel_base {
                            (-Complex::I * wl * h * jj as f64).exp()
                        } else {
                            Complex::ONE
                        }
                } else {
                    // BOSON per-row projection (2026-11-29): the same
                    // scale-preserving row-jj refit as the fermion
                    // branch, on the bath's own delta (d0 forward /
                    // d0_cc backward). At TTI-clean fits this is an
                    // exact no-op (shape = e^{+iwjjh} reproduces the
                    // legacy fold bit-for-bit at the fit level); on
                    // non-stationary rows it carries the row
                    // structure the global c_les compromise misses
                    // (the fix-6 drift class, boson side).
                    match bath {
                        Some(b) => {
                            let delta_b = if cs[li] == nf { &b.d0 } else { &b.d0_cc };
                            let cj = row_les_fit(delta_b, &p.poles, h, jj, n);
                            let c0 = &p.les_row0;
                            let c0max = c0
                                .iter()
                                .map(|c| c.re * c.re + c.im * c.im)
                                .fold(0.0f64, f64::max);
                            let shape = if {
                                let mm2 = c0[combo[li]].re * c0[combo[li]].re
                                    + c0[combo[li]].im * c0[combo[li]].im;
                                mm2 > 1e-16 * c0max
                            } {
                                cj[combo[li]] / c0[combo[li]]
                            } else {
                                (Complex::I * wl * h * jj as f64).exp()
                            };
                            p.c_les[combo[li]]
                                * shape
                                * if rel_base {
                                    (-Complex::I * wl * h * jj as f64).exp()
                                } else {
                                    Complex::ONE
                                }
                        }
                        None => {
                            p.c_les[combo[li]]
                                * if rel_base {
                                    Complex::ONE
                                } else {
                                    (Complex::I * wl * h * jj as f64).exp()
                                }
                        }
                    }
                };
                let les_const = Complex::ONE;
                cprod *= cl * les_const;
                if std::env::var("XCA_PROBE").is_ok() {
                    eprintln!(
                        "PROBE les32: cs {} bwd {} cl {:?} wl {} jj {} const {:?}",
                        cs[li], bwd_line, cl, wl, jj, les_const
                    );
                }
                wsum[rv] += wl;
            }
            (0, 2) | (2, 0) => {
                // Delta^<(t_L, t_R) = c e^{-i w t_L} e^{+i w t_R};
                // BACKWARD: partner c_gtr/pole (see (3,2) above)
                let cl = if bwd_line && ferm_line {
                    p.c_gtr[combo[li]]
                } else {
                    p.c_les[combo[li]]
                };
                let wl = w;
                cprod *= cl;
                let lv = if be == 0 { e } else { l };
                let rv = if be == 2 { e } else { l };
                wsum[lv] += -wl;
                wsum[rv] += wl;
            }
            (0, 5) | (5, 0) => {
                // Delta^<(t_L, n) = c e^{-i w t_L} e^{+i w n};
                // The coefficient is ALWAYS the channel delta's lesser
                // projection - the reference les families project
                // LineSeg::les regardless of channel direction (the
                // bwd->c_gtr partner swap was the (3,2)-class bug:
                // cut#3's backward-line-1 chains sat at a constant
                // 0.76-0.91j mixture ratio while the forward ones
                // were machine-level 0.9998 - XCA_INTDUMP 2026-11-29)
                let cl = p.c_les[combo[li]];
                let wl = w;
                // DIRECTION FIX (fermion lines - re-validated
                // post-lam-fix by the cut#3 forward chains at
                // machine level): Delta^<(t_L, n) = c
                // e^{-iw(n - t_L)h} = c e^{-iwn} e^{+iw t_L}; the
                // L-vertex kernel (wsum -w) gives e^{+iw t_L} so the
                // anchor constant must be e^{-iwn}, NOT e^{+iwn}.
                // BOSON lines take the SAME direction now (2026-11-29
                // per-row session): the d0/d0_cc lesser data is
                // cell-verified c e^{-iw(n-t_L)h} (identical
                // convention to the fermion d1), so the legacy boson
                // e^{+iwn} evaluated the SUM form c e^{+iw(n+t_L)h}
                // - a j-growing error through the L window [0..j].
                // The old "boson anchor worsened under the flip"
                // verdict predates fixes 1-8.
                cprod *= cl * (-Complex::I * wl * h * n as f64).exp();
                let lv = if be == 0 { e } else { l };
                wsum[lv] += -wl;
            }
            // ---- tv chords (real endpoint first) ----
            (3, 1) | (1, 3) | (0, 1) | (1, 0) => {
                // Delta^tv(t, tau) = c_tv e^{-i w t} e^{-w tau}; for
                // a BACKWARD channel the greater-fit pole w is the
                // negation of the bath frequency while the tv
                // stored-tau profile keeps e^{-eps tau} (conjugation
                // preserves real exponents): the tau kernel exponent
                // flips (mode 6). The t-phase e^{-i w t} and c_tv
                // are correct as-is (anchored by the bisection
                // pointwise comparison against the raw d1c data)
                let bwd = bwd_line;
                let rv = if be != 1 { e } else { l };
                let tv = if be != 1 { l } else { e };
                if ferm_line {
                    cprod *= p.c_tv[combo[li]];
                    if be == 3 || bl == 3 {
                        // real endpoint is the lower anchor: constant
                        cprod *= (-Complex::I * w * h * jj as f64).exp();
                    } else {
                        wsum[rv] += -w;
                    }
                    isum[tv].push((
                        w,
                        if bwd && mirrored_line(li, proj, cs) {
                            6
                        } else {
                            0
                        },
                        Complex::ONE,
                    ));
                } else {
                    // TWO-MATE boson tv (2026-11-29, XCA_TVPROBE
                    // instrument): the free-boson D^tv carries two
                    // thermal mates sharing ONE t-phase,
                    //   tv(t, tau) = sum_l e^{-i w_l t h}
                    //       (A_l e^{-w_l tau} + B'_l e^{+w_l tau}),
                    // machine-verified on the full (t, tau) grid
                    // (worst 1e-8); the legacy single-column
                    // c_tv e^{w beta} form phase-flipped the
                    // reflected mate at large tau (8.6e-2 at
                    // (10, beta)). The t-phase e^{-iwt} needs
                    // wsum +w at internal real endpoints (the
                    // legacy -w gave e^{+iwt}); the two kernels
                    // ride per-entry coefficients as modes 0 (A)
                    // and 7 (B')
                    if be == 3 || bl == 3 {
                        cprod *= (-Complex::I * w * h * jj as f64).exp();
                    } else {
                        wsum[rv] += w;
                    }
                    isum[tv].push((w, 0, p.c_tv[combo[li]]));
                    isum[tv].push((w, 7, p.c_tv_r[combo[li]]));
                }
            }
            // ---- vt chords (tau endpoint first) ----
            (1, 2) | (2, 1) | (1, 5) | (5, 1) | (4, 2) | (2, 4) => {
                // Delta^vt(tau, t) = sgn conj(c_tv) e^{-w(beta-tau)}
                // e^{+i w t}; for a BACKWARD channel the greater-fit
                // pole w is the negation of the bath frequency
                // (d1c's greater data carries the conjugated phase),
                // while its vt tau-profile keeps e^{-eps(beta-tau)}:
                // the (beta-tau) kernel exponent therefore flips
                // (mode 5 / the reflected anchor constant). The
                // t-phase e^{+i w t} and conj(c_tv) are correct as-is
                // with the backward pole (anchored by the bisection
                // per-chain comparison against the reference les4).
                let bwd = bwd_line;
                let rv = if be != 1 && be != 4 { e } else { l };
                let tv = if be == 1 || be == 4 { e } else { l };
                if ferm_line {
                    // FERMION: conj(c_tv) with e^{+iwt} phases
                    // (bisection-anchored vs the reference les4)
                    if be == 4 || bl == 4 {
                        // PER-M ROW-FIT (the coefficient probe's
                        // verdict 2026-11-20: the c_tv.fold
                        // factorization is structurally different
                        // from LineSep::vt's per-m fit coefficients
                        // - the per-pole ratios are messy complex):
                        // fold the per-m fit coefficient c_l^{(mm)}
                        // directly - the raw vt data
                        // vt_c(delta, mm, t) fitted on the pole
                        // basis; the anchor is FIXED at tau_m so
                        // this is a constant per pole. The R-vertex
                        // t2-phase rides wsum (+w) as before.
                        // NOTE (calibration 2026-11-20): the per-m fit uses the SAME m
                        // for both classes on fixtures with shared
                        // deltas (no materialized d_cc - the
                        // reflected-index variant made s=1/s=0 mixed:
                        // 0.15/3.43 vs 0.28/2.35 uniform); the
                        // remaining per-class discrepancy lives
                        // OUTSIDE the (4,2) arm (stat-decoration
                        // class - see the per-chain audit)
                        let cf = vt_row_fit(pp_ints[cs[li]].delta, &p.poles, h, mm, n);
                        cprod *= cf[combo[li]];
                        wsum[rv] += w;
                    } else {
                        cprod *= p.c_tv[combo[li]].conj();
                        if be == 5 || bl == 5 {
                            cprod *= (Complex::I * w * h * n as f64).exp();
                        } else {
                            // INTERNAL real endpoint (R vertex, (1,2)/
                            // (2,1) - m >= 3 only): the vt data's
                            // t-phase is e^{+iwt} (FTVPROBE: the s=+1
                            // variant reproduces vt_c at 1.4e-6 over the
                            // full grid; the legacy +w gave e^{-iwt},
                            // 1.1e-1). The flip also makes the internal
                            // phase CONSISTENT with the (1,5) anchor
                            // constant e^{+iwnh} (they ran opposite)
                            wsum[rv] += -w;
                        }
                    }
                    isum[tv].push((
                        w,
                        if bwd && mirrored_line(li, proj, cs) {
                            5
                        } else {
                            1
                        },
                        Complex::ONE,
                    ));
                } else {
                    // TWO-MATE boson vt (2026-11-29): the Bosonic vt
                    // read is vt_c = -conj(lmix(t, beta-tau)) (Sign::
                    // Bose as_i64 = +1, so the MINUS stays; anchored
                    // by LineSep::vt == -conj(lmix) in the logs), so
                    // the two tv mates reflect into
                    //   vt(tau, t) = -sum_l e^{+i w_l t h}
                    //     (conj(B')/st e^{-w tau} + conj(A) st e^{+w tau}),
                    // st = e^{-w beta} (A = c_tv, B' = c_tv_r).
                    // The conjugated t-phase e^{+iwt} keeps wsum -w
                    // at internal real endpoints; the (1,5)/(5,1)
                    // n-side anchor constant flips to e^{+iwnh}; the
                    // (4,2)/(2,4) tau-anchor endpoint is fixed, so
                    // both mates' kernel values fold into cprod
                    let st = Complex::new((-w * beta).exp(), 0.0);
                    let c_norm = -p.c_tv_r[combo[li]].conj() / st;
                    let c_ref = -p.c_tv[combo[li]].conj() * st;
                    if be == 5 || bl == 5 {
                        cprod *= (Complex::I * w * h * n as f64).exp();
                        isum[tv].push((w, 0, c_norm));
                        isum[tv].push((w, 7, c_ref));
                    } else if be == 4 || bl == 4 {
                        let tm = dtau * mm as f64;
                        cprod *= c_norm * Complex::new((-w * tm).exp(), 0.0)
                            + c_ref * Complex::new((w * tm).exp(), 0.0);
                        wsum[rv] += -w;
                    } else {
                        wsum[rv] += -w;
                        isum[tv].push((w, 0, c_norm));
                        isum[tv].push((w, 7, c_ref));
                    }
                }
            }
            // ---- Matsubara K chords: overflow-safe factorization
            // K(t_l - t_e, w) = K(t_l) K(beta - t_e) / K(beta); the
            // 1/K(beta) constant folds into cprod, mode 3 = K(tau)
            // at the later endpoint, mode 4 = K(beta - tau) at the
            // earlier endpoint (both pointwise in the I-chain state)
            (1, 1) => {
                // BACKWARD-channel mat chord: the reflected data
                // d1c.mat(tau) = -d1.mat(beta - tau) has an
                // e^{-eps tau} DECAYING profile, which the K kernel
                // reproduces only at the NEGATED pole (the greater-
                // fit pole w is -eps for backward channels) - the
                // same class of flip as modes 5/6, caught by the
                // independent direct-transcription cross-check
                // (forward/backward ratio groups differed 3.09x)
                let bwd = bwd_line;
                let wm = if bwd
                    && (if ferm_line {
                        mirrored_line(li, proj, cs)
                    } else {
                        true
                    }) {
                    -w
                } else {
                    w
                };
                if ferm_line {
                    cprod *= p.c_mat[combo[li]] / Complex::new(k_tau(beta, wm, beta), 0.0);
                    isum[l].push((wm, 3, Complex::ONE));
                    isum[e].push((wm, 4, Complex::ONE));
                } else {
                    // boson matsubara chord: the B kernel (Bose
                    // statistics; same negated-pole rule for the
                    // backward reflected data)
                    cprod *= p.c_mat[combo[li]]
                        / Complex::new(super::boson_rt::boson_k_tau(beta, wm, beta), 0.0);
                    isum[l].push((wm, 13, Complex::ONE));
                    isum[e].push((wm, 14, Complex::ONE));
                }
            }
            (4, 1) | (1, 4) => {
                // tau-anchor -> I chord: EMPIRICALLY c_mat only, no
                // kernel factors (the m=2 constant gate matched the
                // reference vt(3) exactly with this form; the
                // reference's sigma_mat_core places those K kernels
                // through its own Matsubara backbone, which our mat
                // segments + this bare coefficient reproduce)
                cprod *= -p.c_mat[combo[li]];
            }
            _ => {
                // invalid combination for a well-formed cut
                debug_assert!(false, "invalid chord blocks ({be},{bl})");
            }
        }
    }
    let phi = |v: usize, t: i64| -> Complex {
        if wsum[v] == 0.0 {
            Complex::ONE
        } else {
            (-Complex::I * wsum[v] * h * t as f64).exp()
        }
    };
    let ikern = |v: usize, tau: i64| -> Complex {
        // SUM accumulation: each vertex hosts at most ONE line (the
        // pairing guarantees it), whose kernel factor is the SUM of
        // its mate terms - the boson two-mate chords push TWO entries
        // (the normal and reflected thermal mates) that must ADD.
        // The legacy multiply accumulation was degenerate-equal with
        // single entries but SQUARED the two-mate pushes (the
        // (1,0,1)/(0,2,0) cuts collapsed to machine zero). Empty
        // vertices (no imaginary-side line) carry factor ONE
        if isum[v].is_empty() {
            return Complex::ONE;
        }
        let mut acc = Complex::ZERO;
        let tauf = dtau * tau as f64;
        for &(w, mode, cf) in &isum[v] {
            let k = match mode {
                0 => Complex::new((-w * tauf).exp(), 0.0),
                // mode 7: the BOSON tv/vt REFLECTED mate's kernel
                // e^{+w tau} (the free-boson D^tv's second thermal
                // component; pairs with the mode-0 normal mate under
                // one shared t-phase)
                7 => Complex::new((w * tauf).exp(), 0.0),
                1 => Complex::new((-w * (beta - tauf)).exp(), 0.0),
                5 => Complex::new((w * (beta - tauf)).exp(), 0.0),
                6 => Complex::new((w * tauf).exp(), 0.0),
                3 => Complex::new(k_tau(tauf, w, beta), 0.0),
                4 => Complex::new(k_tau(beta - tauf, w, beta), 0.0),
                13 => Complex::new(super::boson_rt::boson_k_tau(tauf, w, beta), 0.0),
                14 => Complex::new(super::boson_rt::boson_k_tau(beta - tauf, w, beta), 0.0),
                _ => Complex::ONE,
            };
            acc += cf * k;
        }
        acc
    };
    // ---------------- chain construction -------------------------
    // vertices 1..2m-2 in contour order; the state is built phase by
    // phase. L-phase works in DESCENDING real time from j; the
    // stored array is indexed by t in [0..j]. I-phase over
    // [0..ntau]; R-phase over [0..n].
    let j = match anchor {
        Anchor::JUpper(jj) | Anchor::JLower(jj) => *jj,
        Anchor::Tau(_) => 0, // no L block for the tau anchor
    };
    let mut state_l: Vec<Complex> = Vec::new();
    let mut state_i: Vec<Complex> = Vec::new();
    let mut state_r: Vec<Complex> = Vec::new();
    let mut have_l = false;
    let mut have_i = false;
    let mut have_r = false;
    // helper: crossing kernels
    // L -> I: tv read (t, tau); L -> R: lesser (t_L, t_R);
    // I -> R: vt read (tau, t)
    // seed for vertex 1
    {
        let v = 1usize;
        let g1 = &pp_g[chain.sectors[1]];
        match block_of(v) {
            0 => {
                // L: anchor-lower -> L: greater-slot (j, t)
                let mut t = vec![Complex::ZERO; j as usize + 1];
                for tt in 0..=j {
                    t[tt as usize] = phi(v, tt) * ikern(v, 0) * chain.ops[v] * gtr(g1, j, tt);
                }
                state_l = t;
                have_l = true;
            }
            1 => {
                // I: anchor -> I
                let mut t = vec![Complex::ZERO; ntau as usize + 1];
                match anchor {
                    Anchor::JLower(jj) => {
                        if rev_i {
                            // reversed pairing: the chain seeds from
                            // the LAST I vertex's factors (the line-0
                            // chord kernel via isum, its op, and the
                            // n-adjacent segment's vt); at m=2 this
                            // reduces to vertex 2 / sectors[3] (the
                            // reference-anchored form)
                            let vlast = 2 * m - 2;
                            let gc = &pp_g[chain.sectors[2 * m - 1]];
                            for tau in 0..=ntau {
                                t[tau as usize] = phi(vlast, 0)
                                    * ikern(vlast, tau)
                                    * chain.ops[vlast]
                                    * vt_fwd(gc, tau, n);
                            }
                        } else {
                            for tau in 0..=ntau {
                                t[tau as usize] = phi(v, 0)
                                    * ikern(v, tau)
                                    * chain.ops[v]
                                    * tv_slot(g1, *jj, tau);
                            }
                        }
                    }
                    Anchor::Tau(mm) => {
                        // anchor-tau -> I: Matsubara difference read;
                        // the reference's mix families place the
                        // I-vertices on [0, tau_m] (between the
                        // contour start and the anchor - vt(2)'s
                        // lowside-to-m and vt(3)'s m^2 scaling)
                        for tau in 0..=*mm {
                            t[tau as usize] =
                                ikern(v, tau) * chain.ops[v] * mat_read(g1, *mm - tau);
                        }
                    }
                    Anchor::JUpper(_) => unreachable!(),
                }
                state_i = t;
                have_i = true;
            }
            2 => {
                // R
                let mut t = vec![Complex::ZERO; n as usize + 1];
                match anchor {
                    Anchor::JUpper(jj) => {
                        for tt in *jj..=n {
                            t[tt as usize] = phi(v, tt - jj)
                                * rm(&rowmult, v, tt)
                                * ikern(v, 0)
                                * chain.ops[v]
                                * gtr(g1, tt, *jj);
                        }
                    }
                    Anchor::JLower(jj) => {
                        for tt in 0..=n {
                            t[tt as usize] = phi(v, tt - jj)
                                * rm(&rowmult, v, tt)
                                * ikern(v, 0)
                                * chain.ops[v]
                                * les_slot(g1, *jj, tt);
                        }
                    }
                    Anchor::Tau(mm) => {
                        for tt in 0..=n {
                            t[tt as usize] = phi(v, tt - jj)
                                * rm(&rowmult, v, tt)
                                * ikern(v, 0)
                                * chain.ops[v]
                                * vt_fwd(g1, *mm, tt);
                        }
                    }
                }
                state_r = t;
                have_r = true;
            }
            _ => unreachable!(),
        }
    }
    // I-vertex upper bound: the tau anchor confines its I-block to
    // [0, tau_m]; the other anchors use the full imaginary axis
    let i_max = match anchor {
        Anchor::Tau(mm) => *mm,
        _ => ntau,
    };
    // remaining internal vertices 2..=2m-2
    for v in 2..=(2 * m - 2) {
        let g = &pp_g[chain.sectors[if rev_i { 2 * m - v } else { v }]];
        let bv = block_of(v);
        let bprev = block_of(v - 1);
        match (bprev, bv) {
            (0, 0) => {
                // L -> L: descending Volterra with greater-slot
                // segments: T_new(t2) = op * int_{t2}^{t1-range} ...
                // implemented as ascending convolution over reversed
                // indices with the same Gregory weights
                let mut tn = vec![Complex::ZERO; j as usize + 1];
                for t2 in 0..=j {
                    // integrate t1 in [t2, j]
                    let len = (j - t2) as usize;
                    let mut f = vec![Complex::ZERO; len + 1];
                    for q in 0..=len {
                        let t1 = t2 + q as i64;
                        f[q] = gtr(g, t1, t2) * state_l[t1 as usize];
                    }
                    let integral = poly_integral_chain(&f, 0, len, integ);
                    tn[t2 as usize] = phi(v, t2) * ikern(v, 0) * chain.ops[v] * integral;
                }
                state_l = tn;
            }
            (0, 1) => {
                // L -> I: tv kernel crossing
                let mut ti = vec![Complex::ZERO; ntau as usize + 1];
                for tau in 0..=ntau {
                    let mut f = vec![Complex::ZERO; j as usize + 1];
                    for tt in 0..=j {
                        f[tt as usize] = tv_slot(g, tt, tau) * state_l[tt as usize];
                    }
                    let integral = poly_integral_chain(&f, 0, j as usize, integ);
                    ti[tau as usize] = phi(v, 0) * ikern(v, tau) * chain.ops[v] * integral;
                }
                state_i = ti;
                have_i = true;
                have_l = false;
            }
            (0, 2) => {
                // L -> R: lesser crossing
                let mut tr = vec![Complex::ZERO; n as usize + 1];
                for tt in 0..=n {
                    let mut f = vec![Complex::ZERO; j as usize + 1];
                    for tl in 0..=j {
                        f[tl as usize] = les_slot(g, tl, tt) * state_l[tl as usize];
                    }
                    let integral = poly_integral_chain(&f, 0, j as usize, integ);
                    tr[tt as usize] =
                        phi(v, tt) * rm(&rowmult, v, tt) * ikern(v, 0) * chain.ops[v] * integral;
                }
                state_r = tr;
                have_r = true;
                have_l = false;
            }
            (1, 1) => {
                // I -> I: Matsubara Volterra with mat segments; the
                // reversed all-I cut applies vertex 1's factors
                // (ops[1], the line-1 kernel via isum[1]) at this
                // outer step
                let veff = if rev_i { 2 * m - 1 - v } else { v };
                let mut ti = vec![Complex::ZERO; ntau as usize + 1];
                for tau2 in 0..=i_max {
                    let mut f = vec![Complex::ZERO; tau2 as usize + 1];
                    for q in 0..=tau2 as usize {
                        f[q] = mat_read(g, tau2 - q as i64) * state_i[q];
                    }
                    let integral = poly_integral_chain(&f, 0, tau2 as usize, integ);
                    ti[tau2 as usize] =
                        phi(veff, 0) * ikern(veff, tau2) * chain.ops[veff] * integral;
                }
                state_i = ti;
            }
            (1, 2) => {
                // I -> R: vt crossing
                let probe = std::env::var("XCA_PROBE").is_ok();
                let mut tr = vec![Complex::ZERO; n as usize + 1];
                for tt in 0..=n {
                    let mut f = vec![Complex::ZERO; ntau as usize + 1];
                    for tau in 0..=i_max {
                        f[tau as usize] = vt_fwd(g, tau, tt) * state_i[tau as usize];
                    }
                    if probe && tt == 5 {
                        eprintln!(
                            "PROBE xing cs={cs:?} nl={} ni={} nr={}: state_i[32] {:?} gb.vt(32,5) {:?} f[32] {:?} ops[v] {:?}",
                            cut.nl,
                            cut.ni,
                            cut.nr,
                            state_i[32],
                            vt_fwd(g, 32, 5),
                            f[32],
                            chain.ops[v]
                        );
                    }
                    let integral = poly_integral_chain(&f, 0, i_max as usize, integ);
                    tr[tt as usize] =
                        phi(v, tt) * rm(&rowmult, v, tt) * ikern(v, 0) * chain.ops[v] * integral;
                }
                state_r = tr;
                have_r = true;
                have_i = false;
            }
            (2, 2) => {
                // R -> R: greater Volterra (validated path); the
                // JUpper cut starts at the relative lower bound j so
                // the Gregory weights match the evalm interval
                let lb = match anchor {
                    Anchor::JUpper(jx) => *jx as usize,
                    _ => 0,
                };
                let mut tn = vec![Complex::ZERO; n as usize + 1];
                for nsteps in (lb + 1)..=n as usize {
                    let mut f = vec![Complex::ZERO; nsteps + 1];
                    for u in lb..=nsteps {
                        // ABSOLUTE-ARGUMENT read (the reference's
                        // convention): the greater segment between
                        // u1=u and u2=nsteps read at (u, nsteps) -
                        // the accessor picks raw (u>u2) or the
                        // hermitian continuation (u<u2,
                        // -ret(u2,u1).conj()) automatically; the
                        // legacy elapsed form gtr(g, d, 0) assumed
                        // TTI and always read the in-triangle value
                        // - wrong for the les context (and for any
                        // non-TTI data)
                        let gv = match anchor {
                            // MIX reads the CONTINUATION branch too
                            // (the factor-6 probe: the stock's data_bc
                            // reads psc(gb, t2, t1) = the -ret*+less
                            // branch; the elapsed in-triangle form
                            // differs by the real-part sign - the
                            // two-constant audit's source)
                            Anchor::JLower(_) | Anchor::Tau(_) => gtr(g, u as i64, nsteps as i64),
                            _ => gtr(g, (nsteps - u) as i64, 0),
                        };
                        f[u] = gv * state_r[u];
                    }
                    let integral = poly_integral_chain(&f[lb..], 0, nsteps - lb, integ);
                    let trel = nsteps as i64
                        - match anchor {
                            Anchor::JUpper(jx) | Anchor::JLower(jx) => *jx,
                            Anchor::Tau(_) => 0,
                        };
                    tn[nsteps] = phi(v, trel)
                        * rm(&rowmult, v, nsteps as i64)
                        * ikern(v, 0)
                        * chain.ops[v]
                        * integral;
                }
                state_r = tn;
            }
            _ => unreachable!("invalid block transition"),
        }
    }
    // closing: last internal vertex (or the anchor when 2m-2 = 0)
    // -> n, plus the external vertex factors. The closing segment
    // flavor follows the last block.
    let vlast = 2 * m - 2;
    let gfin = &pp_g[chain.sectors[2 * m - 1]];
    let mut total = Complex::ZERO;
    if vlast == 0 {
        // m = 1 boundary: not used (NCA path)
        return Complex::ZERO;
    }
    match block_of(vlast) {
        0 => {
            // L -> n: lesser read (t, n)
            let mut f = vec![Complex::ZERO; j as usize + 1];
            for tl in 0..=j {
                f[tl as usize] = les_slot(gfin, tl, n) * state_l[tl as usize];
            }
            total = poly_integral_chain(&f, 0, j as usize, integ);
        }
        1 => {
            // I -> n: vt read (tau, n); the reversed all-I cut closes
            // with the j-adjacent segment's tv read (ga)
            let mut f = vec![Complex::ZERO; ntau as usize + 1];
            if rev_i {
                let ga = &pp_g[chain.sectors[1]];
                for tau in 0..=i_max {
                    f[tau as usize] = tv_slot(ga, jj, tau) * state_i[tau as usize];
                }
            } else {
                for tau in 0..=i_max {
                    f[tau as usize] = vt_fwd(gfin, tau, n) * state_i[tau as usize];
                }
            }
            total = poly_integral_chain(&f, 0, i_max as usize, integ);
        }
        _ => {
            // R -> n: greater slot (n, t) folded as the final step;
            // JUpper closes over the relative interval [j, n]
            let lb = match anchor {
                Anchor::JUpper(jx) => *jx as usize,
                _ => 0,
            };
            let mut f = vec![Complex::ZERO; n as usize + 1];
            for u in lb..=n as usize {
                // the reference families' ordering: ret1 (the
                // greater cuts) reads gtr(ga, n, i1) FORWARD; the
                // les/vt families read gtr(gc, ., n) REVERSED ->
                // the hermitian continuation -ret(n, .).conj() in
                // the sub-diagonal (hand-verified 4 digits vs
                // eval2's les2 array; the fermion constant fixtures
                // never discriminated it: -conj(z) == z for purely
                // imaginary z)
                // NOTE: the reference's les/vt families read the
                // final greater REVERSED (gtr(gc, ., n) ->
                // -ret(n,.).conj(), hand-verified on the boson-walk
                // les2 array) - but applying the reversal globally
                // to the JLower close BREAKS the fermion calibration
                // (the discriminating exponential fixtures): the cut
                // and family decompositions do NOT share the close
                // structure 1:1 (open item, memory #2575)
                // ORDERING AUDIT (open): the reference's les/vt
                // families read the final greater REVERSED (gtr(gc,
                // ., n) -> -ret(n,.).conj(), hand-verified 4 digits
                // on the boson-walk les2 array). Experimental table
                // (anchor WORST j=0/5/10 + the fermion calibration):
                //   uniform forward:      8.98/11.13/10.68 + green
                //   uniform reversed:     14.62/9.89/3.36 + fermion
                //                          clean j5 9.7->43 BROKEN
                //   boson-conditional:    14.85/10.66/4.14 + green
                // (fermion bit-identical) but j0 worse. The per-cut
                // decomposition boundary differs from the families'
                // (the close may correspond to les1's gb factor or a
                // mixture) - needs the structural derivation, not
                // empirical sweeps (memory #2575).
                // EXPERIMENT 2026-11-29 (post-lam1-fix): the
                // JLower/les-family close reads REVERSED -
                // gtr(gfin, u, n) - matching the reference les
                // families' gtr(gc, ., n) hermitian continuation
                // (XCA_INTDUMP pointwise: REV + fixed lam factors ->
                // CONSTANT per-point ratios ~1 vs les(1); fwd
                // rotates). JUpper keeps the forward read for now
                // (its external anchor vs the reference ret family
                // is a separate 7% audit - evalm's own gap).
                let gv_close = match anchor {
                    Anchor::JUpper(_) => gtr(gfin, n, u as i64),
                    _ => gtr(gfin, u as i64, n),
                };
                f[u] = gv_close * state_r[u];
            }
            total = poly_integral_chain(&f[lb..], 0, n as usize - lb, integ);
        }
    }
    // final vertex phase/operator + vertex-0 operator
    let jref = match anchor {
        Anchor::JUpper(jx) | Anchor::JLower(jx) => *jx,
        Anchor::Tau(_) => 0,
    };
    let closer = if block_of(vlast) == 2 {
        // R closing: the final external vertex factor at the
        // relative time (n - j)
        phi(2 * m - 1, n - jref) * ikern(2 * m - 1, 0)
    } else {
        ikern(2 * m - 1, 0)
    };
    // orientation (-1)^{1+nl} i^{ni} i^{#I->I segments}: one i per
    // imaginary vertex PLUS one extra i per Matsubara-branch
    // propagator segment (the contour measure i dtau of the I-block
    // interior; anchored by the reference les(4)/vt(3) prefactors
    // and the m=2 constant gate). Global measure: one h per real
    // internal vertex, one dtau per imaginary internal vertex (each
    // internal vertex's measure is applied exactly once by the step
    // or closing that consumes its time; all quadrature above is
    // dimensionless)
    let mut oric = Complex::new(if (1 + cut.nl) % 2 == 0 { 1.0 } else { -1.0 }, 0.0);
    for _ in 0..cut.ni {
        oric *= Complex::I;
    }
    for _ in 0..cut.ni.saturating_sub(1) {
        oric *= Complex::I;
    }
    // the tau anchor enters its I-block BACKWARD along the imaginary
    // axis ([0, tau_m]): an extra factor of i^{ni} (i.e. i for odd
    // ni, 1 for even), anchored by the m=2 constant gate ((0,1,1)
    // needs xi, (0,2,0) and (0,0,2) direct) and equivalent to the
    // backward-vertex factor (-i)^{ni} combined with a global
    // (-1)^{ni} pair sign
    if matches!(anchor, Anchor::Tau(_)) && cut.ni % 2 == 1 {
        oric *= Complex::I;
    }
    let meas = Complex::new(
        h.powi((cut.nl + cut.nr) as i32) * dtau.powi(cut.ni as i32),
        0.0,
    );
    // XCA_INTDUMP audit: the (0,0,2) JLower cut's per-point kernel at
    // the target row - the mirror of eval2's les(1) INTREF dump. The
    // chain path is recomputed pointwise below; INTMINE-TOT must
    // reproduce `total` bit-for-bit (the dump's self-check)
    if m == 2 && cut.nl == 0 && cut.ni == 0 && cut.nr == 2 {
        if let Anchor::JLower(jt) = *anchor {
            if std::env::var("XCA_INTDUMP")
                .ok()
                .and_then(|tv| tv.trim().parse::<i64>().ok())
                == Some(jt)
            {
                let g1 = &pp_g[chain.sectors[1]];
                let g2 = &pp_g[chain.sectors[2]];
                let g3 = &pp_g[chain.sectors[3]];
                // per-line chord constants of this combo (cprod's two
                // factors): line 0 = (3,2) les chord, line 1 = (2,5)
                // greater chord with its n-side anchor phase
                let nf = pp_ints.len();
                let w0 = proj[cs[0]].poles[combo[0]];
                let w1 = proj[cs[1]].poles[combo[1]];
                let cl0 = proj[cs[0]].c_les[combo[0]]
                    * if block_of(1) == 2 {
                        Complex::ONE
                    } else {
                        (Complex::I * w0 * h * jj as f64).exp()
                    };
                let cl1 =
                    proj[cs[1]].c_gtr[combo[1]] * (Complex::I * w1 * h * (n - jj) as f64).exp();
                eprintln!(
                    "INTMINE-D cs={:?} sectors={:?} ops={:?} combo={:?} poles={:?} wsum={:?} cprod={} cl0={} cl1={} phi3={} oric={} meas={}",
                    cs,
                    chain.sectors,
                    [chain.ops[0], chain.ops[1], chain.ops[2], chain.ops[3]],
                    combo,
                    [proj[cs[0]].poles[combo[0]], proj[cs[1]].poles[combo[1]]],
                    wsum,
                    cz(cprod),
                    cz(cl0),
                    cz(cl1),
                    cz(phi(2 * m - 1, n - jref)),
                    cz(oric),
                    cz(meas)
                );
                // per-combo vertex phases + seed values
                for u1 in 0..=n {
                    eprintln!(
                        "INTMINE-P {} {} {} {} {} {} {}",
                        cs[0],
                        cs[1],
                        chain.sectors[0],
                        combo[0],
                        combo[1],
                        u1,
                        cz(phi(1, u1 - jt) * chain.ops[1] * les_slot(g1, jt, u1))
                    );
                }
                for u2 in 0..=n {
                    eprintln!(
                        "INTMINE-Q {} {} {} {} {} {} {}",
                        cs[0],
                        cs[1],
                        chain.sectors[0],
                        combo[0],
                        combo[1],
                        u2,
                        cz(phi(2, u2 - jt))
                    );
                }
                // the EFFECTIVE per-row line-2 factor (the (2,5)
                // per-row form): c_row_gtr^{(u1)}[l1] e^{-iw n h} -
                // directly comparable to INTREF-L2 (ls2.gtr(u1, n))
                {
                    let fits_l1 = row_gtr_fits(pp_ints[cs[1]].delta, &proj[cs[1]].poles, h, n);
                    for u1 in 0..=n {
                        let wp = proj[cs[1]].poles[combo[1]];
                        eprintln!(
                            "INTMINE-L2E {} {} {} {} {} {}",
                            cs[0],
                            cs[1],
                            combo[0],
                            combo[1],
                            u1,
                            cz(fits_l1[u1 as usize][combo[1]]
                                * (-Complex::I * wp * h * n as f64).exp())
                        );
                    }
                }
                // per-combo row integrals, close integrand, recomputed total
                let mut total_re = Complex::ZERO;
                {
                    let mut seed = vec![Complex::ZERO; n as usize + 1];
                    for u1 in 0..=n {
                        seed[u1 as usize] = phi(1, u1 - jt) * chain.ops[1] * les_slot(g1, jt, u1);
                    }
                    let mut state2 = vec![Complex::ZERO; n as usize + 1];
                    for u2 in 1..=n {
                        let mut f = vec![Complex::ZERO; u2 as usize + 1];
                        for u in 0..=u2 {
                            f[u as usize] = gtr(g2, u, u2) * seed[u as usize];
                        }
                        let rowint = poly_integral_chain(&f, 0, u2 as usize, integ);
                        eprintln!(
                            "INTMINE-ROW {} {} {} {} {} {} {}",
                            cs[0],
                            cs[1],
                            chain.sectors[0],
                            combo[0],
                            combo[1],
                            u2,
                            cz(rowint)
                        );
                        state2[u2 as usize] = phi(2, u2 - jt) * chain.ops[2] * rowint;
                    }
                    let mut f = vec![Complex::ZERO; n as usize + 1];
                    for u2 in 0..=n {
                        f[u2 as usize] = gtr(g3, n, u2) * state2[u2 as usize];
                        eprintln!(
                            "INTMINE-CLOSE {} {} {} {} {} {} {}",
                            cs[0],
                            cs[1],
                            chain.sectors[0],
                            combo[0],
                            combo[1],
                            u2,
                            cz(f[u2 as usize])
                        );
                    }
                    total_re = poly_integral_chain(&f, 0, n as usize, integ);
                }
                eprintln!(
                    "INTMINE-TOT {} {} {} {} {} total={} total_re={}",
                    cs[0],
                    cs[1],
                    chain.sectors[0],
                    combo[0],
                    combo[1],
                    cz(total),
                    cz(total_re)
                );
                // combo-independent propagator grids (once per chain)
                if combo.iter().all(|&x| x == 0) {
                    for u2 in 0..=n {
                        eprintln!(
                            "INTMINE-GC {} {} {} {} {} {}",
                            cs[0],
                            cs[1],
                            chain.sectors[0],
                            u2,
                            cz(gtr(g3, n, u2)),
                            cz(gtr(g3, u2, n))
                        );
                    }
                    for u2 in 0..=n {
                        for u1 in 0..=u2 {
                            eprintln!(
                                "INTMINE-G2 {} {} {} {} {} {}",
                                cs[0],
                                cs[1],
                                chain.sectors[0],
                                u2,
                                u1,
                                cz(gtr(g2, u1, u2))
                            );
                        }
                    }
                }
            }
        }
    }
    // XCA_INTDUMP audit (cut#3): the (1,0,1) JLower cut's chain - the
    // mirror of eval2's les(3) INTREF3 dump. L-seed -> (0,2) crossing
    // -> R close, recomputed pointwise
    if m == 2 && cut.nl == 1 && cut.ni == 0 && cut.nr == 1 {
        if let Anchor::JLower(jt) = *anchor {
            if std::env::var("XCA_INTDUMP")
                .ok()
                .and_then(|tv| tv.trim().parse::<i64>().ok())
                == Some(jt)
            {
                let g1 = &pp_g[chain.sectors[1]];
                let g2 = &pp_g[chain.sectors[2]];
                let g3 = &pp_g[chain.sectors[3]];
                eprintln!(
                    "INTMINE3-D cs={:?} sectors={:?} ops={:?} combo={:?} poles={:?} wsum={:?} cprod={} oric={} meas={}",
                    cs,
                    chain.sectors,
                    [chain.ops[0], chain.ops[1], chain.ops[2], chain.ops[3]],
                    combo,
                    [proj[cs[0]].poles[combo[0]], proj[cs[1]].poles[combo[1]]],
                    wsum,
                    cz(cprod),
                    cz(oric),
                    cz(meas)
                );
                let mut seed = vec![Complex::ZERO; jt as usize + 1];
                for tl in 0..=jt {
                    seed[tl as usize] = phi(1, tl) * chain.ops[1] * gtr(g1, jt, tl);
                    eprintln!(
                        "INTMINE3-SEED {} {} {} {} {} {} {}",
                        cs[0],
                        cs[1],
                        chain.sectors[0],
                        combo[0],
                        combo[1],
                        tl,
                        cz(seed[tl as usize])
                    );
                }
                let mut tr = vec![Complex::ZERO; n as usize + 1];
                for tt in 0..=n {
                    let mut f = vec![Complex::ZERO; jt as usize + 1];
                    for tl in 0..=jt {
                        f[tl as usize] = les_slot(g2, tl, tt) * seed[tl as usize];
                    }
                    let integral = poly_integral_chain(&f, 0, jt as usize, integ);
                    tr[tt as usize] = phi(2, tt) * chain.ops[2] * integral;
                    eprintln!(
                        "INTMINE3-X {} {} {} {} {} {} {}",
                        cs[0],
                        cs[1],
                        chain.sectors[0],
                        combo[0],
                        combo[1],
                        tt,
                        cz(tr[tt as usize])
                    );
                }
                let mut f = vec![Complex::ZERO; n as usize + 1];
                for tt in 0..=n {
                    f[tt as usize] = gtr(g3, tt, n) * tr[tt as usize];
                    eprintln!(
                        "INTMINE3-CLOSE {} {} {} {} {} {} {}",
                        cs[0],
                        cs[1],
                        chain.sectors[0],
                        combo[0],
                        combo[1],
                        tt,
                        cz(f[tt as usize])
                    );
                }
                let total3 = poly_integral_chain(&f, 0, n as usize, integ);
                eprintln!(
                    "INTMINE3-TOT {} {} {} {} {} total={} total_re={}",
                    cs[0],
                    cs[1],
                    chain.sectors[0],
                    combo[0],
                    combo[1],
                    cz(total),
                    cz(total3)
                );
            }
        }
    }
    // XCA_INTDUMP audit (cut#5): the (2,0,0) all-L JLower cut - the
    // mirror of eval2's les(5) INTREF5 dump. L-seed -> (0,0) L-L
    // descending step -> L close, recomputed pointwise
    if m == 2 && cut.nl == 2 && cut.ni == 0 && cut.nr == 0 {
        if let Anchor::JLower(jt) = *anchor {
            if std::env::var("XCA_INTDUMP")
                .ok()
                .and_then(|tv| tv.trim().parse::<i64>().ok())
                == Some(jt)
            {
                let g1 = &pp_g[chain.sectors[1]];
                let g2 = &pp_g[chain.sectors[2]];
                let g3 = &pp_g[chain.sectors[3]];
                eprintln!(
                    "INTMINE5-D cs={:?} sectors={:?} ops={:?} combo={:?} poles={:?} wsum={:?} cprod={} oric={} meas={}",
                    cs,
                    chain.sectors,
                    [chain.ops[0], chain.ops[1], chain.ops[2], chain.ops[3]],
                    combo,
                    [proj[cs[0]].poles[combo[0]], proj[cs[1]].poles[combo[1]]],
                    wsum,
                    cz(cprod),
                    cz(oric),
                    cz(meas)
                );
                let mut seed = vec![Complex::ZERO; jt as usize + 1];
                for t1 in 0..=jt {
                    seed[t1 as usize] = phi(1, t1) * chain.ops[1] * gtr(g1, jt, t1);
                    eprintln!(
                        "INTMINE5-SEED {} {} {} {} {} {} {}",
                        cs[0],
                        cs[1],
                        chain.sectors[0],
                        combo[0],
                        combo[1],
                        t1,
                        cz(seed[t1 as usize])
                    );
                }
                let mut state2 = vec![Complex::ZERO; jt as usize + 1];
                for t2 in 0..=jt {
                    let len = (jt - t2) as usize;
                    let mut f = vec![Complex::ZERO; len + 1];
                    for q in 0..=len {
                        let t1 = t2 + q as i64;
                        f[q] = gtr(g2, t1, t2) * seed[t1 as usize];
                    }
                    let integral = poly_integral_chain(&f, 0, len, integ);
                    state2[t2 as usize] = phi(2, t2) * chain.ops[2] * integral;
                    eprintln!(
                        "INTMINE5-X {} {} {} {} {} {} {}",
                        cs[0],
                        cs[1],
                        chain.sectors[0],
                        combo[0],
                        combo[1],
                        t2,
                        cz(state2[t2 as usize])
                    );
                }
                let mut f = vec![Complex::ZERO; jt as usize + 1];
                for t2 in 0..=jt {
                    f[t2 as usize] = les_slot(g3, t2, n) * state2[t2 as usize];
                    eprintln!(
                        "INTMINE5-CLOSE {} {} {} {} {} {} {}",
                        cs[0],
                        cs[1],
                        chain.sectors[0],
                        combo[0],
                        combo[1],
                        t2,
                        cz(f[t2 as usize])
                    );
                }
                let total5 = poly_integral_chain(&f, 0, jt as usize, integ);
                eprintln!(
                    "INTMINE5-TOT {} {} {} {} {} total={} total_re={}",
                    cs[0],
                    cs[1],
                    chain.sectors[0],
                    combo[0],
                    combo[1],
                    cz(total),
                    cz(total5)
                );
            }
        }
    }
    // XCA_INTDUMP audit (cut#1): per-chain D/TOT for the (0,1,1)
    // cut - the sector-asymmetry instrument ([0] ~1.97x [3] while the
    // reference les2 is symmetric; the s0=0 chains carry the
    // backward channel 1, the exact s0=3 ones are all-forward)
    if m == 2 && cut.nl == 0 && cut.ni == 1 && cut.nr == 1 {
        if let Anchor::JLower(jt) = *anchor {
            if std::env::var("XCA_INTDUMP")
                .ok()
                .and_then(|tv| tv.trim().parse::<i64>().ok())
                == Some(jt)
            {
                eprintln!(
                    "INTMINE1-D cs={:?} sectors={:?} ops={:?} combo={:?} poles={:?} wsum={:?} cprod={} closer={} oric={} meas={}",
                    cs,
                    chain.sectors,
                    [chain.ops[0], chain.ops[1], chain.ops[2], chain.ops[3]],
                    combo,
                    [proj[cs[0]].poles[combo[0]], proj[cs[1]].poles[combo[1]]],
                    wsum,
                    cz(cprod),
                    cz(closer),
                    cz(oric),
                    cz(meas)
                );
                eprintln!(
                    "INTMINE1-TOT {} {} {} {} {} total={}",
                    cs[0],
                    cs[1],
                    chain.sectors[0],
                    combo[0],
                    combo[1],
                    cz(total)
                );
            }
        }
    }
    #[cfg(test)]
    if std::env::var("XCA_DBG").is_ok() {
        eprintln!(
            "eval_cut dbg: cs {cs:?} cut ({},{},{}) combo {combo:?} cprod {cprod:?} closer {closer:?} total {total:?} meas {meas:?} oric {oric:?} wsum {wsum:?} ops03 {:?} ops12 {:?} sectors {:?}",
            cut.nl,
            cut.ni,
            cut.nr,
            chain.ops[0] * chain.ops[2 * m - 1],
            chain.ops[1] * chain.ops[2],
            chain.sectors
        );
    }
    let val = cprod * oric * meas * closer * chain.ops[0] * total * chain.ops[2 * m - 1];
    if matches!(anchor, Anchor::Tau(_)) {
        // the mix's ADJOINT-SCATTER convention: the stock writes
        // lmix += pref * [op1.S.op4]^dag - at 1x1 the pre-scatter
        // sandwich IS this value, so the written value is its
        // conjugate plus the global -1 (empirically stock =
        // -conj(val), the factor-6 campaign's closing
        // identification; the -1's principled home is under audit)
        -val.conj()
    } else {
        val
    }
}

/// Sum of ALL block cuts for the given anchor over topologies,
/// channels, sectors and pole combinations (natural values).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sigma_cuts_m(
    m: usize,
    hs: &HilbertSpace,
    pp_ints: &[Hybridization],
    bath: Option<&crate::nca::BosonBath>,
    proj: &[CsoeProj],
    pp_g: &[PseudoGreen],
    anchor: &Anchor,
    n: i64,
    h: f64,
    beta: f64,
    dtau: f64,
    kt: i64,
    integ: &Integrator,
) -> Vec<Complex> {
    let ns = hs.ns();
    let mut out = vec![Complex::ZERO; ns];
    if m < 2 {
        return out;
    }
    // #7 threading: the (topology x chain x sector) sums are
    // embarrassingly parallel; chains are prebuilt (moved into the
    // work items) and each thread accumulates a per-sector partial
    let topos = connected_pairings(m);
    let mut cuts = families(m, anchor);
    // diagnostic knob (mirrors the evalfam2m one): "nl,ni,nr" keeps
    // only that family; a leading '!' excludes it
    if let Ok(f) = std::env::var("XCA_MCUT") {
        let (excl, f) = match f.strip_prefix('!') {
            Some(r) => (true, r.to_string()),
            None => (false, f),
        };
        let p: Vec<usize> = f.split(',').map(|x| x.parse().unwrap()).collect();
        cuts.retain(|c| {
            let hit = (c.nl, c.ni, c.nr) == (p[0], p[1], p[2]);
            if excl { !hit } else { hit }
        });
    }
    let chans = super::boson_rt::chans_with_bath(pp_ints, bath);
    let mut work = Vec::new();
    for (ti, topo) in topos.iter().enumerate() {
        for cs in ChannelIter::new(chans.len(), m) {
            for s0 in 0..ns {
                if let Some(chain) =
                    super::boson_rt::build_rt_chain_mixed(hs, &topo.pairs, &cs, &chans, s0)
                {
                    work.push((ti, s0, cs.clone(), chain));
                }
            }
        }
    }
    let accum = |w: &[(usize, usize, Vec<usize>, _)]| -> Vec<Complex> {
        let mut part = vec![Complex::ZERO; ns];
        for item in w {
            let (ti, s0, cs, chain) = (item.0, item.1, &item.2, &item.3);
            let topo = &topos[ti];
            let mut combo = vec![0usize; m];
            loop {
                let mut stat = 1.0f64;
                for &c in cs.iter() {
                    // boson channels carry no statistics factor
                    if c < pp_ints.len() && pp_ints[c].backward {
                        stat *= -1.0;
                    }
                }
                let msign = if m % 2 == 0 { 1.0 } else { -1.0 };
                // The topology parity factor corresponds to the
                // reference's BOTH-FERMION crossing factor
                // (int1.fermion && int2.fermion -> stat_sign *= -1):
                // boson-boson line pairs carry no crossing sign (the
                // stock's boson prefactor is +1), so parity applies
                // only when EVERY line is a fermion channel
                // (instrument: the boson probe's four equal chains
                // were sign-flipped vs stock, 2026-11-29)
                let parity_eff = if cs.iter().all(|&c| c < pp_ints.len()) {
                    topo.parity
                } else {
                    1.0
                };
                for cut in cuts.iter() {
                    let val = eval_cut(
                        m,
                        cut,
                        anchor,
                        chain,
                        &topo.pairs,
                        cs,
                        &combo,
                        proj,
                        pp_ints,
                        bath,
                        pp_g,
                        n,
                        h,
                        beta,
                        dtau,
                        kt,
                        integ,
                    );
                    part[s0] += Complex::new(msign * stat * parity_eff, 0.0) * val;
                }
                let mut idx = 0;
                let mut wrapped = false;
                loop {
                    if idx == m {
                        wrapped = true;
                        break;
                    }
                    combo[idx] += 1;
                    if combo[idx] < proj[cs[idx]].poles.len() {
                        break;
                    }
                    combo[idx] = 0;
                    idx += 1;
                }
                if wrapped {
                    break;
                }
            }
        }
        part
    };
    let nthreads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(4)
        .clamp(1, work.len().max(1));
    let chunk = work.len().div_ceil(nthreads);
    let partials: Vec<Vec<Complex>> = std::thread::scope(|scope| {
        let handles: Vec<_> = work
            .chunks(chunk)
            .map(|w| scope.spawn(move || accum(w)))
            .collect();
        handles.into_iter().map(|hd| hd.join().unwrap()).collect()
    });
    for part in partials {
        for s in 0..ns {
            out[s] += part[s];
        }
    }
    // the mix (lmix) write carries the OUTPUT SECTOR's
    // pseudo-particle sign (reference `d.sig`; bosonic empty/doubly
    // sectors +1, fermionic singly-occupied -1): the [-, +, +, -]
    // sector pattern anchored by the m=2 constant gate
    if matches!(anchor, Anchor::Tau(_)) {
        for s0 in 0..out.len() {
            out[s0] *= Complex::new(pp_g[s0].sign.as_i64() as f64, 0.0);
        }
    }
    out
}

/// Per-cut contributions (diagnostic surface): one vector per cut in
/// `families(m, anchor)` order, each summed over topologies,
/// channels, sectors and pole combinations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sigma_cut_contribs(
    m: usize,
    hs: &HilbertSpace,
    pp_ints: &[Hybridization],
    bath: Option<&crate::nca::BosonBath>,
    proj: &[CsoeProj],
    pp_g: &[PseudoGreen],
    anchor: &Anchor,
    n: i64,
    h: f64,
    beta: f64,
    dtau: f64,
    kt: i64,
    integ: &Integrator,
) -> Vec<Vec<Complex>> {
    let ns = hs.ns();
    let cuts = families(m, anchor);
    let mut outs = vec![vec![Complex::ZERO; ns]; cuts.len()];
    if m < 2 {
        return outs;
    }
    for topo in connected_pairings(m) {
        let pairs = topo.pairs.clone();
        for cs in ChannelIter::new(pp_ints.len(), m) {
            for s0 in 0..ns {
                let Some(chain) = build_rt_chain(hs, &pairs, &cs, pp_ints, s0) else {
                    continue;
                };
                let mut combo = vec![0usize; m];
                loop {
                    let mut stat = 1.0f64;
                    for &c in &cs {
                        if pp_ints[c].fermion && pp_ints[c].backward {
                            stat *= -1.0;
                        }
                    }
                    let msign = if m % 2 == 0 { 1.0 } else { -1.0 };
                    for (ci, cut) in cuts.iter().enumerate() {
                        let val = eval_cut(
                            m, cut, anchor, &chain, &pairs, &cs, &combo, proj, pp_ints, bath, pp_g,
                            n, h, beta, dtau, kt, integ,
                        );
                        outs[ci][s0] += Complex::new(msign * stat * topo.parity, 0.0) * val;
                    }
                    let mut idx = 0;
                    let mut wrapped = false;
                    loop {
                        if idx == m {
                            wrapped = true;
                            break;
                        }
                        combo[idx] += 1;
                        if combo[idx] < proj[cs[idx]].poles.len() {
                            break;
                        }
                        combo[idx] = 0;
                        idx += 1;
                    }
                    if wrapped {
                        break;
                    }
                }
            }
        }
    }
    // the mix (lmix) write carries the OUTPUT SECTOR's
    // pseudo-particle sign (reference `d.sig`; bosonic empty/doubly
    // sectors +1, fermionic singly-occupied -1): the [-, +, +, -]
    // sector pattern anchored by the m=2 constant gate
    if matches!(anchor, Anchor::Tau(_)) {
        for row in outs.iter_mut() {
            for s0 in 0..row.len() {
                row[s0] *= Complex::new(pp_g[s0].sign.as_i64() as f64, 0.0);
            }
        }
    }
    outs
}
