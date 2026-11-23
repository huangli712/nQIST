// Project : nQIST (real-time XCA, multi-band)
// File    : src/xca_rt/evalfam2m.rs
// Purpose : MATRIX path of the block-cut families: `eval_cut_m`
//           mirrors `evalfam2::eval_cut` window-for-window with
//           rectangular block chain states, the stock block
//           propagator reads, and entrywise matrix quadrature. The
//           CHORD SETUP is a verbatim copy of the scalar version
//           (keep the two in sync - every sign there is
//           instrument-anchored); the scalar accessors map onto the
//           stock block primitives as:
//             gtr      -> psc_blk_into   (hermitian continuation)
//             les_slot -> pls_blk_into   (the -[less]^dag fallback)
//             tv_slot  -> ptv_dir_blk    (lmix copy)
//             vt_fwd   -> ptv_blk_into   (the conj-TRANSPOSE view;
//                         the scalar's elementwise conj is its 1x1
//                         degenerate case)
//             mat_read -> the mat block (dim^2 per tau point; the
//                         negative branch takes the adjoint)
//           The scalar eval_cut stays the single-band route,
//           untouched (bit-identical).
//
//           Anchored by the m=2 three-angle degeneracy gates
//           (JUpper/JLower/Tau vs the stock OCA matrix kernel) and
//           the m=3 cut-vs-chain gate (JUpper vs sigma_gtr_xca_m_mat).

use super::boson_rt::chans_with_bath;
use super::evalfam2::{Anchor, FamCut, families, k_tau, mirrored_line, rm, row_les_fit};
use super::evalm::{ChannelIter, RtChainM};
use super::fit::CsoeProj;
use super::quad::poly_integral_chain_mat;
use crate::hilbert::HilbertSpace;
use crate::nca::{BosonBath, Hybridization};
use crate::oca::{pls_blk_into, psc_blk_into, ptv_blk_into, ptv_dir_blk};
use crate::xca::topology::connected_pairings;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::linalg::{DenseMat, scale_c};
use kadanoffbaym::types::PseudoGreen;
use kadanoffbaym::weights::Integrator;

/// Chord coefficient products (verbatim bookkeeping of the scalar
/// evaluator - no chain or propagator data enters).
pub(crate) struct ChordSetupM {
    pub cprod: Complex,
    pub wsum: Vec<f64>,
    pub isum: Vec<Vec<(f64, u8, Complex)>>,
    pub rowmult: Vec<Option<Vec<Complex>>>,
    pub rev_i: bool,
}

/// VERBATIM COPY of evalfam2::eval_cut's chord setup (the signs are
/// instrument-anchored; any chord fix applies to BOTH copies).
#[allow(clippy::too_many_arguments)]
pub(crate) fn chord_setup_m(
    m: usize,
    cut: &FamCut,
    anchor: &Anchor,
    pairs: &[(usize, usize)],
    cs: &[usize],
    combo: &[usize],
    proj: &[CsoeProj],
    pp_ints: &[Hybridization],
    bath: Option<&BosonBath>,
    n: i64,
    h: f64,
    beta: f64,
    dtau: f64,
    kt: i64,
) -> ChordSetupM {
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
    // XCA_M3_REV env knob selects it for comparison runs.
    // MATRIX-ONLY extension (2026-11-20 Tau campaign): the all-I
    // reversal also applies to the Tau anchor (XCA_TAU_REV=off
    // restores the scalar condition) - hypothesis: the scalar's
    // forward Tau all-I is the same compensating structure the
    // JLower rev fixed
    let tau_rev_on = matches!(anchor, Anchor::Tau(_))
        && std::env::var("XCA_TAU_REV").map_or(false, |v| v == "on");
    let rev_i = (matches!(anchor, Anchor::JLower(_)) || tau_rev_on)
        && cut.nl == 0
        && cut.nr == 0
        && (cut.ni == 2 || (cut.ni == 4 && std::env::var("XCA_M3_REV").is_ok()));
    // The REFLECTED I-walk is the DEFAULT for the m=3 ALL-I cuts
    // since the 2026-11-22 contact adjudication (NQIST_TTI_CONTACT):
    // the m=3 increment's t=0 contact with the Matsubara equilibrium
    // is machine-level only with the reflected I-seed/close indices
    // (les 2.4e-8 / mix 1.4e-6 reflected vs 4.3e-2 / 1.4e-3
    // unreflected). E1/E2 isolation showed the contact needs ONLY
    // the ni == 4 scope; the mixed (0,3,1)/(0,2,2) reflection stays
    // OPT-IN (its specialized branches deviate from the raw-data
    // semantics at a ~1/beta scale - the m=3 direct-transcription
    // discriminant) and forward_130 remains opt-in (never
    // contact-validated).
    let mixed_031 = m == 3 && match (cut.nl, cut.ni, cut.nr) {
        (0, 3, 1) => std::env::var_os("XCA_M3_FORWARD_031").is_some(),
        (0, 2, 2) => std::env::var_os("XCA_M3_FORWARD_022").is_some(),
        _ => false,
    };
    let forward_130 = m == 3 && (cut.nl, cut.ni, cut.nr) == (1, 3, 0)
        && std::env::var_os("XCA_M3_FORWARD_130").is_some();
    let ni4_unreflect = std::env::var_os("XCA_M3_NI4_UNREFLECT").is_some();
    // XCA_M2_MODERN: diagnostic - the m=2 families take the m>=3 chord
    // forms (two-mate tv/vt, reflected all-I) instead of the legacy
    // single-mate stock-anchored branches
    let m2_modern = m == 2 && std::env::var_os("XCA_M2_MODERN").is_some();
    let forward_less = (m == 3 || m2_modern) && matches!(anchor, Anchor::JLower(_))
        && ((cut.ni == 4 && !ni4_unreflect) || (m2_modern && cut.ni == 2) || mixed_031 || forward_130)
        && !rev_i && bath.is_none();
    let forward_contact = (m == 3 || m2_modern) && matches!(anchor, Anchor::Tau(_))
        && ((cut.ni == 4 && !ni4_unreflect) || (m2_modern && cut.ni == 2) || mixed_031)
        && !rev_i && bath.is_none();
    let native_mixed = m == 3 && matches!(anchor, Anchor::JLower(_))
        && (cut.nl, cut.ni, cut.nr) == (1, 2, 1)
        && std::env::var_os("XCA_M3_NATIVE_MIXED").is_some();
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
                    // the les-class form is the ABSOLUTE-time greater
                    // gtr(t_e, t_l); the R seed and R->R steps apply
                    // time RELATIVE to j - compensate per endpoint
                    // (both-relative cancels: the all-R cut stays
                    // exact; m=3 direct-transcription verdict)
                    if block_of(l.wrapping_sub(1)) == 2 || l == 1 {
                        cprod *= (Complex::I * w * h * jj as f64).exp();
                    }
                    if block_of(e.wrapping_sub(1)) == 2 || e == 1 {
                        cprod *= (-Complex::I * w * h * jj as f64).exp();
                    }
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
                    // MIX per-row semantics (the two-constant audit
                    // signature): the stock's mix families evaluate
                    // line2 as LineSep::gtr(i1, n) - the PER-ROW gtr
                    // fit (row_gtr_fits[u1][l] . e^{-iwnh}); the
                    // global c_gtr differs per pole and the odd
                    // sectors' cancellations amplify it. Mirrors the
                    // scalar edit (see evalfam2.rs)
                    let cgr = super::evalfam2::row_gtr_fits(pp_ints[cs[li]].delta, &p.poles, h, n);
                    let l0 = combo[li];
                    rowmult[rv] = Some((0..=n).map(|t| cgr[t as usize][l0]).collect());
                    cprod *= (-Complex::I * w * h * n as f64).exp();
                } else {
                    // the n-side constant must match the R endpoint's
                    // phase reference: R->R steps and the R seed apply
                    // time RELATIVE to j ((n - j) + relative = the raw
                    // gtr(t_R, n) form e^{+iw(n - t_R)}); the I->R
                    // crossing applies ABSOLUTE time (constant n)
                    let nref = if rv == 1 || block_of(rv.wrapping_sub(1)) == 2 {
                        (n - jjl) as f64
                    } else {
                        n as f64
                    };
                    cprod *= p.c_gtr[combo[li]] * (Complex::I * w * h * nref).exp();
                    wsum[rv] += w;
                }
            }
            (0, 0) => {
                // Delta^>(t_e, t_l): e = walk-earlier = the anchor-side
                // (HIGHER-time) L vertex hosts the d-side operator and
                // is contour-LATER on the upper branch - the raw read
                // is gtr_c(d, t_e, t_l) with t_e FIRST (the legacy
                // signs encoded the swapped form; m=3
                // direct-transcription verdict on the (3,1,0) cut)
                cprod *= p.c_gtr[combo[li]];
                wsum[e] += w;
                wsum[l] += -w;
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
                // Delta^<(t_L, t_R) = c e^{-i w t_L} e^{+i w t_R}; the
                // coefficient is ALWAYS the channel's own lesser
                // projection (the reference les families project
                // LineSeg::les regardless of direction - the same
                // lesson as the (3,2) anchor; the legacy bwd->c_gtr
                // swap broke every cut whose line-1 chain was
                // backward; m=3 direct-transcription verdict)
                let cl = p.c_les[combo[li]];
                let wl = w;
                cprod *= cl;
                let lv = if be == 0 { e } else { l };
                let rv = if be == 2 { e } else { l };
                // the R endpoint's phase rides ABSOLUTE time at the
                // crossings but RELATIVE time at R->R steps -
                // compensate the relative case to the absolute form
                if block_of(rv.wrapping_sub(1)) == 2 {
                    cprod *= (-Complex::I * w * h * jj as f64).exp();
                }
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
                if native_mixed && ferm_line {
                    // The time fit uses exp(-i*w*t). The imaginary
                    // displacement in t+i*tau gives exp(+w*tau).
                    cprod *= p.c_tv[combo[li]];
                    if be == 3 || bl == 3 {
                        cprod *= (-Complex::I * w * h * jj as f64).exp();
                    } else {
                        wsum[rv] += w;
                    }
                    isum[tv].push((w, 7, Complex::ONE));
                } else if forward_less && forward_130 && (be == 0 || bl == 0) {
                    cprod *= p.c_tv[combo[li]];
                    wsum[rv] += w;
                    isum[tv].push((w, 5, Complex::ONE));
                } else if forward_less {
                    let coeff = super::fit::vt_col_fit(pp_ints[cs[li]].delta, &p.poles, beta, dtau, pp_ints[cs[li]].delta.ntau as i64, jj);
                    isum[tv].push((w, 1, coeff[combo[li]].conj()));
                } else if ferm_line {
                    // TWO-MATE fermion tv (2026-11-22 root-cause fix):
                    // the TRUE Delta^tv of a production bath is the
                    // KMS-complete form (machine-verified 1e-13 on
                    // the dimer deltas); per-entry coefficients as
                    // modes 0 (c_tv) and 7 (c_tv_r). m == 2 keeps
                    // the legacy single-mate form bit-exact (the
                    // np=10 fixture fits make the split deviate on
                    // spurious poles; the m=2 families are
                    // stock-anchored on that path)
                    if m == 2 && !m2_modern {
                        cprod *= p.c_tv[combo[li]];
                        if be == 3 || bl == 3 {
                            cprod *= (-Complex::I * w * h * jj as f64).exp();
                        } else {
                            wsum[rv] += w;
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
                        if be == 3 || bl == 3 {
                            // real endpoint is the lower anchor: constant
                            cprod *= (-Complex::I * w * h * jj as f64).exp();
                        } else {
                            // the tv data form is c_tv e^{-i w t} - the
                            // vertex phase must be e^{-iwt} (wsum +w)
                            wsum[rv] += w;
                        }
                        isum[tv].push((w, 0, p.c_tv_a[combo[li]]));
                        isum[tv].push((w, 7, p.c_tv_b[combo[li]]));
                    }
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
                    isum[tv].push((w, 0, p.c_tv_a[combo[li]]));
                    isum[tv].push((w, 7, p.c_tv_b[combo[li]]));
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
                let mut ferm_two_mate = false;
                if native_mixed && ferm_line {
                    // The reflected, conjugated tv profile gives vt.
                    cprod *= p.c_tv[combo[li]].conj();
                    if be == 5 || bl == 5 {
                        cprod *= (Complex::I * w * h * n as f64).exp();
                    } else {
                        wsum[rv] -= w;
                    }
                    isum[tv].push((w, 5, Complex::ONE));
                } else if (forward_less
                    || (forward_contact && std::env::var_os("XCA_15COL_LES").is_some()))
                    && (be == 5 || bl == 5)
                {
                    // XCA_15COL_LES (diagnostic): give the mixed (Tau) form of
                    // the (1,5) chord the SAME tau-dependent mode-0 kernel the
                    // les form uses, instead of the constant cprod factor of
                    // the forward_contact branch below - the two forms of the
                    // all-imaginary cut otherwise diverge linearly in n
                    // (junction identity).
                    let coeff = super::fit::vt_col_fit(pp_ints[cs[li]].delta, &p.poles, beta, dtau, if forward_contact { mm } else { pp_ints[cs[li]].delta.ntau as i64 }, n);
                    // mixed walk: the I-vertex sits at physical tau_v =
                    // (mma - tau) dtau, so the (1,5) profile cf e^{-w(beta - tau_v)}
                    // is the mode-8 kernel (the les walk has tau_v =
                    // (ntau - tau) dtau, i.e. the mode-0 kernel)
                    if std::env::var("XCA_15COL_LES").as_deref() == Ok("unit") {
                        // diagnostic: the SAME constant (unit) chord in both the
                        // les and the mixed form, to tell whether their
                        // time-dependent divergence lives on this chord
                        isum[tv].push((w, 99, Complex::ONE));
                    } else {
                        let mode15 = if forward_contact
                            && std::env::var("XCA_15COL_LES").as_deref() == Ok("mode8")
                        {
                            8
                        } else {
                            0
                        };
                        isum[tv].push((w, mode15, coeff[combo[li]]));
                    }
                } else if mixed_031 && ferm_line && (be, bl) == (1, 2) {
                    cprod *= p.c_tv[combo[li]].conj();
                    wsum[rv] -= w;
                    if let Anchor::Tau(_) = anchor {
                        isum[tv].push((-w, 8, Complex::ONE));
                    } else {
                        isum[tv].push((w, 7, Complex::ONE));
                    }
                } else if ferm_line {
                    // FERMION: conj(c_tv) with e^{+iwt} phases
                    // (bisection-anchored vs the reference les4)
                    if be == 4 || bl == 4 {
                        // PER-M ROW-FIT (the coefficient probe's
                        // verdict: c_tv.fold is structurally
                        // different from LineSep::vt's per-m fit
                        // coefficients) - mirrors the scalar fix
                        // verbatim (evalfam2's vt_row_fit)
                        // NOTE: uniform m for both classes (see the scalar comment -
                        // shared-delta fixtures have no d_cc; the
                        // remaining per-class discrepancy is outside
                        // the (4,2) arm)
                        let cf =
                            super::evalfam2::vt_row_fit(pp_ints[cs[li]].delta, &p.poles, h, mm, n);
                        cprod *= cf[combo[li]];
                        wsum[rv] += w;
                    } else if (forward_contact || std::env::var("XCA_15COL").is_ok()) && (be == 5 || bl == 5) {
                        // THE (1,5) TAU-COLUMN FIT (the two-sided dump
                        // verdict: the c_tv projection MIS-ASSIGNS
                        // near-degenerate poles - ch2's pole-0 vt
                        // weight lands on pole 1; the same-channel
                        // totals close coincidentally, the mixed
                        // exposes it). vt(delta, tau, n) =
                        // sum_l cf[l].e^{-w_l(beta-tau)} with the
                        // n-phase ABSORBED into cf (no e^{+iwnh}!)
                        if let Anchor::Tau(mma) = anchor {
                            let cf = super::fit::vt_col_fit(
                                pp_ints[cs[li]].delta,
                                &p.poles,
                                beta,
                                dtau,
                                *mma,
                                n,
                            );
                            cprod *= cf[combo[li]];
                        } else {
                            cprod *= p.c_tv[combo[li]].conj();
                            cprod *= (Complex::I * w * h * n as f64).exp();
                        }
                    } else {
                        // TWO-MATE fermion vt (2026-11-22 root-cause
                        // fix): the conjugated mirror of the tv
                        // two-mate,
                        //   vt(tau, t) = sum_l e^{+i w_l t h}
                        //     (conj(a_l) e^{-w_l beta} e^{+w_l tau}
                        //      + conj(b_l) e^{-w_l tau}),
                        // per-entry coefficients as modes 7/0 (the
                        // boson two-mate pattern with the Fermi sign;
                        // on the fixture b = 0 the value reduces to
                        // the legacy conj(c_tv) mode-1 form exactly).
                        // The m=2 all-I TAU V2 vertex-attachment swap
                        // keeps the legacy shared push.
                        let v2_all_i = matches!(anchor, Anchor::Tau(_))
                            && cut.nl == 0
                            && cut.ni == 2
                            && cut.nr == 0
                            && (be == 1 && bl == 5 || be == 5 && bl == 1);
                        if v2_all_i {
                            cprod *= p.c_tv[combo[li]].conj();
                            if be == 5 || bl == 5 {
                                cprod *= (Complex::I * w * h * n as f64).exp();
                            } else {
                                wsum[rv] += -w;
                            }
                        } else {
                            let st = Complex::new((-w * beta).exp(), 0.0);
                            let c_ref = p.c_tv_a[combo[li]].conj() * st;
                            let c_norm = p.c_tv_b[combo[li]].conj() / st;
                            if m == 2 && !m2_modern {
                                // m == 2 keeps the legacy single-mate
                                // form bit-exact (see the tv arm)
                                cprod *= p.c_tv[combo[li]].conj();
                                if be == 5 || bl == 5 {
                                    cprod *= (Complex::I * w * h * n as f64).exp();
                                } else {
                                    if block_of(rv.wrapping_sub(1)) == 2 {
                                        cprod *= (Complex::I * w * h * jj as f64).exp();
                                    }
                                    wsum[rv] += -w;
                                }
                            } else {
                                if be == 5 || bl == 5 {
                                    cprod *= (Complex::I * w * h * n as f64).exp();
                                } else {
                                    // INTERNAL real endpoint (R vertex,
                                    // (1,2)/(2,1) - m >= 3 only): the vt
                                    // data's t-phase is e^{+iwt} ABSOLUTE;
                                    // the R->R steps apply relative time -
                                    // compensate
                                    if block_of(rv.wrapping_sub(1)) == 2 {
                                        cprod *= (Complex::I * w * h * jj as f64).exp();
                                    }
                                    wsum[rv] += -w;
                                }
                                isum[tv].push((w, 0, c_norm));
                                isum[tv].push((w, 7, c_ref));
                                ferm_two_mate = true;
                            }
                        }
                    }
                    // XCA_TAU_V2 (the all-I vertex-attachment swap,
                    // validated by the transcription test's per-tau1
                    // MACHINE match): the (1,5) kernel rides the
                    // I->I output vertex (2) instead of tv=1 - the
                    // stock's lam2/prof_l rides the ANCHOR-side
                    // vertex's time (the outer variable q)
                    let v2_all_i = matches!(anchor, Anchor::Tau(_))
                        && cut.nl == 0
                        && cut.ni == 2
                        && cut.nr == 0
                        && (be == 1 && bl == 5 || be == 5 && bl == 1);
                    let push15 = if v2_all_i { 2 } else { tv };
                    if !ferm_two_mate {
                        isum[push15].push((
                            w,
                            if forward_contact {
                                8
                            } else if bwd && mirrored_line(li, proj, cs) {
                                5
                            } else {
                                1
                            },
                            Complex::ONE,
                        ));
                    }
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
                let reverse_tau = tau_rev_on && rev_i && m == 3;
                // the fermion mat chord pole: NO negation - c_mat is
                // projected on the channel's OWN data in the K basis
                // (the time reflection is already carried by the
                // coefficients); the legacy bwd negation double-flipped
                // the kernel (m=3 direct-transcription verdict: the
                // e^{eps beta}-class scale error on the all-I cuts;
                // (1,1) is m>=3-only and was never stock-anchored).
                // The BOSON negation stays (its c_mat projection is
                // gated on the Holstein anchors).
                let wm = if !ferm_line && bwd { -w } else { w };
                if ferm_line {
                    cprod *= p.c_mat[combo[li]] / Complex::new(k_tau(beta, wm, beta), 0.0);
                    let (later_time, earlier_time) = if reverse_tau { (e, l) } else { (l, e) };
                    isum[later_time].push((wm, 3, Complex::ONE));
                    isum[earlier_time].push((wm, 4, Complex::ONE));
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
                // tau-anchor -> I chord. The BASE form: c_mat only
                // (with the minus sign, single-band-anchored). The
                // XCA_41K variant: the delta mat per pole = A_l
                // e^{-w(tau_m - tau)} = [A_l e^{-w tau_m}] e^{+w
                // tau} - a TAU-DEPENDENT kernel (mode 7) at the
                // I-vertex, mirroring the stock's lam1.mat backbone
                // integration (the per-(family,sector) real-ratio
                // table from the per-chain probe)
                    if forward_contact {
                        cprod *= -p.c_mat[combo[li]];
                        isum[l].push((w, 3, Complex::ONE));
                    } else if m == 2 && !m2_modern && cut.ni == 2 && cut.nl == 0 && cut.nr == 0 && !rev_i {
                        let coef = p.c_mat[combo[li]] * (-w * dtau * mm as f64).exp()
                            / (1.0 + (-beta * w).exp());
                        isum[1].push((w, 7, coef));
                    } else if m == 3 && tau_rev_on && rev_i && ferm_line {
                        let coef = -p.c_mat[combo[li]] * k_tau(dtau * mm as f64, w, beta);
                        isum[l].push((w, 7, coef));
                    } else if std::env::var("XCA_41K").is_ok() {
                        // THE XCA_41X EXPERIMENT (the mixed-channel
                        // arithmetic: the mixed ratios = the c_mat
                        // channel ratio 4.0/6.25 - as if the (4,1)
                        // reads the OTHER line's channel's
                        // projection! Swap ONLY the (4,1)'s channel
                        // index under this knob)
                        let p41: &crate::xca_rt::fit::CsoeProj =
                            if std::env::var("XCA_41X").is_ok() {
                                &proj[cs[1 - li]]
                            } else {
                                p
                            };
                    // the kernel mode is selectable: XCA_41K=7/0/1/5;
                    // XCA_41S=nosign drops the single-band -1;
                    // XCA_41C=refl uses the reflected constant
                    // e^{-w(beta-mm)} (NOTE vacuous at mm=ntau/2);
                    // XCA_41N=fp/fm divides by (1+-e^{-beta w}) - the
                    // Matsubara normalization convention of the
                    // c_mat projection basis
                    let mode: u8 = std::env::var("XCA_41K")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(7);
                    let sgn = if std::env::var("XCA_41S").as_deref() == Ok("nosign") {
                        1.0
                    } else {
                        -1.0
                    };
                    let w41 = p41.poles[combo[li]];
                    let cexp = if std::env::var("XCA_41C").as_deref() == Ok("refl") {
                        -w41 * (beta - dtau * mm as f64)
                    } else {
                        -w41 * dtau * mm as f64
                    };
                    let norm = match std::env::var("XCA_41N").as_deref() {
                        Ok("fp") => 1.0 + (-beta * w41).exp(),
                        Ok("fm") => 1.0 - (-beta * w41).exp(),
                        _ => 1.0,
                    };
                    // ROUTE B (the dump finding): fold the
                    // constant INTO the kernel's per-entry
                    // coefficient - the intermediates stay bounded
                    // (e^{-w(mm-tau)} <= 1 for w>0) and the ~1e4
                    // inter-combo cancellations (which amplify any
                    // quadrature/structural error ~100x) disappear.
                    // The isum entries support per-entry coefficients
                    // (the boson two-mate machinery)
                    let k41_c = Complex::new(sgn, 0.0)
                        * p41.c_mat[combo[li]]
                        * Complex::new(cexp.exp(), 0.0)
                        / Complex::new(norm, 0.0);
                    // XCA_TAU_V2 (the all-I vertex-attachment
                    // swap, validated by the transcription test's
                    // per-tau1 MACHINE match): the (4,1) kernel
                    // rides the SEED vertex (1) instead of l=2 -
                    // the stock's lam1(m-tau2) rides the n-side
                    // vertex's time (the inner variable t2)
                    let v2_41 = std::env::var("XCA_TAU_V2").is_ok()
                        && cut.nl == 0
                        && cut.ni == 2
                        && cut.nr == 0;
                    let push41 = if v2_41 { 1 } else { l };
                    isum[push41].push((w, mode, k41_c));
                } else {
                    // m >= 3 default: the TAU-DEPENDENT kernel form (the
                    // m == 2 stock-anchored branch's semantics) - the
                    // raw line data mat(mm - tau) carries the tau
                    // profile; the old constant-only form cannot
                    // represent it. The SIGN is the canonical
                    // arrangement's (the m == 2 V2 swap differs); the
                    // m=3 direct-transcription verdict: magnitude
                    // exact, sep = -dir with +c_mat
                    let coef = -p.c_mat[combo[li]] * (-w * dtau * mm as f64).exp()
                        / (1.0 + (-beta * w).exp());
                    isum[l].push((w, 7, coef));
                }
            }
            _ => {
                // invalid combination for a well-formed cut
                debug_assert!(false, "invalid chord blocks ({be},{bl})");
            }
        }
    }
    ChordSetupM {
        cprod,
        wsum,
        isum,
        rowmult,
        rev_i,
    }
}

/// Evaluate ONE (topology, channels, s0, pole-combo, cut) on a
/// MATRIX chain: the natural (undecorated) family value as a
/// (dim x dim) block of the chain's starting sector.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_cut_m(
    m: usize,
    cut: &FamCut,
    anchor: &Anchor,
    chain: &RtChainM,
    pairs: &[(usize, usize)],
    cs: &[usize],
    combo: &[usize],
    proj: &[CsoeProj],
    pp_ints: &[Hybridization],
    bath: Option<&BosonBath>,
    pp_g: &[PseudoGreen],
    n: i64,
    h: f64,
    beta: f64,
    dtau: f64,
    kt: i64,
    integ: &Integrator,
) -> DenseMat {
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
    let cs0 = chord_setup_m(
        m, cut, anchor, pairs, cs, combo, proj, pp_ints, bath, n, h, beta, dtau, kt,
    );
    let ChordSetupM {
        cprod,
        wsum,
        isum,
        rowmult,
        rev_i,
    } = cs0;
    // The REFLECTED I-walk is the DEFAULT for the m=3 ALL-I cuts
    // since the 2026-11-22 contact adjudication (NQIST_TTI_CONTACT):
    // the m=3 increment's t=0 contact with the Matsubara equilibrium
    // is machine-level only with the reflected I-seed/close indices
    // (les 2.4e-8 / mix 1.4e-6 reflected vs 4.3e-2 / 1.4e-3
    // unreflected). E1/E2 isolation showed the contact needs ONLY
    // the ni == 4 scope; the mixed (0,3,1)/(0,2,2) reflection stays
    // OPT-IN (its specialized branches deviate from the raw-data
    // semantics at a ~1/beta scale - the m=3 direct-transcription
    // discriminant) and forward_130 remains opt-in (never
    // contact-validated).
    let mixed_031 = m == 3 && match (cut.nl, cut.ni, cut.nr) {
        (0, 3, 1) => std::env::var_os("XCA_M3_FORWARD_031").is_some(),
        (0, 2, 2) => std::env::var_os("XCA_M3_FORWARD_022").is_some(),
        _ => false,
    };
    let forward_130 = m == 3 && (cut.nl, cut.ni, cut.nr) == (1, 3, 0)
        && std::env::var_os("XCA_M3_FORWARD_130").is_some();
    let ni4_unreflect = std::env::var_os("XCA_M3_NI4_UNREFLECT").is_some();
    // XCA_M2_MODERN: diagnostic - the m=2 families take the m>=3 chord
    // forms (two-mate tv/vt, reflected all-I) instead of the legacy
    // single-mate stock-anchored branches
    let m2_modern = m == 2 && std::env::var_os("XCA_M2_MODERN").is_some();
    let forward_less = (m == 3 || m2_modern) && matches!(anchor, Anchor::JLower(_))
        && ((cut.ni == 4 && !ni4_unreflect) || (m2_modern && cut.ni == 2) || mixed_031 || forward_130)
        && !rev_i && bath.is_none();
    let forward_contact = (m == 3 || m2_modern) && matches!(anchor, Anchor::Tau(_))
        && ((cut.ni == 4 && !ni4_unreflect) || (m2_modern && cut.ni == 2) || mixed_031)
        && !rev_i && bath.is_none();
    let v2p = m == 2
        && !m2_modern
        && cut.nl == 0
        && cut.ni == 2
        && cut.nr == 0
        && !rev_i
        && matches!(anchor, Anchor::Tau(_));
    let reflect_mixed = m == 3
        && matches!(anchor, Anchor::JLower(_))
        && (cut.nl, cut.ni, cut.nr) == (1, 2, 1)
        && std::env::var_os("XCA_M3_REFLECT_MIXED").is_some();
    let jj = match anchor {
        Anchor::JUpper(jj) | Anchor::JLower(jj) => *jj,
        Anchor::Tau(_) => 0,
    };
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
        let tauf = dtau * (if reflect_mixed { ntau - tau } else { tau }) as f64;
        for &(w, mode, cf) in &isum[v] {
            let k = match mode {
                0 => Complex::new((-w * tauf).exp(), 0.0),
                // mode 7: the BOSON tv/vt REFLECTED mate's kernel
                // e^{+w tau} (the free-boson D^tv's second thermal
                // component; pairs with the mode-0 normal mate under
                // one shared t-phase)
                7 => Complex::new((w * tauf).exp(), 0.0),
                8 => {
                    let Anchor::Tau(mm) = anchor else { unreachable!() };
                    Complex::new((-w * (beta - dtau * (*mm - tau) as f64)).exp(), 0.0)
                }
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
    // ---------------- chain construction (matrix states) ---------
    // block readers into DenseMat
    let gtr_blk = |g: &PseudoGreen, a: i64, b: i64| -> DenseMat {
        let dim = g.size1 as usize;
        let mut buf = vec![Complex::ZERO; dim * dim];
        psc_blk_into(g, a, b, dim, &mut buf);
        DenseMat::from_vec(dim, dim, buf)
    };
    let les_blk = |g: &PseudoGreen, a: i64, b: i64| -> DenseMat {
        let dim = g.size1 as usize;
        let mut buf = vec![Complex::ZERO; dim * dim];
        pls_blk_into(g, a, b, dim, &mut buf);
        DenseMat::from_vec(dim, dim, buf)
    };
    let tv_blk = |g: &PseudoGreen, t: i64, tau: i64| -> DenseMat {
        let dim = g.size1 as usize;
        let mut buf = vec![Complex::ZERO; dim * dim];
        ptv_dir_blk(g, t, tau, dim, &mut buf);
        DenseMat::from_vec(dim, dim, buf)
    };
    let vt_blk = |g: &PseudoGreen, tau: i64, t: i64| -> DenseMat {
        let dim = g.size1 as usize;
        let mut buf = vec![Complex::ZERO; dim * dim];
        ptv_blk_into(g, tau, t, dim, &mut buf);
        DenseMat::from_vec(dim, dim, buf)
    };
    let mat_blk = |g: &PseudoGreen, d: i64| -> DenseMat {
        let dim = g.size1 as usize;
        let nel = dim * dim;
        let raw = g.mat.raw();
        if d >= 0 {
            DenseMat::from_slice(dim, dim, &raw[d as usize * nel..(d as usize + 1) * nel])
        } else {
            // anti-periodic Matsubara continuation (adjoint form;
            // the scalar's elementwise conj is the 1x1 case)
            let s = g.sign.as_i64() as f64;
            scale_c(
                &DenseMat::from_slice(
                    dim,
                    dim,
                    &raw[(-d) as usize * nel..((-d) as usize + 1) * nel],
                )
                .adjoint(),
                -Complex::new(s, 0.0),
            )
        }
    };
    // entrywise matrix quadrature over a block-valued f slice
    let quad = |f: &[Vec<Complex>], rows: usize, cols: usize, i1: usize| -> DenseMat {
        let val = poly_integral_chain_mat(f, 0, i1, rows * cols, integ);
        DenseMat::from_vec(rows, cols, val)
    };
    let j = match anchor {
        Anchor::JUpper(jx) | Anchor::JLower(jx) => *jx,
        Anchor::Tau(_) => 0, // no L block for the tau anchor
    };
    // XCA_EQT_SKIP_LEND / XCA_EQT_SKIP_REND (diagnostic): drop the
    // equal-time CONTACT point of the L seed (t = j) and/or the R
    // close (u = n). When the external time equals the anchor both
    // the anchor and the internal vertex sit on the same contour
    // point; the continuous integral gives that point zero measure,
    // while the discrete sum carries it with an O(1) weight.
    let skip_l_end = std::env::var_os("XCA_EQT_SKIP_LEND").is_some();
    let skip_r_end = std::env::var_os("XCA_EQT_SKIP_REND").is_some();
    let mut state_l: Vec<DenseMat> = Vec::new();
    let mut state_i: Vec<DenseMat> = Vec::new();
    let mut state_r: Vec<DenseMat> = Vec::new();
    let mut have_l = false;
    let mut have_i = false;
    let mut have_r = false;
    // seed for vertex 1
    {
        let v = 1usize;
        let g1 = &pp_g[chain.sectors[1]];
        let dim1 = g1.size1 as usize;
        match block_of(v) {
            0 => {
                // L: anchor-lower -> L: greater-slot (j, t)
                let mut t = Vec::with_capacity(j as usize + 1);
                for tt in 0..=j {
                    let prod = chain.ops[v].mat_mul(&gtr_blk(g1, j, tt));
                    if skip_l_end && tt == j {
                        t.push(DenseMat::zeros_shape(prod.nrows(), prod.ncols()));
                    } else {
                        t.push(scale_c(&prod, phi(v, tt) * ikern(v, 0)));
                    }
                }
                state_l = t;
                have_l = true;
            }
            1 => {
                // I: anchor -> I
                match anchor {
                    Anchor::JLower(jja) => {
                        if rev_i {
                            // reversed pairing: seed from the LAST I
                            // vertex's factors. MATRIX association:
                            // the rev cut accumulates RIGHTWARD (the
                            // algebra G3.ops2.G2.ops1.G1 - the read
                            // LEFT, the op RIGHT; the scalar text's
                            // `ops * read` is its 1x1 degenerate)
                            let vlast = 2 * m - 2;
                            let gc = &pp_g[chain.sectors[2 * m - 1]];
                            let mut t = Vec::with_capacity(ntau as usize + 1);
                            for tau in 0..=ntau {
                                let prod = vt_blk(gc, tau, n).mat_mul(&chain.ops[vlast]);
                                t.push(scale_c(&prod, phi(vlast, 0) * ikern(vlast, tau)));
                            }
                            state_i = t;
                        } else {
                            let mut t = Vec::with_capacity(ntau as usize + 1);
                            for tau in 0..=ntau {
                                let index = if forward_less { ntau - tau } else { tau };
                                let prod = chain.ops[v].mat_mul(&tv_blk(g1, *jja, index));
                                t.push(scale_c(&prod, phi(v, 0) * ikern(v, tau)));
                            }
                            state_i = t;
                        }
                    }
                    Anchor::Tau(mma) => {
                        if rev_i {
                            // TAU all-I reversal (matrix-only
                            // experiment): seed from the n-side
                            // (vertex vlast's vt read), the anchor
                            // mat walk moves to the close
                            let vlast = 2 * m - 2;
                            let gc = &pp_g[chain.sectors[2 * m - 1]];
                            let mut t = vec![
                                DenseMat::zeros_shape(
                                    gc.size1 as usize,
                                    chain.ops[vlast].ncols()
                                );
                                *mma as usize + 1
                            ];
                            for tau in 0..=*mma {
                                let prod = vt_blk(gc, tau, n).mat_mul(&chain.ops[vlast]);
                                t[tau as usize] = scale_c(&prod, phi(vlast, 0) * ikern(vlast, tau));
                            }
                            state_i = t;
                        } else if (v2p || std::env::var("XCA_TAU_V2").is_ok())
                            && cut.nl == 0
                            && cut.ni == 2
                            && cut.nr == 0
                        {
                            // V2 FINAL (the transcription-validated
                            // LEFTWARD accumulation - the string
                            // ops[3].MAT.ops[2].mat2.ops[1].VT.ops[0]):
                            // the seed reads the VT of its OWN sector
                            // at (tau, n); the anchor mat walk moves to
                            // the close. The (4,1) kernel rides HERE
                            // (isum[1] under V2) - the manual's k41(t2)
                            // XCA_TAU_V2S: the SECTOR SWAP (the final
                            // derivation: the transcription's vt@gc=s3
                            // rides the t2/seed side and mat@sa=s1 the
                            // q/close side - the V2 kept the original
                            // greens; this swaps them; valid where
                            // dim(s1)==dim(s3), e.g. the mirror fixture)
                            let gseed: &kadanoffbaym::types::PseudoGreen =
                                if std::env::var("XCA_TAU_V2S").is_ok() {
                                    &pp_g[chain.sectors[2 * m - 1]]
                                } else {
                                    g1
                                };
                            // XCA_TAU_V2P: THE OPERATOR-PLACEMENT FIX (the
                            // 17th-round mechanism): the stock's string is
                            // mat.ops1.[gb.ops2.vt.k41] - op3(=ops2) INSIDE
                            // the inner integral and op2(=ops1) at the OUTER
                            // (adjacent to the mat read); my default has them
                            // BACKWARDS (ops1 in the seed, ops2 at the I->I
                            // output) - invisible for the same-channel (the
                            // dim-1 scalar corners commute!) and exposed for
                            // the mixed (the dim-2 intermediates)
                            let mut t = vec![
                                DenseMat::zeros_shape(chain.ops[v].nrows(), dim1);
                                ntau as usize + 1
                            ];
                            for tau in 0..=*mma {
                                let prod = if v2p {
                                    // the transpose-aware placement: the
                                    // stock's inner operator is ops[2]^T
                                    // ((s2->s3)) on the s3-green's vt
                                    chain.ops[2]
                                        .adjoint()
                                        .mat_mul(&vt_blk(
                                            &pp_g[chain.sectors[2 * m - 1]],
                                            tau,
                                            n,
                                        ))
                                } else {
                                    chain.ops[v].mat_mul(&vt_blk(gseed, tau, n))
                                };
                                t[tau as usize] = scale_c(&prod, ikern(v, tau));
                            }
                            state_i = t;
                        } else {
                            let mut t = vec![
                                DenseMat::zeros_shape(chain.ops[v].nrows(), dim1);
                                ntau as usize + 1
                            ];
                            for tau in 0..=*mma {
                                 let index = if forward_contact { tau } else { *mma - tau };
                                 let prod = chain.ops[v].mat_mul(&mat_blk(g1, index));
                                t[tau as usize] = scale_c(&prod, ikern(v, tau));
                            }
                            state_i = t;
                        }
                    }
                    Anchor::JUpper(_) => unreachable!(),
                }
                have_i = true;
            }
            2 => {
                // R
                match anchor {
                    Anchor::JUpper(jja) => {
                        let mut t =
                            vec![DenseMat::zeros_shape(chain.ops[v].nrows(), dim1); n as usize + 1];
                        for tt in *jja..=n {
                            let prod = chain.ops[v].mat_mul(&gtr_blk(g1, tt, *jja));
                            t[tt as usize] = scale_c(
                                &prod,
                                phi(v, tt - jja) * rm(&rowmult, v, tt) * ikern(v, 0),
                            );
                        }
                        state_r = t;
                    }
                    Anchor::JLower(jja) => {
                        let mut t =
                            vec![DenseMat::zeros_shape(chain.ops[v].nrows(), dim1); n as usize + 1];
                        for tt in 0..=n {
                            let prod = chain.ops[v].mat_mul(&les_blk(g1, *jja, tt));
                            t[tt as usize] = scale_c(
                                &prod,
                                phi(v, tt - jja) * rm(&rowmult, v, tt) * ikern(v, 0),
                            );
                        }
                        state_r = t;
                    }
                    Anchor::Tau(mma) => {
                        let mut t =
                            vec![DenseMat::zeros_shape(chain.ops[v].nrows(), dim1); n as usize + 1];
                        for tt in 0..=n {
                            let prod = chain.ops[v].mat_mul(&vt_blk(g1, *mma, tt));
                            t[tt as usize] =
                                scale_c(&prod, phi(v, tt - jj) * rm(&rowmult, v, tt) * ikern(v, 0));
                        }
                        state_r = t;
                    }
                }
                have_r = true;
            }
            _ => unreachable!(),
        }
    }
    // I-vertex upper bound: the tau anchor confines its I-block to
    // [0, tau_m]; the other anchors use the full imaginary axis
    let i_max = match anchor {
        Anchor::Tau(mma) => *mma,
        _ => ntau,
    };
    // remaining internal vertices 2..=2m-2
    for v in 2..=(2 * m - 2) {
        let g = &pp_g[chain.sectors[if rev_i { 2 * m - v } else { v }]];
        let bv = block_of(v);
        let bprev = block_of(v - 1);
        match (bprev, bv) {
            (0, 0) => {
                // L -> L: descending Volterra with greater-slot segments
                let (r0, c0) = (state_l[0].nrows(), state_l[0].ncols());
                let mut tn = vec![DenseMat::zeros_shape(chain.ops[v].nrows(), c0); j as usize + 1];
                for t2 in 0..=j {
                    // integrate t1 in [t2, j]
                    let len = (j - t2) as usize;
                    let mut f: Vec<Vec<Complex>> = vec![vec![Complex::ZERO; r0 * c0]; len + 1];
                    for q in 0..=len {
                        let t1 = t2 + q as i64;
                        let prod = gtr_blk(g, t1, t2).mat_mul(&state_l[t1 as usize]);
                        f[q as usize] = prod.raw().to_vec();
                    }
                    let integral = quad(&f, r0, c0, len);
                    let prod = chain.ops[v].mat_mul(&integral);
                    tn[t2 as usize] = scale_c(&prod, phi(v, t2) * ikern(v, 0));
                }
                state_l = tn;
            }
            (0, 1) => {
                // L -> I: tv kernel crossing
                let (r0, c0) = (state_l[0].nrows(), state_l[0].ncols());
                let mut ti =
                    vec![DenseMat::zeros_shape(chain.ops[v].nrows(), c0); ntau as usize + 1];
                for tau in 0..=ntau {
                    let mut f: Vec<Vec<Complex>> =
                        vec![vec![Complex::ZERO; r0 * c0]; j as usize + 1];
                    for tt in 0..=j {
                        let index = if reflect_mixed || (forward_less && forward_130) {
                            ntau - tau
                        } else {
                            tau
                        };
                        let prod = tv_blk(g, tt, index).mat_mul(&state_l[tt as usize]);
                        f[tt as usize] = prod.raw().to_vec();
                    }
                    let integral = quad(&f, r0, c0, j as usize);
                    let prod = chain.ops[v].mat_mul(&integral);
                    ti[tau as usize] = scale_c(&prod, phi(v, 0) * ikern(v, tau));
                }
                state_i = ti;
                have_i = true;
                have_l = false;
            }
            (0, 2) => {
                // L -> R: lesser crossing
                let (r0, c0) = (state_l[0].nrows(), state_l[0].ncols());
                let mut tr = vec![DenseMat::zeros_shape(chain.ops[v].nrows(), c0); n as usize + 1];
                for tt in 0..=n {
                    let mut f: Vec<Vec<Complex>> =
                        vec![vec![Complex::ZERO; r0 * c0]; j as usize + 1];
                    for tl in 0..=j {
                        let prod = les_blk(g, tl, tt).mat_mul(&state_l[tl as usize]);
                        f[tl as usize] = prod.raw().to_vec();
                    }
                    let integral = quad(&f, r0, c0, j as usize);
                    let prod = chain.ops[v].mat_mul(&integral);
                    tr[tt as usize] =
                        scale_c(&prod, phi(v, tt) * rm(&rowmult, v, tt) * ikern(v, 0));
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
                if rev_i {
                    // RIGHTWARD accumulation: state.mat then
                    // .ops[veff] (the rev algebra; see the seed)
                    let (r0r, c0r) = (state_i[0].nrows(), state_i[0].ncols());
                    let mut ti = vec![
                        DenseMat::zeros_shape(r0r, chain.ops[veff].ncols());
                        ntau as usize + 1
                    ];
                    for tau2 in 0..=i_max {
                        let mut f: Vec<Vec<Complex>> =
                            vec![vec![Complex::ZERO; r0r * c0r]; tau2 as usize + 1];
                        for q in 0..=tau2 as usize {
                            let prod = state_i[q].mat_mul(&mat_blk(g, tau2 - q as i64));
                            f[q] = prod.raw().to_vec();
                        }
                        let integral = quad(&f, r0r, c0r, tau2 as usize);
                        let prod = integral.mat_mul(&chain.ops[veff]);
                        ti[tau2 as usize] = scale_c(&prod, phi(veff, 0) * ikern(veff, tau2));
                    }
                    state_i = ti;
                } else {
                    let (r0, c0) = (state_i[0].nrows(), state_i[0].ncols());
                    let rows = if v2p { r0 } else { chain.ops[veff].nrows() };
                    let mut ti = vec![DenseMat::zeros_shape(rows, c0); ntau as usize + 1];
                    // THE MIX DIRECTION FIX (XCA_41D=down): the
                    // physical imaginary walk runs DOWNWARD from the
                    // anchor (mm) to the real junction (0) - the
                    // anchor-adjacent vertex (the seed's) sits at
                    // HIGHER times than the real-adjacent (the
                    // close's). The Volterra then integrates q in
                    // [tau2..=i_max] with mat(q - tau2), matching
                    // the stock's triangle {tau2 <= tau1} (the data
                    // probe verified the (4,1) factorization exact;
                    // the ordering was the application error)
                    let downward = matches!(anchor, Anchor::Tau(_))
                        && std::env::var("XCA_41D").as_deref() == Ok("down");
                    for tau2 in 0..=i_max {
                        if downward {
                            let len = (i_max - tau2) as usize + 1;
                            let mut f: Vec<Vec<Complex>> = vec![vec![Complex::ZERO; r0 * c0]; len];
                            for q in tau2..=i_max {
                                let prod = mat_blk(g, q - tau2).mat_mul(&state_i[q as usize]);
                                f[(q - tau2) as usize] = prod.raw().to_vec();
                            }
                            let integral = quad(&f, r0, c0, len - 1);
                            let prod = if v2p {
                                integral
                            } else {
                                chain.ops[veff].mat_mul(&integral)
                            };
                            ti[tau2 as usize] = scale_c(&prod, phi(veff, 0) * ikern(veff, tau2));
                        } else {
                            let mut f: Vec<Vec<Complex>> =
                                vec![vec![Complex::ZERO; r0 * c0]; tau2 as usize + 1];
                            for q in 0..=tau2 as usize {
                                let prod = mat_blk(g, tau2 - q as i64).mat_mul(&state_i[q]);
                                f[q] = prod.raw().to_vec();
                            }
                            let integral = quad(&f, r0, c0, tau2 as usize);
                            if std::env::var("XCA_V2DBG").is_ok()
                                && (tau2 == 0 || tau2 == 4 || tau2 == 12 || tau2 == 24)
                            {
                                eprintln!(
                                    "  ItoIint tau2={tau2}: {:?}",
                                    &integral.raw()[..4.min(integral.raw().len())]
                                );
                            }
                            let prod = if v2p {
                                integral
                            } else {
                                chain.ops[veff].mat_mul(&integral)
                            };
                            ti[tau2 as usize] =
                                scale_c(&prod, phi(veff, 0) * ikern(veff, tau2));
                        }
                    }
                    state_i = ti;
                }
            }
            (1, 2) => {
                // I -> R: vt crossing
                let (r0, c0) = (state_i[0].nrows(), state_i[0].ncols());
                let mut tr = vec![DenseMat::zeros_shape(chain.ops[v].nrows(), c0); n as usize + 1];
                for tt in 0..=n {
                    let mut f: Vec<Vec<Complex>> =
                        vec![vec![Complex::ZERO; r0 * c0]; ntau as usize + 1];
                    for tau in 0..=i_max {
                        let index = if reflect_mixed || forward_less || forward_contact {
                            i_max - tau
                        } else {
                            tau
                        };
                        let prod = vt_blk(g, index, tt).mat_mul(&state_i[tau as usize]);
                        f[tau as usize] = prod.raw().to_vec();
                    }
                    // Causal slice: the startup rule must see only the
                    // nodes actually present; a zero-padded ntau+1
                    // buffer makes poly_integral_chain take the xiw
                    // branch and read the zeros as data.
                    let integral = quad(&f[..=i_max as usize], r0, c0, i_max as usize);
                    let prod = chain.ops[v].mat_mul(&integral);
                    tr[tt as usize] =
                        scale_c(&prod, phi(v, tt) * rm(&rowmult, v, tt) * ikern(v, 0));
                }
                state_r = tr;
                have_r = true;
                have_i = false;
            }
            (2, 2) => {
                // R -> R: greater Volterra; the JUpper cut starts at
                // the relative lower bound j
                let lb = match anchor {
                    Anchor::JUpper(jx) => *jx as usize,
                    _ => 0,
                };
                let (r0, c0) = (state_r[0].nrows(), state_r[0].ncols());
                let mut tn = vec![DenseMat::zeros_shape(chain.ops[v].nrows(), c0); n as usize + 1];
                for nsteps in (lb + 1)..=n as usize {
                    let mut f: Vec<Vec<Complex>> =
                        vec![vec![Complex::ZERO; r0 * c0]; nsteps - lb + 1];
                    for u in lb..=nsteps {
                        // ABSOLUTE-ARGUMENT read for the les context
                        let gv = match anchor {
                            // MIX reads the CONTINUATION branch too
                            // (mirrors the scalar fix: the stock's
                            // data_bc reads the -ret*+less branch; the
                            // elapsed form differs by the real-part
                            // sign)
                            Anchor::JLower(_) | Anchor::Tau(_) => {
                                gtr_blk(g, u as i64, nsteps as i64)
                            }
                            _ => gtr_blk(g, (nsteps - u) as i64, 0),
                        };
                        let prod = gv.mat_mul(&state_r[u]);
                        f[u - lb] = prod.raw().to_vec();
                    }
                    let integral = quad(&f, r0, c0, nsteps - lb);
                    let trel = nsteps as i64 - jj;
                    let prod = chain.ops[v].mat_mul(&integral);
                    tn[nsteps] = scale_c(
                        &prod,
                        phi(v, trel) * rm(&rowmult, v, nsteps as i64) * ikern(v, 0),
                    );
                }
                state_r = tn;
            }
            _ => unreachable!("invalid block transition"),
        }
    }
    // closing: last internal vertex (or the anchor when 2m-2 = 0)
    // -> n, plus the external vertex factors
    let vlast = 2 * m - 2;
    let gfin = &pp_g[chain.sectors[2 * m - 1]];
    if vlast == 0 {
        // m = 1 boundary: not used (NCA path)
        return DenseMat::zeros_shape(chain.ops[0].ncols(), chain.ops[0].ncols());
    }
    let (r0, c0) = match block_of(vlast) {
        1 if v2p => (chain.ops[3].ncols(), chain.ops[0].nrows()),
        0 => (state_l[0].nrows(), state_l[0].ncols()),
        1 => (state_i[0].nrows(), state_i[0].ncols()),
        _ => (state_r[0].nrows(), state_r[0].ncols()),
    };
    let mut total = DenseMat::zeros_shape(r0, c0);
    match block_of(vlast) {
        0 => {
            // L -> n: lesser read (t, n)
            let mut f: Vec<Vec<Complex>> = vec![vec![Complex::ZERO; r0 * c0]; j as usize + 1];
            for tl in 0..=j {
                let prod = les_blk(gfin, tl, n).mat_mul(&state_l[tl as usize]);
                f[tl as usize] = prod.raw().to_vec();
            }
            total = quad(&f, r0, c0, j as usize);
        }
        1 => {
            // I -> n: vt read (tau, n); the reversed all-I cut closes
            // with the j-adjacent segment's tv read (ga); the TAU
            // reversal closes RIGHTWARD with the anchor mat walk
            // mat(ga, mm - tau)
            let mut f: Vec<Vec<Complex>> = vec![vec![Complex::ZERO; r0 * c0]; ntau as usize + 1];
            if let Anchor::Tau(mma) = anchor {
                if rev_i {
                    // the TAU reversal close: RIGHTWARD anchor mat
                    // walk (pairs with the rev seed)
                    let ga = &pp_g[chain.sectors[1]];
                    for tau in 0..=i_max {
                        let prod = state_i[tau as usize].mat_mul(&mat_blk(ga, *mma - tau));
                        f[tau as usize] = prod.raw().to_vec();
                    }
                } else if (v2p || std::env::var("XCA_TAU_V2").is_ok())
                    && cut.nl == 0
                    && cut.ni == 2
                    && cut.nr == 0
                {
                    // V2 FINAL (the LEFTWARD accumulation - the
                    // string ops[3].MAT.ops[2].mat2.ops[1].VT.ops[0]):
                    // the close reads the ANCHOR mat walk of its OWN
                    // sector mat(gfin, mm - tau) - the MAT slot at
                    // the ops[3]-side
                    // XCA_TAU_V2S: the sector swap - the mat from
                    // sectors[1] (the transcription's ga@sa)
                    let gclose = if v2p || std::env::var("XCA_TAU_V2S").is_ok() {
                        &pp_g[chain.sectors[1]]
                    } else {
                        gfin
                    };
                    for tau in 0..=i_max {
                        let prod = if v2p {
                            // the transpose-aware placement: f =
                            // ti^T.ops[1].mat(m-tau) - the (s3 x s1)
                            // orientation for the final sandwich
                            // (ops[2m-1].total.ops[0]); ti^T = the
                            // adjoint-then-conj (the plain transpose)
                            let mut tr = state_i[tau as usize].adjoint();
                            for q in tr.raw_mut().iter_mut() {
                                *q = q.conj();
                            }
                            let mut mat = mat_blk(gclose, *mma - tau).adjoint();
                            for q in mat.raw_mut().iter_mut() {
                                *q = q.conj();
                            }
                            tr.mat_mul(&chain.ops[1]).mat_mul(&mat)
                        } else {
                            mat_blk(gclose, *mma - tau).mat_mul(&state_i[tau as usize])
                        };
                        if std::env::var("XCA_V2DBG").is_ok() {
                            eprintln!("  v2close tau={tau}: f0 {:?}", prod.raw()[0]);
                        }
                        f[tau as usize] = prod.raw().to_vec();
                    }
                } else {
                    // forward close: the n-side vt read (leftward)
                    for tau in 0..=i_max {
                        let index = if forward_contact { *mma - tau } else { tau };
                        let prod = vt_blk(gfin, index, n).mat_mul(&state_i[tau as usize]);
                        f[tau as usize] = prod.raw().to_vec();
                    }
                }
            } else if rev_i {
                let ga = &pp_g[chain.sectors[1]];
                for tau in 0..=i_max {
                    // RIGHTWARD: state.tv-read (the rev algebra)
                    let prod = state_i[tau as usize].mat_mul(&tv_blk(ga, jj, tau));
                    f[tau as usize] = prod.raw().to_vec();
                }
            } else {
                for tau in 0..=i_max {
                    let index = if forward_less { ntau - tau } else { tau };
                    let prod = vt_blk(gfin, index, n).mat_mul(&state_i[tau as usize]);
                    f[tau as usize] = prod.raw().to_vec();
                }
            }
            total = quad(&f[..=i_max as usize], r0, c0, i_max as usize);
        }
        _ => {
            // R -> n: greater slot (n, t); JUpper closes over the
            // relative interval [j, n]
            let lb = match anchor {
                Anchor::JUpper(jx) => *jx as usize,
                _ => 0,
            };
            let mut f: Vec<Vec<Complex>> = vec![vec![Complex::ZERO; r0 * c0]; n as usize + 1 - lb];
            for u in lb..=n as usize {
                let gv_close = match anchor {
                    Anchor::JUpper(_) => gtr_blk(gfin, n, u as i64),
                    _ => gtr_blk(gfin, u as i64, n),
                };
                let prod = gv_close.mat_mul(&state_r[u]);
                if skip_r_end && u as i64 == n && !matches!(anchor, Anchor::JUpper(_)) {
                    f[u - lb] = vec![Complex::ZERO; prod.raw().len()];
                } else {
                    f[u - lb] = prod.raw().to_vec();
                }
            }
            total = quad(&f, r0, c0, n as usize - lb);
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
    // propagator segment; the tau anchor enters its I-block BACKWARD
    // (an extra i^{ni} for odd ni)
    let mut oric = Complex::new(if (1 + cut.nl) % 2 == 0 { 1.0 } else { -1.0 }, 0.0);
    for _ in 0..cut.ni {
        oric *= Complex::I;
    }
    for _ in 0..cut.ni.saturating_sub(1) {
        oric *= Complex::I;
    }
    if matches!(anchor, Anchor::Tau(_)) && cut.ni % 2 == 1 {
        oric *= Complex::I;
    }
    // Diagnostic: restore the third-order vertex phase relative to
    // OCA and convert imaginary bath lines from Matsubara storage.
    if m == 3
        && matches!(anchor, Anchor::JLower(_))
        && (cut.nl, cut.ni, cut.nr) == (1, 2, 1)
        && std::env::var_os("XCA_M3_CONTOUR_PHASE").is_some()
    {
        oric *= -Complex::I;
    }
    // The m=3 MM phase correction is DEFAULT ON since 2026-11-23 (user
    // approved), for the LES family only. At m=3 every JLower (cut,
    // topology) WITHOUT a Matsubara (imaginary-imaginary) chord comes out
    // a quarter-turn off. The L<->R mirror relation of the les family,
    // Sigma_(nl,ni,nr) = -conj(Sigma_(nr,ni,nl))^T (exact in the
    // native-OCA-validated m=2 architecture), holds only for topologies
    // carrying exactly one (1,1) mat chord (the contact-validated
    // all-imaginary cut and the large topology of (0,3,1)/(1,3,0)); all
    // MM = 0 topologies flip its sign. The +i rotation restores the
    // relation: the equal-time anti-Hermiticity defect drops 566x
    // (3.77e-5 -> 6.67e-8) and the sign is fixed physically by that
    // criterion. The greater/ret (JUpper) family and the mixed (Tau)
    // anchor keep the legacy phase: the JUpper cross-architecture gate
    // (xca_m3_cuts_match_chain_twoband) is machine-exact without the
    // rotation and breaks by |1-i| with it, and the les-vs-mixed
    // junction residual is marginally worse with it.
    // XCA_M3_MM_PHASE=off restores the legacy uncorrected les form;
    // XCA_M3_MM_PHASE=-i selects the other quarter-turn (A/B studies).
    if m == 3 && matches!(anchor, Anchor::JLower(_)) {
        let has_mat = pairs
            .iter()
            .any(|&(a, b)| block_of(a) == 1 && block_of(b) == 1);
        if !has_mat {
            let v = std::env::var("XCA_M3_MM_PHASE").unwrap_or_else(|_| "i".to_string());
            match v.as_str() {
                "off" | "none" | "0" => {}
                "-i" => oric *= -Complex::I,
                _ => oric *= Complex::I,
            }
        }
    }
    let meas = Complex::new(
        h.powi((cut.nl + cut.nr) as i32) * dtau.powi(cut.ni as i32),
        0.0,
    );
    // the matrix association of the scalar return
    // `cprod*oric*meas*closer*ops[0]*total*ops[2m-1]`: the external
    // vertex op LEFT-multiplies the accumulated state and ops[0]
    // (the anchor-side external vertex) RIGHT-multiplies - the
    // evalm-validated algebra (ops[0] rightmost)
    let val = chain.ops[2 * m - 1].mat_mul(&total).mat_mul(&chain.ops[0]);
    // THE LES/TAU TRANSPOSE: the scalar prototype's textual order
    // (`ops * read`, scalars commute) carries a hidden orientation -
    // mirroring it as the forward string yields the TRANSPOSED
    // family value for the les/vt-class anchors (verified
    // machine-exact on the rotated two-band fixture: every
    // off-diagonal entry got = want^T at 1.0000; diagonal entries
    // and 1x1 sectors are transpose-blind - why the single-band
    // discriminator and the corner sectors passed). The correct
    // orientation for JLower/Tau is the transposed product (the
    // stock's les families place the reads LEFT of the ops; the G
    // blocks of real-basis Hamiltonians are symmetric, making the
    // plain transpose the exact general relation)
    let val = if forward_less || !matches!(anchor, Anchor::JLower(_)) {
        val
    } else {
        let mut t = val.adjoint();
        for q in t.raw_mut().iter_mut() {
            *q = q.conj();
        }
        t
    };
    let mut out = scale_c(&val, cprod * oric * meas * closer);
    if matches!(anchor, Anchor::Tau(_)) {
        // MIX ADJOINT-SCATTER (closed 2026-11-20, the factor-6
        // campaign): the stock writes lmix +=
        // pref*[op1.S.op4]^dag with op1.S.op4 = my^T, so the
        // written value = conj(my) (entrywise, NO transpose)
        // times the global -1 - the exact matrix lift of the
        // scalar -conj fix. MUST apply AFTER the decorations
        // (cprod is complex - conjugating before scale_c breaks
        // the 1x1 consistency; that was the 800-point
        // discriminator divergence). XCA_TAU_ORI=id restores the
        // raw form for A/B diagnostics
        // XCA_TAU_ORI: id = raw; T = -transpose; A = -adjoint;
        // default (anything else) = -conj-no-transpose (the C form)
        let ori = std::env::var("XCA_TAU_ORI").unwrap_or_default();
        if ori != "id" {
            match ori.as_str() {
                "T" => {
                    let t = out.adjoint();
                    for q in t.raw().iter() {
                        let _ = q;
                    }
                    let mut tt = t.clone();
                    for q in tt.raw_mut().iter_mut() {
                        *q = q.conj();
                    }
                    // t was adjoint; conj(adjoin) = transpose; negate
                    for q in tt.raw_mut().iter_mut() {
                        *q = -*q;
                    }
                    out = tt;
                }
                "A" => {
                    let mut t = out.adjoint();
                    for q in t.raw_mut().iter_mut() {
                        *q = -*q;
                    }
                    out = t;
                }
                _ => {
                    for q in out.raw_mut().iter_mut() {
                        *q = -q.conj();
                    }
                }
            }
        }
    }
    out
}

/// Sum of ALL block cuts for the given anchor over topologies,
/// channels, matrix chains and pole combinations: one (dim x dim)
/// block per sector (natural values; the Tau mix write carries the
/// output sector's pseudo-particle sign).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sigma_cuts_m_mat(
    m: usize,
    hs: &HilbertSpace,
    pp_ints: &[Hybridization],
    bath: Option<&BosonBath>,
    proj: &[CsoeProj],
    pp_g: &[PseudoGreen],
    anchor: &Anchor,
    n: i64,
    h: f64,
    beta: f64,
    dtau: f64,
    kt: i64,
    integ: &Integrator,
) -> Vec<DenseMat> {
    let ns = hs.ns();
    let mut out: Vec<DenseMat> = (0..ns)
        .map(|s| DenseMat::zeros_shape(hs.sectors[s].dim, hs.sectors[s].dim))
        .collect();
    if m < 2 {
        return out;
    }
    let topos = connected_pairings(m);
    let cuts = families(m, anchor);
    // diagnostic knob: "nl,ni,nr" keeps only that family; a
    // leading '!' excludes it
    let cuts: Vec<_> = match std::env::var("XCA_MCUT") {
        // fam:<code>+<code>... : EXCLUDE whole cut families (diagnostic)
        // L = any lower-branch vertex, R = any upper-branch vertex,
        // L0/R0 = the same without the all-imaginary cut, all = everything
        Ok(f) if f.starts_with("fam:") => {
            let codes: Vec<&str> = f[4..].split('+').collect();
            let full = 2 * m - 2;
            let hit = |c: &FamCut| -> bool {
                codes.iter().any(|code| match *code {
                    "L" => c.nl >= 1,
                    "R" => c.nr >= 1,
                    "L0" => c.nl == 0 && c.ni < full,
                    "R0" => c.nr == 0 && c.ni < full,
                    // two or more imaginary vertices but not all-imaginary:
                    // the only cuts whose I->I kernels exist outside the
                    // reflected all-I walk
                    "MI" => c.ni >= 2 && c.ni < full,
                    "all" => true,
                    _ => false,
                })
            };
            cuts.into_iter().filter(|c| !hit(c)).collect()
        }
        Ok(f) => {
            let (excl, f) = match f.strip_prefix('!') {
                Some(r) => (true, r.to_string()),
                None => (false, f),
            };
            let p: Vec<usize> = f.split(',').map(|x| x.parse().unwrap()).collect();
            cuts.into_iter()
                .filter(|c| {
                    let hit = (c.nl, c.ni, c.nr) == (p[0], p[1], p[2]);
                    if excl { !hit } else { hit }
                })
                .collect()
        }
        Err(_) => cuts,
    };
    let chans = chans_with_bath(pp_ints, bath);
    let endpoint_audit = m == 3 && n == 1
        && (matches!(anchor, Anchor::JLower(0))
            || matches!(anchor, Anchor::Tau(mu) if *mu == pp_g[0].ntau as i64))
        && std::env::var_os("NQIST_TTI_ENDPOINT_AUDIT").is_some();
    let mut audit = (endpoint_audit || (m == 3 && n == 1 && matches!(anchor, Anchor::JLower(1))
        && std::env::var_os("NQIST_TTI_FIRST_ROW").is_some()))
        .then(|| vec![vec![out.clone(); cuts.len()]; topos.len()]);
    for (topology_index, topo) in topos.iter().enumerate() {
        for cs in ChannelIter::new(chans.len(), m) {
            for s0 in 0..ns {
                let Some(chain) =
                    super::boson_rt::build_rt_chain_mixed_m(hs, &topo.pairs, &cs, &chans, s0)
                else {
                    continue;
                };
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
                    // parity applies only when EVERY line is a
                    // fermion channel (the both-fermion crossing
                    // factor; boson-boson pairs carry none)
                    let parity_eff = if cs.iter().all(|&c| c < pp_ints.len()) {
                        topo.parity
                    } else {
                        1.0
                    };
                    let fac = Complex::new(msign * stat * parity_eff, 0.0);
                    for (index, cut) in cuts.iter().enumerate() {
                        let val = eval_cut_m(
                            m,
                            cut,
                            anchor,
                            &chain,
                            &topo.pairs,
                            &cs,
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
                        for q in 0..out[s0].len() {
                            out[s0].raw_mut()[q] += fac * val.raw()[q];
                        }
                        if let Some(blocks) = &mut audit {
                            for (dest, src) in blocks[topology_index][index][s0].raw_mut().iter_mut().zip(val.raw()) {
                                *dest += fac * *src;
                            }
                        }
                    }
                    // pole odometer
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
    // pseudo-particle sign (bosonic empty/doubly sectors +1,
    // fermionic singly-occupied -1)
    if matches!(anchor, Anchor::Tau(_)) {
        for s0 in 0..out.len() {
            let sg = Complex::new(pp_g[s0].sign.as_i64() as f64, 0.0);
            for q in 0..out[s0].len() {
                out[s0].raw_mut()[q] *= sg;
            }
        }
    }
    if let Some(blocks) = audit {
        if endpoint_audit {
            for (index, cut) in cuts.iter().enumerate().filter(|(_, cut)| cut.nl == 0) {
                for s in 0..ns {
                    let dim = hs.sectors[s].dim;
                    for row in 0..dim {
                        for col in 0..dim {
                            let value = blocks.iter().map(|topology| {
                                let block = &topology[index][s];
                                if matches!(anchor, Anchor::Tau(_)) {
                                    block.get(col, row).conj() * pp_g[s].sign.as_i64() as f64
                                } else {
                                    block.get(row, col)
                                }
                            }).fold(Complex::ZERO, |a, b| a + b);
                            eprintln!("endpoint {anchor:?} {} {s} {row} {col} {:.16e} {:.16e}", cut.ni, value.re, value.im);
                        }
                    }
                }
            }
            return out;
        }
        for (topo, cut_blocks) in topos.iter().zip(blocks) {
        for (cut, sectors) in cuts.iter().zip(cut_blocks) {
            if (cut.nl, cut.ni, cut.nr) != (1, 2, 1) {
                continue;
            }
            let mut norm = 0.0_f64;
            let mut defect = 0.0_f64;
            for block in sectors {
                for row in 0..block.nrows() {
                    for col in 0..block.ncols() {
                        let value = block.get(row, col);
                        norm = norm.max(value.abs());
                        defect = defect.max((value + block.get(col, row).conj()).abs());
                    }
                }
            }
            if norm > 0.0 {
                eprintln!("lesser cut (1,2,1) pairs={:?} norm={norm:.6e} defect={defect:.6e}", topo.pairs);
            }
        }
        }
    }
    out
}
