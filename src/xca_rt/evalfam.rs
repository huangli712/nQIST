// Project : nQIST (real-time XCA)
// File    : src/xca_rt/evalfam.rs
// Purpose : general-order contour-family evaluation of the real-time
//           XCA self-energy, built in validated stages on top of the
//           greater chain (`evalm`).
//
//           Stage 1: the all-real families (block cut
//           (nl, ni, nr) = (0, 0, 2m-2)):
//             Anchor::JUpper -> greater (ret-slot) output [== evalm]
//             Anchor::JLower -> lesser  (reference les(1) analog)
//             Anchor::Tau    -> mixed   (reference vt(1) analog)
//
//           Conventions (m=2-anchored against the reference kernel):
//             - JUpper works in relative time u = t - j exactly like
//               evalm (numerical regression target).
//             - JLower / Tau work in absolute time t in [0, n]; the
//               anchor-side endpoint constants of the anchor chord
//               fold into the coefficient product:
//                 Delta^<(j, t) = sum_l c_les e^{-i w j} e^{+i w t}
//                 Delta^vt(m, t) = (fermion ? +1 : -1) * conj(c_tv)
//                                  * e^{-w (beta - tau_m)} e^{+i w t}
//             - all other chords are greater-flavored:
//               Delta^>(t2, t1) = sum_l c_gtr e^{-i w t2} e^{+i w t1}
//             - phases are evaluated at each vertex's own state time.
//             - orientation/stat/msign/parity assembly identical to
//               evalm: msign * stat * parity * (-(h^2)^{m-1}).

use super::evalm::{ChannelIter, build_rt_chain};
use super::fit::CsoeProj;
use super::gtr;
use super::quad::poly_integral_chain;
use crate::hilbert::HilbertSpace;
use crate::nca::Hybridization;
use crate::xca::topology::connected_pairings;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::PseudoGreen;
use kadanoffbaym::weights::Integrator;

/// External anchor of vertex 0.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Anchor {
    /// j on the upper branch: greater (ret-slot) output.
    JUpper(i64),
    /// j on the lower branch: lesser output.
    JLower(i64),
    /// tau_m on the imaginary branch: mixed (lmix) output.
    Tau(i64),
}

/// Lesser read (reference `pls` semantics).
#[inline]
fn les_slot(g: &PseudoGreen, a: i64, b: i64) -> Complex {
    if a <= b {
        g.less.raw_block_at(a, b)[0]
    } else {
        -g.less.raw_block_at(b, a)[0].conj()
    }
}

/// vt read with forward Matsubara index (reference `pvt`):
/// `Delta^vt(tau, t) = -sig * conj(lmix(t, ntau - tau))`.
#[inline]
fn vt_fwd(g: &PseudoGreen, tau: i64, t: i64) -> Complex {
    let ntau = g.ntau as i64;
    -Complex::new(g.sign.as_i64() as f64, 0.0) * g.lmix.raw_block_at(t, ntau - tau)[0].conj()
}

/// Stage-1 backbone: the anchored all-real family for one
/// (topology, channels, sector, pole-combo). `len` is the state-array
/// extent: (n - j) for JUpper, n for the absolute-time anchors.
#[allow(clippy::too_many_arguments)]
fn eval_backbone_anchored(
    m: usize,
    chain: &super::evalm::RtChain,
    pairs: &[(usize, usize)],
    b0: usize,
    cs: &[usize],
    combo: &[usize],
    proj: &[CsoeProj],
    pp_ints: &[Hybridization],
    pp_g: &[PseudoGreen],
    anchor: &Anchor,
    n: i64,
    j: i64,
    h: f64,
    beta: f64,
    dtau: f64,
    integ: &Integrator,
) -> Complex {
    // per-vertex real-phase weights (absolute-time convention; the
    // JUpper path evaluates them at relative times, which is
    // algebraically identical after the anchor constant fold)
    let mut wsum = vec![0.0f64; 2 * m];
    // coefficient product: anchor chord flavor + greater chords
    let mut cprod = Complex::ONE;
    for li in 0..m {
        let (e, l) = pairs[li];
        let p = &proj[cs[li]];
        let w = p.poles[combo[li]];
        if li == 0 {
            // anchor chord (0, b0): endpoint constants folded here
            match anchor {
                Anchor::JUpper(_) => {
                    cprod *= p.c_gtr[combo[0]];
                    // Delta^>(t_b0, j) = c e^{-i w t_b0} e^{+i w j}
                    // with relative times the anchor factor is 1
                }
                Anchor::JLower(_) => {
                    // Delta^<(j, t) = sum c_les e^{-i w (t - j)}:
                    // anchor constant e^{+i w j}, vertex factor
                    // e^{-i w t} (applied through wsum below)
                    cprod *= p.c_les[combo[0]] * (Complex::I * w * h * j as f64).exp();
                }
                Anchor::Tau(_) => {
                    let sgn = if pp_ints[cs[0]].fermion { 1.0 } else { -1.0 };
                    cprod *= Complex::new(sgn, 0.0)
                        * p.c_tv[combo[0]].conj()
                        * Complex::new((-w * (beta - dtau * (j as f64))).exp(), 0.0);
                }
            }
        } else {
            cprod *= p.c_gtr[combo[li]];
        }
        // endpoint phase weights in the ACCESSOR argument order:
        // first argument carries e^{-i w t}, second e^{+i w t}
        let (first, second) = if li == 0 {
            match anchor {
                // Delta^>(t_b0, j) = c e^{-i w t_b0} e^{+i w j}:
                // first = b0 (carries e^{-i w t}), anchor constant 1
                // in the relative-time form
                Anchor::JUpper(_) => (l, e),
                // Delta^<(j, t_b0) = c e^{-i w t_b0} e^{+i w j}:
                // first = b0 (carries e^{-i w t}), the anchor
                // constant e^{+i w j} is folded in cprod above
                Anchor::JLower(_) => (l, e),
                // Delta^vt(tau_m, t_b0) = conj(c_tv) e^{-w(beta-tau)}
                // e^{+i w t_b0}: the real endpoint carries e^{+i w t}
                Anchor::Tau(_) => (e, l),
            }
        } else {
            // greater chords: Delta^>(t2, t1) with the later vertex
            // as the FIRST argument
            (l, e)
        };
        if first != 0 {
            wsum[first] += -w;
        }
        if second != 0 {
            wsum[second] += w;
        }
        // the anchor-side endpoint constant is folded in cprod;
        // vertex 0 never appears in the state phases
    }
    let phi = |v: usize, t: i64| -> Complex {
        if wsum[v] == 0.0 {
            Complex::ONE
        } else {
            (-Complex::I * wsum[v] * h * t as f64).exp()
        }
    };
    // seed: anchor -> vertex 1 segment (anchor-flavored)
    let g1 = &pp_g[chain.sectors[1]];
    let len = match anchor {
        Anchor::JUpper(jj) => (n - jj) as usize,
        _ => n as usize,
    };
    let mut t = vec![Complex::ZERO; len + 1];
    match anchor {
        Anchor::JUpper(jj) => {
            for u in 0..=len {
                t[u] = phi(1, u as i64) * chain.ops[1] * gtr(g1, u as i64, 0);
            }
        }
        Anchor::JLower(jj) => {
            for u in 0..=len {
                t[u] = phi(1, u as i64) * chain.ops[1] * les_slot(g1, *jj, u as i64);
            }
        }
        Anchor::Tau(mm) => {
            for u in 0..=len {
                t[u] = phi(1, u as i64) * chain.ops[1] * vt_fwd(g1, *mm, u as i64);
            }
        }
    }
    // remaining vertices (including the final external vertex
    // 2m-1, whose op/phase apply at the state time): identical
    // greater Volterra to evalm
    for s in 1..(2 * m - 1) {
        let v = s + 1;
        // segment entering vertex v carries the sector after v-1
        let g = &pp_g[chain.sectors[v]];
        if v == b0 {
            let mut conv = vec![Complex::ZERO; len + 1];
            for nsteps in 1..=len {
                let mut f = vec![Complex::ZERO; nsteps + 1];
                for u in 0..=nsteps {
                    f[u] = gtr(g, (nsteps - u) as i64, 0) * t[u];
                }
                conv[nsteps] = poly_integral_chain(&f, 0, nsteps, integ);
            }
            let mut tn = vec![Complex::ZERO; len + 1];
            for u in 0..=len {
                tn[u] = phi(b0, u as i64) * chain.ops[b0] * (conv[u] * chain.ops[0]);
            }
            t = tn;
            continue;
        }
        let mut tn = vec![Complex::ZERO; len + 1];
        for nsteps in 1..=len {
            let mut f = vec![Complex::ZERO; nsteps + 1];
            for u in 0..=nsteps {
                f[u] = gtr(g, (nsteps - u) as i64, 0) * t[u];
            }
            let integral = poly_integral_chain(&f, 0, nsteps, integ);
            tn[nsteps] = phi(v, nsteps as i64) * chain.ops[v] * integral;
        }
        t = tn;
    }
    cprod * t[len]
}

/// Sum over topologies, channels, sectors and pole combinations of
/// the stage-1 all-real family. Returns the natural (undecorated)
/// value per sector.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sigma_allreal_m(
    m: usize,
    hs: &HilbertSpace,
    pp_ints: &[Hybridization],
    proj: &[CsoeProj],
    pp_g: &[PseudoGreen],
    anchor: &Anchor,
    n: i64,
    h: f64,
    beta: f64,
    dtau: f64,
    integ: &Integrator,
) -> Vec<Complex> {
    let ns = hs.ns();
    let mut out = vec![Complex::ZERO; ns];
    if m < 2 {
        return out;
    }
    let j = match anchor {
        Anchor::JUpper(jj) | Anchor::JLower(jj) => *jj,
        Anchor::Tau(mm) => *mm,
    };
    for topo in connected_pairings(m) {
        let pairs = topo.pairs.clone();
        let b0 = pairs[0].1;
        for cs in ChannelIter::new(pp_ints.len(), m) {
            for s0 in 0..ns {
                let Some(chain) = build_rt_chain(hs, &pairs, &cs, pp_ints, s0) else {
                    continue;
                };
                let mut combo = vec![0usize; m];
                loop {
                    let val = eval_backbone_anchored(
                        m, &chain, &pairs, b0, &cs, &combo, proj, pp_ints, pp_g, anchor, n, j, h,
                        beta, dtau, integ,
                    );
                    let mut stat = 1.0f64;
                    for &c in &cs {
                        if pp_ints[c].fermion && pp_ints[c].backward {
                            stat *= -1.0;
                        }
                    }
                    let msign = if m % 2 == 0 { 1.0 } else { -1.0 };
                    out[s0] += Complex::new(msign * stat * topo.parity, 0.0)
                        * Complex::new(-(h * h).powi(m as i32 - 1), 0.0)
                        * val;
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
    out
}
