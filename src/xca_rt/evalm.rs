// Project : nQIST (real-time XCA)
// File    : src/xca_rt/evalm.rs
// Purpose : arbitrary-order all-greater family of the real-time XCA
//           expansion: the direct generalization of the validated
//           m = 2 ret(1) structure.
//
//           Backbone: connected chord topologies (reused from the
//           Matsubara module); sector chains walked on the operator
//           blocks; hybridization lines 1..m-1 are pole-separated
//           with vertex-time phases `exp(-/+i w t_v)` while line 0
//           (touching the external base vertex) keeps raw greater
//           reads. Time integrals are nested causal Volterra chains
//           in the relative time u in [0, n-j] with the reference
//           sub-interval quadrature (quad.rs).
//
//           At m = 2 the result must agree with the ret family of
//           `eval2` / the reference kernel (calibration); at m >= 3
//           it is validated against `direct_gtr_m` (recursive direct
//           transcription). Scalar (1x1) blocks, steady input.

use super::fit::CsoeFit;
use super::gtr;
use crate::hilbert::HilbertSpace;
use crate::nca::Hybridization;
use crate::xca::topology::connected_pairings;
use crate::xca_rt::quad::poly_integral_chain;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{ContourGreen, PseudoGreen};
use kadanoffbaym::weights::Integrator;

/// Scalar sector chain of a real-time backbone: sector after each
/// vertex and the (1x1) operator block applied at it.
pub(crate) struct RtChain {
    pub sectors: Vec<usize>,
    pub ops: Vec<Complex>,
}

/// Walk the vertices of the backbone for channel assignment `cs`
/// from starting sector `s0`; `None` when an operator block is
/// missing or the chain does not close. Vertices: 0 hosts the
/// annihilation operator of line 0, its partner `b0` the creation
/// one; interior vertices carry the pair-owning channel's operators
/// (earlier endpoint -> annihilation, later -> creation), exactly as
/// the Matsubara sigma backbone.
pub(crate) fn build_rt_chain(
    hs: &HilbertSpace,
    pairs: &[(usize, usize)],
    cs: &[usize],
    pp_ints: &[Hybridization],
    s0: usize,
) -> Option<RtChain> {
    let m = pairs.len();
    let nv = 2 * m;
    // owning line of each vertex (0 = earlier side, 1 = later side)
    let mut owner = vec![(0usize, false); nv];
    for (li, &(a, b)) in pairs.iter().enumerate() {
        owner[a] = (li, false);
        owner[b] = (li, true);
    }
    let mut sectors = vec![0usize; nv + 1];
    sectors[0] = s0;
    let mut ops = Vec::with_capacity(nv);
    for v in 0..nv {
        let (li, later) = owner[v];
        let flav = if later {
            pp_ints[cs[li]].op_cre_fl
        } else {
            pp_ints[cs[li]].op_ann_fl
        };
        let (target, blk) = hs.ops[flav].block(sectors[v])?;
        if blk.raw().len() != 1 {
            // scalar prototype scope
            return None;
        }
        sectors[v + 1] = *target;
        ops.push(blk.get(0, 0));
    }
    if sectors[nv] != s0 {
        return None;
    }
    Some(RtChain { sectors, ops })
}

/// Greater component of the ContourGreen (sgtr semantics).
fn gtr_c(g: &ContourGreen, n: i64, j: i64) -> Complex {
    if j < n {
        g.ret.raw_block_at(n, j)[0] - g.less.raw_block_at(j, n)[0].conj()
    } else if j == n {
        g.ret.raw_block_at(n, n)[0] + g.less.raw_block_at(n, n)[0]
    } else {
        -g.ret.raw_block_at(j, n)[0].conj() + g.less.raw_block_at(n, j)[0]
    }
}

#[inline]
fn ph(w: f64, t_rel: i64, h: f64) -> Complex {
    (-Complex::I * w * h * t_rel as f64).exp()
}

/// Per-topology all-greater contribution (diagnostic surface).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sigma_gtr_xca_m_by_topo(
    m: usize,
    hs: &HilbertSpace,
    pp_ints: &[Hybridization],
    fits: &[CsoeFit],
    pp_g: &[PseudoGreen],
    n: i64,
    j: i64,
    h: f64,
    integ: &Integrator,
    topo_ix: usize,
) -> Vec<Complex> {
    let ns = hs.ns();
    let mut out = vec![Complex::ZERO; ns];
    let topos = connected_pairings(m);
    let topo = &topos[topo_ix];
    let pairs = topo.pairs.clone();
    let b0 = pairs[0].1;
    for cs in ChannelIter::new(pp_ints.len(), m) {
        for s0 in 0..ns {
            let Some(chain) = build_rt_chain(hs, &pairs, &cs, pp_ints, s0) else {
                continue;
            };
            let mut combo = vec![0usize; m];
            loop {
                let mut cprod = Complex::ONE;
                for li in 0..m {
                    cprod *= fits[cs[li]].coefs[combo[li]];
                }
                let val = eval_backbone(
                    m,
                    &chain,
                    &pairs,
                    b0,
                    &cs,
                    pp_ints,
                    fits,
                    &combo,
                    pp_g,
                    n,
                    j,
                    h,
                    (n - j) as usize,
                    integ,
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
                    * cprod
                    * val;
                let mut idx = 0;
                let mut wrapped = false;
                loop {
                    if idx == m {
                        wrapped = true;
                        break;
                    }
                    combo[idx] += 1;
                    if combo[idx] < fits[cs[idx]].poles.len() {
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
    out
}

/// Per-topology direct contribution (diagnostic surface).
#[allow(clippy::too_many_arguments)]
pub(crate) fn direct_gtr_m_by_topo(
    m: usize,
    hs: &HilbertSpace,
    pp_ints: &[Hybridization],
    pp_g: &[PseudoGreen],
    n: i64,
    j: i64,
    h: f64,
    integ: &Integrator,
    topo_ix: usize,
) -> Vec<Complex> {
    let ns = hs.ns();
    let mut out = vec![Complex::ZERO; ns];
    let topos = connected_pairings(m);
    let topo = &topos[topo_ix];
    let pairs = topo.pairs.clone();
    for cs in ChannelIter::new(pp_ints.len(), m) {
        for s0 in 0..ns {
            let Some(chain) = build_rt_chain(hs, &pairs, &cs, pp_ints, s0) else {
                continue;
            };
            let mut stat = 1.0f64;
            for &c in &cs {
                if pp_ints[c].fermion && pp_ints[c].backward {
                    stat *= -1.0;
                }
            }
            let msign = if m % 2 == 0 { 1.0 } else { -1.0 };
            let len = (n - j) as usize;
            let mut tv = vec![0usize; 2 * m];
            tv[0] = 0;
            tv[2 * m - 1] = len;
            let val = recurse_simplex(
                m,
                &chain,
                &pairs,
                &cs.iter().map(|&c| pp_ints[c].delta).collect::<Vec<_>>(),
                pp_g,
                j,
                &mut tv,
                2 * m - 2,
                integ,
            );
            out[s0] += Complex::new(msign * stat * topo.parity, 0.0)
                * Complex::new(-(h * h).powi(m as i32 - 1), 0.0)
                * val;
        }
    }
    out
}

/// MIXED fermion/boson direct transcription of the all-greater
/// order-m family (raw data reads through the RtChan view; a boson
/// line's in-triangle greater = gtr_c(d0, later, earlier), the
/// convention the JUpper/ret-class chords implement). Statistical
/// rules mirror sigma_cuts_m: boson channels carry no backward
/// minus, and the topology parity applies only when every line is
/// a fermion channel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn direct_gtr_m_mixed(
    m: usize,
    hs: &HilbertSpace,
    pp_ints: &[Hybridization],
    bath: Option<&crate::nca::BosonBath>,
    pp_g: &[PseudoGreen],
    n: i64,
    j: i64,
    h: f64,
    integ: &Integrator,
) -> Vec<Complex> {
    let ns = hs.ns();
    let mut out = vec![Complex::ZERO; ns];
    if m < 2 {
        return out;
    }
    let len = (n - j) as usize;
    let chans = super::boson_rt::chans_with_bath(pp_ints, bath);
    for topo in connected_pairings(m) {
        let pairs = topo.pairs.clone();
        for cs in ChannelIter::new(chans.len(), m) {
            for s0 in 0..ns {
                let Some(chain) =
                    super::boson_rt::build_rt_chain_mixed(hs, &pairs, &cs, &chans, s0)
                else {
                    continue;
                };
                let mut stat = 1.0f64;
                for &c in &cs {
                    if c < pp_ints.len() && pp_ints[c].backward {
                        stat *= -1.0;
                    }
                }
                let msign = if m % 2 == 0 { 1.0 } else { -1.0 };
                let parity_eff = if cs.iter().all(|&c| c < pp_ints.len()) {
                    topo.parity
                } else {
                    1.0
                };
                let mut tv = vec![0usize; 2 * m];
                tv[0] = 0;
                tv[2 * m - 1] = len;
                let val = recurse_simplex(
                    m,
                    &chain,
                    &pairs,
                    &cs.iter().map(|&c| chans[c].delta()).collect::<Vec<_>>(),
                    pp_g,
                    j,
                    &mut tv,
                    2 * m - 2,
                    integ,
                );
                out[s0] += Complex::new(msign * stat * parity_eff, 0.0)
                    * Complex::new(-(h * h).powi(m as i32 - 1), 0.0)
                    * val;
            }
        }
    }
    out
}

/// All-greater self-energy of order `m` at external times `(n, j)`,
/// summed over topologies, channel assignments, starting sectors and
/// pole combinations. Returns one scalar per sector.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sigma_gtr_xca_m(
    m: usize,
    hs: &HilbertSpace,
    pp_ints: &[Hybridization],
    fits: &[CsoeFit],
    pp_g: &[PseudoGreen],
    n: i64,
    j: i64,
    h: f64,
    integ: &Integrator,
) -> Vec<Complex> {
    let ns = hs.ns();
    let mut out = vec![Complex::ZERO; ns];
    if m < 2 {
        return out;
    }
    let len = (n - j) as usize;
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
                    // coefficient product of all separated lines
                    let mut cprod = Complex::ONE;
                    for li in 0..m {
                        cprod *= fits[cs[li]].coefs[combo[li]];
                    }
                    let val = eval_backbone(
                        m, &chain, &pairs, b0, &cs, pp_ints, fits, &combo, pp_g, n, j, h, len,
                        integ,
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
                        * cprod
                        * val;
                    // pole odometer over all m lines; done when it
                    // wraps back to the all-zero state
                    let mut idx = 0;
                    let mut wrapped = false;
                    loop {
                        if idx == m {
                            wrapped = true;
                            break;
                        }
                        combo[idx] += 1;
                        if combo[idx] < fits[cs[idx]].poles.len() {
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

/// Nested causal Volterra chain with vertex phases; `u` = relative
/// time from the base vertex (steps).
#[allow(clippy::too_many_arguments)]
fn eval_backbone(
    m: usize,
    chain: &RtChain,
    pairs: &[(usize, usize)],
    b0: usize,
    cs: &[usize],
    pp_ints: &[Hybridization],
    fits: &[CsoeFit],
    combo: &[usize],
    pp_g: &[PseudoGreen],
    n: i64,
    j: i64,
    h: f64,
    len: usize,
    integ: &Integrator,
) -> Complex {
    // vertex phase FUNCTIONS from the separated lines: at vertex v
    // with relative time t. IN-READ CONVENTION (2026-11-29 JUpper
    // audit): the ret-class greater families (the reference ret1,
    // direct_gtr_m's raw gtr_c(later, earlier) transcription) read
    // the IN-triangle greater, whose separated form is
    // c e^{-i w (t_l - t_e) h} - later endpoints e^{-i w t h},
    // earlier e^{+i w t h}. The legacy out-read convention
    // (e^{+iw(t_l-t_e)}, the pole-flipped reconstruction) left a
    // 3.5% external anchor residual and the m=3 separated-vs-direct
    // structural disagreement
    let mut phi_w = vec![(0.0f64, 1.0f64); 2 * m]; // (sum of +-w, sign)
    for li in 0..m {
        let (e, l) = pairs[li];
        let w = fits[cs[li]].poles[combo[li]];
        phi_w[l].0 += w;
        phi_w[e].0 += -w;
    }
    let phi = |v: usize, t: i64| -> Complex {
        if phi_w[v].0 == 0.0 {
            Complex::ONE
        } else {
            (-Complex::I * phi_w[v].0 * h * t as f64).exp()
        }
    };
    // segment propagator sector: after vertex v -> sectors[v+1]
    let seg = |v: usize| &pp_g[chain.sectors[v + 1]];
    // seed: vertex-1 factor (phase at time u) times G^>(u)
    let mut t = vec![Complex::ZERO; len + 1];
    let g1 = seg(0);
    for u in 0..=len {
        t[u] = phi(1, u as i64) * chain.ops[1] * gtr(g1, u as i64, 0);
    }
    // loop over the remaining vertices
    for s in 1..(2 * m - 1) {
        let v = s + 1;
        if v == b0 {
            // line 0 also carries a hybridization: fully separated
            // (pole in the combo, phase through phi). The segment
            // (v-1 -> b0) IS convolved first (the Matsubara special
            // step skipped it; here every segment carries a
            // propagator), then the line-0 ops and phase multiply
            let g = seg(s);
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
        // ordinary step: convolve by the segment propagator (sector
        // after vertex s), then apply the vertex-v operator+phase
        let g = seg(s);
        let opv = chain.ops[v];
        let mut tn = vec![Complex::ZERO; len + 1];
        for nsteps in 1..=len {
            let mut f = vec![Complex::ZERO; nsteps + 1];
            for u in 0..=nsteps {
                f[u] = gtr(g, (nsteps - u) as i64, 0) * t[u];
            }
            let integral = poly_integral_chain(&f, 0, nsteps, integ);
            tn[nsteps] = phi(v, nsteps as i64) * opv * integral;
        }
        t = tn;
    }
    // the loop already applied the final vertex (v = 2m-1) factor at
    // every step; the chain closes at u = len
    t[len]
}

/// Channel assignment odometer over `nch^m` combinations.
pub(crate) struct ChannelIter {
    nch: usize,
    m: usize,
    cs: Vec<usize>,
    started: bool,
    done: bool,
}

impl ChannelIter {
    pub fn new(nch: usize, m: usize) -> Self {
        ChannelIter {
            nch,
            m,
            cs: vec![0; m],
            started: false,
            done: nch == 0 || m == 0,
        }
    }
}

impl Iterator for ChannelIter {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Vec<usize>> {
        if self.done {
            return None;
        }
        if !self.started {
            self.started = true;
            return Some(self.cs.clone());
        }
        let mut idx = 0;
        loop {
            if idx == self.m {
                return None;
            }
            self.cs[idx] += 1;
            if self.cs[idx] < self.nch {
                return Some(self.cs.clone());
            }
            self.cs[idx] = 0;
            idx += 1;
        }
    }
}

/// Recursive direct transcription of the all-greater order-m diagram
/// (no separation, explicit simplex sum with the same quadrature):
/// the m = 2 instance must equal the reference ret family, and the
/// m >= 3 instances validate `sigma_gtr_xca_m` (separated == direct).
#[allow(clippy::too_many_arguments)]
pub(crate) fn direct_gtr_m(
    m: usize,
    hs: &HilbertSpace,
    pp_ints: &[Hybridization],
    pp_g: &[PseudoGreen],
    n: i64,
    j: i64,
    h: f64,
    integ: &Integrator,
) -> Vec<Complex> {
    let ns = hs.ns();
    let mut out = vec![Complex::ZERO; ns];
    if m < 2 {
        return out;
    }
    let len = (n - j) as usize;
    for topo in connected_pairings(m) {
        let pairs = topo.pairs.clone();
        for cs in ChannelIter::new(pp_ints.len(), m) {
            for s0 in 0..ns {
                let Some(chain) = build_rt_chain(hs, &pairs, &cs, pp_ints, s0) else {
                    continue;
                };
                let mut stat = 1.0f64;
                for &c in &cs {
                    if pp_ints[c].fermion && pp_ints[c].backward {
                        stat *= -1.0;
                    }
                }
                let msign = if m % 2 == 0 { 1.0 } else { -1.0 };
                // vertex times (relative steps), strictly ordered
                let mut tv = vec![0usize; 2 * m];
                tv[0] = 0;
                tv[2 * m - 1] = len;
                let val = recurse_simplex(
                    m,
                    &chain,
                    &pairs,
                    &cs.iter().map(|&c| pp_ints[c].delta).collect::<Vec<_>>(),
                    pp_g,
                    j,
                    &mut tv,
                    2 * m - 2,
                    integ,
                );
                out[s0] += Complex::new(msign * stat * topo.parity, 0.0)
                    * Complex::new(-(h * h).powi(m as i32 - 1), 0.0)
                    * val;
            }
        }
    }
    out
}

/// Nested causal Volterra transcription of the all-greater order-m
/// diagram (no separation): places the internal vertex times
/// OUTSIDE-IN (the last internal vertex first - the backbone's
/// nesting, where each step's integral covers the full prefix
/// [0, t_v]); each placement carries the Gregory weight
/// giw^{(t_above)}(t) of its point in the parent's integration
/// interval times the segment propagator G_{s_{v+1}}(t_above - t),
/// and the innermost level carries the raw greater delta reads
/// between each line's two placed vertex times, all operators, and
/// the seed segment [0, t_1].
/// CORRECTED 2026-11-29: the legacy inside-out form carried the
/// segment propagator BOTH in the placement weight (as a
/// sub-interval integral) AND in the bottom product (pointwise) -
/// every internal segment squared - and its loop association
/// inverted the backbone nesting; it sat 5-15x off the stock at
/// m=2 (and nonzero at j=n where the true value is 0).
/// The m = 2 instance must equal the reference ret family; the
/// m >= 3 instances validate `sigma_gtr_xca_m` (separated ==
/// direct). `deltas[li]` is line li's hybridization (fermion or
/// boson through the RtChan view).
#[allow(clippy::too_many_arguments)]
fn recurse_simplex(
    m: usize,
    chain: &RtChain,
    pairs: &[(usize, usize)],
    deltas: &[&ContourGreen],
    pp_g: &[PseudoGreen],
    j: i64,
    tv: &mut Vec<usize>,
    vnext: usize,
    integ: &Integrator,
) -> Complex {
    if vnext == 0 {
        // innermost: every line's raw greater read between its two
        // placed vertex times, all operators, the seed segment
        let mut acc = gtr(&pp_g[chain.sectors[1]], tv[1] as i64, 0);
        for v in 0..2 * m {
            acc *= chain.ops[v];
        }
        for li in 0..m {
            let (e, l) = pairs[li];
            acc *= gtr_c(deltas[li], j + tv[l] as i64, j + tv[e] as i64);
        }
        return acc;
    }
    // zero parent time: the backbone's post-seed chain value at 0 is
    // exactly zero (its step loops only write nsteps >= 1)
    if tv[vnext + 1] == 0 {
        return Complex::ZERO;
    }
    let above = tv[vnext + 1];
    let g = &pp_g[chain.sectors[vnext + 1]];
    let mut total = Complex::ZERO;
    for t in 0..=above {
        tv[vnext] = t;
        let w = integ.giw.at(above, t);
        if w == 0.0 {
            continue;
        }
        total += Complex::new(w, 0.0)
            * gtr(g, (above - t) as i64, 0)
            * recurse_simplex(m, chain, pairs, deltas, pp_g, j, tv, vnext - 1, integ);
    }
    total
}
