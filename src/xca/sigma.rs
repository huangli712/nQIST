// Project : nQIST
// File    : src/xca/sigma.rs
// Purpose : arbitrary-order (XCA) pseudo-particle self-energy
//           diagrams on the Matsubara axis
//
// Implements the backbone decomposition of Kaye/Huang/Strand/Golez
// (PRX 14, 031034 (2024); SciPost Phys. 19, 121 (2025)) in the nQIST
// sector conventions. A diagram of order `m` pairs the `2m` backbone
// vertices with `m` hybridization channels (each channel carries its
// direction-corrected Delta: backward channels already hold
// `Delta_cc(tau) = -Delta(beta - tau)`, i.e. `Delta(-tau)`). The line
// touching vertex 0 stays on the grid; every other line is separated
// into SOE kernels placed on the vertices (`w <= 0` poles: `K^-` at
// the earlier, `K^+` at the later endpoint, divided by `K^-(0)`) or
// on the edges (`w > 0` poles: `K^+` on each segment between the
// endpoints, divided by `K^+(0)^(edges - 1)`; the overflow-safe
// branch). The remaining chain of pointwise products and Volterra
// convolutions runs on the equispaced Matsubara grid with the same
// Gregory weights the pseudo-particle Dyson solvers use.
//
// Diagram prefactor: `i^2 * prod(stat_c) * parity(topology) * prod_l
// c_l / denom_l` with the per-channel statistics sign of the NCA/OCA
// kernels (`-1` for backward fermionic channels). At order 1 this
// reproduces `nca::set_sigma_mat` exactly; at order 2 it reproduces
// `oca::add_sigma_mat_oca` up to quadrature error.

use crate::hilbert::HilbertSpace;
use crate::xca::soe::SoeFit;
use crate::xca::topology::connected_pairings;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::types::PseudoGreen;
use kadanoffbaym::weights::Integrator;

/// Grid views shared by the diagram evaluations.
pub(crate) struct Grids<'a> {
    pub ntau: usize,
    pub dtau: f64,
    pub pp_g: &'a [PseudoGreen],
}

impl Grids<'_> {
    /// Sector dimension.
    #[inline]
    fn dim(&self, s: usize) -> usize {
        self.pp_g[s].size1
    }

    /// Matsubara block of sector `s` at grid point `i` (flat row
    /// major, `dim x dim`).
    #[inline]
    fn g_block(&self, s: usize, i: usize) -> &[Complex] {
        self.pp_g[s].mat.block(i)
    }
}

/// One resolved sector chain: sectors `sigma[v]` (before the vertex
/// operator, `2m + 1` entries with `[2m] == [0]`) and the operator
/// blocks at the vertices.
struct Chain<'a> {
    sectors: Vec<usize>,
    ops: Vec<&'a DenseMat>,
}

/// Resolve the sector chain of one diagram from the starting sector
/// `s0`. Operator placement follows the uniform rule shared with the
/// NCA/OCA kernels: the earlier endpoint of a line carries
/// `op_ann_fl`, the later endpoint `op_cre_fl` of its channel (for
/// the vertex-0 line the earlier endpoint is vertex 0).
fn build_chain<'a>(
    m: usize,
    pairs: &[(usize, usize)],
    channels: &[usize],
    pp_ints: &[crate::oca::OcaChannel<'a>],
    hs: &'a HilbertSpace,
    s0: usize,
) -> Option<Chain<'a>> {
    let mut owner = vec![(0usize, false); 2 * m];
    for (li, &(a, b)) in pairs.iter().enumerate() {
        owner[a] = (li, false);
        owner[b] = (li, true);
    }
    let mut sectors = vec![0usize; 2 * m + 1];
    sectors[0] = s0;
    let mut ops = Vec::with_capacity(2 * m);
    for v in 0..2 * m {
        let (li, later) = owner[v];
        let src = if later {
            &pp_ints[channels[li]].cre
        } else {
            &pp_ints[channels[li]].ann
        };
        let (target, block) = match src {
            crate::oca::VertSrc::Flavor(f) => hs.ops[*f].block(sectors[v])?,
            crate::oca::VertSrc::Diag(list) => list.iter().find(|(s, _)| *s == sectors[v])?,
        };
        sectors[v + 1] = *target;
        ops.push(block);
    }
    if sectors[2 * m] != s0 {
        return None;
    }
    Some(Chain { sectors, ops })
}

/// SOE kernel placement for the decomposed lines (topology lines
/// `1..m`; line 0 touches vertex 0 and stays on the grid).
#[derive(Clone)]
struct Kernels {
    /// `vertex[v] = Some((line, pole, sign))`: multiply the vertex
    /// operator by `K(tau_i, sign * w_l)`.
    vertex: Vec<Option<(usize, usize, f64)>>,
    /// `edge[s]`: `(line, pole)` pairs whose `K^+(tau_i, w_l)`
    /// multiplies segment `s` (between vertices `s` and `s + 1`).
    edge: Vec<Vec<(usize, usize)>>,
}

#[allow(clippy::needless_range_loop)]
fn build_kernels(
    m: usize,
    pairs: &[(usize, usize)],
    channels: &[usize],
    pole_l: &[usize],
    fits: &[SoeFit],
) -> Kernels {
    let mut vertex = vec![None; 2 * m];
    let mut edge = vec![Vec::new(); 2 * m - 1];
    for li in 1..m {
        let l = pole_l[li - 1];
        let fit_ix = channels[li];
        let (e, late) = pairs[li];
        let w = fits[fit_ix].poles[l];
        if w <= 0.0 {
            vertex[e] = Some((fit_ix, l, -1.0));
            vertex[late] = Some((fit_ix, l, 1.0));
        } else {
            for s in e..late {
                edge[s].push((fit_ix, l));
            }
        }
    }
    Kernels { vertex, edge }
}

/// Scalar kernel factor of vertex `v` at grid point `i`.
#[inline]
fn vertex_kernel(kern: &Kernels, v: usize, fits: &[SoeFit], i: usize, dtau: f64) -> f64 {
    match kern.vertex[v] {
        Some((li, l, sgn)) if sgn > 0.0 => fits[li].kp(i, l, dtau),
        Some((li, l, _)) => fits[li].km(i, l, dtau),
        None => 1.0,
    }
}

/// Product of edge kernel factors of segment `s` at grid point `i`.
#[inline]
fn edge_kernel(kern: &Kernels, s: usize, fits: &[SoeFit], i: usize, dtau: f64) -> f64 {
    let mut acc = 1.0;
    for &(li, l) in &kern.edge[s] {
        acc *= fits[li].kp(i, l, dtau);
    }
    acc
}

/// Volterra convolution `(L * T)(t_n) = int_0^{t_n} L(t_n - t_j)
/// T(t_j) dt_j` with Gregory weights; `L` is square (`rows x rows`),
/// `T` is `rows x cols`, and the result replaces `t` through `buf`.
fn convolve(
    ntau: usize,
    rows: usize,
    cols: usize,
    lpos: &[Complex],
    t: &mut [Complex],
    buf: &mut Vec<Complex>,
    integ: &Integrator,
) {
    debug_assert_eq!(lpos.len(), (ntau + 1) * rows * rows);
    debug_assert_eq!(t.len(), (ntau + 1) * rows * cols);
    buf.clear();
    buf.resize((ntau + 1) * rows * cols, Complex::ZERO);
    let k = integ.k as usize;
    for n in 1..=ntau {
        let off = n * rows * cols;
        if n >= k {
            for j in 0..=n {
                let w = integ.giw.at(n, j);
                if w == 0.0 {
                    continue;
                }
                let wc = Complex::new(w, 0.0);
                let li = (n - j) * rows * rows;
                let ti = j * rows * cols;
                for r in 0..rows {
                    for c in 0..cols {
                        let mut acc = Complex::ZERO;
                        for q in 0..rows {
                            acc += lpos[li + r * rows + q] * t[ti + q * cols + c];
                        }
                        buf[off + r * cols + c] += wc * acc;
                    }
                }
            }
        } else {
            // short-interval startup (blueprint sdiagram warmup): a
            // k-point sub-grid on [0, t_n] with the integrand built
            // from polynomial interpolation of the two factors over
            // the grid nodes 0..=k, then a single gregory row
            let dsub = n as f64 / k as f64;
            // interpolated samples f[l] = L(t_n - tau_l) T(tau_l)
            let mut f = vec![Complex::ZERO; (k + 1) * rows * cols];
            for l in 0..=k {
                let x = dsub * l as f64; // T argument, grid units
                let dl = n as f64 - x; // L difference, grid units
                for r in 0..rows {
                    for c in 0..cols {
                        let mut tv = Complex::ZERO;
                        for q in 0..rows {
                            let lv = poly_interp_elem(lpos, rows, r, q, dl, integ);
                            tv += lv * poly_interp_t(t, rows, cols, q, c, x, integ);
                        }
                        f[l * rows * cols + r * cols + c] = tv;
                    }
                }
            }
            for j in 0..=k {
                let w = integ.giw.at(k, j);
                if w == 0.0 {
                    continue;
                }
                let wc = Complex::new(w, 0.0);
                let fi = j * rows * cols;
                for q in 0..rows * cols {
                    buf[off + q] += wc * f[fi + q];
                }
            }
            let sc = Complex::new(dsub, 0.0);
            for q in 0..rows * cols {
                buf[off + q] *= sc;
            }
        }
    }
    t.copy_from_slice(buf);
}

/// Polynomial interpolation of the `L` buffer element `(r, q)` at the
/// fractional difference `x` (grid units), stencil over the nodes
/// `0..=k` of the difference index (blueprint `poly_interpol`).
#[inline]
pub(crate) fn poly_interp_elem(
    lpos: &[Complex],
    rows: usize,
    r: usize,
    q: usize,
    x: f64,
    integ: &Integrator,
) -> Complex {
    let k = integ.k as usize;
    let mut sum = Complex::ZERO;
    let mut xp = 1.0;
    for l in 0..=k {
        let mut fl = Complex::ZERO;
        for i in 0..=k {
            let w = integ.piw.at(l, i);
            if w != 0.0 {
                fl += Complex::new(w, 0.0) * lpos[i * rows * rows + r * rows + q];
            }
        }
        sum += fl * Complex::new(xp, 0.0);
        xp *= x;
    }
    sum
}

/// Polynomial interpolation of the `T` buffer element `(q, c)` at the
/// fractional grid point `x` (grid units), stencil over `0..=k`.
#[inline]
pub(crate) fn poly_interp_t(
    t: &[Complex],
    rows: usize,
    cols: usize,
    q: usize,
    c: usize,
    x: f64,
    integ: &Integrator,
) -> Complex {
    let k = integ.k as usize;
    let mut sum = Complex::ZERO;
    let mut xp = 1.0;
    for l in 0..=k {
        let mut fl = Complex::ZERO;
        for i in 0..=k {
            let w = integ.piw.at(l, i);
            if w != 0.0 {
                fl += Complex::new(w, 0.0) * t[i * rows * cols + q * cols + c];
            }
        }
        sum += fl * Complex::new(xp, 0.0);
        xp *= x;
    }
    sum
}

/// Scratch buffers reused across backbone evaluations of a topology.
struct Workspace {
    t: Vec<Complex>,
    buf: Vec<Complex>,
    lblocks: Vec<Complex>,
    inner: Vec<Complex>,
}

impl Workspace {
    fn new() -> Self {
        Workspace {
            t: Vec::new(),
            buf: Vec::new(),
            lblocks: Vec::new(),
            inner: Vec::new(),
        }
    }
}

/// Add the order-`m` XCA self-energy over all topologies, channel
/// assignments and pole combinations into `pp_sigma[s0].mat`
/// (`i = 1..=ntau`; `i = 0` carries no time integrals and stays zero,
/// matching the OCA convention).
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_sigma_mat_xca(
    m: usize,
    hs: &HilbertSpace,
    pp_ints: &[crate::oca::OcaChannel],
    fits: &[SoeFit],
    grids: &Grids,
    integ: &Integrator,
    pp_sigma: &mut [PseudoGreen],
) {
    assert!(m >= 1, "add_sigma_mat_xca: order must be >= 1");
    assert_eq!(
        pp_ints.len(),
        fits.len(),
        "add_sigma_mat_xca: one SOE fit per channel"
    );
    let nch = pp_ints.len();

    for topo in connected_pairings(m) {
        let b0 = topo.pairs[0].1;
        let mut ws = Workspace::new();
        let mut cs = vec![0usize; m];
        loop {
            // a line with an empty fit (zero hybridization, e.g. an
            // uncoupled channel) carries no poles: for orders >= 2
            // the channel assignments that involve it contribute
            // nothing
            if !(m >= 2 && cs.iter().any(|&c| fits[c].poles.is_empty())) {
                let mut stat = 1.0f64;
                for &c in &cs {
                    if pp_ints[c].fermion && pp_ints[c].backward {
                        stat *= -1.0;
                    }
                }
                // net prefactor (-1)^m * prod(stat) * parity: reproduces
                // the NCA (i*stat x kernel i) and OCA (i^2 x stat x
                // crossing x debug -1) totals of the calibrated kernels
                let msign = if m.is_multiple_of(2) { 1.0 } else { -1.0 };
                // The calibrated OCA pair prefactor carries the two-line
                // fermion-loop sign: -1 for a fermion-fermion pair, +1
                // for every other pair. The m = 2 backbone already
                // carries that loop sign inside `eval_into`, so the
                // prefactor must cancel it for non-fermion pairs -
                // without this the boson-boson diagrams enter with the
                // opposite sign (measured ratio -1.0000000 against the
                // OCA reference; the fermion-fermion ratio is +1).
                // Orders >= 3 with boson lines remain uncalibrated.
                let pair_loop_sign = if m == 2 && !cs.iter().all(|&c| pp_ints[c].fermion) {
                    -1.0
                } else {
                    1.0
                };
                let base_pref =
                    Complex::new(msign * stat * topo.parity * pair_loop_sign, 0.0);

                for s0 in 0..hs.ns() {
                    let Some(chain) = build_chain(m, &topo.pairs, &cs, pp_ints, hs, s0) else {
                        continue;
                    };
                    let empty_kern = || Kernels {
                        vertex: vec![None; 2 * m],
                        edge: vec![Vec::new(); 2 * m - 1],
                    };
                    if m == 1 {
                        eval_into(
                            m,
                            b0,
                            &cs,
                            &chain,
                            &empty_kern(),
                            fits,
                            pp_ints,
                            grids,
                            integ,
                            base_pref,
                            s0,
                            pp_sigma,
                            &mut ws,
                        );
                    } else {
                        let mut pole_l = vec![0usize; m - 1];
                        'poles: loop {
                            let kern = build_kernels(m, &topo.pairs, &cs, &pole_l, fits);
                            let mut pref = base_pref;
                            for li in 1..m {
                                let l = pole_l[li - 1];
                                let fit = &fits[cs[li]];
                                let w = fit.poles[l];
                                let denom = if w <= 0.0 {
                                    fit.km0(l)
                                } else {
                                    let edges = (topo.pairs[li].1 - topo.pairs[li].0 - 1) as i32;
                                    fit.kp0(l).powi(edges)
                                };
                                pref *= fit.coefs[l] / Complex::new(denom, 0.0);
                            }
                            eval_into(
                                m, b0, &cs, &chain, &kern, fits, pp_ints, grids, integ, pref, s0,
                                pp_sigma, &mut ws,
                            );
                            // pole odometer over lines 1..m-1 (bound:
                            // the pole count of the channel on that line)
                            let mut idx = 0;
                            loop {
                                if idx == m - 1 {
                                    break 'poles;
                                }
                                pole_l[idx] += 1;
                                if pole_l[idx] < fits[cs[idx + 1]].len() {
                                    break;
                                }
                                pole_l[idx] = 0;
                                idx += 1;
                            }
                        }
                    }
                }
            }
            // channel odometer over cs[0..m]
            let mut idx = 0;
            loop {
                if idx == m {
                    break;
                }
                cs[idx] += 1;
                if cs[idx] < nch {
                    break;
                }
                cs[idx] = 0;
                idx += 1;
            }
            if idx == m {
                break;
            }
        }
    }
}

/// Evaluate one fully resolved backbone and add
/// `pref * dtau^(2m-2) * T` into `pp_sigma[s0].mat`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn eval_into(
    m: usize,
    b0: usize,
    cs: &[usize],
    chain: &Chain,
    kern: &Kernels,
    fits: &[SoeFit],
    pp_ints: &[crate::oca::OcaChannel],
    grids: &Grids,
    integ: &Integrator,
    pref: Complex,
    s0: usize,
    pp_sigma: &mut [PseudoGreen],
    ws: &mut Workspace,
) {
    let ntau = grids.ntau;
    let dtau = grids.dtau;
    let d1 = grids.dim(chain.sectors[1]);
    let d0 = grids.dim(chain.sectors[0]);

    // --- init: vertex 1 (plain) or the b0 == 1 special (m == 1)
    let mut rows;
    let mut cols;
    if b0 == 1 {
        // T = op_cre (d2 x d1) * Delta(tau) * (G_{s1} (d1 x d1) op_ann
        // (d1 x d0))
        rows = grids.dim(chain.sectors[2]);
        cols = d0;
        ws.t.clear();
        ws.t.resize((ntau + 1) * rows * cols, Complex::ZERO);
        ws.inner.clear();
        ws.inner.resize(d1 * cols, Complex::ZERO);
        let op_cre = chain.ops[1];
        let op_ann = chain.ops[0];
        let delta = pp_ints[cs[0]].delta;
        for i in 0..=ntau {
            let g = grids.g_block(chain.sectors[1], i);
            let lam = delta.mat.block(i)[0];
            for r in 0..d1 {
                for c in 0..cols {
                    let mut acc = Complex::ZERO;
                    for q in 0..d1 {
                        acc += g[r * d1 + q] * op_ann.get(q, c);
                    }
                    ws.inner[r * cols + c] = acc;
                }
            }
            let off = i * rows * cols;
            for r in 0..rows {
                for c in 0..cols {
                    let mut acc = Complex::ZERO;
                    for q in 0..d1 {
                        acc += op_cre.get(r, q) * lam * ws.inner[q * cols + c];
                    }
                    ws.t[off + r * cols + c] = acc;
                }
            }
        }
    } else {
        // T = (K_1(tau) op(1)) (d2 x d1) G_{s1}(tau) (d1 x d1)
        rows = grids.dim(chain.sectors[2]);
        cols = d1;
        ws.t.clear();
        ws.t.resize((ntau + 1) * rows * cols, Complex::ZERO);
        let op1 = chain.ops[1];
        for i in 0..=ntau {
            let g = grids.g_block(chain.sectors[1], i);
            let kf = Complex::new(vertex_kernel(kern, 1, fits, i, dtau), 0.0);
            let off = i * rows * cols;
            for r in 0..rows {
                for c in 0..cols {
                    let mut acc = Complex::ZERO;
                    for q in 0..d1 {
                        acc += op1.get(r, q) * g[q * d1 + c];
                    }
                    ws.t[off + r * cols + c] = kf * acc;
                }
            }
        }
    }

    // --- segment loop: convolve by the segment propagator (with edge
    // kernels), then apply the vertex s+1 operator; the b0 vertex
    // applies the vertex-0 Delta line of its channel
    for s in 1..=2 * m - 2 {
        let ds = grids.dim(chain.sectors[s + 1]);
        debug_assert_eq!(rows, ds, "segment sector mismatch");
        ws.lblocks.clear();
        ws.lblocks.resize((ntau + 1) * ds * ds, Complex::ZERO);
        for i in 0..=ntau {
            let g = grids.g_block(chain.sectors[s + 1], i);
            let kf = Complex::new(edge_kernel(kern, s, fits, i, dtau), 0.0);
            let off = i * ds * ds;
            for q in 0..ds * ds {
                ws.lblocks[off + q] = kf * g[q];
            }
        }
        convolve(ntau, ds, cols, &ws.lblocks, &mut ws.t, &mut ws.buf, integ);

        let v = s + 1;
        if v == b0 {
            // T = op_cre(c0) (Delta(tau) (T op_ann(c0)))
            let op_cre = chain.ops[v];
            let op_ann = chain.ops[0];
            let delta = pp_ints[cs[0]].delta;
            let dnew_rows = grids.dim(chain.sectors[v + 1]);
            let dnew_cols = grids.dim(chain.sectors[0]);
            ws.buf.clear();
            ws.buf
                .resize((ntau + 1) * dnew_rows * dnew_cols, Complex::ZERO);
            for i in 0..=ntau {
                let lam = delta.mat.block(i)[0];
                let src = &ws.t[i * ds * cols..(i + 1) * ds * cols];
                let dst = i * dnew_rows * dnew_cols;
                for r in 0..dnew_rows {
                    for c in 0..dnew_cols {
                        // T op_ann then op_cre, with Delta in between
                        let mut acc = Complex::ZERO;
                        for q in 0..ds {
                            let mut tann = Complex::ZERO;
                            for c1 in 0..cols {
                                tann += src[q * cols + c1] * op_ann.get(c1, c);
                            }
                            acc += op_cre.get(r, q) * lam * tann;
                        }
                        ws.buf[dst + r * dnew_cols + c] = acc;
                    }
                }
            }
            std::mem::swap(&mut ws.t, &mut ws.buf);
            rows = dnew_rows;
            cols = dnew_cols;
        } else {
            // T = (K_v(tau) op(v)) T
            let op = chain.ops[v];
            let dnew_rows = grids.dim(chain.sectors[v + 1]);
            ws.buf.clear();
            ws.buf.resize((ntau + 1) * dnew_rows * cols, Complex::ZERO);
            for i in 0..=ntau {
                let kf = Complex::new(vertex_kernel(kern, v, fits, i, dtau), 0.0);
                let src = &ws.t[i * ds * cols..(i + 1) * ds * cols];
                let dst = i * dnew_rows * cols;
                for r in 0..dnew_rows {
                    for c in 0..cols {
                        let mut acc = Complex::ZERO;
                        for q in 0..ds {
                            acc += op.get(r, q) * src[q * cols + c];
                        }
                        ws.buf[dst + r * cols + c] = kf * acc;
                    }
                }
            }
            std::mem::swap(&mut ws.t, &mut ws.buf);
            rows = dnew_rows;
        }
    }

    // --- accumulate: Sigma.mat(i) += pref * dtau^(2m-2) * T(i);
    // i = 0 carries the contact term (nonzero only for m = 1)
    let scale = Complex::new(dtau.powi((2 * m - 2) as i32), 0.0) * pref;
    debug_assert_eq!((rows, cols), (d0, d0), "backbone must close on s0");
    for i in 0..=ntau {
        let blk = pp_sigma[s0].mat.block_mut(i);
        let src = &ws.t[i * rows * cols..(i + 1) * rows * cols];
        for q in 0..rows * cols {
            blk[q] += scale * src[q];
        }
    }
}

// ------------------------------------------------------------------
// TEMPORARY PROBE: single-pair XCA(2) vs OCA comparison
// ------------------------------------------------------------------
#[cfg(test)]
pub(crate) mod probe {
    use super::*;
    use crate::hilbert::{SectorClass, build};
    use crate::nca::Hybridization;
    use crate::oca::{self, OcaChannel};
    use kadanoffbaym::complex::Complex;
    use kadanoffbaym::types::{Contour, ContourGreen, PseudoGreen, Sign};
    use kadanoffbaym::weights::Integrator;

    pub(crate) struct ProbeFx {
        pub hs: crate::hilbert::HilbertSpace,
        pub pp_ints: Vec<Hybridization<'static>>,
        pub pp_g: Vec<PseudoGreen>,
        pub contour: Contour,
    }

    // SAFETY-free construction: channels borrow leaked ContourGreens
    // (test-only, process-lifetime)
    pub(crate) fn fixture() -> ProbeFx {
        let beta = 5.0f64;
        let ntau = 96usize;
        let hs = build(1, SectorClass::SpinResolved);
        let contour = Contour::new(4, ntau as i64, 1, 1, 0.08, beta);
        let dtau = beta / ntau as f64;
        let kf = |t: f64, w: f64| -(-w * t).exp() / (1.0 + (-beta * w).exp());
        let mk = |src: f64| -> &'static ContourGreen {
            let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
            for i in 0..=ntau {
                let v = src * kf(dtau * i as f64, -1.2) + 0.5 * src * kf(dtau * i as f64, 0.8);
                g.mat.block_mut(i)[0] = Complex::new(v, 0.0);
            }
            Box::leak(Box::new(g))
        };
        let mk_cc = |src: &ContourGreen| -> &'static ContourGreen {
            let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
            for i in 0..=ntau {
                let v = -src.mat.block(ntau - i)[0];
                g.mat.block_mut(i)[0] = v;
            }
            Box::leak(Box::new(g))
        };
        let d_up = mk(0.09);
        let d_dn = mk(0.12);
        let d_up_cc = mk_cc(d_up);
        let d_dn_cc = mk_cc(d_dn);
        let cua = hs.flavor(0, 0, true);
        let cda = hs.flavor(0, 1, true);
        let cud = hs.flavor(0, 0, false);
        let cdd = hs.flavor(0, 1, false);
        let pp_ints = vec![
            Hybridization {
                delta: d_up,
                op_cre_fl: cua,
                op_ann_fl: cud,
                fermion: true,
                backward: false,
            },
            Hybridization {
                delta: d_up_cc,
                op_cre_fl: cud,
                op_ann_fl: cua,
                fermion: true,
                backward: true,
            },
            Hybridization {
                delta: d_dn,
                op_cre_fl: cda,
                op_ann_fl: cdd,
                fermion: true,
                backward: false,
            },
            Hybridization {
                delta: d_dn_cc,
                op_cre_fl: cdd,
                op_ann_fl: cda,
                fermion: true,
                backward: true,
            },
        ];
        let energies = [0.0, -1.5, -1.5, 0.0];
        let mu = -1.6387400224070612;
        let mut pp_g = Vec::new();
        for s in 0..4 {
            let mut g = PseudoGreen::new(&contour, hs.sectors[s].sig, Complex::ZERO);
            for i in 0..=ntau {
                let v = -(dtau * i as f64 * (energies[s] - mu)).exp();
                g.mat.block_mut(i)[0] = Complex::new(v, 0.0);
            }
            pp_g.push(g);
        }
        ProbeFx {
            hs,
            pp_ints,
            pp_g,
            contour,
        }
    }

        /// XCA(2) restricted to the channel assignment `cs` (single
    /// topology), evaluated into a fresh sigma buffer.
    pub(crate) fn xca2_single(fx: &ProbeFx, cs: [usize; 2]) -> Vec<PseudoGreen> {
        let channels = fermion_channels(fx);
        xca2_single_ch(fx, &channels, cs)
    }

    /// Fermion-only channel view of the fixture.
    pub(crate) fn fermion_channels(fx: &ProbeFx) -> Vec<OcaChannel<'static>> {
        fx.pp_ints
            .iter()
            .map(|h| oca::OcaChannel {
                delta: h.delta,
                fermion: h.fermion,
                backward: h.backward,
                ann: oca::VertSrc::Flavor(h.op_ann_fl),
                cre: oca::VertSrc::Flavor(h.op_cre_fl),
            })
            .collect()
    }

    /// XCA(2) for one channel assignment over an arbitrary channel
    /// set (the boson-extended diagnosis driver).
    pub(crate) fn xca2_single_ch(
        fx: &ProbeFx,
        channels: &[OcaChannel],
        cs: [usize; 2],
    ) -> Vec<PseudoGreen> {
        let ntau = 96usize;
        let beta = 5.0f64;
        let fits = crate::xca::fit_channels(channels, beta, ntau, 12, 6.0);
        let grids = Grids {
            ntau,
            dtau: beta / ntau as f64,
            pp_g: &fx.pp_g,
        };
        let integ = Integrator::new(5);
        let mut out: Vec<PseudoGreen> = (0..4)
            .map(|s| PseudoGreen::new(&fx.contour, fx.hs.sectors[s].sig, Complex::ZERO))
            .collect();
        let topos = crate::xca::topology::connected_pairings(2);
        for topo in &topos {
            let b0 = topo.pairs[0].1;
            let mut ws = Workspace::new();
            let mut stat = 1.0f64;
            for &c in &cs {
                if channels[c].fermion && channels[c].backward {
                    stat *= -1.0;
                }
            }
            let msign = 1.0f64; // m even
            let base_pref = Complex::new(msign * stat * topo.parity, 0.0);
            for s0 in 0..fx.hs.ns() {
                let Some(chain) = build_chain(2, &topo.pairs, &cs, channels, &fx.hs, s0) else {
                    continue;
                };
                // pole loop over line 1
                let mut pole_l = vec![0usize; 1];
                'poles: loop {
                    let kern = build_kernels(2, &topo.pairs, &cs, &pole_l, &fits);
                    let mut pref = base_pref;
                    let l = pole_l[0];
                    let fit = &fits[cs[1]];
                    let w = fit.poles[l];
                    let denom = if w <= 0.0 {
                        fit.km0(l)
                    } else {
                        let edges = (topo.pairs[1].1 - topo.pairs[1].0 - 1) as i32;
                        fit.kp0(l).powi(edges)
                    };
                    pref *= fit.coefs[l] / Complex::new(denom, 0.0);
                    eval_into(
                        2, b0, &cs, &chain, &kern, &fits, channels, &grids, &integ, pref, s0,
                        &mut out, &mut ws,
                    );
                    pole_l[0] += 1;
                    if pole_l[0] >= fits[cs[1]].len() {
                        break 'poles;
                    }
                }
            }
        }
        out
    }

        /// Native OCA diagrams filtered to one ordered channel pair.
    pub(crate) fn oca_single(fx: &ProbeFx, c1: usize, c2: usize) -> (Vec<PseudoGreen>, Vec<usize>) {
        let channels = fermion_channels(fx);
        oca_single_ch(fx, &channels, c1, c2)
    }

    /// Native OCA diagrams filtered to one ordered channel pair over
    /// an arbitrary channel set (the boson-extended diagnosis driver).
    pub(crate) fn oca_single_ch(
        fx: &ProbeFx,
        channels: &[OcaChannel],
        c1: usize,
        c2: usize,
    ) -> (Vec<PseudoGreen>, Vec<usize>) {
        let all = oca::build_sigma_diagrams_oca_channels(&fx.hs, channels);
        let mut picked = Vec::new();
        for (idx, d) in all.iter().enumerate() {
            if std::ptr::eq(d.lam1, channels[c1].delta)
                && std::ptr::eq(d.lam2, channels[c2].delta)
            {
                picked.push(idx);
            }
        }
        let integ = Integrator::new(5);
        let mut out: Vec<PseudoGreen> = (0..4)
            .map(|s| PseudoGreen::new(&fx.contour, fx.hs.sectors[s].sig, Complex::ZERO))
            .collect();
        for &idx in &picked {
            oca::add_sigma_mat_oca(&all[idx], &fx.pp_g, 5.0, 5, &integ, &mut out);
        }
        (out, picked)
    }
}
