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
use crate::nca::Hybridization;
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
    pp_ints: &[Hybridization],
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
        let flavor = if later {
            pp_ints[channels[li]].op_cre_fl
        } else {
            pp_ints[channels[li]].op_ann_fl
        };
        let (target, block) = hs.ops[flavor].block(sectors[v])?;
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
    lblocks: &[Complex],
    t: &mut [Complex],
    buf: &mut Vec<Complex>,
    integ: &Integrator,
) {
    debug_assert_eq!(lblocks.len(), (ntau + 1) * rows * rows);
    debug_assert_eq!(t.len(), (ntau + 1) * rows * cols);
    buf.clear();
    buf.resize((ntau + 1) * rows * cols, Complex::ZERO);
    for n in 1..=ntau {
        let off = n * rows * cols;
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
                        acc += lblocks[li + r * rows + q] * t[ti + q * cols + c];
                    }
                    buf[off + r * cols + c] += wc * acc;
                }
            }
        }
    }
    t.copy_from_slice(buf);
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
    pp_ints: &[Hybridization],
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
    // an empty fit (e.g. the all-zero hybridization of the first DMFT
    // iteration) carries no poles: orders >= 2 contribute nothing
    if m >= 2 && fits.iter().any(|f| f.poles.is_empty()) {
        return;
    }
    let nch = pp_ints.len();

    for topo in connected_pairings(m) {
        let b0 = topo.pairs[0].1;
        let mut ws = Workspace::new();
        let mut cs = vec![0usize; m];
        loop {
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
            let base_pref = Complex::new(msign * stat * topo.parity, 0.0);

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
    pp_ints: &[Hybridization],
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
