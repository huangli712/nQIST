// Project : nQIST
// File    : src/xca/gf.rs
// Purpose : arbitrary-order (XCA) single-particle Green's function
//           diagrams on the Matsubara axis (circular backbones)
//
// Green's function diagrams of order `m` use the same connected
// chord topologies as the self-energy, but the pair containing
// vertex 0 hosts the two external operators (creation at time 0,
// annihilation at the partner vertex, which is the external time
// `tau`), and the remaining `m - 1` hybridization lines are interior
// and all SOE-separated. The circle is evaluated as two backbones:
// the 0 -> tau portion (plain Volterra chain) and the tau -> beta
// portion (chain built in the reflected time `u = beta - t`, with
// the kernel signs swapped through `K(beta - u, w) = K(u, -w)`),
// closed by the segment propagator `G_{s0}(beta - t)` and contracted
// as `tr(T_l op_ann T_r op_cre)` (SciPost Phys. 19, 121 (2025),
// Appendix B; TRIQS/xca `eval_one_diagram_G`).
//
// Prefactor: `-prod(stat_c) * parity(topology) * prod_l c_l /
// denom_l` times `dtau^(2m-2)`; at order 1 this reproduces
// `nca::gf_mat` exactly and at order 2 `oca::add_gf_mat_oca` up to
// quadrature error (the calibration tests pin these conventions).

use crate::hilbert::HilbertSpace;
use crate::nca::{GfVertex, Hybridization};
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
    #[inline]
    fn dim(&self, s: usize) -> usize {
        self.pp_g[s].size1
    }

    #[inline]
    fn g_block(&self, s: usize, i: usize) -> &[Complex] {
        self.pp_g[s].mat.block(i)
    }
}

/// One resolved circular sector chain with the operator blocks.
struct Chain<'a> {
    sectors: Vec<usize>,
    /// Interior vertex operators (per backbone vertex).
    ops: Vec<Option<&'a DenseMat>>,
    /// External creation block at vertex 0 (`sigma[0] -> sigma[1]`).
    op_cre: &'a DenseMat,
    /// External annihilation block at vertex `btau`.
    op_ann: &'a DenseMat,
}

/// Resolve the circular chain of a GF diagram: vertex 0 carries the
/// external creation operator of the measurement vertex, the `tau`
/// vertex (partner of vertex 0) the external annihilation operator,
/// and every interior vertex the operator of its line (earlier
/// endpoint `op_ann_fl`, later endpoint `op_cre_fl`).
#[allow(clippy::too_many_arguments)]
fn build_chain<'a>(
    m: usize,
    pairs: &[(usize, usize)],
    channels: &[usize],
    pp_ints: &[Hybridization],
    hs: &'a HilbertSpace,
    verts: &[GfVertex],
    vx: usize,
    s0: usize,
) -> Option<Chain<'a>> {
    let btau = pairs[0].1;
    let mut owner = vec![(0usize, false); 2 * m];
    for (li, &(a, b)) in pairs.iter().enumerate() {
        owner[a] = (li, false);
        owner[b] = (li, true);
    }
    let mut sectors = vec![0usize; 2 * m + 1];
    sectors[0] = s0;
    let mut ops: Vec<Option<&DenseMat>> = vec![None; 2 * m];
    // vertex 0: external creation
    let (t1, b1) = hs.ops[verts[vx].op_cre_fl].block(s0)?;
    sectors[1] = *t1;
    let op_cre = &b1;
    // interior vertices 1..2m-1
    for v in 1..2 * m {
        let flavor = if v == btau {
            verts[vx].op_ann_fl
        } else {
            let (li, later) = owner[v];
            if later {
                pp_ints[channels[li]].op_cre_fl
            } else {
                pp_ints[channels[li]].op_ann_fl
            }
        };
        let (t, b) = hs.ops[flavor].block(sectors[v])?;
        sectors[v + 1] = *t;
        ops[v] = if v == btau { None } else { Some(b) };
    }
    if sectors[2 * m] != s0 {
        return None;
    }
    // external annihilation block at btau: re-derive from the chain
    let (_, bann) = hs.ops[verts[vx].op_ann_fl].block(sectors[btau])?;
    Some(Chain {
        sectors,
        ops,
        op_cre,
        op_ann: bann,
    })
}

/// SOE kernel placement (same rules as the self-energy backbones).
#[derive(Clone)]
struct Kernels {
    /// `vertex[v] = Some((line, pole, sign))` with `K(tau, sign*w)`.
    vertex: Vec<Option<(usize, usize, f64)>>,
    /// `edge[s]`: `(line, pole)` pairs multiplying segment `s`.
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
    let mut edge = vec![Vec::new(); 2 * m];
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

/// Vertex kernel factor; `flip` swaps `K^+ <-> K^-` (reflected time).
#[inline]
fn vertex_kernel(
    kern: &Kernels,
    v: usize,
    fits: &[SoeFit],
    i: usize,
    dtau: f64,
    flip: bool,
) -> f64 {
    match kern.vertex[v] {
        Some((li, l, sgn)) => {
            let sgn = if flip { -sgn } else { sgn };
            if sgn > 0.0 {
                fits[li].kp(i, l, dtau)
            } else {
                fits[li].km(i, l, dtau)
            }
        }
        None => 1.0,
    }
}

/// Edge kernel product of segment `s`; `flip` swaps `K^+ <-> K^-`.
#[inline]
fn edge_kernel(kern: &Kernels, s: usize, fits: &[SoeFit], i: usize, dtau: f64, flip: bool) -> f64 {
    let mut acc = 1.0;
    for &(li, l) in &kern.edge[s] {
        acc *= if flip {
            fits[li].km(i, l, dtau)
        } else {
            fits[li].kp(i, l, dtau)
        };
    }
    acc
}

/// Volterra convolution `(L * T)` with `L` on the left (matrix).
fn convolve_left(
    ntau: usize,
    rows: usize,
    cols: usize,
    lblocks: &[Complex],
    t: &mut [Complex],
    buf: &mut Vec<Complex>,
    integ: &Integrator,
) {
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
                            acc += lblocks[li + r * rows + q] * t[ti + q * cols + c];
                        }
                        buf[off + r * cols + c] += wc * acc;
                    }
                }
            }
        } else {
            // short-interval startup (mirror of the sigma-side kernel:
            // k-point sub-grid, polynomial interpolation of both
            // factors, single gregory row)
            let dsub = n as f64 / k as f64;
            for l in 0..=k {
                let x = dsub * l as f64;
                let dl = n as f64 - x;
                for r in 0..rows {
                    for c in 0..cols {
                        let mut acc = Complex::ZERO;
                        for q in 0..rows {
                            let lv = super::sigma::poly_interp_elem(lblocks, rows, r, q, dl, integ);
                            let tv = super::sigma::poly_interp_t(t, rows, cols, q, c, x, integ);
                            acc += lv * tv;
                        }
                        buf[off + r * cols + c] +=
                            Complex::new(integ.giw.at(k, l) * dsub, 0.0) * acc;
                    }
                }
            }
        }
    }
    t.copy_from_slice(buf);
}

/// Volterra convolution accumulating OUTWARD in reflected time:
/// `(T circ R)(t_n) = int_0^{t_n} T(t_j) R(t_n - t_j) dt_j` with the
/// accumulated `T` (rows x cols) at the SMALL argument and the new
/// square segment `R` (cols x cols) at the LARGE argument — the
/// correct pairing for the tau -> beta side of a circular backbone.
fn convolve_right(
    ntau: usize,
    rows: usize,
    cols: usize,
    rblocks: &[Complex],
    t: &mut [Complex],
    buf: &mut Vec<Complex>,
    integ: &Integrator,
) {
    buf.clear();
    buf.resize((ntau + 1) * rows * cols, Complex::ZERO);
    for n in 1..=ntau {
        let off = n * rows * cols;
        let k = integ.k as usize;
        if n >= k {
            for j in 0..=n {
                let w = integ.giw.at(n, j);
                if w == 0.0 {
                    continue;
                }
                let wc = Complex::new(w, 0.0);
                let ri = (n - j) * cols * cols;
                let ti = j * rows * cols;
                for r in 0..rows {
                    for c in 0..cols {
                        let mut acc = Complex::ZERO;
                        for q in 0..cols {
                            acc += t[ti + r * cols + q] * rblocks[ri + q * cols + c];
                        }
                        buf[off + r * cols + c] += wc * acc;
                    }
                }
            }
        } else {
            // short-interval startup (right pairing: T at the small
            // argument, R at the large one)
            let dsub = n as f64 / k as f64;
            for l in 0..=k {
                let x = dsub * l as f64;
                let dl = n as f64 - x;
                for r in 0..rows {
                    for c in 0..cols {
                        let mut acc = Complex::ZERO;
                        for q in 0..cols {
                            let tv = super::sigma::poly_interp_t(t, rows, cols, r, q, x, integ);
                            let rv = super::sigma::poly_interp_elem(rblocks, cols, q, c, dl, integ);
                            acc += tv * rv;
                        }
                        buf[off + r * cols + c] +=
                            Complex::new(integ.giw.at(k, l) * dsub, 0.0) * acc;
                    }
                }
            }
        }
    }
    t.copy_from_slice(buf);
}

struct Workspace {
    t: Vec<Complex>,
    tl: Vec<Complex>,
    buf: Vec<Complex>,
    blocks: Vec<Complex>,
}

/// Add the order-`m` XCA single-particle Green's function over all
/// topologies, channel assignments and pole combinations into
/// `out[vertex][i]`, `i = 0..=ntau`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_gf_mat_xca(
    m: usize,
    hs: &HilbertSpace,
    pp_ints: &[Hybridization],
    fits: &[SoeFit],
    verts: &[GfVertex],
    grids: &Grids,
    integ: &Integrator,
    out: &mut [Vec<Complex>],
) {
    assert!(m >= 1, "add_gf_mat_xca: order must be >= 1");
    assert_eq!(
        pp_ints.len(),
        fits.len(),
        "add_gf_mat_xca: one SOE fit per channel"
    );
    if m >= 2 && fits.iter().any(|f| f.poles.is_empty()) {
        return;
    }
    let nch = pp_ints.len();
    let ntau = grids.ntau;
    let dtau = grids.dtau;

    for topo in connected_pairings(m) {
        let btau = topo.pairs[0].1;
        let mut ws = Workspace {
            t: Vec::new(),
            tl: Vec::new(),
            buf: Vec::new(),
            blocks: Vec::new(),
        };
        let mut cs = vec![0usize; m];
        loop {
            // the external pair carries no channel: only cs[1..m]
            // are hybridization lines (plain statistics factors; the
            // circular wrap signs are folded into the reflected-time
            // evaluation)
            let mut stat = 1.0f64;
            for &c in &cs[1..] {
                if pp_ints[c].fermion && pp_ints[c].backward {
                    stat *= -1.0;
                }
            }
            // net prefactor (-1)^m * prod(stat) * parity, matching
            // the self-energy convention (order 1 reproduces the NCA
            // bubble prefactor -1)
            let msign = if m.is_multiple_of(2) { 1.0 } else { -1.0 };
            let base_pref = Complex::new(msign * stat * topo.parity, 0.0);

            for vx in 0..verts.len() {
                for s0 in 0..hs.ns() {
                    let Some(chain) = build_chain(m, &topo.pairs, &cs, pp_ints, hs, verts, vx, s0)
                    else {
                        continue;
                    };
                    let run = |kern: &Kernels,
                               pref: Complex,
                               ws: &mut Workspace,
                               out: &mut [Vec<Complex>]| {
                        eval_gf_backbone(
                            m, btau, &chain, kern, fits, grids, integ, pref, vx, out, ws,
                        );
                    };
                    if m == 1 {
                        run(
                            &Kernels {
                                vertex: vec![None; 2],
                                edge: vec![Vec::new(); 2],
                            },
                            base_pref,
                            &mut ws,
                            out,
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
                            run(&kern, pref, &mut ws, out);
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
            // channel odometer over the m - 1 interior lines only
            // (cs[0] belongs to the external pair and carries no
            // channel)
            let mut idx = 1;
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
    let _ = (ntau, dtau);
}

/// Evaluate one resolved circular backbone and add
/// `pref * dtau^(2m-2) * tr(T_l op_ann T_r op_cre)` to
/// `out[vx][i]`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn eval_gf_backbone(
    m: usize,
    btau: usize,
    chain: &Chain,
    kern: &Kernels,
    fits: &[SoeFit],
    grids: &Grids,
    integ: &Integrator,
    pref: Complex,
    vx: usize,
    out: &mut [Vec<Complex>],
    ws: &mut Workspace,
) {
    let ntau = grids.ntau;
    let dtau = grids.dtau;
    let d0 = grids.dim(chain.sectors[0]);
    let d1 = grids.dim(chain.sectors[1]);
    let dbt = grids.dim(chain.sectors[btau]);

    // ---- right part: T_r = op(btau-1) ... op(1) G_{s1}, then
    // convolved by the segments up to (and including) s = btau-1
    let mut rows;
    let cols;
    if btau == 1 {
        rows = d1;
        cols = d1;
        ws.t.clear();
        ws.t.resize((ntau + 1) * rows * cols, Complex::ZERO);
        for i in 0..=ntau {
            let g = grids.g_block(chain.sectors[1], i);
            ws.t[i * rows * cols..(i + 1) * rows * cols].copy_from_slice(g);
        }
    } else {
        rows = grids.dim(chain.sectors[2]);
        cols = d1;
        ws.t.clear();
        ws.t.resize((ntau + 1) * rows * cols, Complex::ZERO);
        let op1 = chain.ops[1].expect("interior vertex 1 op");
        for i in 0..=ntau {
            let g = grids.g_block(chain.sectors[1], i);
            let kf = Complex::new(vertex_kernel(kern, 1, fits, i, dtau, false), 0.0);
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
        for s in 1..btau {
            let ds = grids.dim(chain.sectors[s + 1]);
            debug_assert_eq!(rows, ds);
            ws.blocks.clear();
            ws.blocks.resize((ntau + 1) * ds * ds, Complex::ZERO);
            for i in 0..=ntau {
                let g = grids.g_block(chain.sectors[s + 1], i);
                let kf = Complex::new(edge_kernel(kern, s, fits, i, dtau, false), 0.0);
                let off = i * ds * ds;
                for q in 0..ds * ds {
                    ws.blocks[off + q] = kf * g[q];
                }
            }
            convolve_left(ntau, ds, cols, &ws.blocks, &mut ws.t, &mut ws.buf, integ);
            if s + 1 < btau {
                // apply the interior vertex op at s+1
                let op = chain.ops[s + 1].expect("interior vertex op");
                let dnew = grids.dim(chain.sectors[s + 2]);
                ws.buf.clear();
                ws.buf.resize((ntau + 1) * dnew * cols, Complex::ZERO);
                for i in 0..=ntau {
                    let kf = Complex::new(vertex_kernel(kern, s + 1, fits, i, dtau, false), 0.0);
                    let src = &ws.t[i * ds * cols..(i + 1) * ds * cols];
                    let dst = i * dnew * cols;
                    for r in 0..dnew {
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
                rows = dnew;
            }
        }
    }
    // T_r is now (d_{btau} x d_1)
    debug_assert_eq!(rows, dbt);

    // ---- left part in reflected time u = beta - t:
    // T_l = G_{s0}(u) (K^r op(2m-1)), then convolutions
    // right-multiplied down to vertex btau + 1
    {
        let dlast = if m == 1 {
            d0
        } else {
            grids.dim(chain.sectors[2 * m - 1])
        };
        ws.tl.clear();
        ws.tl.resize((ntau + 1) * d0 * dlast, Complex::ZERO);
        // closing propagator in reflected time (flip calibrated:
        // unity; the statistics structure of the circular trace is
        // fully carried by the channel statistics factors)
        let flip = Complex::ONE;
        for i in 0..=ntau {
            let g = grids.g_block(chain.sectors[0], i);
            let kf = if m == 1 {
                flip
            } else {
                let op = chain.ops[2 * m - 1].expect("interior last vertex op");
                let kf =
                    Complex::new(vertex_kernel(kern, 2 * m - 1, fits, i, dtau, true), 0.0) * flip;
                let off = i * d0 * dlast;
                for r in 0..d0 {
                    for c in 0..dlast {
                        let mut acc = Complex::ZERO;
                        for q in 0..d0 {
                            acc += g[r * d0 + q] * op.get(q, c);
                        }
                        ws.tl[off + r * dlast + c] = kf * acc;
                    }
                }
                continue;
            };
            let _ = kf;
            let off = i * d0 * dlast;
            for r in 0..d0 {
                for c in 0..dlast {
                    ws.tl[off + r * dlast + c] = flip * g[r * d0 + c];
                }
            }
        }
        let lrows = d0;
        let mut lcols = dlast;
        // convolutions and interior vertex ops down to btau + 1.
        // Edge kernels ride the segment elapsed time (the conv's
        // LARGE argument) -> plain kp, no reflection swap; vertex
        // kernels sit at the vertex real time beta - u -> swapped.
        for s in (btau + 1..2 * m).rev() {
            // segment (s-1, s) carries G_{sigma_s}, reflected
            let ds = grids.dim(chain.sectors[s]);
            debug_assert_eq!(lcols, ds);
            ws.blocks.clear();
            ws.blocks.resize((ntau + 1) * ds * ds, Complex::ZERO);
            for i in 0..=ntau {
                let g = grids.g_block(chain.sectors[s], i);
                let kf = Complex::new(edge_kernel(kern, s - 1, fits, i, dtau, false), 0.0);
                let off = i * ds * ds;
                for q in 0..ds * ds {
                    ws.blocks[off + q] = kf * g[q];
                }
            }
            convolve_right(
                ntau,
                lrows,
                lcols,
                &ws.blocks,
                &mut ws.tl,
                &mut ws.buf,
                integ,
            );
            if s - 1 == btau {
                // vertex btau hosts the external annihilation operator
                // and is applied in the final contraction
                continue;
            }
            // vertex op at s-1 (right multiplication)
            let op = chain.ops[s - 1].expect("interior vertex op");
            let dnew = grids.dim(chain.sectors[s - 1]);
            ws.buf.clear();
            ws.buf.resize((ntau + 1) * lrows * dnew, Complex::ZERO);
            for i in 0..=ntau {
                let kf = Complex::new(vertex_kernel(kern, s - 1, fits, i, dtau, true), 0.0);
                let src = &ws.tl[i * lrows * lcols..(i + 1) * lrows * lcols];
                let dst = i * lrows * dnew;
                for r in 0..lrows {
                    for c in 0..dnew {
                        let mut acc = Complex::ZERO;
                        for q in 0..lcols {
                            acc += src[r * lcols + q] * op.get(q, c);
                        }
                        ws.buf[dst + r * dnew + c] = kf * acc;
                    }
                }
            }
            std::mem::swap(&mut ws.tl, &mut ws.buf);
            lcols = dnew;
        }
        // fold the reflection back: T_l_real[i] = T_l[ntau - i],
        // contract with the external operators and the trace
        let scale = Complex::new(dtau.powi((2 * m - 2) as i32), 0.0) * pref;
        let op_cre = chain.op_cre;
        let op_ann = chain.op_ann;
        let dnext = grids.dim(chain.sectors[btau + 1]);
        let mut acc_mat = vec![Complex::ZERO; d0 * d0];
        for i in 0..=ntau {
            let j = ntau - i;
            let tl = &ws.tl[j * d0 * lcols..(j + 1) * d0 * lcols];
            let tr = &ws.t[i * dbt * d1..(i + 1) * dbt * d1];
            // A = op_ann (dnext x dbt) applied between T_l and T_r:
            // C = T_l (d0 x lcols) ... lcols must equal dnext here
            debug_assert_eq!(lcols, dnext);
            for r in 0..d0 {
                for c in 0..d0 {
                    // C[r][c] = sum over the chain: T_l . op_ann . T_r
                    // . op_cre
                    let mut acc = Complex::ZERO;
                    for q1 in 0..lcols {
                        // tl[r][q1] * (op_ann . T_r . op_cre)[q1][c]
                        let mut mid = Complex::ZERO;
                        for q2 in 0..dbt {
                            let ann_op = op_ann.get(q1, q2);
                            if ann_op == Complex::ZERO {
                                continue;
                            }
                            // (T_r . op_cre)[q2][c]
                            let mut rc = Complex::ZERO;
                            for q3 in 0..d1 {
                                rc += tr[q2 * d1 + q3] * op_cre.get(q3, c);
                            }
                            mid += ann_op * rc;
                        }
                        acc += tl[r * lcols + q1] * mid;
                    }
                    acc_mat[r * d0 + c] = acc;
                }
            }
            let mut tr_val = Complex::ZERO;
            for q in 0..d0 {
                tr_val += acc_mat[q * d0 + q];
            }
            out[vx][i] += scale * tr_val;
        }
    }
}
