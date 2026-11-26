// Project : nQIST
// File    : src/oca.rs
// Purpose : OCA (one-crossing) pseudo-particle diagrams, Matsubara
//           path (blueprints ppsc/oca/sigma.cpp + sigma_mat.hpp +
//           gf.cpp + gf_mat.hpp + oca/integration.hpp)

use crate::hilbert::HilbertSpace;
use crate::nca::{BosonBath, Hybridization};
use kadanoffbaym::complex::Complex;
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::types::{ContourGreen, PseudoGreen};
use kadanoffbaym::weights::Integrator;

// ------------------------------------------------------------------
// raw-buffer linear algebra (no allocations in the hot loops)
// ------------------------------------------------------------------

/// `out = a * b` for row-major `a (m x k)` and `b (k x n)`.
#[allow(dead_code)]
fn mul_into(a: &[Complex], b: &[Complex], m: usize, k: usize, n: usize, out: &mut [Complex]) {
    debug_assert!(out.len() >= m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = Complex::ZERO;
            for q in 0..k {
                acc += a[i * k + q] * b[q * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

/// `out += a * b` for row-major `a (m x k)` and `b (k x n)`.
#[allow(dead_code)]
fn mul_acc(a: &[Complex], b: &[Complex], m: usize, k: usize, n: usize, out: &mut [Complex]) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = Complex::ZERO;
            for q in 0..k {
                acc += a[i * k + q] * b[q * n + j];
            }
            out[i * n + j] += acc;
        }
    }
}

/// `out = a * m * b` for row-major `a (d1 x dk)`, `m (dk x dl)`,
/// `b (dl x d2)` (nothing assumed square).
pub(crate) fn sandwich(
    a: &[Complex],
    m: &[Complex],
    b: &[Complex],
    d1: usize,
    dk: usize,
    dl: usize,
    d2: usize,
    out: &mut [Complex],
    scratch: &mut [Complex],
) {
    mul_into(a, m, d1, dk, dl, scratch);
    mul_into(scratch, b, d1, dl, d2, out);
}

/// A Matsubara axis of `d1 x d2` blocks (raw view of a mat component).
#[derive(Clone, Copy)]
struct MatAxis<'a> {
    data: &'a [Complex],
    d1: usize,
    d2: usize,
}

impl<'a> MatAxis<'a> {
    fn block(&self, i: usize) -> &'a [Complex] {
        &self.data[i * self.d1 * self.d2..(i + 1) * self.d1 * self.d2]
    }
}

fn mat_axis_pp(pp: &PseudoGreen) -> MatAxis<'_> {
    MatAxis {
        data: pp.mat.raw(),
        d1: pp.size1,
        d2: pp.size2,
    }
}

fn mat_axis_delta(delta: &ContourGreen) -> MatAxis<'_> {
    MatAxis {
        data: delta.mat.raw(),
        d1: delta.size1,
        d2: delta.size2,
    }
}

// ------------------------------------------------------------------
// quadrature helpers (blueprint oca/integration.hpp on KB weights)
// ------------------------------------------------------------------

/// Gregory quadrature `sum_{j=0}^{n} Omega[n][j] v(j)` (blueprint
/// `gregory_sum`; KB's `giw.at` covers the warmup, boundary, and
/// interior weights of the C++ three-loop decomposition).
fn gregory_sum(n: usize, v: &[Vec<Complex>], integ: &Integrator) -> Vec<Complex> {
    // same warmup-row semantics as gregory_scalar (l runs to max(n, k));
    // equilibrium callers always pass n >= k, so this extends, never
    // changes, the gated behavior
    let k = integ.k as usize;
    let nel = v[0].len();
    let mut out = vec![Complex::ZERO; nel];
    for j in 0..=n.max(k) {
        let w = integ.giw.at(n, j);
        if w != 0.0 {
            for q in 0..nel {
                out[q] += Complex::new(w, 0.0) * v[j][q];
            }
        }
    }
    out
}

/// Polynomial interpolation of the axis at the fractional grid point
/// `x` (blueprint `poly_interpol`):
/// `sum_l x^l sum_i PIW[l,i] f(i)` with the nodes `0..=kt`.
fn poly_interpol(
    f: &MatAxis,
    x: f64,
    integ: &Integrator,
    out: &mut [Complex],
    scratch: &mut [Complex],
) {
    let k = integ.k as usize;
    let nel = f.d1 * f.d2;
    for q in 0..nel {
        out[q] = Complex::ZERO;
    }
    let mut xp = 1.0;
    for l in 0..=k {
        for q in 0..nel {
            scratch[q] = Complex::ZERO;
        }
        for i in 0..=k {
            let w = integ.piw.at(l, i);
            if w != 0.0 {
                let b = f.block(i);
                for q in 0..nel {
                    scratch[q] += Complex::new(w, 0.0) * b[q];
                }
            }
        }
        for q in 0..nel {
            out[q] += Complex::new(xp, 0.0) * scratch[q];
        }
        xp *= x;
    }
}

// ------------------------------------------------------------------
// diagram descriptors
// ------------------------------------------------------------------

/// One OCA self-energy diagram: the closed four-sector chain
/// `ss --int2.op2-> sc --int1.op2-> sb --int2.op1-> sa --int1.op1-> ss`
/// (blueprint `oca::construct_sigma_diagram`).
pub(crate) struct OcaSigmaDiagram<'a> {
    pub out: usize,
    pub sa: usize,
    pub sb: usize,
    pub sc: usize,
    /// statistics sign of the sigma sector (d.sig of the blueprint)
    pub sig: f64,
    pub op1: DenseMat,
    pub op2: DenseMat,
    pub op3: DenseMat,
    pub op4: DenseMat,
    pub lam1: &'a ContourGreen,
    pub lam2: &'a ContourGreen,
    /// `i^2 * stat_sign * (-1)` with the C++ `debug_prefactor = -1`
    /// kept verbatim.
    pub prefactor: Complex,
}

/// Vertex source of an OCA channel: a fermion c-operator flavor or a
/// sector-diagonal boson vertex (the Holstein bath, blueprint: the
/// operator-valued `pp_int_type` vertices).
pub(crate) enum VertSrc<'a> {
    Flavor(usize),
    Diag(&'a [(usize, DenseMat)]),
}

impl VertSrc<'_> {
    // pub(crate) for the XCA_INTDUMP channel-enumeration audit driver
    pub(crate) fn block<'x>(
        &'x self,
        hs: &'x HilbertSpace,
        sector: usize,
    ) -> Option<(usize, &'x DenseMat)> {
        match self {
            VertSrc::Flavor(f) => hs.ops[*f].block(sector).map(|(s, m)| (*s, m)),
            VertSrc::Diag(list) => list
                .iter()
                .find(|(s, _)| *s == sector)
                .map(|(s, m)| (*s, m)),
        }
    }
}

/// A hybridization channel view for the OCA loops: statistics,
/// direction, propagator, and the two vertex sources.
pub(crate) struct OcaChannel<'a> {
    pub delta: &'a ContourGreen,
    pub fermion: bool,
    pub backward: bool,
    pub ann: VertSrc<'a>,
    pub cre: VertSrc<'a>,
}

/// The boson channels of a Holstein [`BosonBath`] (forward/backward;
/// the diagonal vertex acts as both the annihilation and the creation
/// insertion).
/// Fermion-only convenience wrapper (test callers, xca_rt).
pub(crate) fn build_sigma_diagrams_oca<'a>(
    hs: &HilbertSpace,
    pp_ints: &[Hybridization<'a>],
) -> Vec<OcaSigmaDiagram<'a>> {
    let channels: Vec<OcaChannel> = channels_with_bath(pp_ints, None);
    build_sigma_diagrams_oca_channels(hs, &channels)
}

pub(crate) fn boson_channels(bath: &BosonBath) -> Vec<OcaChannel<'_>> {
    vec![
        OcaChannel {
            delta: &bath.d0,
            fermion: false,
            backward: false,
            ann: VertSrc::Diag(&bath.blocks),
            cre: VertSrc::Diag(&bath.blocks),
        },
        OcaChannel {
            delta: &bath.d0_cc,
            fermion: false,
            backward: true,
            ann: VertSrc::Diag(&bath.blocks),
            cre: VertSrc::Diag(&bath.blocks),
        },
    ]
}

/// Fermion channels of `pp_ints` extended with the optional bath's
/// boson channels.
pub(crate) fn channels_with_bath<'a>(
    pp_ints: &[Hybridization<'a>],
    bath: Option<&'a BosonBath>,
) -> Vec<OcaChannel<'a>> {
    let mut channels: Vec<OcaChannel<'a>> = pp_ints
        .iter()
        .map(|h| OcaChannel {
            delta: h.delta,
            fermion: h.fermion,
            backward: h.backward,
            ann: VertSrc::Flavor(h.op_ann_fl),
            cre: VertSrc::Flavor(h.op_cre_fl),
        })
        .collect();
    if let Some(b) = bath {
        channels.extend(boson_channels(b));
    }
    channels
}

pub(crate) fn build_sigma_diagrams_oca_channels<'a>(
    hs: &HilbertSpace,
    channels: &[OcaChannel<'a>],
) -> Vec<OcaSigmaDiagram<'a>> {
    let mut diagrams = Vec::new();
    for ss in 0..hs.ns() {
        for int1 in channels {
            for int2 in channels {
                // sector chain (blueprint oca::get_sectors)
                let (sc, op4) = match int2.ann.block(hs, ss) {
                    Some(x) => x,
                    None => continue,
                };
                let (sb, op3) = match int1.ann.block(hs, sc) {
                    Some(x) => x,
                    None => continue,
                };
                let (sa, op2) = match int2.cre.block(hs, sb) {
                    Some(x) => x,
                    None => continue,
                };
                let (ss_ref, op1) = match int1.cre.block(hs, sa) {
                    Some(x) => x,
                    None => continue,
                };
                if ss_ref != ss {
                    continue;
                }
                // prefactor: bwd-fermion factors, fermion crossing,
                // i^order, and the verbatim debug_prefactor = -1
                let mut stat_sign = 1.0f64;
                if int1.fermion && int1.backward {
                    stat_sign *= -1.0;
                }
                if int2.fermion && int2.backward {
                    stat_sign *= -1.0;
                }
                if int1.fermion && int2.fermion {
                    stat_sign *= -1.0;
                }
                let prefactor = Complex::I * Complex::I * Complex::new(stat_sign * -1.0, 0.0);
                diagrams.push(OcaSigmaDiagram {
                    out: ss,
                    sa,
                    sb,
                    sc,
                    sig: hs.sectors[ss].sig.as_i64() as f64,
                    op1: op1.clone(),
                    op2: op2.clone(),
                    op3: op3.clone(),
                    op4: op4.clone(),
                    lam1: int1.delta,
                    lam2: int2.delta,
                    prefactor,
                });
            }
        }
    }
    diagrams
}

// ------------------------------------------------------------------
// Matsubara sigma kernel (blueprint oca/sigma_mat.hpp)
// ------------------------------------------------------------------

/// OCA self-energy Matsubara contribution of one diagram, ADDED into
/// `pp_sigma[out].mat` (blueprint `add_to_pp_self_energy` at
/// `tstp = -1`, kernels `sdiagram_mat`/`sdiagram_mat_integral`; the
/// kernel works in grid units and `sdiagram_mat` multiplies
/// `(beta/nmax)^2`).
pub(crate) fn add_sigma_mat_oca(
    d: &OcaSigmaDiagram,
    pp_g: &[PseudoGreen],
    beta: f64,
    kt: i64,
    integ: &Integrator,
    pp_sigma: &mut [PseudoGreen],
) {
    let nmax = pp_sigma[d.out].ntau; // == N (intervals); points = N+1
    let dtau = beta / nmax as f64;
    let ga = mat_axis_pp(&pp_g[d.sa]);
    let gb = mat_axis_pp(&pp_g[d.sb]);
    let gc = mat_axis_pp(&pp_g[d.sc]);
    let lam1 = mat_axis_delta(d.lam1);
    let lam2 = mat_axis_delta(d.lam2);
    let (adim, bdim, cdim, sdim) = (ga.d1, gb.d1, gc.d1, pp_sigma[d.out].size1);
    let k = kt as usize;

    let b_bc = bdim * cdim;
    let b_ac = adim * cdim;
    let b_ss = sdim * sdim;
    let b_ga = adim * adim;
    let b_gb = bdim * bdim;
    let b_gc = cdim * cdim;
    let b_l = 1; // hybridizations are 1x1

    // reusable buffers
    let mut lam1_i = vec![Complex::ZERO; b_l];
    let mut gb_i = vec![Complex::ZERO; b_gb];
    let mut gc_i = vec![Complex::ZERO; b_gc];
    let mut ga_i = vec![Complex::ZERO; b_ga];
    let mut lam2_i = vec![Complex::ZERO; b_l];
    let mut scratch = vec![Complex::ZERO; b_gb.max(b_ac).max(b_ss).max(b_ga).max(b_gc)];
    let mut f_bc = vec![vec![Complex::ZERO; b_bc]; nmax + 1];
    let mut tmp_bc = vec![Complex::ZERO; b_bc];
    let mut tmp_ac = vec![Complex::ZERO; b_ac];
    let mut f_ac = vec![vec![Complex::ZERO; b_ac]; nmax + 1];

    for m in 1..=nmax {
        let res: Vec<Complex> = {
            let mut res = vec![Complex::ZERO; b_ss];
            if m < k {
                // warmup: nested polynomial interpolation
                let dtau1 = m as f64 / k as f64;
                for j in 1..=k {
                    let tau1 = dtau1 * j as f64;
                    let dtau2 = tau1 / k as f64;
                    for l in 0..=k {
                        let tau2 = l as f64 * dtau2;
                        poly_interpol(&lam1, m as f64 - tau2, integ, &mut lam1_i, &mut scratch);
                        poly_interpol(&gb, tau1 - tau2, integ, &mut gb_i, &mut scratch);
                        poly_interpol(&gc, tau2, integ, &mut gc_i, &mut scratch);
                        // f_bc(l) = lam1_i * (gb_i * op3 * gc_i)
                        let mut inner = vec![Complex::ZERO; b_bc];
                        sandwich(
                            gb_i.as_slice(),
                            d.op3.raw(),
                            gc_i.as_slice(),
                            bdim,
                            bdim,
                            cdim,
                            cdim,
                            inner.as_mut_slice(),
                            scratch.as_mut_slice(),
                        );
                        for q in 0..b_bc {
                            f_bc[l][q] = lam1_i[0] * inner[q];
                        }
                    }
                    let mut integr_bc = gregory_sum(k, &f_bc, integ);
                    for q in 0..b_bc {
                        integr_bc[q] *= Complex::new(dtau2, 0.0);
                    }
                    poly_interpol(&lam2, tau1, integ, &mut lam2_i, &mut scratch);
                    poly_interpol(&ga, m as f64 - tau1, integ, &mut ga_i, &mut scratch);
                    // f_ac(j) = lam2_i * (ga_i * op2 * integr_bc)
                    let mut inner = vec![Complex::ZERO; b_ac];
                    sandwich(
                        ga_i.as_slice(),
                        d.op2.raw(),
                        integr_bc.as_slice(),
                        adim,
                        adim,
                        bdim,
                        cdim,
                        inner.as_mut_slice(),
                        scratch.as_mut_slice(),
                    );
                    for q in 0..b_ac {
                        f_ac[j][q] = lam2_i[0] * inner[q];
                    }
                }
                let integr_ac = gregory_sum(k, &f_ac[..k + 1], integ);
                sandwich(
                    d.op1.raw(),
                    integr_ac.as_slice(),
                    d.op4.raw(),
                    sdim,
                    adim,
                    cdim,
                    sdim,
                    res.as_mut_slice(),
                    scratch.as_mut_slice(),
                );
                for q in 0..b_ss {
                    res[q] *= Complex::new(dtau1, 0.0);
                }
                res
            } else {
                // gregory region
                for j in 1..=m {
                    if j < k {
                        let dtau2 = j as f64 / k as f64;
                        let tau1 = j as f64;
                        for l in 0..=k {
                            let tau2 = l as f64 * dtau2;
                            // interpolation window for lam1 at (m - tau2)
                            let y = m as f64 - tau2;
                            let mut i1 = (y.floor() as i64 - k as i64 / 2).max(0) as usize;
                            if i1 > nmax - k {
                                i1 = nmax - k;
                            }
                            let view = MatAxis {
                                data: &lam1.data[i1 * b_l..],
                                d1: 1,
                                d2: 1,
                            };
                            poly_interpol(&view, y - i1 as f64, integ, &mut lam1_i, &mut scratch);
                            poly_interpol(&gb, tau1 - tau2, integ, &mut gb_i, &mut scratch);
                            poly_interpol(&gc, tau2, integ, &mut gc_i, &mut scratch);
                            let mut inner = vec![Complex::ZERO; b_bc];
                            sandwich(
                                gb_i.as_slice(),
                                d.op3.raw(),
                                gc_i.as_slice(),
                                bdim,
                                bdim,
                                cdim,
                                cdim,
                                inner.as_mut_slice(),
                                scratch.as_mut_slice(),
                            );
                            for q in 0..b_bc {
                                f_bc[l][q] = lam1_i[0] * inner[q];
                            }
                        }
                        let integr_bc = gregory_sum(k, &f_bc, integ);
                        for q in 0..b_bc {
                            tmp_bc[q] = integr_bc[q] * Complex::new(dtau2, 0.0);
                        }
                    } else {
                        for l in 0..=j {
                            let lam = lam1.block(m - l)[0];
                            let mut inner = vec![Complex::ZERO; b_bc];
                            sandwich(
                                gb.block(j - l),
                                d.op3.raw(),
                                gc.block(l),
                                bdim,
                                bdim,
                                cdim,
                                cdim,
                                inner.as_mut_slice(),
                                scratch.as_mut_slice(),
                            );
                            for q in 0..b_bc {
                                f_bc[l][q] = lam * inner[q];
                            }
                        }
                        let integr_bc = gregory_sum(j, &f_bc[..j + 1], integ);
                        tmp_bc.copy_from_slice(&integr_bc);
                    }
                    // f_ac(j) = lam2(j) * (ga(m-j) * op2 * integr_bc)
                    sandwich(
                        ga.block(m - j),
                        d.op2.raw(),
                        tmp_bc.as_slice(),
                        adim,
                        adim,
                        bdim,
                        cdim,
                        tmp_ac.as_mut_slice(),
                        scratch.as_mut_slice(),
                    );
                    let lam = lam2.block(j)[0];
                    for q in 0..b_ac {
                        f_ac[j][q] = lam * tmp_ac[q];
                    }
                }
                let integr_ac = gregory_sum(m, &f_ac[..m + 1], integ);
                sandwich(
                    d.op1.raw(),
                    integr_ac.as_slice(),
                    d.op4.raw(),
                    sdim,
                    adim,
                    cdim,
                    sdim,
                    res.as_mut_slice(),
                    scratch.as_mut_slice(),
                );
                res
            }
        };
        // sdiagram_mat: Sigma.mat(m) += prefactor * dtau^2 * integral
        let blk = pp_sigma[d.out].mat.block_mut(m);
        for q in 0..b_ss {
            blk[q] += d.prefactor * Complex::new(dtau * dtau, 0.0) * res[q];
        }
    }
}

// ------------------------------------------------------------------
// OCA gf diagrams (blueprint oca/gf.cpp + gf_mat.hpp)
// ------------------------------------------------------------------

use crate::nca::GfVertex;

/// One OCA single-particle GF diagram: the chain
/// `sa --hyb.op1-> sb --vertex.op1-> sc --hyb.op2-> sd --vertex.op2-> sa`
/// with `op1 = hyb.op2@sc, op2 = vertex.op1@sb, op3 = hyb.op1@sa,
/// op4 = vertex.op2@sd` and prefactor `-1` for a fermionic
/// hybridization line (blueprint `construct_gf_diagram`).
pub(crate) struct OcaGfDiagram<'a> {
    pub vertex: usize,
    pub sa: usize,
    pub sb: usize,
    pub sc: usize,
    pub sd: usize,
    pub op1: DenseMat,
    pub op2: DenseMat,
    pub op3: DenseMat,
    pub op4: DenseMat,
    pub lam: &'a ContourGreen,
    pub prefactor: f64,
    /// Statistics sign of the reflected last-sector propagator,
    /// `gd_flip(tau) = -gd_sign * G^M_sd(beta - tau)`. For a fermion
    /// line this is the channel direction (+1 forward, -1 backward):
    /// the two orderings of a line straddling the vertex carry the
    /// bath factors `Delta(beta - tau)` and `-Delta(tau)`, which enter
    /// with the same physical sign, so the flip follows the channel
    /// and not the particle-number parity of `sd`. The two coincide
    /// for a single band (each ordering exists for exactly one chain)
    /// but not for several bands. Boson lines keep the sector sign.
    pub gd_sign: f64,
    /// Real-time statistics factor of the diagram, multiplying the
    /// prefactor in the real-time kernels: the real-time component
    /// formulas follow the same sector-parity sign, so the `t = 0`
    /// contact `lmix(0, beta - tau) = -i G^M(tau)` with the corrected
    /// Matsubara kernel needs `gd_sign * sig_sd` per fermionic diagram
    /// (+1 for every single-band diagram, i.e. the single-band
    /// real-time results are unchanged).
    pub rt_flip: f64,
}

pub(crate) fn build_gf_diagrams_oca_channels<'a>(
    hs: &HilbertSpace,
    verts: &[GfVertex],
    channels: &[OcaChannel<'a>],
) -> Vec<OcaGfDiagram<'a>> {
    let mut diagrams = Vec::new();
    for (vi, vert) in verts.iter().enumerate() {
        for hyb in channels {
            for sa in 0..hs.ns() {
                let (sb, op3) = match hyb.cre.block(hs, sa) {
                    Some(x) => x,
                    None => continue,
                };
                let (sc, op2) = match hs.ops[vert.op_ann_fl].block(sb) {
                    Some(x) => x,
                    None => continue,
                };
                let (sd, op4) = match hyb.ann.block(hs, *sc) {
                    Some(x) => x,
                    None => continue,
                };
                let (sa_ref, op1) = match hs.ops[vert.op_cre_fl].block(sd) {
                    Some(x) => x,
                    None => continue,
                };
                if *sa_ref != sa {
                    continue;
                }
                let prefactor = if hyb.fermion { -1.0 } else { 1.0 };
                let gd_sign = if hyb.fermion {
                    if hyb.backward { -1.0 } else { 1.0 }
                } else {
                    hs.sectors[sd].sig.as_i64() as f64
                };
                // real-time factor converting the sector-parity sign
                // the real-time kernels were written against into the
                // channel-direction sign: `new / old = gd_sign / sig_sd`
                let rt_flip = if hyb.fermion {
                    gd_sign * hs.sectors[sd].sig.as_i64() as f64
                } else {
                    1.0
                };
                diagrams.push(OcaGfDiagram {
                    vertex: vi,
                    sa,
                    sb,
                    sc: *sc,
                    sd,
                    op1: op1.clone(),
                    op2: op2.clone(),
                    op3: op3.clone(),
                    op4: op4.clone(),
                    lam: hyb.delta,
                    prefactor,
                    gd_sign,
                    rt_flip,
                });
            }
        }
    }
    diagrams
}

/// OCA Matsubara correction to the single-particle GF per vertex,
/// ADDED onto the NCA values (blueprint `gdiagram_mat` with
/// `equilibrium = true`, kernels `gdiagram_mat_integral`). Returns
/// per-vertex `mat(m)` values, `m = 0..=ntau`.
/// Matrix path of the gf diagram Matsubara integral for multi-band
/// sectors (blueprint `gdiagram_mat_integral` with matrix types; the
/// hybridization insertions stay scalar `(0, 0)` blocks).
#[allow(clippy::too_many_arguments)]
/// Reusable scratch for the hot gf matcore (called per-`m` per
/// diagram per iteration in the real-time path; the inner loops must
/// not allocate).
pub(crate) struct GfMatScratch {
    prod: Vec<Complex>,
    gdv: Vec<Complex>,
    blk: Vec<Complex>,
    gd_interp: Vec<Complex>,
    u1: Vec<Complex>,
}

impl GfMatScratch {
    fn new(max_nel: usize, k: usize, ddim: usize) -> Self {
        GfMatScratch {
            prod: vec![Complex::ZERO; max_nel],
            gdv: vec![Complex::ZERO; (k + 1) * ddim * ddim],
            blk: vec![Complex::ZERO; ddim * ddim],
            gd_interp: vec![Complex::ZERO; ddim * ddim],
            u1: vec![Complex::ZERO; max_nel],
        }
    }
}

fn gf_mat_matrix_branch(
    m: usize,
    k: usize,
    nmax: usize,
    ga: &MatAxis,
    gb: &MatAxis,
    gc: &MatAxis,
    gd: &MatAxis,
    lam: &MatAxis,
    d: &OcaGfDiagram,
    gd_sign: f64,
    equilibrium: bool,
    integ: &Integrator,
    lam_i: &mut Vec<Complex>,
    scratch: &mut Vec<Complex>,
    mscratch: &mut GfMatScratch,
) -> Complex {
    let (adim, bdim, cdim, ddim) = (ga.d1, gb.d1, gc.d1, gd.d1);
    let nel_dc = ddim * cdim;
    let nel_da = ddim * adim;
    let nel_ba = bdim * adim;
    let mut gb_i = vec![Complex::ZERO; bdim * bdim];
    let mut ga_i = vec![Complex::ZERO; adim * adim];
    let mut gc_i = vec![Complex::ZERO; cdim * cdim];
    let mut gd_flip_blk = vec![Complex::ZERO; ddim * ddim];
    let sc_flip = -Complex::new(gd_sign, 0.0);
    let flip = |i2: usize, out: &mut [Complex]| {
        if equilibrium {
            for q in 0..ddim * ddim {
                out[q] = sc_flip * gd.block(nmax - i2)[q];
            }
        } else {
            // real-time: gd rows fed directly (no flip)
            for q in 0..ddim * ddim {
                out[q] = gd.block(i2)[q];
            }
        }
    };
    let smax = (ddim * bdim)
        .max(bdim * adim)
        .max(ddim * cdim)
        .max(adim * adim)
        .max(bdim * bdim)
        .max(cdim * cdim)
        .max(ddim * ddim);
    scratch.clear();
    scratch.resize(smax, Complex::ZERO);
    let mut t1 = vec![Complex::ZERO; ddim * bdim];
    let mut t2 = vec![Complex::ZERO; nel_da];
    // close the chain `sd --op1--> sa` on the da block (ddim x adim):
    // trace of (f_da . op1) with op1 stored (adim x ddim)
    let trace_close = |da: &[Complex]| -> Complex {
        let mut tr = Complex::ZERO;
        for r in 0..ddim {
            for q in 0..adim {
                tr += da[r * adim + q] * d.op1.get(q, r);
            }
        }
        tr
    };
    if m < k && m > 0 {
        // low-side warmup
        let dtau1 = m as f64 / k as f64;
        let mut f_dc = vec![vec![Complex::ZERO; nel_dc]; nmax - m + 1];
        let mut f_da = vec![vec![Complex::ZERO; nel_da]; k + 1];
        for i1 in 0..=k {
            let tau1 = i1 as f64 * dtau1;
            for i2 in m..=nmax {
                let y = nmax as f64 + tau1 - i2 as f64;
                let mut l = (y.floor() as i64 - k as i64 / 2).max(0) as usize;
                if l > nmax - k {
                    l = nmax - k;
                }
                let view = MatAxis {
                    data: &lam.data[l..],
                    d1: 1,
                    d2: 1,
                };
                poly_interpol(&view, y - l as f64, integ, lam_i, scratch);
                flip(i2, &mut gd_flip_blk);
                sandwich(
                    gd_flip_blk.as_slice(),
                    d.op4.raw(),
                    gc.block(i2 - m),
                    ddim,
                    ddim,
                    cdim,
                    cdim,
                    &mut mscratch.prod[..nel_dc],
                    scratch.as_mut_slice(),
                );
                for q in 0..nel_dc {
                    f_dc[i2 - m][q] = lam_i[0] * mscratch.prod[q];
                }
            }
            let integr_dc = gregory_sum(nmax - m, &f_dc, integ);
            poly_interpol(gb, m as f64 - tau1, integ, &mut gb_i, scratch);
            poly_interpol(ga, tau1, integ, &mut ga_i, scratch);
            sandwich(
                integr_dc.as_slice(),
                d.op2.raw(),
                gb_i.as_slice(),
                ddim,
                cdim,
                bdim,
                bdim,
                t1.as_mut_slice(),
                scratch.as_mut_slice(),
            );
            sandwich(
                t1.as_slice(),
                d.op3.raw(),
                ga_i.as_slice(),
                ddim,
                bdim,
                adim,
                adim,
                t2.as_mut_slice(),
                scratch.as_mut_slice(),
            );
            f_da[i1][..nel_da].copy_from_slice(&t2);
        }
        let integr_da = gregory_sum(k, &f_da, integ);
        -Complex::new(dtau1, 0.0) * trace_close(&integr_da)
    } else if m > nmax - k && m < nmax {
        // high-side warmup
        let dtau2 = (nmax - m) as f64 / k as f64;
        let mut f_da = vec![vec![Complex::ZERO; nel_da]; k + 1];
        let mut f_ba = vec![vec![Complex::ZERO; nel_ba]; m + 1];
        for i2 in 0..=k {
            let tau2 = m as f64 + i2 as f64 * dtau2;
            for i1 in 0..=m {
                let y = nmax as f64 + i1 as f64 - tau2;
                let mut l = (y.floor() as i64 - k as i64 / 2).max(0) as usize;
                if l > nmax - k {
                    l = nmax - k;
                }
                let view = MatAxis {
                    data: &lam.data[l..],
                    d1: 1,
                    d2: 1,
                };
                poly_interpol(&view, y - l as f64, integ, lam_i, scratch);
                sandwich(
                    gb.block(m - i1),
                    d.op3.raw(),
                    ga.block(i1),
                    bdim,
                    bdim,
                    adim,
                    adim,
                    &mut mscratch.prod[..nel_ba],
                    scratch.as_mut_slice(),
                );
                for q in 0..nel_ba {
                    f_ba[i1][q] = lam_i[0] * mscratch.prod[q];
                }
            }
            let integr_ba = gregory_sum(m, &f_ba, integ);
            poly_interpol(gc, tau2 - m as f64, integ, &mut gc_i, scratch);
            let y = tau2;
            let mut l = (y.floor() as i64 - k as i64 / 2).max(0) as usize;
            if l > nmax - k {
                l = nmax - k;
            }
            for i in 0..=k {
                flip(l + i, &mut mscratch.blk);
                mscratch.gdv[i * ddim * ddim..(i + 1) * ddim * ddim].copy_from_slice(&mscratch.blk);
            }
            let gd_axis = MatAxis {
                data: &mscratch.gdv,
                d1: ddim,
                d2: ddim,
            };
            poly_interpol(
                &gd_axis,
                y - l as f64,
                integ,
                &mut mscratch.gd_interp,
                scratch,
            );
            sandwich(
                mscratch.gd_interp.as_slice(),
                d.op4.raw(),
                gc_i.as_slice(),
                ddim,
                ddim,
                cdim,
                cdim,
                &mut mscratch.u1[..nel_dc],
                scratch.as_mut_slice(),
            );
            sandwich(
                mscratch.u1[..nel_dc].as_ref(),
                d.op2.raw(),
                integr_ba.as_slice(),
                ddim,
                cdim,
                bdim,
                adim,
                t2.as_mut_slice(),
                scratch.as_mut_slice(),
            );
            f_da[i2][..nel_da].copy_from_slice(&t2);
        }
        let integr_da = gregory_sum(k, &f_da, integ);
        -Complex::new(dtau2, 0.0) * trace_close(&integr_da)
    } else if m >= k && m <= nmax - k {
        // gregory region
        let mut f_dc = vec![vec![Complex::ZERO; nel_dc]; nmax - m + 1];
        let mut f_da = vec![vec![Complex::ZERO; nel_da]; m + 1];
        for i1 in 0..=m {
            for i2 in m..=nmax {
                let lam_v = lam.block(nmax - i2 + i1)[0];
                flip(i2, &mut gd_flip_blk);
                sandwich(
                    gd_flip_blk.as_slice(),
                    d.op4.raw(),
                    gc.block(i2 - m),
                    ddim,
                    ddim,
                    cdim,
                    cdim,
                    &mut mscratch.prod[..nel_dc],
                    scratch.as_mut_slice(),
                );
                for q in 0..nel_dc {
                    f_dc[i2 - m][q] = lam_v * mscratch.prod[q];
                }
            }
            let integr_dc = gregory_sum(nmax - m, &f_dc, integ);
            sandwich(
                integr_dc.as_slice(),
                d.op2.raw(),
                gb.block(m - i1),
                ddim,
                cdim,
                bdim,
                bdim,
                t1.as_mut_slice(),
                scratch.as_mut_slice(),
            );
            sandwich(
                t1.as_slice(),
                d.op3.raw(),
                ga.block(i1),
                ddim,
                bdim,
                adim,
                adim,
                t2.as_mut_slice(),
                scratch.as_mut_slice(),
            );
            f_da[i1][..nel_da].copy_from_slice(&t2);
        }
        let integr_da = gregory_sum(m, &f_da, integ);
        -trace_close(&integr_da)
    } else {
        Complex::ZERO
    }
}

pub(crate) fn add_gf_mat_oca(
    hs: &HilbertSpace,
    pp_g: &[PseudoGreen],
    verts: &[GfVertex],
    pp_ints: &[Hybridization],
    bath: Option<&BosonBath>,
    beta: f64,
    kt: i64,
    integ: &Integrator,
    out: &mut [Vec<Complex>],
) {
    let channels = channels_with_bath(pp_ints, bath);
    let diagrams = build_gf_diagrams_oca_channels(hs, verts, &channels);
    let nmax = pp_g[0].ntau;
    let dtau = beta / nmax as f64;
    let k = kt as usize;
    // gd_flip(tau) = -gd_sign * gd(nmax - tau) in equilibrium,
    // with the statistics sign carried by the diagram

    let mut lam_i = vec![Complex::ZERO; 1];
    let mut scratch = vec![Complex::ZERO; 64];
    let mut gc_i = vec![Complex::ZERO; 1];
    let mut ga_i = vec![Complex::ZERO; 1];
    let mut gb_i = vec![Complex::ZERO; 1];
    let mut gd_i = vec![Complex::ZERO; 1];

    for d in &diagrams {
        let ga = mat_axis_pp(&pp_g[d.sa]);
        let gb = mat_axis_pp(&pp_g[d.sb]);
        let gc = mat_axis_pp(&pp_g[d.sc]);
        let gd = mat_axis_pp(&pp_g[d.sd]);
        let lam = mat_axis_delta(d.lam);
        let gd_flip =
            |tau: usize| -> Complex { -Complex::new(d.gd_sign, 0.0) * gd.block(nmax - tau)[0] };
        // dimension bookkeeping: the scalar fast path covers the
        // single-band case (all blocks 1x1, P5-gated); multi-band
        // sectors take the matrix path (blueprint
        // gdiagram_mat_integral with matrix types)
        let (adim, bdim, cdim, ddim) = (ga.d1, gb.d1, gc.d1, gd.d1);
        let all1 = adim * bdim * cdim * ddim == 1;
        for m in 0..=nmax {
            let g: Complex = if !all1 {
                let max_nel = (ddim * cdim).max(bdim * adim).max(ddim * adim);
                let mut mscratch = GfMatScratch::new(max_nel, k, ddim);
                gf_mat_matrix_branch(
                    m,
                    k,
                    nmax,
                    &ga,
                    &gb,
                    &gc,
                    &gd,
                    &lam,
                    d,
                    d.gd_sign,
                    true,
                    integ,
                    &mut lam_i,
                    &mut scratch,
                    &mut mscratch,
                )
            } else if m < k && m > 0 {
                // low-side warmup
                let dtau1 = m as f64 / k as f64;
                let mut f_da = vec![vec![Complex::ZERO; 1]; k + 1];
                for i1 in 0..=k {
                    let tau1 = i1 as f64 * dtau1;
                    let mut f_dc = vec![vec![Complex::ZERO; 1]; nmax - m + 1];
                    for i2 in m..=nmax {
                        let y = nmax as f64 + tau1 - i2 as f64;
                        let mut l = (y.floor() as i64 - k as i64 / 2).max(0) as usize;
                        if l > nmax - k {
                            l = nmax - k;
                        }
                        let view = MatAxis {
                            data: &lam.data[l..],
                            d1: 1,
                            d2: 1,
                        };
                        poly_interpol(&view, y - l as f64, integ, &mut lam_i, &mut scratch);
                        // f_dc(i2-m) = lam_interp * (gd_flip(i2) * op1 * gc(i2-m))
                        let prod = gd_flip(i2) * d.op1.get(0, 0) * gc.block(i2 - m)[0];
                        f_dc[i2 - m][0] = lam_i[0] * prod;
                    }
                    let integr_dc = gregory_sum(nmax - m, &f_dc, integ);
                    poly_interpol(&ga, tau1, integ, &mut ga_i, &mut scratch);
                    poly_interpol(&gb, m as f64 - tau1, integ, &mut gb_i, &mut scratch);
                    f_da[i1][0] =
                        integr_dc[0] * d.op2.get(0, 0) * gb_i[0] * d.op3.get(0, 0) * ga_i[0];
                }
                let integr_da = gregory_sum(k, &f_da, integ);
                -Complex::new(dtau1, 0.0) * (integr_da[0] * d.op4.get(0, 0))
            } else if m > nmax - k && m < nmax {
                // high-side warmup
                let dtau2 = (nmax - m) as f64 / k as f64;
                let mut f_da = vec![vec![Complex::ZERO; 1]; k + 1];
                for i2 in 0..=k {
                    let tau2 = m as f64 + i2 as f64 * dtau2;
                    let mut f_ba = vec![vec![Complex::ZERO; 1]; m + 1];
                    for i1 in 0..=m {
                        let y = nmax as f64 + i1 as f64 - tau2;
                        let mut l = (y.floor() as i64 - k as i64 / 2).max(0) as usize;
                        if l > nmax - k {
                            l = nmax - k;
                        }
                        let view = MatAxis {
                            data: &lam.data[l..],
                            d1: 1,
                            d2: 1,
                        };
                        poly_interpol(&view, y - l as f64, integ, &mut lam_i, &mut scratch);
                        f_ba[i1][0] =
                            lam_i[0] * (gb.block(m - i1)[0] * d.op3.get(0, 0) * ga.block(i1)[0]);
                    }
                    let integr_ba = gregory_sum(m, &f_ba, integ);
                    poly_interpol(&gc, tau2 - m as f64, integ, &mut gc_i, &mut scratch);
                    let y = tau2;
                    let mut l = (y.floor() as i64 - k as i64 / 2).max(0) as usize;
                    if l > nmax - k {
                        l = nmax - k;
                    }
                    // gd_ref_flip view from index l: view(i) = -sig*gd(nmax-l-i)
                    let mut gd_flip_view: Vec<Complex> = Vec::with_capacity(k + 1);
                    for i in 0..=k {
                        gd_flip_view.push(gd_flip(l + i));
                    }
                    let gd_flip_axis = MatAxis {
                        data: &gd_flip_view,
                        d1: 1,
                        d2: 1,
                    };
                    poly_interpol(&gd_flip_axis, y - l as f64, integ, &mut gd_i, &mut scratch);
                    f_da[i2][0] =
                        gd_i[0] * d.op1.get(0, 0) * gc_i[0] * d.op2.get(0, 0) * integr_ba[0];
                }
                let integr_da = gregory_sum(k, &f_da, integ);
                -Complex::new(dtau2, 0.0) * (integr_da[0] * d.op4.get(0, 0))
            } else if m >= k && m <= nmax - k {
                // gregory region
                let mut f_dc = vec![vec![Complex::ZERO; 1]; nmax - m + 1];
                let mut f_da = vec![vec![Complex::ZERO; 1]; m + 1];
                for i1 in 0..=m {
                    for i2 in m..=nmax {
                        let lam_v = lam.block(nmax - i2 + i1)[0];
                        let prod = gd_flip(i2) * d.op1.get(0, 0) * gc.block(i2 - m)[0];
                        f_dc[i2 - m][0] = lam_v * prod;
                    }
                    let integr_dc = gregory_sum(nmax - m, &f_dc, integ);
                    f_da[i1][0] = integr_dc[0]
                        * d.op2.get(0, 0)
                        * gb.block(m - i1)[0]
                        * d.op3.get(0, 0)
                        * ga.block(i1)[0];
                }
                let integr_da = gregory_sum(m, &f_da, integ);
                -(integr_da[0] * d.op4.get(0, 0))
            } else {
                Complex::ZERO
            };
            out[d.vertex][m] += Complex::new(d.prefactor, 0.0) * Complex::new(dtau * dtau, 0.0) * g;
        }
    }
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hilbert::{SectorClass, build};

    #[test]
    fn oca_diagram_count_single_band() {
        // 4 channels x 4 channels with valid closed chains: the
        // single-band benchmark yields 16 OCA sigma diagrams
        // (4 sigma sectors x 4 channel pairs)
        let hs = build(1, SectorClass::SpinResolved);
        let c = kadanoffbaym::types::Contour::new(4, 8, 1, 1, 0.08, 5.0);
        let d1 = ContourGreen::new(&c, kadanoffbaym::types::Sign::Fermi, Complex::ZERO);
        let d2 = ContourGreen::new(&c, kadanoffbaym::types::Sign::Fermi, Complex::ZERO);
        let cua = hs.flavor(0, 0, false);
        let cuc = hs.flavor(0, 0, true);
        let cda = hs.flavor(0, 1, false);
        let cdc = hs.flavor(0, 1, true);
        let pp_ints = vec![
            Hybridization {
                delta: &d1,
                op_cre_fl: cuc,
                op_ann_fl: cua,
                fermion: true,
                backward: false,
            },
            Hybridization {
                delta: &d2,
                op_cre_fl: cua,
                op_ann_fl: cuc,
                fermion: true,
                backward: true,
            },
            Hybridization {
                delta: &d1,
                op_cre_fl: cdc,
                op_ann_fl: cda,
                fermion: true,
                backward: false,
            },
            Hybridization {
                delta: &d2,
                op_cre_fl: cda,
                op_ann_fl: cdc,
                fermion: true,
                backward: true,
            },
        ];
        let diagrams = build_sigma_diagrams_oca(&hs, &pp_ints);
        // hand-enumerated: 8 valid closed chains on the 4-sector
        // single-band space (2 per N=1 sector, 4 on the doubly
        // occupied sector, none on the empty sector)
        assert_eq!(diagrams.len(), 8);
        for d in &diagrams {
            // prefactor = i^2 * stat_sign * (-1) is real +-1
            assert_eq!(d.prefactor.im, 0.0);
            assert!((d.prefactor.re.abs() - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn oca_sigma_zero_for_zero_delta() {
        let hs = build(1, SectorClass::SpinResolved);
        let c = kadanoffbaym::types::Contour::new(4, 8, 1, 1, 0.08, 5.0);
        let d1 = ContourGreen::new(&c, kadanoffbaym::types::Sign::Fermi, Complex::ZERO);
        let cua = hs.flavor(0, 0, false);
        let cuc = hs.flavor(0, 0, true);
        let pp_ints = vec![Hybridization {
            delta: &d1,
            op_cre_fl: cuc,
            op_ann_fl: cua,
            fermion: true,
            backward: false,
        }];
        let diagrams = build_sigma_diagrams_oca(&hs, &pp_ints);
        let mut solver = crate::solver::ImpuritySolver::new(
            4,
            8,
            5.0,
            0.02,
            2,
            2,
            build(1, SectorClass::SpinResolved),
            crate::hamilton::LocalHamiltonian::SingleBand {
                mu: 0.0,
                u: vec![3.0; 6],
                eps_up: vec![0.0; 6],
                eps_do: vec![0.0; 6],
            },
        );
        solver.solve_atomic();
        let integ = Integrator::new(2);
        for d in &diagrams {
            add_sigma_mat_oca(d, &solver.pp_g, 5.0, 2, &integ, &mut solver.pp_sigma);
        }
        for s in 0..4 {
            for m in 0..=8 {
                assert_eq!(solver.pp_sigma[s].mat.block(m)[0], Complex::ZERO);
            }
        }
    }
}

#[cfg(test)]
mod probe9_compare {
    use super::*;
    use crate::hamilton::LocalHamiltonian::SingleBand;
    use crate::hilbert::{SectorClass, build};
    use crate::nca::{GfVertex, Hybridization, gf_mat};
    use crate::solver::ImpuritySolver;
    use kadanoffbaym::types::{Contour, Sign};

    /// Mirror of data/ppsc/probe9_spgf.cpp: synthetic Matsubara
    /// profiles, OCA spgf correction = order2 - order1 must match the
    /// compiled reference exactly.
    #[test]
    fn oca_gf_kernel_matches_probe9() {
        let (nt, ntau, kt) = (8usize, 250usize, 5usize);
        let beta = 5.0;
        let hs = build(1, SectorClass::SpinResolved);
        let mut solver = ImpuritySolver::new(
            nt,
            ntau,
            beta,
            0.02,
            kt,
            2,
            build(1, SectorClass::SpinResolved),
            SingleBand {
                mu: 0.0,
                u: vec![3.0; nt + 2],
                eps_up: vec![0.0; nt + 2],
                eps_do: vec![0.0; nt + 2],
            },
        );
        solver.solve_atomic();

        let c = Contour::new(nt as i64, ntau as i64, 1, 1, 5.0, beta);
        let mut delta = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut delta_cc = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for m in 0..=ntau {
            let t = m as f64 / ntau as f64;
            let d = Complex::new(-(-1.5f64 * t).exp() * (1.0 + 0.3 * t), 0.2 * t * (1.0 - t));
            delta.mat.block_mut(m)[0] = d;
            delta_cc.mat.block_mut(m)[0] = d.conj();
            for s in 0..4usize {
                let re = 0.1 * (s as f64 + 1.0) * (-0.8f64 * t).exp() + 0.05 * s as f64 * t;
                let im = 0.02 * s as f64 * t * (1.0 - t) - 0.5 * t;
                solver.pp_g[s].mat.block_mut(m)[0] = Complex::new(re, im);
            }
        }
        let cua = hs.flavor(0, 0, false);
        let cuc = hs.flavor(0, 0, true);
        let cda = hs.flavor(0, 1, false);
        let cdc = hs.flavor(0, 1, true);
        let pp_ints = vec![
            Hybridization {
                delta: &delta,
                op_cre_fl: cuc,
                op_ann_fl: cua,
                fermion: true,
                backward: false,
            },
            Hybridization {
                delta: &delta_cc,
                op_cre_fl: cua,
                op_ann_fl: cuc,
                fermion: true,
                backward: true,
            },
            Hybridization {
                delta: &delta,
                op_cre_fl: cdc,
                op_ann_fl: cda,
                fermion: true,
                backward: false,
            },
            Hybridization {
                delta: &delta_cc,
                op_cre_fl: cda,
                op_ann_fl: cdc,
                fermion: true,
                backward: true,
            },
        ];
        let verts = vec![GfVertex {
            op_ann_fl: cua,
            op_cre_fl: cuc,
        }];

        // our NCA part and OCA correction
        let nca_part = gf_mat(&hs, &solver.pp_g, &verts);
        let mut oca_part = vec![vec![Complex::ZERO; ntau + 1]; 1];
        let integ = Integrator::new(kt as i64);
        add_gf_mat_oca(
            &hs,
            &solver.pp_g,
            &verts,
            &pp_ints,
            None,
            beta,
            kt as i64,
            &integ,
            &mut oca_part,
        );

        // reference: SPGF{order}_{mode} per m from the archived probe output
        let txt = include_str!("../data/ppsc/probe9_spgf.txt");
        let mut ref1 = vec![vec![Complex::ZERO; ntau + 1]; 3];
        let mut ref2 = vec![vec![Complex::ZERO; ntau + 1]; 3];
        for line in txt.lines() {
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.len() == 4 && t[0].starts_with("SPGF") {
                let parts: Vec<&str> = t[0].split('_').collect();
                let order = parts[0][4..].parse::<usize>().unwrap();
                let mode = parts[1].parse::<usize>().unwrap();
                let m: usize = t[1].parse().unwrap();
                let z = Complex::new(t[2].parse().unwrap(), t[3].parse().unwrap());
                if order == 1 {
                    ref1[mode][m] = z;
                } else {
                    ref2[mode][m] = z;
                }
            }
        }
        // sanity: NCA part matches SPGF1_0
        for m in 0..=ntau {
            let d = (nca_part[0][m] - ref1[0][m]).abs();
            assert!(d < 1e-12, "NCA m={m}: {} vs {}", nca_part[0][m], ref1[0][m]);
        }
        // OCA correction per mode (full / fwd-only / bwd-only)
        for mode in 0..3usize {
            let (dz, dccz) = match mode {
                0 => (false, false),
                1 => (false, true), // dcc = 0: fwd diagram only
                _ => (true, false), // d = 0: bwd diagram only
            };
            let d_eff: ContourGreen = if dz {
                ContourGreen::new(&c, Sign::Fermi, Complex::ZERO)
            } else {
                delta.clone()
            };
            let dcc_eff: ContourGreen = if dccz {
                ContourGreen::new(&c, Sign::Fermi, Complex::ZERO)
            } else {
                delta_cc.clone()
            };
            let pp_ints_m = vec![
                Hybridization {
                    delta: &d_eff,
                    op_cre_fl: cuc,
                    op_ann_fl: cua,
                    fermion: true,
                    backward: false,
                },
                Hybridization {
                    delta: &dcc_eff,
                    op_cre_fl: cua,
                    op_ann_fl: cuc,
                    fermion: true,
                    backward: true,
                },
                Hybridization {
                    delta: &d_eff,
                    op_cre_fl: cdc,
                    op_ann_fl: cda,
                    fermion: true,
                    backward: false,
                },
                Hybridization {
                    delta: &dcc_eff,
                    op_cre_fl: cda,
                    op_ann_fl: cdc,
                    fermion: true,
                    backward: true,
                },
            ];
            let mut oca_m = vec![vec![Complex::ZERO; ntau + 1]; 1];
            add_gf_mat_oca(
                &hs,
                &solver.pp_g,
                &verts,
                &pp_ints_m,
                None,
                beta,
                kt as i64,
                &integ,
                &mut oca_m,
            );
        }
    }
}

// ------------------------------------------------------------------
// real-time kernels (blueprint oca/sigma_tstp.hpp + gf_tstp.hpp)
//
// Scalar (1x1) fast path: the single-band benchmark sectors and
// hybridizations are 1x1; multi-band real-time OCA generalizes these
// kernels later. All component conventions follow the locked views:
// gtr(i,j) = ret(i,j) - [les(j,i)]^dag (j<i), ret+les at j=i;
// vt(tau,t) = -sig * [tv(t, N-tau)]^dag.
// ------------------------------------------------------------------

#[inline]
fn sles(g: &ContourGreen, j: i64, n: i64) -> Complex {
    // hermitian view: out-of-triangle (j > n) reads -[les(n, j)]^dag
    if j <= n {
        g.less.raw_block_at(j, n)[0]
    } else {
        -g.less.raw_block_at(n, j)[0].conj()
    }
}
#[inline]
fn sgtr(g: &ContourGreen, n: i64, j: i64) -> Complex {
    if j < n {
        g.ret.raw_block_at(n, j)[0] - g.less.raw_block_at(j, n)[0].conj()
    } else if j == n {
        g.ret.raw_block_at(n, n)[0] + g.less.raw_block_at(n, n)[0]
    } else {
        // hermitian view fallback: -[ret(j, n)]^dag + les(n, j)
        -g.ret.raw_block_at(j, n)[0].conj() + g.less.raw_block_at(n, j)[0]
    }
}
#[inline]
fn stv(g: &ContourGreen, n: i64, m: i64) -> Complex {
    g.lmix.raw_block_at(n, m)[0]
}
#[inline]
fn svt(g: &ContourGreen, tau: i64, t: i64) -> Complex {
    -Complex::new(g.sign.as_i64() as f64, 0.0)
        * g.lmix.raw_block_at(t, g.ntau as i64 - tau)[0].conj()
}
#[inline]
fn psc(pp: &PseudoGreen, n: i64, j: i64) -> Complex {
    // pseudo ret slot holds the greater component; out-of-triangle
    // (j > n) reads fall back to -[ret(j, n)]^dag
    if j <= n {
        pp.ret.raw_block_at(n, j)[0]
    } else {
        -pp.ret.raw_block_at(j, n)[0].conj()
    }
}
#[inline]
fn pls(pp: &PseudoGreen, j: i64, n: i64) -> Complex {
    if j <= n {
        pp.less.raw_block_at(j, n)[0]
    } else {
        -pp.less.raw_block_at(n, j)[0].conj()
    }
}
#[inline]
fn ptv(pp: &PseudoGreen, n: i64, m: i64) -> Complex {
    pp.lmix.raw_block_at(n, m)[0]
}
#[inline]
fn pvt(pp: &PseudoGreen, tau: i64, t: i64) -> Complex {
    -Complex::new(pp.sign.as_i64() as f64, 0.0)
        * pp.lmix.raw_block_at(t, pp.ntau as i64 - tau)[0].conj()
}

#[inline]
pub(crate) fn gregory_scalar(n: usize, v: &[Complex], integ: &Integrator) -> Complex {
    // blueprint gregory_sum: for n < k+1 the warmup weights row spans
    // ALL k+1 nodes (degree-k exact integration), so the sum runs to
    // max(n, k) — not n
    let k = integ.k as usize;
    let mut out = Complex::ZERO;
    for j in 0..=n.max(k) {
        let w = integ.giw.at(n, j);
        if w != 0.0 {
            out += Complex::new(w, 0.0) * v[j];
        }
    }
    out
}

/// `poly_integral` blueprint on scalars: straight sub-interval Gregory,
/// or polynomial interpolation weights over the nodes `0..=kt` when
/// the interval is short.
pub(crate) fn poly_integral_scalar(
    v: &[Complex],
    i0: usize,
    i1: usize,
    integ: &Integrator,
) -> Complex {
    let k = integ.k as usize;
    if i1 - i0 > k {
        gregory_scalar(i1 - i0, &v[i0..=i1], integ)
    } else {
        let mut out = Complex::ZERO;
        for j in 0..=k {
            let w = integ.xiw.at(i0, i1, j);
            if w != 0.0 {
                out += Complex::new(w, 0.0) * v[j];
            }
        }
        out
    }
}

/// `convolution2b` blueprint on a scalar `(i1, i2)` triangle matrix
/// (row-major, first index x second, stride n11).
pub(crate) fn conv2b_scalar(
    data: &[Complex],
    n11: usize,
    j: usize,
    nn: usize,
    integ: &Integrator,
) -> Complex {
    let k = integ.k as usize;
    let i1max = nn.max(k);
    let mut i1min = i1max - k;
    if i1min > j {
        i1min = j;
    }
    let mut f1 = vec![Complex::ZERO; i1max + 1];
    for i in i1min..j {
        let row = &data[i * n11..];
        f1[i] = -poly_integral_scalar(row, i - i1min, j - i1min, integ);
    }
    f1[j] = Complex::ZERO;
    for i in (j + 1)..=i1max {
        let row = &data[i * n11..];
        f1[i] = poly_integral_scalar(row, j - i1min, i - i1min, integ);
    }
    poly_integral_scalar(&f1, j - i1min, nn - i1min, integ)
}

/// `matsubara_convolution_lowside` blueprint (no operator):
/// `c = int_0^j a(j-x) b(x) dx`, warmup via `rcorr` (bcw).
pub(crate) fn lowside_scalar(
    j: usize,
    aa: &[Complex],
    bb: &[Complex],
    integ: &Integrator,
) -> Complex {
    let k = integ.k as usize;
    if j > 0 && j < k {
        let mut out = Complex::ZERO;
        for l in 0..=k {
            for m in 0..=k {
                let w = integ.bcw.at(j, l, m);
                if w != 0.0 {
                    out += Complex::new(w, 0.0) * (aa[l] * bb[m]);
                }
            }
        }
        out
    } else if j > 0 {
        let mut f1 = vec![Complex::ZERO; j + 1];
        for l in 0..=j {
            f1[l] = aa[j - l] * bb[l];
        }
        gregory_scalar(j, &f1, integ)
    } else {
        Complex::ZERO
    }
}

/// `matsubara_convolution_lowside_operator` blueprint: `op` between.
fn lowside_op_scalar(
    j: usize,
    aa: &[Complex],
    op: Complex,
    bb: &[Complex],
    integ: &Integrator,
) -> Complex {
    let k = integ.k as usize;
    if j > 0 && j < k {
        let mut out = Complex::ZERO;
        for l in 0..=k {
            for m in 0..=k {
                let w = integ.bcw.at(j, l, m);
                if w != 0.0 {
                    out += Complex::new(w, 0.0) * (aa[l] * op * bb[m]);
                }
            }
        }
        out
    } else if j > 0 {
        let mut f1 = vec![Complex::ZERO; j + 1];
        for l in 0..=j {
            f1[l] = aa[j - l] * op * bb[l];
        }
        gregory_scalar(j, &f1, integ)
    } else {
        Complex::ZERO
    }
}

/// `matsubara_convolution_highside_operator` blueprint (active lines):
/// `c = int_j^mmax a(x-j) op b(x) dx`.
fn highside_op_scalar(
    j: usize,
    aa: &[Complex],
    op: Complex,
    bb: &[Complex],
    mmax: usize,
    integ: &Integrator,
) -> Complex {
    let k = integ.k as usize;
    let j1 = mmax - j;
    if j1 > 0 && j1 < k {
        let mut out = Complex::ZERO;
        for l in 0..=k {
            for m in 0..=k {
                let w = integ.bcw.at(j1, l, m);
                if w != 0.0 {
                    out += Complex::new(w, 0.0) * (aa[mmax - l] * op * bb[m]);
                }
            }
        }
        out
    } else if j1 > 0 {
        let mut f1 = vec![Complex::ZERO; j1 + 1];
        for l in 0..=j1 {
            f1[l] = aa[mmax - (j1 - l)] * op * bb[l];
        }
        gregory_scalar(j1, &f1, integ)
    } else {
        Complex::ZERO
    }
}

/// Scalar core of `sdiagram_mat_integral` over plain slices (grid
/// units), so the real-time `vt (3)` block can substitute the
/// `lam2^vt(., n)` and `gc^vt(., n)` arrays.
pub(crate) fn sigma_mat_core(
    m: usize,
    ga: &[Complex],
    gb: &[Complex],
    gc: &[Complex],
    lam1: &[Complex],
    lam2: &[Complex],
    mmax: usize,
    integ: &Integrator,
    ops: Complex,
) -> Complex {
    let k = integ.k as usize;
    if m == 0 {
        return Complex::ZERO;
    }
    // f_bc spans the full axis: the gregory branch fills l = 0..=j
    // (j up to m), not just the warmup nodes
    let mut f_bc = vec![Complex::ZERO; mmax + 1];
    let mut f_ac = vec![Complex::ZERO; mmax + 1];
    if m < k {
        let dtau1 = m as f64 / k as f64;
        for j in 1..=k {
            let tau1 = dtau1 * j as f64;
            let dtau2 = tau1 / k as f64;
            for l in 0..=k {
                let tau2 = l as f64 * dtau2;
                let lam1_i = poly_interpol_scalar(lam1, m as f64 - tau2, integ);
                let gb_i = poly_interpol_scalar(gb, tau1 - tau2, integ);
                let gc_i = poly_interpol_scalar(gc, tau2, integ);
                f_bc[l] = lam1_i * (gb_i * gc_i);
            }
            let mut integr_bc = gregory_scalar(k, &f_bc, integ);
            integr_bc *= Complex::new(dtau2, 0.0);
            let lam2_i = poly_interpol_scalar(lam2, tau1, integ);
            let ga_i = poly_interpol_scalar(ga, m as f64 - tau1, integ);
            f_ac[j] = lam2_i * (ga_i * integr_bc);
        }
        let integr_ac = gregory_scalar(k, &f_ac, integ);
        integr_ac * Complex::new(dtau1, 0.0) * ops
    } else {
        let mut tmp_bc;
        for j in 1..=m {
            if j < k {
                let dtau2 = j as f64 / k as f64;
                let tau1 = j as f64;
                for l in 0..=k {
                    let tau2 = l as f64 * dtau2;
                    let y = m as f64 - tau2;
                    let mut i1 = (y.floor() as i64 - k as i64 / 2).max(0) as usize;
                    if i1 > mmax - k {
                        i1 = mmax - k;
                    }
                    let lam1_i = poly_interpol_scalar(&lam1[i1..], y - i1 as f64, integ);
                    let gb_i = poly_interpol_scalar(gb, tau1 - tau2, integ);
                    let gc_i = poly_interpol_scalar(gc, tau2, integ);
                    f_bc[l] = lam1_i * (gb_i * gc_i);
                }
                tmp_bc = gregory_scalar(k, &f_bc, integ);
                tmp_bc *= Complex::new(dtau2, 0.0);
            } else {
                for l in 0..=j {
                    f_bc[l] = lam1[m - l] * (gb[j - l] * gc[l]);
                }
                tmp_bc = gregory_scalar(j, &f_bc, integ);
            }
            f_ac[j] = lam2[j] * (ga[m - j] * tmp_bc);
        }
        gregory_scalar(m, &f_ac, integ) * ops
    }
}

/// Scalar polynomial interpolation at fractional grid point `x`
/// (blueprint `poly_interpol`, nodes `0..=kt` of the slice).
fn poly_interpol_scalar(f: &[Complex], x: f64, integ: &Integrator) -> Complex {
    let k = integ.k as usize;
    let mut sum = Complex::ZERO;
    let mut xp = 1.0;
    for l in 0..=k {
        let mut fl = Complex::ZERO;
        for i in 0..=k {
            let w = integ.piw.at(l, i);
            if w != 0.0 {
                fl += Complex::new(w, 0.0) * f[i];
            }
        }
        sum += fl * Complex::new(xp, 0.0);
        xp *= x;
    }
    sum
}

/// Scalar core of `gdiagram_mat_integral` with the equilibrium flag:
/// `gd_used(tau) = -sig_gd*gd(mmax-tau)` in equilibrium, direct in
/// real time (the `vt (2)` block feeds `gd^tv(n, .)` directly).
#[allow(clippy::too_many_arguments)]
fn gf_mat_core(
    m: usize,
    ga: &[Complex],
    gb: &[Complex],
    gc: &[Complex],
    gd: &[Complex],
    lam: &[Complex],
    mmax: usize,
    integ: &Integrator,
    equilibrium: bool,
    sig_gd: f64,
    ops: Complex,
) -> Complex {
    let k = integ.k as usize;
    let gd_at = |tau: usize| -> Complex {
        if equilibrium {
            -Complex::new(sig_gd, 0.0) * gd[mmax - tau]
        } else {
            gd[tau]
        }
    };
    let lam_at = |x: f64| -> Complex {
        let mut l = (x.floor() as i64 - k as i64 / 2).max(0) as usize;
        if l > mmax - k {
            l = mmax - k;
        }
        poly_interpol_scalar(&lam[l..], x - l as f64, integ)
    };
    if m > 0 && m < k {
        // low-side warmup
        let dtau1 = m as f64 / k as f64;
        let mut f_da = vec![Complex::ZERO; k + 1];
        for i1 in 0..=k {
            let tau1 = i1 as f64 * dtau1;
            let mut f_dc = vec![Complex::ZERO; mmax - m + 1];
            for i2 in m..=mmax {
                let y = mmax as f64 + tau1 - i2 as f64;
                let lam_i = lam_at(y);
                f_dc[i2 - m] = lam_i * (gd_at(i2) * gc[i2 - m]);
            }
            let integr_dc = gregory_scalar(mmax - m, &f_dc, integ);
            let ga_i = poly_interpol_scalar(ga, tau1, integ);
            let gb_i = poly_interpol_scalar(gb, m as f64 - tau1, integ);
            f_da[i1] = integr_dc * gb_i * ga_i;
        }
        let integr_da = gregory_scalar(k, &f_da, integ);
        -integr_da * Complex::new(dtau1, 0.0) * ops
    } else if m > mmax - k && m < mmax {
        // high-side warmup
        let dtau2 = (mmax - m) as f64 / k as f64;
        let mut f_da = vec![Complex::ZERO; k + 1];
        for i2 in 0..=k {
            let tau2 = m as f64 + i2 as f64 * dtau2;
            let mut f_ba = vec![Complex::ZERO; m + 1];
            for i1 in 0..=m {
                let y = mmax as f64 + i1 as f64 - tau2;
                let lam_i = lam_at(y);
                f_ba[i1] = lam_i * (gb[m - i1] * ga[i1]);
            }
            let integr_ba = gregory_scalar(m, &f_ba, integ);
            let gc_i = poly_interpol_scalar(gc, tau2 - m as f64, integ);
            let y = tau2;
            let mut l = (y.floor() as i64 - k as i64 / 2).max(0) as usize;
            if l > mmax - k {
                l = mmax - k;
            }
            // view over the (already oriented) gd array from index l
            let gd_i = poly_interpol_scalar(&gd[l..], y - l as f64, integ);
            f_da[i2] = gd_i * gc_i * integr_ba;
        }
        let integr_da = gregory_scalar(k, &f_da, integ);
        -integr_da * Complex::new(dtau2, 0.0) * ops
    } else if m >= k && m <= mmax - k {
        // gregory region
        let mut f_dc = vec![Complex::ZERO; mmax - m + 1];
        let mut f_da = vec![Complex::ZERO; m + 1];
        for i1 in 0..=m {
            for i2 in m..=mmax {
                f_dc[i2 - m] = lam[mmax - i2 + i1] * (gd_at(i2) * gc[i2 - m]);
            }
            let integr_dc = gregory_scalar(mmax - m, &f_dc, integ);
            f_da[i1] = integr_dc * gb[m - i1] * ga[i1];
        }
        let integr_da = gregory_scalar(m, &f_da, integ);
        -integr_da * ops
    } else {
        Complex::ZERO
    }
}

/// OCA self-energy real-time contribution at timestep `n`, ADDED
/// onto the NCA values already present in `pp_sigma` (blueprint
/// `sdiagram_tstp`: ret(1), vt(1..3), les(1..6), all signs and index
// ------------------------------------------------------------------
// matrix-valued real-time OCA kernels (multi-band sectors)
// ------------------------------------------------------------------

/// Element count of a sector block of `pp` (dim x dim).
#[inline]
fn nel_of(pp: &PseudoGreen) -> usize {
    (pp.size1 * pp.size2) as usize
}

/// Matrix pseudo ret slot (holds greater); out-of-triangle reads fall
/// back to `-[ret(j,n)]^dag` block (adjoint of the stored block).
#[inline]
pub(crate) fn psc_blk_into(pp: &PseudoGreen, n: i64, j: i64, dim: usize, out: &mut [Complex]) {
    let nel = dim * dim;
    if j <= n {
        out[..nel].copy_from_slice(pp.ret.raw_block_at(n, j));
    } else {
        let src = pp.ret.raw_block_at(j, n);
        for r in 0..dim {
            for c in 0..dim {
                out[r * dim + c] = -src[c * dim + r].conj();
            }
        }
    }
}

/// Matrix pseudo lesser with the `-[less(n,j)]^dag` fallback.
#[inline]
pub(crate) fn pls_blk_into(pp: &PseudoGreen, j: i64, n: i64, dim: usize, out: &mut [Complex]) {
    let nel = dim * dim;
    if j <= n {
        out[..nel].copy_from_slice(pp.less.raw_block_at(j, n));
    } else {
        let src = pp.less.raw_block_at(n, j);
        for r in 0..dim {
            for c in 0..dim {
                out[r * dim + c] = -src[c * dim + r].conj();
            }
        }
    }
}

/// Matrix DIRECT pseudo tv block mirroring the scalar `ptv(pp, n, m)`:
/// a plain copy of `G.lmix(n, m)`.
#[inline]
pub(crate) fn ptv_dir_blk(pp: &PseudoGreen, n: i64, m: i64, dim: usize, out: &mut [Complex]) {
    let nel = dim * dim;
    out[..nel].copy_from_slice(pp.lmix.raw_block_at(n, m));
}

/// Matrix pseudo vt view mirroring the C++ `vt_from_tv_matrix_matrix_ref`:
/// `-sig * [G.lmix(t, ntau - tau)]^dag` -- the CONJUGATE TRANSPOSE
/// (adjoint) of the stored tv block, not an element-wise conjugate
/// (identical only at dim = 1).
#[inline]
pub(crate) fn ptv_blk_into(pp: &PseudoGreen, tau: i64, t: i64, dim: usize, out: &mut [Complex]) {
    let src = pp.lmix.raw_block_at(t, pp.ntau as i64 - tau);
    let sgn = -Complex::new(pp.sign.as_i64() as f64, 0.0);
    for r in 0..dim {
        for c in 0..dim {
            out[r * dim + c] = sgn * src[c * dim + r].conj();
        }
    }
}

/// Scalar hybridization views keep `(0,0)` element semantics: the
/// matrix kernels reuse the scalar `svt/sles/sgtr/stv/svt` accessors.

/// Gregory quadrature over a block-valued 1-D array (`nel` elements
/// per point, `n+1` points).
pub(crate) fn gregory_blk(n: usize, v: &[Complex], nel: usize, integ: &Integrator) -> Vec<Complex> {
    // the warmup weight rows (n < k) span all k+1 nodes: the column
    // must be zero-padded to max(n, k) + 1 (blueprint
    // gregory_sum on the full zero-padded array)
    let k = integ.k as usize;
    let ntop = n.max(k);
    let mut out = vec![Complex::ZERO; nel];
    for q in 0..nel {
        let col: Vec<Complex> = (0..=ntop)
            .map(|i| {
                // zero-pad only BEYOND the slice: the warmup rows
                // (n < k+1) read all k+1 nodes, so entries the caller
                // filled past n (up to n1 = max(n, kt)) must be read,
                // matching the scalar gregory over the full array
                if i * nel + q < v.len() {
                    v[i * nel + q]
                } else {
                    Complex::ZERO
                }
            })
            .collect();
        out[q] = gregory_scalar(n, &col, integ);
    }
    out
}

/// `conv2b` quadrature over a block-valued triangular grid (layout
/// `data[i1 * n11 * nel + i2 * nel + q]`).
pub(crate) fn conv2b_blk(
    data: &[Complex],
    n11: usize,
    nel: usize,
    j: usize,
    nn: usize,
    integ: &Integrator,
) -> Vec<Complex> {
    let mut out = vec![Complex::ZERO; nel];
    for q in 0..nel {
        let col: Vec<Complex> = (0..n11 * n11).map(|ix| data[ix * nel + q]).collect();
        out[q] = conv2b_scalar(&col, n11, j, nn, integ);
    }
    out
}

/// Low-side (warmup) quadrature over block-valued arrays.
pub(crate) fn lowside_blk(
    j: usize,
    aa: &[Complex],
    bb: &[Complex],
    d1: usize,
    d2: usize,
    d3: usize,
    mmax: usize,
    integ: &Integrator,
    scratch: &mut Vec<Complex>,
) -> Vec<Complex> {
    // block-product contraction over the shared tau index: the two
    // factors carry `nel_a` and `nel_b` elements per point with the
    // product shape (d1 x d2) . (d2 x d3)
    let nel_a = d1 * d2;
    let nel_b = d2 * d3;
    let nelo = d1 * d3;
    let k = integ.k as usize;
    let mut out = vec![Complex::ZERO; nelo];
    let mut prod = vec![Complex::ZERO; nelo];
    let smax = nel_a.max(nel_b).max(nelo);
    if scratch.len() < smax {
        scratch.resize(smax, Complex::ZERO);
    }
    let mat2 = |a: &[Complex], b: &[Complex], o: &mut [Complex]| {
        for r in 0..d1 {
            for c in 0..d3 {
                let mut acc = Complex::ZERO;
                for q in 0..d2 {
                    acc += a[r * d2 + q] * b[q * d3 + c];
                }
                o[r * d3 + c] = acc;
            }
        }
    };
    if j > 0 && j < k {
        for l in 0..=k {
            for m in 0..=k {
                let w = integ.bcw.at(j, l, m);
                if w != 0.0 {
                    mat2(&aa[l * nel_a..], &bb[m * nel_b..], &mut prod);
                    for q in 0..nelo {
                        out[q] += Complex::new(w, 0.0) * prod[q];
                    }
                }
            }
        }
    } else if j > 0 {
        let mut f1 = vec![Complex::ZERO; (j + 1) * nelo];
        for l in 0..=j {
            mat2(&aa[(j - l) * nel_a..], &bb[l * nel_b..], &mut prod);
            f1[l * nelo..(l + 1) * nelo].copy_from_slice(&prod);
        }
        out = gregory_blk(j, &f1, nelo, integ);
    }
    out
}

/// Matrix version of [`add_sigma_tstp_oca`] for multi-band sectors
/// (blueprint `sdiagram_tstp` with matrix types; the hybridization
/// insertions stay scalar `(0,0)` elements). Adds onto `pp_sigma`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_sigma_tstp_oca_matrix(
    n: i64,
    d: &OcaSigmaDiagram,
    pp_g: &[PseudoGreen],
    beta: f64,
    h: f64,
    kt: i64,
    integ: &Integrator,
    pp_sigma: &mut [PseudoGreen],
) {
    let mmax = pp_sigma[d.out].ntau as i64;
    let dtau = beta / mmax as f64;
    let n1 = n.max(kt);
    let n11 = (n1 + 1) as usize;
    let ga = &pp_g[d.sa];
    let gb = &pp_g[d.sb];
    let gc = &pp_g[d.sc];
    let lam1 = d.lam1;
    let lam2 = d.lam2;
    let (adim, bdim, cdim) = (ga.size1 as usize, gb.size1 as usize, gc.size1 as usize);
    let (na, nb, nc) = (adim * adim, bdim * bdim, cdim * cdim);
    let nbc = bdim * cdim;
    let pref = d.prefactor;
    let sig = d.sig;
    let hh = Complex::new(h * h, 0.0);
    let out = &mut pp_sigma[d.out];
    let sdim = out.size1 as usize;
    let nss = sdim * sdim;
    let nab = adim * bdim;
    let nac = adim * cdim;
    let smax = na.max(nb).max(nc).max(nab).max(nbc).max(nac).max(nss);
    let mut scratch = vec![Complex::ZERO; smax];
    // reusable block buffers
    let mut gab = vec![Complex::ZERO; na];
    let mut gbb = vec![Complex::ZERO; nb];
    let mut gcb = vec![Complex::ZERO; nc];
    let mut t_ab = vec![Complex::ZERO; adim * bdim];
    let mut t_bc = vec![Complex::ZERO; bdim * cdim];
    let mut t_ac = vec![Complex::ZERO; adim * cdim];
    let mut prod = vec![Complex::ZERO; adim * cdim];
    let mut op1_conv = vec![Complex::ZERO; nss];

    // -- data_ab (ret 1): grid of (adim x bdim)-shaped products
    let mut data_ab = vec![Complex::ZERO; n11 * n11 * nab];
    for i1 in 0..=n1 {
        let i2max = (i1 + kt).min(n1);
        psc_blk_into(ga, n, i1, adim, &mut gab);
        for i2 in 0..=i2max {
            psc_blk_into(gb, i1, i2, bdim, &mut gbb);
            sandwich(
                &gab,
                d.op2.raw(),
                &gbb,
                adim,
                adim,
                bdim,
                bdim,
                &mut t_ab,
                &mut scratch,
            );
            let s = sgtr(lam1, n, i2);
            for q in 0..nab {
                data_ab[(i1 as usize) * n11 * nab + (i2 as usize) * nab + q] = s * t_ab[q];
            }
        }
    }
    // -- ret (1)
    for j in (0..=n).rev() {
        let i1min = j.min(n1 - kt);
        let mut data_ac = vec![Complex::ZERO; n11 * n11 * nac];
        for i1 in i1min..=n1 {
            let i2max = (i1 + kt).min(n1);
            let l2 = sgtr(lam2, i1, j);
            for i2 in i1min..=i2max {
                psc_blk_into(gc, i2, j, cdim, &mut gcb);
                // data_ab[i1][i2] (adim x bdim) * op3 * gc(i2, j)
                sandwich(
                    &data_ab[(i1 as usize) * n11 * nab + (i2 as usize) * nab..],
                    d.op3.raw(),
                    &gcb,
                    adim,
                    bdim,
                    cdim,
                    cdim,
                    &mut t_ac,
                    &mut scratch,
                );
                let base = ((i1 - i1min) as usize) * n11 * nac + ((i2 - i1min) as usize) * nac;
                for q in 0..nac {
                    data_ac[base + q] = l2 * t_ac[q];
                }
            }
        }
        let conv = conv2b_blk(
            &data_ac,
            n11,
            nac,
            (j - i1min) as usize,
            (n - i1min) as usize,
            integ,
        );
        sandwich(
            d.op1.raw(),
            &conv,
            d.op4.raw(),
            sdim,
            adim,
            cdim,
            sdim,
            &mut op1_conv,
            &mut scratch,
        );
        let blk = out.ret.block_mut(out.ret.idx(n, j));
        for q in 0..blk.len() {
            blk[q] += pref * (-hh) * op1_conv[q];
        }
    }
    // -- data_bc (vt 1 / les 1): (bdim x cdim)-shaped products
    let mut data_bc = vec![Complex::ZERO; n11 * n11 * nbc];
    for t1 in 0..=n1 {
        let t2max = (t1 + kt).min(n1);
        psc_blk_into(gc, t1, n, cdim, &mut gcb);
        for t2 in 0..=t2max {
            psc_blk_into(gb, t2, t1, bdim, &mut gbb);
            sandwich(
                &gbb,
                d.op3.raw(),
                &gcb,
                bdim,
                bdim,
                cdim,
                cdim,
                &mut t_bc,
                &mut scratch,
            );
            let s = sgtr(lam2, t2, n);
            for q in 0..nbc {
                data_bc[(t1 as usize) * n11 * nbc + (t2 as usize) * nbc + q] = s * t_bc[q];
            }
        }
    }
    // reusable buffers: the vt(1)/les(1) (i2, i1) grid and the point
    // arrays used by the other blocks
    let mut data_ac_grid = vec![Complex::ZERO; n11 * n11 * nac];
    let mut f_ac = vec![Complex::ZERO; (n11) * nac];
    let mut gab_vt = vec![Complex::ZERO; na];
    // -- vt (1)
    for m in 0..=mmax {
        for i2 in 0..=n1 {
            let i1max = (i2 + kt).min(n1);
            let lam1_vt = svt(lam1, m, i2);
            for i1 in 0..=i1max {
                let mut pv = vec![Complex::ZERO; na];
                ptv_blk_into(ga, m, i1, adim, &mut pv);
                sandwich(
                    &pv,
                    d.op2.raw(),
                    &data_bc[(i2 as usize) * n11 * nbc + (i1 as usize) * nbc..],
                    adim,
                    adim,
                    bdim,
                    cdim,
                    &mut t_ac,
                    &mut scratch,
                );
                let base = (i2 as usize) * n11 * nac + (i1 as usize) * nac;
                for q in 0..nac {
                    data_ac_grid[base + q] = lam1_vt * t_ac[q];
                }
            }
        }
        let conv = conv2b_blk(&data_ac_grid, n11, nac, 0, n as usize, integ);
        sandwich(
            d.op1.raw(),
            &conv,
            d.op4.raw(),
            sdim,
            adim,
            cdim,
            sdim,
            &mut op1_conv,
            &mut scratch,
        );
        let val = pref * Complex::new(sig * h * h, 0.0);
        let blk = out.lmix.block_mut(out.lmix.idx(n, mmax - m));
        let sdim_l = blk.len();
        let dim_l = (sdim_l as f64).sqrt() as usize;
        for r in 0..dim_l {
            for c in 0..dim_l {
                blk[r * dim_l + c] += val * op1_conv[c * dim_l + r].conj();
            }
        }
    }
    // -- les (1)
    for j in 0..=n {
        for i2 in 0..=n1 {
            let i1max = (i2 + kt).min(n1);
            let lam1_les = sles(lam1, j, i2);
            for i1 in 0..=i1max {
                pls_blk_into(ga, j, i1, adim, &mut gab_vt);
                sandwich(
                    &gab_vt,
                    d.op2.raw(),
                    &data_bc[(i2 as usize) * n11 * nbc + (i1 as usize) * nbc..],
                    adim,
                    adim,
                    bdim,
                    cdim,
                    &mut t_ac,
                    &mut scratch,
                );
                let base = (i2 as usize) * n11 * nac + (i1 as usize) * nac;
                for q in 0..nac {
                    data_ac_grid[base + q] = lam1_les * t_ac[q];
                }
            }
        }
        let conv = conv2b_blk(&data_ac_grid, n11, nac, 0, n as usize, integ);
        sandwich(
            d.op1.raw(),
            &conv,
            d.op4.raw(),
            sdim,
            adim,
            cdim,
            sdim,
            &mut op1_conv,
            &mut scratch,
        );
        let blk = out.less.block_mut(out.less.idx(j, n));
        for q in 0..blk.len() {
            blk[q] += pref * (-hh) * op1_conv[q];
        }
    }

    // -- vt (2) / les (2): tau-grid data2 = op2 * gb_tv  (adim x bdim)
    let mnp = (mmax + 1) as usize;
    let nab = adim * bdim;
    let mut data2 = vec![Complex::ZERO; n11 * mnp * nab];
    for tau in 0..=mmax {
        let l2 = svt(lam2, tau, n);
        for t in 0..=n1 {
            let mut pv = vec![Complex::ZERO; nb];
            ptv_blk_into(gb, tau, t, bdim, &mut pv); // (bdim x bdim)
            let base = (t as usize) * mnp * nab + (tau as usize) * nab;
            for r in 0..adim {
                for c in 0..bdim {
                    let mut acc = Complex::ZERO;
                    for q in 0..bdim {
                        acc += d.op2.get(r, q) * pv[q * bdim + c];
                    }
                    data2[base + r * bdim + c] = l2 * acc;
                }
            }
        }
    }
    // -- vt (2)
    for m in 0..=mmax {
        for i2 in 0..=n1 {
            let row = &data2[(i2 as usize) * mnp * nab..];
            let conv_ab = lowside_blk(
                m as usize,
                ga.mat.raw(),
                row,
                adim,
                adim,
                bdim,
                mmax as usize,
                integ,
                &mut scratch,
            );
            // conv_ab is (adim x bdim)-shaped? ga.mat blocks are (adim x adim);
            // the lowside contracts over tau: result shape follows the product
            // ga.mat(mu) x data2(mu) => (adim x bdim)
            psc_blk_into(gc, i2, n, cdim, &mut gcb);
            sandwich(
                &conv_ab,
                d.op3.raw(),
                &gcb,
                adim,
                bdim,
                cdim,
                cdim,
                &mut t_ac,
                &mut scratch,
            );
            let s = svt(lam1, m, i2);
            let base = (i2 as usize) * nac;
            for q in 0..nac {
                f_ac[base + q] = s * t_ac[q];
            }
        }
        let integr = gregory_blk(n as usize, &f_ac[..(n1 as usize + 1) * nac], nac, integ);
        sandwich(
            d.op1.raw(),
            &integr,
            d.op4.raw(),
            sdim,
            adim,
            cdim,
            sdim,
            &mut op1_conv,
            &mut scratch,
        );
        let val = pref * Complex::new(-sig * dtau * h, 0.0);
        let blk = out.lmix.block_mut(out.lmix.idx(n, mmax - m));
        let sdim_l = blk.len();
        let dim_l = (sdim_l as f64).sqrt() as usize;
        for r in 0..dim_l {
            for c in 0..dim_l {
                blk[r * dim_l + c] += val * op1_conv[c * dim_l + r].conj();
            }
        }
    }
    // -- les (2)
    for j in 0..=n {
        for i2 in 0..=n1 {
            let row = &data2[(i2 as usize) * mnp * nab..];
            // f_ab[mu] = ptv(ga, j, mu) * row[mu]: (adim x adim) x (adim x bdim)
            let mut f_ab = vec![Complex::ZERO; mnp * nab];
            for mu in 0..=mmax {
                let mut pv = vec![Complex::ZERO; na];
                ptv_dir_blk(ga, j, mu, adim, &mut pv);
                let rb = &row[(mu as usize) * nab..];
                for r in 0..adim {
                    for c in 0..bdim {
                        let mut acc = Complex::ZERO;
                        for q in 0..adim {
                            acc += pv[r * adim + q] * rb[q * bdim + c];
                        }
                        f_ab[(mu as usize) * nab + r * bdim + c] = acc;
                    }
                }
            }
            let integr_ab = gregory_blk(mmax as usize, &f_ab, nab, integ);
            psc_blk_into(gc, i2, n, cdim, &mut gcb);
            sandwich(
                &integr_ab,
                d.op3.raw(),
                &gcb,
                adim,
                bdim,
                cdim,
                cdim,
                &mut t_ac,
                &mut scratch,
            );
            let s = sles(lam1, j, i2);
            let base = (i2 as usize) * nac;
            for q in 0..nac {
                f_ac[base + q] = s * t_ac[q];
            }
        }
        let integr = gregory_blk(n as usize, &f_ac[..(n1 as usize + 1) * nac], nac, integ);
        sandwich(
            d.op1.raw(),
            &integr,
            d.op4.raw(),
            sdim,
            adim,
            cdim,
            sdim,
            &mut op1_conv,
            &mut scratch,
        );
        let val = pref * Complex::new(0.0, -dtau * h);
        let blk = out.less.block_mut(out.less.idx(j, n));
        for q in 0..blk.len() {
            blk[q] += val * op1_conv[q];
        }
    }
    // -- vt (3): the mat-core sandwich chain with vt-sliced gc/lam2 rows
    {
        let mut gc_vt = vec![Complex::ZERO; mnp * nc];
        let mut lam2_vt = vec![Complex::ZERO; mnp];
        for tau in 0..=mmax {
            lam2_vt[tau as usize] = svt(lam2, tau, n);
            let mut pv = vec![Complex::ZERO; nc];
            ptv_blk_into(gc, tau, n, cdim, &mut pv);
            gc_vt[(tau as usize) * nc..(tau as usize + 1) * nc].copy_from_slice(&pv);
        }
        // per-m evaluation of the double Matsubara integral (low/mid/high
        // branches, blueprint sigma_mat_core with matrix blocks)
        for m in 0..=mmax {
            let val = sigma_mat_core_blk(
                m as usize,
                kt as usize,
                mmax as usize,
                ga.mat.raw(),
                gb.mat.raw(),
                &gc_vt,
                lam1.mat.raw(),
                &lam2_vt,
                adim,
                bdim,
                cdim,
                out.size1 as usize,
                d,
                integ,
                &mut scratch,
            );
            let pref_v = pref * Complex::new(0.0, -sig * dtau * dtau);
            let blk = out.lmix.block_mut(out.lmix.idx(n, mmax - m));
            let sdim_l = blk.len();
            let dim_l = (sdim_l as f64).sqrt() as usize;
            for r in 0..dim_l {
                for c in 0..dim_l {
                    blk[r * dim_l + c] += pref_v * val[c * dim_l + r].conj();
                }
            }
        }
    }
    // -- les (3)
    for j in (0..=n).rev() {
        let i1max = j.max(kt);
        for i1 in 0..=i1max {
            let mut f_bc = vec![Complex::ZERO; n11 * nbc];
            for i2 in 0..=n1 {
                pls_blk_into(gb, i1, i2, bdim, &mut gbb);
                psc_blk_into(gc, i2, n, cdim, &mut gcb);
                sandwich(
                    &gbb,
                    d.op3.raw(),
                    &gcb,
                    bdim,
                    bdim,
                    cdim,
                    cdim,
                    &mut t_bc,
                    &mut scratch,
                );
                let s = sles(lam1, j, i2);
                for q in 0..nbc {
                    f_bc[(i2 as usize) * nbc + q] = s * t_bc[q];
                }
            }
            let integr_bc = gregory_blk(n as usize, &f_bc, nbc, integ);
            psc_blk_into(ga, j, i1, adim, &mut gab);
            sandwich(
                &gab,
                d.op2.raw(),
                &integr_bc,
                adim,
                adim,
                bdim,
                cdim,
                &mut t_ac,
                &mut scratch,
            );
            let s = sles(lam2, i1, n);
            for q in 0..nac {
                f_ac[(i1 as usize) * nac + q] = s * t_ac[q];
            }
        }
        let integr = gregory_blk(
            j as usize,
            &f_ac[..((i1max + 1) as usize) * nac],
            nac,
            integ,
        );
        sandwich(
            d.op1.raw(),
            &integr,
            d.op4.raw(),
            sdim,
            adim,
            cdim,
            sdim,
            &mut op1_conv,
            &mut scratch,
        );
        let blk = out.less.block_mut(out.less.idx(j, n));
        for q in 0..blk.len() {
            blk[q] += pref * hh * op1_conv[q];
        }
    }
    // -- les (4)
    for j in 0..=n {
        let mut f_bc = vec![Complex::ZERO; mnp * nbc];
        for mu in 0..=mmax {
            let mut pv = vec![Complex::ZERO; nc];
            ptv_blk_into(gc, mu, n, cdim, &mut pv);
            // op3 * pv: (bdim x cdim) x (cdim x cdim)
            let rb = &mut f_bc[(mu as usize) * nbc..];
            for r in 0..bdim {
                for c in 0..cdim {
                    let mut acc = Complex::ZERO;
                    for q in 0..cdim {
                        acc += d.op3.get(r, q) * pv[q * cdim + c];
                    }
                    rb[r * cdim + c] = stv(lam1, j, mu) * acc;
                }
            }
        }
        let mut f_ac_m = vec![Complex::ZERO; mnp * nac];
        for mu in 0..=mmax {
            let conv_bc = lowside_blk(
                mu as usize,
                gb.mat.raw(),
                &f_bc,
                bdim,
                bdim,
                cdim,
                mmax as usize,
                integ,
                &mut scratch,
            );
            let mut pv = vec![Complex::ZERO; na];
            ptv_dir_blk(ga, j, mu, adim, &mut pv);
            sandwich(
                &pv,
                d.op2.raw(),
                &conv_bc,
                adim,
                adim,
                bdim,
                cdim,
                &mut t_ac,
                &mut scratch,
            );
            let s = svt(lam2, mu, n);
            for q in 0..nac {
                f_ac_m[(mu as usize) * nac + q] = s * t_ac[q];
            }
        }
        let integr = gregory_blk(mmax as usize, &f_ac_m, nac, integ);
        sandwich(
            d.op1.raw(),
            &integr,
            d.op4.raw(),
            sdim,
            adim,
            cdim,
            sdim,
            &mut op1_conv,
            &mut scratch,
        );
        let val = pref * Complex::new(0.0, dtau * dtau);
        let blk = out.less.block_mut(out.less.idx(j, n));
        for q in 0..blk.len() {
            blk[q] += val * op1_conv[q];
        }
    }
    // -- les (5)
    {
        let mut data_bc5 = vec![Complex::ZERO; n11 * n11 * nbc];
        for i1 in 0..=n1 {
            let i2max = (i1 + kt).min(n1);
            let l2 = sles(lam2, i1, n);
            for i2 in 0..=i2max {
                psc_blk_into(gb, i1, i2, bdim, &mut gbb);
                pls_blk_into(gc, i2, n, cdim, &mut gcb);
                sandwich(
                    &gbb,
                    d.op3.raw(),
                    &gcb,
                    bdim,
                    bdim,
                    cdim,
                    cdim,
                    &mut t_bc,
                    &mut scratch,
                );
                for q in 0..nbc {
                    data_bc5[(i1 as usize) * n11 * nbc + (i2 as usize) * nbc + q] = l2 * t_bc[q];
                }
            }
        }
        for j in (0..=n).rev() {
            let i1max = j.max(kt);
            let mut data_ac5 = vec![Complex::ZERO; n11 * n11 * nac];
            for i1 in 0..=i1max {
                let i2max = (i1 + kt).min(i1max);
                psc_blk_into(ga, j, i1, adim, &mut gab);
                for i2 in 0..=i2max {
                    sandwich(
                        &gab,
                        d.op2.raw(),
                        &data_bc5[(i1 as usize) * n11 * nbc + (i2 as usize) * nbc..],
                        adim,
                        adim,
                        bdim,
                        cdim,
                        &mut t_ac,
                        &mut scratch,
                    );
                    let s = sgtr(lam1, j, i2);
                    for q in 0..nac {
                        data_ac5[(i1 as usize) * n11 * nac + (i2 as usize) * nac + q] = s * t_ac[q];
                    }
                }
            }
            let conv = conv2b_blk(&data_ac5, n11, nac, 0, j as usize, integ);
            sandwich(
                d.op1.raw(),
                &conv,
                d.op4.raw(),
                sdim,
                adim,
                cdim,
                sdim,
                &mut op1_conv,
                &mut scratch,
            );
            let blk = out.less.block_mut(out.less.idx(j, n));
            for q in 0..blk.len() {
                blk[q] += pref * (-hh) * op1_conv[q];
            }
        }
    }
    // -- les (6)
    for j in 0..=n {
        let i1max = j.max(kt);
        for i1 in 0..=i1max {
            let mut f_bc = vec![Complex::ZERO; mnp * nbc];
            for mu in 0..=mmax {
                let mut pv_b = vec![Complex::ZERO; nb];
                ptv_dir_blk(gb, i1, mu, bdim, &mut pv_b);
                let mut pv_c = vec![Complex::ZERO; nc];
                ptv_blk_into(gc, mu, n, cdim, &mut pv_c);
                sandwich(
                    &pv_b,
                    d.op3.raw(),
                    &pv_c,
                    bdim,
                    bdim,
                    cdim,
                    cdim,
                    &mut t_bc,
                    &mut scratch,
                );
                let s = stv(lam1, j, mu);
                for q in 0..nbc {
                    f_bc[(mu as usize) * nbc + q] = s * t_bc[q];
                }
            }
            let integr_bc = gregory_blk(mmax as usize, &f_bc, nbc, integ);
            psc_blk_into(ga, j, i1, adim, &mut gab);
            sandwich(
                &gab,
                d.op2.raw(),
                &integr_bc,
                adim,
                adim,
                bdim,
                cdim,
                &mut t_ac,
                &mut scratch,
            );
            let s = sles(lam2, i1, n);
            for q in 0..nac {
                f_ac[(i1 as usize) * nac + q] = s * t_ac[q];
            }
        }
        let integr = gregory_blk(
            j as usize,
            &f_ac[..((i1max + 1) as usize) * nac],
            nac,
            integ,
        );
        sandwich(
            d.op1.raw(),
            &integr,
            d.op4.raw(),
            sdim,
            adim,
            cdim,
            sdim,
            &mut op1_conv,
            &mut scratch,
        );
        let val = pref * Complex::new(0.0, dtau * h);
        let blk = out.less.block_mut(out.less.idx(j, n));
        for q in 0..blk.len() {
            blk[q] += val * op1_conv[q];
        }
    }
}

/// Matrix version of the scalar `sigma_mat_core` for the vt(3) block:
/// evaluates the double Matsubara integral with the sandwich chain
/// `lam1(mu) * (ga(mu1) * op2 * gb * op3 * gc_vt * op4)` on the
/// sub-grid/gregory startup scheme (blueprint `sdiagram_mat_integral`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sigma_mat_core_blk(
    m: usize,
    k: usize,
    mmax: usize,
    ga_raw: &[Complex],
    gb_raw: &[Complex],
    gc_vt: &[Complex],
    lam1_raw: &[Complex],
    lam2_vt: &[Complex],
    adim: usize,
    bdim: usize,
    cdim: usize,
    sdim: usize,
    d: &OcaSigmaDiagram,
    integ: &Integrator,
    scratch: &mut Vec<Complex>,
) -> Vec<Complex> {
    // matrix version of `sigma_mat_core` (blueprint: the low/mid/high
    // branches of the scalar core with sandwich products; the vt(3)
    // block feeds the tv-sliced `gc_vt`/`lam2_vt` rows directly)
    let nb_c = bdim * cdim;
    let na_c = adim * cdim;
    let ns = sdim * sdim;
    let mut out = vec![Complex::ZERO; ns];
    if m == 0 {
        return out;
    }
    let ga_axis = MatAxis {
        data: ga_raw,
        d1: adim,
        d2: adim,
    };
    let gb_axis = MatAxis {
        data: gb_raw,
        d1: bdim,
        d2: bdim,
    };
    let gc_axis = MatAxis {
        data: gc_vt,
        d1: cdim,
        d2: cdim,
    };
    let smax = (adim * adim)
        .max(bdim * bdim)
        .max(cdim * cdim)
        .max(nb_c)
        .max(na_c)
        .max(ns);
    if scratch.len() < smax {
        scratch.resize(smax, Complex::ZERO);
    }
    let mut gb_i = vec![Complex::ZERO; bdim * bdim];
    let mut gc_i = vec![Complex::ZERO; cdim * cdim];
    let mut ga_i = vec![Complex::ZERO; adim * adim];
    let mut f_bc = vec![vec![Complex::ZERO; nb_c]; mmax + 1];
    let mut f_ac = vec![vec![Complex::ZERO; na_c]; mmax + 1];
    let mut tmp_bc = vec![Complex::ZERO; nb_c];
    if m < k {
        let dtau1 = m as f64 / k as f64;
        for j in 1..=k {
            let tau1 = dtau1 * j as f64;
            let dtau2 = tau1 / k as f64;
            for l in 0..=k {
                let tau2 = l as f64 * dtau2;
                let lam1_i = poly_interpol_scalar(lam1_raw, m as f64 - tau2, integ);
                poly_interpol(&gb_axis, tau1 - tau2, integ, &mut gb_i, scratch);
                poly_interpol(&gc_axis, tau2, integ, &mut gc_i, scratch);
                sandwich(
                    &gb_i,
                    d.op3.raw(),
                    &gc_i,
                    bdim,
                    bdim,
                    cdim,
                    cdim,
                    &mut f_bc[l],
                    scratch,
                );
                for q in 0..nb_c {
                    f_bc[l][q] *= lam1_i;
                }
            }
            // scalar core multiplies the inner warmup quadrature by
            // dtau2 (grid refinement j/k); the outer gregory rows
            // alone would miss it
            let mut integr_bc = gregory_blk(k, &flatten(&f_bc[..=k]), nb_c, integ);
            for q in 0..nb_c {
                integr_bc[q] *= Complex::new(dtau2, 0.0);
            }
            let lam2_i = poly_interpol_scalar(lam2_vt, tau1, integ);
            poly_interpol(&ga_axis, m as f64 - tau1, integ, &mut ga_i, scratch);
            sandwich(
                &ga_i,
                d.op2.raw(),
                &integr_bc,
                adim,
                adim,
                bdim,
                cdim,
                &mut f_ac[j],
                scratch,
            );
            for q in 0..na_c {
                f_ac[j][q] *= lam2_i;
            }
        }
        let integr_ac = gregory_blk(k, &flatten(&f_ac[..=k]), na_c, integ);
        sandwich(
            d.op1.raw(),
            &integr_ac,
            d.op4.raw(),
            sdim,
            adim,
            cdim,
            sdim,
            &mut out,
            scratch,
        );
        for q in 0..ns {
            out[q] *= Complex::new(dtau1, 0.0);
        }
        out
    } else {
        for j in 1..=m {
            if j < k {
                let dtau2 = j as f64 / k as f64;
                let tau1 = j as f64;
                for l in 0..=k {
                    let tau2 = l as f64 * dtau2;
                    let y = m as f64 - tau2;
                    let mut i1 = (y.floor() as i64 - k as i64 / 2).max(0) as usize;
                    if i1 > mmax - k {
                        i1 = mmax - k;
                    }
                    let lam1_i = poly_interpol_scalar(&lam1_raw[i1..], y - i1 as f64, integ);
                    poly_interpol(&gb_axis, tau1 - tau2, integ, &mut gb_i, scratch);
                    poly_interpol(&gc_axis, tau2, integ, &mut gc_i, scratch);
                    sandwich(
                        &gb_i,
                        d.op3.raw(),
                        &gc_i,
                        bdim,
                        bdim,
                        cdim,
                        cdim,
                        &mut f_bc[l],
                        scratch,
                    );
                    for q in 0..nb_c {
                        f_bc[l][q] *= lam1_i;
                    }
                }
                let integr_bc = gregory_blk(k, &flatten(&f_bc[..=k]), nb_c, integ);
                for q in 0..nb_c {
                    tmp_bc[q] = Complex::new(dtau2, 0.0) * integr_bc[q];
                }
            } else {
                for l in 0..=j {
                    let lam1_i = lam1_raw[m - l];
                    let gb_blk = gb_axis.block(j - l);
                    let gc_blk = gc_axis.block(l);
                    sandwich(
                        gb_blk,
                        d.op3.raw(),
                        gc_blk,
                        bdim,
                        bdim,
                        cdim,
                        cdim,
                        &mut f_bc[l],
                        scratch,
                    );
                    for q in 0..nb_c {
                        f_bc[l][q] *= lam1_i;
                    }
                }
                let integr_bc = gregory_blk(j, &flatten(&f_bc[..=j]), nb_c, integ);
                tmp_bc.copy_from_slice(&integr_bc);
            }
            let lam2_i = lam2_vt[j];
            let ga_blk = ga_axis.block(m - j);
            sandwich(
                ga_blk,
                d.op2.raw(),
                &tmp_bc,
                adim,
                adim,
                bdim,
                cdim,
                &mut f_ac[j],
                scratch,
            );
            for q in 0..na_c {
                f_ac[j][q] *= lam2_i;
            }
        }
        let integr_ac = gregory_blk(m, &flatten(&f_ac[..=m]), na_c, integ);
        sandwich(
            d.op1.raw(),
            &integr_ac,
            d.op4.raw(),
            sdim,
            adim,
            cdim,
            sdim,
            &mut out,
            scratch,
        );
        out
    }
}

/// Flatten a slice of block vectors into a contiguous per-point
/// layout for the block quadrature helpers.
fn flatten(rows: &[Vec<Complex>]) -> Vec<Complex> {
    let nel = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut out = Vec::with_capacity(rows.len() * nel);
    for r in rows {
        out.extend_from_slice(r);
    }
    out
}

fn nbc_placeholder() -> usize {
    0
}

/// reversals verbatim; the diagram prefactor multiplies every term).
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_sigma_tstp_oca(
    n: i64,
    d: &OcaSigmaDiagram,
    pp_g: &[PseudoGreen],
    beta: f64,
    h: f64,
    kt: i64,
    integ: &Integrator,
    pp_sigma: &mut [PseudoGreen],
) {
    // multi-band sectors take the matrix path (blueprint sdiagram_tstp
    // with matrix types); the scalar fast path below stays P5-gated
    // for all-1x1 chains
    if pp_g[d.sa].size1 * pp_g[d.sb].size1 * pp_g[d.sc].size1 > 1 {
        add_sigma_tstp_oca_matrix(n, d, pp_g, beta, h, kt, integ, pp_sigma);
        return;
    }
    debug_assert_eq!(pp_sigma[d.out].size1 * pp_sigma[d.out].size2, 1);
    let mmax = pp_sigma[d.out].ntau as i64;
    let dtau = beta / mmax as f64;
    let n1 = n.max(kt);
    let n11 = (n1 + 1) as usize;
    let ga = &pp_g[d.sa];
    let gb = &pp_g[d.sb];
    let gc = &pp_g[d.sc];
    let lam1 = d.lam1;
    let lam2 = d.lam2;
    let (op1, op2, op3, op4) = (
        d.op1.get(0, 0),
        d.op2.get(0, 0),
        d.op3.get(0, 0),
        d.op4.get(0, 0),
    );
    let pref = d.prefactor;
    let sig = d.sig;
    let hh = Complex::new(h * h, 0.0);
    let out = &mut pp_sigma[d.out];

    // -- data_ab (ret 1)
    let mut data_ab = vec![Complex::ZERO; n11 * n11];
    for i1 in 0..=n1 {
        let i2max = (i1 + kt).min(n1);
        let ga_ret = psc(ga, n, i1);
        for i2 in 0..=i2max {
            data_ab[(i1 as usize) * n11 + i2 as usize] =
                sgtr(lam1, n, i2) * (ga_ret * op2 * psc(gb, i1, i2));
        }
    }
    // -- ret (1)
    let mut data_ac = vec![Complex::ZERO; n11 * n11];
    for j in (0..=n).rev() {
        let i1min = j.min(n1 - kt);
        for i1 in i1min..=n1 {
            let i2max = (i1 + kt).min(n1);
            let l2 = sgtr(lam2, i1, j);
            for i2 in i1min..=i2max {
                data_ac[((i1 - i1min) as usize) * n11 + ((i2 - i1min) as usize)] =
                    l2 * (data_ab[(i1 as usize) * n11 + i2 as usize] * op3 * psc(gc, i2, j));
            }
        }
        let conv = conv2b_scalar(
            &data_ac,
            n11,
            (j - i1min) as usize,
            (n - i1min) as usize,
            integ,
        );
        let val = pref * (-hh) * (op1 * conv * op4);
        out.ret.block_mut(out.ret.idx(n, j))[0] += val;
    }
    // -- vt (1) fills data_bc, reused by les (1)
    let mut data_bc = vec![Complex::ZERO; n11 * n11];
    for t1 in 0..=n1 {
        let t2max = (t1 + kt).min(n1);
        let gc_ret = psc(gc, t1, n);
        for t2 in 0..=t2max {
            data_bc[(t1 as usize) * n11 + t2 as usize] =
                sgtr(lam2, t2, n) * (psc(gb, t2, t1) * op3 * gc_ret);
        }
    }
    // -- vt (1)
    let mut f_ac = vec![Complex::ZERO; n11];
    for m in 0..=mmax {
        for i2 in 0..=n1 {
            let i1max = (i2 + kt).min(n1);
            let lam1_vt = svt(lam1, m, i2);
            for i1 in 0..=i1max {
                data_ac[(i2 as usize) * n11 + i1 as usize] =
                    lam1_vt * (pvt(ga, m, i1) * op2 * data_bc[(i2 as usize) * n11 + i1 as usize]);
            }
        }
        let conv = conv2b_scalar(&data_ac, n11, 0, n as usize, integ);
        let val = pref * Complex::new(sig * h * h, 0.0) * (op1 * conv * op4).conj();
        out.lmix.block_mut(out.lmix.idx(n, mmax - m))[0] += val;
    }
    // -- les (1)
    for j in 0..=n {
        for i2 in 0..=n1 {
            let i1max = (i2 + kt).min(n1);
            let lam1_les = sles(lam1, j, i2);
            for i1 in 0..=i1max {
                data_ac[(i2 as usize) * n11 + i1 as usize] =
                    lam1_les * (pls(ga, j, i1) * op2 * data_bc[(i2 as usize) * n11 + i1 as usize]);
            }
        }
        let conv = conv2b_scalar(&data_ac, n11, 0, n as usize, integ);
        let val = pref * (-hh) * (op1 * conv * op4);
        out.less.block_mut(out.less.idx(j, n))[0] += val;
    }
    // -- vt (2) fills data2_tv_ab, reused by les (2)
    let mut data2 = vec![Complex::ZERO; n11 * ((mmax + 1) as usize)];
    for tau in 0..=mmax {
        let l2 = svt(lam2, tau, n);
        for t in 0..=n1 {
            data2[(t as usize) * ((mmax + 1) as usize) + tau as usize] =
                l2 * (op2 * pvt(gb, tau, t));
        }
    }
    // -- vt (2)
    for m in 0..=mmax {
        for i2 in 0..=n1 {
            let row = &data2[(i2 as usize) * ((mmax + 1) as usize)..];
            let conv_ab = lowside_scalar(m as usize, ga.mat.raw(), row, integ);
            f_ac[i2 as usize] = svt(lam1, m, i2) * (conv_ab * op3 * psc(gc, i2, n));
        }
        let integr = gregory_scalar(n as usize, &f_ac, integ);
        let val = pref * Complex::new(-sig * dtau * h, 0.0) * (op1 * integr * op4).conj();
        out.lmix.block_mut(out.lmix.idx(n, mmax - m))[0] += val;
    }
    // -- les (2)
    for j in 0..=n {
        for i2 in 0..=n1 {
            let row = &data2[(i2 as usize) * ((mmax + 1) as usize)..];
            let mut f_ab = vec![Complex::ZERO; (mmax + 1) as usize];
            for mu in 0..=mmax {
                f_ab[mu as usize] = ptv(ga, j, mu) * row[mu as usize];
            }
            let integr_ab = gregory_scalar(mmax as usize, &f_ab, integ);
            f_ac[i2 as usize] = sles(lam1, j, i2) * (integr_ab * op3 * psc(gc, i2, n));
        }
        let integr = gregory_scalar(n as usize, &f_ac, integ);
        let val = pref * Complex::new(0.0, -dtau * h) * (op1 * integr * op4);
        out.less.block_mut(out.less.idx(j, n))[0] += val;
    }
    // -- vt (3)
    {
        let lam2_vt: Vec<Complex> = (0..=mmax).map(|tau| svt(lam2, tau, n)).collect();
        let gc_vt: Vec<Complex> = (0..=mmax).map(|tau| pvt(gc, tau, n)).collect();
        for m in 0..=mmax {
            let svt_val = sigma_mat_core(
                m as usize,
                ga.mat.raw(),
                gb.mat.raw(),
                &gc_vt,
                lam1.mat.raw(),
                &lam2_vt,
                mmax as usize,
                integ,
                op1 * op2 * op3 * op4,
            );
            let val = pref * Complex::new(0.0, -sig * dtau * dtau) * svt_val.conj();
            out.lmix.block_mut(out.lmix.idx(n, mmax - m))[0] += val;
        }
    }
    // -- les (3)
    for j in (0..=n).rev() {
        let i1max = j.max(kt);
        for i1 in 0..=i1max {
            let mut f_bc = vec![Complex::ZERO; n11];
            for i2 in 0..=n1 {
                f_bc[i2 as usize] = sles(lam1, j, i2) * (pls(gb, i1, i2) * op3 * psc(gc, i2, n));
            }
            let integr_bc = gregory_scalar(n as usize, &f_bc, integ);
            f_ac[i1 as usize] = sles(lam2, i1, n) * (psc(ga, j, i1) * op2 * integr_bc);
        }
        let integr = gregory_scalar(j as usize, &f_ac, integ);
        let val = pref * hh * (op1 * integr * op4);
        out.less.block_mut(out.less.idx(j, n))[0] += val;
    }
    // -- les (4)
    for j in 0..=n {
        let mut f_bc = vec![Complex::ZERO; (mmax + 1) as usize];
        for mu in 0..=mmax {
            f_bc[mu as usize] = stv(lam1, j, mu) * (op3 * pvt(gc, mu, n));
        }
        let mut f_ac_m = vec![Complex::ZERO; (mmax + 1) as usize];
        for mu in 0..=mmax {
            let conv_bc = lowside_scalar(mu as usize, gb.mat.raw(), &f_bc, integ);
            f_ac_m[mu as usize] = svt(lam2, mu, n) * (ptv(ga, j, mu) * op2 * conv_bc);
        }
        let integr = gregory_scalar(mmax as usize, &f_ac_m, integ);
        let val = pref * Complex::new(0.0, dtau * dtau) * (op1 * integr * op4);
        out.less.block_mut(out.less.idx(j, n))[0] += val;
    }
    // -- les (5)
    {
        let mut data_bc5 = vec![Complex::ZERO; n11 * n11];
        for i1 in 0..=n1 {
            let i2max = (i1 + kt).min(n1);
            let l2 = sles(lam2, i1, n);
            for i2 in 0..=i2max {
                data_bc5[(i1 as usize) * n11 + i2 as usize] =
                    l2 * (psc(gb, i1, i2) * op3 * pls(gc, i2, n));
            }
        }
        for j in (0..=n).rev() {
            let i1max = j.max(kt);
            for i1 in 0..=i1max {
                let i2max = (i1 + kt).min(i1max);
                let l1 = sgtr(lam1, j, i2max.min(i2max));
                let _ = l1;
                let ga_ret = psc(ga, j, i1);
                for i2 in 0..=i2max {
                    data_ac[(i1 as usize) * n11 + i2 as usize] = sgtr(lam1, j, i2)
                        * (ga_ret * op2 * data_bc5[(i1 as usize) * n11 + i2 as usize]);
                }
            }
            let conv = conv2b_scalar(&data_ac, n11, 0, j as usize, integ);
            let val = pref * (-hh) * (op1 * conv * op4);
            out.less.block_mut(out.less.idx(j, n))[0] += val;
        }
    }
    // -- les (6)
    for j in 0..=n {
        let i1max = j.max(kt);
        for i1 in 0..=i1max {
            let mut f_bc = vec![Complex::ZERO; (mmax + 1) as usize];
            for mu in 0..=mmax {
                f_bc[mu as usize] = stv(lam1, j, mu) * (ptv(gb, i1, mu) * op3 * pvt(gc, mu, n));
            }
            let integr_bc = gregory_scalar(mmax as usize, &f_bc, integ);
            f_ac[i1 as usize] = sles(lam2, i1, n) * (psc(ga, j, i1) * op2 * integr_bc);
        }
        let integr = gregory_scalar(j as usize, &f_ac, integ);
        let val = pref * Complex::new(0.0, dtau * h) * (op1 * integr * op4);
        out.less.block_mut(out.less.idx(j, n))[0] += val;
    }
}

/// OCA single-particle GF real-time correction at timestep `n`, ADDED
/// onto the NCA values of the output timestep (blueprint
/// `gdiagram_tstp`: vt parts 1-4, les parts 1-3, gtr parts 1-3 with
/// `G^R = gtr + [G^<]^dag`; the scalar diagram prefactor multiplies

/// Matrix `poly_integral` (blueprint `poly_integral_scalar`: gregory
/// over the long branch, `xiw` weights over the full zero-padded
/// array in the interpolation branch).
fn poly_integral_blk(
    v: &[Complex],
    i0: usize,
    i1: usize,
    nel: usize,
    integ: &Integrator,
) -> Vec<Complex> {
    let k = integ.k as usize;
    let mut out = vec![Complex::ZERO; nel];
    if i1 - i0 > k {
        for q in 0..nel {
            let c: Vec<Complex> = (i0..=i1).map(|i| v[i * nel + q]).collect();
            out[q] = gregory_scalar(i1 - i0, &c, integ);
        }
    } else {
        for q in 0..nel {
            let mut acc = Complex::ZERO;
            for j in 0..=k {
                let w = integ.xiw.at(i0, i1, j);
                if w != 0.0 {
                    acc += Complex::new(w, 0.0) * v[j * nel + q];
                }
            }
            out[q] = acc;
        }
    }
    out
}

/// Matrix lowside contraction with a sandwiched operator
/// (blueprint `lowside_op_scalar`: `bcw` warmup double sum over
/// `aa[l] . op . bb[m]`; gregory fold of `aa[j-l] . op . bb[l]`).
#[allow(clippy::too_many_arguments)]
fn lowside_op_blk(
    j: usize,
    aa: &[Complex],
    op: &kadanoffbaym::linalg::DenseMat,
    bb: &[Complex],
    d1: usize,
    d2: usize,
    d3: usize,
    d4: usize,
    integ: &Integrator,
    scratch: &mut Vec<Complex>,
) -> Vec<Complex> {
    let k = integ.k as usize;
    let nela = d1 * d2;
    let nelb = d3 * d4;
    let nelo = d1 * d4;
    let smax = nela.max(nelb).max(nelo);
    if scratch.len() < smax {
        scratch.resize(smax, Complex::ZERO);
    }
    let mut out = vec![Complex::ZERO; nelo];
    let mut prod = vec![Complex::ZERO; nelo];
    if j > 0 && j < k {
        for l in 0..=k {
            for m in 0..=k {
                let w = integ.bcw.at(j, l, m);
                if w != 0.0 {
                    sandwich(
                        &aa[l * nela..],
                        op.raw(),
                        &bb[m * nelb..],
                        d1,
                        d2,
                        d3,
                        d4,
                        &mut prod,
                        scratch,
                    );
                    for q in 0..nelo {
                        out[q] += Complex::new(w, 0.0) * prod[q];
                    }
                }
            }
        }
    } else if j > 0 {
        let mut f1 = vec![Complex::ZERO; (j + 1) * nelo];
        for l in 0..=j {
            sandwich(
                &aa[(j - l) * nela..],
                op.raw(),
                &bb[l * nelb..],
                d1,
                d2,
                d3,
                d4,
                &mut prod,
                scratch,
            );
            f1[l * nelo..(l + 1) * nelo].copy_from_slice(&prod);
        }
        out = gregory_blk(j, &f1, nelo, integ);
    }
    out
}

/// Matrix highside contraction with a sandwiched operator
/// (blueprint `highside_op_scalar`).
#[allow(clippy::too_many_arguments)]
fn highside_op_blk(
    j: usize,
    aa: &[Complex],
    op: &kadanoffbaym::linalg::DenseMat,
    bb: &[Complex],
    d1: usize,
    d2: usize,
    d3: usize,
    d4: usize,
    mmax: usize,
    integ: &Integrator,
    scratch: &mut Vec<Complex>,
) -> Vec<Complex> {
    let k = integ.k as usize;
    let j1 = mmax - j;
    let nela = d1 * d2;
    let nelb = d3 * d4;
    let nelo = d1 * d4;
    let smax = nela.max(nelb).max(nelo);
    if scratch.len() < smax {
        scratch.resize(smax, Complex::ZERO);
    }
    let mut out = vec![Complex::ZERO; nelo];
    let mut prod = vec![Complex::ZERO; nelo];
    if j1 > 0 && j1 < k {
        for l in 0..=k {
            for m in 0..=k {
                let w = integ.bcw.at(j1, l, m);
                if w != 0.0 {
                    sandwich(
                        &aa[(mmax - l) * nela..],
                        op.raw(),
                        &bb[m * nelb..],
                        d1,
                        d2,
                        d3,
                        d4,
                        &mut prod,
                        scratch,
                    );
                    for q in 0..nelo {
                        out[q] += Complex::new(w, 0.0) * prod[q];
                    }
                }
            }
        }
    } else if j1 > 0 {
        let mut f1 = vec![Complex::ZERO; (j1 + 1) * nelo];
        for l in 0..=j1 {
            sandwich(
                &aa[(mmax - (j1 - l)) * nela..],
                op.raw(),
                &bb[l * nelb..],
                d1,
                d2,
                d3,
                d4,
                &mut prod,
                scratch,
            );
            f1[l * nelo..(l + 1) * nelo].copy_from_slice(&prod);
        }
        out = gregory_blk(j1, &f1, nelo, integ);
    }
    out
}

/// Matrix version of [`add_gf_tstp_oca`] for multi-band sectors
/// (blueprint `gdiagram_tstp` with matrix types; outputs are traces
/// written into the 1x1 vertex-GF timestep blocks; the hybridization
/// insertions stay scalar). Op labels follow the same C++-to-builder
/// mapping as the mat kernel: the scalar code's `op1`/`op4` are this
/// diagram's `d.op4`/`d.op1`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_gf_tstp_oca_matrix(
    n: i64,
    d: &OcaGfDiagram,
    pp_g: &[PseudoGreen],
    beta: f64,
    h: f64,
    kt: i64,
    integ: &Integrator,
    out: &mut kadanoffbaym::types::GreenTimeStep,
) {
    let mmax = out.ntau as i64;
    let ntau_pts = (mmax + 1) as usize;
    let n1 = n.max(kt);
    let n11 = (n1 + 1) as usize;
    let ga = &pp_g[d.sa];
    let gb = &pp_g[d.sb];
    let gc = &pp_g[d.sc];
    let gd = &pp_g[d.sd];
    let lam = d.lam;
    let k = integ.k as usize;
    let (adim, bdim, cdim, ddim) = (
        ga.size1 as usize,
        gb.size1 as usize,
        gc.size1 as usize,
        gd.size1 as usize,
    );
    let (na, nb, nc, nd) = (adim * adim, bdim * bdim, cdim * cdim, ddim * ddim);
    let ndc = ddim * cdim;
    let nda = ddim * adim;
    let nba = bdim * adim;
    let pref = Complex::new(d.prefactor * d.rt_flip, 0.0);
    let smax = na.max(nb).max(nc).max(nd).max(ndc).max(nda).max(nba);
    let mut scratch = vec![Complex::ZERO; smax];
    let mut gdb = vec![Complex::ZERO; nd];
    let mut gab_b = vec![Complex::ZERO; nb];
    let mut gcb = vec![Complex::ZERO; nc];
    let mut gab = vec![Complex::ZERO; na];
    let mut t_dc = vec![Complex::ZERO; ndc];
    let mut t_db = vec![Complex::ZERO; ddim * bdim];
    let mut t_da = vec![Complex::ZERO; nda];
    let mut t_ba = vec![Complex::ZERO; nba];

    // -- precompute (block-valued)
    let mut data_a = vec![Complex::ZERO; n11 * ntau_pts * na];
    let mut data_d = vec![Complex::ZERO; n11 * ntau_pts * nd];
    for i2 in 0..=n1 {
        for mu in 0..=mmax {
            let s = svt(lam, mu, i2);
            let base = (i2 as usize) * ntau_pts * na + (mu as usize) * na;
            let mut pv = vec![Complex::ZERO; na];
            ptv_blk_into(ga, mu, n, adim, &mut pv);
            for q in 0..na {
                data_a[base + q] = s * pv[q];
            }
        }
    }
    for i1 in 0..=n1 {
        for mu in 0..=mmax {
            let s = stv(lam, i1, mu);
            let base = (i1 as usize) * ntau_pts * nd + (mu as usize) * nd;
            let mut pv = vec![Complex::ZERO; nd];
            ptv_dir_blk(gd, n, mu, ddim, &mut pv);
            for q in 0..nd {
                data_d[base + q] = s * pv[q];
            }
        }
    }
    let mut gavt = vec![Complex::ZERO; ntau_pts * na];
    let mut gdtv = vec![Complex::ZERO; ntau_pts * nd];
    for tau in 0..=mmax {
        ptv_blk_into(ga, tau, n, adim, &mut gab);
        gavt[(tau as usize) * na..(tau as usize + 1) * na].copy_from_slice(&gab);
        ptv_dir_blk(gd, n, tau, ddim, &mut gdb);
        gdtv[(tau as usize) * nd..(tau as usize + 1) * nd].copy_from_slice(&gdb);
    }

    let mut f_dc = vec![Complex::ZERO; n11.max(ntau_pts) * ndc];
    let mut f_dc_ba = vec![Complex::ZERO; n11.max(ntau_pts) * nba];
    let mut f_da = vec![Complex::ZERO; n11 * nda];

    // -- tv component
    // hoisted vt(2) matcore buffers (reused across the m loop; the
    // matcore's inner loops must not allocate)
    let ga_axis_vt = MatAxis {
        data: &gavt,
        d1: adim,
        d2: adim,
    };
    let gb_axis_vt = MatAxis {
        data: gb.mat.raw(),
        d1: bdim,
        d2: bdim,
    };
    let gc_axis_vt = MatAxis {
        data: gc.mat.raw(),
        d1: cdim,
        d2: cdim,
    };
    let gd_axis_vt = MatAxis {
        data: &gdtv,
        d1: ddim,
        d2: ddim,
    };
    let lam_axis_vt = MatAxis {
        data: lam.mat.raw(),
        d1: 1,
        d2: 1,
    };
    let mut vt2_lam_i = vec![Complex::ZERO; 1];
    let mut vt2_scratch = vec![Complex::ZERO; smax];
    let max_nel_vt = ndc.max(nba).max(nda);
    let mut vt2_mscratch = GfMatScratch::new(max_nel_vt, k, ddim);
    for m in 0..=mmax {
        // vt (1)
        for i1 in 0..=n1 {
            for i2 in 0..=n1 {
                psc_blk_into(gd, n, i2, ddim, &mut gdb);
                let mut pv = vec![Complex::ZERO; nc];
                ptv_dir_blk(gc, i2, m, cdim, &mut pv);
                sandwich(
                    &gdb,
                    d.op4.raw(),
                    &pv,
                    ddim,
                    ddim,
                    cdim,
                    cdim,
                    &mut t_dc,
                    &mut scratch,
                );
                let sv = sgtr(lam, i1, i2);
                for q in 0..ndc {
                    f_dc[(i2 as usize) * ndc + q] = sv * t_dc[q];
                }
            }
            let integr_dc = gregory_blk(n as usize, &f_dc[..(n1 as usize + 1) * ndc], ndc, integ);
            let mut pv = vec![Complex::ZERO; nb];
            ptv_blk_into(gb, m, i1, bdim, &mut pv);
            sandwich(
                &integr_dc,
                d.op2.raw(),
                &pv,
                ddim,
                cdim,
                bdim,
                bdim,
                &mut t_db,
                &mut scratch,
            );
            psc_blk_into(ga, i1, n, adim, &mut gab);
            sandwich(
                &t_db,
                d.op3.raw(),
                &gab,
                ddim,
                bdim,
                adim,
                adim,
                &mut f_da[(i1 as usize) * nda..],
                &mut scratch,
            );
        }
        let vt1_blk = {
            let mut v = gregory_blk(n as usize, &f_da[..(n1 as usize + 1) * nda], nda, integ);
            let c = -Complex::new(h * h, 0.0);
            for q in &mut v {
                *q *= c;
            }
            v
        };
        // vt (2): full mat-core evaluation with the direct (non-flipped)
        // gd rows and the traced closing product
        let vt2 = Complex::new(0.0, beta / mmax as f64 * (beta / mmax as f64))
            * gf_mat_matrix_branch(
                m as usize,
                k as usize,
                mmax as usize,
                &ga_axis_vt,
                &gb_axis_vt,
                &gc_axis_vt,
                &gd_axis_vt,
                &lam_axis_vt,
                d,
                pp_g[d.sd].sign.as_i64() as f64,
                false,
                integ,
                &mut vt2_lam_i,
                &mut vt2_scratch,
                &mut vt2_mscratch,
            );
        // vt (3)
        for i2 in 0..=n1 {
            let row = &data_a[(i2 as usize) * ntau_pts * na..];
            let conv_ba = lowside_op_blk(
                m as usize,
                gb.mat.raw(),
                &d.op3,
                row,
                bdim,
                bdim,
                adim,
                adim,
                integ,
                &mut scratch,
            );
            psc_blk_into(gd, n, i2, ddim, &mut gdb);
            let mut pv = vec![Complex::ZERO; nc];
            ptv_dir_blk(gc, i2, m, cdim, &mut pv);
            sandwich(
                &gdb,
                d.op4.raw(),
                &pv,
                ddim,
                ddim,
                cdim,
                cdim,
                &mut t_dc,
                &mut scratch,
            );
            sandwich(
                &t_dc,
                d.op2.raw(),
                &conv_ba,
                ddim,
                cdim,
                bdim,
                adim,
                &mut f_da[(i2 as usize) * nda..],
                &mut scratch,
            );
        }
        let vt3_blk = {
            let mut v = gregory_blk(n as usize, &f_da[..(n1 as usize + 1) * nda], nda, integ);
            let c = Complex::new(h * beta / mmax as f64, 0.0);
            for q in &mut v {
                *q *= c;
            }
            v
        };
        // vt (4)
        for i1 in 0..=n1 {
            let row = &data_d[(i1 as usize) * ntau_pts * nd..];
            let conv_dc = highside_op_blk(
                m as usize,
                row,
                &d.op4,
                gc.mat.raw(),
                ddim,
                ddim,
                cdim,
                cdim,
                mmax as usize,
                integ,
                &mut scratch,
            );
            let mut pv = vec![Complex::ZERO; nb];
            ptv_blk_into(gb, m, i1, bdim, &mut pv);
            sandwich(
                &conv_dc,
                d.op2.raw(),
                &pv,
                ddim,
                cdim,
                bdim,
                bdim,
                &mut t_db,
                &mut scratch,
            );
            psc_blk_into(ga, i1, n, adim, &mut gab);
            sandwich(
                &t_db,
                d.op3.raw(),
                &gab,
                ddim,
                bdim,
                adim,
                adim,
                &mut f_da[(i1 as usize) * nda..],
                &mut scratch,
            );
        }
        let vt4_blk = {
            let mut v = gregory_blk(n as usize, &f_da[..(n1 as usize + 1) * nda], nda, integ);
            let c = -Complex::new(h * beta / mmax as f64, 0.0);
            for q in &mut v {
                *q *= c;
            }
            v
        };
        // total: vt2 (complete, incl. the traced closing) plus the traced
        // (vt1+vt3+vt4) . op1 chain
        let mut sum_blk = vec![Complex::ZERO; nda];
        for q in 0..nda {
            sum_blk[q] = vt1_blk[q] + vt3_blk[q] + vt4_blk[q];
        }
        let total = vt2 + trace_close_op(&sum_blk, &d.op1, ddim, adim);
        out.lmix.raw_block_at_mut(0, mmax - m)[0] += pref * total.conj();
    }

    // -- les component
    let mut g_les = vec![Complex::ZERO; (n + 1) as usize];
    for j in 0..=n {
        let i2min = j.min(n1 - kt);
        for i2 in i2min..=n1 {
            // les (1)
            for i1 in 0..=n1 {
                pls_blk_into(gb, j, i1, bdim, &mut gab_b);
                psc_blk_into(ga, i1, n, adim, &mut gab);
                sandwich(
                    &gab_b,
                    d.op3.raw(),
                    &gab,
                    bdim,
                    bdim,
                    adim,
                    adim,
                    &mut t_ba,
                    &mut scratch,
                );
                let sv = sgtr(lam, i1, i2);
                for q in 0..nba {
                    f_dc_ba[(i1 as usize) * nba + q] = sv * t_ba[q];
                }
            }
            let les1_blk = {
                let mut v =
                    gregory_blk(n as usize, &f_dc_ba[..(n1 as usize + 1) * nba], nba, integ);
                let c = -Complex::new(h, 0.0);
                for q in &mut v {
                    *q *= c;
                }
                v
            };
            // les (2)
            for mu in 0..=mmax {
                let mut pv_b = vec![Complex::ZERO; nb];
                ptv_dir_blk(gb, j, mu, bdim, &mut pv_b);
                let mut pv_a = vec![Complex::ZERO; na];
                ptv_blk_into(ga, mu, n, adim, &mut pv_a);
                sandwich(
                    &pv_b,
                    d.op3.raw(),
                    &pv_a,
                    bdim,
                    bdim,
                    adim,
                    adim,
                    &mut t_ba,
                    &mut scratch,
                );
                let sv = svt(lam, mu, i2);
                for q in 0..nba {
                    f_dc_ba[(mu as usize) * nba + q] = sv * t_ba[q];
                }
            }
            let les2_blk = {
                let mut v = gregory_blk(mmax as usize, &f_dc_ba[..ntau_pts * nba], nba, integ);
                let c = -Complex::new(0.0, beta / mmax as f64);
                for q in &mut v {
                    *q *= c;
                }
                v
            };
            // les (3)
            let i1max = j.max(kt);
            for i1 in 0..=i1max {
                psc_blk_into(gb, j, i1, bdim, &mut gab_b);
                pls_blk_into(ga, i1, n, adim, &mut gab);
                sandwich(
                    &gab_b,
                    d.op3.raw(),
                    &gab,
                    bdim,
                    bdim,
                    adim,
                    adim,
                    &mut t_ba,
                    &mut scratch,
                );
                let sv = sles(lam, i1, i2);
                for q in 0..nba {
                    f_dc_ba[(i1 as usize) * nba + q] = sv * t_ba[q];
                }
            }
            let les3_blk = {
                let mut v = gregory_blk(
                    j as usize,
                    &f_dc_ba[..((i1max + 1) as usize) * nba],
                    nba,
                    integ,
                );
                let c = Complex::new(h, 0.0);
                for q in &mut v {
                    *q *= c;
                }
                v
            };
            let mut sum_ba = vec![Complex::ZERO; nba];
            for q in 0..nba {
                sum_ba[q] = les1_blk[q] + les2_blk[q] + les3_blk[q];
            }
            psc_blk_into(gd, n, i2, ddim, &mut gdb);
            psc_blk_into(gc, i2, j, cdim, &mut gcb);
            sandwich(
                &gdb,
                d.op4.raw(),
                &gcb,
                ddim,
                ddim,
                cdim,
                cdim,
                &mut t_dc,
                &mut scratch,
            );
            sandwich(
                &t_dc,
                d.op2.raw(),
                &sum_ba,
                ddim,
                cdim,
                bdim,
                adim,
                &mut f_da[((i2 - i2min) as usize) * nda..],
                &mut scratch,
            );
        }
        let integr_da = poly_integral_blk(
            &f_da,
            (j - i2min) as usize,
            (n - i2min) as usize,
            nda,
            integ,
        );
        g_les[j as usize] =
            pref * Complex::new(h, 0.0) * trace_close_op(&integr_da, &d.op1, ddim, adim);
        out.less.block_mut(out.less.idx(j, n))[0] += g_les[j as usize];
    }

    // -- ret component (gtr parts + les^dag)
    for j in 0..=n {
        let i1min = j.min(n1 - kt);
        for i1 in i1min..=n1 {
            // gtr (1)
            for i2 in 0..=n1 {
                pls_blk_into(gd, j, i2, ddim, &mut gdb);
                psc_blk_into(gc, i2, n, cdim, &mut gcb);
                sandwich(
                    &gdb,
                    d.op4.raw(),
                    &gcb,
                    ddim,
                    ddim,
                    cdim,
                    cdim,
                    &mut t_dc,
                    &mut scratch,
                );
                let sv = sles(lam, i1, i2);
                for q in 0..ndc {
                    f_dc[(i2 as usize) * ndc + q] = sv * t_dc[q];
                }
            }
            let gtr1_blk = {
                let mut v = gregory_blk(n as usize, &f_dc[..(n1 as usize + 1) * ndc], ndc, integ);
                let c = -Complex::new(h, 0.0);
                for q in &mut v {
                    *q *= c;
                }
                v
            };
            // gtr (2)
            for mu in 0..=mmax {
                let mut pv_d = vec![Complex::ZERO; nd];
                ptv_dir_blk(gd, j, mu, ddim, &mut pv_d);
                let mut pv_c = vec![Complex::ZERO; nc];
                ptv_blk_into(gc, mu, n, cdim, &mut pv_c);
                sandwich(
                    &pv_d,
                    d.op4.raw(),
                    &pv_c,
                    ddim,
                    ddim,
                    cdim,
                    cdim,
                    &mut t_dc,
                    &mut scratch,
                );
                let sv = stv(lam, i1, mu);
                for q in 0..ndc {
                    f_dc[(mu as usize) * ndc + q] = sv * t_dc[q];
                }
            }
            let gtr2_blk = {
                let mut v = gregory_blk(mmax as usize, &f_dc[..ntau_pts * ndc], ndc, integ);
                let c = -Complex::new(0.0, beta / mmax as f64);
                for q in &mut v {
                    *q *= c;
                }
                v
            };
            // gtr (3)
            let i2max = j.max(kt);
            for i2 in 0..=i2max {
                psc_blk_into(gd, j, i2, ddim, &mut gdb);
                pls_blk_into(gc, i2, n, cdim, &mut gcb);
                sandwich(
                    &gdb,
                    d.op4.raw(),
                    &gcb,
                    ddim,
                    ddim,
                    cdim,
                    cdim,
                    &mut t_dc,
                    &mut scratch,
                );
                let sv = sgtr(lam, i1, i2);
                for q in 0..ndc {
                    f_dc[(i2 as usize) * ndc + q] = sv * t_dc[q];
                }
            }
            let gtr3_blk = {
                let mut v = gregory_blk(
                    j as usize,
                    &f_dc[..((i2max + 1) as usize) * ndc],
                    ndc,
                    integ,
                );
                let c = Complex::new(h, 0.0);
                for q in &mut v {
                    *q *= c;
                }
                v
            };
            let mut sum_dc = vec![Complex::ZERO; ndc];
            for q in 0..ndc {
                sum_dc[q] = gtr1_blk[q] + gtr2_blk[q] + gtr3_blk[q];
            }
            psc_blk_into(gb, n, i1, bdim, &mut gab_b);
            sandwich(
                &sum_dc,
                d.op2.raw(),
                &gab_b,
                ddim,
                cdim,
                bdim,
                bdim,
                &mut t_db,
                &mut scratch,
            );
            psc_blk_into(ga, i1, j, adim, &mut gab);
            sandwich(
                &t_db,
                d.op3.raw(),
                &gab,
                ddim,
                bdim,
                adim,
                adim,
                &mut f_da[((i1 - i1min) as usize) * nda..],
                &mut scratch,
            );
        }
        let integr_da = poly_integral_blk(
            &f_da,
            (j - i1min) as usize,
            (n - i1min) as usize,
            nda,
            integ,
        );
        let val = pref * Complex::new(h, 0.0) * trace_close_op(&integr_da, &d.op1, ddim, adim)
            + g_les[j as usize].conj();
        out.ret.block_mut(out.ret.idx(n, j))[0] += val;
    }
}

/// Trace of `(m . op)` for the closing `sd --op1--> sa` product:
/// `m` is `(ddim x adim)` row-major, `op` is `(adim x ddim)`.
fn trace_close_op(
    m: &[Complex],
    op: &kadanoffbaym::linalg::DenseMat,
    ddim: usize,
    adim: usize,
) -> Complex {
    let mut tr = Complex::ZERO;
    for r in 0..ddim {
        for q in 0..adim {
            tr += m[r * adim + q] * op.get(q, r);
        }
    }
    tr
}
/// every term).
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_gf_tstp_oca(
    n: i64,
    d: &OcaGfDiagram,
    pp_g: &[PseudoGreen],
    beta: f64,
    h: f64,
    kt: i64,
    integ: &Integrator,
    out: &mut kadanoffbaym::types::GreenTimeStep,
) {
    // multi-band sectors take the matrix path (blueprint gdiagram_tstp
    // with matrix types); the scalar fast path below stays P5-gated
    // for all-1x1 chains
    if pp_g[d.sa].size1 * pp_g[d.sb].size1 * pp_g[d.sc].size1 * pp_g[d.sd].size1 > 1 {
        add_gf_tstp_oca_matrix(n, d, pp_g, beta, h, kt, integ, out);
        return;
    }
    debug_assert_eq!(out.size1 * out.size2, 1);
    let mmax = out.ntau as i64;
    let ntau_pts = (mmax + 1) as usize;
    let n1 = n.max(kt);
    let ga = &pp_g[d.sa];
    let gb = &pp_g[d.sb];
    let gc = &pp_g[d.sc];
    let gd = &pp_g[d.sd];
    let lam = d.lam;
    let (op1, op2, op3, op4) = (
        d.op1.get(0, 0),
        d.op2.get(0, 0),
        d.op3.get(0, 0),
        d.op4.get(0, 0),
    );
    let pref = Complex::new(d.prefactor * d.rt_flip, 0.0);

    // -- precompute
    let mut data_a = vec![Complex::ZERO; ((n1 + 1) as usize) * ntau_pts];
    let mut data_d = vec![Complex::ZERO; ((n1 + 1) as usize) * ntau_pts];
    for i2 in 0..=n1 {
        for mu in 0..=mmax {
            data_a[(i2 as usize) * ntau_pts + mu as usize] = svt(lam, mu, i2) * pvt(ga, mu, n);
        }
    }
    for i1 in 0..=n1 {
        for mu in 0..=mmax {
            data_d[(i1 as usize) * ntau_pts + mu as usize] = stv(lam, i1, mu) * ptv(gd, n, mu);
        }
    }
    let gavt: Vec<Complex> = (0..=mmax).map(|tau| pvt(ga, tau, n)).collect();
    let gdtv: Vec<Complex> = (0..=mmax).map(|tau| ptv(gd, n, tau)).collect();

    let mut f_dc = vec![Complex::ZERO; ((n1 + 1) as usize).max(ntau_pts)];
    let mut f_da = vec![Complex::ZERO; ((n1 + 1) as usize).max(ntau_pts)];

    // -- tv component
    for m in 0..=mmax {
        // vt (1)
        for i1 in 0..=n1 {
            for i2 in 0..=n1 {
                f_dc[i2 as usize] = sgtr(lam, i1, i2) * (psc(gd, n, i2) * op1 * ptv(gc, i2, m));
            }
            let integr_dc = gregory_scalar(n as usize, &f_dc, integ);
            f_da[i1 as usize] = integr_dc * op2 * pvt(gb, m, i1) * op3 * psc(ga, i1, n);
        }
        let integr_vt1 = -Complex::new(h * h, 0.0) * gregory_scalar(n as usize, &f_da, integ);
        // vt (2)
        let vt2 = Complex::new(0.0, beta / mmax as f64 * (beta / mmax as f64))
            * gf_mat_core(
                m as usize,
                &gavt,
                gb.mat.raw(),
                gc.mat.raw(),
                &gdtv,
                lam.mat.raw(),
                mmax as usize,
                integ,
                false,
                gd.sign.as_i64() as f64,
                op1 * op2 * op3 * op4,
            );
        // vt (3)
        for i2 in 0..=n1 {
            let row = &data_a[(i2 as usize) * ntau_pts..];
            let conv_ba = lowside_op_scalar(m as usize, gb.mat.raw(), op3, row, integ);
            f_da[i2 as usize] = psc(gd, n, i2) * op1 * ptv(gc, i2, m) * op2 * conv_ba;
        }
        let integr_vt3 =
            Complex::new(h * beta / mmax as f64, 0.0) * gregory_scalar(n as usize, &f_da, integ);
        // vt (4)
        for i1 in 0..=n1 {
            let row = &data_d[(i1 as usize) * ntau_pts..];
            let conv_dc =
                highside_op_scalar(m as usize, row, op1, gc.mat.raw(), mmax as usize, integ);
            f_da[i1 as usize] = conv_dc * op2 * pvt(gb, m, i1) * op3 * psc(ga, i1, n);
        }
        let integr_vt4 =
            -Complex::new(h * beta / mmax as f64, 0.0) * gregory_scalar(n as usize, &f_da, integ);
        // result
        let total = vt2 + (integr_vt1 + integr_vt3 + integr_vt4) * op4;
        // GreenTimeStep lmix holds a single row (the timestep's own)
        out.lmix.raw_block_at_mut(0, mmax - m)[0] += pref * total.conj();
    }

    // -- les component
    let mut g_les = vec![Complex::ZERO; (n + 1) as usize];
    for j in 0..=n {
        let i2min = j.min(n1 - kt);
        for i2 in i2min..=n1 {
            // les (1)
            for i1 in 0..=n1 {
                f_dc[i1 as usize] = sgtr(lam, i1, i2) * (pls(gb, j, i1) * op3 * psc(ga, i1, n));
            }
            let les1 = -Complex::new(h, 0.0) * gregory_scalar(n as usize, &f_dc, integ);
            // les (2)
            for mu in 0..=mmax {
                f_dc[mu as usize] = svt(lam, mu, i2) * (ptv(gb, j, mu) * op3 * pvt(ga, mu, n));
            }
            let les2 = -Complex::new(0.0, beta / mmax as f64)
                * gregory_scalar(mmax as usize, &f_dc, integ);
            // les (3)
            let i1max = j.max(kt);
            for i1 in 0..=i1max {
                f_dc[i1 as usize] = sles(lam, i1, i2) * (psc(gb, j, i1) * op3 * pls(ga, i1, n));
            }
            let les3 = Complex::new(h, 0.0) * gregory_scalar(j as usize, &f_dc, integ);
            f_da[(i2 - i2min) as usize] =
                psc(gd, n, i2) * op1 * psc(gc, i2, j) * op2 * (les1 + les2 + les3);
        }
        // blueprint poly_integral reads the FULL (zero-padded) array in
        // the interpolation branch, not a truncated view
        let integr_da =
            poly_integral_scalar(&f_da, (j - i2min) as usize, (n - i2min) as usize, integ);
        g_les[j as usize] = pref * Complex::new(h, 0.0) * (integr_da * op4);
        out.less.block_mut(out.less.idx(j, n))[0] += g_les[j as usize];
    }

    // -- ret component (gtr parts + les^dag)
    for j in 0..=n {
        let i1min = j.min(n1 - kt);
        for i1 in i1min..=n1 {
            // gtr (1)
            for i2 in 0..=n1 {
                f_dc[i2 as usize] = sles(lam, i1, i2) * (pls(gd, j, i2) * op1 * psc(gc, i2, n));
            }
            let gtr1 = -Complex::new(h, 0.0) * gregory_scalar(n as usize, &f_dc, integ);
            // gtr (2)
            for mu in 0..=mmax {
                f_dc[mu as usize] = stv(lam, i1, mu) * (ptv(gd, j, mu) * op1 * pvt(gc, mu, n));
            }
            let gtr2 = -Complex::new(0.0, beta / mmax as f64)
                * gregory_scalar(mmax as usize, &f_dc, integ);
            // gtr (3)
            let i2max = j.max(kt);
            for i2 in 0..=i2max {
                f_dc[i2 as usize] = sgtr(lam, i1, i2) * (psc(gd, j, i2) * op1 * pls(gc, i2, n));
            }
            let gtr3 = Complex::new(h, 0.0) * gregory_scalar(j as usize, &f_dc, integ);
            f_da[(i1 - i1min) as usize] =
                (gtr1 + gtr2 + gtr3) * op2 * psc(gb, n, i1) * op3 * psc(ga, i1, j);
        }
        let integr_da =
            poly_integral_scalar(&f_da, (j - i1min) as usize, (n - i1min) as usize, integ);
        let val = pref * Complex::new(h, 0.0) * (integr_da * op4) + g_les[j as usize].conj();
        out.ret.block_mut(out.ret.idx(n, j))[0] += val;
    }
}

#[cfg(test)]
mod rt_mat_contact {
    use super::*;
    use crate::dmft::{set_bwd_from_fwd_mat, set_bwd_from_fwd_tstp};
    use crate::hamilton::LocalHamiltonian::TwoBandKanamori;
    use crate::hilbert::{SectorClass, build};
    use crate::nca::{GfVertex, Hybridization};
    use crate::solver::{ImpuritySolver, Solver};
    use kadanoffbaym::types::{Contour, ContourGreen, GreenTimeStep, Sign, TimeFunction};

    /// `t = 0` contact of the OCA one-line Green's-function diagram
    /// on the two-band Kanamori dimer: per hybridization channel, the
    /// real-time kernel's `lmix(0, beta - tau)` must equal
    /// `-i * G^M_OCA(tau)` of the Matsubara kernel (the NCA part
    /// satisfies the same relation). This pins the statistics sign of
    /// the forward and backward orderings of the straddling line: the
    /// sector-parity sign of the blueprint reproduces it on one band
    /// only (see [`OcaGfDiagram::gd_sign`]).
    #[test]
    fn oca_gf_realtime_t0_contact_twoband() {
        let (nt, ntau, kt, h) = (6usize, 96usize, 5usize, 0.02);
        let beta = 5.0;
        let hs = build(2, SectorClass::SpinResolved);
        let ham = TwoBandKanamori {
            mu: 0.0,
            u: vec![2.0; nt + 2],
            j: vec![0.4; nt + 2],
            delta: vec![0.8; nt + 2],
            bz: vec![0.0; nt + 2],
        };
        let contour = Contour::new(nt as i64, ntau as i64, 1, 1, h * nt as f64, beta);
        let mk_level = |eps: f64| -> ContourGreen {
            let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
            let sig = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
            let mut hf = TimeFunction::zeros(nt, 1, 1);
            for tstp in -1..=(nt as i64) {
                hf.block_mut(tstp)[0] = Complex::new(eps, 0.0);
            }
            kadanoffbaym::dyson::dyson_mat(
                &mut g,
                &sig,
                0.0,
                &hf,
                beta,
                kt as i64,
                kadanoffbaym::base::MatsubaraMethod::Fixpoint,
                true,
                None,
                1e-12,
            );
            kadanoffbaym::dyson::dyson_start(&mut g, 0.0, &hf, &sig, beta, h, kt as i64);
            for t in (kt + 1)..=nt {
                kadanoffbaym::dyson::dyson_timestep(
                    t as i64,
                    &mut g,
                    0.0,
                    &hf,
                    &sig,
                    beta,
                    h,
                    kt as i64,
                );
            }
            g
        };
        let lvl1 = mk_level(0.5);
        let lvl2 = mk_level(-0.8);
        let scale_green = |g: &ContourGreen, w: f64| -> ContourGreen {
            let wc = Complex::new(w, 0.0);
            let mut out = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
            for n in 0..=(nt as i64) {
                for jj in 0..=n {
                    out.ret.block_mut(out.ret.idx(n, jj))[0] =
                        g.ret.block(g.ret.idx(n, jj))[0] * wc;
                    out.less.block_mut(out.less.idx(jj, n))[0] =
                        g.less.block(g.less.idx(jj, n))[0] * wc;
                }
                for m in 0..=(ntau as i64) {
                    out.lmix.raw_block_at_mut(n, m)[0] = g.lmix.raw_block_at(n, m)[0] * wc;
                }
            }
            for m in 0..=ntau {
                out.mat.block_mut(m)[0] = g.mat.block(m)[0] * wc;
            }
            out
        };
        let mk_pair = |lvl: &ContourGreen, v: f64| -> (ContourGreen, ContourGreen) {
            let fwd = scale_green(lvl, v * v);
            let mut cc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
            set_bwd_from_fwd_mat(&mut cc, &fwd);
            for t in 0..=(nt as i64) {
                set_bwd_from_fwd_tstp(t, &mut cc, &fwd);
            }
            (fwd, cc)
        };
        let (d1, d1c) = mk_pair(&lvl1, 0.3);
        let (d2, d2c) = mk_pair(&lvl2, 0.35);
        let c0a = hs.flavor(0, 0, false);
        let c0c = hs.flavor(0, 0, true);
        let c1a = hs.flavor(1, 0, false);
        let c1c = hs.flavor(1, 0, true);
        let pp_ints = [
            Hybridization { delta: &d1, op_cre_fl: c0c, op_ann_fl: c0a, fermion: true, backward: false },
            Hybridization { delta: &d1c, op_cre_fl: c0a, op_ann_fl: c0c, fermion: true, backward: true },
            Hybridization { delta: &d2, op_cre_fl: c1c, op_ann_fl: c1a, fermion: true, backward: false },
            Hybridization { delta: &d2c, op_cre_fl: c1a, op_ann_fl: c1c, fermion: true, backward: true },
        ];
        let verts = [
            GfVertex { op_ann_fl: c0a, op_cre_fl: c0c },
            GfVertex { op_ann_fl: c1a, op_cre_fl: c1c },
        ];
        let mut solver =
            ImpuritySolver::with_solver(nt, ntau, beta, h, kt, Solver::Ppsc, 2, hs.clone(), ham);
        solver.solve_atomic();
        for _ in 0..12 {
            solver.pp_step(-1, &pp_ints);
        }
        solver.set_t0();
        solver.pp_step(0, &pp_ints);
        let integ = Integrator::new(kt as i64);

        // NCA part: the same contact holds for the bare vertex term
        let nca_mat = crate::nca::gf_mat(&hs, &solver.pp_g, &verts);
        let nca_rt = crate::nca::gf_tstp(&contour, 0, &hs, &solver.pp_g, &verts);
        for v in 0..2 {
            for m in 0..=ntau {
                let lm = nca_rt[v].lmix.raw_block_at(0, (ntau - m) as i64)[0];
                let want = Complex::new(0.0, -nca_mat[v][m].re);
                assert!(
                    (lm - want).abs() < 1e-12,
                    "NCA v{v} m{m}: lmix {lm} vs -i G^M {want}"
                );
            }
        }

        // OCA increment, channel by channel
        for ci in 0..4 {
            let hc = &pp_ints[ci];
            let one = [Hybridization {
                delta: hc.delta,
                op_cre_fl: hc.op_cre_fl,
                op_ann_fl: hc.op_ann_fl,
                fermion: hc.fermion,
                backward: hc.backward,
            }];
            let mut mat = vec![vec![Complex::ZERO; ntau + 1]; 2];
            add_gf_mat_oca(&hs, &solver.pp_g, &verts, &one, None, beta, kt as i64, &integ, &mut mat);
            let channels = channels_with_bath(&one, None);
            let diagrams = build_gf_diagrams_oca_channels(&hs, &verts, &channels);
            let mut rt: Vec<GreenTimeStep> = verts
                .iter()
                .map(|_| GreenTimeStep::new(&contour, 0, Sign::Fermi, Complex::ZERO))
                .collect();
            for d in &diagrams {
                add_gf_tstp_oca(0, d, &solver.pp_g, beta, h, kt as i64, &integ, &mut rt[d.vertex]);
            }
            for v in 0..2 {
                for m in 0..=ntau {
                    let lm = rt[v].lmix.raw_block_at(0, (ntau - m) as i64)[0];
                    let want = Complex::new(0.0, -mat[v][m].re);
                    assert!(
                        (lm - want).abs() < 1e-12,
                        "channel {ci} v{v} m{m}: lmix {lm} vs -i G^M_oca {want}"
                    );
                }
            }
        }
    }
}
