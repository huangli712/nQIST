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

/// `out = a * m * b`: sandwich of the square matrix `m (k x k)`.
fn sandwich(
    a: &[Complex],
    m: &[Complex],
    b: &[Complex],
    d1: usize,
    dk: usize,
    d2: usize,
    out: &mut [Complex],
    scratch: &mut [Complex],
) {
    mul_into(a, m, d1, dk, dk, scratch);
    mul_into(scratch, b, dk, dk, d2, out);
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
    fn block<'x>(&'x self, hs: &'x HilbertSpace, sector: usize) -> Option<(usize, &'x DenseMat)> {
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
/// Fermion-only convenience wrapper (test callers).
#[cfg(test)]
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
    let mut scratch = vec![Complex::ZERO; b_gb.max(b_ac).max(b_ss)];
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
    let gd_sig = |d: usize| pp_g[d].sign.as_i64() as f64;

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
        // gd_flip(tau) = -sig_gd * gd(nmax - tau) in equilibrium
        let gd_flip =
            |tau: usize| -> Complex { -Complex::new(gd_sig(d.sd), 0.0) * gd.block(nmax - tau)[0] };
        // dimension bookkeeping (single-band: all 1x1)
        let (adim, cdim) = (ga.d1, gc.d1);
        debug_assert_eq!(adim * cdim, 1);
        for m in 0..=nmax {
            let g: Complex = if m < k && m > 0 {
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
fn gregory_scalar(n: usize, v: &[Complex], integ: &Integrator) -> Complex {
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
fn poly_integral_scalar(v: &[Complex], i0: usize, i1: usize, integ: &Integrator) -> Complex {
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
fn conv2b_scalar(data: &[Complex], n11: usize, j: usize, nn: usize, integ: &Integrator) -> Complex {
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
fn lowside_scalar(j: usize, aa: &[Complex], bb: &[Complex], integ: &Integrator) -> Complex {
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
fn sigma_mat_core(
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
    let pref = Complex::new(d.prefactor, 0.0);

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
