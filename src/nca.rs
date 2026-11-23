// Project : nQIST
// File    : src/nca.rs
// Purpose : NCA pseudo-particle self-energy and single-particle
//           Green's function diagrams (blueprints ppsc/nca/{sigma,gf}.cpp)
//
// Layered prefactor structure, exactly like the C++ dispatch: the
// kernels below carry their own factor (`i` for the self-energy,
// `i * sig_b` for the GF), and the diagram `prefactor` (`i*stat_sign`
// resp. `i*sig_b`) is applied on top. Do not fold the layers.
//
// Both the Matsubara and the real-time kernels are implemented here
// (the real-time ones follow below the component accessors).

use crate::hilbert::HilbertSpace;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::types::{Contour, ContourGreen, GreenTimeStep, PseudoGreen, Sign};

// ------------------------------------------------------------------
// component accessors (mam view semantics, probes 5b/5c verified)
// ------------------------------------------------------------------

/// Adjoints a raw `s1 x s2` block: `out[j*s1 + i] = conj(src[i*s2 + j])`.
pub(crate) fn adjoint_raw(src: &[Complex], s1: usize, s2: usize) -> Vec<Complex> {
    let mut out = vec![Complex::ZERO; src.len()];
    for i in 0..s1 {
        for j in 0..s2 {
            out[j * s1 + i] = src[i * s2 + j].conj();
        }
    }
    out
}

/// `Delta^R(n, j)` for `j <= n` (direct stored read).
pub(crate) fn delta_ret(g: &ContourGreen, n: i64, j: i64) -> DenseMat {
    DenseMat::from_slice(g.size1, g.size2, g.ret.raw_block_at(n, j))
}

/// `Delta^<(j, n)` for `j <= n` (direct stored read).
pub(crate) fn delta_les(g: &ContourGreen, j: i64, n: i64) -> DenseMat {
    DenseMat::from_slice(g.size1, g.size2, g.less.raw_block_at(j, n))
}

/// Greater component of a regular contour Green's function, mam
/// `gtr` view: `G^>(n, j) = G^R(n, j) - [G^<(j, n)]^dagger` for
/// `j < n`, `G^R + G^<` at `j == n` (out-of-triangle les reads carry
/// the hardcoded minus of the hermitian fallback).
pub(crate) fn delta_gtr(g: &ContourGreen, n: i64, j: i64) -> DenseMat {
    let r = delta_ret(g, n, j);
    if j == n {
        let l = delta_les(g, n, n);
        let mut out = r.clone();
        out.axpy(1.0, &l);
        out
    } else {
        let l = delta_les(g, j, n);
        let mut out = r.clone();
        for a in 0..g.size1 {
            for b in 0..g.size2 {
                let sub = -l.get(b, a).conj();
                out.set(a, b, out.get(a, b) + sub);
            }
        }
        out
    }
}

/// `Delta^tv(n, m)` (direct lmix-row read; the KB lmix component IS
/// the NESSi tv storage).
pub(crate) fn delta_tv(g: &ContourGreen, n: i64, m: i64) -> DenseMat {
    DenseMat::from_slice(g.size1, g.size2, g.lmix.raw_block_at(n, m))
}

/// One hybridization channel entering the pseudo-particle diagrams
/// (blueprint `pp_int_type`): `op1 = op_cre_fl` (creation), `op2 =
/// op_ann_fl` (annihilation), `fermion` the channel statistics,
/// `backward` the propagation direction.
pub struct Hybridization<'a> {
    pub delta: &'a ContourGreen,
    pub op_cre_fl: usize,
    pub op_ann_fl: usize,
    pub fermion: bool,
    pub backward: bool,
}

/// Single-particle Green's function vertex (blueprint `gf_vert_type`,
/// `(orb, spin, op1 = ann, op2 = cre)`).
pub struct GfVertex {
    pub op_ann_fl: usize,
    pub op_cre_fl: usize,
}

/// One NCA self-energy diagram: sector chain `ss --op2(ann)--> sa
/// --op1(cre)--> ss` with propagator `G_a = pp_g[sa]`, the channel
/// hybridization `delta`, and the dispatch prefactor `i * stat_sign`
/// (`stat_sign = -1` for backward fermionic hybridization,
/// Eckstein-Werner PRB 82, 115155).
pub(crate) struct SigmaDiagram<'a> {
    /// Self-energy sector `ss` this diagram contributes to.
    pub out: usize,
    /// Adjacent propagator sector `sa`.
    pub sa: usize,
    pub delta: &'a ContourGreen,
    pub op1: DenseMat,
    pub op2: DenseMat,
    pub prefactor: Complex,
}

/// Build all valid NCA self-energy diagrams (blueprint
/// `build_all_sigma_diagrams`: sector loop outer, hybridization loop
/// inner).
pub(crate) fn build_sigma_diagrams<'a>(
    hs: &HilbertSpace,
    pp_ints: &[Hybridization<'a>],
) -> Vec<SigmaDiagram<'a>> {
    let mut diagrams = Vec::new();
    for ss in 0..hs.ns() {
        for hyb in pp_ints {
            // sa = op2 (annihilation) applied to ss; closure op1 (creation) back to ss
            let Some((sa, op2)) = hs.ops[hyb.op_ann_fl].block(ss) else {
                continue;
            };
            let Some((ss_ref, op1)) = hs.ops[hyb.op_cre_fl].block(*sa) else {
                continue;
            };
            if *ss_ref != ss {
                continue;
            }
            let stat_sign = if hyb.fermion && hyb.backward {
                -1.0
            } else {
                1.0
            };
            let prefactor = Complex::I * Complex::new(stat_sign, 0.0);
            diagrams.push(SigmaDiagram {
                out: ss,
                sa: *sa,
                delta: hyb.delta,
                op1: op1.clone(),
                op2: op2.clone(),
                prefactor,
            });
        }
    }
    diagrams
}

/// Set the Matsubara component of all pseudo-particle self-energies
/// from the NCA diagrams (blueprint `set_pp_self_energy` at
/// `tstp = -1`): `Sigma^M_s(m) = sum over diagrams of prefactor *
/// [ i * Delta^M(m)(0,0) * (op1 G^M_a(m) op2) ]`, `m = 0..=ntau`
/// (set semantics: cleared first, accumulated diagram by diagram in
/// build order).
pub(crate) fn set_sigma_mat(
    hs: &HilbertSpace,
    pp_g: &[PseudoGreen],
    pp_ints: &[Hybridization],
    pp_sigma: &mut [PseudoGreen],
) {
    let diagrams = build_sigma_diagrams(hs, pp_ints);
    for s in 0..hs.ns() {
        for m in 0..=pp_sigma[s].ntau {
            for z in pp_sigma[s].mat.block_mut(m) {
                *z = Complex::ZERO;
            }
        }
    }
    accumulate_sigma_mat(&diagrams, pp_g, pp_sigma);
}

/// Kernel of [`set_sigma_mat`]: accumulate the Matsubara
/// self-energy contributions of the given diagrams (no clearing).
pub(crate) fn accumulate_sigma_mat(
    diagrams: &[SigmaDiagram],
    pp_g: &[PseudoGreen],
    pp_sigma: &mut [PseudoGreen],
) {
    for d in diagrams {
        let ntau = pp_g[d.sa].ntau;
        let dim_out = pp_sigma[d.out].size1;
        for m in 0..=ntau {
            // kernel: i * Delta^M(m)(0,0) * (op1 G^M_a(m) op2)
            let lam = d.delta.mat.block(m)[0];
            let ga =
                DenseMat::from_slice(pp_g[d.sa].size1, pp_g[d.sa].size2, pp_g[d.sa].mat.block(m));
            let prod = d.op1.mat_mul(&ga).mat_mul(&d.op2);
            for i in 0..dim_out {
                for j in 0..dim_out {
                    let kernel = Complex::I * lam * prod.get(i, j);
                    let idx = i * dim_out + j;
                    let old = pp_sigma[d.out].mat.block(m)[idx];
                    pp_sigma[d.out].mat.block_mut(m)[idx] = old + d.prefactor * kernel;
                }
            }
        }
    }
}

/// One NCA single-particle GF diagram: sector chain `sa --op1(ann)-->
/// sb --op2(cre)--> sa` with propagators `G_a = pp_g[sa]`, `G_b =
/// pp_g[sb]` and dispatch prefactor `i * sig_b` (the statistics sign
/// of sector `sb`).
pub(crate) struct GfDiagram {
    pub vertex: usize,
    pub sa: usize,
    pub sb: usize,
    pub op1: DenseMat,
    pub op2: DenseMat,
    pub prefactor: Complex,
}

/// Build all valid NCA GF diagrams (blueprint
/// `build_all_gf_diagrams`: vertex loop outer, `sa` loop inner).
pub(crate) fn build_gf_diagrams(hs: &HilbertSpace, verts: &[GfVertex]) -> Vec<GfDiagram> {
    let mut diagrams = Vec::new();
    for (vertex, v) in verts.iter().enumerate() {
        for sa in 0..hs.ns() {
            let Some((sb, op1)) = hs.ops[v.op_ann_fl].block(sa) else {
                continue;
            };
            let Some((sa_ref, op2)) = hs.ops[v.op_cre_fl].block(*sb) else {
                continue;
            };
            if *sa_ref != sa {
                continue;
            }
            let sig_b = hs.sectors[*sb].sig.as_i64() as f64;
            let prefactor = Complex::I * Complex::new(sig_b, 0.0);
            diagrams.push(GfDiagram {
                vertex,
                sa,
                sb: *sb,
                op1: op1.clone(),
                op2: op2.clone(),
                prefactor,
            });
        }
    }
    diagrams
}

/// Matsubara component of the NCA single-particle Green's functions
/// per vertex (blueprint `evaluate_to_out_idx` at `tstp = -1`):
/// `G^M(m) = sum over diagrams of prefactor * [ i*sig_b * tr(op1
/// G^M_a(m) op2 G^M_b(ntau - m)) ]`, `m = 0..=ntau`.
pub(crate) fn gf_mat(
    hs: &HilbertSpace,
    pp_g: &[PseudoGreen],
    verts: &[GfVertex],
) -> Vec<Vec<Complex>> {
    let diagrams = build_gf_diagrams(hs, verts);
    let ntau = pp_g.first().map(|g| g.ntau).unwrap_or(0);
    let mut out = vec![vec![Complex::ZERO; ntau + 1]; verts.len()];
    for d in &diagrams {
        let sig_b = hs.sectors[d.sb].sig.as_i64() as f64;
        for m in 0..=ntau {
            let ga =
                DenseMat::from_slice(pp_g[d.sa].size1, pp_g[d.sa].size2, pp_g[d.sa].mat.block(m));
            let gb = DenseMat::from_slice(
                pp_g[d.sb].size1,
                pp_g[d.sb].size2,
                pp_g[d.sb].mat.block(ntau - m),
            );
            let prod = d.op1.mat_mul(&ga).mat_mul(&d.op2).mat_mul(&gb);
            let kernel = Complex::I * Complex::new(sig_b, 0.0) * prod.trace();
            out[d.vertex][m] += d.prefactor * kernel;
        }
    }
    out
}

// ------------------------------------------------------------------
// real-time kernels (tstp >= 0)
// ------------------------------------------------------------------

/// Pseudo-particle greater component: the ret slot at `(n, j)`.
pub(crate) fn pp_greater(pp: &PseudoGreen, n: i64, j: i64) -> DenseMat {
    DenseMat::from_slice(pp.size1, pp.size2, pp.ret.raw_block_at(n, j))
}

/// Pseudo-particle lesser component `(j, n)`, `j <= n` (stored).
pub(crate) fn pp_les(pp: &PseudoGreen, j: i64, n: i64) -> DenseMat {
    DenseMat::from_slice(pp.size1, pp.size2, pp.less.raw_block_at(j, n))
}

/// Pseudo-particle tv component `(n, m)` (lmix row read).
pub(crate) fn pp_tv(pp: &PseudoGreen, n: i64, m: i64) -> DenseMat {
    DenseMat::from_slice(pp.size1, pp.size2, pp.lmix.raw_block_at(n, m))
}

/// Pseudo-particle vt component, mam `vt_from_tv` view:
/// `G^vt(m, n) = -sig * [G^tv(n, ntau - m)]^dagger`.
pub(crate) fn pp_vt(pp: &PseudoGreen, m: i64, n: i64) -> DenseMat {
    let src = adjoint_raw(
        pp.lmix.raw_block_at(n, pp.ntau as i64 - m),
        pp.size1,
        pp.size2,
    );
    let mut out = DenseMat::from_slice(pp.size1, pp.size2, &src);
    out.scale(-(pp.sign.as_i64() as f64));
    out
}

/// Set the real-time components of the pseudo-particle self-energies
/// at `tstp = n >= 0` from the NCA diagrams (blueprint
/// `set_pp_self_energy` at `tstp >= 0`, kernels `sdiagram_tstp`).
/// Unlike the Matsubara kernel the real-time kernels carry NO
/// explicit `i`; only the dispatch prefactor `i * stat_sign`
/// multiplies them: non-equal-time `Sigma^>(n,j) = pref *
/// [Delta^>(n,j) (op1 G^>_a(n,j) op2)]`, lesser `Sigma^<(j,n) =
/// pref * [Delta^<(j,n) (op1 G^<_a(j,n) op2)]`, mixed `Sigma^tv(n,m)
/// = pref * [Delta^tv(n,m) (op1 G^tv_a(n,m) op2)]`, and the
/// equal-time specials `Sigma^>(n,n) = pref * [(-Delta^R(n,n)^dag +
/// Delta^<(n,n)) (op1 G^>_a(n,n) op2)]`, `Sigma^<(n,n) = pref *
/// [(-Delta^<(n,n))^dag (op1 G^<_a(n,n) op2)]`. Set semantics (fresh
/// timestep: rows zeroed first, accumulated per diagram).
pub(crate) fn set_sigma_tstp(
    n: i64,
    hs: &HilbertSpace,
    pp_g: &[PseudoGreen],
    pp_ints: &[Hybridization],
    pp_sigma: &mut [PseudoGreen],
) {
    let diagrams = build_sigma_diagrams(hs, pp_ints);
    // fresh timestep: zero the touched rows
    for s in 0..hs.ns() {
        for j in 0..=n {
            for z in pp_sigma[s].ret.raw_block_at_mut(n, j) {
                *z = Complex::ZERO;
            }
            for z in pp_sigma[s].less.raw_block_at_mut(j, n) {
                *z = Complex::ZERO;
            }
        }
        for m in 0..=(pp_sigma[s].ntau as i64) {
            for z in pp_sigma[s].lmix.raw_block_at_mut(n, m) {
                *z = Complex::ZERO;
            }
        }
    }
    accumulate_sigma_tstp(n, &diagrams, pp_g, pp_sigma);
}

/// Kernel of [`set_sigma_tstp`]: accumulate the real-time
/// self-energy contributions of the given diagrams into row `n` (no
/// row clearing).
pub(crate) fn accumulate_sigma_tstp(
    n: i64,
    diagrams: &[SigmaDiagram],
    pp_g: &[PseudoGreen],
    pp_sigma: &mut [PseudoGreen],
) {
    for d in diagrams {
        let dim_out = pp_sigma[d.out].size1;
        let ntau = pp_g[d.sa].ntau as i64;
        let ga = &pp_g[d.sa];
        let lam = d.delta;
        // non-equal-time ret (greater) and lesser
        for j in 0..n {
            let l = delta_gtr(lam, n, j).get(0, 0);
            let prod = d.op1.mat_mul(&pp_greater(ga, n, j)).mat_mul(&d.op2);
            for i in 0..dim_out {
                for k in 0..dim_out {
                    let add = d.prefactor * l * prod.get(i, k);
                    let idx = pp_sigma[d.out].ret.idx(n, j);
                    pp_sigma[d.out].ret.block_mut(idx)[i * dim_out + k] += add;
                }
            }
            let l = delta_les(lam, j, n).get(0, 0);
            let prod = d.op1.mat_mul(&pp_les(ga, j, n)).mat_mul(&d.op2);
            for i in 0..dim_out {
                for k in 0..dim_out {
                    let add = d.prefactor * l * prod.get(i, k);
                    let idx = pp_sigma[d.out].less.idx(j, n);
                    pp_sigma[d.out].less.block_mut(idx)[i * dim_out + k] += add;
                }
            }
        }
        // mixed component
        for m in 0..=ntau {
            let l = delta_tv(lam, n, m).get(0, 0);
            let prod = d.op1.mat_mul(&pp_tv(ga, n, m)).mat_mul(&d.op2);
            for i in 0..dim_out {
                for k in 0..dim_out {
                    let add = d.prefactor * l * prod.get(i, k);
                    pp_sigma[d.out].lmix.raw_block_at_mut(n, m)[i * dim_out + k] += add;
                }
            }
        }
        // equal-time retarded (greater slot): (-Delta^R(n,n)^dag + Delta^<(n,n))(0,0)
        {
            let r_nn = delta_ret(lam, n, n);
            let l_nn = delta_les(lam, n, n);
            let mut lam_et = DenseMat::zeros(lam.size1);
            for a in 0..lam.size1 {
                for b in 0..lam.size2 {
                    lam_et.set(a, b, l_nn.get(a, b) - r_nn.get(b, a).conj());
                }
            }
            let l = lam_et.get(0, 0);
            let prod = d.op1.mat_mul(&pp_greater(ga, n, n)).mat_mul(&d.op2);
            for i in 0..dim_out {
                for k in 0..dim_out {
                    let add = d.prefactor * l * prod.get(i, k);
                    let idx = pp_sigma[d.out].ret.idx(n, n);
                    pp_sigma[d.out].ret.block_mut(idx)[i * dim_out + k] += add;
                }
            }
        }
        // equal-time lesser: (-Delta^<(n,n))^dag (0,0)
        {
            let l = -delta_les(lam, n, n).get(0, 0).conj();
            let prod = d.op1.mat_mul(&pp_les(ga, n, n)).mat_mul(&d.op2);
            for i in 0..dim_out {
                for k in 0..dim_out {
                    let add = d.prefactor * l * prod.get(i, k);
                    let idx = pp_sigma[d.out].less.idx(n, n);
                    pp_sigma[d.out].less.block_mut(idx)[i * dim_out + k] += add;
                }
            }
        }
    }
}

/// A static boson bath (Holstein/Hirsch): the coupling-scaled free
/// phonon propagator with its backward partner and the diagonal
/// vertex blocks (e.g. the doublon operator `n_up n_dn` from
/// [`crate::boson::docc_blocks`]).
pub struct BosonBath {
    pub d0: ContourGreen,
    pub d0_cc: ContourGreen,
    pub blocks: Vec<(usize, DenseMat)>,
}

/// Build the bosonic NCA self-energy diagrams: per sector with a
/// vertex block, a forward and a backward channel, each a same-sector
/// chain `ss --vertex--> ss` with statistics prefactor `i * (+1)`
/// (boson hybridizations carry no backward minus; blueprint
/// `construct_sigma_diagram` with `sig = +1`).
pub(crate) fn build_boson_diagrams(bath: &BosonBath) -> Vec<SigmaDiagram<'_>> {
    let mut out = Vec::new();
    for delta in [&bath.d0, &bath.d0_cc] {
        for (s, blk) in &bath.blocks {
            out.push(SigmaDiagram {
                out: *s,
                sa: *s,
                delta,
                op1: blk.clone(),
                op2: blk.clone(),
                prefactor: Complex::I,
            });
        }
    }
    out
}

/// Add the bosonic Matsubara self-energy contributions on top of the
/// fermionic set.
pub(crate) fn add_boson_sigma_mat(
    bath: &BosonBath,
    pp_g: &[PseudoGreen],
    pp_sigma: &mut [PseudoGreen],
) {
    accumulate_sigma_mat(&build_boson_diagrams(bath), pp_g, pp_sigma);
}

/// Add the bosonic real-time self-energy contributions of row `tstp`
/// on top of the fermionic set.
pub(crate) fn add_boson_sigma_tstp(
    tstp: i64,
    bath: &BosonBath,
    pp_g: &[PseudoGreen],
    pp_sigma: &mut [PseudoGreen],
) {
    accumulate_sigma_tstp(tstp, &build_boson_diagrams(bath), pp_g, pp_sigma);
}

/// Real-time NCA single-particle Green's functions per vertex at
/// `tstp = n >= 0` (blueprint `evaluate_to_out_idx` at `tstp >= 0`,
/// kernels `gdiagram_tstp`): per diagram
/// `G^<(j,n) = tr(op1 G^<_a(j,n) op2 G^R_b(n,j))`,
/// `G^>(n,j) = tr(op1 G^>_a(n,j) op2 G^<_b(j,n))` with
/// `G^R(n,j) = G^>(n,j) - [G^<(j,n)]^dag`, and
/// `G^tv(n,m) = tr(op1 G^tv_a(n,m) op2 G^vt_b(m,n))`, all times the
/// dispatch prefactor `i * sig_b`.
pub(crate) fn gf_tstp(
    c: &Contour,
    n: i64,
    hs: &HilbertSpace,
    pp_g: &[PseudoGreen],
    verts: &[GfVertex],
) -> Vec<GreenTimeStep> {
    let diagrams = build_gf_diagrams(hs, verts);
    let mut out: Vec<GreenTimeStep> = verts
        .iter()
        .map(|_| GreenTimeStep::new(c, n, Sign::Fermi, Complex::ZERO))
        .collect();
    for d in &diagrams {
        let sig_b = hs.sectors[d.sb].sig.as_i64() as f64;
        let pref = Complex::I * Complex::new(sig_b, 0.0);
        let ga = &pp_g[d.sa];
        let gb = &pp_g[d.sb];
        let g = &mut out[d.vertex];
        // per-diagram evaluation (blueprint: fresh timestep per
        // diagram, incr into the output): the retarded kernel pairs
        // with THIS diagram's unfactored lesser trace
        let mut les_kernel = vec![Complex::ZERO; (n + 1) as usize];
        for j in 0..=n {
            let prod = d
                .op1
                .mat_mul(&pp_les(ga, j, n))
                .mat_mul(&d.op2)
                .mat_mul(&pp_greater(gb, n, j));
            les_kernel[j as usize] = prod.trace();
            let idx = g.less.idx(j, n);
            g.less.block_mut(idx)[0] += pref * les_kernel[j as usize];
        }
        // retarded from greater and lesser: ret(n,j) = pref*(g_gtr - conj(les))
        for j in 0..=n {
            let prod = d
                .op1
                .mat_mul(&pp_greater(ga, n, j))
                .mat_mul(&d.op2)
                .mat_mul(&pp_les(gb, j, n));
            let gtr = prod.trace();
            let idx = g.ret.idx(n, j);
            g.ret.block_mut(idx)[0] += pref * (gtr - les_kernel[j as usize].conj());
        }
        // mixed component
        for m in 0..=(g.ntau as i64) {
            let prod = d
                .op1
                .mat_mul(&pp_tv(ga, n, m))
                .mat_mul(&d.op2)
                .mat_mul(&pp_vt(gb, m, n));
            g.lmix.raw_block_at_mut(0, m)[0] += pref * prod.trace();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hilbert::{SectorClass, build};
    use kadanoffbaym::types::{Contour, Sign};

    /// 1x1 fermionic contour Green's function filled with a known
    /// Matsubara profile `mat(m) = c * exp(-dtau * m * x)`.
    fn filled_delta(ntau: usize, beta: f64, c: f64, x: f64) -> ContourGreen {
        let contour = Contour::new(4, ntau as i64, 1, 1, 0.08, beta);
        let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        let dtau = beta / ntau as f64;
        for m in 0..=ntau {
            let v = c * (-dtau * m as f64 * x).exp();
            g.mat.block_mut(m)[0] = Complex::new(v, 0.0);
        }
        g
    }

    #[test]
    fn single_band_sigma_diagram_structure() {
        // benchmark sector layout: 0=(0,0) B, 1=(0,1) F, 2=(1,0) F, 3=(1,1) B
        let hs = build(1, SectorClass::SpinResolved);
        let delta = filled_delta(8, 5.0, 1.0, 1.0);
        let delta_cc = filled_delta(8, 5.0, 2.0, 1.0);
        let cua = hs.flavor(0, 0, false);
        let cuc = hs.flavor(0, 0, true);
        let cda = hs.flavor(0, 1, false);
        let cdc = hs.flavor(0, 1, true);
        let pp_ints = [
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
        let diagrams = build_sigma_diagrams(&hs, &pp_ints);
        // all eight diagrams of the single-band case, in build order
        // (ss major, channel minor): fwd-up (2,0),(3,1); bwd-up
        // (0,2),(1,3); fwd-dn (1,0),(3,2); bwd-dn (0,1),(2,3)
        let pairs: Vec<(usize, usize)> = diagrams.iter().map(|d| (d.out, d.sa)).collect();
        assert_eq!(
            pairs,
            vec![
                (0, 2),
                (0, 1),
                (1, 3),
                (1, 0),
                (2, 0),
                (2, 3),
                (3, 1),
                (3, 2)
            ]
        );
        // prefactors: fwd +i, bwd -i (fermion && backward); the bwd
        // channels are the delta_cc ones: (0,2),(0,1),(1,3),(2,3)
        for d in &diagrams {
            let is_bwd = matches!((d.out, d.sa), (0, 2) | (0, 1) | (1, 3) | (2, 3));
            assert_eq!(
                d.prefactor,
                if is_bwd { -Complex::I } else { Complex::I },
                "out = {} sa = {}",
                d.out,
                d.sa
            );
        }
        // all operator blocks are 1x1 with element +-1
        for d in &diagrams {
            let z = d.op1.get(0, 0);
            assert!(z == Complex::ONE || z == -Complex::ONE);
        }
    }

    #[test]
    fn sigma_mat_pointwise_product() {
        // Sigma^M(m) = prefactor * i * Delta^M(m) * (op1 G_a^M(m) op2):
        // with unit operator blocks and G_a = -exp(-tau (E - mu)),
        // sector 2 (fwd, prefactor i) gets Sigma = i*i*Delta*G_a = -Delta*G_a
        let hs = build(1, SectorClass::SpinResolved);
        let ntau = 8;
        let beta = 5.0;
        let delta = filled_delta(ntau, beta, 0.3, 0.7);
        let contour = Contour::new(4, ntau as i64, 1, 1, 0.08, beta);
        let mut pp_g = Vec::new();
        let mut pp_sigma = Vec::new();
        for s in 0..4 {
            pp_g.push(PseudoGreen::new(&contour, hs.sectors[s].sig, Complex::ZERO));
            pp_sigma.push(PseudoGreen::new(&contour, hs.sectors[s].sig, Complex::ZERO));
        }
        let e = -1.5;
        let mu = -1.6387400224070612;
        let dtau = beta / ntau as f64;
        for m in 0..=ntau {
            let g = -(dtau * m as f64 * (e - mu)).exp();
            pp_g[0].mat.block_mut(m)[0] = Complex::new(g, 0.0);
        }
        let cua = hs.flavor(0, 0, false);
        let cuc = hs.flavor(0, 0, true);
        let pp_ints = [Hybridization {
            delta: &delta,
            op_cre_fl: cuc,
            op_ann_fl: cua,
            fermion: true,
            backward: false,
        }];
        set_sigma_mat(&hs, &pp_g, &pp_ints, &mut pp_sigma);
        // only sectors 2 and 3 receive contributions from the fwd up channel
        for m in 0..=ntau {
            let want = -(delta.mat.block(m)[0] * pp_g[0].mat.block(m)[0]);
            assert!(
                (pp_sigma[2].mat.block(m)[0] - want).abs() < 1e-15,
                "m = {m}"
            );
            assert_eq!(pp_sigma[0].mat.block(m)[0], Complex::ZERO);
        }
    }

    #[test]
    fn gf_mat_two_diagrams_sum() {
        // benchmark up vertex: diagrams (sa=2, sb=0) and (sa=3, sb=1);
        // kernel per diagram: prefactor * i * sig_b * tr(op1 G_a(m) op2 G_b(N-m))
        // with unit ops: tr = G_a(m) * G_b(N-m); sig_b = +1 for sb=0,
        // -1 for sb=1; prefactor = i*sig_b -> net per diagram -G_a*G_b
        let hs = build(1, SectorClass::SpinResolved);
        let ntau = 8;
        let beta = 5.0;
        let contour = Contour::new(4, ntau as i64, 1, 1, 0.08, beta);
        let mut pp_g = Vec::new();
        for s in 0..4 {
            pp_g.push(PseudoGreen::new(&contour, hs.sectors[s].sig, Complex::ZERO));
        }
        let (e0, e1, e2, e3) = (0.0, -1.5, -1.5, 0.0);
        let mu = -1.6387400224070612;
        let dtau = beta / ntau as f64;
        for m in 0..=ntau {
            let boltz = |e: f64| -(dtau * m as f64 * (e - mu)).exp();
            pp_g[0].mat.block_mut(m)[0] = Complex::new(boltz(e0), 0.0);
            pp_g[1].mat.block_mut(m)[0] = Complex::new(boltz(e1), 0.0);
            pp_g[2].mat.block_mut(m)[0] = Complex::new(boltz(e2), 0.0);
            pp_g[3].mat.block_mut(m)[0] = Complex::new(boltz(e3), 0.0);
        }
        let cua = hs.flavor(0, 0, false);
        let cuc = hs.flavor(0, 0, true);
        let verts = [GfVertex {
            op_ann_fl: cua,
            op_cre_fl: cuc,
        }];
        let out = gf_mat(&hs, &pp_g, &verts);
        assert_eq!(out.len(), 1);
        for m in 0..=ntau {
            let g = |s: usize| pp_g[s].mat.block(m)[0];
            let gb = |s: usize, mm: usize| pp_g[s].mat.block(ntau - mm)[0];
            // net -[G_2(m) G_0(N-m) + G_3(m) G_1(N-m)]
            let want = -(g(2) * gb(0, m) + g(3) * gb(1, m));
            assert!((out[0][m] - want).abs() < 1e-14, "m = {m}");
        }
    }
}
