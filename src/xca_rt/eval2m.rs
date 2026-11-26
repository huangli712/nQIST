// Project : nQIST (real-time XCA, multi-band)
// File    : src/xca_rt/eval2m.rs
// Purpose : order-2 real-time XCA evaluator for MATRIX blocks
//           (norb >= 2): the ten component families of the scalar
//           `eval2` (ret1, vt1, les1, vt2, les2, vt3, les3..les6)
//           mirrored line-for-line onto the stock OCA matrix
//           primitives (sandwich / psc / pls / ptv / ptv_dir blocks,
//           gregory_blk / conv2b_blk / lowside_blk quadrature,
//           sigma_mat_core_blk), with every raw hybridization read
//           (sgtr / svt / sles / stv) replaced by the LineSep
//           pole-separated reconstruction. The line factors are
//           SCALAR (the deltas are scalar per flavor channel); only
//           the assembly is matrix-valued.
//
//           Anchored by `xca2_matches_oca_realtime_twoband` against
//           `oca::add_sigma_tstp_oca_matrix` (the PPSC-gated stock);
//           the scalar eval2 stays the single-band production route,
//           bit-identical (this file is reached only through the
//           size dispatch in `add_sigma_tstp_xca2`).

use super::eval2::{
    FAM_LES1, FAM_LES2, FAM_LES3, FAM_LES4, FAM_LES5, FAM_LES6, FAM_RET1, FAM_VT1, FAM_VT2,
    FAM_VT3, LineSep,
};

/// Full-family entry (all ten families; see `_fam` for the bitmask).
pub(crate) fn add_sigma_tstp_xca2m(
    n: i64,
    d: &OcaSigmaDiagram,
    pp_g: &[PseudoGreen],
    beta: f64,
    h: f64,
    kt: i64,
    integ: &Integrator,
    fit1: &CsoeFit,
    fit2: &CsoeFit,
    pp_sigma: &mut [PseudoGreen],
) {
    add_sigma_tstp_xca2m_fam(n, d, pp_g, beta, h, kt, integ, fit1, fit2, pp_sigma, !0u32)
}
use super::fit::CsoeFit;
use super::tv_c;
use crate::oca::OcaSigmaDiagram;
use crate::oca::{
    conv2b_blk, gregory_blk, lowside_blk, pls_blk_into, psc_blk_into, ptv_blk_into, ptv_dir_blk,
    sandwich, sigma_mat_core_blk,
};
use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::PseudoGreen;
use kadanoffbaym::weights::Integrator;

#[inline]
fn phase(w: f64, a: i64, b: i64, h: f64) -> Complex {
    (-Complex::I * w * h * (a - b) as f64).exp()
}

/// Order-2 separated evaluator, matrix path. Adds the ret / lmix /
/// less blocks of `pp_sigma` for one diagram at timestep `n`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_sigma_tstp_xca2m_fam(
    n: i64,
    d: &OcaSigmaDiagram,
    pp_g: &[PseudoGreen],
    beta: f64,
    h: f64,
    kt: i64,
    integ: &Integrator,
    fit1: &CsoeFit,
    fit2: &CsoeFit,
    pp_sigma: &mut [PseudoGreen],
    fam: u32,
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
    let nac = adim * cdim;
    let nab = adim * bdim;
    let pref = d.prefactor;
    let sig = d.sig;
    let hh = Complex::new(h * h, 0.0);
    let out = &mut pp_sigma[d.out];
    let sdim = out.size1 as usize;
    let nss = sdim * sdim;
    let mnp = (mmax + 1) as usize;
    let nsamp = n1 as usize;
    let smax = na.max(nb).max(nc).max(nab).max(nbc).max(nac).max(nss);
    let mut scratch = vec![Complex::ZERO; smax];
    let mut gab = vec![Complex::ZERO; na];
    let mut gbb = vec![Complex::ZERO; nb];
    let mut gcb = vec![Complex::ZERO; nc];
    let mut t_ab = vec![Complex::ZERO; nab];
    let mut t_bc = vec![Complex::ZERO; nbc];
    let mut t_ac = vec![Complex::ZERO; nac];
    let mut op1_conv = vec![Complex::ZERO; nss];
    let ls1 = LineSep::new(lam1, fit1, h);
    let ls2 = LineSep::new(lam2, fit2, h);

    // conj-transpose scatter of the final sandwich into lmix
    // (the vt-family convention, identical to the stock matrix
    // kernel); takes the product slice as a parameter so it captures
    // nothing (the sandwich calls below reuse op1_conv mutably)
    let scatter_vt = |blk: &mut [Complex], val: Complex, prod: &[Complex]| {
        let dim_l = (blk.len() as f64).sqrt() as usize;
        for r in 0..dim_l {
            for c in 0..dim_l {
                blk[r * dim_l + c] += val * prod[c * dim_l + r].conj();
            }
        }
    };

    let mut data_bc = vec![Complex::ZERO; n11 * n11 * nbc];
    let mut data_ac_grid = vec![Complex::ZERO; n11 * n11 * nac];
    let mut f_ac = vec![Complex::ZERO; n11 * nac];
    let mut gab_vt = vec![Complex::ZERO; na];
    let mut data2 = vec![Complex::ZERO; n11 * mnp * nab];
    if fam & FAM_RET1 != 0 {
        // ---- ret (1): ls1.gtr(n, i2) x [Ga^>(n,i1) op2 Gb^>(i1,i2)]
        //      x ls2.gtr(i1,j) op3 Gc^>(i2,j), causal-triangle conv2b per j
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
                let s = ls1.gtr(n, i2, nsamp);
                let base = (i1 as usize) * n11 * nab + (i2 as usize) * nab;
                data_ab[base..base + nab].copy_from_slice(&t_ab);
                for q in 0..nab {
                    data_ab[base + q] *= s;
                }
            }
        }
        for j in (0..=n).rev() {
            let i1min = j.min(n1 - kt);
            let mut data_ac = vec![Complex::ZERO; n11 * n11 * nac];
            for i1 in i1min..=n1 {
                let i2max = (i1 + kt).min(n1);
                let l2 = ls2.gtr(i1, j, nsamp);
                for i2 in i1min..=i2max {
                    psc_blk_into(gc, i2, j, cdim, &mut gcb);
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
                    data_ac[base..base + nac].copy_from_slice(&t_ac);
                    for q in 0..nac {
                        data_ac[base + q] *= l2;
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
    }
    if fam & (FAM_VT1 | FAM_LES1) != 0 {
        // ---- data_bc shared by vt(1)/les(1):
        //      ls2.gtr(t2, n) x [Gb^>(t2,t1) op3 Gc^>(t1,n)]
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
                let s = ls2.gtr(t2, n, nsamp);
                let base = (t1 as usize) * n11 * nbc + (t2 as usize) * nbc;
                data_bc[base..base + nbc].copy_from_slice(&t_bc);
                for q in 0..nbc {
                    data_bc[base + q] *= s;
                }
            }
        }
    }
    if fam & FAM_VT1 != 0 {
        // ---- vt (1): lmix(n, mmax - m) += pref * sig h^2 * (op1 c op4)^{dag}
        for m in 0..=mmax {
            for i2 in 0..=n1 {
                let i1max = (i2 + kt).min(n1);
                let lam1_vt = ls1.vt(m, i2, nsamp);
                for i1 in 0..=i1max {
                    ptv_blk_into(ga, m, i1, adim, &mut gab_vt);
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
                    data_ac_grid[base..base + nac].copy_from_slice(&t_ac);
                    for q in 0..nac {
                        data_ac_grid[base + q] *= lam1_vt;
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
            scatter_vt(blk, val, &op1_conv);
        }
    }
    if fam & FAM_LES1 != 0 {
        // ---- les (1): less(j, n) += pref * (-h^2) * op1 c op4
        for j in 0..=n {
            for i2 in 0..=n1 {
                let i1max = (i2 + kt).min(n1);
                let lam1_les = ls1.les(j, i2, nsamp);
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
                    data_ac_grid[base..base + nac].copy_from_slice(&t_ac);
                    for q in 0..nac {
                        data_ac_grid[base + q] *= lam1_les;
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
    }
    if fam & (FAM_VT2 | FAM_LES2) != 0 {
        // ---- vt (2) / les (2) shared data2: ls2.vt(tau, n) x [op2 Gb^vt(tau, t)]
        for tau in 0..=mmax {
            let l2 = ls2.vt(tau, n, nsamp);
            for t in 0..=n1 {
                let mut pv = vec![Complex::ZERO; nb];
                ptv_blk_into(gb, tau, t, bdim, &mut pv);
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
    }
    if fam & FAM_VT2 != 0 {
        // ---- vt (2)
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
                let s = ls1.vt(m, i2, nsamp);
                let base = (i2 as usize) * nac;
                f_ac[base..base + nac].copy_from_slice(&t_ac);
                for q in 0..nac {
                    f_ac[base + q] *= s;
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
            scatter_vt(blk, val, &op1_conv);
        }
    }
    if fam & FAM_LES2 != 0 {
        // ---- les (2)
        for j in 0..=n {
            for i2 in 0..=n1 {
                let row = &data2[(i2 as usize) * mnp * nab..];
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
                let s = ls1.les(j, i2, nsamp);
                let base = (i2 as usize) * nac;
                f_ac[base..base + nac].copy_from_slice(&t_ac);
                for q in 0..nac {
                    f_ac[base + q] *= s;
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
    }
    if fam & FAM_VT3 != 0 {
        // ---- vt (3): the mat-core sandwich chain with the vt data of
        //      line 2 separated per pole (phase at t = n hoisted)
        {
            let mut gc_vt = vec![Complex::ZERO; mnp * nc];
            for tau in 0..=mmax {
                let mut pv = vec![Complex::ZERO; nc];
                ptv_blk_into(gc, tau, n, cdim, &mut pv);
                gc_vt[(tau as usize) * nc..(tau as usize + 1) * nc].copy_from_slice(&pv);
            }
            for m in 0..=mmax {
                let mut acc = vec![Complex::ZERO; nss];
                for l in 0..fit2.poles.len() {
                    // per-tau vt coefficients of line 2 at this pole
                    let mut prof = vec![Complex::ZERO; mnp];
                    for tau in 0..=mmax {
                        prof[tau as usize] = ls2.vt_coef_at(tau, l, nsamp);
                    }
                    let core = sigma_mat_core_blk(
                        m as usize,
                        kt as usize,
                        mmax as usize,
                        ga.mat.raw(),
                        gb.mat.raw(),
                        &gc_vt,
                        lam1.mat.raw(),
                        &prof,
                        adim,
                        bdim,
                        cdim,
                        sdim,
                        d,
                        integ,
                        &mut scratch,
                    );
                    let ph = phase(fit2.poles[l], 0, n, h);
                    for q in 0..nss {
                        acc[q] += ph * core[q];
                    }
                }
                let pref_v = pref * Complex::new(0.0, -sig * dtau * dtau);
                let blk = out.lmix.block_mut(out.lmix.idx(n, mmax - m));
                let dim_l = (blk.len() as f64).sqrt() as usize;
                for r in 0..dim_l {
                    for c in 0..dim_l {
                        blk[r * dim_l + c] += pref_v * acc[c * dim_l + r].conj();
                    }
                }
            }
        }
    }
    if fam & FAM_LES3 != 0 {
        // ---- les (3): two-level real-time gregory with les reads of both lines
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
                    let s = ls1.les(j, i2, nsamp);
                    let base = (i2 as usize) * nbc;
                    f_bc[base..base + nbc].copy_from_slice(&t_bc);
                    for q in 0..nbc {
                        f_bc[base + q] *= s;
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
                let s = ls2.les(i1, n, nsamp);
                let base = (i1 as usize) * nac;
                f_ac[base..base + nac].copy_from_slice(&t_ac);
                for q in 0..nac {
                    f_ac[base + q] *= s;
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
    }
    if fam & FAM_LES4 != 0 {
        // ---- les (4): stv reads of line 1 at fixed j (raw point values),
        //      Matsubara lowside with gb, vt of line 2 separated
        for j in 0..=n {
            let mut f_bc = vec![Complex::ZERO; mnp * nbc];
            for mu in 0..=mmax {
                let mut pv = vec![Complex::ZERO; nc];
                ptv_blk_into(gc, mu, n, cdim, &mut pv);
                let rb = &mut f_bc[(mu as usize) * nbc..];
                for r in 0..bdim {
                    for c in 0..cdim {
                        let mut acc = Complex::ZERO;
                        for q in 0..cdim {
                            acc += d.op3.get(r, q) * pv[q * cdim + c];
                        }
                        rb[r * cdim + c] = tv_c(lam1, j, mu) * acc;
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
                let s = ls2.vt(mu, n, nsamp);
                let base = (mu as usize) * nac;
                f_ac_m[base..base + nac].copy_from_slice(&t_ac);
                for q in 0..nac {
                    f_ac_m[base + q] *= s;
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
    }
    if fam & FAM_LES5 != 0 {
        // ---- les (5): greater reads of line 1 anchored at j, les of line
        //      2 anchored at n, conv2b over [0, j]
        {
            let mut data_bc5 = vec![Complex::ZERO; n11 * n11 * nbc];
            for i1 in 0..=n1 {
                let i2max = (i1 + kt).min(n1);
                let l2 = ls2.les(i1, n, nsamp);
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
                    let base = (i1 as usize) * n11 * nbc + (i2 as usize) * nbc;
                    data_bc5[base..base + nbc].copy_from_slice(&t_bc);
                    for q in 0..nbc {
                        data_bc5[base + q] *= l2;
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
                        let s = ls1.gtr(j, i2, nsamp);
                        let base = (i1 as usize) * n11 * nac + (i2 as usize) * nac;
                        data_ac5[base..base + nac].copy_from_slice(&t_ac);
                        for q in 0..nac {
                            data_ac5[base + q] *= s;
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
    }
    if fam & FAM_LES6 != 0 {
        // ---- les (6): tv of line 1 at fixed j (raw point values) times
        //      the tv/vt chain through gb/gc, les of line 2
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
                    let s = tv_c(lam1, j, mu);
                    let base = (mu as usize) * nbc;
                    f_bc[base..base + nbc].copy_from_slice(&t_bc);
                    for q in 0..nbc {
                        f_bc[base + q] *= s;
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
                let s = ls2.les(i1, n, nsamp);
                let base = (i1 as usize) * nac;
                f_ac[base..base + nac].copy_from_slice(&t_ac);
                for q in 0..nac {
                    f_ac[base + q] *= s;
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
}
