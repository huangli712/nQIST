// Project : nQIST (real-time XCA)
// File    : src/xca_rt/eval2.rs
// Purpose : order-2 real-time XCA evaluator: the five
//           component families of `oca::add_sigma_tstp_oca`
//           (ret, vt1, les1, vt2, les2) with every hybridization
//           read replaced by its pole-sum reconstruction
//
//           Each `lam` read (greater / vt / les component) is
//           expressed through the SAME pole basis fitted to the
//           greater component; vt/les coefficients are obtained by
//           projecting the line component onto that basis per
//           Matsubara/lesser anchor. With the quadrature mirrored
//           exactly (quad.rs), agreement with the reference kernel is
//           limited only by the reconstruction residual.
//
//           Prototype scope: scalar (1x1) blocks, steady input.

use super::fit::{CsoeFit, csoe_coeffs};
use super::quad::{conv2b_scalar, gregory_scalar, lowside_scalar, poly_integral_scalar};
use super::{gtr, les_c, les_pp, tv_c, tv_pp, vt_c, vt_pp};
use crate::oca::OcaSigmaDiagram;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{ContourGreen, PseudoGreen};
use kadanoffbaym::weights::Integrator;

#[inline]
fn phase(w: f64, a: i64, b: i64, h: f64) -> Complex {
    (-Complex::I * w * h * (a - b) as f64).exp()
}

/// Audit helper (XCA_INTDUMP instrument): complex as "re,im" with
/// full double precision for point-by-point machine comparison.
fn cz(z: Complex) -> String {
    format!("{:+.15e},{:+.15e}", z.re, z.im)
}

/// Pole-separated view of one hybridization line: the greater
/// coefficients come from the direct fit; the vt/les components are
/// projected onto the same basis per anchor, lazily.
pub(crate) struct LineSep<'a> {
    pub lam: &'a ContourGreen,
    pub fit: &'a CsoeFit,
    pub h: f64,
    /// cached vt coefficients per Matsubara anchor: `sum_l
    /// vt_coef[m][l] e^{-i w_l t h} = svt(lam, m, t)`
    vt_coef: std::cell::RefCell<Vec<Option<Vec<Complex>>>>,
    /// cached les coefficients per real-time anchor
    les_coef: std::cell::RefCell<Vec<Option<Vec<Complex>>>>,
}

impl<'a> LineSep<'a> {
    pub fn new(lam: &'a ContourGreen, fit: &'a CsoeFit, h: f64) -> Self {
        let nm = (lam.ntau + 1) as usize;
        LineSep {
            lam,
            fit,
            h,
            vt_coef: std::cell::RefCell::new(vec![None; nm]),
            les_coef: std::cell::RefCell::new(vec![None; nm.max(1)]),
        }
    }

    /// Raw pole reconstruction of the positive-direction greater
    /// values (the quantity the fit samples).
    #[inline]
    fn gtr_raw(&self, a: i64, b: i64) -> Complex {
        let mut acc = Complex::ZERO;
        for (l, &c) in self.fit.coefs.iter().enumerate() {
            acc += c * phase(self.fit.poles[l], a, b, self.h);
        }
        acc
    }

    /// greater read with `sgtr` semantics (ret - less* in triangle,
    /// hermitian continuation out of it): the reconstruction honours
    /// the continuation instead of naively phasing across zero.
    pub fn gtr(&self, a: i64, b: i64, nsamp: usize) -> Complex {
        // note: -gtr_raw(b, a)* = -(ret - less*)* = -ret* + less,
        // i.e. the hermitian continuation already carries the lesser
        // term - no extra les add
        if a > b {
            self.gtr_raw(a, b)
        } else if a == b {
            let l = self.les(a, a, nsamp);
            self.gtr_raw(a, a) + l + l.conj()
        } else {
            -self.gtr_raw(b, a).conj()
        }
    }

    /// Cached per-tau vt coefficient of pole `l` (projecting the vt
    /// component over the real-time axis).
    pub fn vt_coef_at(&self, tau: i64, l: usize, nsamp: usize) -> Complex {
        self.vt(tau, 0, nsamp); // ensures the cache row exists
        self.vt_coef.borrow()[tau as usize].clone().unwrap()[l]
    }

    /// vt read `vt_c(lam, m, t)` through the projected pole basis.
    pub fn vt(&self, m: i64, t: i64, nsamp: usize) -> Complex {
        let midx = m as usize;
        if self.vt_coef.borrow().len() <= midx {
            self.vt_coef.borrow_mut().resize(midx + 1, None);
        }
        if self.vt_coef.borrow()[midx].is_none() {
            let values: Vec<Complex> = (0..=nsamp as i64).map(|t| vt_c(self.lam, m, t)).collect();
            let coefs = csoe_coeffs(&self.fit.poles, self.h, &values);
            self.vt_coef.borrow_mut()[midx] = Some(coefs);
        }
        let coefs = self.vt_coef.borrow()[midx].clone().unwrap();
        let mut acc = Complex::ZERO;
        for (l, &c) in coefs.iter().enumerate() {
            acc += c * phase(self.fit.poles[l], t, 0, self.h);
        }
        acc
    }

    /// les read `les_c(lam, j, t)` through the projected pole basis.
    pub fn les(&self, j: i64, t: i64, nsamp: usize) -> Complex {
        let jidx = j as usize;
        if self.les_coef.borrow().len() <= jidx {
            self.les_coef.borrow_mut().resize(jidx + 1, None);
        }
        if self.les_coef.borrow()[jidx].is_none() {
            let values: Vec<Complex> = (0..=nsamp as i64).map(|t| les_c(self.lam, j, t)).collect();
            let coefs = csoe_coeffs(&self.fit.poles, self.h, &values);
            self.les_coef.borrow_mut()[jidx] = Some(coefs);
        }
        let coefs = self.les_coef.borrow()[jidx].clone().unwrap();
        let mut acc = Complex::ZERO;
        for (l, &c) in coefs.iter().enumerate() {
            acc += c * phase(self.fit.poles[l], t, 0, self.h);
        }
        acc
    }
}

/// Order-2 real-time self-energy through the pole-separated five
/// families; signature mirrors `oca::add_sigma_tstp_oca` plus the two
/// line separations. Adds ret / lmix / less blocks of `pp_sigma`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_sigma_tstp_xca2(
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
    add_sigma_tstp_xca2_fam(n, d, pp_g, beta, h, kt, integ, fit1, fit2, pp_sigma, !0u32)
}

/// Family bitmask bits (stage oracle surface).
pub(crate) const FAM_RET1: u32 = 1 << 0;
pub(crate) const FAM_VT1: u32 = 1 << 1;
pub(crate) const FAM_LES1: u32 = 1 << 2;
pub(crate) const FAM_VT2: u32 = 1 << 3;
pub(crate) const FAM_LES2: u32 = 1 << 4;
pub(crate) const FAM_VT3: u32 = 1 << 5;
pub(crate) const FAM_LES3: u32 = 1 << 6;
pub(crate) const FAM_LES4: u32 = 1 << 7;
pub(crate) const FAM_LES5: u32 = 1 << 8;
pub(crate) const FAM_LES6: u32 = 1 << 9;

pub(crate) fn add_sigma_tstp_xca2_fam(
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
    debug_assert_eq!(pp_sigma[d.out].size1 * pp_sigma[d.out].size2, 1);
    let mmax = pp_sigma[d.out].ntau as i64;
    let dtau = beta / mmax as f64;
    let n1 = n.max(kt);
    let n11 = (n1 + 1) as usize;
    let ga = &pp_g[d.sa];
    let gb = &pp_g[d.sb];
    let gc = &pp_g[d.sc];
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
    let ls1 = LineSep::new(d.lam1, fit1, h);
    let ls2 = LineSep::new(d.lam2, fit2, h);

    let mut f_ac = vec![Complex::ZERO; n11];
    let mut data_ac = vec![Complex::ZERO; n11 * n11];
    let mut data_bc = vec![Complex::ZERO; n11 * n11];
    let nm = (mmax + 1) as usize;
    let mut data2 = vec![Complex::ZERO; n11 * nm];

    if fam & 1 != 0 {
        // -- ret (1): Delta1^>(n, i2) x [Ga^>(n, i1) op2 Gb^>(i1, i2)] x
        //    Delta2^>(i1, j) op3 Gc^>(i2, j), causal-triangle conv2b per j
        let mut data_ab = vec![Complex::ZERO; n11 * n11];
        for i1 in 0..=n1 {
            let i2max = (i1 + kt).min(n1);
            let ga_ret = gtr(ga, n, i1);
            for i2 in 0..=i2max {
                data_ab[(i1 as usize) * n11 + i2 as usize] =
                    ls1.gtr(n, i2, n1 as usize) * (ga_ret * op2 * gtr(gb, i1, i2));
            }
        }
        for j in (0..=n).rev() {
            let i1min = j.min(n1 - kt);
            for i1 in i1min..=n1 {
                let i2max = (i1 + kt).min(n1);
                let l2 = ls2.gtr(i1, j, n1 as usize);
                for i2 in i1min..=i2max {
                    data_ac[((i1 - i1min) as usize) * n11 + ((i2 - i1min) as usize)] =
                        l2 * (data_ab[(i1 as usize) * n11 + i2 as usize] * op3 * gtr(gc, i2, j));
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
    }
    if fam & 6 != 0 {
        // -- data_bc shared by vt (1) / les (1):
        //    Delta2^>(i2, n) x [Gb^>(i2, i1) op3 Gc^>(i1, n)]
        for t1 in 0..=n1 {
            let t2max = (t1 + kt).min(n1);
            let gc_ret = gtr(gc, t1, n);
            for t2 in 0..=t2max {
                data_bc[(t1 as usize) * n11 + t2 as usize] =
                    ls2.gtr(t2, n, n1 as usize) * (gtr(gb, t2, t1) * op3 * gc_ret);
            }
        }
    }
    if fam & 2 != 0 {
        // -- vt (1): lmix(n, mmax - m) += pref * sig h^2 * conj(...)
        for m in 0..=mmax {
            for i2 in 0..=n1 {
                let i1max = (i2 + kt).min(n1);
                let lam1_vt = ls1.vt(m, i2, n1 as usize);
                for i1 in 0..=i1max {
                    data_ac[(i2 as usize) * n11 + i1 as usize] = lam1_vt
                        * (vt_pp(ga, m, i1) * op2 * data_bc[(i2 as usize) * n11 + i1 as usize]);
                }
            }
            let conv = conv2b_scalar(&data_ac, n11, 0, n as usize, integ);
            let val = pref * Complex::new(sig * h * h, 0.0) * (op1 * conv * op4).conj();
            out.lmix.block_mut(out.lmix.idx(n, mmax - m))[0] += val;
        }
    }
    if fam & 4 != 0 {
        // -- les (1): less(j, n) += pref * (-h^2) * op1 conv op4
        for j in 0..=n {
            for i2 in 0..=n1 {
                let i1max = (i2 + kt).min(n1);
                let lam1_les = ls1.les(j, i2, n1 as usize);
                for i1 in 0..=i1max {
                    data_ac[(i2 as usize) * n11 + i1 as usize] = lam1_les
                        * (les_pp(ga, j, i1) * op2 * data_bc[(i2 as usize) * n11 + i1 as usize]);
                }
            }
            // XCA_INTDUMP audit: the les(1) integrand factor chain at
            // the target row j - the reference side of the exhaustive
            // point-by-point comparison against the (0,0,2) block cut
            // (FEASIBILITY.md, the architecture-level audit)
            if let Ok(tv) = std::env::var("XCA_INTDUMP") {
                if tv.trim().parse::<i64>() == Ok(j) {
                    eprintln!(
                        "INTREF-D {} {} {} {} {} {} {} {} pref={} sig={}",
                        d.out,
                        d.sa,
                        d.sb,
                        d.sc,
                        cz(op1),
                        cz(op2),
                        cz(op3),
                        cz(op4),
                        cz(d.prefactor),
                        d.sig
                    );
                    for i2 in 0..=n1 {
                        eprintln!(
                            "INTREF-L1 {} {} {} {} {} {}",
                            d.out,
                            d.sa,
                            d.sb,
                            d.sc,
                            i2,
                            cz(ls1.les(j, i2, n1 as usize))
                        );
                    }
                    for i1 in 0..=n1 {
                        eprintln!(
                            "INTREF-L2 {} {} {} {} {} {}",
                            d.out,
                            d.sa,
                            d.sb,
                            d.sc,
                            i1,
                            cz(ls2.gtr(i1, n, n1 as usize))
                        );
                        eprintln!(
                            "INTREF-GA {} {} {} {} {} {}",
                            d.out,
                            d.sa,
                            d.sb,
                            d.sc,
                            i1,
                            cz(les_pp(ga, j, i1))
                        );
                    }
                    for i2 in 0..=n1 {
                        eprintln!(
                            "INTREF-GC {} {} {} {} {} {} {}",
                            d.out,
                            d.sa,
                            d.sb,
                            d.sc,
                            i2,
                            cz(gtr(gc, i2, n)),
                            cz(gtr(gc, n, i2))
                        );
                    }
                    for i2 in 1..=n1 {
                        let row = poly_integral_scalar(
                            &data_ac[(i2 as usize) * n11..],
                            0,
                            i2 as usize,
                            integ,
                        );
                        eprintln!(
                            "INTREF-ROW {} {} {} {} {} {}",
                            d.out,
                            d.sa,
                            d.sb,
                            d.sc,
                            i2,
                            cz(row)
                        );
                    }
                    for i2 in 0..=n1 {
                        let i1max = (i2 + kt).min(n1);
                        for i1 in 0..=i1max {
                            eprintln!(
                                "INTREF-G2 {} {} {} {} {} {} {} {}",
                                d.out,
                                d.sa,
                                d.sb,
                                d.sc,
                                i2,
                                i1,
                                cz(data_ac[(i2 as usize) * n11 + i1 as usize]),
                                cz(gtr(gb, i1, i2))
                            );
                        }
                    }
                }
            }
            let conv = conv2b_scalar(&data_ac, n11, 0, n as usize, integ);
            if let Ok(tv) = std::env::var("XCA_INTDUMP") {
                if tv.trim().parse::<i64>() == Ok(j) {
                    eprintln!(
                        "INTREF-TOT {} {} {} {} {} {}",
                        d.out,
                        d.sa,
                        d.sb,
                        d.sc,
                        cz(conv),
                        cz(pref * (-hh) * (op1 * conv * op4))
                    );
                }
            }
            let val = pref * (-hh) * (op1 * conv * op4);
            out.less.block_mut(out.less.idx(j, n))[0] += val;
        }
    }
    if fam & 24 != 0 {
        // -- vt (2) / les (2): Delta2^vt(tau, t) rows over the Matsubara
        //    axis (projected basis), mixed with the Matsubara component
        //    of Ga (lowside) or its tv row (les); fills the shared
        //    preamble `data2`
        // the reference evaluates the line-2 vt factor ONCE per tau at
        // the fixed time t = n (a per-tau constant), not per t
        for tau in 0..=mmax {
            let l2 = ls2.vt(tau, n, n1 as usize);
            for t in 0..=n1 {
                data2[(t as usize) * nm + tau as usize] = l2 * (op2 * vt_pp(gb, tau, t));
            }
        }
    }
    if fam & 8 != 0 {
        for m in 0..=mmax {
            for i2 in 0..=n1 {
                let row = &data2[(i2 as usize) * nm..];
                let conv_ab = lowside_scalar(m as usize, ga.mat.raw(), row, integ);
                f_ac[i2 as usize] = ls1.vt(m, i2, n1 as usize) * (conv_ab * op3 * gtr(gc, i2, n));
            }
            let integr = gregory_scalar(n as usize, &f_ac, integ);
            let val = pref * Complex::new(-sig * dtau * h, 0.0) * (op1 * integr * op4).conj();
            out.lmix.block_mut(out.lmix.idx(n, mmax - m))[0] += val;
        }
    }
    if fam & 16 != 0 {
        for j in 0..=n {
            for i2 in 0..=n1 {
                let row = &data2[(i2 as usize) * nm..];
                let mut f_ab = vec![Complex::ZERO; nm];
                for mu in 0..=mmax {
                    f_ab[mu as usize] = tv_pp(ga, j, mu) * row[mu as usize];
                }
                let integr_ab = gregory_scalar(mmax as usize, &f_ab, integ);
                f_ac[i2 as usize] =
                    ls1.les(j, i2, n1 as usize) * (integr_ab * op3 * gtr(gc, i2, n));
            }
            let integr = gregory_scalar(n as usize, &f_ac, integ);
            let val = pref * Complex::new(0.0, -dtau * h) * (op1 * integr * op4);
            if std::env::var("XCA_PROBE").is_ok() && j == 0 {
                eprintln!(
                    "PROBE eval2-les2: f_ac[5] {:?} | integr {:?} | mmax {} n1 {} n {} mu-inner uses row (data2)",
                    f_ac[5usize], integr, mmax, n1, n
                );
            }
            out.less.block_mut(out.less.idx(j, n))[0] += val;
        }
    }
    if fam & 32 != 0 {
        // -- vt (3): Matsubara-axis core with the vt data of line 2
        //    separated per pole (phase at t = n hoisted)
        {
            let gc_vt: Vec<Complex> = (0..=mmax).map(|tau| vt_pp(gc, tau, n)).collect();
            for m in 0..=mmax {
                let mut acc = Complex::ZERO;
                for l in 0..fit2.poles.len() {
                    // per-tau vt coefficients of line 2 at this pole
                    let mut prof = vec![Complex::ZERO; (mmax + 1) as usize];
                    for tau in 0..=mmax {
                        prof[tau as usize] = ls2.vt_coef_at(tau, l, n1 as usize);
                    }
                    let core = crate::oca::sigma_mat_core(
                        m as usize,
                        ga.mat.raw(),
                        gb.mat.raw(),
                        &gc_vt,
                        d.lam1.mat.raw(),
                        &prof,
                        mmax as usize,
                        integ,
                        op1 * op2 * op3 * op4,
                    );
                    acc += phase(fit2.poles[l], n, 0, h) * core;
                }
                let val = pref * Complex::new(0.0, -sig * dtau * dtau) * acc.conj();
                out.lmix.block_mut(out.lmix.idx(n, mmax - m))[0] += val;
            }
        }
    }
    if fam & 64 != 0 {
        // -- les (3): two-level real-time gregory with les reads of both
        //    lines
        let intdump3 = std::env::var("XCA_INTDUMP")
            .ok()
            .and_then(|tv| tv.trim().parse::<i64>().ok());
        for j in (0..=n).rev() {
            let i1max = j.max(kt);
            for i1 in 0..=i1max {
                let mut f_bc = vec![Complex::ZERO; n11];
                for i2 in 0..=n1 {
                    f_bc[i2 as usize] =
                        ls1.les(j, i2, n1 as usize) * (les_pp(gb, i1, i2) * op3 * gtr(gc, i2, n));
                }
                let integr_bc = gregory_scalar(n as usize, &f_bc, integ);
                f_ac[i1 as usize] =
                    ls2.les(i1, n, n1 as usize) * (gtr(ga, j, i1) * op2 * integr_bc);
            }
            let integr = gregory_scalar(j as usize, &f_ac, integ);
            let val = pref * hh * (op1 * integr * op4);
            // XCA_INTDUMP audit: the les(3) factor chain at the target
            // row (the reference side of the cut#3 (1,0,1) comparison)
            if intdump3 == Some(j) {
                eprintln!(
                    "INTREF3-D {} {} {} {} {} {} {} {} {} {}",
                    d.out,
                    d.sa,
                    d.sb,
                    d.sc,
                    cz(op1),
                    cz(op2),
                    cz(op3),
                    cz(op4),
                    cz(pref),
                    d.sig
                );
                for i2 in 0..=n1 {
                    eprintln!(
                        "INTREF3-L1 {} {} {}",
                        d.out,
                        i2,
                        cz(ls1.les(j, i2, n1 as usize))
                    );
                }
                for i1 in 0..=i1max {
                    eprintln!(
                        "INTREF3-L2 {} {} {}",
                        d.out,
                        i1,
                        cz(ls2.les(i1, n, n1 as usize))
                    );
                    eprintln!("INTREF3-GA {} {} {}", d.out, i1, cz(gtr(ga, j, i1)));
                }
                for i2 in 0..=n1 {
                    eprintln!(
                        "INTREF3-GC {} {} {} {}",
                        d.out,
                        i2,
                        cz(gtr(gc, i2, n)),
                        cz(gtr(gc, n, i2))
                    );
                }
                for i1 in 0..=i1max {
                    for i2 in 0..=n1 {
                        eprintln!(
                            "INTREF3-GB {} {} {} {}",
                            d.out,
                            i1,
                            i2,
                            cz(les_pp(gb, i1, i2))
                        );
                    }
                }
                // recompute and dump the inner integrals + f_ac rows
                for i1 in 0..=i1max {
                    let mut f_bc = vec![Complex::ZERO; n11];
                    for i2 in 0..=n1 {
                        f_bc[i2 as usize] = ls1.les(j, i2, n1 as usize)
                            * (les_pp(gb, i1, i2) * op3 * gtr(gc, i2, n));
                    }
                    let integr_bc = gregory_scalar(n as usize, &f_bc, integ);
                    eprintln!(
                        "INTREF3-ROW {} {} {} {}",
                        d.out,
                        i1,
                        cz(integr_bc),
                        cz(ls2.les(i1, n, n1 as usize) * (gtr(ga, j, i1) * op2 * integr_bc))
                    );
                }
                eprintln!("INTREF3-TOT {} {}", d.out, cz(val));
            }
            out.less.block_mut(out.less.idx(j, n))[0] += val;
        }
    }
    if fam & 128 != 0 {
        // -- les (4): stv reads of line 1 at fixed j (raw point values),
        //    Matsubara lowside with gb, vt of line 2 separated per pole
        for j in 0..=n {
            let mut f_bc = vec![Complex::ZERO; (mmax + 1) as usize];
            for mu in 0..=mmax {
                f_bc[mu as usize] = tv_c(d.lam1, j, mu) * (op3 * vt_pp(gc, mu, n));
            }
            let mut f_ac_m = vec![Complex::ZERO; (mmax + 1) as usize];
            for mu in 0..=mmax {
                let conv_bc = lowside_scalar(mu as usize, gb.mat.raw(), &f_bc, integ);
                f_ac_m[mu as usize] =
                    ls2.vt(mu, n, n1 as usize) * (tv_pp(ga, j, mu) * op2 * conv_bc);
            }
            let integr = gregory_scalar(mmax as usize, &f_ac_m, integ);
            let val = pref * Complex::new(0.0, dtau * dtau) * (op1 * integr * op4);
            out.less.block_mut(out.less.idx(j, n))[0] += val;
        }
    }
    if fam & 256 != 0 {
        // -- les (5): greater reads of line 1 anchored at j, les of line
        //    2 anchored at n, conv2b over [0, j]
        let intdump5 = std::env::var("XCA_INTDUMP")
            .ok()
            .and_then(|tv| tv.trim().parse::<i64>().ok());
        {
            let mut data_bc5 = vec![Complex::ZERO; n11 * n11];
            for i1 in 0..=n1 {
                let i2max = (i1 + kt).min(n1);
                let l2 = ls2.les(i1, n, n1 as usize);
                for i2 in 0..=i2max {
                    data_bc5[(i1 as usize) * n11 + i2 as usize] =
                        l2 * (gtr(gb, i1, i2) * op3 * les_pp(gc, i2, n));
                }
            }
            for j in (0..=n).rev() {
                let i1max = j.max(kt);
                for i1 in 0..=i1max {
                    let i2max = (i1 + kt).min(i1max);
                    let ga_ret = gtr(ga, j, i1);
                    for i2 in 0..=i2max {
                        data_ac[(i1 as usize) * n11 + i2 as usize] = ls1.gtr(j, i2, n1 as usize)
                            * (ga_ret * op2 * data_bc5[(i1 as usize) * n11 + i2 as usize]);
                    }
                }
                let conv = conv2b_scalar(&data_ac, n11, 0, j as usize, integ);
                let val = pref * (-hh) * (op1 * conv * op4);
                if intdump5 == Some(j) {
                    eprintln!(
                        "INTREF5-D {} {} {} {} {} {} {} {} {} {}",
                        d.out,
                        d.sa,
                        d.sb,
                        d.sc,
                        cz(op1),
                        cz(op2),
                        cz(op3),
                        cz(op4),
                        cz(pref),
                        d.sig
                    );
                    for i2 in 0..=n1 {
                        eprintln!(
                            "INTREF5-L1 {} {} {}",
                            d.out,
                            i2,
                            cz(ls1.gtr(j, i2, n1 as usize))
                        );
                    }
                    for i1 in 0..=i1max {
                        eprintln!(
                            "INTREF5-L2 {} {} {}",
                            d.out,
                            i1,
                            cz(ls2.les(i1, n, n1 as usize))
                        );
                        eprintln!("INTREF5-GA {} {} {}", d.out, i1, cz(gtr(ga, j, i1)));
                    }
                    for i2 in 0..=n1 {
                        eprintln!(
                            "INTREF5-GC {} {} {} {}",
                            d.out,
                            i2,
                            cz(les_pp(gc, i2, n)),
                            cz(gtr(gc, i2, n))
                        );
                    }
                    for i1 in 0..=i1max {
                        let i2max2 = (i1 + kt).min(n1);
                        for i2 in 0..=i2max2 {
                            eprintln!("INTREF5-GB {} {} {} {}", d.out, i1, i2, cz(gtr(gb, i1, i2)));
                        }
                    }
                    // row integrals of data_ac matching conv2b(j,0)
                    for i1 in 1..=i1max {
                        let row = poly_integral_scalar(
                            &data_ac[(i1 as usize) * n11..],
                            0,
                            i1 as usize,
                            integ,
                        );
                        eprintln!("INTREF5-ROW {} {} {}", d.out, i1, cz(row));
                    }
                    eprintln!("INTREF5-TOT {} {}", d.out, cz(val));
                }
                out.less.block_mut(out.less.idx(j, n))[0] += val;
            }
        }
    }
    if fam & 512 != 0 {
        // -- les (6): tv of line 1 at fixed j (raw point values) times
        //    the tv/vt chain through gb/gc, les of line 2
        for j in 0..=n {
            let i1max = j.max(kt);
            for i1 in 0..=i1max {
                let mut f_bc = vec![Complex::ZERO; (mmax + 1) as usize];
                for mu in 0..=mmax {
                    f_bc[mu as usize] =
                        tv_c(d.lam1, j, mu) * (tv_pp(gb, i1, mu) * op3 * vt_pp(gc, mu, n));
                }
                let integr_bc = gregory_scalar(mmax as usize, &f_bc, integ);
                f_ac[i1 as usize] =
                    ls2.les(i1, n, n1 as usize) * (gtr(ga, j, i1) * op2 * integr_bc);
            }
            let integr = gregory_scalar(j as usize, &f_ac, integ);
            let val = pref * Complex::new(0.0, dtau * h) * (op1 * integr * op4);
            out.less.block_mut(out.less.idx(j, n))[0] += val;
        }
    }
}
