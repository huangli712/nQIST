// Project : nQIST
// File    : src/xca/mod.rs
// Purpose : arbitrary Crossing Approximation (XCA) module
//
// XCA generalizes the NCA (order 1) and OCA (order 2) bold
// hybridization expansions to arbitrary order by enumerating the
// connected chord-diagram topologies of the skeleton series and
// evaluating them through a sum-of-exponentials separation of the
// hybridization lines:
//
// - J. Kaye, Z. Huang, H. U. R. Strand, D. Golez, Phys. Rev. X 14,
//   031034 (2024)
// - Z. Huang, D. Golež, H. U. R. Strand, J. Kaye, SciPost Phys. 19,
//   121 (2025)
// - reference implementation: https://github.com/TRIQS/xca
//   (analyzed at pinned SHA 35b0c593, read-only)
//
// This module reimplements the algorithm natively on the nQIST
// Matsubara grid with the KadanoffBaym Gregory quadrature instead of
// the DLR discretization used upstream; orders 1 and 2 reproduce the
// existing NCA/OCA kernels and orders >= 3 add the crossing
// topologies in between.

pub(crate) mod gf;
pub(crate) mod sigma;
pub(crate) mod soe;
pub(crate) mod topology;

pub(crate) use sigma::add_sigma_mat_xca;
pub(crate) use soe::{FIT_TOL_DEFAULT, SoeFit, soe_fit_tol};

/// Pole budget of the real-time XCA fits (`rt_xca_*` routes): every
/// pole costs `poles^m` in the diagram sums, so the budget is the main
/// performance knob there.
pub const MAX_POLES_DEFAULT: usize = 6;

/// Upper bound on the number of SOE poles of the equilibrium XCA fit.
/// The pole count is set by the fit tolerance (blueprint: TRIQS/xca
/// `max_n_poles`): the fit grows until the tolerance is met or this
/// bound is reached. Every surviving pole costs `p^(m-1)` in the
/// diagram sums, so the bound caps the kernel cost on data whose
/// error floor lies above the tolerance.
pub const EQ_MAX_POLES_DEFAULT: usize = 32;

/// Default spectral cutoff passed to the SOE fit (`w_max`).
pub const W_MAX_DEFAULT: f64 = 12.0;

/// Fit every hybridization channel of `pp_ints` on the Matsubara grid
/// with the default tolerance; one `SoeFit` per channel, in order.
pub(crate) fn fit_channels(
    channels: &[crate::oca::OcaChannel],
    beta: f64,
    ntau: usize,
    max_poles: usize,
    w_max: f64,
) -> Vec<SoeFit> {
    fit_channels_tol(channels, beta, ntau, max_poles, w_max, FIT_TOL_DEFAULT)
}

/// Pole-free fits of `n` channels: order 1 reads the raw hybridization
/// and never touches the fits, so none is computed for it.
pub(crate) fn unused_fits(n: usize, beta: f64) -> Vec<SoeFit> {
    (0..n).map(|_| soe::SoeFit::unused(beta)).collect()
}

/// `fit_channels` with an explicit tolerance (root-mean-square error of
/// each channel's SOE on the Matsubara grid). Channels with identical
/// sample data share one fit.
pub(crate) fn fit_channels_tol(
    channels: &[crate::oca::OcaChannel],
    beta: f64,
    ntau: usize,
    max_poles: usize,
    w_max: f64,
    tol: f64,
) -> Vec<SoeFit> {
    use kadanoffbaym::complex::Complex;
    let dtau = beta / ntau as f64;
    let data: Vec<Vec<Complex>> = channels
        .iter()
        .map(|c| (0..=ntau).map(|i| c.delta.mat.block(i)[0]).collect())
        .collect();
    let mut fits: Vec<Option<SoeFit>> = vec![None; channels.len()];
    for k in 0..channels.len() {
        if fits[k].is_some() {
            continue;
        }
        let fit = soe_fit_tol(beta, dtau, &data[k], w_max, max_poles, tol);
        if std::env::var_os("NQIST_FIT_DEBUG").is_some() {
            eprintln!(
                "FITDEBUG: ch {k}, {} poles, rms {:.3e}, max-abs {:.3e} (tol {tol:.1e})",
                fit.len(),
                (fit.resid_l2 / (ntau + 1) as f64).sqrt(),
                fit.resid
            );
        }
        for (j, dj) in data.iter().enumerate().skip(k) {
            if (0..=ntau).all(|i| dj[i].re == data[k][i].re && dj[i].im == data[k][i].im) {
                fits[j] = Some(fit.clone());
            }
        }
    }
    fits.into_iter()
        .map(|f| f.expect("every channel fitted"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hilbert::{SectorClass, build};
    use crate::nca::{self, GfVertex, Hybridization};
    use crate::oca;
    use kadanoffbaym::complex::Complex;
    use kadanoffbaym::types::{Contour, ContourGreen, PseudoGreen, Sign};
    use kadanoffbaym::weights::Integrator;

    const BETA: f64 = 5.0;
    const NTAU: usize = 96;

    struct Fixture {
        hs: crate::hilbert::HilbertSpace,
        delta: ContourGreen,
        delta_cc: ContourGreen,
        pp_g: Vec<PseudoGreen>,
    }

    impl Fixture {
        /// Two-site discrete bath `Delta(tau) = v1^2 K(tau, e1) +
        /// v2^2 K(tau, e2)` and atomic-limit propagators.
        #[allow(clippy::needless_range_loop)]
        fn new() -> Self {
            let hs = build(1, SectorClass::SpinResolved);
            let contour = Contour::new(4, NTAU as i64, 1, 1, 0.08, BETA);
            let dtau = BETA / NTAU as f64;
            let kf = |t: f64, w: f64| -(-w * t).exp() / (1.0 + (-BETA * w).exp());
            let mut delta = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
            let mut delta_cc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
            for i in 0..=NTAU {
                let t = dtau * i as f64;
                let v = 0.09 * kf(t, -1.2) + 0.16 * kf(t, 0.8);
                delta.mat.block_mut(i)[0] = Complex::new(v, 0.0);
                // Delta_cc(tau) = -Delta(beta - tau)
                let tb = BETA - t;
                let vb = -(0.09 * kf(tb, -1.2) + 0.16 * kf(tb, 0.8));
                delta_cc.mat.block_mut(i)[0] = Complex::new(vb, 0.0);
            }
            // atomic-limit pseudo-particle propagators (shifted by
            // pp_mu so the density is one)
            let energies = [0.0, -1.5, -1.5, 0.0];
            let mu = -1.6387400224070612;
            let mut pp_g = Vec::new();
            for s in 0..4 {
                let mut g = PseudoGreen::new(&contour, hs.sectors[s].sig, Complex::ZERO);
                for i in 0..=NTAU {
                    let v = -(dtau * i as f64 * (energies[s] - mu)).exp();
                    g.mat.block_mut(i)[0] = Complex::new(v, 0.0);
                }
                pp_g.push(g);
            }
            Fixture {
                hs,
                delta,
                delta_cc,
                pp_g,
            }
        }

        fn pp_ints(&self) -> Vec<Hybridization<'_>> {
            let hs = &self.hs;
            let cua = hs.flavor(0, 0, false);
            let cuc = hs.flavor(0, 0, true);
            let cda = hs.flavor(0, 1, false);
            let cdc = hs.flavor(0, 1, true);
            vec![
                Hybridization {
                    delta: &self.delta,
                    op_cre_fl: cuc,
                    op_ann_fl: cua,
                    fermion: true,
                    backward: false,
                },
                Hybridization {
                    delta: &self.delta_cc,
                    op_cre_fl: cua,
                    op_ann_fl: cuc,
                    fermion: true,
                    backward: true,
                },
                Hybridization {
                    delta: &self.delta,
                    op_cre_fl: cdc,
                    op_ann_fl: cda,
                    fermion: true,
                    backward: false,
                },
                Hybridization {
                    delta: &self.delta_cc,
                    op_cre_fl: cda,
                    op_ann_fl: cdc,
                    fermion: true,
                    backward: true,
                },
            ]
        }

        fn empty_sigma(&self) -> Vec<PseudoGreen> {
            let contour = Contour::new(4, NTAU as i64, 1, 1, 0.08, BETA);
            (0..4)
                .map(|s| PseudoGreen::new(&contour, self.hs.sectors[s].sig, Complex::ZERO))
                .collect()
        }
    }

    /// Multi-band probe: the native dimension-general OCA kernels vs
    /// the (dimer-ED-validated) XCA m=2 backbone on a two-band
    /// fixture. Reports per-sector max deviation to localize any
    /// matrix-convention bug in the native path.
    /// Two-band GF probe: the native gf-mat kernel (matrix branch) vs
    /// the XCA m=2 gf backbone on the same fixture as the sigma probe.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn oca_gfmat_twoband_matches_xca_m2() {
        use crate::hamilton::LocalHamiltonian::TwoBand;
        use kadanoffbaym::linalg::eigh;
        let hs = build(2, SectorClass::SpinResolved);
        let ns = hs.ns();
        let contour = Contour::new(4, NTAU as i64, 1, 1, 0.08, BETA);
        let dtau = BETA / NTAU as f64;
        let kf = |t: f64, w: f64| -(-w * t).exp() / (1.0 + (-BETA * w).exp());
        let bath_val = |t: f64| 0.09 * kf(t, -1.2) + 0.16 * kf(t, 0.8);
        let mut deltas = Vec::new();
        let mut deltas_cc = Vec::new();
        for _ in 0..4 {
            let mut d = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
            let mut dc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
            for i in 0..=NTAU {
                let t = dtau * i as f64;
                d.mat.block_mut(i)[0] = Complex::new(bath_val(t), 0.0);
                dc.mat.block_mut(i)[0] = Complex::new(-bath_val(BETA - t), 0.0);
            }
            deltas.push(d);
            deltas_cc.push(dc);
        }
        let ham = TwoBand {
            mu: 0.0,
            u: vec![2.0; 6],
            j: vec![0.4; 6],
            delta: vec![0.8; 6],
            bz: vec![0.0; 6],
        };
        let mut pp_g = Vec::new();
        for sec in 0..ns {
            let h0 = ham.sector_block(&hs, sec, -1);
            let dim = hs.sectors[sec].dim;
            let (evals, _) = eigh(&h0);
            let csec = Contour::new(4, NTAU as i64, dim as i64, dim as i64, 0.08, BETA);
            let mut g = PseudoGreen::new(&csec, hs.sectors[sec].sig, Complex::ZERO);
            for m in 0..=NTAU {
                for i in 0..dim {
                    let x = -(dtau * m as f64 * evals[i]).exp();
                    g.mat.block_mut(m as usize)[i * dim + i] = Complex::new(x, 0.0);
                }
            }
            pp_g.push(g);
        }
        let mut pp_ints = Vec::new();
        for (orb, spin) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
            let c = hs.flavor(orb, spin, false);
            let a = hs.flavor(orb, spin, true);
            let ix = 2 * orb + spin;
            pp_ints.push(Hybridization {
                delta: &deltas[ix],
                op_cre_fl: a,
                op_ann_fl: c,
                fermion: true,
                backward: false,
            });
            pp_ints.push(Hybridization {
                delta: &deltas_cc[ix],
                op_cre_fl: c,
                op_ann_fl: a,
                fermion: true,
                backward: true,
            });
        }
        let verts: Vec<GfVertex> = (0..2usize)
            .map(|orb| GfVertex {
                op_ann_fl: hs.flavor(orb, 0, false),
                op_cre_fl: hs.flavor(orb, 0, true),
            })
            .collect();
        let integ = Integrator::new(5);
        let mut want = nca::gf_mat(&hs, &pp_g, &verts);
        oca::add_gf_mat_oca(
            &hs, &pp_g, &verts, &pp_ints, None, BETA, 5, &integ, &mut want,
        );
        let channels = oca::channels_with_bath(&pp_ints, None);
        let fits = fit_channels(&channels, BETA, NTAU, 12, 6.0);
        let grids = gf::Grids {
            ntau: NTAU,
            dtau,
            pp_g: &pp_g,
        };
        let mut got = vec![vec![Complex::ZERO; NTAU + 1]; 2];
        gf::add_gf_mat_xca(1, &hs, &channels, &fits, &verts, &grids, &integ, &mut got);
        gf::add_gf_mat_xca(2, &hs, &channels, &fits, &verts, &grids, &integ, &mut got);
        // baseline: XCA(1) vs NCA
        let nca_only = nca::gf_mat(&hs, &pp_g, &verts);
        let mut got1 = vec![vec![Complex::ZERO; NTAU + 1]; 2];
        gf::add_gf_mat_xca(1, &hs, &channels, &fits, &verts, &grids, &integ, &mut got1);
        for v in 0..2 {
            let mut worst = 0.0f64;
            for m in 1..=NTAU {
                let dev = (nca_only[v][m] - got1[v][m]).abs() / nca_only[v][m].abs().max(1e-3);
                worst = worst.max(dev);
            }
            eprintln!("gf probe baseline v{v}: XCA1-vs-NCA max rel {worst:.3e}");
        }
        // OCA part: (want - NCA) vs (got - got1)
        for v in 0..2 {
            let mut dev_at = vec![0.0f64; NTAU + 1];
            for m in 1..=NTAU {
                let a = want[v][m] - nca_only[v][m];
                let b = got[v][m] - got1[v][m];
                dev_at[m] = (a - b).abs() / a.abs().max(1e-6);
                if m == 25 || m == 33 {
                    eprintln!(
                        "gf raw v{v} m{m}: oca_native {a:?} oca_xca {b:?} ratio {:?}",
                        b / if a.abs() > 1e-30 { a } else { Complex::ONE }
                    );
                }
            }
            let mut s = String::new();
            for m in (1..=NTAU).step_by(8) {
                s.push_str(&format!("{m}:{:.0e} ", dev_at[m]));
            }
            eprintln!("gf probe oca-part v{v}: {s}");
        }
        // per-channel native decomposition (the valid reference; the
        // XCA single-channel calls are not the full's per-channel
        // terms and are left out so the debug instrumentation below
        // only sees the full add_gf_mat_xca(2) run)
        for ix in 0..pp_ints.len() {
            let mut w1 = nca::gf_mat(&hs, &pp_g, &verts);
            oca::add_gf_mat_oca(
                &hs,
                &pp_g,
                &verts,
                &pp_ints[ix..ix + 1],
                None,
                BETA,
                5,
                &integ,
                &mut w1,
            );
            for v in 0..2 {
                let a = w1[v][33] - nca_only[v][33];
                eprintln!("chan {ix} v{v} m33: native {a:?}");
            }
        }
    }

    /// Differential probe: on all-1x1 (single-band-shaped) data the
    /// scalar tstp kernels (P5-gated ground truth) and the matrix
    /// kernels must agree EXACTLY. Any element-wise deviation
    /// localizes a matrix-transcription value bug.

    #[allow(dead_code)]
    fn cdev(a: Complex, b: Complex) -> f64 {
        let d = a - b;
        let dn = (d * d.conj()).re.sqrt();
        let an = (a * a.conj()).re.sqrt();
        dn / an.max(1e-12)
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn oca_tstp_matrix_matches_scalar_1x1() {
        use crate::hamilton::LocalHamiltonian::SingleBand;
        use kadanoffbaym::linalg::eigh;
        let nt: i64 = 8;
        let hs = build(1, SectorClass::SpinResolved);
        let ns = hs.ns();
        let fcontour = Contour::new(nt, NTAU as i64, 1, 1, 0.16, BETA);
        let dtau = BETA / NTAU as f64;
        let h = 0.02;
        // one decaying bath channel + backward
        let mk_delta = |sign_flip: bool| {
            let mut d = ContourGreen::new(&fcontour, Sign::Fermi, Complex::ZERO);
            for m in 0..=(NTAU as i64) {
                let t = dtau * m as f64;
                let v = -((1.3 * t).exp()) * 0.2;
                d.mat.block_mut(m as usize)[0] = Complex::new(if sign_flip { -v } else { v }, 0.0);
            }
            for n in 0..=nt {
                for j in 0..=n {
                    let dt = (n - j) as f64 * h;
                    let vr = Complex::new(0.0, -0.3) * Complex::new((1.1 * dt).exp(), 0.0);
                    d.ret.raw_block_at_mut(n, j)[0] = vr;
                    d.less.raw_block_at_mut(j, n)[0] = Complex::new((0.9 * dt).exp() * 0.25, 0.0);
                }
                for m in 0..=(NTAU as i64) {
                    d.lmix.raw_block_at_mut(n, m)[0] =
                        Complex::new((0.5 * n as f64 * h + 0.7 * m as f64 * dtau).exp(), 0.0)
                            * 0.15;
                }
            }
            d
        };
        let delta = mk_delta(false);
        let delta_cc = mk_delta(true);
        let delta_d = mk_delta(false);
        let delta_d_cc = mk_delta(true);
        let ham = SingleBand {
            mu: 0.0,
            u: vec![2.0; (nt + 2) as usize],
            eps_up: vec![0.0; (nt + 2) as usize],
            eps_do: vec![0.0; (nt + 2) as usize],
        };
        let mut pp_g = Vec::new();
        for sec in 0..ns {
            let h0 = ham.sector_block(&hs, sec, -1);
            let (evals, _) = eigh(&h0);
            let dim = hs.sectors[sec].dim;
            let csec = Contour::new(nt, NTAU as i64, dim as i64, dim as i64, 0.16, BETA);
            let mut g = PseudoGreen::new(&csec, hs.sectors[sec].sig, Complex::ZERO);
            for m in 0..=(NTAU as i64) {
                g.mat.block_mut(m as usize)[0] =
                    Complex::new(-(dtau * m as f64 * evals[0]).exp(), 0.0);
            }
            for n in 0..=nt {
                for j in 0..=n {
                    let dt = (n - j) as f64 * h;
                    g.ret.raw_block_at_mut(n, j)[0] =
                        Complex::new(0.0, -1.0) * Complex::new(evals[0] * dt, 0.0).exp() * 0.5;
                    g.less.raw_block_at_mut(j, n)[0] = Complex::new(evals[0] * dt, 0.0).exp() * 0.3;
                }
                for m in 0..=(NTAU as i64) {
                    g.lmix.raw_block_at_mut(n, m)[0] =
                        Complex::new(evals[0] * (n as f64 * h + m as f64 * dtau), 0.0).exp() * 0.2;
                }
            }
            pp_g.push(g);
        }
        let c = hs.flavor(0, 0, false);
        let a = hs.flavor(0, 0, true);
        let cd = hs.flavor(0, 1, false);
        let ad = hs.flavor(0, 1, true);
        // NB: a single spin species closes NO sigma/gf OCA chain (the
        // four-sector loop needs alternating up/down operators); both
        // spins are required for nonzero diagram counts.
        let pp_ints = vec![
            Hybridization {
                delta: &delta,
                op_cre_fl: a,
                op_ann_fl: c,
                fermion: true,
                backward: false,
            },
            Hybridization {
                delta: &delta_cc,
                op_cre_fl: c,
                op_ann_fl: a,
                fermion: true,
                backward: true,
            },
            Hybridization {
                delta: &delta_d,
                op_cre_fl: ad,
                op_ann_fl: cd,
                fermion: true,
                backward: false,
            },
            Hybridization {
                delta: &delta_d_cc,
                op_cre_fl: cd,
                op_ann_fl: ad,
                fermion: true,
                backward: true,
            },
        ];
        let channels = oca::channels_with_bath(&pp_ints, None);
        let integ = Integrator::new(5);
        let sig_diagrams = oca::build_sigma_diagrams_oca_channels(&hs, &channels);
        let verts: Vec<GfVertex> = vec![GfVertex {
            op_ann_fl: c,
            op_cre_fl: a,
        }];
        let gf_diagrams = oca::build_gf_diagrams_oca_channels(&hs, &verts, &channels);
        let mk_out = || -> Vec<PseudoGreen> {
            (0..ns)
                .map(|sec| {
                    let dim = hs.sectors[sec].dim as i64;
                    let csec = Contour::new(nt, NTAU as i64, dim, dim, 0.16, BETA);
                    PseudoGreen::new(&csec, hs.sectors[sec].sig, Complex::ZERO)
                })
                .collect()
        };
        let mut worst = 0.0f64;
        for n in [0i64, 2, 5, 7] {
            let mut got_scalar = mk_out();
            for d in &sig_diagrams {
                oca::add_sigma_tstp_oca(n, d, &pp_g, BETA, h, 5, &integ, &mut got_scalar);
            }
            let mut got_matrix = mk_out();
            for d in &sig_diagrams {
                oca::add_sigma_tstp_oca_matrix(n, d, &pp_g, BETA, h, 5, &integ, &mut got_matrix);
            }
            for sec in 0..ns {
                for j in 0..=n {
                    let a1 = got_scalar[sec].ret.raw_block_at(n, j)[0];
                    let b1 = got_matrix[sec].ret.raw_block_at(n, j)[0];
                    let dev = cdev(a1, b1);
                    if dev > worst {
                        worst = dev;
                        eprintln!(
                            "SIG ret n={n} sec={sec} j={j}: scalar={a1:?} matrix={b1:?} dev={dev:.3e}"
                        );
                    }
                    let a2 = got_scalar[sec].less.raw_block_at(j, n)[0];
                    let b2 = got_matrix[sec].less.raw_block_at(j, n)[0];
                    let dev2 = cdev(a2, b2);
                    if dev2 > worst {
                        worst = dev2;
                        eprintln!(
                            "SIG les n={n} sec={sec} j={j}: scalar={a2:?} matrix={b2:?} dev={dev2:.3e}"
                        );
                    }
                }
                for m in 0..=(NTAU as i64) {
                    let a3 = got_scalar[sec].lmix.raw_block_at(n, m)[0];
                    let b3 = got_matrix[sec].lmix.raw_block_at(n, m)[0];
                    let dev3 = cdev(a3, b3);
                    if dev3 > worst {
                        worst = dev3;
                        eprintln!(
                            "SIG lmix n={n} sec={sec} m={m}: scalar={a3:?} matrix={b3:?} dev={dev3:.3e}"
                        );
                    }
                }
            }
            eprintln!("sigma n={n}: worst so far {worst:.3e}");
        }
        assert!(worst < 1e-10, "sigma scalar-vs-matrix deviation {worst}");
        // gf side
        let mut worst_gf = 0.0f64;
        for n in [0i64, 2, 5, 7] {
            let tc = Contour::new(nt, NTAU as i64, 1, 1, 0.16, BETA);
            let mk_ts =
                || kadanoffbaym::types::GreenTimeStep::new(&tc, n, Sign::Fermi, Complex::ZERO);
            let mut ts_scalar = mk_ts();
            for d in &gf_diagrams {
                oca::add_gf_tstp_oca(n, d, &pp_g, BETA, h, 5, &integ, &mut ts_scalar);
            }
            let mut ts_matrix = mk_ts();
            for d in &gf_diagrams {
                oca::add_gf_tstp_oca_matrix(n, d, &pp_g, BETA, h, 5, &integ, &mut ts_matrix);
            }
            for j in 0..=n {
                let a1 = ts_scalar.ret.raw_block_at(n, j)[0];
                let b1 = ts_matrix.ret.raw_block_at(n, j)[0];
                let dev = cdev(a1, b1);
                if dev > worst_gf {
                    worst_gf = dev;
                    eprintln!("GF ret n={n} j={j}: scalar={a1:?} matrix={b1:?} dev={dev:.3e}");
                }
                let a2 = ts_scalar.less.raw_block_at(j, n)[0];
                let b2 = ts_matrix.less.raw_block_at(j, n)[0];
                let dev2 = cdev(a2, b2);
                if dev2 > worst_gf {
                    worst_gf = dev2;
                    eprintln!("GF les n={n} j={j}: scalar={a2:?} matrix={b2:?} dev={dev2:.3e}");
                }
            }
            for m in 0..=(NTAU as i64) {
                let a3 = ts_scalar.lmix.raw_block_at(0, m)[0];
                let b3 = ts_matrix.lmix.raw_block_at(0, m)[0];
                let dev3 = cdev(a3, b3);
                if dev3 > worst_gf {
                    worst_gf = dev3;
                    eprintln!("GF lmix n={n} m={m}: scalar={a3:?} matrix={b3:?} dev={dev3:.3e}");
                }
            }
            eprintln!("gf n={n}: worst so far {worst_gf:.3e}");
        }
        assert!(worst_gf < 1e-10, "gf scalar-vs-matrix deviation {worst_gf}");
    }

    /// Crash-hunting probe for the matrix real-time OCA kernels:
    /// fills a two-band fixture on the FULL contour (mat + ret + les +
    /// lmix) with smooth analytic data and evaluates one sigma and one
    /// gf diagram at a small timestep. Catches stride/dimension bugs in
    /// seconds (the full driver run takes ~35 min to reach the same
    /// code).
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn oca_tstp_matrix_probe() {
        use crate::hamilton::LocalHamiltonian::TwoBand;
        use kadanoffbaym::linalg::eigh;
        let nt: i64 = 8;
        let hs = build(2, SectorClass::SpinResolved);
        let ns = hs.ns();
        let fcontour = Contour::new(nt, NTAU as i64, 1, 1, 0.16, BETA);
        let dtau = BETA / NTAU as f64;
        let h = 0.02;
        let kf = |t: f64, w: f64| -(-w * t).exp() / (1.0 + (-BETA * w).exp());
        let bath_val = |t: f64| 0.09 * kf(t, -1.2) + 0.16 * kf(t, 0.8);
        // hybridizations with full contour data
        let mut deltas = Vec::new();
        let mut deltas_cc = Vec::new();
        for ix in 0..4 {
            let w = 0.5 * ix as f64 - 0.8;
            let mut d = ContourGreen::new(&fcontour, Sign::Fermi, Complex::ZERO);
            let mut dc = ContourGreen::new(&fcontour, Sign::Fermi, Complex::ZERO);
            for m in 0..=(NTAU as i64) {
                let t = dtau * m as f64;
                d.mat.block_mut(m as usize)[0] = Complex::new(bath_val(t), 0.0);
                dc.mat.block_mut(m as usize)[0] = Complex::new(-bath_val(BETA - t), 0.0);
            }
            for n in 0..=nt {
                for j in 0..=n {
                    let dt = (n - j) as f64 * h;
                    let v = Complex::new(0.0, -1.0) * Complex::new(w * dt, 0.0).exp() * 0.3;
                    d.ret.raw_block_at_mut(n, j)[0] = v;
                    dc.ret.raw_block_at_mut(n, j)[0] = v.conj();
                    let vl = Complex::new(w * dt, 0.0).exp() * 0.2;
                    d.less.raw_block_at_mut(j, n)[0] = vl;
                    dc.less.raw_block_at_mut(j, n)[0] = vl.conj();
                }
                for m in 0..=(NTAU as i64) {
                    let t = n as f64 * h;
                    let tau = m as f64 * dtau;
                    let v = Complex::new(w * (t + tau), 0.0).exp() * 0.1;
                    d.lmix.raw_block_at_mut(n, m)[0] = v;
                    dc.lmix.raw_block_at_mut(n, m)[0] = v.conj();
                }
            }
            deltas.push(d);
            deltas_cc.push(dc);
        }
        let ham = TwoBand {
            mu: 0.0,
            u: vec![2.0; (nt + 2) as usize],
            j: vec![0.4; (nt + 2) as usize],
            delta: vec![0.8; (nt + 2) as usize],
            bz: vec![0.0; (nt + 2) as usize],
        };
        // propagators with full contour data per sector
        let mut pp_g = Vec::new();
        for sec in 0..ns {
            let h0 = ham.sector_block(&hs, sec, -1);
            let dim = hs.sectors[sec].dim;
            let (evals, _) = eigh(&h0);
            let csec = Contour::new(nt, NTAU as i64, dim as i64, dim as i64, 0.16, BETA);
            let mut g = PseudoGreen::new(&csec, hs.sectors[sec].sig, Complex::ZERO);
            for m in 0..=(NTAU as i64) {
                for i in 0..dim {
                    let x = -(dtau * m as f64 * evals[i]).exp();
                    g.mat.block_mut(m as usize)[i * dim + i] = Complex::new(x, 0.0);
                }
            }
            for n in 0..=nt {
                for j in 0..=n {
                    let dt = (n - j) as f64 * h;
                    for i in 0..dim {
                        let e = evals[i % dim];
                        g.ret.raw_block_at_mut(n, j)[i * dim + i] =
                            Complex::new(0.0, -1.0) * Complex::new(e * dt, 0.0).exp() * 0.5;
                        g.less.raw_block_at_mut(j, n)[i * dim + i] =
                            Complex::new(e * dt, 0.0).exp() * 0.3;
                    }
                }
                for m in 0..=(NTAU as i64) {
                    for i in 0..dim {
                        let e = evals[i % dim];
                        g.lmix.raw_block_at_mut(n, m)[i * dim + i] =
                            Complex::new(e * (n as f64 * h + m as f64 * dtau), 0.0).exp() * 0.2;
                    }
                }
            }
            pp_g.push(g);
        }
        let mut pp_ints = Vec::new();
        for (orb, spin) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
            let c = hs.flavor(orb, spin, false);
            let a = hs.flavor(orb, spin, true);
            let ix = 2 * orb + spin;
            pp_ints.push(Hybridization {
                delta: &deltas[ix],
                op_cre_fl: a,
                op_ann_fl: c,
                fermion: true,
                backward: false,
            });
            pp_ints.push(Hybridization {
                delta: &deltas_cc[ix],
                op_cre_fl: c,
                op_ann_fl: a,
                fermion: true,
                backward: true,
            });
        }
        let channels = oca::channels_with_bath(&pp_ints, None);
        let integ = Integrator::new(5);
        // sigma diagrams: evaluate each at a few timesteps
        let sig_diagrams = oca::build_sigma_diagrams_oca_channels(&hs, &channels);
        eprintln!("probe: {} sigma diagrams", sig_diagrams.len());
        let mut sig_out: Vec<PseudoGreen> = (0..ns)
            .map(|sec| {
                let dim = hs.sectors[sec].dim as i64;
                let csec = Contour::new(nt, NTAU as i64, dim, dim, 0.16, BETA);
                PseudoGreen::new(&csec, hs.sectors[sec].sig, Complex::ZERO)
            })
            .collect();
        for n in [0i64, 1, 2, 5, 7] {
            for (di, d) in sig_diagrams.iter().enumerate() {
                oca::add_sigma_tstp_oca(n, d, &pp_g, BETA, h, 5, &integ, &mut sig_out);
                // finiteness check per diagram: catches value bugs the
                // index panics cannot
                for sec in 0..ns {
                    let dim = hs.sectors[sec].dim;
                    for j in 0..=n {
                        for z in sig_out[sec].ret.raw_block_at(n, j) {
                            assert!(
                                z.re.is_finite() && z.im.is_finite(),
                                "sigma diagram {di} produced non-finite ret at n={n} sec={sec}"
                            );
                        }
                        for z in sig_out[sec].less.raw_block_at(j, n) {
                            assert!(
                                z.re.is_finite() && z.im.is_finite(),
                                "sigma diagram {di} produced non-finite less at n={n} sec={sec}"
                            );
                        }
                    }
                    for m in 0..=(NTAU as i64) {
                        for z in sig_out[sec].lmix.raw_block_at(n, m) {
                            assert!(
                                z.re.is_finite() && z.im.is_finite(),
                                "sigma diagram {di} produced non-finite lmix at n={n} sec={sec}"
                            );
                        }
                    }
                }
            }
            eprintln!("probe: sigma tstp {n} done (finite)");
        }
        // gf diagrams: evaluate at the same timesteps
        let verts: Vec<GfVertex> = (0..2usize)
            .map(|orb| GfVertex {
                op_ann_fl: hs.flavor(orb, 0, false),
                op_cre_fl: hs.flavor(orb, 0, true),
            })
            .collect();
        let gf_diagrams = oca::build_gf_diagrams_oca_channels(&hs, &verts, &channels);
        eprintln!("probe: {} gf diagrams", gf_diagrams.len());
        for n in [0i64, 1, 2, 5, 7] {
            let tc = Contour::new(nt, NTAU as i64, 1, 1, 0.16, BETA);
            let mut ts =
                kadanoffbaym::types::GreenTimeStep::new(&tc, n, Sign::Fermi, Complex::ZERO);
            for (di, d) in gf_diagrams.iter().enumerate() {
                oca::add_gf_tstp_oca(n, d, &pp_g, BETA, h, 5, &integ, &mut ts);
                for j in 0..=n {
                    for z in ts.ret.raw_block_at(n, j) {
                        assert!(
                            z.re.is_finite() && z.im.is_finite(),
                            "gf diagram {di} non-finite ret n={n}"
                        );
                    }
                    for z in ts.less.raw_block_at(j, n) {
                        assert!(
                            z.re.is_finite() && z.im.is_finite(),
                            "gf diagram {di} non-finite less n={n}"
                        );
                    }
                }
                for m in 0..=(NTAU as i64) {
                    for z in ts.lmix.raw_block_at(0, m) {
                        assert!(
                            z.re.is_finite() && z.im.is_finite(),
                            "gf diagram {di} non-finite lmix n={n}"
                        );
                    }
                }
            }
            eprintln!("probe: gf tstp {n} done (finite)");
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn oca_mat_twoband_matches_xca_m2() {
        use crate::hamilton::LocalHamiltonian::TwoBand;
        use kadanoffbaym::linalg::eigh;
        let hs = build(2, SectorClass::SpinResolved);
        let ns = hs.ns();
        let contour = Contour::new(4, NTAU as i64, 1, 1, 0.08, BETA);
        let dtau = BETA / NTAU as f64;
        let kf = |t: f64, w: f64| -(-w * t).exp() / (1.0 + (-BETA * w).exp());
        let bath_val = |t: f64| 0.09 * kf(t, -1.2) + 0.16 * kf(t, 0.8);
        let mut deltas = Vec::new();
        let mut deltas_cc = Vec::new();
        for _ in 0..4 {
            let mut d = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
            let mut dc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
            for i in 0..=NTAU {
                let t = dtau * i as f64;
                d.mat.block_mut(i)[0] = Complex::new(bath_val(t), 0.0);
                dc.mat.block_mut(i)[0] = Complex::new(-bath_val(BETA - t), 0.0);
            }
            deltas.push(d);
            deltas_cc.push(dc);
        }
        let ham = TwoBand {
            mu: 0.0,
            u: vec![2.0; 6],
            j: vec![0.4; 6],
            delta: vec![0.8; 6],
            bz: vec![0.0; 6],
        };
        let mut pp_g = Vec::new();
        for sec in 0..ns {
            let h0 = ham.sector_block(&hs, sec, -1);
            let dim = hs.sectors[sec].dim;
            let (evals, _) = eigh(&h0);
            let csec = Contour::new(4, NTAU as i64, dim as i64, dim as i64, 0.08, BETA);
            let mut g = PseudoGreen::new(&csec, hs.sectors[sec].sig, Complex::ZERO);
            for m in 0..=NTAU {
                for i in 0..dim {
                    let x = -(dtau * m as f64 * evals[i]).exp();
                    g.mat.block_mut(m as usize)[i * dim + i] = Complex::new(x, 0.0);
                }
            }
            pp_g.push(g);
        }
        let mut pp_ints = Vec::new();
        for (orb, spin) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
            let c = hs.flavor(orb, spin, false);
            let a = hs.flavor(orb, spin, true);
            let ix = 2 * orb + spin;
            pp_ints.push(Hybridization {
                delta: &deltas[ix],
                op_cre_fl: a,
                op_ann_fl: c,
                fermion: true,
                backward: false,
            });
            pp_ints.push(Hybridization {
                delta: &deltas_cc[ix],
                op_cre_fl: c,
                op_ann_fl: a,
                fermion: true,
                backward: true,
            });
        }
        let empty = || -> Vec<PseudoGreen> {
            (0..ns)
                .map(|sec| {
                    let dim = hs.sectors[sec].dim as i64;
                    let csec = Contour::new(4, NTAU as i64, dim, dim, 0.08, BETA);
                    PseudoGreen::new(&csec, hs.sectors[sec].sig, Complex::ZERO)
                })
                .collect()
        };
        let integ = Integrator::new(5);
        let mut want = empty();
        nca::set_sigma_mat(&hs, &pp_g, &pp_ints, &mut want);
        let channels = oca::channels_with_bath(&pp_ints, None);
        for d in &oca::build_sigma_diagrams_oca_channels(&hs, &channels) {
            oca::add_sigma_mat_oca(d, &pp_g, BETA, 5, &integ, &mut want);
        }
        let fits = fit_channels(&channels, BETA, NTAU, 12, 6.0);
        let grids = sigma::Grids {
            ntau: NTAU,
            dtau,
            pp_g: &pp_g,
        };
        let mut got = empty();
        add_sigma_mat_xca(1, &hs, &channels, &fits, &grids, &integ, &mut got);
        add_sigma_mat_xca(2, &hs, &channels, &fits, &grids, &integ, &mut got);
        for sec in 0..ns {
            let mut worst = 0.0f64;
            let mut wi = 0;
            for m in 1..=NTAU {
                for q in 0..hs.sectors[sec].dim * hs.sectors[sec].dim {
                    let a = want[sec].mat.block(m)[q];
                    let b = got[sec].mat.block(m)[q];
                    let dev = (a - b).abs() / a.abs().max(1e-3);
                    if dev > worst {
                        worst = dev;
                        wi = m;
                    }
                }
            }
            eprintln!("twoband probe sector {sec}: max rel {worst:.3e} at m={wi}");
        }
    }

    /// Two-band m=1 anchor (the cell the m=2 twoband anchors
    /// imply but no dedicated gate covered): on the norb = 2
    /// Kanamori fixture, XCA(1) must reproduce the stock NCA
    /// Matsubara self-energy exactly (fit-residual level; the
    /// single-band twin asserts 1e-12, xca_order1_sigma_equals_
    /// nca_exactly).
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn xca_order1_twoband_matches_nca() {
        use crate::hamilton::LocalHamiltonian::TwoBand;
        use kadanoffbaym::linalg::eigh;
        let hs = build(2, SectorClass::SpinResolved);
        let ns = hs.ns();
        let contour = Contour::new(4, NTAU as i64, 1, 1, 0.08, BETA);
        let dtau = BETA / NTAU as f64;
        let kf = |t: f64, w: f64| -(-w * t).exp() / (1.0 + (-BETA * w).exp());
        let bath_val = |t: f64| 0.09 * kf(t, -1.2) + 0.16 * kf(t, 0.8);
        let mut deltas = Vec::new();
        let mut deltas_cc = Vec::new();
        for _ in 0..4 {
            let mut d = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
            let mut dc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
            for i in 0..=NTAU {
                let t = dtau * i as f64;
                d.mat.block_mut(i)[0] = Complex::new(bath_val(t), 0.0);
                dc.mat.block_mut(i)[0] = Complex::new(-bath_val(BETA - t), 0.0);
            }
            deltas.push(d);
            deltas_cc.push(dc);
        }
        let ham = TwoBand {
            mu: 0.0,
            u: vec![2.0; 6],
            j: vec![0.4; 6],
            delta: vec![0.8; 6],
            bz: vec![0.0; 6],
        };
        let mut pp_g = Vec::new();
        for sec in 0..ns {
            let h0 = ham.sector_block(&hs, sec, -1);
            let dim = hs.sectors[sec].dim;
            let (evals, _) = eigh(&h0);
            let csec = Contour::new(4, NTAU as i64, dim as i64, dim as i64, 0.08, BETA);
            let mut g = PseudoGreen::new(&csec, hs.sectors[sec].sig, Complex::ZERO);
            for m in 0..=NTAU {
                for i in 0..dim {
                    let x = -(dtau * m as f64 * evals[i]).exp();
                    g.mat.block_mut(m as usize)[i * dim + i] = Complex::new(x, 0.0);
                }
            }
            pp_g.push(g);
        }
        let mut pp_ints = Vec::new();
        for (orb, spin) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
            let c = hs.flavor(orb, spin, false);
            let a = hs.flavor(orb, spin, true);
            let ix = 2 * orb + spin;
            pp_ints.push(Hybridization {
                delta: &deltas[ix],
                op_cre_fl: a,
                op_ann_fl: c,
                fermion: true,
                backward: false,
            });
            pp_ints.push(Hybridization {
                delta: &deltas_cc[ix],
                op_cre_fl: c,
                op_ann_fl: a,
                fermion: true,
                backward: true,
            });
        }
        let empty = || -> Vec<PseudoGreen> {
            (0..ns)
                .map(|sec| {
                    let dim = hs.sectors[sec].dim as i64;
                    let csec = Contour::new(4, NTAU as i64, dim, dim, 0.08, BETA);
                    PseudoGreen::new(&csec, hs.sectors[sec].sig, Complex::ZERO)
                })
                .collect()
        };
        let integ = Integrator::new(5);
        let mut want = empty();
        nca::set_sigma_mat(&hs, &pp_g, &pp_ints, &mut want);
        let channels = oca::channels_with_bath(&pp_ints, None);
        let fits = fit_channels(&channels, BETA, NTAU, 12, 6.0);
        for f in &fits {
            assert!(f.resid < 1.0e-10, "SOE resid {}", f.resid);
        }
        let grids = sigma::Grids {
            ntau: NTAU,
            dtau,
            pp_g: &pp_g,
        };
        let mut got = empty();
        add_sigma_mat_xca(1, &hs, &channels, &fits, &grids, &integ, &mut got);
        for sec in 0..ns {
            let dim2 = hs.sectors[sec].dim * hs.sectors[sec].dim;
            for m in 0..=NTAU {
                for q in 0..dim2 {
                    let a = want[sec].mat.block(m)[q];
                    let b = got[sec].mat.block(m)[q];
                    assert!((a - b).abs() < 1e-12, "s {sec} m {m} q {q}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn xca_order1_sigma_equals_nca_exactly() {
        let fx = Fixture::new();
        let pp_ints = fx.pp_ints();
        let mut want = fx.empty_sigma();
        nca::set_sigma_mat(&fx.hs, &fx.pp_g, &pp_ints, &mut want);
        let mut got = fx.empty_sigma();
        let chans = oca::channels_with_bath(&pp_ints, None);
        let fits = fit_channels(&chans, BETA, NTAU, 12, 6.0);
        for f in &fits {
            assert!(f.resid < 1.0e-10, "SOE resid {}", f.resid);
        }
        let grids = sigma::Grids {
            ntau: NTAU,
            dtau: BETA / NTAU as f64,
            pp_g: &fx.pp_g,
        };
        let integ = Integrator::new(5);
        add_sigma_mat_xca(1, &fx.hs, &chans, &fits, &grids, &integ, &mut got);
        for s in 0..4 {
            for i in 0..=NTAU {
                let a = want[s].mat.block(i)[0];
                let b = got[s].mat.block(i)[0];
                assert!((a - b).abs() < 1e-12, "s {s} i {i}: {a} vs {b}");
            }
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn xca_order3_sigma_stays_on_scale() {
        // per-sector magnitude of the order-3 self-energy on the
        // atomic fixture: must stay on the V^6 beta^4 scale (O(1..10)),
        // guarding against exponential-growth regressions in the
        // kernel placement
        let fx = Fixture::new();
        let pp_ints = fx.pp_ints();
        let mut got = fx.empty_sigma();
        let chans = oca::channels_with_bath(&pp_ints, None);
        let fits = fit_channels(&chans, BETA, NTAU, 12, 6.0);
        let grids = sigma::Grids {
            ntau: NTAU,
            dtau: BETA / NTAU as f64,
            pp_g: &fx.pp_g,
        };
        let integ = Integrator::new(5);
        add_sigma_mat_xca(3, &fx.hs, &chans, &fits, &grids, &integ, &mut got);
        for s in 0..4 {
            let mx = (0..=NTAU)
                .map(|i| got[s].mat.block(i)[0].abs())
                .fold(0.0f64, f64::max);
            assert!(mx < 10.0, "sector {s} blew up: {mx}");
        }
    }

    /// Holstein low-order anchors: with a boson bath attached, XCA(1)
    /// must reproduce NCA+boson and XCA(2) must reproduce
    /// NCA+boson+OCA(channels) exactly (the calibrated equivalence,
    /// now carried by the freshly C++-gated Holstein kernels).
    #[test]
    #[allow(clippy::too_many_lines)]
    fn xca_holstein_m1_anchor_matches_nca_boson() {
        let fx = Fixture::new();
        let pp_ints = fx.pp_ints();

        // Holstein bath: free phonon at omega = 1, coupling g = 0.3
        let mut d0 = crate::boson::green_from_eps_phonon(4, NTAU, BETA, 0.02, 1.0);
        let gvec = vec![0.3; 4 + 2];
        crate::boson::scale_phonon(&mut d0, &gvec);
        let contour = Contour::new(4, NTAU as i64, 1, 1, 0.08, BETA);
        let mut d0_cc = ContourGreen::new(&contour, Sign::Bose, Complex::ZERO);
        crate::boson::set_bwd_from_fwd_boson_mat(&mut d0_cc, &d0);
        let bath = nca::BosonBath {
            d0,
            d0_cc,
            blocks: crate::boson::docc_blocks(&fx.hs),
        };

        let chans = oca::channels_with_bath(&pp_ints, Some(&bath));
        assert_eq!(chans.len(), 6, "4 fermion + 2 boson channels");
        let fits = fit_channels(&chans, BETA, NTAU, 12, 6.0);
        for (c, f) in fits.iter().enumerate() {
            eprintln!(
                "DBG fit chan {c}: poles={} resid={}",
                f.poles.len(),
                f.resid
            );
            assert!(f.resid < 1.0e-10, "SOE resid chan {c}: {}", f.resid);
        }
        let grids = sigma::Grids {
            ntau: NTAU,
            dtau: BETA / NTAU as f64,
            pp_g: &fx.pp_g,
        };
        let integ = Integrator::new(5);

        // --- m = 1 anchor: XCA(1) == NCA + boson ---
        let mut want = fx.empty_sigma();
        nca::set_sigma_mat(&fx.hs, &fx.pp_g, &pp_ints, &mut want);
        nca::add_boson_sigma_mat(&bath, &fx.pp_g, &mut want);
        let mut got = fx.empty_sigma();
        add_sigma_mat_xca(1, &fx.hs, &chans, &fits, &grids, &integ, &mut got);
        for s in 0..4 {
            for i in 0..=NTAU {
                let a = want[s].mat.block(i)[0];
                let b = got[s].mat.block(i)[0];
                assert!((a - b).abs() < 1e-10, "m1 anchor s {s} i {i}: {a} vs {b}");
            }
        }
    }

    /// m = 2 boson-inclusive self-energy anchor: XCA(1) + XCA(2)
    /// with the fermion + Holstein channel set vs NCA + native OCA
    /// (boson channels included).
    ///
    /// CLOSED (2026-11-24): the historical ~0.518 deviation was a
    /// pure sign flip of the boson-boson pair class. The per-pair
    /// diagnosis (`a1_boson_pair_decomposition`) gives XCA/OCA =
    /// -1.0000000 for b-b, +1 for f-f, and NO diagrams for mixed
    /// pairs (the diagonal docc vertex plus a single fermion line
    /// cannot close the sector chain), while the companion GF anchor
    /// passed at 7.2e-11 - so the channel/chain resolution was never
    /// the culprit. `add_sigma_mat_xca` now applies the calibrated
    /// OCA two-line fermion-loop sign for m = 2.
    ///
    /// Still open (separate item): orders >= 3 with boson lines (the
    /// mixed m = 3 R-cut calibration).
    #[test]
    fn xca_holstein_m2_anchor_matches_oca_boson() {
        let fx = Fixture::new();
        let pp_ints = fx.pp_ints();
        let mut d0 = crate::boson::green_from_eps_phonon(4, NTAU, BETA, 0.02, 1.0);
        let gvec = vec![0.3; 4 + 2];
        crate::boson::scale_phonon(&mut d0, &gvec);
        let contour = Contour::new(4, NTAU as i64, 1, 1, 0.08, BETA);
        let mut d0_cc = ContourGreen::new(&contour, Sign::Bose, Complex::ZERO);
        crate::boson::set_bwd_from_fwd_boson_mat(&mut d0_cc, &d0);
        let bath = nca::BosonBath {
            d0,
            d0_cc,
            blocks: crate::boson::docc_blocks(&fx.hs),
        };
        let chans = oca::channels_with_bath(&pp_ints, Some(&bath));
        let fits = fit_channels(&chans, BETA, NTAU, 12, 6.0);
        let grids = sigma::Grids {
            ntau: NTAU,
            dtau: BETA / NTAU as f64,
            pp_g: &fx.pp_g,
        };
        let integ = Integrator::new(5);
        let mut want2 = fx.empty_sigma();
        nca::set_sigma_mat(&fx.hs, &fx.pp_g, &pp_ints, &mut want2);
        nca::add_boson_sigma_mat(&bath, &fx.pp_g, &mut want2);
        let diagrams = oca::build_sigma_diagrams_oca_channels(&fx.hs, &chans);
        for d in &diagrams {
            oca::add_sigma_mat_oca(d, &fx.pp_g, BETA, 5, &integ, &mut want2);
        }
        let mut got2 = fx.empty_sigma();
        add_sigma_mat_xca(1, &fx.hs, &chans, &fits, &grids, &integ, &mut got2);
        add_sigma_mat_xca(2, &fx.hs, &chans, &fits, &grids, &integ, &mut got2);
        let mut max_rel = 0.0f64;
        for s in 0..4 {
            for i in 1..=NTAU {
                let a = want2[s].mat.block(i)[0];
                let b = got2[s].mat.block(i)[0];
                let scale = a.abs().max(1e-3);
                max_rel = max_rel.max((a - b).abs() / scale);
            }
        }
        eprintln!("[m2 holstein sigma anchor] max_rel = {max_rel:.3e}");
        assert!(
            max_rel < 1.0e-3,
            "m2 anchor max relative deviation {max_rel}"
        );
    }

    /// Bath with one Einstein phonon coupled through the docc vertex
    /// (`g = 0.3`), the bath of the Holstein m = 2 anchors below.
    fn holstein_bath(fx: &Fixture) -> nca::BosonBath {
        let mut d0 = crate::boson::green_from_eps_phonon(4, NTAU, BETA, 0.02, 1.0);
        let gvec = vec![0.3; 4 + 2];
        crate::boson::scale_phonon(&mut d0, &gvec);
        let contour = Contour::new(4, NTAU as i64, 1, 1, 0.08, BETA);
        let mut d0_cc = ContourGreen::new(&contour, Sign::Bose, Complex::ZERO);
        crate::boson::set_bwd_from_fwd_boson_mat(&mut d0_cc, &d0);
        nca::BosonBath {
            d0,
            d0_cc,
            blocks: crate::boson::docc_blocks(&fx.hs),
        }
    }

    /// m = 2 self-energy of the boson channels only (no fermion
    /// hybridization: the atomic leg of the Holstein real-time gate),
    /// XCA(1)+XCA(2) against the native NCA + OCA boson diagrams.
    #[test]
    fn xca_holstein_m2_anchor_boson_only() {
        let fx = Fixture::new();
        let bath = holstein_bath(&fx);
        let chans = oca::channels_with_bath(&[], Some(&bath));
        let fits = fit_channels_tol(
            &chans,
            BETA,
            NTAU,
            EQ_MAX_POLES_DEFAULT,
            W_MAX_DEFAULT,
            FIT_TOL_DEFAULT,
        );
        let grids = sigma::Grids {
            ntau: NTAU,
            dtau: BETA / NTAU as f64,
            pp_g: &fx.pp_g,
        };
        let integ = Integrator::new(5);
        let mut want = fx.empty_sigma();
        nca::add_boson_sigma_mat(&bath, &fx.pp_g, &mut want);
        let diagrams = oca::build_sigma_diagrams_oca_channels(&fx.hs, &chans);
        for d in &diagrams {
            oca::add_sigma_mat_oca(d, &fx.pp_g, BETA, 5, &integ, &mut want);
        }
        let mut got = fx.empty_sigma();
        add_sigma_mat_xca(1, &fx.hs, &chans, &fits, &grids, &integ, &mut got);
        add_sigma_mat_xca(2, &fx.hs, &chans, &fits, &grids, &integ, &mut got);
        let max_rel = sigma_rel_gap(&want, &got);
        eprintln!("[m2 holstein boson-only sigma anchor] max_rel = {max_rel:.3e}");
        assert!(max_rel < 1.0e-9, "max relative deviation {max_rel}");
    }

    /// Zero-hybridization fermion channels next to the boson ones must
    /// not switch the m = 2 diagrams on the other channels off (the
    /// atomic leg of the Holstein real-time gate passes the zero
    /// fermion channels on the Matsubara axis). Regression: an empty
    /// fit used to cancel the whole order >= 2 self-energy.
    #[test]
    fn xca_m2_keeps_boson_diagrams_with_zero_fermion_hybridization() {
        let fx = Fixture::new();
        let bath = holstein_bath(&fx);
        let contour = Contour::new(4, NTAU as i64, 1, 1, 0.08, BETA);
        let zero = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        let mut zero_ints = Vec::new();
        for spin in 0..2 {
            for backward in [false, true] {
                zero_ints.push(Hybridization {
                    delta: &zero,
                    op_cre_fl: fx.hs.flavor(0, spin, !backward),
                    op_ann_fl: fx.hs.flavor(0, spin, backward),
                    fermion: true,
                    backward,
                });
            }
        }
        let chans = oca::channels_with_bath(&zero_ints, Some(&bath));
        let fits = fit_channels_tol(
            &chans,
            BETA,
            NTAU,
            EQ_MAX_POLES_DEFAULT,
            W_MAX_DEFAULT,
            FIT_TOL_DEFAULT,
        );
        let grids = sigma::Grids {
            ntau: NTAU,
            dtau: BETA / NTAU as f64,
            pp_g: &fx.pp_g,
        };
        let integ = Integrator::new(5);
        let mut want = fx.empty_sigma();
        nca::set_sigma_mat(&fx.hs, &fx.pp_g, &zero_ints, &mut want);
        nca::add_boson_sigma_mat(&bath, &fx.pp_g, &mut want);
        let diagrams = oca::build_sigma_diagrams_oca_channels(&fx.hs, &chans);
        for d in &diagrams {
            oca::add_sigma_mat_oca(d, &fx.pp_g, BETA, 5, &integ, &mut want);
        }
        let mut got = fx.empty_sigma();
        add_sigma_mat_xca(1, &fx.hs, &chans, &fits, &grids, &integ, &mut got);
        add_sigma_mat_xca(2, &fx.hs, &chans, &fits, &grids, &integ, &mut got);
        let max_rel = sigma_rel_gap(&want, &got);
        eprintln!("[m2 holstein zero-fermion sigma anchor] max_rel = {max_rel:.3e}");
        assert!(
            max_rel < 1.0e-9,
            "zero-fermion channels changed the m = 2 self-energy: {max_rel}"
        );
    }

    /// Largest relative deviation of the Matsubara self-energies over
    /// all sectors and grid points.
    fn sigma_rel_gap(want: &[PseudoGreen], got: &[PseudoGreen]) -> f64 {
        let mut max_rel = 0.0f64;
        for s in 0..want.len() {
            for i in 1..=NTAU {
                let a = want[s].mat.block(i)[0];
                let b = got[s].mat.block(i)[0];
                let scale = a.abs().max(1e-3);
                max_rel = max_rel.max((a - b).abs() / scale);
            }
        }
        max_rel
    }

    /// m = 2 boson-inclusive GF anchor: XCA(2) with boson channels
    /// vs `nca::gf_mat` + `oca::add_gf_mat_oca` (boson included).
    /// Measured 7.2e-11 relative (OCA-part scale 1.6e2) - the GF
    /// chain/vertex handling of boson channels is correct. This is
    /// the probe that localized the old ~0.5 self-energy deviation to
    /// the sigma-side pair prefactor (fixed above by the m = 2
    /// fermion-loop sign).
    #[test]
    fn xca_gf_holstein_m2_anchor_matches_oca_boson() {
        let fx = Fixture::new();
        let pp_ints = fx.pp_ints();
        let mut d0 = crate::boson::green_from_eps_phonon(4, NTAU, BETA, 0.02, 1.0);
        let gvec = vec![0.3; 4 + 2];
        crate::boson::scale_phonon(&mut d0, &gvec);
        let contour = Contour::new(4, NTAU as i64, 1, 1, 0.08, BETA);
        let mut d0_cc = ContourGreen::new(&contour, Sign::Bose, Complex::ZERO);
        crate::boson::set_bwd_from_fwd_boson_mat(&mut d0_cc, &d0);
        let bath = nca::BosonBath {
            d0,
            d0_cc,
            blocks: crate::boson::docc_blocks(&fx.hs),
        };
        let chans = oca::channels_with_bath(&pp_ints, Some(&bath));
        let fits = fit_channels(&chans, BETA, NTAU, 12, 6.0);
        let verts = [GfVertex {
            op_ann_fl: fx.hs.flavor(0, 0, false),
            op_cre_fl: fx.hs.flavor(0, 0, true),
        }];
        let integ = Integrator::new(5);
        let nca_only = nca::gf_mat(&fx.hs, &fx.pp_g, &verts);
        let mut want = nca_only.clone();
        oca::add_gf_mat_oca(
            &fx.hs,
            &fx.pp_g,
            &verts,
            &pp_ints,
            Some(&bath),
            BETA,
            5,
            &integ,
            &mut want,
        );
        let grids = gf::Grids {
            ntau: NTAU,
            dtau: BETA / NTAU as f64,
            pp_g: &fx.pp_g,
        };
        let mut got = vec![vec![Complex::ZERO; NTAU + 1]; 1];
        gf::add_gf_mat_xca(1, &fx.hs, &chans, &fits, &verts, &grids, &integ, &mut got);
        gf::add_gf_mat_xca(2, &fx.hs, &chans, &fits, &verts, &grids, &integ, &mut got);
        let mut max_rel = 0.0f64;
        let mut max_dev = 0.0f64;
        let mut oca_scale = 0.0f64;
        for i in 1..=NTAU {
            let a = want[0][i];
            let b = got[0][i];
            max_dev = max_dev.max((a - b).abs());
            oca_scale = oca_scale.max((a - nca_only[0][i]).abs());
            max_rel = max_rel.max((a - b).abs() / a.abs().max(1e-3));
        }
        eprintln!(
            "[gf-boson anchor] OCA-part scale {oca_scale:.3e}, absolute dev {max_dev:.3e}, max_rel {max_rel:.3e}"
        );
        assert!(max_rel < 1.0e-3, "m2 GF+boson max relative deviation {max_rel}");
    }

    #[test]
    fn xca_order2_sigma_matches_oca() {
        let fx = Fixture::new();
        let pp_ints = fx.pp_ints();
        let mut want = fx.empty_sigma();
        nca::set_sigma_mat(&fx.hs, &fx.pp_g, &pp_ints, &mut want);
        let diagrams = oca::build_sigma_diagrams_oca(&fx.hs, &pp_ints);
        let integ = Integrator::new(5);
        for d in &diagrams {
            oca::add_sigma_mat_oca(d, &fx.pp_g, BETA, 5, &integ, &mut want);
        }

        let mut got = fx.empty_sigma();
        let chans = oca::channels_with_bath(&pp_ints, None);
        let fits = fit_channels(&chans, BETA, NTAU, 12, 6.0);
        let grids = sigma::Grids {
            ntau: NTAU,
            dtau: BETA / NTAU as f64,
            pp_g: &fx.pp_g,
        };
        add_sigma_mat_xca(1, &fx.hs, &chans, &fits, &grids, &integ, &mut got);
        add_sigma_mat_xca(2, &fx.hs, &chans, &fits, &grids, &integ, &mut got);

        let mut max_rel = 0.0f64;
        for s in 0..4 {
            for i in 1..=NTAU {
                let a = want[s].mat.block(i)[0];
                let b = got[s].mat.block(i)[0];
                let scale = a.abs().max(1e-3);
                max_rel = max_rel.max((a - b).abs() / scale);
            }
        }
        // residual deviation: Gregory warmup rows (the OCA reference
        // refines them by polynomial interpolation), converging to
        // zero with ntau (verified: ~ntau^-2)
        assert!(max_rel < 1.0e-3, "max relative deviation {max_rel}");
    }

    #[test]
    fn xca_order1_gf_equals_nca_exactly() {
        let fx = Fixture::new();
        let pp_ints = fx.pp_ints();
        let cua = fx.hs.flavor(0, 0, false);
        let cuc = fx.hs.flavor(0, 0, true);
        let verts = [GfVertex {
            op_ann_fl: cua,
            op_cre_fl: cuc,
        }];
        let want = nca::gf_mat(&fx.hs, &fx.pp_g, &verts);
        let mut got = vec![vec![Complex::ZERO; NTAU + 1]; 1];
        let chans = oca::channels_with_bath(&pp_ints, None);
        let fits = fit_channels(&chans, BETA, NTAU, 12, 6.0);
        let grids = gf::Grids {
            ntau: NTAU,
            dtau: BETA / NTAU as f64,
            pp_g: &fx.pp_g,
        };
        let integ = Integrator::new(5);
        gf::add_gf_mat_xca(1, &fx.hs, &chans, &fits, &verts, &grids, &integ, &mut got);
        for i in 0..=NTAU {
            let a = want[0][i];
            let b = got[0][i];
            assert!((a - b).abs() < 1e-12, "i {i}: {a} vs {b}");
        }
    }

    #[test]
    fn xca_gf2_formula_matches_oca_reference() {
        // Direct 2D quadrature of ONE live mixed-spin chain with the
        // paper formula and the nQIST chain conventions: the value
        // must agree with the OCA reference diagram-for-diagram (up
        // to the net prefactor -1 of the channel).
        // line = dn-fwd, external ops on up, sector path 1->3->2->0->1
        let fx = Fixture::new();
        let hs = &fx.hs;
        let cua = hs.flavor(0, 0, false);
        let cuc = hs.flavor(0, 0, true);
        let cda = hs.flavor(0, 1, false);
        let cdc = hs.flavor(0, 1, true);
        let verts = [GfVertex {
            op_ann_fl: cua,
            op_cre_fl: cuc,
        }];
        // nqist OCA reference for the dn-fwd channel only
        let pp_ints_dn = [Hybridization {
            delta: &fx.delta,
            op_cre_fl: cdc,
            op_ann_fl: cda,
            fermion: true,
            backward: false,
        }];
        let integ = Integrator::new(5);
        let mut want = vec![vec![Complex::ZERO; NTAU + 1]; 1];
        oca::add_gf_mat_oca(
            hs,
            &fx.pp_g,
            &verts,
            &pp_ints_dn,
            None,
            BETA,
            5,
            &integ,
            &mut want,
        );

        // direct 2D: chain sectors [1, 3, 2, 0, 1]; ops from blocks
        let g = |s: usize, i: usize| fx.pp_g[s].mat.block(i)[0];
        let delta = |i: usize| fx.delta.mat.block(i)[0];
        let (op_cre0, _) = hs.ops[cuc].block(1usize).unwrap();
        let _ = op_cre0;
        let oc = hs.ops[cuc].block(1).unwrap().1.get(0, 0); // sigma 1 -> 3
        let o1 = hs.ops[cda].block(3).unwrap().1.get(0, 0); // 3 -> 2
        let oa = hs.ops[cua].block(2).unwrap().1.get(0, 0); // 2 -> 0
        let o3 = hs.ops[cdc].block(0).unwrap().1.get(0, 0); // 0 -> 1
        let dtau = BETA / NTAU as f64;
        for m in [8usize, 32, 64, 90] {
            // tau1 = j in [0, m], t3 = m + l in [m, ntau]
            let mut acc = Complex::ZERO;
            for j in 0..=m {
                let w1 = integ.giw.at(m, j);
                if w1 == 0.0 {
                    continue;
                }
                let n3 = NTAU - m;
                for l in 0..=n3 {
                    let w3 = integ.giw.at(n3, l);
                    if w3 == 0.0 {
                        continue;
                    }
                    let i3 = m + l;
                    // circle product: op_cre@0 G3(tau1) o1 G2(tau-tau1)
                    // oa G0(t3-tau) o3 G1(beta-t3), Delta(t3 - tau1)
                    let val = oc
                        * g(3, j)
                        * o1
                        * g(2, m - j)
                        * oa
                        * g(0, i3 - m)
                        * o3
                        * g(1, NTAU - i3)
                        * delta(i3 - j);
                    acc += Complex::new(w1 * w3, 0.0) * val;
                }
            }
            let direct = acc * Complex::new(dtau * dtau, 0.0);
            // the net channel prefactor of this diagram is -1: the
            // direct quadrature must reproduce the OCA reference
            // diagram-for-diagram
            let want_val = want[0][m];
            let scaled = -direct;
            let rel = (scaled - want_val).abs() / want_val.abs().max(1e-6);
            assert!(rel < 1.0e-8, "i{m}: direct {scaled:?} vs oca {want_val:?}");
        }
    }

    #[test]
    fn xca_order2_gf_matches_oca() {
        let fx = Fixture::new();
        let pp_ints = fx.pp_ints();
        let cua = fx.hs.flavor(0, 0, false);
        let cuc = fx.hs.flavor(0, 0, true);
        let verts = [GfVertex {
            op_ann_fl: cua,
            op_cre_fl: cuc,
        }];
        let mut want = nca::gf_mat(&fx.hs, &fx.pp_g, &verts);
        let integ = Integrator::new(5);
        oca::add_gf_mat_oca(
            &fx.hs, &fx.pp_g, &verts, &pp_ints, None, BETA, 5, &integ, &mut want,
        );

        let mut got = vec![vec![Complex::ZERO; NTAU + 1]; 1];
        let chans = oca::channels_with_bath(&pp_ints, None);
        let fits = fit_channels(&chans, BETA, NTAU, 12, 6.0);
        let grids = gf::Grids {
            ntau: NTAU,
            dtau: BETA / NTAU as f64,
            pp_g: &fx.pp_g,
        };
        gf::add_gf_mat_xca(1, &fx.hs, &chans, &fits, &verts, &grids, &integ, &mut got);
        gf::add_gf_mat_xca(2, &fx.hs, &chans, &fits, &verts, &grids, &integ, &mut got);

        let mut max_rel = 0.0f64;
        for i in 0..=NTAU {
            let a = want[0][i];
            let b = got[0][i];
            let scale = a.abs().max(1e-3);
            max_rel = max_rel.max((a - b).abs() / scale);
        }
        // the residual deviation lives in the Gregory warmup rows
        // (the OCA reference refines them with polynomial
        // interpolation, the xca backbone uses plain rows) and
        // converges to zero with ntau (verified: ~ntau^-1)
        assert!(max_rel < 5.0e-3, "max relative deviation {max_rel}");
    }
}

#[cfg(test)]
mod probe_test {
    use super::sigma::probe;

    /// Warmup-row regression: with the sub-grid startup the XCA(2)
    /// single-pair evaluation reproduces the native OCA diagram to
    /// SOE-fit precision at every sampled point, warmup region
    /// included (the historical truncation showed ratios up to 2.6
    /// at i = 1).
    #[test]
    fn xca2_single_pair_matches_oca_incl_warmup() {
        let fx = probe::fixture();
        for (c1, c2) in [(0usize, 2usize)] {
            let xca = probe::xca2_single(&fx, [c1, c2]);
            let (oca, picked) = probe::oca_single(&fx, c1, c2);
            assert_eq!(picked.len(), 1);
            for s in 0..4usize {
                for i in 1..=12usize {
                    let a = oca[s].mat.block(i)[0];
                    let b = xca[s].mat.block(i)[0];
                    if a.re.abs() > 1e-13 {
                        assert!(
                            (b - a).abs() < 1e-7 * a.abs().max(1e-4),
                            "s{s}i{i}: oca={a:?} xca={b:?}"
                        );
                    }
                }
            }
        }
    }

    /// A1 diagnosis (2026-11-24): per ordered channel-pair comparison
    /// of XCA(2) against the native OCA diagram over the fermion +
    /// Holstein channel set (6 channels). Prints dev/scale/rel per
    /// channel class so the bad pair rules can be isolated.
    #[test]
    #[ignore = "A1 diagnosis driver (prints the per-class deviation table)"]
    fn a1_boson_pair_decomposition() {
        use crate::nca;
        use crate::oca;
        use kadanoffbaym::complex::Complex;
        use kadanoffbaym::types::{Contour, ContourGreen, Sign};

        const BETA: f64 = 5.0;
        const NTAU: usize = 96;

        let fx = probe::fixture();
        let pp_ints = &fx.pp_ints;
        let mut d0 = crate::boson::green_from_eps_phonon(4, NTAU, BETA, 0.02, 1.0);
        crate::boson::scale_phonon(&mut d0, &vec![0.3; 4 + 2]);
        let contour = Contour::new(4, NTAU as i64, 1, 1, 0.08, BETA);
        let mut d0_cc = ContourGreen::new(&contour, Sign::Bose, Complex::ZERO);
        crate::boson::set_bwd_from_fwd_boson_mat(&mut d0_cc, &d0);
        let bath = nca::BosonBath {
            d0,
            d0_cc,
            blocks: crate::boson::docc_blocks(&fx.hs),
        };
        let chans = oca::channels_with_bath(&pp_ints, Some(&bath));
        let class = |c: usize| -> &'static str {
            let bos = c >= pp_ints.len();
            match (bos, chans[c].backward) {
                (false, false) => "f-fwd",
                (false, true) => "f-bwd",
                (true, false) => "b-fwd",
                (true, true) => "b-bwd",
            }
        };
        for c1 in 0..chans.len() {
            for c2 in 0..chans.len() {
                let xca = probe::xca2_single_ch(&fx, &chans, [c1, c2]);
                let (oca, picked) = probe::oca_single_ch(&fx, &chans, c1, c2);
                let mut dev = 0.0f64;
                let mut plus = 0.0f64;
                let mut scale = 0.0f64;
                let mut ratio = Complex::ZERO;
                let mut worst = 0.0f64;
                for s in 0..4usize {
                    for i in 1..=NTAU {
                        let a = oca[s].mat.block(i)[0];
                        let b = xca[s].mat.block(i)[0];
                        dev = dev.max((a - b).abs());
                        plus = plus.max((a + b).abs());
                        scale = scale.max(a.abs());
                        if (a - b).abs() > worst && a.abs() > 1e-9 {
                            worst = (a - b).abs();
                            ratio = b / a;
                        }
                    }
                }
                println!(
                    "pair ({c1},{c2}) {}/{} picked={} dev={dev:.3e} plus={plus:.3e} scale={scale:.3e} rel={:.3e} ratio={:?}",
                    class(c1),
                    class(c2),
                    picked.len(),
                    dev / scale.max(1e-6),
                    ratio
                );
            }
        }
    }

        /// Kernel-level diagnostics for the continuous Bethe OCA gap: the
    /// native NCA+OCA and the XCA(1)+XCA(2) Sigma / G are compared on
    /// the same converged state at several pole budgets, plus a probe of
    /// the SOE fit smoothness (the XCA output under a tiny change of
    /// the hybridization).
    mod continuous_bethe_diag {
        use crate::dmft::{get_gf_verts, get_pp_ints, run_equilibrium_probe};
        use crate::hamilton::LocalHamiltonian::SingleBand;
        use crate::hilbert::{HilbertSpace, SectorClass, build};
        use crate::nca::{self, GfVertex, Hybridization};
        use crate::oca::{self, OcaChannel};
        use crate::solver::ImpuritySolver;
        use crate::xca::soe::SoeFit;
        use crate::xca::{EQ_MAX_POLES_DEFAULT, W_MAX_DEFAULT, fit_channels_tol};
        use kadanoffbaym::complex::Complex;
        use kadanoffbaym::types::{Contour, MATSUBARA, PseudoGreen, Sign};
        use kadanoffbaym::weights::Integrator;

        #[test]
        fn kernel_gap_continuous_bethe() {
            kernel_gap(300, 250, 1.0e-9);
            kernel_gap(300, 250, 1.0e-10);
        }


        /// Tolerance scan of the SOE fit on the converged Bethe
        /// hybridization: pole count, achieved error and cost vs the
        /// requested tolerance (the pole count is an output).
        #[test]
        fn fit_tolerance_scan_continuous_bethe() {
            let (nt, ntau, kt, beta, h) = (300usize, 250usize, 5i64, 5.0, 0.02);
            let hs = build(1, SectorClass::SpinResolved);
            let mut u = vec![3.5; nt + 2];
            u[0] = 3.0;
            let ham = SingleBand {
                mu: 0.0,
                u,
                eps_up: vec![0.0; nt + 2],
                eps_do: vec![0.0; nt + 2],
            };
            let mut solver =
                ImpuritySolver::new(nt, ntau, beta, h, kt as usize, 1, hs, ham);
            let (log, _, delta, delta_cc) =
                run_equilibrium_probe(&mut solver, 0.5, 1e-8, 300, |_, _, _, _, _, _| {});
            assert!(log.converged, "NCA equilibrium did not converge");
            let pp_ints = get_pp_ints(&delta, &delta_cc, &solver);
            let channels = oca::channels_with_bath(&pp_ints, None);
            let scale = (0..=ntau)
                .map(|m| delta.mat.block(m)[0].abs())
                .fold(0.0f64, f64::max);
            println!("Delta max |value| on grid: {scale:.3e}");
            for tol in [
                1.0e-4f64, 1.0e-6, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12,
            ] {
                let t0 = std::time::Instant::now();
                let fits = fit_channels_tol(
                    &channels,
                    beta,
                    ntau,
                    EQ_MAX_POLES_DEFAULT,
                    W_MAX_DEFAULT,
                    tol,
                );
                let dt = t0.elapsed().as_secs_f64();
                let n: Vec<usize> = fits.iter().map(|f| f.poles.len()).collect();
                let max_resid = fits.iter().map(|f| f.resid).fold(0.0f64, f64::max);
                let rms = (fits.iter().map(|f| f.resid_l2).fold(0.0f64, f64::max)
                    / (ntau + 1) as f64)
                    .sqrt();
                println!(
                    "tol={tol:.0e}: poles {n:?}, max-abs err {max_resid:.3e}, rms {rms:.3e}, {dt:.3}s"
                );
            }
        }

        fn zero_ppgf(nt: usize, ntau: usize, beta: f64, dim: usize, sig: Sign) -> PseudoGreen {
            let c = Contour::new(
                nt as i64,
                ntau as i64,
                dim as i64,
                dim as i64,
                nt as f64 * 0.02,
                beta,
            );
            PseudoGreen::new(&c, sig, Complex::ZERO)
        }

        /// XCA Sigma = sum_{m=1..=order} add_sigma_mat_xca(m) on zero.
        fn xca_sigma_sum(
            order: usize,
            nt: usize,
            ntau: usize,
            beta: f64,
            kt: i64,
            hs: &HilbertSpace,
            channels: &[OcaChannel],
            fits: &[SoeFit],
            pp_g: &[PseudoGreen],
        ) -> Vec<PseudoGreen> {
            let integ = Integrator::new(kt);
            let grids = crate::xca::sigma::Grids {
                ntau,
                dtau: beta / ntau as f64,
                pp_g,
            };
            let mut out: Vec<PseudoGreen> = hs
                .sectors
                .iter()
                .map(|s| zero_ppgf(nt, ntau, beta, s.dim, s.sig))
                .collect();
            for m in 1..=order {
                crate::xca::sigma::add_sigma_mat_xca(
                    m,
                    hs,
                    channels,
                    fits,
                    &grids,
                    &integ,
                    &mut out,
                );
            }
            out
        }

        /// XCA G = sum_{m=1..=order} add_gf_mat_xca(m) on zero.
        fn xca_gf_sum(
            order: usize,
            ntau: usize,
            beta: f64,
            kt: i64,
            hs: &HilbertSpace,
            channels: &[OcaChannel],
            fits: &[SoeFit],
            verts: &[GfVertex],
            pp_g: &[PseudoGreen],
        ) -> Vec<Vec<Complex>> {
            let integ = Integrator::new(kt);
            let grids = crate::xca::gf::Grids {
                ntau,
                dtau: beta / ntau as f64,
                pp_g,
            };
            let mut out = vec![vec![Complex::ZERO; ntau + 1]; verts.len()];
            for m in 1..=order {
                crate::xca::gf::add_gf_mat_xca(
                    m,
                    hs,
                    channels,
                    fits,
                    verts,
                    &grids,
                    &integ,
                    &mut out,
                );
            }
            out
        }

        /// Native order-2 G (NCA + OCA vertex diagrams): the
        /// `Solver::Ppsc` order-2 branch of `ImpuritySolver::get_spgf_mat`.
        fn native_gf_order2(
            hs: &HilbertSpace,
            pp_g: &[PseudoGreen],
            verts: &[GfVertex],
            pp_ints: &[Hybridization],
            beta: f64,
            kt: i64,
        ) -> Vec<Vec<Complex>> {
            let mut out = nca::gf_mat(hs, pp_g, verts);
            let integ = Integrator::new(kt);
            let mut oca_part = vec![vec![Complex::ZERO; out[0].len()]; verts.len()];
            oca::add_gf_mat_oca(
                hs,
                pp_g,
                verts,
                pp_ints,
                None,
                beta,
                kt,
                &integ,
                &mut oca_part,
            );
            for (o, add) in out.iter_mut().zip(&oca_part) {
                for q in 0..o.len() {
                    o[q] += add[q];
                }
            }
            out
        }

        fn is_warmup(m: usize, ntau: usize) -> bool {
            m < 8 || m + 8 > ntau
        }

        /// (worst, warmup-rows worst, scale = max |a|)
        fn sigma_gap(a: &[PseudoGreen], b: &[PseudoGreen]) -> (f64, f64, f64) {
            let ntau = a[0].ntau;
            let (mut worst, mut warm, mut scale) = (0.0f64, 0.0f64, 0.0f64);
            for (x, y) in a.iter().zip(b) {
                for m in 0..=ntau {
                    let d = (x.mat.block(m)[0] - y.mat.block(m)[0]).abs();
                    worst = worst.max(d);
                    if is_warmup(m, ntau) {
                        warm = warm.max(d);
                    }
                    scale = scale.max(x.mat.block(m)[0].abs());
                }
            }
            (worst, warm, scale)
        }

        /// (worst, warmup-rows worst, scale = max |a|, argmax m)
        fn gf_gap(a: &[Vec<Complex>], b: &[Vec<Complex>]) -> (f64, f64, f64, usize) {
            let ntau = a[0].len() - 1;
            let (mut worst, mut warm, mut scale, mut worst_m) = (0.0f64, 0.0f64, 0.0f64, 0usize);
            for (x, y) in a.iter().zip(b) {
                for m in 0..=ntau {
                    let d = (x[m] - y[m]).abs();
                    if d > worst {
                        worst = d;
                        worst_m = m;
                    }
                    if is_warmup(m, ntau) {
                        warm = warm.max(d);
                    }
                    scale = scale.max(x[m].abs());
                }
            }
            (worst, warm, scale, worst_m)
        }

        fn kernel_gap(nt: usize, ntau: usize, tol: f64) {
            let kt: i64 = 5;
            let beta = 5.0;
            let h = 0.02;
            let hs = build(1, SectorClass::SpinResolved);
            let mut u = vec![3.5; nt + 2];
            u[0] = 3.0;
            let ham = SingleBand {
                mu: 0.0,
                u,
                eps_up: vec![0.0; nt + 2],
                eps_do: vec![0.0; nt + 2],
            };
            let mut solver =
                ImpuritySolver::new(nt, ntau, beta, h, kt as usize, 1, hs.clone(), ham);
            let (log, _, mut delta, mut delta_cc) =
                run_equilibrium_probe(&mut solver, 0.5, 1e-8, 300, |_, _, _, _, _, _| {});
            assert!(log.converged, "NCA equilibrium did not converge");
            let verts = get_gf_verts(&solver);
            let integ = Integrator::new(kt);

            // native order-2 state consistent with the returned Delta
            let pp_ints = get_pp_ints(&delta, &delta_cc, &solver);
            let channels = oca::channels_with_bath(&pp_ints, None);
            solver.pp_step(MATSUBARA, &pp_ints);
            let diagrams = oca::build_sigma_diagrams_oca_channels(&hs, &channels);
            for d in &diagrams {
                oca::add_sigma_mat_oca(d, &solver.pp_g, beta, kt, &integ, &mut solver.pp_sigma);
            }
            solver.solve_dyson(MATSUBARA);
            // re-evaluate the native NCA + OCA Sigma on the same final
            // pp_g the XCA Sigma is evaluated on (apples to apples)
            nca::set_sigma_mat(&hs, &solver.pp_g, &pp_ints, &mut solver.pp_sigma);
            for d in &diagrams {
                oca::add_sigma_mat_oca(d, &solver.pp_g, beta, kt, &integ, &mut solver.pp_sigma);
            }

            let fits = fit_channels_tol(
                &channels,
                beta,
                ntau,
                EQ_MAX_POLES_DEFAULT,
                W_MAX_DEFAULT,
                tol,
            );
            let resid = |fs: &[SoeFit]| {
                fs.iter()
                    .map(|f| f.resid)
                    .fold(0.0f64, f64::max)
            };
            let npoles = |fs: &[SoeFit]| -> Vec<usize> {
                fs.iter().map(|f| f.poles.len()).collect()
            };
            let sig_xca = xca_sigma_sum(
                2,
                nt,
                ntau,
                beta,
                kt,
                &hs,
                &channels,
                &fits,
                &solver.pp_g,
            );
            let (sg, sg_warm, sg_scale) = sigma_gap(&solver.pp_sigma, &sig_xca);
            let g_nat = native_gf_order2(&hs, &solver.pp_g, &verts, &pp_ints, beta, kt);
            let g_xca = xca_gf_sum(
                2,
                ntau,
                beta,
                kt,
                &hs,
                &channels,
                &fits,
                &verts,
                &solver.pp_g,
            );
            let (gg, gg_warm, gg_scale, gg_m) = gf_gap(&g_nat, &g_xca);
            println!(
                "[ntau={ntau} tol={tol:.0e}] fit resid max {:.3e}, poles used {:?}",
                resid(&fits),
                npoles(&fits)
            );
            println!(
                "  Sigma gap {sg:.3e} (warmup rows {sg_warm:.3e}), scale {sg_scale:.3e}"
            );
            println!(
                "  G gap    {gg:.3e} (warmup rows {gg_warm:.3e}) at m={gg_m}, scale {gg_scale:.3e}"
            );

            // smoothness scan: relative noise on the forward
            // hybridization, from far below the 1e-8 DMFT tolerance up
            // to the change between stalled DMFT iterations (~1e-5)
            let orig: Vec<Complex> = (0..=ntau).map(|m| delta.mat.block(m)[0]).collect();
            for (k, eps) in [1e-10f64, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4].into_iter().enumerate() {
                for (m, v) in orig.iter().enumerate() {
                    let n = (((m + 7 * k) as f64 * 0.618_033_988_749_894_9f64).fract() - 0.5) * 2.0;
                    delta.mat.block_mut(m)[0] = *v * Complex::new(1.0 + eps * n, 0.0);
                }
                let pp_ints_p = get_pp_ints(&delta, &delta_cc, &solver);
                let channels_p = oca::channels_with_bath(&pp_ints_p, None);
                let fits_p = fit_channels_tol(
                    &channels_p,
                    beta,
                    ntau,
                    EQ_MAX_POLES_DEFAULT,
                    W_MAX_DEFAULT,
                    tol,
                );
                let same_poles = fits_p.iter().zip(&fits).all(|(a, b)| a.poles == b.poles);
                let sig_xca_p = xca_sigma_sum(
                    2,
                    nt,
                    ntau,
                    beta,
                    kt,
                    &hs,
                    &channels_p,
                    &fits_p,
                    &solver.pp_g,
                );
                let g_xca_p = xca_gf_sum(
                    2,
                    ntau,
                    beta,
                    kt,
                    &hs,
                    &channels_p,
                    &fits_p,
                    &verts,
                    &solver.pp_g,
                );
                let g_nat_p = native_gf_order2(&hs, &solver.pp_g, &verts, &pp_ints_p, beta, kt);
                let (ds, _, _) = sigma_gap(&sig_xca, &sig_xca_p);
                let (dg, _, _, _) = gf_gap(&g_xca, &g_xca_p);
                let (dg_nat, _, _, _) = gf_gap(&g_nat, &g_nat_p);
                println!(
                    "  noise {eps:.0e}: same pole set = {same_poles}, poles {:?}, resid {:.3e}, dSigma xca {ds:.3e}, dG xca {dg:.3e}, dG native {dg_nat:.3e}",
                    npoles(&fits_p),
                    resid(&fits_p)
                );
            }
            assert!(sg < 1e-1 && gg < 1e-1, "sanity: kernels disagree");
        }
    }
}
