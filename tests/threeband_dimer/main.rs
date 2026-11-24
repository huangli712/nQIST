// Project : nQIST
// File    : tests/threeband_dimer/main.rs
// Purpose : three-band benchmark: crystal-field-split Kanamori
//           impurity (norb = 3), each orbital coupled to its own bath
//           site (spin-up channels only), against exact
//           diagonalization of the 512-state Fock space through the ED
//           module (`nqist::ed`: impurity bits 0..5 = 1u, 2u, 3u, 1d,
//           2d, 3d plus the bath bits 6, 7, 8).
//
//           Mirrors `tests/xca_twoband_dimer`: equilibrium Matsubara
//           Green's functions against `EdReference`, real-time
//           propagators (stationarity + J quench) against
//           `EdEvolution`. The impurity energies of the exact
//           reference come from `LocalHamiltonian::sector_block`
//           itself, so the ED Hamiltonian is convention-identical to
//           the solver's.

use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{Contour, ContourGreen, Sign};
use nqist::dmft::set_bwd_from_fwd_mat;
use nqist::hamilton::LocalHamiltonian;
use nqist::hamilton::LocalHamiltonian::ThreeBand;
use nqist::hilbert::{HilbertSpace, SectorClass, build};
use nqist::nca::{GfVertex, Hybridization};
use nqist::solver::ImpuritySolver;

const BETA: f64 = 5.0;
const NTAU: usize = 64;
const NT: usize = 6;
const U_HUB: f64 = 2.0;
const J_HUND: f64 = 0.4;
const CRYSTAL: f64 = 0.8; // 0.5 * delta * (n1 - n2 - n3)
const V: [f64; 3] = [0.3, 0.35, 0.3];
const EPS_B: [f64; 3] = [0.5, -0.8, 0.9];

/// Constant three-band Hamiltonian series of length `len`.
fn mk_ham(u: f64, j: f64, delta: f64, len: usize) -> LocalHamiltonian {
    ThreeBand {
        mu: 0.0,
        u: vec![u; len],
        j: vec![j; len],
        delta: vec![delta; len],
        bz: vec![0.0; len],
    }
}

/// Dimer bath: one site per spin-up orbital (mode `o` = orbital
/// `o`), couplings scaled by `vscale`.
fn bath_scaled(vscale: f64) -> [nqist::ed::BathCoupling; 3] {
    [
        nqist::ed::BathCoupling {
            imp_mode: 0,
            eps: EPS_B[0],
            v: V[0] * vscale,
        },
        nqist::ed::BathCoupling {
            imp_mode: 1,
            eps: EPS_B[1],
            v: V[1] * vscale,
        },
        nqist::ed::BathCoupling {
            imp_mode: 2,
            eps: EPS_B[2],
            v: V[2] * vscale,
        },
    ]
}

/// Dimer bath at full hybridization.
fn bath() -> [nqist::ed::BathCoupling; 3] {
    bath_scaled(1.0)
}

fn k_tau(tau: f64, w: f64) -> f64 {
    -(-w * tau).exp() / (1.0 + (-BETA * w).exp())
}

/// Spin-up Green's-function vertices of the three orbitals.
fn up_verts(hs: &HilbertSpace) -> [GfVertex; 3] {
    [
        GfVertex {
            op_ann_fl: hs.flavor(0, 0, false),
            op_cre_fl: hs.flavor(0, 0, true),
        },
        GfVertex {
            op_ann_fl: hs.flavor(1, 0, false),
            op_cre_fl: hs.flavor(1, 0, true),
        },
        GfVertex {
            op_ann_fl: hs.flavor(2, 0, false),
            op_cre_fl: hs.flavor(2, 0, true),
        },
    ]
}

/// Solve the three-band impurity at expansion order `order` on the
/// Matsubara axis and return the spin-up Green's functions of the
/// three orbitals (`G[o][m]`, `m = 0..=ntau`).
fn solve_threeband(
    ham: &LocalHamiltonian,
    order: usize,
    pp_iters: usize,
    vscale: f64,
    ntau: usize,
) -> Vec<Vec<f64>> {
    let hs = build(3, SectorClass::SpinResolved);
    let mut solver = ImpuritySolver::new(NT, ntau, BETA, 0.02, 5, order, hs.clone(), ham.clone());
    solver.solve_atomic();

    let contour = Contour::new(NT as i64, ntau as i64, 1, 1, 0.1, BETA);
    let dtau = BETA / ntau as f64;
    // per-orbital single-level baths
    let mut fwd = Vec::new();
    let mut cc = Vec::new();
    for o in 0..3 {
        let mut d = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        for m in 0..=ntau {
            let t = dtau * m as f64;
            d.mat.block_mut(m)[0] =
                Complex::new((V[o] * vscale).powi(2) * k_tau(t, EPS_B[o]), 0.0);
        }
        let mut c = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        set_bwd_from_fwd_mat(&mut c, &d);
        fwd.push(d);
        cc.push(c);
    }

    let verts = up_verts(&hs);
    let mut last: Vec<Vec<f64>> = Vec::new();
    let t0 = std::time::Instant::now();
    for it in 0..pp_iters {
        let mut ints = Vec::new();
        for o in 0..3 {
            ints.push(Hybridization {
                delta: &fwd[o],
                op_cre_fl: hs.flavor(o, 0, true),
                op_ann_fl: hs.flavor(o, 0, false),
                fermion: true,
                backward: false,
            });
            ints.push(Hybridization {
                delta: &cc[o],
                op_cre_fl: hs.flavor(o, 0, false),
                op_ann_fl: hs.flavor(o, 0, true),
                fermion: true,
                backward: true,
            });
        }
        solver.pp_step(-1, &ints);
        let g = solver.get_spgf_mat(&verts, &ints);
        let cur: Vec<Vec<f64>> = g.iter().map(|row| row.iter().map(|z| z.re).collect()).collect();
        if it == 0 || it % 10 == 0 {
            eprintln!(
                "threeband-o{order} pp iter {it:3} (t {:>7.1?}): G(0) = {:.6} {:.6} {:.6}",
                t0.elapsed(),
                cur[0][0],
                cur[1][0],
                cur[2][0]
            );
        }
        if !last.is_empty() {
            let diff = last
                .iter()
                .flatten()
                .zip(cur.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            if diff < 1.0e-10 {
                eprintln!("threeband-o{order} converged after {it} iterations");
                last = cur;
                break;
            }
        }
        last = cur;
    }
    last
}

fn main() {
    println!("run via cargo test --test threeband_dimer");
}

/// Diagnostic: NCA error profile along tau for several hybridization
/// strengths (locates a V-independent offset versus a diagrammatic
/// error); solver and ED bath are scaled together.
#[test]
#[ignore = "diagnostic profile printer"]
fn threeband_diag_profile() {
    use nqist::ed::EdReference;
    let hs = build(3, SectorClass::SpinResolved);
    let ham = mk_ham(U_HUB, J_HUND, CRYSTAL, NT + 2);
    let dtau = BETA / NTAU as f64;
    for (vscale, ntau) in [(1.0f64, NTAU), (0.5, NTAU), (0.25, NTAU), (0.1, NTAU), (1.0, 128)] {
        let ed = EdReference::build(&ham, &hs, &bath_scaled(vscale), &[], BETA);
        let g = solve_threeband(&ham, 1, 200, vscale, ntau);
        let step = ntau / 16;
        for o in 0..3 {
            let mut worst = (0.0f64, 0usize);
            for m in 1..ntau {
                let d = (g[o][m] - ed.greens_tau(o, BETA * m as f64 / ntau as f64)).abs();
                if d > worst.0 {
                    worst = (d, m);
                }
            }
            let mut line = String::new();
            for m in (0..=ntau).step_by(step) {
                let tau = if ntau == NTAU { dtau * m as f64 } else { BETA * m as f64 / ntau as f64 };
                line.push_str(&format!(" {:+.5}/{:+.5}", g[o][m], ed.greens_tau(o, tau)));
            }
            println!("vscale {vscale} ntau {ntau} orb {o}: worst {:+.3e} at m {} | solver/ed:{}", worst.0, worst.1, line);
        }
    }
}

// ------------------------------------------------------------------
// ED module self-checks on the 512-state space
// ------------------------------------------------------------------

#[test]
fn threeband_ed_selfchecks() {
    use nqist::ed::EdReference;
    let hs = build(3, SectorClass::SpinResolved);
    let ham = mk_ham(U_HUB, J_HUND, CRYSTAL, 2);
    let ed = EdReference::build(&ham, &hs, &bath(), &[], BETA);
    assert_eq!(ed.dim(), 512);
    assert_eq!(ed.nmodes(), 9);
    // canonical identity G^>(0) - G^<(0) = -i on every mode
    for m in 0..9 {
        let d = ed.greens_gtr(m, 0.0) - ed.greens_les(m, 0.0) + Complex::new(0.0, 1.0);
        assert!((d * d.conj()).re < 1e-18, "mode {m}: {d:?}");
    }
    // tau -> 0+ limit: G(0+) = -<c c^dag> = -(1 - <n>) per impurity mode
    for m in 0..6 {
        let n = ed.expectation(&[(m, true), (m, false)]).re;
        let g0 = ed.greens_tau(m, 1e-9);
        assert!(
            (g0 + (1.0 - n)).abs() < 1e-6,
            "mode {m}: G(0+) {g0} vs -(1 - n) {}",
            -(1.0 - n)
        );
    }
    // particle-hole symmetric point (delta = 0, zero-energy bath
    // levels): every mode is half filled and the equivalent orbitals
    // 1 and 3 (same V) have identical Green's functions
    let bath_ph = [
        nqist::ed::BathCoupling {
            imp_mode: 0,
            eps: 0.0,
            v: V[0],
        },
        nqist::ed::BathCoupling {
            imp_mode: 1,
            eps: 0.0,
            v: V[1],
        },
        nqist::ed::BathCoupling {
            imp_mode: 2,
            eps: 0.0,
            v: V[2],
        },
    ];
    let ed_ph = EdReference::build(&mk_ham(U_HUB, J_HUND, 0.0, 2), &hs, &bath_ph, &[], BETA);
    for m in 0..9 {
        let n = ed_ph.expectation(&[(m, true), (m, false)]).re;
        assert!((n - 0.5).abs() < 1e-10, "PH: mode {m} <n> = {n}");
    }
    for m in [1usize, 16, 32, 63] {
        let tau = BETA * m as f64 / 64.0;
        let (g1, g3) = (ed_ph.greens_tau(0, tau), ed_ph.greens_tau(2, tau));
        assert!((g1 - g3).abs() < 1e-10, "PH: G1 vs G3 at tau {tau}: {g1} vs {g3}");
    }
}

/// Atomic-limit cross-check: with zero hybridization the solver's
/// equilibrium G(tau) IS the atomic propagator (up to the Matsubara
/// Dyson quadrature error) and must equal the ED Lehmann result. The
/// residual must also shrink under grid refinement, which identifies
/// it as quadrature error.
fn atomic_crosscheck_error(ntau: usize) -> f64 {
    use nqist::ed::EdReference;
    let hs = build(3, SectorClass::SpinResolved);
    let ham = mk_ham(U_HUB, J_HUND, CRYSTAL, NT + 2);
    let contour = Contour::new(NT as i64, ntau as i64, 1, 1, 0.1, BETA);
    let z0 = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let mut ints = Vec::new();
    for o in 0..3 {
        ints.push(Hybridization {
            delta: &z0,
            op_cre_fl: hs.flavor(o, 0, true),
            op_ann_fl: hs.flavor(o, 0, false),
            fermion: true,
            backward: false,
        });
        ints.push(Hybridization {
            delta: &z0,
            op_cre_fl: hs.flavor(o, 0, false),
            op_ann_fl: hs.flavor(o, 0, true),
            fermion: true,
            backward: true,
        });
    }
    let verts = up_verts(&hs);
    let mut solver = ImpuritySolver::new(NT, ntau, BETA, 0.02, 5, 1, hs.clone(), ham.clone());
    solver.solve_atomic();
    solver.pp_step(-1, &ints);
    solver.pp_step(-1, &ints);
    let g = solver.get_spgf_mat(&verts, &ints);
    let ed = EdReference::build(&ham, &hs, &[], &[], BETA);
    let mut worst = 0.0f64;
    for (orb, row) in g.iter().enumerate() {
        for i in 1..ntau {
            let tau = BETA * i as f64 / ntau as f64;
            worst = worst.max((row[i].re - ed.greens_tau(orb, tau)).abs());
        }
    }
    worst
}

#[test]
fn threeband_atomic_limit_crosscheck() {
    let coarse = atomic_crosscheck_error(NTAU);
    let fine = atomic_crosscheck_error(2 * NTAU);
    println!(
        "atomic crosscheck [three-band]: worst |G_solver - G_ed| = {coarse:.3e} (ntau {NTAU}), {fine:.3e} (ntau {})",
        2 * NTAU
    );
    assert!(
        coarse < 5.0e-6,
        "atomic: quadrature-level expected, got {coarse:.3e}"
    );
    assert!(
        fine < 0.2 * coarse,
        "atomic: residual must shrink under grid refinement: {coarse:.3e} -> {fine:.3e}"
    );
}

// ------------------------------------------------------------------
// equilibrium dimer: NCA / OCA against ED
// ------------------------------------------------------------------

fn dimer_errors(ham: &LocalHamiltonian, order: usize, vscale: f64) -> [f64; 3] {
    use nqist::ed::EdReference;
    let hs = build(3, SectorClass::SpinResolved);
    let ed = EdReference::build(ham, &hs, &bath_scaled(vscale), &[], BETA);
    let g = solve_threeband(ham, order, 200, vscale, NTAU);
    let dtau = BETA / NTAU as f64;
    let mut err = [0.0f64; 3];
    for o in 0..3 {
        for m in 1..NTAU {
            err[o] = err[o].max((g[o][m] - ed.greens_tau(o, dtau * m as f64)).abs());
        }
    }
    err
}

/// Hybridization-scaling diagnostic: the NCA dimer error follows the
/// leading missing diagrams (~ v^4); a structural defect in the
/// three-band vertex algebra would not follow the scaling.
#[test]
fn threeband_dimer_nca_scaling() {
    let ham = mk_ham(U_HUB, J_HUND, CRYSTAL, NT + 2);
    let full = dimer_errors(&ham, 1, 1.0);
    let half = dimer_errors(&ham, 1, 0.5);
    println!(
        "scaling [three-band]: NCA err {full:?} at V, {half:?} at V/2, ratios {:.4} {:.4} {:.4}",
        half[0] / full[0],
        half[1] / full[1],
        half[2] / full[2]
    );
    for o in 0..3 {
        assert!(
            half[o] < 0.5 * full[o],
            "orbital {o}: errors must shrink with the hybridization: {full:?} vs {half:?}"
        );
    }
}

/// Equilibrium gate: NCA against the ED module's 512-state exact
/// reference. The budget is absolute and calibrated on the validated
/// run (the bold series is non-monotonic at these couplings,
/// mirroring the two-band dimer).
#[test]
fn threeband_dimer_nca_vs_ed() {
    let ham = mk_ham(U_HUB, J_HUND, CRYSTAL, NT + 2);
    let err = dimer_errors(&ham, 1, 1.0);
    println!(
        "threeband order 1: max |G-Ged| orb1 {:+.6e} orb2 {:+.6e} orb3 {:+.6e}",
        err[0], err[1], err[2]
    );
    assert!(
        err.iter().all(|e| *e < 2.0e-2),
        "NCA: {err:?}"
    );
}

/// Equilibrium gate: OCA against the same exact reference. HEAVY
/// (the Matsubara OCA kernel costs ~30 s per pseudo-particle
/// iteration at ntau = 64) - ignored for the default suite; run with
/// `cargo test --release --test threeband_dimer -- --ignored`. The
/// OCA error exceeds the NCA error at these couplings (non-monotonic
/// bold series, as in the two-band dimer), so the budget is absolute.
#[test]
#[ignore = "heavy: three-band OCA equilibrium dimer (~3 min); run manually"]
fn threeband_dimer_oca_vs_ed() {
    let ham = mk_ham(U_HUB, J_HUND, CRYSTAL, NT + 2);
    let err = dimer_errors(&ham, 2, 1.0);
    println!(
        "threeband order 2: max |G-Ged| orb1 {:+.6e} orb2 {:+.6e} orb3 {:+.6e}",
        err[0], err[1], err[2]
    );
    assert!(
        err.iter().all(|e| *e < 4.0e-2),
        "OCA: {err:?}"
    );
}

// ------------------------------------------------------------------
// real-time dimer: stationarity + J quench against EdEvolution
// ------------------------------------------------------------------

fn check_threeband_dimer_realtime(order: usize, nt_rt: usize, ntau: usize, with_stationarity: bool) {
    use std::time::Instant;

    use nqist::ed::{EdEvolution, hamiltonian_at};

    // NQIST_H: diagnostic real-time step override (keeps the physical
    // window nt_rt * 0.025 fixed, so nt_rt scales as 1/h)
    let h_rt: f64 = std::env::var("NQIST_H")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.025);
    let nt_rt = if std::env::var_os("NQIST_H").is_some() {
        (nt_rt as f64 * 0.025 / h_rt).round() as usize
    } else {
        nt_rt
    };
    let (h_rt, kt) = (h_rt, 5usize);
    let (j0, j1) = (0.4f64, 0.55f64);
    let hs = build(3, SectorClass::SpinResolved);
    let ham_eq = mk_ham(U_HUB, j0, CRYSTAL, nt_rt + 2);
    let mut j = vec![j0; nt_rt + 2];
    for k in 1..j.len() {
        j[k] = j1;
    }
    let ham_q = ThreeBand {
        mu: 0.0,
        u: vec![U_HUB; nt_rt + 2],
        j,
        delta: vec![CRYSTAL; nt_rt + 2],
        bz: vec![0.0; nt_rt + 2],
    };

    // static analytic single-level up-only bath on the full contour:
    // TTI propagators generated by the KB dyson solvers (Sigma = 0,
    // H = eps), scaled by v^2 (validated dimer-driver construction of
    // tests/xca_twoband_dimer)
    let contour = Contour::new(nt_rt as i64, ntau as i64, 1, 1, h_rt * nt_rt as f64, BETA);
    let mk_level = |eps: f64| -> ContourGreen {
        let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        let sig = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        let mut hf = kadanoffbaym::types::TimeFunction::zeros(nt_rt, 1, 1);
        for tstp in -1..=(nt_rt as i64) {
            hf.block_mut(tstp)[0] = Complex::new(eps, 0.0);
        }
        kadanoffbaym::dyson::dyson_mat(
            &mut g,
            &sig,
            0.0,
            &hf,
            BETA,
            kt as i64,
            kadanoffbaym::base::MatsubaraMethod::Fixpoint,
            true,
            None,
            1e-12,
        );
        kadanoffbaym::dyson::dyson_start(&mut g, 0.0, &hf, &sig, BETA, h_rt, kt as i64);
        for tstp in (kt as i64 + 1)..=(nt_rt as i64) {
            kadanoffbaym::dyson::dyson_timestep(tstp, &mut g, 0.0, &hf, &sig, BETA, h_rt, kt as i64);
        }
        g
    };
    let lvl: Vec<ContourGreen> = EPS_B.iter().map(|&e| mk_level(e)).collect();
    let scale_green = |g: &ContourGreen, w: f64| -> ContourGreen {
        let wc = Complex::new(w, 0.0);
        let mut out = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        for n in 0..=(nt_rt as i64) {
            for jj in 0..=n {
                out.ret.block_mut(out.ret.idx(n, jj))[0] = g.ret.block(g.ret.idx(n, jj))[0] * wc;
                out.less.block_mut(out.less.idx(jj, n))[0] =
                    g.less.block(g.less.idx(jj, n))[0] * wc;
            }
            for m in 0..=(ntau as i64) {
                out.lmix.raw_block_at_mut(n, m)[0] = g.lmix.raw_block_at(n, m)[0] * wc;
            }
        }
        for m in 0..=(ntau as i64) {
            out.mat.block_mut(m as usize)[0] = g.mat.block(m as usize)[0] * wc;
        }
        out
    };
    let mk_pair = |lvl: &ContourGreen, v: f64| -> (ContourGreen, ContourGreen) {
        let fwd = scale_green(lvl, v * v);
        let mut cc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        set_bwd_from_fwd_mat(&mut cc, &fwd);
        for t in 0..=(nt_rt as i64) {
            nqist::dmft::set_bwd_from_fwd_tstp(t, &mut cc, &fwd);
        }
        (fwd, cc)
    };
    let verts = up_verts(&hs);

    let run = |case: &str, ham: &LocalHamiltonian, vscale: f64| -> ImpuritySolver {
        let started = Instant::now();
        eprintln!("[threeband m={order} {case}] starting equilibrium, nt={nt_rt}, ntau={ntau}");
        let mut solver = ImpuritySolver::new(
            nt_rt,
            ntau,
            BETA,
            h_rt,
            kt,
            order,
            hs.clone(),
            ham.clone(),
        );
        solver.solve_atomic();
        let zero = || ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        let pairs: Vec<(ContourGreen, ContourGreen)> = (0..3)
            .map(|o| {
                if vscale == 0.0 {
                    (zero(), zero())
                } else {
                    mk_pair(&lvl[o], V[o] * vscale)
                }
            })
            .collect();
        let mut pp_ints = Vec::new();
        for o in 0..3 {
            pp_ints.push(Hybridization {
                delta: &pairs[o].0,
                op_cre_fl: hs.flavor(o, 0, true),
                op_ann_fl: hs.flavor(o, 0, false),
                fermion: true,
                backward: false,
            });
            pp_ints.push(Hybridization {
                delta: &pairs[o].1,
                op_cre_fl: hs.flavor(o, 0, false),
                op_ann_fl: hs.flavor(o, 0, true),
                fermion: true,
                backward: true,
            });
        }
        let mut last: Vec<f64> = Vec::new();
        for iter in 1..=200 {
            solver.pp_step(-1, &pp_ints);
            let g = solver.get_spgf_mat(&verts, &pp_ints);
            let cur: Vec<f64> = g.iter().flatten().map(|z| z.re).collect();
            let converged = !last.is_empty()
                && last.iter().zip(&cur).all(|(a, b)| (a - b).abs() < 1.0e-10);
            if iter == 1 || iter % 10 == 0 || converged {
                eprintln!(
                    "[threeband m={order} {case}] equilibrium iteration {iter}/200 finished, converged={converged}, elapsed={:.2?}",
                    started.elapsed()
                );
            }
            if converged {
                break;
            }
            last = cur;
        }
        eprintln!(
            "[threeband m={order} {case}] starting real-time propagation after {:.2?}",
            started.elapsed()
        );
        solver.set_t0();
        let ints: &[Hybridization] = if vscale == 0.0 { &[] } else { &pp_ints[..] };
        for tstp in 0..=(kt as i64) {
            solver.pp_step(tstp, ints);
        }
        for tstp in (kt + 1) as i64..=(nt_rt as i64) {
            solver.extrapolate_timestep(tstp - 1);
            solver.pp_step(tstp, ints);
        }
        eprintln!(
            "[threeband m={order} {case}] done in {:.2?}",
            started.elapsed()
        );
        solver
    };

    // -- leg 1: no-quench stationarity (TTI)
    if with_stationarity {
        // v = 0 bisect: the atomic leg must be exactly stationary
        let mut solver = run("atomic", &ham_eq, 0.0);
        solver.update_density_matrix(0);
        let obs0 = nqist::observable::update_exp_vals_three_band(&solver, 0);
        let mut drift = 0.0f64;
        for tstp in 1..=nt_rt as i64 {
            solver.update_density_matrix(tstp);
            let obs = nqist::observable::update_exp_vals_three_band(&solver, tstp);
            let parts = [
                (obs.n1 - obs0.n1).abs(),
                (obs.n2 - obs0.n2).abs(),
                (obs.n3 - obs0.n3).abs(),
                (obs.e_int - obs0.e_int).abs(),
                (obs.q - obs0.q).abs(),
            ];
            eprintln!("atomic TTI tstp {tstp}: n1 {:+.3e} n2 {:+.3e} n3 {:+.3e} e_int {:+.3e} Q {:+.3e}", parts[0], parts[1], parts[2], parts[3], parts[4]);
            drift = drift.max(parts[0]).max(parts[1]).max(parts[2]).max(parts[3]);
        }
        println!("realtime TTI drift [three-band, v=0]: {drift:.3e}");
        assert!(drift < 1e-8, "atomic leg must be stationary: {drift:.3e}");

        let mut solver = run("steady", &ham_eq, 1.0);
        solver.update_density_matrix(0);
        let obs0 = nqist::observable::update_exp_vals_three_band(&solver, 0);
        let mut drift = 0.0f64;
        for tstp in 1..=nt_rt as i64 {
            solver.update_density_matrix(tstp);
            let obs = nqist::observable::update_exp_vals_three_band(&solver, tstp);
            let row_drift = 0.0f64
                .max((obs.n1 - obs0.n1).abs())
                .max((obs.n2 - obs0.n2).abs())
                .max((obs.n3 - obs0.n3).abs())
                .max((obs.e_int - obs0.e_int).abs());
            drift = drift.max(row_drift);
        }
        println!("realtime TTI drift [three-band] (n1, n2, n3, e_int): {drift:.3e}");
        // The OCA leg runs at ntau = 64 (the OCA matrix kernels cost
        // ~35 s per real-time step already at ntau = 32, so ntau = 128
        // would take ~10 min per step), where the tstp = 0 Matsubara
        // quadrature error is 2.6e-6 (`threeband_atomic_limit_crosscheck`);
        // that leg is gated at 5e-6, the NCA leg (ntau = 128) keeps 1e-6.
        // The atomic leg and the ED quench gate stay untouched.
        let steady_gate = if order >= 2 { 5e-6 } else { 1e-6 };
        assert!(
            drift < steady_gate,
            "TTI stationarity broken: {drift:.3e} (gate {steady_gate:.0e})"
        );
    }

    // -- leg 2: J quench vs the exact unitary evolution
    let ed_start = Instant::now();
    eprintln!("[threeband m={order}] building 512-state ED quench reference");
    let h0 = hamiltonian_at(&ham_q, &hs, &bath(), &[], -1);
    let h1 = hamiltonian_at(&ham_q, &hs, &bath(), &[], 0);
    let ev = EdEvolution::build_quench(&h0, &h1, nt_rt, h_rt, BETA, &[0, 1, 2]);
    eprintln!(
        "[threeband m={order}] ED reference ready in {:.2?}",
        ed_start.elapsed()
    );
    let solver = run("J quench", &ham_q, 1.0);
    let points: Vec<(usize, usize)> = if nt_rt <= 12 {
        vec![(5, 0), (nt_rt, 0), (nt_rt, nt_rt / 2), (nt_rt, nt_rt - 1)]
    } else {
        vec![
            (5, 0),
            (nt_rt / 2, 0),
            (nt_rt / 2, nt_rt / 4),
            (nt_rt, 0),
            (nt_rt, nt_rt / 2),
            (nt_rt, nt_rt - 1),
        ]
    };
    let mut w = 0.0f64;
    for mode in [0usize, 1, 2] {
        let points_mode = points.clone();
        for (n, j) in points_mode {
            let gf = solver.get_spgf(&verts, n as i64, &[]);
            let g_ret = gf[mode].ret.raw_block_at(n as i64, j as i64)[0];
            let g_les = gf[mode].less.raw_block_at(j as i64, n as i64)[0];
            for (tag, a, b) in [
                ("ret", g_ret, ev.greens_ret(mode, n, j)),
                ("les", g_les, ev.greens_les(mode, j, n)),
            ] {
                let d = (a - b).abs();
                if d > w {
                    println!(
                        "quench [threeband m={order}] worst so far: mode {mode} {tag} ({n},{j}): {d:.3e}"
                    );
                }
                w = w.max(d);
            }
        }
    }
    println!("realtime J-quench order {order} [three-band]: worst |G - G_ed| {w:.3e}");
    // absolute budget calibrated on the validated NCA run (worst
    // 6.55e-2 at the longest separation t = 0.6; the error grows with
    // the separation like in the two-band dimer)
    assert!(
        w < 8.0e-2,
        "quench error {w:.3e} exceeds the three-band budget"
    );
}

/// Tier 2 gate: three-band real-time NCA with the full dimer. Two
/// legs: (i) no-quench stationarity pre-gate - the real-time stack
/// (analytic single-level bath fill + set_t0/extrapolate/pp_step)
/// propagated from the converged Matsubara state must keep the
/// equal-time observables static to quadrature level (TTI); (ii) J
/// quench 0.4 -> 0.55 vs the exact unitary evolution (EdEvolution)
/// at NCA accuracy.
///
/// Grid notes (measured): the tstp = 0 observables carry the
/// Matsubara quadrature error (2.6e-6 at 64 points, 4.4e-8 at 128,
/// see `threeband_atomic_limit_crosscheck`), and the real-time
/// quadrature noise of the extensive `e_int` (|E| ~ 11, 16 sectors)
/// is 1.7e-8 at h = 0.05 but 3.1e-10 at h = 0.025. The two-band gates
/// (1e-8 atomic, 1e-6 steady) are therefore run at ntau = 128 and
/// h = 0.025 (window 0.6, nt = 24).
#[test]
fn threeband_dimer_realtime_nca_vs_ed() {
    check_threeband_dimer_realtime(1, 24, 128, true);
}

/// Tier 2 gate: three-band real-time OCA with the full dimer
/// (stationarity + J quench vs EdEvolution). Runs on the ntau = 64
/// grid with the OCA steady-state gate of 5e-6 (see the gate note in
/// `check_threeband_dimer_realtime`); the atomic leg (1e-8) and the
/// quench budget are those of the NCA gate. HEAVY (measured 55m46s:
/// ~80 s per real-time step in the OCA matrix kernels at ntau = 64)
/// - ignored for the default suite; run with
/// `cargo test --release --test threeband_dimer -- --ignored`.
#[test]
#[ignore = "heavy: three-band OCA real-time dimer (~1 h); run manually"]
fn threeband_dimer_realtime_oca_vs_ed() {
    check_threeband_dimer_realtime(2, 24, 64, true);
}
