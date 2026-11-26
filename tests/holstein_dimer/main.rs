// Project : nQIST
// File    : tests/holstein_dimer/main.rs
// Purpose : Hubbard-Holstein benchmark: single-band Hubbard impurity
//           with an Einstein phonon coupled to the doublon (the PPSC
//           Holstein vertex, D0 = g D_free g), both spins hybridizing
//           with one bath site each, against exact diagonalization of
//           the joint fermion + phonon space (16 Fock states x nph
//           phonon levels) through the ED module.
//
//           Boson-line convention: `BosonBath` carries the forward and
//           the backward boson channel, which for bosons are the same
//           physical line (`D(beta - tau) = D(tau)`). The solver
//           counts the vertex once (`ImpuritySolver::set_boson_bath`
//           scales each channel by 1/2), so the bath coupling `g` is
//           the Hamiltonian coupling `lambda` in
//           `H_ph = omega0 b^dag b + lambda docc (b + b^dag)`: the
//           X = b + b^dag propagator of
//           `boson::green_from_eps_phonon` scaled by g(t) g(t') is
//           the correlator of `g docc X`. The PPSC Holstein driver
//           counts the line twice; its coupling maps to
//           `g = sqrt(2) g_ppsc` here (see tests/ppsc_ref). The
//           frequency is chosen != 1 so that a wrong normalization
//           (g omega0, g / omega0, ...) gives a visibly different
//           exact reference.

use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{Contour, ContourGreen, Sign};
use nqist::dmft::set_bwd_from_fwd_mat;
use nqist::ed::{BathCoupling, EdReference, PhononCoupling, PhononSpec};
use nqist::hamilton::LocalHamiltonian;
use nqist::hamilton::LocalHamiltonian::SingleBand;
use nqist::hilbert::{HilbertSpace, SectorClass, build};
use nqist::nca::{BosonBath, GfVertex, Hybridization};
use nqist::solver::{ImpuritySolver, Solver};

const BETA: f64 = 5.0;
const NTAU: usize = 64;
const NT: usize = 6;
const U_HUB: f64 = 2.0;
const V: f64 = 0.4;
const EPS_B: f64 = 0.5;
const OMEGA: f64 = 0.7;
const G_EP: f64 = 0.4;
const NPH: usize = 12;

/// Doublon string `n_up n_dn` (the Holstein vertex).
const DOCC: [(usize, bool); 4] = [(0, true), (0, false), (1, true), (1, false)];

/// Particle-hole symmetric single-band Hubbard series (`eps = 0`).
fn mk_ham(u: f64, len: usize) -> LocalHamiltonian {
    SingleBand {
        mu: 0.0,
        u: vec![u; len],
        eps_up: vec![0.0; len],
        eps_do: vec![0.0; len],
    }
}

/// Dimer bath: one site per spin, couplings scaled by `vscale`.
fn bath_scaled(vscale: f64) -> [BathCoupling; 2] {
    [
        BathCoupling {
            imp_mode: 0,
            eps: EPS_B,
            v: V * vscale,
        },
        BathCoupling {
            imp_mode: 1,
            eps: EPS_B,
            v: V * vscale,
        },
    ]
}

/// Dimer bath at full hybridization.
fn bath() -> [BathCoupling; 2] {
    bath_scaled(1.0)
}

/// Phonon mode coupled to the doublon through the constant `lambda`.
fn phonon(lambda: f64, nph: usize, len: usize) -> PhononSpec {
    PhononSpec {
        nph,
        omega: OMEGA,
        couplings: vec![PhononCoupling {
            ops: DOCC.to_vec(),
            lambda: vec![lambda; len],
        }],
    }
}

/// Phonon mode with a time-dependent coupling series `gvec`.
fn phonon_series(gvec: &[f64]) -> PhononSpec {
    PhononSpec {
        nph: NPH,
        omega: OMEGA,
        couplings: vec![PhononCoupling {
            ops: DOCC.to_vec(),
            lambda: gvec.to_vec(),
        }],
    }
}

/// Exact reference with the phonon coupled to the doublon with the
/// constant `lambda` on `nph` retained levels.
fn ed_reference(ham: &LocalHamiltonian, lambda: f64, nph: usize, vscale: f64) -> EdReference {
    let hs = build(1, SectorClass::SpinResolved);
    EdReference::build_with_phonon(
        ham,
        &hs,
        &bath_scaled(vscale),
        &[],
        BETA,
        Some(&phonon(lambda, nph, NT + 2)),
    )
}

fn k_tau(tau: f64, w: f64) -> f64 {
    -(-w * tau).exp() / (1.0 + (-BETA * w).exp())
}

/// Boson bath of the coupling series `gvec`: the `g(t) g(t')`-scaled
/// free phonon propagator with both boson channels (the solver counts
/// the vertex once) and the doublon vertex blocks.
fn boson_bath(hs: &HilbertSpace, nt: usize, ntau: usize, h: f64, gvec: &[f64]) -> BosonBath {
    let mut d0 = nqist::boson::green_from_eps_phonon(nt, ntau, BETA, h, OMEGA);
    nqist::boson::scale_phonon(&mut d0, gvec);
    let bcontour = Contour::new(nt as i64, ntau as i64, 1, 1, nt as f64 * h, BETA);
    let mut d0_cc = ContourGreen::new(&bcontour, Sign::Bose, Complex::ZERO);
    nqist::boson::set_bwd_from_fwd_boson_mat(&mut d0_cc, &d0);
    for tstp in 0..=(nt as i64) {
        nqist::boson::set_bwd_from_fwd_boson_tstp(tstp, &mut d0_cc, &d0);
    }
    BosonBath {
        d0,
        d0_cc,
        blocks: nqist::boson::docc_blocks(hs),
    }
}

/// Spin up / down Green's-function vertices.
fn spin_verts(hs: &HilbertSpace) -> [GfVertex; 2] {
    [
        GfVertex {
            op_ann_fl: hs.flavor(0, 0, false),
            op_cre_fl: hs.flavor(0, 0, true),
        },
        GfVertex {
            op_ann_fl: hs.flavor(0, 1, false),
            op_cre_fl: hs.flavor(0, 1, true),
        },
    ]
}

/// Hybridization channels of the spin-symmetric single-level bath
/// `delta` (forward + backward for both spins).
fn spin_ints<'a>(
    hs: &HilbertSpace,
    delta: &'a ContourGreen,
    delta_cc: &'a ContourGreen,
) -> [Hybridization<'a>; 4] {
    [
        Hybridization {
            delta,
            op_cre_fl: hs.flavor(0, 0, true),
            op_ann_fl: hs.flavor(0, 0, false),
            fermion: true,
            backward: false,
        },
        Hybridization {
            delta: delta_cc,
            op_cre_fl: hs.flavor(0, 0, false),
            op_ann_fl: hs.flavor(0, 0, true),
            fermion: true,
            backward: true,
        },
        Hybridization {
            delta,
            op_cre_fl: hs.flavor(0, 1, true),
            op_ann_fl: hs.flavor(0, 1, false),
            fermion: true,
            backward: false,
        },
        Hybridization {
            delta: delta_cc,
            op_cre_fl: hs.flavor(0, 1, false),
            op_ann_fl: hs.flavor(0, 1, true),
            fermion: true,
            backward: true,
        },
    ]
}

/// Solve the Holstein dimer impurity at expansion order `order` on
/// the Matsubara axis and return the spin-up / spin-down Green's
/// functions plus the number of pseudo-particle iterations.
fn solve_holstein(
    ham: &LocalHamiltonian,
    solver_kind: Solver,
    order: usize,
    vscale: f64,
    g: f64,
    ntau: usize,
    pp_iters: usize,
) -> (Vec<Vec<f64>>, usize) {
    let hs = build(1, SectorClass::SpinResolved);
    let mut solver = ImpuritySolver::with_solver(
        NT,
        ntau,
        BETA,
        0.02,
        5,
        solver_kind,
        order,
        hs.clone(),
        ham.clone(),
    );
    solver.solve_atomic();
    solver.set_boson_bath(Some(boson_bath(
        &hs,
        NT,
        ntau,
        0.02,
        &vec![g; NT + 2],
    )));

    let contour = Contour::new(NT as i64, ntau as i64, 1, 1, 0.1, BETA);
    let mut delta = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    for m in 0..=ntau {
        let t = BETA * m as f64 / ntau as f64;
        delta.mat.block_mut(m)[0] = Complex::new((V * vscale).powi(2) * k_tau(t, EPS_B), 0.0);
    }
    let mut delta_cc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    set_bwd_from_fwd_mat(&mut delta_cc, &delta);
    let pp_ints = spin_ints(&hs, &delta, &delta_cc);
    let verts = spin_verts(&hs);

    let mut last: Vec<Vec<f64>> = Vec::new();
    let mut iters = 0usize;
    for it in 0..pp_iters {
        solver.pp_step(-1, &pp_ints);
        let g = solver.get_spgf_mat(&verts, &pp_ints);
        let cur: Vec<Vec<f64>> = g.iter().map(|row| row.iter().map(|z| z.re).collect()).collect();
        iters = it + 1;
        if !last.is_empty() {
            let diff = last
                .iter()
                .flatten()
                .zip(cur.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            if diff < 1.0e-10 {
                last = cur;
                break;
            }
        }
        last = cur;
    }
    (last, iters)
}

/// Matsubara solution with the library's single-line convention:
/// `G_EP` is the Hamiltonian coupling.
fn solve_matsubara(
    ham: &LocalHamiltonian,
    solver_kind: Solver,
    order: usize,
    vscale: f64,
) -> (Vec<Vec<f64>>, usize) {
    solve_holstein(ham, solver_kind, order, vscale, G_EP, NTAU, 200)
}

fn main() {
    println!("run via cargo test --test holstein_dimer");
}

// ------------------------------------------------------------------
// ED module self-checks with the phonon mode
// ------------------------------------------------------------------

#[test]
fn holstein_dimer_ed_selfchecks() {
    let ham = mk_ham(U_HUB, NT + 2);
    let ed = ed_reference(&ham, G_EP, NPH, 1.0);
    assert_eq!(ed.dim(), 16 * NPH);
    assert_eq!(ed.nmodes(), 4);
    // canonical identity G^>(0) - G^<(0) = -i on every fermionic mode
    for m in 0..4 {
        let d = ed.greens_gtr(m, 0.0) - ed.greens_les(m, 0.0) + Complex::new(0.0, 1.0);
        assert!((d * d.conj()).re < 1e-18, "mode {m}: {d:?}");
    }
    // lambda = 0: the phonon degrees of freedom decouple and the
    // exact Green's functions equal the fermion-only reference
    let hs = build(1, SectorClass::SpinResolved);
    let ed0 = EdReference::build(&ham, &hs, &bath(), &[], BETA);
    let ed_ph0 = ed_reference(&ham, 0.0, 4, 1.0);
    for m in 0..2 {
        for i in [1usize, 8, 16, 32, 48, 63] {
            let tau = BETA * i as f64 / 64.0;
            let d = (ed_ph0.greens_tau(m, tau) - ed0.greens_tau(m, tau)).abs();
            assert!(d < 1e-12, "lambda = 0, mode {m} tau {tau}: {d:.3e}");
        }
    }
    // spin symmetry of the exact reference
    for i in [1usize, 8, 16, 32, 48, 63] {
        let tau = BETA * i as f64 / 64.0;
        let d = (ed.greens_tau(0, tau) - ed.greens_tau(1, tau)).abs();
        assert!(d < 1e-12, "spin symmetry at tau {tau}: {d:.3e}");
    }
    // phonon-number closed form in the atomic limit (no bath, the
    // doublon is an exact eigenstate projector): <n_ph> = n_B(omega0)
    // + P_d (g / omega0)^2 with the Boltzmann doublon weight P_d
    // (particle-hole form: E_0 = 0, E_s = -U/2, E_d = -g^2 / omega0)
    let ed_at = EdReference::build_with_phonon(
        &ham,
        &hs,
        &[],
        &[],
        BETA,
        Some(&phonon(G_EP, NPH, NT + 2)),
    );
    let e_d = -G_EP * G_EP / OMEGA;
    let z = 1.0 + 2.0 * (BETA * U_HUB * 0.5).exp() + (-BETA * e_d).exp();
    let p_d = (-BETA * e_d).exp() / z;
    let n_b = 1.0 / ((BETA * OMEGA).exp() - 1.0);
    let want = n_b + p_d * (G_EP / OMEGA).powi(2);
    let got = ed_at.phonon_number();
    let p_d_ed = ed_at
        .expectation(&[(0, true), (0, false), (1, true), (1, false)])
        .re;
    println!(
        "holstein atomic <n_ph>: {got:.12} vs closed form {want:.12} (P_d {p_d:.6}, ED {p_d_ed:.6})"
    );
    assert!((p_d_ed - p_d).abs() < 1e-8, "doublon weight {p_d_ed} vs {p_d}");
    assert!((got - want).abs() < 1e-7, "{got} vs {want}");
    // truncation convergence of the exact Green's function
    let g = |nph: usize| -> Vec<f64> {
        let e = ed_reference(&ham, G_EP, nph, 1.0);
        (1..=32)
            .map(|i| e.greens_tau(0, BETA * i as f64 / 32.0))
            .collect()
    };
    let (g6, g9, g12) = (g(6), g(9), g(12));
    let dist = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    };
    let (d69, d912) = (dist(&g6, &g9), dist(&g9, &g12));
    println!(
        "holstein dimer phonon truncation: |G6 - G9| {d69:.3e}, |G9 - G12| {d912:.3e}"
    );
    assert!(d912 < 1e-2 * d69 && d912 < 1e-7, "{d69:.3e} -> {d912:.3e}");
}

// ------------------------------------------------------------------
// boson-line convention of the library
// ------------------------------------------------------------------

/// Library convention gate: the bath coupling `g` is the Hamiltonian
/// `lambda` (the phonon line is counted once). Near the atomic limit
/// the NCA error (a few 1e-4) is far below the model difference of
/// the PPSC doubled convention (`lambda = sqrt(2) g`), so the two
/// references are clearly separated.
#[test]
fn holstein_library_counts_single_boson_line() {
    let ham = mk_ham(U_HUB, NT + 2);
    let vscale = 0.1;
    let (g, _) = solve_matsubara(&ham, Solver::Ppsc, 1, vscale);
    let error = |lambda: f64| -> f64 {
        let ed = ed_reference(&ham, lambda, NPH, vscale);
        let mut err = 0.0f64;
        for m in 1..NTAU {
            err = err.max((g[0][m] - ed.greens_tau(0, BETA * m as f64 / NTAU as f64)).abs());
        }
        err
    };
    let physical = error(G_EP);
    let ppsc = error(G_EP * 2.0f64.sqrt());
    println!(
        "library convention: |G - ED(lambda=g)| {physical:.3e}, |G - ED(lambda=sqrt(2) g)| {ppsc:.3e}"
    );
    assert!(
        physical * 5.0 < ppsc,
        "the library coupling must be the Hamiltonian lambda: {physical:.3e} vs {ppsc:.3e}"
    );
}

// ------------------------------------------------------------------
// equilibrium dimer: NCA / OCA against the phonon ED (single line)
// ------------------------------------------------------------------

fn dimer_error(solver_kind: Solver, order: usize, vscale: f64) -> f64 {
    let ham = mk_ham(U_HUB, NT + 2);
    let ed = ed_reference(&ham, G_EP, NPH, vscale);
    let (g, _) = solve_matsubara(&ham, solver_kind, order, vscale);
    let mut err = 0.0f64;
    for o in 0..2 {
        for m in 1..NTAU {
            err = err.max((g[o][m] - ed.greens_tau(o, BETA * m as f64 / NTAU as f64)).abs());
        }
    }
    err
}

/// Coupling-normalization scan: the solver counts one boson line of
/// the bath coupling `g`; the exact reference must use the SAME
/// `lambda = g` in `H_ph = lambda docc (b + b^dag)`. With omega0 = 0.7
/// the candidate normalizations `g omega0`, `g / omega0`,
/// `g omega0^2`, `g / omega0^2` and the PPSC doubled convention
/// (`sqrt(2) g`) give different references, and only `lambda = g`
/// agrees with the solver to the NCA accuracy. Run near the atomic
/// limit (V = 0.04), where the NCA error (~5e-4) is far below the
/// model change of a wrong normalization (the phonon effect on G is
/// ~1.7e-2).
#[test]
fn holstein_coupling_normalization_scan() {
    check_holstein_coupling_normalization_scan(Solver::Ppsc);
}

/// Solver-selection parity: XCA order 1 on the same gates as `holstein_coupling_normalization_scan`.
#[test]
fn holstein_coupling_normalization_scan_xca() {
    check_holstein_coupling_normalization_scan(Solver::Xca);
}

/// `holstein_coupling_normalization_scan` for a solver selection.
fn check_holstein_coupling_normalization_scan(sel: Solver) {
    let ham = mk_ham(U_HUB, NT + 2);
    let vscale = 0.1;
    let (g, _) = solve_matsubara(&ham, sel, 1, vscale);
    let error = |lambda: f64| -> f64 {
        let ed = ed_reference(&ham, lambda, NPH, vscale);
        let mut err = 0.0f64;
        for m in 1..NTAU {
            err = err.max((g[0][m] - ed.greens_tau(0, BETA * m as f64 / NTAU as f64)).abs());
        }
        err
    };
    let right = error(G_EP);
    let wrong: Vec<(&str, f64)> = [
        ("g omega0", G_EP * OMEGA),
        ("g / omega0", G_EP / OMEGA),
        ("g omega0^2", G_EP * OMEGA * OMEGA),
        ("g / omega0^2", G_EP / (OMEGA * OMEGA)),
        ("sqrt(2) g (PPSC convention)", G_EP * 2.0f64.sqrt()),
        ("0", 0.0),
    ]
    .iter()
    .map(|(name, lam)| (*name, error(*lam)))
    .collect();
    println!(
        "normalization scan: lambda = g err {right:.3e}; other candidates {wrong:?}"
    );
    for (name, e) in &wrong {
        assert!(
            right < 0.2 * e,
            "lambda = g must be the best match: {right:.3e} vs {name} {e:.3e}"
        );
    }
}

/// Hybridization-scaling diagnostic with the phonon: the NCA dimer
/// error follows the leading missing diagrams; a defect in the boson
/// vertex algebra would not follow the scaling.
#[test]
fn holstein_dimer_nca_scaling() {
    let full = dimer_error(Solver::Ppsc, 1, 1.0);
    let half = dimer_error(Solver::Ppsc, 1, 0.5);
    println!(
        "scaling [holstein]: NCA err {full:.6e} at V, {half:.6e} at V/2"
    );
    assert!(
        half < 0.5 * full,
        "errors must shrink with the hybridization: {full:.6e} vs {half:.6e}"
    );
}

/// Equilibrium gate: NCA and OCA against the phonon ED with the
/// physical single boson line. The budgets are absolute and
/// calibrated on the validated run.
#[test]
fn holstein_dimer_nca_oca_vs_ed() {
    let ham = mk_ham(U_HUB, NT + 2);
    let ed = ed_reference(&ham, G_EP, NPH, 1.0);
    let mut errors = Vec::new();
    for order in [1usize, 2] {
        let (g, iters) = solve_matsubara(&ham, Solver::Ppsc, order, 1.0);
        let mut err = [0.0f64; 2];
        for o in 0..2 {
            for m in 1..NTAU {
                err[o] = err[o].max((g[o][m] - ed.greens_tau(o, BETA * m as f64 / NTAU as f64)).abs());
            }
        }
        println!(
            "holstein order {order} ({iters} pp iterations): max |G-Ged| up {:+.6e} dn {:+.6e}",
            err[0], err[1]
        );
        assert!((err[0] - err[1]).abs() < 1e-8, "spin symmetry: {err:?}");
        errors.push(err[0]);
    }
    assert!(errors[0] < 5.0e-2, "NCA: {errors:?}");
    assert!(errors[1] < 1.0e-2, "OCA: {errors:?}");
}

/// Solver-selection parity: the XCA backbone (order 1 = XCA(1),
/// order 2 = XCA(1)+XCA(2)) against the same phonon ED, same budgets
/// as the PPSC NCA/OCA gate above.
#[test]
fn holstein_dimer_xca_nca_oca_vs_ed() {
    let ham = mk_ham(U_HUB, NT + 2);
    let ed = ed_reference(&ham, G_EP, NPH, 1.0);
    let mut errors = Vec::new();
    for order in [1usize, 2] {
        let (g, iters) = solve_matsubara(&ham, Solver::Xca, order, 1.0);
        let mut err = [0.0f64; 2];
        for o in 0..2 {
            for m in 1..NTAU {
                err[o] = err[o].max(
                    (g[o][m] - ed.greens_tau(o, BETA * m as f64 / NTAU as f64)).abs(),
                );
            }
        }
        println!(
            "holstein XCA order {order} ({iters} pp iterations): max |G-Ged| up {:+.6e} dn {:+.6e}",
            err[0], err[1]
        );
        assert!((err[0] - err[1]).abs() < 1e-8, "spin symmetry: {err:?}");
        errors.push(err[0]);
    }
    assert!(errors[0] < 5.0e-2, "XCA(1): {errors:?}");
    assert!(errors[1] < 1.0e-2, "XCA(2): {errors:?}");
}

// ------------------------------------------------------------------
// real-time dimer: stationarity + g quench against EdEvolution
// ------------------------------------------------------------------

/// Real-time solver state of the Holstein dimer, with its fermionic
/// bath (forward / backward hybridization): the OCA vertex diagrams of
/// the real-time G need the hybridization list, which the caller
/// rebuilds with `spin_ints` from this pair.
struct RtRun {
    solver: ImpuritySolver,
    verts: [GfVertex; 2],
    bath: (ContourGreen, ContourGreen),
}

#[allow(clippy::too_many_arguments)]
fn run_realtime(
    solver_kind: Solver,
    order: usize,
    gvec: &[f64],
    vscale: f64,
    nt_rt: usize,
    ntau: usize,
    h_rt: f64,
) -> RtRun {
    use std::time::Instant;

    let kt = 5usize;
    let len = nt_rt + 2;
    let ham = mk_ham(U_HUB, len);

    let contour = Contour::new(
        nt_rt as i64,
        ntau as i64,
        1,
        1,
        h_rt * nt_rt as f64,
        BETA,
    );
    // TTI single-level bath propagator from the KB dyson solvers
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
    let lvl = mk_level(EPS_B);
    let scale_green = |g: &ContourGreen, w: f64| -> ContourGreen {
        let wc = Complex::new(w, 0.0);
        let mut out = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        for n in 0..=(nt_rt as i64) {
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
    let hs = build(1, SectorClass::SpinResolved);
    let verts = spin_verts(&hs);

    let started = Instant::now();
    let mut solver = ImpuritySolver::with_solver(
        nt_rt,
        ntau,
        BETA,
        h_rt,
        kt,
        solver_kind,
        order,
        hs.clone(),
        ham,
    );
    // the real-time XCA route: the dimer bath is static (time
    // translation invariant)
    if solver_kind == Solver::Xca {
        solver.rt_xca_m1 = true;
        solver.rt_xca_m2 = order >= 2;
    }
    solver.solve_atomic();
    solver.set_boson_bath(Some(boson_bath(
        &hs,
        nt_rt,
        ntau,
        h_rt,
        gvec,
    )));
    let zero = || ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let (delta, delta_cc) = if vscale == 0.0 {
        (zero(), zero())
    } else {
        mk_pair(&lvl, V * vscale)
    };
    let pp_ints = spin_ints(&hs, &delta, &delta_cc);
    let mut last: Vec<f64> = Vec::new();
    let mut iters = 0usize;
    for iter in 1..=200 {
        solver.pp_step(-1, &pp_ints);
        let g = solver.get_spgf_mat(&verts, &pp_ints);
        let cur: Vec<f64> = g.iter().flatten().map(|z| z.re).collect();
        iters = iter;
        let converged = !last.is_empty()
            && last.iter().zip(&cur).all(|(a, b)| (a - b).abs() < 1.0e-10);
        if converged {
            break;
        }
        last = cur;
    }
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
        "[holstein m={order} vscale {vscale}] equilibrium {iters} iterations, total {:.2?}",
        started.elapsed()
    );
    RtRun {
        solver,
        verts,
        bath: (delta, delta_cc),
    }
}

/// Largest drift of the equal-time observables away from their
/// `tstp = 0` values over the stored real-time rows.
fn obs_drift(run: &mut RtRun, nt_rt: usize) -> f64 {
    run.solver.update_density_matrix(0);
    let obs0 = nqist::observable::update_exp_vals(&run.solver, 0);
    let mut drift = 0.0f64;
    for tstp in 1..=nt_rt as i64 {
        run.solver.update_density_matrix(tstp);
        let obs = nqist::observable::update_exp_vals(&run.solver, tstp);
        drift = drift
            .max((obs.n_up - obs0.n_up).abs())
            .max((obs.n_dn - obs0.n_dn).abs())
            .max((obs.docc - obs0.docc).abs())
            .max((obs.e_int - obs0.e_int).abs());
    }
    drift
}

fn check_holstein_dimer_realtime(solver_kind: Solver, order: usize) {
    use nqist::ed::{EdEvolution, hamiltonian_at_phonon};

    let h_rt = std::env::var("NQIST_H")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.025);
    let nt_rt = (0.6 / h_rt).round() as usize;
    let ntau = 128usize;
    let len = nt_rt + 2;
    let (g0, g1) = (G_EP, 0.6f64);
    let ham = mk_ham(U_HUB, len);
    let gvec_eq = vec![g0; len];
    let mut gvec_q = vec![g1; len];
    gvec_q[0] = g0;
    let hs = build(1, SectorClass::SpinResolved);

    // -- leg 1: no-quench stationarity (TTI), with and without the
    // fermionic bath (the phonon stays coupled in both legs)
    let mut atomic = run_realtime(solver_kind, order, &gvec_eq, 0.0, nt_rt, ntau, h_rt);
    let drift = obs_drift(&mut atomic, nt_rt);
    println!("realtime TTI drift [holstein m={order}, v=0]: {drift:.3e}");
    assert!(
        drift < 1e-8,
        "atomic leg must be stationary: {drift:.3e}"
    );

    let mut steady = run_realtime(solver_kind, order, &gvec_eq, 1.0, nt_rt, ntau, h_rt);
    let drift = obs_drift(&mut steady, nt_rt);
    println!(
        "realtime TTI drift [holstein m={order}] (n_up, n_dn, docc, e_int): {drift:.3e}"
    );
    assert!(
        drift < 1e-6,
        "TTI stationarity broken: {drift:.3e} (gate 1e-6)"
    );

    // -- leg 2: phonon-coupling quench g0 -> g1 vs the exact unitary
    // evolution of the fermion + phonon space
    let h0 = hamiltonian_at_phonon(&ham, &hs, &bath(), &[], -1, &phonon_series(&gvec_eq));
    let h1 = hamiltonian_at_phonon(&ham, &hs, &bath(), &[], 0, &phonon_series(&gvec_q));
    let ev = EdEvolution::build_quench_with_layout(&h0, &h1, nt_rt, h_rt, BETA, &[0, 1], 4);
    let quench = run_realtime(solver_kind, order, &gvec_q, 1.0, nt_rt, ntau, h_rt);
    let q_ints = spin_ints(&hs, &quench.bath.0, &quench.bath.1);
    let mut w = 0.0f64;
    for mode in [0usize, 1] {
        for (n, j) in [
            (5usize, 0usize),
            (nt_rt / 2, 0),
            (nt_rt / 2, nt_rt / 4),
            (nt_rt, 0),
            (nt_rt, nt_rt - 1),
        ] {
            let gf = quench.solver.get_spgf(&quench.verts, n as i64, &q_ints);
            for (tag, a, b) in [
                (
                    "ret",
                    gf[mode].ret.raw_block_at(n as i64, j as i64)[0],
                    ev.greens_ret(mode, n, j),
                ),
                (
                    "les",
                    gf[mode].less.raw_block_at(j as i64, n as i64)[0],
                    ev.greens_les(mode, j, n),
                ),
            ] {
                let d = (a - b).abs();
                w = w.max(d);
                println!(
                    "g quench [holstein m={order}] mode {mode} {tag} ({n},{j}): {d:.3e}"
                );
            }
        }
    }
    println!(
        "realtime g-quench order {order} [holstein]: worst |G - G_ed| {w:.3e}"
    );
    // Measured: order 1 (NCA) 4.380e-2, order 2 (OCA with its
    // one-line vertex diagrams of G) 8.651e-3 (4.847e-2 while the
    // gate passed an empty hybridization list).
    let budget = if order == 1 { 8.0e-2 } else { 2.0e-2 };
    assert!(
        w < budget,
        "quench error {w:.3e} exceeds the Holstein budget"
    );
}

/// Tier 2 gate: Holstein real-time NCA (stationarity + g quench vs
/// EdEvolution on the fermion + phonon space, single boson line).
#[test]
fn holstein_dimer_realtime_nca_vs_ed() {
    check_holstein_dimer_realtime(Solver::Ppsc, 1);
}

/// Tier 2 gate: Holstein real-time OCA.
#[test]
fn holstein_dimer_realtime_oca_vs_ed() {
    check_holstein_dimer_realtime(Solver::Ppsc, 2);
}

/// Solver-selection parity: XCA order 1 through the real-time XCA
/// route with the single boson line, the same stationarity and quench
/// gates as the native NCA gate.
#[test]
fn holstein_dimer_realtime_xca_nca_vs_ed() {
    check_holstein_dimer_realtime(Solver::Xca, 1);
}

/// Solver-selection parity: XCA order 2 through the real-time XCA
/// route with the single boson line, the same gates as the native OCA
/// gate.
#[test]
fn holstein_dimer_realtime_xca_oca_vs_ed() {
    check_holstein_dimer_realtime(Solver::Xca, 2);
}
