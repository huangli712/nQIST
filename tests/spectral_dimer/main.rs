// Project : nQIST
// File    : tests/spectral_dimer/main.rs
// Purpose : equilibrium spectral function A(omega) extracted from the
//           solver's real-time retarded row, against the ED Lehmann
//           reference (single-band dimer, U != 0).

use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{Contour, ContourGreen, Sign};
use nqist::dmft::set_bwd_from_fwd_mat;
use nqist::ed::{hamiltonian_at, BathCoupling, EdEvolution, EdReference};
use nqist::hamilton::LocalHamiltonian;
use nqist::hamilton::LocalHamiltonian::SingleBand;
use nqist::hilbert::{build, HilbertSpace, SectorClass};
use nqist::nca::{GfVertex, Hybridization};
use nqist::solver::{ImpuritySolver, Solver};
use nqist::spectral::{
    integrate_spectrum, omega_grid, spectral_from_gf, spectral_from_gf_windowed,
    spectral_from_greater, spectral_from_less, spectral_from_ret, spectral_from_ret_windowed,
    wigner_greater_from_gf, wigner_less_from_gf, wigner_spectral_at, wigner_spectral_from_gf,
    wigner_spectral_from_greater, wigner_spectral_from_less, wigner_spectral_from_ret,
    wigner_spectral_series, Window,
};

const BUDGET_LESS: f64 = 1.0e-1;
const BUDGET_GREATER: f64 = 1.0e-1;
const BUDGET_HANN: f64 = 5.0e-2;
const BUDGET_WIGNER: f64 = 2.5e-2;
const BUDGET_WIGNER_LESS: f64 = 2.0e-2;
const BUDGET_WIGNER_GREATER: f64 = 2.0e-2;
const BETA: f64 = 5.0;
const NTAU: usize = 96;
const NT_RT: usize = 400;
const H_RT: f64 = 0.02;
const KT: usize = 5;
const U_HUB: f64 = 2.0;
const EPS_UP: f64 = 0.15;
const EPS_DO: f64 = -0.10;
const EPS_B: [f64; 2] = [0.5, -0.3];
const V: [f64; 2] = [0.40, 0.35];

/// Constant single-band Hamiltonian series of length `len`.
fn mk_ham(u: f64, len: usize) -> LocalHamiltonian {
    SingleBand {
        mu: 0.0,
        u: vec![u; len],
        eps_up: vec![EPS_UP; len],
        eps_do: vec![EPS_DO; len],
    }
}

/// Two bath levels, one per spin (mode 0 = up, mode 1 = down).
fn bath(vscale: f64) -> [BathCoupling; 2] {
    [
        BathCoupling {
            imp_mode: 0,
            eps: EPS_B[0],
            v: V[0] * vscale,
        },
        BathCoupling {
            imp_mode: 1,
            eps: EPS_B[1],
            v: V[1] * vscale,
        },
    ]
}

/// Spin-resolved Green's-function vertices (mode 0 = up, 1 = down).
fn verts(hs: &HilbertSpace) -> [GfVertex; 2] {
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

#[test]
fn spectral_dimer_nca_vs_ed() {
    check_spectral_dimer_vs_ed(Solver::Ppsc);
}

/// Solver-selection parity: XCA order 1 on the same gates as `spectral_dimer_nca_vs_ed`.
#[test]
fn spectral_dimer_xca_nca_vs_ed() {
    check_spectral_dimer_vs_ed(Solver::Xca);
}

/// `spectral_dimer_nca_vs_ed` for a solver selection.
fn check_spectral_dimer_vs_ed(sel: Solver) {
    use std::time::Instant;

    let hs = build(1, SectorClass::SpinResolved);
    let ham = mk_ham(U_HUB, NT_RT + 2);
    let contour = Contour::new(NT_RT as i64, NTAU as i64, 1, 1, H_RT * NT_RT as f64, BETA);

    // static analytic single-level bath on the full contour (same
    // construction as the single-band / two-band dimer benches)
    let mk_level = |eps: f64| -> ContourGreen {
        let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        let sig = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        let mut hf = kadanoffbaym::types::TimeFunction::zeros(NT_RT, 1, 1);
        for tstp in -1..=(NT_RT as i64) {
            hf.block_mut(tstp)[0] = Complex::new(eps, 0.0);
        }
        kadanoffbaym::dyson::dyson_mat(
            &mut g,
            &sig,
            0.0,
            &hf,
            BETA,
            KT as i64,
            kadanoffbaym::base::MatsubaraMethod::Fixpoint,
            true,
            None,
            1e-12,
        );
        kadanoffbaym::dyson::dyson_start(&mut g, 0.0, &hf, &sig, BETA, H_RT, KT as i64);
        for tstp in (KT as i64 + 1)..=(NT_RT as i64) {
            kadanoffbaym::dyson::dyson_timestep(
                tstp, &mut g, 0.0, &hf, &sig, BETA, H_RT, KT as i64,
            );
        }
        g
    };
    let lvl: Vec<ContourGreen> = EPS_B.iter().map(|&e| mk_level(e)).collect();
    let scale_green = |g: &ContourGreen, w: f64| -> ContourGreen {
        let wc = Complex::new(w, 0.0);
        let mut out = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        for n in 0..=(NT_RT as i64) {
            for jj in 0..=n {
                out.ret.block_mut(out.ret.idx(n, jj))[0] = g.ret.block(g.ret.idx(n, jj))[0] * wc;
                out.less.block_mut(out.less.idx(jj, n))[0] =
                    g.less.block(g.less.idx(jj, n))[0] * wc;
            }
            for m in 0..=(NTAU as i64) {
                out.lmix.raw_block_at_mut(n, m)[0] = g.lmix.raw_block_at(n, m)[0] * wc;
            }
        }
        for m in 0..=(NTAU as i64) {
            out.mat.block_mut(m as usize)[0] = g.mat.block(m as usize)[0] * wc;
        }
        out
    };
    let mk_pair = |lvl: &ContourGreen, v: f64| -> (ContourGreen, ContourGreen) {
        let fwd = scale_green(lvl, v * v);
        let mut cc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        set_bwd_from_fwd_mat(&mut cc, &fwd);
        for t in 0..=(NT_RT as i64) {
            nqist::dmft::set_bwd_from_fwd_tstp(t, &mut cc, &fwd);
        }
        (fwd, cc)
    };
    let vts = verts(&hs);
    let mut solver = ImpuritySolver::with_solver(
        NT_RT,
        NTAU,
        BETA,
        H_RT,
        KT,
        sel,
        1,
        hs.clone(),
        ham.clone(),
    );
    solver.solve_atomic();
    let pairs: Vec<(ContourGreen, ContourGreen)> =
        (0..2).map(|spin| mk_pair(&lvl[spin], V[spin])).collect();
    let mut ints = Vec::new();
    for spin in 0..2 {
        ints.push(Hybridization {
            delta: &pairs[spin].0,
            op_cre_fl: hs.flavor(0, spin, true),
            op_ann_fl: hs.flavor(0, spin, false),
            fermion: true,
            backward: false,
        });
        ints.push(Hybridization {
            delta: &pairs[spin].1,
            op_cre_fl: hs.flavor(0, spin, false),
            op_ann_fl: hs.flavor(0, spin, true),
            fermion: true,
            backward: true,
        });
    }

    let started = Instant::now();
    let mut last: Vec<f64> = Vec::new();
    for iter in 1..=200 {
        solver.pp_step(-1, &ints);
        let g = solver.get_spgf_mat(&vts, &ints);
        let cur: Vec<f64> = g.iter().flatten().map(|z| z.re).collect();
        let converged =
            !last.is_empty() && last.iter().zip(&cur).all(|(a, b)| (a - b).abs() < 1.0e-10);
        if iter == 1 || iter % 25 == 0 || converged {
            eprintln!(
                "[spectral] equilibrium iteration {iter}/200 converged={converged} elapsed={:.2?}",
                started.elapsed()
            );
        }
        if converged {
            break;
        }
        last = cur;
    }

    solver.set_t0();
    for tstp in 0..=(KT as i64) {
        solver.pp_step(tstp, &ints);
    }
    for tstp in (KT + 1) as i64..=(NT_RT as i64) {
        solver.extrapolate_timestep(tstp - 1);
        solver.pp_step(tstp, &ints);
    }
    eprintln!(
        "[spectral] real-time propagation done in {:.2?}",
        started.elapsed()
    );

    let gf = solver.get_spgf(&vts, NT_RT as i64, &ints);
    let omega = omega_grid(4.0, 321);
    let eta = 0.4;
    let ed = EdReference::build(&ham, &hs, &bath(1.0), &[], BETA);
    for mode in 0..2 {
        let a = spectral_from_gf(&gf[mode], NT_RT as i64, H_RT, &omega, eta);
        let ae = ed.spectral_function(mode, &omega, eta);
        let mut worst = 0.0f64;
        let mut at = 0.0f64;
        for (&om, (&x, &y)) in omega.iter().zip(a.iter().zip(&ae)) {
            if om.abs() <= 3.0 && (x - y).abs() > worst {
                worst = (x - y).abs();
                at = om;
            }
        }
        let sa = integrate_spectrum(&a, &omega);
        let se = integrate_spectrum(&ae, &omega);
        // Transform sanity on the exact ED ret: the same sampling +
        // damping applied to G^R_ED must reproduce the Lehmann A(omega)
        // up to the finite-window truncation (~exp(-eta T)).
        let samples: Vec<Complex> = (0..=NT_RT)
            .map(|k| ed.greens_ret(mode, k as f64 * H_RT))
            .collect();
        let a_ft = spectral_from_ret(&samples, H_RT, &omega, eta);
        let mut worst_ft = 0.0f64;
        for (&om, (&x, &y)) in omega.iter().zip(a_ft.iter().zip(&ae)) {
            if om.abs() <= 3.0 {
                worst_ft = worst_ft.max((x - y).abs());
            }
        }
        // Occupied / unoccupied: in equilibrium A^< = f A and
        // A^> = (1-f) A; the solver's lesser/greater rows must satisfy
        // the fluctuation-dissipation decomposition, and A^< + A^> = A.
        let a_less = spectral_from_less(&gf[mode], NT_RT as i64, H_RT, &omega, eta);
        let a_gtr = spectral_from_greater(&gf[mode], NT_RT as i64, H_RT, &omega, eta);
        let (mut wl, mut wg, mut ws) = (0.0f64, 0.0f64, 0.0f64);
        for idx in 0..omega.len() {
            if omega[idx].abs() > 3.0 {
                continue;
            }
            let fw = 1.0 / (1.0 + (BETA * omega[idx]).exp());
            wl = wl.max((a_less[idx] - fw * ae[idx]).abs());
            wg = wg.max((a_gtr[idx] - (1.0 - fw) * ae[idx]).abs());
            ws = ws.max((a_less[idx] + a_gtr[idx] - a[idx]).abs());
        }
        // Windowed transform: the solver's Hann spectrum is compared
        // with the exact ED ret transformed with the SAME window.
        let a_hann = spectral_from_gf_windowed(
            &gf[mode],
            NT_RT as i64,
            H_RT,
            &omega,
            eta,
            Window::Hann,
        );
        let a_hann_e = spectral_from_ret_windowed(&samples, H_RT, &omega, eta, Window::Hann);
        let mut wh = 0.0f64;
        for (&om, (&x, &y)) in omega.iter().zip(a_hann.iter().zip(&a_hann_e)) {
            if om.abs() <= 3.0 {
                wh = wh.max((x - y).abs());
            }
        }
        println!("[spectral] mode {mode}: ED ret FT vs Lehmann worst {worst_ft:.3e}");
        assert!(
            worst_ft < 3.0e-2,
            "mode {mode}: ED transform mismatch {worst_ft:.3e}"
        );
        println!(
            "[spectral] mode {mode}: worst |A_solver - A_ed| {worst:.3e} at omega {at:+.2}, \
             sum {sa:.4} (ED {se:.4}); A^< vs fA {wl:.3e}; A^> vs (1-f)A {wg:.3e}; \
             A^<+A^>-A {ws:.3e}; Hann vs ED-Hann {wh:.3e}"
        );
        assert!(worst < 8.0e-2, "mode {mode}: spectral mismatch {worst:.3e}");
        assert!(
            (sa - se).abs() < 1.0e-2,
            "mode {mode}: integrated weight {sa:.4} vs {se:.4}"
        );
        assert!(wl < BUDGET_LESS, "mode {mode}: A^< mismatch {wl:.3e}");
        assert!(wg < BUDGET_GREATER, "mode {mode}: A^> mismatch {wg:.3e}");
        assert!(ws < 1.0e-12, "mode {mode}: A^<+A^>-A {ws:.3e}");
        assert!(wh < BUDGET_HANN, "mode {mode}: Hann mismatch {wh:.3e}");
    }
}

/// Solve the equilibrium and propagate a single-band impurity to
/// `NT_RT` for `ham` (order 1) with the solver selection `sel`.
/// Duplicates the fixture above so the verified equilibrium test is
/// left untouched.
fn run_solver(sel: Solver, ham: &LocalHamiltonian) -> ImpuritySolver {
    let hs = build(1, SectorClass::SpinResolved);
    let contour = Contour::new(NT_RT as i64, NTAU as i64, 1, 1, H_RT * NT_RT as f64, BETA);
    let mk_level = |eps: f64| -> ContourGreen {
        let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        let sig = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        let mut hf = kadanoffbaym::types::TimeFunction::zeros(NT_RT, 1, 1);
        for tstp in -1..=(NT_RT as i64) {
            hf.block_mut(tstp)[0] = Complex::new(eps, 0.0);
        }
        kadanoffbaym::dyson::dyson_mat(
            &mut g,
            &sig,
            0.0,
            &hf,
            BETA,
            KT as i64,
            kadanoffbaym::base::MatsubaraMethod::Fixpoint,
            true,
            None,
            1e-12,
        );
        kadanoffbaym::dyson::dyson_start(&mut g, 0.0, &hf, &sig, BETA, H_RT, KT as i64);
        for tstp in (KT as i64 + 1)..=(NT_RT as i64) {
            kadanoffbaym::dyson::dyson_timestep(
                tstp, &mut g, 0.0, &hf, &sig, BETA, H_RT, KT as i64,
            );
        }
        g
    };
    let lvl: Vec<ContourGreen> = EPS_B.iter().map(|&e| mk_level(e)).collect();
    let scale_green = |g: &ContourGreen, w: f64| -> ContourGreen {
        let wc = Complex::new(w, 0.0);
        let mut out = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        for n in 0..=(NT_RT as i64) {
            for jj in 0..=n {
                out.ret.block_mut(out.ret.idx(n, jj))[0] = g.ret.block(g.ret.idx(n, jj))[0] * wc;
                out.less.block_mut(out.less.idx(jj, n))[0] =
                    g.less.block(g.less.idx(jj, n))[0] * wc;
            }
            for m in 0..=(NTAU as i64) {
                out.lmix.raw_block_at_mut(n, m)[0] = g.lmix.raw_block_at(n, m)[0] * wc;
            }
        }
        for m in 0..=(NTAU as i64) {
            out.mat.block_mut(m as usize)[0] = g.mat.block(m as usize)[0] * wc;
        }
        out
    };
    let mk_pair = |lvl: &ContourGreen, v: f64| -> (ContourGreen, ContourGreen) {
        let fwd = scale_green(lvl, v * v);
        let mut cc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        set_bwd_from_fwd_mat(&mut cc, &fwd);
        for t in 0..=(NT_RT as i64) {
            nqist::dmft::set_bwd_from_fwd_tstp(t, &mut cc, &fwd);
        }
        (fwd, cc)
    };
    let vts = verts(&hs);
    let mut solver = ImpuritySolver::with_solver(
        NT_RT,
        NTAU,
        BETA,
        H_RT,
        KT,
        sel,
        1,
        hs.clone(),
        ham.clone(),
    );
    solver.solve_atomic();
    let pairs: Vec<(ContourGreen, ContourGreen)> =
        (0..2).map(|spin| mk_pair(&lvl[spin], V[spin])).collect();
    let mut ints = Vec::new();
    for spin in 0..2 {
        ints.push(Hybridization {
            delta: &pairs[spin].0,
            op_cre_fl: hs.flavor(0, spin, true),
            op_ann_fl: hs.flavor(0, spin, false),
            fermion: true,
            backward: false,
        });
        ints.push(Hybridization {
            delta: &pairs[spin].1,
            op_cre_fl: hs.flavor(0, spin, false),
            op_ann_fl: hs.flavor(0, spin, true),
            fermion: true,
            backward: true,
        });
    }
    let mut last: Vec<f64> = Vec::new();
    for _ in 1..=200 {
        solver.pp_step(-1, &ints);
        let g = solver.get_spgf_mat(&vts, &ints);
        let cur: Vec<f64> = g.iter().flatten().map(|z| z.re).collect();
        let converged =
            !last.is_empty() && last.iter().zip(&cur).all(|(a, b)| (a - b).abs() < 1.0e-10);
        if converged {
            break;
        }
        last = cur;
    }
    solver.set_t0();
    for tstp in 0..=(KT as i64) {
        solver.pp_step(tstp, &ints);
    }
    for tstp in (KT + 1) as i64..=(NT_RT as i64) {
        solver.extrapolate_timestep(tstp - 1);
        solver.pp_step(tstp, &ints);
    }
    solver
}

/// Non-equilibrium gate: after a U quench 2.0 -> 2.4 the causal Wigner
/// spectrum at t = 4T/5 is compared with the exact unitary evolution
/// (`EdEvolution`), using the same Hann window on both sides. The
/// retarded, lesser and greater components come from the full two-time
/// Green's function (`get_spgf_full`).
#[test]
fn spectral_quench_nca_vs_ed() {
    check_spectral_quench_vs_ed(Solver::Ppsc);
}

/// Solver-selection parity: XCA order 1 on the same gates as `spectral_quench_nca_vs_ed`.
#[test]
fn spectral_quench_xca_nca_vs_ed() {
    check_spectral_quench_vs_ed(Solver::Xca);
}

/// `spectral_quench_nca_vs_ed` for a solver selection.
fn check_spectral_quench_vs_ed(sel: Solver) {
    let hs = build(1, SectorClass::SpinResolved);
    let u0 = U_HUB;
    let u1 = U_HUB + 0.4;
    let mut u = vec![u0; NT_RT + 2];
    for k in 1..u.len() {
        u[k] = u1;
    }
    let ham_q = SingleBand {
        mu: 0.0,
        u,
        eps_up: vec![EPS_UP; NT_RT + 2],
        eps_do: vec![EPS_DO; NT_RT + 2],
    };
    let solver = run_solver(sel, &ham_q);
    let vts = verts(&hs);

    let h0 = hamiltonian_at(&ham_q, &hs, &bath(1.0), &[], -1);
    let h1 = hamiltonian_at(&ham_q, &hs, &bath(1.0), &[], 0);
    let ev = EdEvolution::build_quench(&h0, &h1, NT_RT, H_RT, BETA, &[0, 1]);

    let omega = omega_grid(4.0, 321);
    let eta = 0.4;
    let t = (4 * NT_RT / 5) as i64;
    let full = solver.get_spgf_full(&vts, &[], NT_RT as i64);
    let jmax = 80i64;
    for mode in 0..2 {
        let a_s = wigner_spectral_from_gf(
            &full[mode],
            t,
            jmax,
            NT_RT as i64,
            H_RT,
            &omega,
            eta,
            Window::Hann,
        );
        let ed_ret: Vec<Complex> = (0..=jmax)
            .map(|j| ev.greens_ret(mode, (t + j) as usize, (t - j) as usize))
            .collect();
        let a_e = wigner_spectral_from_ret(&ed_ret, H_RT, &omega, eta, Window::Hann);
        let mut worst = 0.0f64;
        let mut at = 0.0f64;
        for (&om, (&x, &y)) in omega.iter().zip(a_s.iter().zip(&a_e)) {
            if om.abs() <= 3.0 && (x - y).abs() > worst {
                worst = (x - y).abs();
                at = om;
            }
        }
        let a_less_s = wigner_less_from_gf(
            &full[mode],
            t,
            jmax,
            NT_RT as i64,
            H_RT,
            &omega,
            eta,
            Window::Hann,
        );
        let ed_less: Vec<Complex> = (0..=jmax)
            .map(|j| ev.greens_les(mode, (t + j) as usize, (t - j) as usize))
            .collect();
        let a_less_e = wigner_spectral_from_less(&ed_less, H_RT, &omega, eta, Window::Hann);
        let mut wl = 0.0f64;
        for (&om, (&x, &y)) in omega.iter().zip(a_less_s.iter().zip(&a_less_e)) {
            if om.abs() <= 3.0 {
                wl = wl.max((x - y).abs());
            }
        }
        let a_gtr_s = wigner_greater_from_gf(
            &full[mode],
            t,
            jmax,
            NT_RT as i64,
            H_RT,
            &omega,
            eta,
            Window::Hann,
        );
        let ed_gtr: Vec<Complex> = (0..=jmax)
            .map(|j| ev.greens_gtr(mode, (t + j) as usize, (t - j) as usize))
            .collect();
        let a_gtr_e = wigner_spectral_from_greater(&ed_gtr, H_RT, &omega, eta, Window::Hann);
        let mut wg = 0.0f64;
        for (&om, (&x, &y)) in omega.iter().zip(a_gtr_s.iter().zip(&a_gtr_e)) {
            if om.abs() <= 3.0 {
                wg = wg.max((x - y).abs());
            }
        }
        let series = wigner_spectral_series(
            &full[mode],
            &[t - 1, t],
            jmax,
            NT_RT as i64,
            H_RT,
            &omega,
            eta,
            Window::Hann,
        );
        let a_tm1 = wigner_spectral_from_gf(
            &full[mode],
            t - 1,
            jmax,
            NT_RT as i64,
            H_RT,
            &omega,
            eta,
            Window::Hann,
        );
        let mut wseries = 0.0f64;
        for idx in 0..omega.len() {
            wseries = wseries.max((series[0][idx] - a_tm1[idx]).abs());
            wseries = wseries.max((series[1][idx] - a_s[idx]).abs());
        }
        let a_mid = wigner_spectral_at(
            &full[mode],
            t as f64 - 0.5,
            jmax,
            NT_RT as i64,
            H_RT,
            &omega,
            eta,
            Window::Hann,
        );
        let mut wmid = 0.0f64;
        for idx in 0..omega.len() {
            wmid = wmid.max((a_mid[idx] - 0.5 * (a_tm1[idx] + a_s[idx])).abs());
        }
        println!(
            "[spectral quench] mode {mode}: Wigner ret {worst:.3e} (at {at:+.2}), less {wl:.3e}, \
             greater {wg:.3e}, series {wseries:.3e}, mid {wmid:.3e}"
        );
        assert!(worst < BUDGET_WIGNER, "mode {mode}: Wigner ret {worst:.3e}");
        assert!(wl < BUDGET_WIGNER_LESS, "mode {mode}: Wigner less {wl:.3e}");
        assert!(
            wg < BUDGET_WIGNER_GREATER,
            "mode {mode}: Wigner greater {wg:.3e}"
        );
        assert!(
            wseries < 1.0e-12,
            "mode {mode}: series inconsistency {wseries:.3e}"
        );
        assert!(
            wmid < 1.0e-12,
            "mode {mode}: interpolation inconsistency {wmid:.3e}"
        );
    }
}
