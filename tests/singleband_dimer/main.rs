// Project : nQIST
// File    : tests/singleband_dimer/main.rs
// Purpose : single-band Hubbard benchmark with U != 0 against exact
//           diagonalization. The impurity is coupled to two bath
//           levels (one per spin, different eps/v), so both spin
//           channels are non-trivial. Equilibrium Green's functions
//           are compared with `EdReference`, real-time propagators
//           (stationarity + U quench) with `EdEvolution`. Mirrors
//           `tests/threeband_dimer` (same bath / Dyson machinery).
//
//           This closes the last missing NCA/OCA validation cell: the
//           single-band interacting case had PPSC references only.

use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{Contour, ContourGreen, Sign};
use nqist::dmft::set_bwd_from_fwd_mat;
use nqist::ed::{hamiltonian_at, BathCoupling, EdEvolution, EdReference};
use nqist::hamilton::LocalHamiltonian;
use nqist::hamilton::LocalHamiltonian::SingleBand;
use nqist::hilbert::{build, HilbertSpace, SectorClass};
use nqist::nca::{GfVertex, Hybridization};
use nqist::observable;
use nqist::solver::ImpuritySolver;

const BETA: f64 = 5.0;
const NTAU: usize = 96;
const NT: usize = 6;
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

/// Two bath levels, one per spin (mode 0 = up, mode 1 = down),
/// couplings scaled by `vscale`.
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

fn k_tau(tau: f64, w: f64) -> f64 {
    -(-w * tau).exp() / (1.0 + (-BETA * w).exp())
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

/// Solve the single-band impurity at expansion order `order` on the
/// Matsubara axis; returns `(G_up[m], G_dn[m])`, `m = 0..=ntau`.
fn solve_singleband(
    ham: &LocalHamiltonian,
    order: usize,
    pp_iters: usize,
    vscale: f64,
    ntau: usize,
) -> (Vec<f64>, Vec<f64>) {
    let hs = build(1, SectorClass::SpinResolved);
    let mut solver = ImpuritySolver::new(NT, ntau, BETA, 0.02, 5, order, hs.clone(), ham.clone());
    solver.solve_atomic();

    let contour = Contour::new(NT as i64, ntau as i64, 1, 1, 0.1, BETA);
    let dtau = BETA / ntau as f64;
    let mut fwd = Vec::new();
    let mut cc = Vec::new();
    for spin in 0..2 {
        let mut d = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        for m in 0..=ntau {
            let t = dtau * m as f64;
            d.mat.block_mut(m)[0] =
                Complex::new((V[spin] * vscale).powi(2) * k_tau(t, EPS_B[spin]), 0.0);
        }
        let mut c = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        set_bwd_from_fwd_mat(&mut c, &d);
        fwd.push(d);
        cc.push(c);
    }

    let vts = verts(&hs);
    let mut ints = Vec::new();
    for spin in 0..2 {
        ints.push(Hybridization {
            delta: &fwd[spin],
            op_cre_fl: hs.flavor(0, spin, true),
            op_ann_fl: hs.flavor(0, spin, false),
            fermion: true,
            backward: false,
        });
        ints.push(Hybridization {
            delta: &cc[spin],
            op_cre_fl: hs.flavor(0, spin, false),
            op_ann_fl: hs.flavor(0, spin, true),
            fermion: true,
            backward: true,
        });
    }

    let mut last = (vec![0.0; ntau + 1], vec![0.0; ntau + 1]);
    for it in 0..pp_iters {
        solver.pp_step(-1, &ints);
        let g = solver.get_spgf_mat(&vts, &ints);
        let up: Vec<f64> = g[0].iter().map(|z| z.re).collect();
        let dn: Vec<f64> = g[1].iter().map(|z| z.re).collect();
        if it > 0 {
            let diff = up
                .iter()
                .chain(dn.iter())
                .zip(last.0.iter().chain(last.1.iter()))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            if diff < 1e-10 {
                return (up, dn);
            }
        }
        last = (up, dn);
    }
    last
}

/// Max |G_solver - G_ed| over both spins and the interior Matsubara
/// points (the endpoints carry the equal-time convention).
fn dimer_errors(ham: &LocalHamiltonian, order: usize, vscale: f64) -> [f64; 2] {
    let hs = build(1, SectorClass::SpinResolved);
    let ed = EdReference::build(ham, &hs, &bath(vscale), &[], BETA);
    let (up, dn) = solve_singleband(ham, order, 200, vscale, NTAU);
    let dtau = BETA / NTAU as f64;
    let mut err = [0.0f64; 2];
    for m in 1..NTAU {
        let tau = dtau * m as f64;
        err[0] = err[0].max((up[m] - ed.greens_tau(0, tau)).abs());
        err[1] = err[1].max((dn[m] - ed.greens_tau(1, tau)).abs());
    }
    err
}

/// Atomic-limit cross-check: with zero hybridization the solver G(tau)
/// is the exact atomic propagator up to the Matsubara quadrature
/// error; this pins the quadrature level used by the stationarity
/// gates below.
#[test]
fn singleband_atomic_limit_crosscheck() {
    let ham = mk_ham(U_HUB, NT + 2);
    let hs = build(1, SectorClass::SpinResolved);
    let ed = EdReference::build(&ham, &hs, &[], &[], BETA);
    for ntau in [NTAU, 64] {
        let (up, dn) = solve_singleband(&ham, 1, 200, 0.0, ntau);
        let mut w = 0.0f64;
        for m in 1..ntau {
            let tau = BETA * m as f64 / ntau as f64;
            w = w
                .max((up[m] - ed.greens_tau(0, tau)).abs())
                .max((dn[m] - ed.greens_tau(1, tau)).abs());
        }
        println!("atomic crosscheck ntau {ntau}: max |G-Ged| {w:.3e}");
        assert!(w < 1e-6, "atomic quadrature error too large: {w:.3e}");
    }
}

/// Equilibrium gate: single-band NCA against the ED module's exact
/// reference (both spins hybridized, U != 0).
#[test]
fn singleband_dimer_nca_vs_ed() {
    let ham = mk_ham(U_HUB, NT + 2);
    let err = dimer_errors(&ham, 1, 1.0);
    println!(
        "single-band order 1: max |G-Ged| up {:+.6e} dn {:+.6e}",
        err[0], err[1]
    );
    assert!(err.iter().all(|e| *e < 4.5e-2), "NCA: {err:?}");
}

/// Equilibrium gate: single-band OCA against the same reference.
#[test]
fn singleband_dimer_oca_vs_ed() {
    let ham = mk_ham(U_HUB, NT + 2);
    let err = dimer_errors(&ham, 2, 1.0);
    println!(
        "single-band order 2: max |G-Ged| up {:+.6e} dn {:+.6e}",
        err[0], err[1]
    );
    // The bold series is non-monotonic at these couplings (OCA is
    // far better than NCA here); the budget is absolute.
    assert!(err.iter().all(|e| *e < 1.0e-2), "OCA: {err:?}");
}

/// Real-time single-band bench: converged Matsubara state, then
/// (i) no-quench stationarity (TTI) and (ii) a U quench
/// 2.0 -> 2.4 against the exact unitary evolution.
fn check_singleband_dimer_realtime(
    order: usize,
    nt_rt: usize,
    ntau: usize,
    with_stationarity: bool,
) {
    use std::time::Instant;

    let h_rt = 0.025;
    let kt = 5usize;
    let hs = build(1, SectorClass::SpinResolved);
    let ham_eq = mk_ham(U_HUB, nt_rt + 2);
    let mut u_q = vec![U_HUB; nt_rt + 2];
    for k in 1..u_q.len() {
        u_q[k] = U_HUB + 0.4;
    }
    let ham_q = SingleBand {
        mu: 0.0,
        u: u_q,
        eps_up: vec![EPS_UP; nt_rt + 2],
        eps_do: vec![EPS_DO; nt_rt + 2],
    };

    // static analytic single-level bath on the full contour (same
    // construction as the three-band / two-band dimer benches)
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
            kadanoffbaym::dyson::dyson_timestep(
                tstp, &mut g, 0.0, &hf, &sig, BETA, h_rt, kt as i64,
            );
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
    let vts = verts(&hs);

    let run = |case: &str, ham: &LocalHamiltonian, vscale: f64| -> ImpuritySolver {
        let started = Instant::now();
        eprintln!("[singleband m={order} {case}] starting equilibrium, nt={nt_rt}, ntau={ntau}");
        let mut solver =
            ImpuritySolver::new(nt_rt, ntau, BETA, h_rt, kt, order, hs.clone(), ham.clone());
        solver.solve_atomic();
        let zero = || ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        let pairs: Vec<(ContourGreen, ContourGreen)> = (0..2)
            .map(|spin| {
                if vscale == 0.0 {
                    (zero(), zero())
                } else {
                    mk_pair(&lvl[spin], V[spin] * vscale)
                }
            })
            .collect();
        let mut pp_ints = Vec::new();
        for spin in 0..2 {
            pp_ints.push(Hybridization {
                delta: &pairs[spin].0,
                op_cre_fl: hs.flavor(0, spin, true),
                op_ann_fl: hs.flavor(0, spin, false),
                fermion: true,
                backward: false,
            });
            pp_ints.push(Hybridization {
                delta: &pairs[spin].1,
                op_cre_fl: hs.flavor(0, spin, false),
                op_ann_fl: hs.flavor(0, spin, true),
                fermion: true,
                backward: true,
            });
        }
        let mut last: Vec<f64> = Vec::new();
        for iter in 1..=200 {
            solver.pp_step(-1, &pp_ints);
            let g = solver.get_spgf_mat(&vts, &pp_ints);
            let cur: Vec<f64> = g.iter().flatten().map(|z| z.re).collect();
            let converged =
                !last.is_empty() && last.iter().zip(&cur).all(|(a, b)| (a - b).abs() < 1.0e-10);
            if iter == 1 || iter % 10 == 0 || converged {
                eprintln!(
                    "[singleband m={order} {case}] equilibrium iteration {iter}/200 finished, converged={converged}, elapsed={:.2?}",
                    started.elapsed()
                );
            }
            if converged {
                break;
            }
            last = cur;
        }
        eprintln!(
            "[singleband m={order} {case}] starting real-time propagation after {:.2?}",
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
            "[singleband m={order} {case}] done in {:.2?}",
            started.elapsed()
        );
        solver
    };

    // -- leg 1: no-quench stationarity (TTI)
    if with_stationarity {
        let mut solver = run("atomic", &ham_eq, 0.0);
        solver.update_density_matrix(0);
        let obs0 = observable::update_exp_vals(&solver, 0);
        let mut drift = 0.0f64;
        for tstp in 1..=nt_rt as i64 {
            solver.update_density_matrix(tstp);
            let obs = observable::update_exp_vals(&solver, tstp);
            drift = drift
                .max((obs.n_up - obs0.n_up).abs())
                .max((obs.n_dn - obs0.n_dn).abs())
                .max((obs.docc - obs0.docc).abs())
                .max((obs.e_int - obs0.e_int).abs());
        }
        println!("realtime TTI drift [single-band, v=0]: {drift:.3e}");
        assert!(drift < 1e-8, "atomic leg must be stationary: {drift:.3e}");

        let mut solver = run("steady", &ham_eq, 1.0);
        solver.update_density_matrix(0);
        let obs0 = observable::update_exp_vals(&solver, 0);
        let mut drift = 0.0f64;
        for tstp in 1..=nt_rt as i64 {
            solver.update_density_matrix(tstp);
            let obs = observable::update_exp_vals(&solver, tstp);
            drift = drift
                .max((obs.n_up - obs0.n_up).abs())
                .max((obs.n_dn - obs0.n_dn).abs())
                .max((obs.e_int - obs0.e_int).abs());
        }
        println!("realtime TTI drift [single-band] (n_up, n_dn, e_int): {drift:.3e}");
        // OCA runs at ntau = 64 (the Matsubara quadrature level of the
        // atomic cross-check), NCA at ntau = 96; the atomic leg and
        // the ED quench gate stay tight.
        let steady_gate = if order >= 2 { 5e-6 } else { 1e-6 };
        assert!(
            drift < steady_gate,
            "TTI stationarity broken: {drift:.3e} (gate {steady_gate:.0e})"
        );
    }

    // -- leg 2: U quench vs the exact unitary evolution
    let ed_start = Instant::now();
    eprintln!("[singleband m={order}] building 16-state ED quench reference");
    let h0 = hamiltonian_at(&ham_q, &hs, &bath(1.0), &[], -1);
    let h1 = hamiltonian_at(&ham_q, &hs, &bath(1.0), &[], 0);
    let ev = EdEvolution::build_quench(&h0, &h1, nt_rt, h_rt, BETA, &[0, 1]);
    eprintln!(
        "[singleband m={order}] ED reference ready in {:.2?}",
        ed_start.elapsed()
    );
    let solver = run("U quench", &ham_q, 1.0);
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
    for mode in [0usize, 1] {
        for (n, j) in points.clone() {
            let gf = solver.get_spgf(&vts, n as i64, &[]);
            let g_ret = gf[mode].ret.raw_block_at(n as i64, j as i64)[0];
            let g_les = gf[mode].less.raw_block_at(j as i64, n as i64)[0];
            for (tag, a, b) in [
                ("ret", g_ret, ev.greens_ret(mode, n, j)),
                ("les", g_les, ev.greens_les(mode, j, n)),
            ] {
                if (a - b).abs() > w {
                    println!(
                        "quench [singleband m={order}] worst so far: mode {mode} {tag} ({n},{j}): {:.3e}",
                        (a - b).abs()
                    );
                }
                w = w.max((a - b).abs());
            }
        }
    }
    println!("realtime U-quench order {order} [single-band]: worst |G - G_ed| {w:.3e}");
    assert!(
        w < 8.0e-2,
        "quench error {w:.3e} exceeds the single-band budget"
    );
}

/// Tier 2 gate: single-band real-time NCA (stationarity + U quench).
#[test]
fn singleband_dimer_realtime_nca_vs_ed() {
    check_singleband_dimer_realtime(1, 40, NTAU, true);
}

/// Tier 2 gate: single-band real-time OCA (stationarity + U quench).
#[test]
fn singleband_dimer_realtime_oca_vs_ed() {
    check_singleband_dimer_realtime(2, 24, 64, true);
}
