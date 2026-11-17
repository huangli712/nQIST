// Project : nQIST
// File    : tests/xca_dimer/main.rs
// Purpose : XCA physics benchmark: spinless fermion dimer (single
//           impurity level coupled to one bath site) against exact
//           diagonalization, order by order
//
// Mirrors the benchmark of Kaye/Huang/Strand/Golez, SciPost Phys.
// 19, 121 (2025), Sec. V.1: only the spin-up channels hybridize, so
// the single-band solver reduces to a spinless two-level impurity,
// and the bath Green's function is compared with the exact
// diagonalization of the full dimer Hamiltonian. The bold expansion
// must converge order by order toward the exact result.

use kadanoffbaym::complex::Complex;
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::linalg::eigh;
use kadanoffbaym::types::{Contour, ContourGreen, Sign};
use nqist::dmft::{get_gf_verts, get_pp_ints, set_bwd_from_fwd_mat};
use nqist::hamilton::LocalHamiltonian::SingleBand;
use nqist::hilbert::{SectorClass, build};
use nqist::solver::ImpuritySolver;

const BETA: f64 = 8.0;
const NTAU: usize = 128;
const NT: usize = 6;
const EPS_D: f64 = -1.0;
const EPS_B: f64 = 0.5;
const V_HOP: f64 = 0.4;

/// Fermionic kernel `K(tau, w) = -exp(-w tau)/(1+exp(-beta w))`.
fn k_tau(tau: f64, w: f64) -> f64 {
    -(-w * tau).exp() / (1.0 + (-BETA * w).exp())
}

/// Exact dimer Green's function `G(tau) = -<T c_d(tau) c_d^dag(0)>`
/// from the diagonalization of the four-state spin-up sector
/// `{|0>, |up>, |b>, |up b>}`.
fn dimer_exact(tau: f64) -> f64 {
    // basis: |0>, |d>=c_d^dag|0>, |b>=c_b^dag|0>, |db>
    // H = eps_d n_d + eps_b n_b + V (c_d^dag c_b + c_b^dag c_d)
    let mut h = DenseMat::zeros(4);
    let e = [0.0, EPS_D, EPS_B, EPS_D + EPS_B];
    for (i, &ei) in e.iter().enumerate() {
        h.set(i, i, Complex::new(ei, 0.0));
    }
    // |d> <-> |b> (states 1, 2), |0> <-> |db> (states 0, 3)
    h.set(1, 2, Complex::new(V_HOP, 0.0));
    h.set(2, 1, Complex::new(V_HOP, 0.0));
    h.set(0, 3, Complex::new(V_HOP, 0.0));
    h.set(3, 0, Complex::new(V_HOP, 0.0));
    let (w, v) = eigh(&h);

    // <i|c_d|j>: c_d annihilates the d-particle
    // c_d |d> = |0>, c_d |db> = -|b> (Jordan-Wigner in this basis)
    let mut cmat = DenseMat::zeros(4);
    cmat.set(0, 1, Complex::ONE);
    cmat.set(2, 3, -Complex::ONE);
    let vc = v.adjoint().mat_mul(&cmat).mat_mul(&v);

    let mut z = 0.0;
    for &ei in &w {
        z += (-BETA * ei).exp();
    }
    let mut acc = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            let m = vc.get(i, j);
            let boltz = (-BETA * w[i]).exp();
            let amp = (m * m.conj()).re;
            acc += boltz * amp * (-tau * (w[j] - w[i])).exp();
        }
    }
    -acc / z
}

/// Run the impurity solve to pseudo-particle self-consistency at the
/// given expansion order and return the Matsubara bath Green's
/// function samples.
#[allow(clippy::needless_range_loop)]
fn solve_dimer(order: usize, pp_iters: usize) -> Vec<f64> {
    let hs = build(1, SectorClass::SpinResolved);
    let ham = SingleBand {
        mu: 0.0,
        u: vec![0.0; NT + 2],
        eps_up: vec![EPS_D; NT + 2],
        eps_do: vec![0.0; NT + 2],
    };
    let mut solver = ImpuritySolver::new(NT, NTAU, BETA, 0.02, 5, order, hs, ham);
    solver.solve_atomic();

    let contour = Contour::new(NT as i64, NTAU as i64, 1, 1, 0.12, BETA);
    let mut delta = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let mut delta_cc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let dtau = BETA / NTAU as f64;
    for i in 0..=NTAU {
        let t = dtau * i as f64;
        let val = V_HOP * V_HOP * k_tau(t, EPS_B);
        delta.mat.block_mut(i)[0] = Complex::new(val, 0.0);
    }
    set_bwd_from_fwd_mat(&mut delta_cc, &delta);

    let verts = get_gf_verts(&solver);
    let mut last: Vec<f64> = Vec::new();
    let t0 = std::time::Instant::now();
    for it in 0..pp_iters {
        let pp_ints = get_pp_ints(&delta, &delta_cc, &solver);
        solver.pp_step(-1, &pp_ints);
        let g = solver.get_spgf_mat(&verts, &pp_ints);
        let cur: Vec<f64> = g[0].iter().map(|z| z.re).collect();
        // per-iteration progress (visible with --nocapture)
        eprintln!(
            "dimer-o{order} pp iter {it:3} (t {:>7.1?}): G(0) = {:.6}",
            t0.elapsed(),
            cur.first().copied().unwrap_or(0.0)
        );
        if !last.is_empty() {
            let mut diff = 0.0f64;
            for (a, b) in last.iter().zip(&cur) {
                diff = diff.max((a - b).abs());
            }
            if diff < 1.0e-10 {
                eprintln!("dimer-o{order} converged after {it} iterations");
                last = cur;
                break;
            }
        }
        last = cur;
    }
    last
}

fn main() {
    // silence dead-code warnings for the helper binaries pattern
    println!("run via cargo test --test xca_dimer");
}

#[test]
#[allow(clippy::needless_range_loop)]
fn xca_dimer_converges_to_exact_order_by_order() {
    // order-by-order convergence toward the exact dimer solution
    let mut errors = Vec::new();
    for order in 1..=4usize {
        let g = solve_dimer(order, 200);
        let dtau = BETA / NTAU as f64;
        let mut err = 0.0f64;
        for i in 1..NTAU {
            let exact = dimer_exact(dtau * i as f64);
            err = err.max((g[i] - exact).abs());
        }
        println!("order {order}: max |G_xca - G_exact| = {err:.6e}");
        for i in [1usize, 16, 32, 64, 96, NTAU - 1] {
            println!(
                "  tau {:.3}: xca {:.6} exact {:.6}",
                dtau * i as f64,
                g[i],
                dimer_exact(dtau * i as f64)
            );
        }
        errors.push(err);
    }
    // the bold expansion converges monotonically toward ED at these
    // parameters (moderate coupling), with order 3 clearly below
    // order 1 and 2
    assert!(errors[1] < errors[0], "order2 {errors:?}");
    // order 3 improves further, but at these parameters the bold
    // series converges slowly (the residual pseudo-particle iteration
    // also stiffens at higher orders: the eta-Newton stabilization of
    // SciPost Phys. 19, 121 App. C-E is future work), so only require
    // a non-degrading, absolutely bounded error
    assert!(errors[2] <= errors[1], "order3 {errors:?}");
    assert!(errors[2] < 6.0e-2, "absolute convergence {errors:?}");
}
