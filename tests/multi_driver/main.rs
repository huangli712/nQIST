// Project : nQIST
// File    : tests/multi_driver/main.rs
// Purpose : driver-plumbing gate for the multi-band Bethe DMFT
//           routines, which previously had no test coverage:
//           (i) the AFM cross bath map
//               `Delta_c = t^2 Gloc[bath_map[c]]` and
//           (ii) the time-dependent hopping rows
//               `Delta(t_n, t_j) = t(t_n) t(t_j) Gloc(t_n, t_j)`,
//               `Delta(t_n, tau) = t(t_n) t_init Gloc(t_n, tau)`.
//           Small grids, order 1 (NCA); this is a driver check, not an
//           accuracy gate.

use kadanoffbaym::complex::Complex;
use nqist::dmft::{run_equilibrium_multi, run_realtime_multi, run_start_window_multi};
use nqist::hamilton::LocalHamiltonian::SingleBand;
use nqist::hilbert::{build, SectorClass};
use nqist::solver::ImpuritySolver;

const BETA: f64 = 5.0;
const NT: usize = 8;
const NTAU: usize = 24;
const H: f64 = 0.05;
const KT: usize = 5;
const U_HUB: f64 = 2.0;

fn ham() -> nqist::hamilton::LocalHamiltonian {
    SingleBand {
        mu: 0.0,
        u: vec![U_HUB; NT + 2],
        eps_up: vec![0.2; NT + 2],
        eps_do: vec![-0.2; NT + 2],
    }
}

fn new_solver() -> ImpuritySolver {
    let hs = build(1, SectorClass::SpinResolved);
    let mut s = ImpuritySolver::new(NT, NTAU, BETA, H, KT, 1, hs, ham());
    s.solve_atomic();
    s
}

#[test]
fn bath_map_cross_bethe_is_honored() {
    // identity map: Delta_c = t^2 Gloc_c with t = 1
    let tvec = vec![1.0; NT + 2];
    let mut s = new_solver();
    let (_log, gloc, delta, _dcc) = run_equilibrium_multi(&mut s, &tvec, &[0, 1], 0.5, 1e-10, 100);
    for c in 0..2 {
        for m in 0..=NTAU {
            let d = (delta[c].mat.block(m)[0] - gloc[c].mat.block(m)[0]).abs();
            assert!(d < 1e-14, "identity map channel {c} slice {m}: {d:.3e}");
        }
    }

    // cross map [1, 0]: Delta_0 = Gloc_1, Delta_1 = Gloc_0. The
    // driver mixes a channel's Gloc and immediately writes its Delta,
    // so Delta_0 lags Gloc_1 by one mixing step; at convergence
    // (errmax = 1e-10) the residual is a few 1e-12.
    let mut s = new_solver();
    let (_log, gloc_x, delta_x, _dcc) =
        run_equilibrium_multi(&mut s, &tvec, &[1, 0], 0.5, 1e-10, 100);
    for m in 0..=NTAU {
        let d0 = (delta_x[0].mat.block(m)[0] - gloc_x[1].mat.block(m)[0]).abs();
        let d1 = (delta_x[1].mat.block(m)[0] - gloc_x[0].mat.block(m)[0]).abs();
        assert!(
            d0 < 1e-9 && d1 < 1e-9,
            "cross map slice {m}: {d0:.3e} {d1:.3e}"
        );
    }

    // the cross map must actually alter the solution (not silently
    // treated as the identity)
    let mut w = 0.0f64;
    for m in 0..=NTAU {
        w = w.max((gloc_x[0].mat.block(m)[0] - gloc[0].mat.block(m)[0]).abs());
    }
    assert!(w > 1e-6, "cross bath map must alter the solution: {w:.3e}");
}

#[test]
fn time_dependent_hopping_rows_follow_the_series() {
    // equilibrium hopping 1.0, then a quench to 1.2 at t = 0
    let mut tvec = vec![1.2; NT + 2];
    tvec[0] = 1.0;
    let t_at = |n: i64| tvec[(n + 1) as usize];

    let mut s = new_solver();
    let (_log, mut gloc, mut delta, mut delta_cc) =
        run_equilibrium_multi(&mut s, &tvec, &[1, 0], 0.4, 1e-10, 200);
    // the Matsubara bath used t_init^2 = 1; Delta_0 lags Gloc_1 by
    // one mixing step (see the cross-map comment above)
    for m in 0..=NTAU {
        let d = (delta[0].mat.block(m)[0] - gloc[1].mat.block(m)[0]).abs();
        assert!(d < 1e-9, "Matsubara row slice {m}: {d:.3e}");
    }

    let start = run_start_window_multi(
        &mut s,
        &tvec,
        &[1, 0],
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        1e-10,
        50,
    );
    let kt = KT as i64;
    let mut checked = 0usize;
    for n in 1..=kt {
        for j in 0..=n {
            let f = t_at(n) * t_at(j);
            let d =
                (delta[0].ret.raw_block_at(n, j)[0] - f * gloc[1].ret.raw_block_at(n, j)[0]).abs();
            assert!(d < 1e-13, "START ret row ({n},{j}): {d:.3e}");
            let dl = (delta[0].less.raw_block_at(j, n)[0] - f * gloc[1].less.raw_block_at(j, n)[0])
                .abs();
            assert!(dl < 1e-13, "START less row ({j},{n}): {dl:.3e}");
        }
        let fm = t_at(n) * tvec[0];
        for m in 0..=NTAU as i64 {
            let d = (delta[0].lmix.raw_block_at(n, m)[0] - fm * gloc[1].lmix.raw_block_at(n, m)[0])
                .abs();
            assert!(d < 1e-13, "START lmix row ({n},{m}): {d:.3e}");
        }
        checked += 1;
    }
    assert!(checked > 0);
    assert!(start.start_lines.len() > 0);
    // the factor at (1,1) is genuinely 1.44, i.e. t(t) is not inert
    assert!((t_at(1) * t_at(1) - 1.44).abs() < 1e-14);

    // one real-time step beyond the START window
    let (tlines, obs_lines) = run_realtime_multi(
        &mut s,
        &tvec,
        &[1, 0],
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        1,
        kt + 1,
        kt + 1,
        |_, _| {},
    );
    assert_eq!(tlines.len(), 1);
    assert_eq!(obs_lines.len(), 1);
    let n = kt + 1;
    for j in 0..=n {
        let f = t_at(n) * t_at(j);
        let d = (delta[1].ret.raw_block_at(n, j)[0] - f * gloc[0].ret.raw_block_at(n, j)[0]).abs();
        assert!(d < 1e-13, "realtime ret row ({n},{j}): {d:.3e}");
    }
    let fm = t_at(n) * tvec[0];
    for m in 0..=NTAU as i64 {
        let d =
            (delta[1].lmix.raw_block_at(n, m)[0] - fm * gloc[0].lmix.raw_block_at(n, m)[0]).abs();
        assert!(d < 1e-13, "realtime lmix row ({n},{m}): {d:.3e}");
    }
}

/// Sanity: the equilibrium bath rows are real (no accidental complex
/// factor from the hopping series) for both maps.
#[test]
fn equilibrium_bath_rows_are_real() {
    let tvec = vec![1.1; NT + 2];
    let mut s = new_solver();
    let (_log, _gloc, delta, _dcc) = run_equilibrium_multi(&mut s, &tvec, &[1, 0], 0.5, 1e-10, 50);
    for c in 0..2 {
        for m in 0..=NTAU {
            let z: Complex = delta[c].mat.block(m)[0];
            assert!(z.im.abs() < 1e-14, "channel {c} slice {m}: {z:?}");
        }
    }
}

/// A bath map that indexes a non-existent channel must fail fast
/// instead of silently reading out of range.
#[test]
#[should_panic(expected = "bath_map")]
fn bath_map_out_of_range_is_rejected() {
    let mut s = new_solver();
    let tvec = vec![1.0; NT + 2];
    let _ = run_equilibrium_multi(&mut s, &tvec, &[0, 2], 0.5, 1e-10, 1);
}

#[test]
#[should_panic(expected = "bath_map must have")]
fn bath_map_wrong_length_is_rejected() {
    let mut s = new_solver();
    let tvec = vec![1.0; NT + 2];
    let _ = run_equilibrium_multi(&mut s, &tvec, &[0], 0.5, 1e-10, 1);
}
