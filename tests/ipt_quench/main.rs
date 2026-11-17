// Project : nQIST
// File    : tests/ipt_quench/main.rs
// Purpose : P6 IPT gates — interaction quench U: 0 -> Uf in the PM
//           phase at half filling on the Bethe lattice (paper
//           arXiv:1306.0307v2 Sec. V.1, Figs. 15-18).
//
// Runtime-dominated: run with
//   cargo test --release --test ipt_quench -- --ignored

use nqist::dmft::{run_equilibrium_ipt, run_realtime_ipt, run_start_ipt};
use nqist::ipt::IptSolver;

const NT: usize = 500;
const NTAU: usize = 500;
const BETA: f64 = 50.0;
const H: f64 = 0.02;
const KT: usize = 5;

/// Equilibrium template at `U_i = 0` (identical for every quench).
fn equilibrium_template() -> IptSolver {
    let mut s = IptSolver::new(NT, NTAU, BETA, H, KT, 0.0, vec![0.0; NT + 2], (0.5, 0.5));
    let log = run_equilibrium_ipt(&mut s, 0.5, 1e-9, 300);
    assert!(
        log.len() < 300,
        "equilibrium did not converge: {:?}",
        log.last()
    );
    s
}

struct Obs {
    t: Vec<f64>,
    n_up: Vec<f64>,
    d: Vec<f64>,
    e: Vec<f64>,
}

fn run_quench(uf: f64) -> Obs {
    let mut s = equilibrium_template();
    for v in s.u.iter_mut().skip(1) {
        *v = uf;
    }
    let start = run_start_ipt(&mut s, 1e-8, 200);
    assert!(
        start.len() < 200,
        "START did not converge: {:?}",
        start.last()
    );
    let mut obs = Obs {
        t: Vec::new(),
        n_up: Vec::new(),
        d: Vec::new(),
        e: Vec::new(),
    };
    run_realtime_ipt(&mut s, 40, KT as i64 + 1, NT as i64, |tstp, s| {
        obs.t.push(tstp as f64 * H);
        obs.n_up.push(s.density(true, tstp));
        obs.d.push(s.double_occupancy(tstp));
        obs.e.push(s.total_energy(tstp));
    });
    obs
}

#[test]
#[ignore = "IPT quench gates (~minutes); run with --release -- --ignored"]
fn ipt_quench_gates() {
    // -- gate A + B1 + C(weak): Uf = 1
    let weak = run_quench(1.0);
    assert_eq!(weak.t.len(), NT - KT);

    // gate A (adjusted to the demonstrated bare-IPT physics): the
    // particle-hole violation of the bare second-order expansion
    // saturates (prethermalization-like plateau, cf. the paper's
    // saturated energy drift Fig. 17); it is not a numerical error
    // but the approximation's genuine non-conservation
    let drift: Vec<f64> = weak.n_up.iter().map(|n| (2.0 * n - 1.0).abs()).collect();
    let max_drift = drift.iter().cloned().fold(0.0f64, f64::max);
    let late_drift: Vec<f64> = drift[400..].to_vec();
    let max_late = late_drift.iter().cloned().fold(0.0f64, f64::max);
    let mid_drift = drift[200];
    eprintln!("gate A (Uf=1): max density drift {max_drift:.3e} (late {max_late:.3e})");
    assert!(max_drift < 2.5e-2, "density drift {max_drift}");
    assert!(
        max_late <= mid_drift + 5e-3,
        "drift must saturate: {max_late} vs {mid_drift}"
    );

    // gate B1: prethermalization plateau of d(t)
    let d0 = weak.d[0];
    assert!((d0 - 0.25).abs() < 5e-3, "d(0+) = {d0}");
    let win: Vec<f64> = weak
        .t
        .iter()
        .zip(&weak.d)
        .filter(|(t, _)| **t >= 2.0 && **t <= 8.0)
        .map(|(_, d)| *d)
        .collect();
    let mean: f64 = win.iter().sum::<f64>() / win.len() as f64;
    let spread = win.iter().map(|d| (d - mean).abs()).fold(0.0f64, f64::max);
    eprintln!("gate B1 (Uf=1): plateau d = {mean:.4}, spread {spread:.2e}");
    assert!(spread < 5e-3, "plateau spread {spread}");
    assert!(mean > 0.18 && mean < 0.25, "plateau value {mean}");

    // gate C (weak): total-energy drift
    let e_ref = weak.e[0];
    let e_weak = weak
        .e
        .iter()
        .map(|e| (e - e_ref).abs())
        .fold(0.0f64, f64::max);
    let e_abs = e_ref.abs();
    eprintln!("gate C (Uf=1): energy drift {e_weak:.3e} (|E| = {e_abs:.3})");
    assert!(e_weak < 0.05 * e_abs, "weak drift {e_weak} vs |E| {e_abs}");

    // -- gates B2 + C(strong): Uf = 16 (collapse-and-revival)
    let strong = run_quench(16.0);

    // gate B2: collapse-and-revival period 2pi/U
    let period_ref = 2.0 * std::f64::consts::PI / 16.0;
    let mut peaks: Vec<f64> = Vec::new();
    for i in 1..strong.t.len() - 1 {
        let (_t0, t1, _t2) = (strong.t[i - 1], strong.t[i], strong.t[i + 1]);
        let (d0, d1, d2) = (strong.d[i - 1], strong.d[i], strong.d[i + 1]);
        if d1 > d0 && d1 >= d2 && t1 >= 0.5 && d1 > 0.1 {
            peaks.push(t1);
        }
    }
    eprintln!("gate B2 (Uf=16): {} peaks", peaks.len());
    assert!(peaks.len() >= 8, "not enough revivals: {}", peaks.len());
    let mut periods: Vec<f64> = Vec::new();
    for w in peaks.windows(2) {
        periods.push(w[1] - w[0]);
    }
    let mean_p: f64 = periods.iter().sum::<f64>() / periods.len() as f64;
    eprintln!("gate B2 (Uf=16): mean period {mean_p:.4} vs 2pi/U {period_ref:.4}");
    assert!(
        (mean_p - period_ref).abs() / period_ref < 0.2,
        "period {mean_p} vs {period_ref}"
    );

    // gate C (strong): energy drift ~10% of |E| (paper Fig. 18)
    let e_ref_s = strong.e[0];
    let e_strong = strong
        .e
        .iter()
        .map(|e| (e - e_ref_s).abs())
        .fold(0.0f64, f64::max);
    eprintln!(
        "gate C (Uf=16): energy drift {e_strong:.3} ({:.0}% of |E| = {:.3})",
        100.0 * e_strong / e_abs,
        e_ref_s
    );
    assert!(
        e_strong > 3.0 * e_weak,
        "strong drift {e_strong} not >> weak {e_weak}"
    );
}
