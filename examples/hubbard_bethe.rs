// Project : nQIST
// File    : examples/hubbard_bethe.rs
// Purpose : single-band Hubbard Bethe quench, mirror of
//           ppsc_programs/single_band_hubbard_bethe.cpp

use nqist::dmft::{get_gf_verts, run_equilibrium, run_realtime, run_start_window};
use nqist::hamilton::LocalHamiltonian::SingleBand;
use nqist::hilbert::{SectorClass, build};
use nqist::io::CaseConfig;
use nqist::solver::ImpuritySolver;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/ppsc/quench_u35/case.toml".to_string());
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let cfg = CaseConfig::from_str(&src).unwrap_or_else(|e| panic!("parse {path}: {e}"));

    // expand the scalar series to the C++ time-array convention
    // (length nt + 2, index tstp + 1, U[0] = U0 equilibrium value)
    let mut u = vec![cfg.u[0]; cfg.nt + 2];
    u[0] = cfg.u0;
    let eps_up = vec![cfg.eps[0]; cfg.nt + 2];
    let eps_do = eps_up.clone();

    let hs = build(1, SectorClass::SpinResolved);
    let mut solver = ImpuritySolver::new(
        cfg.nt,
        cfg.ntau,
        cfg.beta,
        cfg.h,
        cfg.kt,
        cfg.order as usize,
        hs,
        SingleBand {
            mu: cfg.mu,
            u,
            eps_up,
            eps_do,
        },
    );

    // -- equilibrium (Matsubara) stage
    let (log, mut gloc, mut delta, mut delta_cc) =
        run_equilibrium(&mut solver, cfg.linear_mixing, cfg.errmax, cfg.itermax);
    println!("{}", log.atomic_line);
    for (dens, iter) in log.dens_lines.iter().zip(&log.iter_lines) {
        println!("{dens}");
        println!("{iter}");
    }

    // -- START window
    let start = run_start_window(
        &mut solver,
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        cfg.errmax,
        cfg.itermax,
    );
    for line in &start.start_lines {
        println!("{line}");
    }

    // -- real-time stage with local observables
    let (tstp_lines, obs_lines) = run_realtime(
        &mut solver,
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        cfg.iter_rtime,
        (cfg.kt + 1) as i64,
        cfg.nt as i64,
        |_, _, _, _| {},
    );
    let _ = get_gf_verts(&solver);
    for (tstp, obs) in tstp_lines.iter().zip(&obs_lines) {
        println!("{tstp}");
        println!("{obs}");
    }
}
