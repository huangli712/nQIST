// Project : nQIST
// File    : examples/single_band_hubbard_afm_bethe.rs
// Purpose : antiferromagnetic two-sublattice Bethe DMFT, mirror of
//           ppsc_programs/single_band_hubbard_afm_bethe.cpp: the A/B
//           sublattices map to a spin-resolved impurity with the
//           cross Bethe self-consistency `Delta_up = v*^2 G_dn`,
//           `Delta_dn = v*^2 G_up`, and a staggered seed field
//           `eps_up = +s`, `eps_do = -s` acting on the Matsubara
//           branch only (time-array slot 0).

use nqist::dmft::{run_equilibrium_multi, run_realtime_multi, run_start_window_multi};
use nqist::hamilton::LocalHamiltonian::SingleBand;
use nqist::hilbert::{SectorClass, build};
use nqist::io::{TomlDoc, fmt_g6};
use nqist::solver::ImpuritySolver;

struct AfmConfig {
    nt: usize,
    ntau: usize,
    beta: f64,
    h: f64,
    itermax: usize,
    errmax: f64,
    iter_rtime: usize,
    kt: usize,
    order: usize,
    linear_mixing: f64,
    mu: f64,
    u0: f64,
    u: f64,
    seed: f64,
}

impl AfmConfig {
    fn from_doc(doc: &TomlDoc) -> Result<AfmConfig, String> {
        let r = doc.root();
        let f = |k: &str| -> Result<f64, String> { r.get_f64(k).map_err(|e| format!("{k}: {e}")) };
        Ok(AfmConfig {
            nt: r.get_usize("nt")?,
            ntau: r.get_usize("ntau")?,
            beta: f("beta")?,
            h: f("h")?,
            itermax: r.get_usize("itermax")?,
            errmax: f("errmax")?,
            iter_rtime: r.get_usize("iter_rtime")?,
            kt: r.get_usize("kt")?,
            order: r.get_usize("order")?,
            linear_mixing: f("linear_mixing")?,
            mu: f("mu")?,
            u0: f("U0")?,
            u: f("U")?,
            seed: f("seed")?,
        })
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/ppsc/afm_bethe/case.toml".to_string());
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let doc = nqist::io::parse_toml(&src).unwrap_or_else(|e| panic!("parse {path}: {e}"));
    let cfg = AfmConfig::from_doc(&doc).unwrap_or_else(|e| panic!("config: {e}"));

    // time arrays; the staggered seed acts on the Matsubara branch
    // only (slot 0), real-time entries are zero
    let mut u = vec![cfg.u; cfg.nt + 2];
    u[0] = cfg.u0;
    let mut eps_up = vec![0.0; cfg.nt + 2];
    let mut eps_do = vec![0.0; cfg.nt + 2];
    eps_up[0] = cfg.seed;
    eps_do[0] = -cfg.seed;
    let tvec = vec![1.0f64; cfg.nt + 2];
    let bath_map = [1usize, 0]; // AFM cross Bethe

    let hs = build(1, SectorClass::SpinResolved);
    let mut solver = ImpuritySolver::new(
        cfg.nt,
        cfg.ntau,
        cfg.beta,
        cfg.h,
        cfg.kt,
        cfg.order,
        hs,
        SingleBand {
            mu: cfg.mu,
            u,
            eps_up,
            eps_do,
        },
    );

    // -- equilibrium (seeded AFM Matsubara state)
    let (log, mut gloc, mut delta, mut delta_cc) = run_equilibrium_multi(
        &mut solver,
        &tvec,
        &bath_map,
        cfg.linear_mixing,
        cfg.errmax,
        cfg.itermax,
    );
    println!("{}", log.atomic_line);
    for (dens, iter) in log.dens_lines.iter().zip(&log.iter_lines) {
        println!("{dens}");
        println!("{iter}");
    }
    let eq = nqist::observable::update_exp_vals(&mut solver, -1);
    println!(
        "eqobs: n_up={} n_dn={} d={} m={} Q={}",
        fmt_g6(eq.n_up),
        fmt_g6(eq.n_dn),
        fmt_g6(eq.docc),
        fmt_g6(0.5 * (eq.n_up - eq.n_dn)),
        fmt_g6(eq.trace)
    );

    // -- START window
    let start = run_start_window_multi(
        &mut solver,
        &tvec,
        &bath_map,
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        cfg.errmax,
        cfg.itermax,
    );
    for line in &start.start_lines {
        println!("{line}");
    }

    // -- real-time stage (AFM obs format, m printed as (n_up-n_dn)/2)
    let (tstp_lines, obs_lines) = run_realtime_multi(
        &mut solver,
        &tvec,
        &bath_map,
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        cfg.iter_rtime,
        (cfg.kt + 1) as i64,
        cfg.nt as i64,
        |_, _| {},
    );
    for (tstp, obs) in tstp_lines.iter().zip(&obs_lines) {
        println!("{tstp}");
        println!("{obs}");
    }
}
