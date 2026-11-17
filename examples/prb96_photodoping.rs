// Project : nQIST
// File    : examples/prb96_photodoping.rs
// Purpose : prb96 (arXiv:1704.06101) photodoping reproduction: the
//           half-filled two-band Mott insulator (U = 15, W = 4,
//           J/U = 0.04, beta = 1) driven by a two-cycle hopping
//           modulation (Omega = 2.25 U, Gaussian envelope), mirror
//           of the two_band_hubbard_bethe.cpp reference run with the
//           same hopping time series (paper Fig. 6/7 setup).

use nqist::dmft::{run_equilibrium_multi, run_realtime_multi, run_start_window_multi};
use nqist::hamilton::LocalHamiltonian::TwoBand;
use nqist::hilbert::{SectorClass, build};
use nqist::io::{TomlDoc, fmt_g6};
use nqist::solver::ImpuritySolver;

struct PhotodopingConfig {
    nt: usize,
    ntau: usize,
    beta: f64,
    h: f64,
    itermax: usize,
    errmax: f64,
    iter_rtime: usize,
    kt: usize,
    linear_mixing: f64,
    mu: f64,
    u0: f64,
    u: f64,
    j0: f64,
    j: f64,
    delta0: f64,
    delta: f64,
    bz0: f64,
    bz: f64,
    t_file: String,
}

impl PhotodopingConfig {
    fn from_doc(doc: &TomlDoc) -> Result<PhotodopingConfig, String> {
        let r = doc.root();
        let f = |k: &str| -> Result<f64, String> { r.get_f64(k).map_err(|e| format!("{k}: {e}")) };
        Ok(PhotodopingConfig {
            nt: r.get_usize("nt")?,
            ntau: r.get_usize("ntau")?,
            beta: f("beta")?,
            h: f("h")?,
            itermax: r.get_usize("itermax")?,
            errmax: f("errmax")?,
            iter_rtime: r.get_usize("iter_rtime")?,
            kt: r.get_usize("kt")?,
            linear_mixing: f("linear_mixing")?,
            mu: f("mu")?,
            u0: f("u0")?,
            u: f("u")?,
            j0: f("j0")?,
            j: f("j")?,
            delta0: f("delta0")?,
            delta: f("delta")?,
            bz0: f("bz0")?,
            bz: f("bz")?,
            t_file: r.get_str("t_file")?.to_string(),
        })
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/ppsc/prb96_photodoping/case.toml".to_string());
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let doc = nqist::io::parse_toml(&src).unwrap_or_else(|e| panic!("parse {path}: {e}"));
    let cfg = PhotodopingConfig::from_doc(&doc).unwrap_or_else(|e| panic!("config: {e}"));

    // hopping time series (nt + 2 entries, C++ convention)
    let tdir = std::path::Path::new(&path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    let tpath = tdir.join(&cfg.t_file);
    let tsrc =
        std::fs::read_to_string(&tpath).unwrap_or_else(|e| panic!("read {}: {e}", tpath.display()));
    let tvec: Vec<f64> = tsrc
        .lines()
        .map(|l| l.trim().parse().expect("t-series value"))
        .collect();
    assert_eq!(tvec.len(), cfg.nt + 2, "t-series length");

    // time arrays (constant interactions; only the hopping pulses)
    let u = vec![cfg.u; cfg.nt + 2];
    let j = vec![cfg.j; cfg.nt + 2];
    let delta = vec![cfg.delta; cfg.nt + 2];
    let bz = vec![cfg.bz; cfg.nt + 2];

    let hs = build(2, SectorClass::SpinResolved);
    let mut solver = ImpuritySolver::new(
        cfg.nt,
        cfg.ntau,
        cfg.beta,
        cfg.h,
        cfg.kt,
        1, // NCA: exact enough in the deep Mott regime (paper Fig. 2)
        hs,
        TwoBand {
            mu: cfg.mu,
            u,
            j,
            delta,
            bz,
        },
    );

    // -- equilibrium (Matsubara) stage
    let (log, mut gloc, mut delta, mut delta_cc) = run_equilibrium_multi(
        &mut solver,
        &tvec,
        &[0, 1, 2, 3],
        cfg.linear_mixing,
        cfg.errmax,
        cfg.itermax,
    );
    println!("{}", log.atomic_line);
    for (dens, iter) in log.dens_lines.iter().zip(&log.iter_lines) {
        println!("{dens}");
        println!("{iter}");
    }

    // -- equilibrium observables
    let obs = nqist::observable::update_exp_vals_two_band(&mut solver, -1);
    println!(
        "eqobs: Q={} n={} m={} n1={} n2={} Eint={}",
        fmt_g6(obs.q),
        fmt_g6(obs.n1 + obs.n2),
        fmt_g6(obs.m),
        fmt_g6(obs.n1),
        fmt_g6(obs.n2),
        fmt_g6(obs.e_int)
    );

    // -- START window
    let start = run_start_window_multi(
        &mut solver,
        &tvec,
        &[0, 1, 2, 3],
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        cfg.errmax,
        cfg.itermax,
    );
    for line in &start.start_lines {
        println!("{line}");
    }

    // -- real-time stage with multiplet observables
    let (tstp_lines, obs_lines) = run_realtime_multi(
        &mut solver,
        &tvec,
        &[0, 1, 2, 3],
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
