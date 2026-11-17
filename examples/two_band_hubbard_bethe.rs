use nqist::dmft::{run_equilibrium_multi, run_realtime_multi, run_start_window_multi};
use nqist::hamilton::LocalHamiltonian::TwoBand;
use nqist::hilbert::{SectorClass, build};
use nqist::io::{TomlDoc, fmt_g6};
use nqist::solver::ImpuritySolver;

/// Two-band case configuration (mirrors the fields of the C++
/// `two_band_hubbard_bethe.cpp` input file).
struct TwoBandConfig {
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
    t_hop: f64,
}

impl TwoBandConfig {
    fn from_doc(doc: &TomlDoc) -> Result<TwoBandConfig, String> {
        let r = doc.root();
        let f = |k: &str| -> Result<f64, String> { r.get_f64(k).map_err(|e| format!("{k}: {e}")) };
        Ok(TwoBandConfig {
            nt: r.get_usize("nt")? as usize,
            ntau: r.get_usize("ntau")? as usize,
            beta: f("beta")?,
            h: f("h")?,
            itermax: r.get_usize("itermax")? as usize,
            errmax: f("errmax")?,
            iter_rtime: r.get_usize("iter_rtime")? as usize,
            kt: r.get_usize("kt")? as usize,
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
            t_hop: f("t")?,
        })
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/ppsc/two_band_u35/case.toml".to_string());
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let doc = nqist::io::parse_toml(&src).unwrap_or_else(|e| panic!("parse {path}: {e}"));
    let cfg = TwoBandConfig::from_doc(&doc).unwrap_or_else(|e| panic!("config: {e}"));

    // time arrays (length nt + 2, index tstp + 1, [0] = equilibrium)
    let mut u = vec![cfg.u; cfg.nt + 2];
    u[0] = cfg.u0;
    let mut j = vec![cfg.j; cfg.nt + 2];
    j[0] = cfg.j0;
    let mut delta = vec![cfg.delta; cfg.nt + 2];
    delta[0] = cfg.delta0;
    let mut bz = vec![cfg.bz; cfg.nt + 2];
    bz[0] = cfg.bz0;

    let hs = build(2, SectorClass::SpinResolved);
    let mut solver = ImpuritySolver::new(
        cfg.nt,
        cfg.ntau,
        cfg.beta,
        cfg.h,
        cfg.kt,
        1, // NCA (order 1), as the reference case
        hs,
        TwoBand {
            mu: cfg.mu,
            u,
            j,
            delta,
            bz,
        },
    );

    // hopping series (constant here; the prb96 pulse example uses a
    // time-dependent one)
    let tvec = vec![cfg.t_hop; cfg.nt + 2];

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
