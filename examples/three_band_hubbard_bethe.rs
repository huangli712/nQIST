use nqist::dmft::{run_equilibrium_multi, run_realtime_multi, run_start_window_multi};
use nqist::hamilton::LocalHamiltonian::ThreeBand;
use nqist::hilbert::{SectorClass, build};
use nqist::io::{TomlDoc, fmt_g6};
use nqist::solver::ImpuritySolver;

/// Three-band case configuration (mirrors the fields of the C++
/// `three_band_hubbard_bethe.cpp` input file).
struct ThreeBandConfig {
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
    order: usize,
}

impl ThreeBandConfig {
    fn from_doc(doc: &TomlDoc) -> Result<ThreeBandConfig, String> {
        let r = doc.root();
        let f = |k: &str| -> Result<f64, String> { r.get_f64(k).map_err(|e| format!("{k}: {e}")) };
        Ok(ThreeBandConfig {
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
            order: r.get_usize("order").unwrap_or(1),
        })
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/ppsc/three_band_u35/case.toml".to_string());
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let doc = nqist::io::parse_toml(&src).unwrap_or_else(|e| panic!("parse {path}: {e}"));
    let cfg = ThreeBandConfig::from_doc(&doc).unwrap_or_else(|e| panic!("config: {e}"));

    // time arrays (length nt + 2, index tstp + 1, [0] = equilibrium)
    let mut u = vec![cfg.u; cfg.nt + 2];
    u[0] = cfg.u0;
    let mut j = vec![cfg.j; cfg.nt + 2];
    j[0] = cfg.j0;
    let mut delta = vec![cfg.delta; cfg.nt + 2];
    delta[0] = cfg.delta0;
    let mut bz = vec![cfg.bz; cfg.nt + 2];
    bz[0] = cfg.bz0;

    let hs = build(3, SectorClass::SpinResolved);
    let mut solver = ImpuritySolver::new(
        cfg.nt,
        cfg.ntau,
        cfg.beta,
        cfg.h,
        cfg.kt,
        cfg.order, // 1 = NCA (the reference case), 2 = OCA
        hs,
        ThreeBand {
            mu: cfg.mu,
            u,
            j,
            delta,
            bz,
        },
    );

    // hopping series (constant here); paramagnetic Bethe bath
    // (channel c = 2 * orb + spin takes its own Gloc)
    let tvec = vec![cfg.t_hop; cfg.nt + 2];
    let bath_map: Vec<usize> = (0..6).collect();

    // -- equilibrium (Matsubara) stage
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

    // -- equilibrium observables
    let obs = nqist::observable::update_exp_vals_three_band(&solver, -1);
    println!(
        "eqobs: Q={} n={} m={} n1={} n2={} n3={} Eint={}",
        fmt_g6(obs.q),
        fmt_g6(obs.n1 + obs.n2 + obs.n3),
        fmt_g6(obs.m),
        fmt_g6(obs.n1),
        fmt_g6(obs.n2),
        fmt_g6(obs.n3),
        fmt_g6(obs.e_int)
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

    // -- real-time stage with the three-band observable table
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
