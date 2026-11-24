// Project : nQIST
// File    : examples/single_band_hubbard_holstein_bethe.rs
// Purpose : Holstein-coupled single-band Hubbard impurity on the Bethe
//           lattice, mirror of ppsc_programs/
//           single_band_hubbard_holstein_bethe.cpp: a static free
//           phonon propagator D0 = g(t) D0_free(t,t') g(t') with its
//           backward partner couples through the doublon vertex
//           (boson pseudo-particle diagrams on top of the fermion NCA
//           hybridization; the solver counts the phonon vertex once,
//           so `g` is the Hamiltonian coupling `lambda`, and the
//           PPSC doubled convention maps to sqrt(2) * g_ppsc).
//           Equilibrium uses plain iteration (no
//           mixing), the START window and the real-time stage use the
//           reference print formats.

use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{Contour, ContourGreen, MATSUBARA, Sign};
use nqist::dmft::{
    get_gf_verts, get_pp_ints, set_bwd_from_fwd_mat, set_bwd_from_fwd_tstp, set_gloc_row,
};
use nqist::hamilton::LocalHamiltonian::SingleBand;
use nqist::hilbert::{SectorClass, build};
use nqist::io::{TomlDoc, fmt_g6, parse_toml};
use nqist::nca::BosonBath;
use nqist::solver::ImpuritySolver;

struct HolsteinConfig {
    nt: usize,
    ntau: usize,
    beta: f64,
    h: f64,
    itermax: usize,
    order: usize,
    errmax: f64,
    iter_rtime: usize,
    kt: usize,
    mu: f64,
    u0: f64,
    u: f64,
    eps: f64,
    g: f64,
    omega: f64,
}

impl HolsteinConfig {
    fn from_doc(doc: &TomlDoc) -> Result<HolsteinConfig, String> {
        let r = doc.root();
        let f = |k: &str| -> Result<f64, String> { r.get_f64(k).map_err(|e| format!("{k}: {e}")) };
        Ok(HolsteinConfig {
            nt: r.get_usize("nt")?,
            ntau: r.get_usize("ntau")?,
            beta: f("beta")?,
            h: f("h")?,
            itermax: r.get_usize("itermax")?,
            order: r.get_usize("order")?,
            errmax: f("errmax")?,
            iter_rtime: r.get_usize("iter_rtime")?,
            kt: r.get_usize("kt")?,
            mu: f("mu")?,
            u0: f("U0")?,
            u: f("U")?,
            eps: f("eps")?,
            g: f("g")?,
            omega: f("omega")?,
        })
    }
}

/// Per-sector pseudo-particle densities `n_s = -tr G^M_s(beta)`,
/// printed in the reference format `n0: v n1: v ... `.
fn sector_line(solver: &ImpuritySolver, prefix: &str) -> String {
    // blueprint: complex-valued sector densities printed as "(re,im)"
    let mut line = String::from(prefix);
    for s in 0..solver.hs.ns() {
        let mut v = 0.0;
        let d = solver.hs.sectors[s].dim;
        for i in 0..d {
            v -= solver.pp_g[s].mat.block(solver.ntau)[i * d + i].re;
        }
        let _ = fmt_g6(-0.0);
        line.push_str(&format!("n{s}: ({},-0) ", fmt_g6(v)));
    }
    line
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/ppsc/holstein_u46/case.toml".to_string());
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let doc = parse_toml(&src).unwrap_or_else(|e| panic!("parse {path}: {e}"));
    let cfg = HolsteinConfig::from_doc(&doc).unwrap_or_else(|e| panic!("config: {e}"));

    // local problem: spin-conserved four-sector single band (the
    // blueprint single_band_fermi_diag keeps spin conservation, i.e.
    // the (N_up, N_dn) sectors)
    let hs = build(1, SectorClass::SpinResolved);
    let mut u = vec![cfg.u; cfg.nt + 2];
    u[0] = cfg.u0;
    let eps_up = vec![cfg.eps; cfg.nt + 2];
    let eps_do = vec![cfg.eps; cfg.nt + 2];
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

    // boson bath: free phonon propagator, coupling-scaled, with the
    // bosonic backward channel (the solver counts the vertex once, so
    // the coupling below is the Hamiltonian lambda; PPSC's doubled
    // convention maps to sqrt(2) * g_ppsc)
    let mut d0 = nqist::boson::green_from_eps_phonon(cfg.nt, cfg.ntau, cfg.beta, cfg.h, cfg.omega);
    let gvec = vec![cfg.g; cfg.nt + 2];
    nqist::boson::scale_phonon(&mut d0, &gvec);
    let tmax = cfg.nt as f64 * cfg.h;
    let bcontour = Contour::new(cfg.nt as i64, cfg.ntau as i64, 1, 1, tmax, cfg.beta);
    let mut d0_cc = ContourGreen::new(&bcontour, Sign::Bose, Complex::ZERO);
    nqist::boson::set_bwd_from_fwd_boson_mat(&mut d0_cc, &d0);
    for tstp in 0..=(cfg.nt as i64) {
        nqist::boson::set_bwd_from_fwd_boson_tstp(tstp, &mut d0_cc, &d0);
    }
    let blocks = nqist::boson::docc_blocks(&solver.hs);
    solver.set_boson_bath(Some(BosonBath { d0, d0_cc, blocks }));

    // fermion channel (Bethe, v* = 1)
    let contour = Contour::new(cfg.nt as i64, cfg.ntau as i64, 1, 1, tmax, cfg.beta);
    let mk = || ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let mut gloc = mk();
    let mut delta = mk();
    let mut delta_cc = mk();
    let mut gtmp = mk();
    let verts = get_gf_verts(&solver);

    // -- equilibrium (plain iteration, no mixing)
    let atomic = solver.solve_atomic();
    println!("{atomic}");
    println!("{}", sector_line(&solver, "INIT: "));
    let mut converged = false;
    for iter in 1..=cfg.itermax {
        let pp_ints = get_pp_ints(&delta, &delta_cc, &solver);
        solver.pp_step(MATSUBARA, &pp_ints);
        let g_new = solver.get_spgf_mat(&verts, &pp_ints);
        for m in 0..=cfg.ntau {
            gloc.mat.block_mut(m)[0] = g_new[0][m];
        }
        let err = gtmp.distance_at(&gloc, MATSUBARA);
        for m in 0..=cfg.ntau {
            let v = gloc.mat.block(m)[0];
            gtmp.mat.block_mut(m)[0] = v;
        }
        for m in 0..=cfg.ntau {
            let v = gloc.mat.block(m)[0];
            delta.mat.block_mut(m)[0] = v;
        }
        set_bwd_from_fwd_mat(&mut delta_cc, &delta);
        println!("iter:  {iter} err: {}", fmt_g6(err));
        println!("{}", sector_line(&solver, ""));
        if err < cfg.errmax {
            converged = true;
            break;
        }
    }
    if !converged {
        eprintln!("WARNING: Matsubara not converged");
        return;
    }

    // -- START window
    solver.set_t0();
    let kt = cfg.kt as i64;
    let mut converged = false;
    for iter in 1..=cfg.itermax {
        let pp_ints = get_pp_ints(&delta, &delta_cc, &solver);
        solver.pp_step(kt, &pp_ints);
        for n in 0..=kt {
            let pp_ints = get_pp_ints(&delta, &delta_cc, &solver);
            let gf = solver.get_spgf(&verts, n, &pp_ints).remove(0);
            set_gloc_row(&mut gloc, &gf);
            let ts = kadanoffbaym::storage::timestep_at(&gloc, n);
            kadanoffbaym::storage::set_timestep_at(&mut delta, &ts);
            set_bwd_from_fwd_tstp(n, &mut delta_cc, &delta);
        }
        let mut err = 0.0;
        for n in 0..=kt {
            err = gtmp.distance_at(&gloc, n);
            let ts = kadanoffbaym::storage::timestep_at(&gloc, n);
            kadanoffbaym::storage::set_timestep_at(&mut gtmp, &ts);
            println!("START time: {n} iter:  {iter} err: {}", fmt_g6(err));
        }
        println!();
        if err < cfg.errmax {
            converged = true;
            break;
        }
    }
    if !converged {
        eprintln!("WARNING: START not converged");
        return;
    }

    // -- real-time stage (adaptive per-timestep iteration)
    for tstp in (kt + 1)..=(cfg.nt as i64) {
        solver.extrapolate_timestep(tstp - 1);
        for iter in 1..=cfg.iter_rtime {
            let pp_ints = get_pp_ints(&delta, &delta_cc, &solver);
            solver.pp_step(tstp, &pp_ints);
            let gf = solver.get_spgf(&verts, tstp, &pp_ints);
            set_gloc_row(&mut gloc, &gf[0]);
            let ts = kadanoffbaym::storage::timestep_at(&gloc, tstp);
            kadanoffbaym::storage::set_timestep_at(&mut delta, &ts);
            set_bwd_from_fwd_tstp(tstp, &mut delta_cc, &delta);
            let err = gtmp.distance_at(&gloc, tstp);
            let ts = kadanoffbaym::storage::timestep_at(&gloc, tstp);
            kadanoffbaym::storage::set_timestep_at(&mut gtmp, &ts);
            println!("time: {tstp} iter  {iter} err: {}", fmt_g6(err));
            if err < cfg.errmax {
                break;
            } else if iter == cfg.iter_rtime {
                println!(
                    "AT TIME {tstp} NOT CONVERGED AFTER {} ITERATIONS ",
                    cfg.iter_rtime
                );
            }
        }
        println!();
    }
}
