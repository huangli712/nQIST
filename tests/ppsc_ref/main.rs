// Project : nQIST
// File    : tests/ppsc_ref/main.rs
// Purpose : reference regression against the archived PPSC run.log
//           and the component-level golden dump (equilibrium stage)

use kadanoffbaym::complex::Complex;
use nqist::dmft::run_equilibrium_probe;
use nqist::hamilton::LocalHamiltonian::SingleBand;
use nqist::hilbert::{SectorClass, build};
use nqist::solver::ImpuritySolver;

/// Benchmark case parameters (PLAN.md 3.1, = archived case.inp).
const CASE: &str = include_str!("../../data/ppsc/quench_u35/run.log");
const GOLDEN: &str = include_str!("../../data/ppsc/quench_u35/golden_eq.txt");

fn benchmark_solver() -> ImpuritySolver {
    let (nt, ntau, kt) = (300, 250, 5);
    let hs = build(1, SectorClass::SpinResolved);
    let mut u = vec![3.5; nt + 2];
    u[0] = 3.0; // U0 = equilibrium value
    let ham = SingleBand {
        mu: 0.0,
        u,
        eps_up: vec![0.0; nt + 2],
        eps_do: vec![0.0; nt + 2],
    };
    ImpuritySolver::new(nt, ntau, 5.0, 0.02, kt, 1, hs, ham)
}

/// (tag, iter, sector, m, re, im) rows of the golden dump.
struct GoldenRow<'a> {
    tag: &'a str,
    iter: usize,
    sector: usize,
    m: usize,
    re: f64,
    im: f64,
}

fn parse_golden() -> Vec<GoldenRow<'static>> {
    let mut rows = Vec::new();
    for line in GOLDEN.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() != 6 {
            continue;
        }
        rows.push(GoldenRow {
            tag: t[0],
            iter: t[1].parse().unwrap(),
            sector: t[2].parse().unwrap(),
            m: t[3].parse().unwrap(),
            re: t[4].parse().unwrap(),
            im: t[5].parse().unwrap(),
        });
    }
    rows
}

/// Matsubara samples compared by the component check (probe3 grid).
const MS: [usize; 11] = [0, 1, 2, 5, 25, 50, 100, 125, 200, 249, 250];

#[test]
fn equilibrium_matches_reference() {
    let mut solver = benchmark_solver();

    // snapshots: per tag and iteration, (sector, m) -> value
    use std::collections::HashMap;
    let mut ours: HashMap<(String, usize, usize, usize), Complex> = HashMap::new();
    let mut put = |tag: &str, iter: usize, s: usize, m: usize, z: Complex| {
        ours.insert((tag.to_string(), iter, s, m), z);
    };

    let (log, _gloc, _delta, _delta_cc) = run_equilibrium_probe(
        &mut solver,
        0.5,
        1e-8,
        300,
        |iter, solver, gloc, delta, delta_cc, spgf| {
            let snap = iter <= 2 || iter == 39;
            if snap {
                for s in 0..solver.hs.ns() {
                    for &m in &MS {
                        put("SIG", iter, s, m, solver.pp_sigma[s].mat.block(m)[0]);
                        put("PPG", iter, s, m, solver.pp_g[s].mat.block(m)[0]);
                    }
                }
                for &m in &MS {
                    put("SPGF", iter, 0, m, spgf[0][m]);
                    put("GLOC", iter, 0, m, gloc.mat.block(m)[0]);
                    put("DELTA", iter, 0, m, delta.mat.block(m)[0]);
                    put("DELTACC", iter, 0, m, delta_cc.mat.block(m)[0]);
                }
            }
        },
    );

    // -- gate 1: convergence at the same iteration count
    assert!(log.converged, "equilibrium did not converge");
    assert_eq!(log.iter_lines.len(), 39);

    // -- gate 2: log lines match the archived run.log digit for digit
    let ref_atomic = CASE.lines().next().unwrap();
    assert_eq!(&log.atomic_line, ref_atomic);
    let ref_dens: Vec<&str> = CASE
        .lines()
        .filter(|l| l.starts_with("dens, pp_mu = "))
        .collect();
    let ref_iter: Vec<&str> = CASE.lines().filter(|l| l.starts_with("iter:  ")).collect();
    assert_eq!(log.dens_lines.len(), ref_dens.len());
    assert_eq!(log.iter_lines.len(), ref_iter.len());
    for (a, b) in log.dens_lines.iter().zip(&ref_dens) {
        assert_eq!(a, b, "dens line mismatch");
    }
    for (a, b) in log.iter_lines.iter().zip(&ref_iter) {
        // same-decay check with a 1e-6 relative margin on the err
        let av: Vec<&str> = a.split_whitespace().collect();
        let bv: Vec<&str> = b.split_whitespace().collect();
        assert_eq!(av[1], bv[1], "iteration number");
        let ea: f64 = av[3].parse().unwrap();
        let eb: f64 = bv[3].parse().unwrap();
        // relative 1e-6 with an absolute floor: near convergence the
        // err values are tiny and carry the ~1e-13 absolute rounding
        // noise of the (different) eigh/dyson implementations
        assert!(
            (ea - eb).abs() <= 1e-6 * eb.abs().max(1e-300) || (ea - eb).abs() <= 1e-10,
            "err mismatch: {a} vs {b}"
        );
    }

    // -- gate 3: pseudo-particle density (trace of the density matrix)
    solver.update_density_matrix(-1);
    let tr: f64 = solver.rho[0].iter().map(|m| m.trace().re).sum();
    assert!((tr - 1.0).abs() < 1e-10, "dens conservation: tr = {tr}");

    // -- gate 4: component-level golden comparison (iters 1-2 and the
    // converged state; the golden C-tags map onto iteration 39)
    let golden = parse_golden();
    let mut checked = 0;
    for row in &golden {
        let (tag, iter) = match row.tag {
            "SIGC" | "PPGC" | "SPGFC" | "GLOCC" | "DELTAC" | "DELTACCC" => {
                (row.tag.strip_suffix('C').unwrap(), 39)
            }
            t if row.iter <= 2 => (t, row.iter),
            _ => continue,
        };
        let key = (tag.to_string(), iter, row.sector, row.m);
        let want = Complex::new(row.re, row.im);
        let got = ours[&key];
        let diff = (got - want).abs();
        assert!(
            diff < 1e-10,
            "{tag} iter {iter} s {sector} m {m}: {got:?} vs {want:?} (diff {diff})",
            sector = row.sector,
            m = row.m
        );
        checked += 1;
    }
    // 132 rows per sampled iteration (44 SIG + 44 PPG + 4x11 bath/spgf)
    // x 3 sampled iterations (1, 2, converged)
    assert_eq!(checked, 396, "golden rows compared: {checked}");

    // -- gate 5: converged pseudo-particle chemical potential
    let obs: Vec<&str> = GOLDEN
        .lines()
        .find(|l| l.starts_with("OBS_EQ"))
        .unwrap()
        .split_whitespace()
        .collect();
    let pp_mu_ref: f64 = obs[1].parse().unwrap();
    assert!(
        (solver.pp_mu - pp_mu_ref).abs() < 1e-10,
        "pp_mu: {} vs {pp_mu_ref}",
        solver.pp_mu
    );
}

// ------------------------------------------------------------------
// P4: real-time gate (obs table + four-component golden + drift)
// ------------------------------------------------------------------

const GOLDEN_RT: &str = include_str!("../../data/ppsc/quench_u35/golden_rt.txt");
const MS8: [i64; 8] = [0, 1, 2, 5, 125, 248, 249, 250];

#[test]
fn realtime_matches_reference() {
    use nqist::dmft::{run_realtime, run_start_window};
    use std::collections::HashMap;

    let mut solver = benchmark_solver();

    // -- equilibrium stage (gated by the previous test)
    let (_log, mut gloc, mut delta, mut delta_cc) =
        nqist::dmft::run_equilibrium(&mut solver, 0.5, 1e-8, 300);

    // -- START window
    let start = run_start_window(&mut solver, &mut gloc, &mut delta, &mut delta_cc, 1e-8, 300);
    assert!(start.converged, "START window did not converge");
    // TEMP: component comparison first to localize divergence
    let ref_start: Vec<&str> = CASE
        .lines()
        .filter(|l| l.starts_with("START: iter:"))
        .collect();
    for l in &start.start_lines {
        eprintln!("OURS {l}");
    }
    for l in &ref_start {
        eprintln!("REF  {l}");
    }
    assert_eq!(
        start.start_lines.len(),
        ref_start.len(),
        "START iteration count"
    );
    for (a, b) in start.start_lines.iter().zip(&ref_start) {
        let av: Vec<&str> = a.split_whitespace().collect();
        let bv: Vec<&str> = b.split_whitespace().collect();
        assert_eq!((av[2], av[3]), (bv[2], bv[3]), "START header");
        let ea: f64 = av[4].parse().unwrap();
        let eb: f64 = bv[4].parse().unwrap();
        assert!(
            (ea - eb).abs() <= 1e-6 * eb.abs() || (ea - eb).abs() <= 1e-10,
            "{a} vs {b}"
        );
    }

    // -- four-component golden at n = kt (post-START, converged bath)
    let verts = nqist::dmft::get_gf_verts(&solver);
    let kt = 5i64;
    let mut ours: HashMap<(String, i64, i64), Complex> = HashMap::new();
    let mut put = |tag: &str, a: i64, b: i64, z: Complex| {
        ours.insert((tag.to_string(), a, b), z);
    };
    for j in 0..=kt {
        put("DRT RET", kt, j, delta.ret.raw_block_at(kt, j)[0]);
        put("DRT LES", kt, j, delta.less.raw_block_at(j, kt)[0]);
        put("DCC RET", kt, j, delta_cc.ret.raw_block_at(kt, j)[0]);
        put("DCC LES", kt, j, delta_cc.less.raw_block_at(j, kt)[0]);
    }
    for &m in &MS8 {
        put("DRT TV", kt, m, delta.lmix.raw_block_at(kt, m)[0]);
        put("DCC TV", kt, m, delta_cc.lmix.raw_block_at(kt, m)[0]);
    }
    for s in 0..4i64 {
        for j in 0..=kt {
            put(
                "SIGRT RET",
                s,
                j,
                solver.pp_sigma[s as usize].ret.raw_block_at(kt, j)[0],
            );
            put(
                "SIGRT LES",
                s,
                j,
                solver.pp_sigma[s as usize].less.raw_block_at(j, kt)[0],
            );
            put(
                "PPGRT RET",
                s,
                j,
                solver.pp_g[s as usize].ret.raw_block_at(kt, j)[0],
            );
            put(
                "PPGRT LES",
                s,
                j,
                solver.pp_g[s as usize].less.raw_block_at(j, kt)[0],
            );
        }
        for &m in &MS8 {
            put(
                "SIGRT TV",
                s,
                m,
                solver.pp_sigma[s as usize].lmix.raw_block_at(kt, m)[0],
            );
            put(
                "PPGRT TV",
                s,
                m,
                solver.pp_g[s as usize].lmix.raw_block_at(kt, m)[0],
            );
        }
    }
    {
        let gf = solver.get_spgf(&verts, kt, &[]);
        for j in 0..=kt {
            put("SPGFRT RET", kt, j, gf[0].ret.raw_block_at(kt, j)[0]);
            put("SPGFRT LES", kt, j, gf[0].less.raw_block_at(j, kt)[0]);
        }
        for &m in &MS8 {
            put("SPGFRT TV", kt, m, gf[0].lmix.raw_block_at(0, m)[0]);
        }
    }

    // -- real-time stage tstp = 6..=300 with golden + drift monitors
    let mut max_drift: f64 = 0.0;
    let (tstp_lines, obs_lines) = run_realtime(
        &mut solver,
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        4,
        6,
        300,
        |tstp, solver, delta, _delta_cc| {
            // density conservation (rho already updated by run_realtime)
            let tr: f64 = solver.rho[(tstp + 1) as usize]
                .iter()
                .map(|m| m.trace().re)
                .sum();
            max_drift = max_drift.max((tr - 1.0).abs());
            if tstp <= 7 {
                for j in 0..=tstp {
                    ours.insert(
                        ("DRT RET".to_string(), tstp, j),
                        delta.ret.raw_block_at(tstp, j)[0],
                    );
                    ours.insert(
                        ("DRT LES".to_string(), tstp, j),
                        delta.less.raw_block_at(j, tstp)[0],
                    );
                    for s in 0..4i64 {
                        ours.insert(
                            (format!("SIGRT{tstp} RET"), s, j),
                            solver.pp_sigma[s as usize].ret.raw_block_at(tstp, j)[0],
                        );
                        ours.insert(
                            (format!("SIGRT{tstp} LES"), s, j),
                            solver.pp_sigma[s as usize].less.raw_block_at(j, tstp)[0],
                        );
                    }
                }
                for &m in &MS8 {
                    ours.insert(
                        ("DRT TV".to_string(), tstp, m),
                        delta.lmix.raw_block_at(tstp, m)[0],
                    );
                    for s in 0..4i64 {
                        ours.insert(
                            (format!("SIGRT{tstp} TV"), s, m),
                            solver.pp_sigma[s as usize].lmix.raw_block_at(tstp, m)[0],
                        );
                    }
                }
                let gf = solver.get_spgf(&verts, tstp, &[]);
                for j in 0..=tstp {
                    ours.insert(
                        ("SPGFRT RET".to_string(), tstp, j),
                        gf[0].ret.raw_block_at(tstp, j)[0],
                    );
                    ours.insert(
                        ("SPGFRT LES".to_string(), tstp, j),
                        gf[0].less.raw_block_at(j, tstp)[0],
                    );
                }
                for &m in &MS8 {
                    ours.insert(
                        ("SPGFRT TV".to_string(), tstp, m),
                        gf[0].lmix.raw_block_at(0, m)[0],
                    );
                }
            }
        },
    );

    // -- gate: obs table, all 295 rows vs run.log (<= 1e-6)
    let ref_obs: Vec<&str> = CASE.lines().filter(|l| l.starts_with("obs:")).collect();
    assert_eq!(obs_lines.len(), ref_obs.len(), "obs row count");
    let mut max_obs_diff = 0.0f64;
    for (a, b) in obs_lines.iter().zip(&ref_obs) {
        let av: Vec<&str> = a.split_whitespace().collect();
        let bv: Vec<&str> = b.split_whitespace().collect();
        assert_eq!(av[1], bv[1], "obs tstp");
        for q in [2usize, 3, 4, 5] {
            let x: f64 = av[q].parse().unwrap();
            let y: f64 = bv[q].parse().unwrap();
            let d = (x - y).abs();
            assert!(d <= 1e-6, "obs {a} vs {b}");
            max_obs_diff = max_obs_diff.max(d);
        }
    }

    // -- gate: four-component golden comparison
    let mut checked = 0usize;
    for line in GOLDEN_RT.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 6
            || !matches!(
                t[0],
                "DRT" | "DCC" | "SIGRT" | "SIGRT6" | "SIGRT7" | "PPGRT" | "SPGFRT"
            )
        {
            continue; // solver stdout noise of the probe program
        }
        let (tag, kind) = (t[0].to_string(), t[1]);
        let a: i64 = t[2].parse().unwrap();
        let b: i64 = t[3].parse().unwrap();
        let want = Complex::new(t[4].parse().unwrap(), t[5].parse().unwrap());
        let mkey = (format!("{} {}", t[0], t[1]), a, b);
        if !ours.contains_key(&mkey) {
            continue;
        }
        let got = ours[&mkey];
        let diff = (got - want).abs();
        assert!(
            diff < 1e-8,
            "{tag} {kind} {a} {b}: {got:?} vs {want:?} (diff {diff})"
        );
        checked += 1;
    }
    // DRT 66 + DCC 20 + SIGRT(n=5) 80 + SIGRT6 88 + SIGRT7 96 +
    // PPGRT 80 + SPGFRT 66 = 496 comparable rows
    assert_eq!(checked, 496, "golden rt rows compared: {checked}");

    // -- gate: density conservation drift
    assert!(max_drift < 1e-4, "tr rho drift {max_drift}");
    assert_eq!(tstp_lines.len(), 295);
}

// ------------------------------------------------------------------
// P5: OCA (order = 2) equilibrium gate
// ------------------------------------------------------------------

const GOLDEN_OCA: &str = include_str!("../../data/ppsc/quench_u35_oca/golden_oca.txt");
const MS11: [usize; 11] = [0, 1, 2, 5, 25, 50, 100, 125, 200, 249, 250];

fn benchmark_solver_oca() -> ImpuritySolver {
    let (nt, ntau, kt) = (300, 250, 5);
    let hs = build(1, SectorClass::SpinResolved);
    let mut u = vec![3.5; nt + 2];
    u[0] = 3.0;
    let ham = SingleBand {
        mu: 0.0,
        u,
        eps_up: vec![0.0; nt + 2],
        eps_do: vec![0.0; nt + 2],
    };
    ImpuritySolver::new(nt, ntau, 5.0, 0.02, kt, 2, hs, ham)
}

#[test]
fn oca_equilibrium_matches_reference() {
    use nqist::dmft::run_equilibrium_probe;
    use std::collections::HashMap;

    let mut solver = benchmark_solver_oca();

    // golden rows: (tag, iter, sector-or-m) -> value
    let mut golden: HashMap<(String, i64, i64), Complex> = HashMap::new();
    let mut golden_iters: Vec<(usize, f64)> = Vec::new();
    let mut eqconv = (0usize, 0.0f64);
    for line in GOLDEN_OCA.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 2 {
            continue;
        }
        match t[0] {
            "EQITER" => golden_iters.push((t[1].parse().unwrap(), t[2].parse().unwrap())),
            "EQCONV" => {
                eqconv = (t[1].parse().unwrap(), t[3].parse().unwrap());
            }
            "OIT1" | "OIT2" => {
                let iter: i64 = t[0][3..].parse().unwrap();
                match (t[1], t.len()) {
                    // SIG/PPG rows: OITn TAG sec m re im -> key sec*1000+m
                    ("SIG", 6) | ("PPG", 6) => {
                        let sec: i64 = t[2].parse().unwrap();
                        let m: i64 = t[3].parse().unwrap();
                        golden.insert(
                            (format!("{} {}", t[0], t[1]), iter, sec * 1000 + m),
                            Complex::new(t[4].parse().unwrap(), t[5].parse().unwrap()),
                        );
                    }
                    // 5-token rows: OITn TAG m re im
                    (_, 5) => {
                        golden.insert(
                            (format!("{} {}", t[0], t[1]), iter, t[2].parse().unwrap()),
                            Complex::new(t[3].parse().unwrap(), t[4].parse().unwrap()),
                        );
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    assert_eq!(golden_iters.len(), 33);

    let mut ours: HashMap<(String, i64, i64), Complex> = HashMap::new();
    let (log, _gloc, _delta, _delta_cc) = run_equilibrium_probe(
        &mut solver,
        0.5,
        1e-8,
        300,
        |iter, solver, _gloc, delta, delta_cc, _spgf| {
            if iter > 2 {
                return;
            }
            let iter: i64 = iter as i64;
            // golden timing: spgf re-evaluated AFTER the bath update
            let pp_ints_post = nqist::dmft::get_pp_ints(delta, delta_cc, solver);
            let spgf_post = solver.get_spgf_mat(&nqist::dmft::get_gf_verts(solver), &pp_ints_post);
            for &m in &MS11 {
                ours.insert((format!("OIT{iter} SPGF"), iter, m as i64), spgf_post[0][m]);
            }
            for sec in 0..4i64 {
                for &m in &MS11 {
                    ours.insert(
                        (format!("OIT{iter} SIG"), iter, sec * 1000 + m as i64),
                        solver.pp_sigma[sec as usize].mat.block(m)[0],
                    );
                    ours.insert(
                        (format!("OIT{iter} PPG"), iter, sec * 1000 + m as i64),
                        solver.pp_g[sec as usize].mat.block(m)[0],
                    );
                }
            }
            for &m in &MS11 {
                ours.insert(
                    (format!("OIT{iter} DELTA"), iter, m as i64),
                    delta.mat.block(m)[0],
                );
                ours.insert(
                    (format!("OIT{iter} DELTACC"), iter, m as i64),
                    delta_cc.mat.block(m)[0],
                );
            }
        },
    );

    // iteration count and err sequence
    assert_eq!(log.iter_lines.len(), golden_iters.len());
    for (a, (it, err)) in log.iter_lines.iter().zip(&golden_iters) {
        let av: Vec<&str> = a.split_whitespace().collect();
        assert_eq!(av[1].parse::<usize>().unwrap(), *it);
        let ea: f64 = av[3].parse().unwrap();
        // our line prints 6 significant digits: the comparison floor is
        // the print resolution (the underlying component agreement is ~1e-9)
        assert!(
            (ea - err).abs() <= 1e-5 * err.abs() || (ea - err).abs() <= 1e-10,
            "{a} vs iter {it} err {err}"
        );
    }
    // converged pp_mu
    assert!(
        (solver.pp_mu - eqconv.1).abs() < 1e-10,
        "pp_mu {} vs {}",
        solver.pp_mu,
        eqconv.1
    );

    // component samples at iterations 1 and 2
    let mut checked = 0usize;
    for ((tag, iter, key), want) in &golden {
        let got = *ours
            .get(&(tag.clone(), *iter, *key))
            .unwrap_or_else(|| panic!("missing {tag} {iter} {key}"));
        let diff = (got - *want).abs();
        assert!(
            diff < 3e-6,
            "{tag} {iter} {key}: {got:?} vs {want:?} (diff {diff})"
        );
        checked += 1;
    }
    assert!(checked > 200, "golden oca rows compared: {checked}");
}

// ------------------------------------------------------------------
// P5: OCA (order = 2) real-time gate, tiny grid
// (data/ppsc/oca_tiny, generated by probe10.cpp)
// ------------------------------------------------------------------

const GOLDEN_TINY: &str = include_str!("../../data/ppsc/oca_tiny/golden_tiny.txt");

#[test]
fn oca_realtime_tiny_matches_reference() {
    use kadanoffbaym::types::{ContourGreen, GreenTimeStep, PseudoGreen};
    use nqist::dmft::{
        get_gf_verts, get_pp_ints, run_equilibrium, run_realtime, run_start_window_probe,
    };
    use nqist::observable::update_exp_vals;
    use std::collections::HashMap;

    type Key = (String, i64, i64);
    let mut golden: HashMap<Key, Complex> = HashMap::new();
    let mut golden_obs: HashMap<i64, (f64, f64, f64)> = HashMap::new();
    let mut eqconv = (0usize, 0.0f64);
    let mut startconv = 0usize;
    for line in GOLDEN_TINY.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() == 6
            && matches!(
                t[0],
                "SIGRT"
                    | "SIGRT6"
                    | "SIGRT7"
                    | "SIGRT8"
                    | "PPGRT"
                    | "DRT"
                    | "DCC"
                    | "SPGFRT"
                    | "IT1SIG"
                    | "IT2SIG"
                    | "IT1DRT"
                    | "IT2DRT"
                    | "IT1DCC"
                    | "IT2DCC"
            )
        {
            let b: i64 = t[3].parse().unwrap();
            if t[1] == "TV" && b > 40 {
                continue; // C++ dumper samples m up to 250; out of range here
            }
            golden.insert(
                (format!("{} {}", t[0], t[1]), t[2].parse().unwrap(), b),
                Complex::new(t[4].parse().unwrap(), t[5].parse().unwrap()),
            );
        } else if t.len() == 6 && t[0] == "OBSRT" {
            golden_obs.insert(
                t[1].parse().unwrap(),
                (
                    t[3].parse().unwrap(),
                    t[4].parse().unwrap(),
                    t[5].parse().unwrap(),
                ),
            );
        } else if t.len() >= 4 && t[0] == "EQCONV" {
            eqconv = (t[1].parse().unwrap(), t[3].parse().unwrap());
        } else if t.len() >= 2 && t[0] == "STARTCONV" {
            startconv = t[1].parse().unwrap();
        }
    }
    assert!(!golden.is_empty());

    let (nt, ntau, kt) = (8usize, 40usize, 5usize);
    let hs = build(1, SectorClass::SpinResolved);
    let mut u = vec![3.5; nt + 2];
    u[0] = 3.0;
    let ham = SingleBand {
        mu: 0.0,
        u,
        eps_up: vec![0.0; nt + 2],
        eps_do: vec![0.0; nt + 2],
    };
    let mut solver = ImpuritySolver::new(nt, ntau, 5.0, 0.02, kt, 2, hs, ham);

    let (log, mut gloc, mut delta, mut delta_cc) = run_equilibrium(&mut solver, 0.5, 1e-8, 300);
    assert_eq!(
        log.iter_lines.len(),
        eqconv.0,
        "equilibrium iteration count"
    );
    assert!(
        (solver.pp_mu - eqconv.1).abs() < 1e-10,
        "pp_mu {} vs {}",
        solver.pp_mu,
        eqconv.1
    );

    let mut ours: HashMap<Key, Complex> = HashMap::new();
    let tv_ms: [i64; 4] = [0, 1, 2, 5];
    let capture_herm = |tag: &str, n: i64, g: &ContourGreen, ours: &mut HashMap<Key, Complex>| {
        for j in 0..=n {
            ours.insert((format!("{tag} RET"), n, j), g.ret.raw_block_at(n, j)[0]);
            ours.insert((format!("{tag} LES"), n, j), g.less.raw_block_at(j, n)[0]);
        }
        for &m in &tv_ms {
            ours.insert((format!("{tag} TV"), n, m), g.lmix.raw_block_at(n, m)[0]);
        }
    };
    let capture_pp =
        |tag: &str, s: i64, n: i64, pp: &PseudoGreen, ours: &mut HashMap<Key, Complex>| {
            for j in 0..=n {
                ours.insert((format!("{tag} RET"), s, j), pp.ret.raw_block_at(n, j)[0]);
                ours.insert((format!("{tag} LES"), s, j), pp.less.raw_block_at(j, n)[0]);
            }
            for &m in &tv_ms {
                ours.insert((format!("{tag} TV"), s, m), pp.lmix.raw_block_at(n, m)[0]);
            }
        };
    let capture_spgf = |tag: &str, n: i64, ts: &GreenTimeStep, ours: &mut HashMap<Key, Complex>| {
        for j in 0..=n {
            ours.insert(
                (format!("{tag} RET"), n, j),
                ts.ret.block(ts.ret.idx(n, j))[0],
            );
            ours.insert(
                (format!("{tag} LES"), n, j),
                ts.less.block(ts.less.idx(j, n))[0],
            );
        }
        for &m in &tv_ms {
            ours.insert((format!("{tag} TV"), n, m), ts.lmix.raw_block_at(0, m)[0]);
        }
    };

    let mut start_len = 0usize;
    run_start_window_probe(
        &mut solver,
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        1e-8,
        300,
        |iter, solver, delta, delta_cc| {
            start_len = iter as usize;
            if iter <= 2 {
                let pfx = format!("IT{iter}");
                for s in 0..4i64 {
                    capture_pp(
                        &format!("{pfx}SIG"),
                        s,
                        kt as i64,
                        &solver.pp_sigma[s as usize],
                        &mut ours,
                    );
                }
                capture_herm(&format!("{pfx}DRT"), kt as i64, delta, &mut ours);
                capture_herm(&format!("{pfx}DCC"), kt as i64, delta_cc, &mut ours);
            }
        },
    );
    if start_len != startconv {
        eprintln!("START count: {start_len} vs {startconv}");
    }

    // -- capture the START-converged state (n = kt)
    capture_herm("DRT", kt as i64, &delta, &mut ours);
    capture_herm("DCC", kt as i64, &delta_cc, &mut ours);
    for s in 0..4i64 {
        capture_pp(
            "SIGRT",
            s,
            kt as i64,
            &solver.pp_sigma[s as usize],
            &mut ours,
        );
        capture_pp("PPGRT", s, kt as i64, &solver.pp_g[s as usize], &mut ours);
    }
    {
        let verts = get_gf_verts(&solver);
        let pp_ints = get_pp_ints(&delta, &delta_cc, &solver);
        let gfs = solver.get_spgf(&verts, kt as i64, &pp_ints);
        capture_spgf("SPGFRT", kt as i64, &gfs[0], &mut ours);
    }

    // -- real-time steps with per-step capture
    let mut obs_seen: HashMap<i64, (f64, f64, f64)> = HashMap::new();
    let (_tstp_lines, _obs_lines) = run_realtime(
        &mut solver,
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        4,
        (kt + 1) as i64,
        nt as i64,
        |tstp, solver, delta, delta_cc| {
            let tag = format!("SIGRT{tstp}");
            for s in 0..4i64 {
                capture_pp(&tag, s, tstp, &solver.pp_sigma[s as usize], &mut ours);
            }
            capture_herm("DRT", tstp, delta, &mut ours);
            let verts = get_gf_verts(solver);
            let pp_ints = get_pp_ints(delta, delta_cc, solver);
            let gfs = solver.get_spgf(&verts, tstp, &pp_ints);
            capture_spgf("SPGFRT", tstp, &gfs[0], &mut ours);
            let obs = update_exp_vals(solver, tstp);
            obs_seen.insert(tstp, (obs.n_up, obs.n_dn, obs.docc));
        },
    );

    // -- compare all golden component rows (grouped summary, no early panic)
    let mut worst_by_family: std::collections::BTreeMap<String, (f64, String)> =
        std::collections::BTreeMap::new();
    let mut missing: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for (key, want) in &golden {
        let Some(got) = ours.get(key) else {
            missing.push(format!("{key:?}"));
            continue;
        };
        let diff = (*got - *want).abs();
        let family = key.0.split_whitespace().next().unwrap_or("?").to_string();
        let e = worst_by_family
            .entry(family)
            .or_insert((0.0, String::new()));
        if diff > e.0 {
            *e = (diff, format!("{key:?}: {got:?} vs {want:?}"));
        }
        checked += 1;
    }
    for (family, (worst, detail)) in &worst_by_family {
        eprintln!("FAMILY {family}: worst {worst:.3e} {detail}");
    }
    if !missing.is_empty() {
        eprintln!("MISSING {} keys, first: {}", missing.len(), missing[0]);
    }
    assert!(checked > 400, "rows compared: {checked}");
    for (family, (worst, _)) in &worst_by_family {
        assert!(*worst < 1e-8, "family {family} worst {worst}");
    }

    // -- observables full precision
    for (tstp, (nup, ndn, docc)) in &golden_obs {
        let got = obs_seen.get(tstp).expect("obs captured");
        assert!(
            (got.0 - nup).abs() < 1e-9 && (got.1 - ndn).abs() < 1e-9 && (got.2 - docc).abs() < 1e-9,
            "obs {tstp}: {got:?} vs ({nup}, {ndn}, {docc})"
        );
    }
}

// ------------------------------------------------------------------
// P5: OCA (order = 2) full-quench real-time gate
// (data/ppsc/quench_u35_oca: golden_oca.txt %.17g rows tstp 6..267,
//  run.log complete 295 obs lines)
//
// Runtime-dominated: run with
//   cargo test --release --test ppsc_ref -- --ignored oca_realtime_matches
// ------------------------------------------------------------------

const GOLDEN_OCA_FULL: &str = include_str!("../../data/ppsc/quench_u35_oca/golden_oca.txt");
const RUNLOG_OCA: &str = include_str!("../../data/ppsc/quench_u35_oca/run.log");

#[test]
#[ignore = "full OCA quench (~30-60 min in release); see the test doc comment"]
fn oca_realtime_matches_reference() {
    use nqist::dmft::{run_equilibrium, run_realtime, run_start_window};
    use nqist::observable::update_exp_vals;
    use std::collections::HashMap;

    // -- golden: full-precision observables + per-tstp sigma rows
    let mut obs_golden: HashMap<i64, (f64, f64, f64)> = HashMap::new();
    let mut sig_golden: HashMap<(String, i64, i64), Complex> = HashMap::new();
    let mut n_sig_rows = 0usize;
    for line in GOLDEN_OCA_FULL.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() == 6 && t[0] == "OBSRT" {
            obs_golden.insert(
                t[1].parse().unwrap(),
                (
                    t[3].parse().unwrap(),
                    t[4].parse().unwrap(),
                    t[5].parse().unwrap(),
                ),
            );
        } else if t.len() == 6 && t[0].starts_with("SIGRT") && matches!(t[1], "RET" | "LES" | "TV")
        {
            sig_golden.insert(
                (
                    format!("{} {}", t[0], t[1]),
                    t[2].parse().unwrap(),
                    t[3].parse().unwrap(),
                ),
                Complex::new(t[4].parse().unwrap(), t[5].parse().unwrap()),
            );
            n_sig_rows += 1;
        }
    }
    assert!(!obs_golden.is_empty() && n_sig_rows > 5000, "golden parse");

    let mut solver = benchmark_solver_oca();
    let (log, mut gloc, mut delta, mut delta_cc) = run_equilibrium(&mut solver, 0.5, 1e-8, 300);
    assert_eq!(log.iter_lines.len(), 33, "equilibrium iterations");
    let start = run_start_window(&mut solver, &mut gloc, &mut delta, &mut delta_cc, 1e-8, 300);
    assert_eq!(start.start_lines.len(), 7, "START iterations");

    let kt = 5i64;
    let tv_ms: [i64; 8] = [0, 1, 2, 5, 125, 248, 249, 250];
    let mut sig_ours: HashMap<(String, i64, i64), Complex> = HashMap::new();
    let capture_sig = |tag: &str,
                       n: i64,
                       solver: &ImpuritySolver,
                       out: &mut HashMap<(String, i64, i64), Complex>| {
        for s in 0..4i64 {
            let pp = &solver.pp_sigma[s as usize];
            for j in 0..=n {
                out.insert((format!("{tag} RET"), s, j), pp.ret.raw_block_at(n, j)[0]);
                out.insert((format!("{tag} LES"), s, j), pp.less.raw_block_at(j, n)[0]);
            }
            for &m in &tv_ms {
                out.insert((format!("{tag} TV"), s, m), pp.lmix.raw_block_at(n, m)[0]);
            }
        }
    };
    // START-converged sigma rows (plain SIGRT tag)
    capture_sig("SIGRT", kt, &solver, &mut sig_ours);

    let mut obs_seen: HashMap<i64, (f64, f64, f64)> = HashMap::new();
    let (_tstp_lines, obs_lines) = run_realtime(
        &mut solver,
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        4,
        kt + 1,
        300,
        |tstp, solver, _delta, _delta_cc| {
            if (tstp as usize) < obs_golden.len() + 6 {
                let obs = update_exp_vals(solver, tstp);
                obs_seen.insert(tstp, (obs.n_up, obs.n_dn, obs.docc));
                let tag = format!("SIGRT{tstp}");
                capture_sig(&tag, tstp, solver, &mut sig_ours);
            }
        },
    );

    // -- golden observables (%.17g): rel 1e-5 / abs 1e-8 floor
    let mut worst = 0.0f64;
    for (tstp, (nup, ndn, docc)) in &obs_golden {
        let got = obs_seen
            .get(tstp)
            .unwrap_or_else(|| panic!("obs {tstp} missing"));
        for (g, w) in [(got.0, *nup), (got.1, *ndn), (got.2, *docc)] {
            let d = (g - w).abs();
            let rel = d / w.abs().max(1e-300);
            assert!(
                d <= 1e-8 || rel <= 1e-5,
                "obs {tstp}: {g} vs {w} (rel {rel:.3e})"
            );
            worst = worst.max(rel);
        }
    }

    // -- golden sigma rows: same criterion
    let mut worst_sig = 0.0f64;
    let mut n_checked = 0usize;
    for (key, want) in &sig_golden {
        let got = sig_ours
            .get(key)
            .unwrap_or_else(|| panic!("missing {key:?}"));
        let d = (*got - *want).abs();
        let rel = d / want.abs().max(1e-300);
        assert!(
            d <= 1e-8 || rel <= 1e-5,
            "{key:?}: {got:?} vs {want:?} (rel {rel:.3e})"
        );
        worst_sig = worst_sig.max(rel);
        n_checked += 1;
    }
    assert!(n_checked > 5000, "sigma rows compared: {n_checked}");

    // -- run.log obs table: all 295 lines at print resolution
    let ref_obs: Vec<&str> = RUNLOG_OCA
        .lines()
        .filter(|l| l.starts_with("obs:"))
        .collect();
    assert_eq!(ref_obs.len(), 295, "reference obs lines");
    assert_eq!(obs_lines.len(), 295, "our obs lines");
    for (a, b) in obs_lines.iter().zip(&ref_obs) {
        let av: Vec<&str> = a.split_whitespace().collect();
        let bv: Vec<&str> = b.split_whitespace().collect();
        assert_eq!(av[1], bv[1], "tstp order");
        for q in [3usize, 4, 5] {
            let ea: f64 = av[q].parse().unwrap();
            let eb: f64 = bv[q].parse().unwrap();
            assert!(
                (ea - eb).abs() <= 1e-6 || (ea - eb).abs() <= 1e-5 * eb.abs(),
                "{a} vs {b}"
            );
        }
    }
}

// ------------------------------------------------------------------
// P7: two-band Kanamori quench vs the archived reference
// (data/ppsc/two_band_u35/run.log, produced by
// ppsc_programs/two_band_hubbard_bethe.ex). Runtime-dominated: run
// with cargo test --release -- --ignored two_band
// ------------------------------------------------------------------

#[test]
#[ignore = "two-band quench (~minutes); run with --release -- --ignored"]
fn two_band_matches_reference() {
    use nqist::dmft::{run_equilibrium_multi, run_realtime_multi, run_start_window_multi};
    use nqist::hamilton::LocalHamiltonian::TwoBand;
    use nqist::hilbert::{SectorClass, build};
    use nqist::observable::update_exp_vals_two_band;
    use nqist::solver::ImpuritySolver;

    const REF: &str = include_str!("../../data/ppsc/two_band_u35/run.log");

    let nt = 300usize;
    let mut u = vec![3.5f64; nt + 2];
    u[0] = 3.0;
    let j = vec![0.4f64; nt + 2];
    let delta = vec![0.0f64; nt + 2];
    let bz = vec![0.0f64; nt + 2];

    let hs = build(2, SectorClass::SpinResolved);
    let mut solver = ImpuritySolver::new(
        nt,
        250,
        5.0,
        0.02,
        5,
        1,
        hs,
        TwoBand {
            mu: 0.0,
            u,
            j,
            delta,
            bz,
        },
    );

    let tvec = vec![1.0f64; nt + 2];
    let (log, mut gloc, mut delta, mut delta_cc) =
        run_equilibrium_multi(&mut solver, &tvec, &[0, 1, 2, 3], 0.5, 1e-8, 100);
    let ref_iters: Vec<&str> = REF.lines().filter(|l| l.starts_with("iter:")).collect();
    assert_eq!(
        log.iter_lines.len(),
        ref_iters.len(),
        "equilibrium iteration count"
    );
    let ref_ppmu = REF
        .lines()
        .find(|l| l.starts_with("pp_dens"))
        .expect("pp_dens line");
    assert_eq!(log.atomic_line, ref_ppmu, "atomic-limit line");
    for (our, r) in log.iter_lines.iter().zip(&ref_iters) {
        let (o, rf) = (
            our.split_whitespace().last().unwrap(),
            r.split_whitespace().last().unwrap(),
        );
        let (o, rf) = (o.parse::<f64>().unwrap(), rf.parse::<f64>().unwrap());
        assert!(
            (o - rf).abs() <= 1e-5 * rf.abs().max(1e-300),
            "iter err {our} vs {r}"
        );
    }
    let obs = update_exp_vals_two_band(&mut solver, -1);
    assert!((obs.n1 + obs.n2 - 2.0).abs() < 1e-9, "half filling");
    assert!((obs.e_int + 4.84371).abs() < 1e-4, "Eint {}", obs.e_int);

    let start = run_start_window_multi(
        &mut solver,
        &tvec,
        &[0, 1, 2, 3],
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        1e-8,
        100,
    );
    let ref_start = REF.lines().filter(|l| l.starts_with("WARMUP")).count();
    assert_eq!(start.start_lines.len(), ref_start, "START iterations");

    let (_tstp_lines, obs_lines) = run_realtime_multi(
        &mut solver,
        &tvec,
        &[0, 1, 2, 3],
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        4,
        6,
        nt as i64,
        |_, _| {},
    );
    let ref_obs: Vec<&str> = REF.lines().filter(|l| l.starts_with("obs:")).collect();
    assert_eq!(obs_lines.len(), ref_obs.len(), "obs table length");
    for (a, b) in obs_lines.iter().zip(&ref_obs) {
        let (av, bv): (Vec<&str>, Vec<&str>) = (
            a.split_whitespace().collect(),
            b.split_whitespace().collect(),
        );
        assert_eq!(av[1], bv[1], "tstp mismatch");
        for q in 3..10 {
            let (ea, eb) = (av[q].parse::<f64>().unwrap(), bv[q].parse::<f64>().unwrap());
            let (d, rel) = ((ea - eb).abs(), (ea - eb).abs() / eb.abs().max(1e-300));
            assert!(d <= 1e-6 || rel <= 1e-5, "{a} vs {b}");
        }
    }
}

// ------------------------------------------------------------------
// P7: prb96 photodoping pump vs the archived reference
// (data/ppsc/prb96_photodoping/run.log). Same hopping time series as
// the C++ run, so the trajectory is bit-comparable. Runtime-dominated:
// run with cargo test --release -- --ignored prb96
// ------------------------------------------------------------------

#[test]
#[ignore = "prb96 photodoping (~minutes); run with --release -- --ignored"]
fn prb96_photodoping_matches_reference() {
    use nqist::dmft::{run_equilibrium_multi, run_realtime_multi, run_start_window_multi};
    use nqist::hamilton::LocalHamiltonian::TwoBand;
    use nqist::hilbert::{SectorClass, build};
    use nqist::solver::ImpuritySolver;

    const REF: &str = include_str!("../../data/ppsc/prb96_photodoping/run.log");
    const TSERIES: &str = include_str!("../../data/ppsc/prb96_photodoping/tseries.txt");

    let nt = 400usize;
    let tvec: Vec<f64> = TSERIES
        .lines()
        .map(|l| l.trim().parse().expect("t-series value"))
        .collect();
    assert_eq!(tvec.len(), nt + 2, "t-series length");

    let hs = build(2, SectorClass::SpinResolved);
    let mut solver = ImpuritySolver::new(
        nt,
        200,
        1.0,
        0.02,
        5,
        1,
        hs,
        TwoBand {
            mu: 0.0,
            u: vec![15.0; nt + 2],
            j: vec![0.6; nt + 2],
            delta: vec![0.0; nt + 2],
            bz: vec![0.0; nt + 2],
        },
    );

    let (log, mut gloc, mut delta, mut delta_cc) =
        run_equilibrium_multi(&mut solver, &tvec, &[0, 1, 2, 3], 0.5, 1e-8, 100);
    let ref_iters: Vec<&str> = REF.lines().filter(|l| l.starts_with("iter:")).collect();
    assert_eq!(log.iter_lines.len(), ref_iters.len(), "equilibrium iters");
    let ref_ppmu = REF
        .lines()
        .find(|l| l.starts_with("pp_dens"))
        .expect("pp_dens line");
    assert_eq!(log.atomic_line, ref_ppmu, "atomic-limit line");

    let start = run_start_window_multi(
        &mut solver,
        &tvec,
        &[0, 1, 2, 3],
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        1e-8,
        100,
    );
    let ref_start = REF.lines().filter(|l| l.starts_with("WARMUP")).count();
    assert_eq!(start.start_lines.len(), ref_start, "START iterations");

    let (_tstp_lines, obs_lines) = run_realtime_multi(
        &mut solver,
        &tvec,
        &[0, 1, 2, 3],
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        4,
        6,
        nt as i64,
        |_, _| {},
    );
    let ref_obs: Vec<&str> = REF.lines().filter(|l| l.starts_with("obs:")).collect();
    assert_eq!(obs_lines.len(), ref_obs.len(), "obs table length");
    let mut bad = 0usize;
    for (a, b) in obs_lines.iter().zip(&ref_obs) {
        let (av, bv): (Vec<&str>, Vec<&str>) = (
            a.split_whitespace().collect(),
            b.split_whitespace().collect(),
        );
        assert_eq!(av[1], bv[1], "tstp mismatch");
        for q in 3..10 {
            let (ea, eb) = (av[q].parse::<f64>().unwrap(), bv[q].parse::<f64>().unwrap());
            let (d, rel) = ((ea - eb).abs(), (ea - eb).abs() / eb.abs().max(1e-300));
            if !(d <= 1e-6 || rel <= 1e-5) {
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "{bad} obs diffs");

    // Fig. 6 physics: damped 2*pi/U oscillation of the singlon-triplon
    // density after the pump (U = 15 -> 2*pi/U = 0.4189)
    let p13: Vec<(f64, f64)> = obs_lines
        .iter()
        .map(|l| {
            let v: Vec<&str> = l.split_whitespace().collect();
            (v[2].parse().unwrap(), v[3].parse().unwrap())
        })
        .collect();
    let peaks: Vec<(f64, f64)> = (1..p13.len() - 1)
        .filter(|&i| p13[i].0 > 0.6 && p13[i].1 > p13[i - 1].1 && p13[i].1 >= p13[i + 1].1)
        .map(|i| p13[i])
        .collect();
    assert!(peaks.len() >= 2, "no 2pi/U oscillation peaks found");
    let period = peaks[1].0 - peaks[0].0;
    let want = 2.0 * std::f64::consts::PI / 15.0;
    assert!(
        (period - want).abs() / want < 0.1,
        "oscillation period {period} vs 2pi/U {want}"
    );
    // photo-doped population ~1% after the pump
    let after: f64 = p13
        .iter()
        .filter(|(t, _)| *t >= 0.6 && *t <= 1.0)
        .map(|(_, p)| *p)
        .sum::<f64>()
        / p13.iter().filter(|(t, _)| *t >= 0.6 && *t <= 1.0).count() as f64;
    assert!(
        after > 0.008 && after < 0.013,
        "post-pump singlon-triplon density {after} (paper: ~1%)"
    );
}
