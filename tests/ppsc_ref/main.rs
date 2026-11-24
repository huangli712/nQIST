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
// P7b: two-band Kanamori quench at OCA (order = 2) vs the archived
// reference (data/ppsc/two_band_oca/golden.txt, produced by
// ppsc_programs/two_band_hubbard_bethe.ex on the 30 x 64 grid). Same
// comparison as the NCA gate above (atomic line, iteration errors,
// START iterations, real-time multiplet table), plus the equilibrium
// `eqobs` line and the START errors. HEAVY: the Matsubara OCA kernel
// costs ~30 s per DMFT iteration here (C++: 51 s for the whole
// case); run with `cargo test --release -- --ignored two_band_oca`.
// ------------------------------------------------------------------

#[test]
#[ignore = "heavy: two-band OCA (~15 min); run with --release -- --ignored"]
fn two_band_oca_matches_reference() {
    use nqist::dmft::{run_equilibrium_multi, run_realtime_multi, run_start_window_multi};
    use nqist::hamilton::LocalHamiltonian::TwoBand;
    use nqist::observable::update_exp_vals_two_band;

    const REF: &str = include_str!("../../data/ppsc/two_band_oca/golden.txt");

    // print-resolution budget of the default C++ stream output
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 || (a - b).abs() <= 1e-5 * b.abs();
    let err_of = |l: &str| l.split_whitespace().last().unwrap().parse::<f64>().unwrap();

    let (nt, ntau) = (30usize, 64usize);
    let mut u = vec![3.5f64; nt + 2];
    u[0] = 3.0;
    let hs = build(2, SectorClass::SpinResolved);
    let mut solver = ImpuritySolver::new(
        nt,
        ntau,
        5.0,
        0.02,
        5,
        2,
        hs,
        TwoBand {
            mu: 0.0,
            u,
            j: vec![0.4; nt + 2],
            delta: vec![0.0; nt + 2],
            bz: vec![0.0; nt + 2],
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
        assert!(close(err_of(our), err_of(r)), "iter err {our} vs {r}");
    }

    // equilibrium observables: eqobs: Q n m n1 n2 Eint
    let obs = update_exp_vals_two_band(&solver, -1);
    let ours_eqobs = [
        obs.q,
        obs.n1 + obs.n2,
        obs.m,
        obs.n1,
        obs.n2,
        obs.e_int,
    ];
    let ref_eqobs_line = REF.lines().find(|l| l.starts_with("eqobs:")).expect("eqobs line");
    let ref_eqobs: Vec<f64> = ref_eqobs_line
        .split_whitespace()
        .skip(1)
        .map(|kv| kv.split('=').nth(1).unwrap().parse::<f64>().unwrap())
        .collect();
    assert_eq!(ref_eqobs.len(), 6, "eqobs fields");
    for (name, a, b) in [
        ("Q", ours_eqobs[0], ref_eqobs[0]),
        ("n", ours_eqobs[1], ref_eqobs[1]),
        ("m", ours_eqobs[2], ref_eqobs[2]),
        ("n1", ours_eqobs[3], ref_eqobs[3]),
        ("n2", ours_eqobs[4], ref_eqobs[4]),
        ("Eint", ours_eqobs[5], ref_eqobs[5]),
    ] {
        assert!(close(a, b), "eqobs {name}: {a} vs {b}");
    }
    assert!((obs.n1 + obs.n2 - 2.0).abs() < 1e-9, "half filling");

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
    let ref_start: Vec<&str> = REF.lines().filter(|l| l.starts_with("WARMUP:")).collect();
    assert_eq!(start.start_lines.len(), ref_start.len(), "START iterations");
    for (our, r) in start.start_lines.iter().zip(&ref_start) {
        assert!(close(err_of(our), err_of(r)), "START err {our} vs {r}");
    }

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
            if !close(ea, eb) {
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "{bad} obs diffs");
}

// ------------------------------------------------------------------
// Three-band Kanamori U quench 3.0 -> 3.5 at J = 0.4 vs the
// archived references (data/ppsc/three_band_u35{,_oca}/golden.txt,
// produced by ppsc_programs/three_band_hubbard_bethe.ex). The C++
// driver prints the `update_exp_vals` row "tstp, n, d1, d2, d3, m, Q"
// inside every pp_step call: once per equilibrium iteration, kt + 1
// rows per START iteration, and iter_rtime rows per real-time step
// (the last one is the converged row). NOTE the C++ "n" column is
// n1 + 2 n2 (n3 is replaced by a second n2 in
// three_band_hubbard.hpp), which equals n1 + n2 + n3 for the
// equivalent orbitals of this case.
// ------------------------------------------------------------------

/// One `(tstp, n, d1, d2, d3, m, Q)` row of the C++ three-band
/// `update_exp_vals` print.
fn three_band_ref_row(line: &str) -> [f64; 7] {
    let mut vals = line.split('=').nth(1).unwrap().split(',');
    let mut row = [0.0f64; 7];
    for v in row.iter_mut() {
        *v = vals.next().unwrap().trim().parse().unwrap();
    }
    row
}

/// Reference sections: the equilibrium rows (one per iteration), the
/// converged START-window rows (tstp 0..=kt, printed right before the
/// last WARMUP line) and the converged real-time rows (last
/// `iter_rtime` row of each step).
fn three_band_ref_sections(ref_log: &str) -> (Vec<[f64; 7]>, Vec<[f64; 7]>, Vec<[f64; 7]>) {
    let (mut eq, mut pending, mut start) = (Vec::new(), Vec::new(), Vec::new());
    for line in ref_log.lines() {
        if line.starts_with("tstp, n, d1") {
            let row = three_band_ref_row(line);
            // the equilibrium rows carry tstp = -1; the rows of the
            // START iterations (tstp 0..=kt) precede their WARMUP line
            if row[0] < 0.0 {
                eq.push(row);
            } else {
                pending.push(row);
            }
        } else if line.starts_with("WARMUP:") {
            start = std::mem::take(&mut pending);
        }
    }
    let mut rt: Vec<[f64; 7]> = Vec::new();
    for row in &pending {
        if rt.last().map(|l| l[0]) != Some(row[0]) {
            rt.push(*row);
        } else {
            *rt.last_mut().unwrap() = *row;
        }
    }
    (eq, start, rt)
}

fn check_three_band_reference(ref_log: &str, nt: usize, ntau: usize, order: usize) {
    use nqist::dmft::{run_equilibrium_multi, run_realtime_multi, run_start_window_multi};
    use nqist::hamilton::LocalHamiltonian::ThreeBand;
    use nqist::observable::update_exp_vals_three_band;

    const KT: usize = 5;
    let (kt, iter_rtime) = (KT, 4usize);
    let mut u = vec![3.5f64; nt + 2];
    u[0] = 3.0; // U0 = equilibrium value
    let hs = build(3, SectorClass::SpinResolved);
    let mut solver = ImpuritySolver::new(
        nt,
        ntau,
        5.0,
        0.02,
        kt,
        order,
        hs,
        ThreeBand {
            mu: 0.0,
            u,
            j: vec![0.4; nt + 2],
            delta: vec![0.0; nt + 2],
            bz: vec![0.0; nt + 2],
        },
    );
    let tvec = vec![1.0f64; nt + 2];
    let bath_map: Vec<usize> = (0..6).collect();
    // print-resolution budget of the default C++ stream output
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 || (a - b).abs() <= 1e-5 * b.abs();
    let row_ok = |tag: &str, o: &nqist::observable::ThreeBandObservables, r: &[f64; 7]| {
        for (name, a, b) in [
            ("n", o.n1 + o.n2 + o.n3, r[1]),
            ("d1", o.d1, r[2]),
            ("d2", o.d2, r[3]),
            ("d3", o.d3, r[4]),
            ("m", o.m, r[5]),
            ("Q", o.q, r[6]),
        ] {
            assert!(close(a, b), "{tag} {name}: {a} vs {b}");
        }
    };
    let (ref_eq, ref_start, ref_rt) = three_band_ref_sections(ref_log);

    // -- equilibrium (Matsubara) stage
    let (log, mut gloc, mut delta, mut delta_cc) =
        run_equilibrium_multi(&mut solver, &tvec, &bath_map, 0.5, 1e-8, 100);
    let ref_iters: Vec<&str> = ref_log.lines().filter(|l| l.starts_with("iter:")).collect();
    assert_eq!(log.iter_lines.len(), ref_iters.len(), "equilibrium iteration count");
    assert_eq!(ref_eq.len(), ref_iters.len(), "equilibrium observable rows");
    let ref_ppmu = ref_log
        .lines()
        .find(|l| l.starts_with("pp_dens"))
        .expect("pp_dens line");
    assert_eq!(log.atomic_line, ref_ppmu, "atomic-limit line");
    let err_of = |l: &str| l.split_whitespace().last().unwrap().parse::<f64>().unwrap();
    // the residual errors approach 1e-9, where print resolution is
    // the absolute floor: same combined budget as the observables
    for (our, r) in log.iter_lines.iter().zip(&ref_iters) {
        assert!(close(err_of(our), err_of(r)), "iter err {our} vs {r}");
    }
    row_ok(
        "equilibrium",
        &update_exp_vals_three_band(&solver, -1),
        ref_eq.last().unwrap(),
    );

    // -- START window: iteration errors and the converged observables
    let start = run_start_window_multi(
        &mut solver,
        &tvec,
        &bath_map,
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        1e-8,
        100,
    );
    let ref_start_iters: Vec<&str> = ref_log.lines().filter(|l| l.starts_with("WARMUP:")).collect();
    assert_eq!(start.start_lines.len(), ref_start_iters.len(), "START iterations");
    assert_eq!(ref_start.len(), kt + 1, "START observable rows");
    for (our, r) in start.start_lines.iter().zip(&ref_start_iters) {
        assert!(close(err_of(our), err_of(r)), "START err {our} vs {r}");
    }
    for n in 0..=kt as i64 {
        solver.update_density_matrix(n);
        let tag = format!("START tstp {n}");
        row_ok(&tag, &update_exp_vals_three_band(&solver, n), &ref_start[n as usize]);
    }

    // -- real-time stage (converged row of every step)
    let (_tstp_lines, obs_lines) = run_realtime_multi(
        &mut solver,
        &tvec,
        &bath_map,
        &mut gloc,
        &mut delta,
        &mut delta_cc,
        iter_rtime,
        (kt + 1) as i64,
        nt as i64,
        |_, _| {},
    );
    assert_eq!(obs_lines.len(), ref_rt.len(), "real-time steps");
    for (n, (our, r)) in ((kt + 1) as i64..=nt as i64).zip(obs_lines.iter().zip(&ref_rt)) {
        assert_eq!(r[0], n as f64, "real-time tstp order");
        // obs: tstp t n d1 d2 d3 m Q
        let v: Vec<&str> = our.split_whitespace().collect();
        for (k, name, b) in [(3usize, "n", r[1]), (4, "d1", r[2]), (5, "d2", r[3]), (6, "d3", r[4]), (7, "m", r[5]), (8, "Q", r[6])] {
            let a = v[k].parse::<f64>().unwrap();
            assert!(close(a, b), "real-time tstp {n} {name}: {a} vs {b}");
        }
    }
}

#[test]
fn three_band_matches_reference() {
    check_three_band_reference(
        include_str!("../../data/ppsc/three_band_u35/golden.txt"),
        30,
        64,
        1,
    );
}

/// HEAVY: the Matsubara OCA kernel costs ~30 s per DMFT iteration on
/// this grid (480 diagrams, sector dimensions up to 9; the C++ driver
/// needs 108 s for the whole case), so the gate is ignored for the
/// default suite; run with
/// `cargo test --release -- --ignored three_band_oca`.
#[test]
#[ignore = "heavy: three-band OCA Bethe gate (~1 h); run with --release -- --ignored"]
fn three_band_oca_matches_reference() {
    check_three_band_reference(
        include_str!("../../data/ppsc/three_band_u35_oca/golden.txt"),
        16,
        40,
        2,
    );
}

// ------------------------------------------------------------------
// P9: Hubbard-Holstein U quench 4.0 -> 6.0 (omega0 = 1, g = 0.3) vs
// the archived references (data/ppsc/holstein_u46{,_oca}/golden.txt,
// produced by ppsc_programs/single_band_hubbard_holstein_bethe.ex).
// The C++ driver prints the complex sector densities
// `n_s = -tr G^M_s(beta)` after solve_atomic (INIT) and after every
// Matsubara iteration, the DMFT errors of the equilibrium, START and
// real-time iterations (the real-time iteration is NOT converged
// within iter_rtime = 4 in this case: every row of the fixed-point
// sequence is compared), and the "AT TIME ... NOT CONVERGED"
// markers. Boson-line convention: the PPSC driver feeds D0 through
// the forward AND the backward boson channel (the library's
// `BosonBath` default), i.e. an effective coupling sqrt(2) g; see
// tests/holstein_dimer for the ED side of that convention.
// ------------------------------------------------------------------

/// The four sector densities `n_s: (re,-0)` of a reference / our line.
fn holstein_sector_values(line: &str) -> Vec<f64> {
    (0..4)
        .map(|s| {
            let tag = format!("n{s}: (");
            let i = line
                .find(&tag)
                .unwrap_or_else(|| panic!("sector {s} missing in {line}"));
            line[i + tag.len()..]
                .split(',')
                .next()
                .unwrap()
                .parse::<f64>()
                .unwrap()
        })
        .collect()
}

/// `n_s = -tr G^M_s(beta)` of the solver, in the reference print order.
fn holstein_sector_densities(solver: &ImpuritySolver) -> Vec<f64> {
    (0..solver.hs.ns())
        .map(|s| {
            let d = solver.hs.sectors[s].dim;
            let mut v = 0.0;
            for i in 0..d {
                v -= solver.pp_g[s].mat.block(solver.ntau)[i * d + i].re;
            }
            v
        })
        .collect()
}

fn check_holstein_reference(ref_log: &str, nt: usize, ntau: usize, order: usize) {
    use kadanoffbaym::types::{Contour, ContourGreen, MATSUBARA, Sign};
    use nqist::dmft::{
        get_gf_verts, get_pp_ints, set_bwd_from_fwd_mat, set_bwd_from_fwd_tstp, set_gloc_row,
    };
    use nqist::nca::BosonBath;

    let (beta, h, kt, itermax, errmax, iter_rtime) = (5.0, 0.02, 5usize, 100usize, 1e-8, 4usize);
    // PPSC counts the phonon line twice; the library counts it once,
    // so the equivalent nQIST coupling is sqrt(2) * g_ppsc.
    let (u0, u1, g_ppsc, omega) = (4.0f64, 6.0, 0.3, 1.0);
    let g = g_ppsc * 2.0f64.sqrt();
    // print-resolution budget of the default C++ stream output
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 || (a - b).abs() <= 1e-5 * b.abs();

    // -- reference sections
    // (err, sector densities) per equilibrium iteration;
    // (row, iter, err) per START row and real-time iteration
    let mut ref_eq: Vec<(f64, Vec<f64>)> = Vec::new();
    let mut ref_start: Vec<(i64, usize, f64)> = Vec::new();
    let mut ref_rt: Vec<(i64, usize, f64)> = Vec::new();
    let mut ref_not_converged = 0usize;
    {
        let lines: Vec<&str> = ref_log.lines().collect();
        let mut i = 0usize;
        while i < lines.len() {
            let l = lines[i];
            if l.starts_with("iter:") {
                let err = l.split_whitespace().last().unwrap().parse::<f64>().unwrap();
                let sectors = holstein_sector_values(lines[i + 1]);
                assert_eq!(sectors.len(), 4, "sector line after {l}");
                ref_eq.push((err, sectors));
                i += 2;
                continue;
            } else if l.starts_with("START time:") {
                let t: Vec<&str> = l.split_whitespace().collect();
                ref_start.push((t[2].parse().unwrap(), t[4].parse().unwrap(), t[6].parse().unwrap()));
            } else if l.starts_with("time: ") {
                let t: Vec<&str> = l.split_whitespace().collect();
                ref_rt.push((t[1].parse().unwrap(), t[3].parse().unwrap(), t[5].parse().unwrap()));
            } else if l.starts_with("AT TIME") {
                ref_not_converged += 1;
            }
            i += 1;
        }
    }
    assert!(!ref_eq.is_empty() && !ref_rt.is_empty(), "reference sections");

    // -- solver in the PPSC driver flow (plain iteration, no mixing)
    let hs = build(1, SectorClass::SpinResolved);
    let mut u = vec![u1; nt + 2];
    u[0] = u0;
    let mut solver = ImpuritySolver::new(
        nt,
        ntau,
        beta,
        h,
        kt,
        order,
        hs,
        SingleBand {
            mu: 0.0,
            u,
            eps_up: vec![0.0; nt + 2],
            eps_do: vec![0.0; nt + 2],
        },
    );
    // boson bath: both channels, coupling sqrt(2) * g_ppsc
    let mut d0 = nqist::boson::green_from_eps_phonon(nt, ntau, beta, h, omega);
    nqist::boson::scale_phonon(&mut d0, &vec![g; nt + 2]);
    let bcontour = Contour::new(nt as i64, ntau as i64, 1, 1, nt as f64 * h, beta);
    let mut d0_cc = ContourGreen::new(&bcontour, Sign::Bose, Complex::ZERO);
    nqist::boson::set_bwd_from_fwd_boson_mat(&mut d0_cc, &d0);
    for tstp in 0..=(nt as i64) {
        nqist::boson::set_bwd_from_fwd_boson_tstp(tstp, &mut d0_cc, &d0);
    }
    solver.set_boson_bath(Some(BosonBath {
        d0,
        d0_cc,
        blocks: nqist::boson::docc_blocks(&solver.hs),
    }));

    let contour = Contour::new(nt as i64, ntau as i64, 1, 1, nt as f64 * h, beta);
    let mk = || ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let (mut gloc, mut delta, mut delta_cc, mut gtmp) = (mk(), mk(), mk(), mk());
    let verts = get_gf_verts(&solver);

    // -- equilibrium (Matsubara) stage
    let atomic = solver.solve_atomic();
    let ref_ppmu = ref_log
        .lines()
        .find(|l| l.starts_with("pp_dens"))
        .expect("pp_dens line");
    assert_eq!(atomic, ref_ppmu, "atomic-limit line");
    let ref_init = ref_log
        .lines()
        .find(|l| l.starts_with("INIT:"))
        .expect("INIT line");
    for (a, b) in holstein_sector_densities(&solver)
        .iter()
        .zip(holstein_sector_values(ref_init))
    {
        assert!(close(*a, b), "INIT sector density {a} vs {b}");
    }
    let mut our_eq: Vec<(f64, Vec<f64>)> = Vec::new();
    let mut converged = false;
    for _iter in 1..=itermax {
        let pp_ints = get_pp_ints(&delta, &delta_cc, &solver);
        solver.pp_step(MATSUBARA, &pp_ints);
        let g_new = solver.get_spgf_mat(&verts, &pp_ints);
        for m in 0..=ntau {
            gloc.mat.block_mut(m)[0] = g_new[0][m];
        }
        let err = gtmp.distance_at(&gloc, MATSUBARA);
        for m in 0..=ntau {
            let v = gloc.mat.block(m)[0];
            gtmp.mat.block_mut(m)[0] = v;
        }
        for m in 0..=ntau {
            let v = gloc.mat.block(m)[0];
            delta.mat.block_mut(m)[0] = v;
        }
        set_bwd_from_fwd_mat(&mut delta_cc, &delta);
        our_eq.push((err, holstein_sector_densities(&solver)));
        if err < errmax {
            converged = true;
            break;
        }
    }
    assert!(converged, "Matsubara not converged");
    assert_eq!(our_eq.len(), ref_eq.len(), "equilibrium iteration count");
    for (k, ((e, s), (re, rs))) in our_eq.iter().zip(&ref_eq).enumerate() {
        assert!(close(*e, *re), "iter {} err {e} vs {re}", k + 1);
        for (a, b) in s.iter().zip(rs) {
            assert!(close(*a, *b), "iter {} sector density {a} vs {b}", k + 1);
        }
    }

    // -- START window
    solver.set_t0();
    let kti = kt as i64;
    let mut our_start: Vec<(i64, usize, f64)> = Vec::new();
    let mut converged = false;
    for iter in 1..=itermax {
        let pp_ints = get_pp_ints(&delta, &delta_cc, &solver);
        solver.pp_step(kti, &pp_ints);
        for n in 0..=kti {
            let pp_ints = get_pp_ints(&delta, &delta_cc, &solver);
            let gf = solver.get_spgf(&verts, n, &pp_ints).remove(0);
            set_gloc_row(&mut gloc, &gf);
            let ts = kadanoffbaym::storage::timestep_at(&gloc, n);
            kadanoffbaym::storage::set_timestep_at(&mut delta, &ts);
            set_bwd_from_fwd_tstp(n, &mut delta_cc, &delta);
        }
        let mut err = 0.0;
        for n in 0..=kti {
            err = gtmp.distance_at(&gloc, n);
            let ts = kadanoffbaym::storage::timestep_at(&gloc, n);
            kadanoffbaym::storage::set_timestep_at(&mut gtmp, &ts);
            our_start.push((n, iter, err));
        }
        if err < errmax {
            converged = true;
            break;
        }
    }
    assert!(converged, "START not converged");
    assert_eq!(our_start.len(), ref_start.len(), "START rows");
    for ((n, it, e), (rn, rit, re)) in our_start.iter().zip(&ref_start) {
        assert_eq!((n, it), (rn, rit), "START row order");
        assert!(close(*e, *re), "START row {n} iter {it} err {e} vs {re}");
    }

    // -- real-time stage (adaptive per-timestep iteration)
    let mut our_rt: Vec<(i64, usize, f64)> = Vec::new();
    let mut not_converged = 0usize;
    for tstp in (kt + 1) as i64..=(nt as i64) {
        solver.extrapolate_timestep(tstp - 1);
        for iter in 1..=iter_rtime {
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
            our_rt.push((tstp, iter, err));
            if err < errmax {
                break;
            } else if iter == iter_rtime {
                not_converged += 1;
            }
        }
    }
    assert_eq!(our_rt.len(), ref_rt.len(), "real-time iteration rows");
    assert_eq!(not_converged, ref_not_converged, "NOT CONVERGED markers");
    for ((t, it, e), (rt, rit, re)) in our_rt.iter().zip(&ref_rt) {
        assert_eq!((t, it), (rt, rit), "real-time row order");
        assert!(close(*e, *re), "real-time tstp {t} iter {it} err {e} vs {re}");
    }
}

#[test]
fn holstein_matches_reference() {
    check_holstein_reference(include_str!("../../data/ppsc/holstein_u46/golden.txt"), 30, 64, 1);
}

#[test]
fn holstein_oca_matches_reference() {
    check_holstein_reference(
        include_str!("../../data/ppsc/holstein_u46_oca/golden.txt"),
        16,
        40,
        2,
    );
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
