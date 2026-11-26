// Project : nQIST
// File    : examples/accuracy_vs_ed.rs
// Purpose : accuracy of the PPSC-like NCA and OCA and of the XCA
//           series at orders 1 and 2 against the built-in exact
//           diagonalization, on the dimer fixtures of the solver gates.
//
//           For every fixture, hybridization scale and solver the
//           program prints the largest deviation from the exact
//           reference in (i) the Matsubara Green's functions of the
//           hybridized modes, (ii) the orbital densities and (iii) the
//           orbital double occupancies. The fixtures are those of
//           `tests/singleband_dimer`, `tests/xca_twoband_dimer`,
//           `tests/threeband_dimer` and `tests/holstein_dimer`.
//
//           XCA(1) and NCA are the same approximation computed by two
//           independent kernels, and XCA(2) and OCA agree to the
//           convolution quadrature on every fixture (the NCA Green's
//           function has an error ~ V^2, the OCA one ~ V^4).
//
//           cargo run --release --example accuracy_vs_ed -- [--only a,b] [--quick]
//
//           `--only` keeps the fixtures whose name contains one of the
//           comma-separated keys; `--quick` runs the full hybridization
//           only. `NQIST_ACC_PROFILE=1` additionally prints the
//           deviation from the exact reference on the first ten
//           Matsubara points. The Matsubara OCA kernel of the
//           three-band fixtures costs minutes per solve.

use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{Contour, ContourGreen, Sign};
use nqist::dmft::set_bwd_from_fwd_mat;
use nqist::ed::{BathCoupling, EdReference, PhononCoupling, PhononSpec};
use nqist::hamilton::LocalHamiltonian;
use nqist::hamilton::LocalHamiltonian::{
    SingleBand, ThreeBand, ThreeBandKanamori, TwoBand, TwoBandKanamori,
};
use nqist::hilbert::{build, HilbertSpace, SectorClass};
use nqist::nca::{BosonBath, GfVertex, Hybridization};
use nqist::solver::{ImpuritySolver, Solver};

const BETA: f64 = 5.0;
const NT: usize = 6;
const OMEGA: f64 = 0.7;
const NPH: usize = 12;

/// One bath level coupled to one impurity mode at the full scale.
struct Bath {
    imp_mode: usize,
    eps: f64,
    v: f64,
}

struct Fixture {
    name: &'static str,
    norb: usize,
    ntau: usize,
    ham: LocalHamiltonian,
    bath: Vec<Bath>,
    /// impurity modes whose Matsubara Green's functions are compared
    modes: Vec<usize>,
    /// e-ph coupling of the doublon (None: no phonon)
    lambda: Option<f64>,
    scales: &'static [f64],
}

fn fixtures() -> Vec<Fixture> {
    let len = NT + 2;
    let s3: &'static [f64] = &[1.0, 0.5, 0.25];
    let s2: &'static [f64] = &[1.0, 0.5];
    let two = |ham: LocalHamiltonian, name: &'static str| Fixture {
        name,
        norb: 2,
        ntau: 96,
        ham,
        bath: vec![
            Bath {
                imp_mode: 0,
                eps: 0.5,
                v: 0.3,
            },
            Bath {
                imp_mode: 1,
                eps: -0.8,
                v: 0.35,
            },
        ],
        modes: vec![0, 1],
        lambda: None,
        scales: s3,
    };
    let three = |ham: LocalHamiltonian, name: &'static str| Fixture {
        name,
        norb: 3,
        ntau: 64,
        ham,
        bath: vec![
            Bath {
                imp_mode: 0,
                eps: 0.5,
                v: 0.3,
            },
            Bath {
                imp_mode: 1,
                eps: -0.8,
                v: 0.35,
            },
            Bath {
                imp_mode: 2,
                eps: 0.9,
                v: 0.3,
            },
        ],
        modes: vec![0, 1, 2],
        lambda: None,
        scales: s2,
    };
    vec![
        Fixture {
            name: "singleband",
            norb: 1,
            ntau: 96,
            ham: SingleBand {
                mu: 0.0,
                u: vec![2.0; len],
                eps_up: vec![0.15; len],
                eps_do: vec![-0.10; len],
            },
            bath: vec![
                Bath {
                    imp_mode: 0,
                    eps: 0.5,
                    v: 0.40,
                },
                Bath {
                    imp_mode: 1,
                    eps: -0.3,
                    v: 0.35,
                },
            ],
            modes: vec![0, 1],
            lambda: None,
            scales: s3,
        },
        two(
            TwoBand {
                mu: 0.0,
                u: vec![2.0; len],
                j: vec![0.4; len],
                delta: vec![0.8; len],
                bz: vec![0.0; len],
            },
            "twoband_density",
        ),
        two(
            TwoBandKanamori {
                mu: 0.0,
                u: vec![2.0; len],
                j: vec![0.4; len],
                delta: vec![0.8; len],
                bz: vec![0.0; len],
            },
            "twoband_kanamori",
        ),
        two(
            TwoBandKanamori {
                mu: 0.0,
                u: vec![2.0; len],
                j: vec![0.4; len],
                delta: vec![0.8; len],
                bz: vec![0.3; len],
            },
            "twoband_kanamori_bz",
        ),
        three(
            ThreeBand {
                mu: 0.0,
                u: vec![2.0; len],
                j: vec![0.4; len],
                delta: vec![0.8; len],
                bz: vec![0.0; len],
            },
            "threeband_density",
        ),
        three(
            ThreeBandKanamori {
                mu: 0.0,
                u: vec![2.0; len],
                j: vec![0.4; len],
                delta: vec![0.8; len],
                bz: vec![0.0; len],
            },
            "threeband_kanamori",
        ),
        three(
            ThreeBandKanamori {
                mu: 0.0,
                u: vec![2.0; len],
                j: vec![0.4; len],
                delta: vec![0.8; len],
                bz: vec![0.2; len],
            },
            "threeband_kanamori_bz",
        ),
        Fixture {
            name: "holstein",
            norb: 1,
            ntau: 64,
            ham: SingleBand {
                mu: 0.0,
                u: vec![2.0; len],
                eps_up: vec![0.0; len],
                eps_do: vec![0.0; len],
            },
            bath: vec![
                Bath {
                    imp_mode: 0,
                    eps: 0.5,
                    v: 0.4,
                },
                Bath {
                    imp_mode: 1,
                    eps: 0.5,
                    v: 0.4,
                },
            ],
            modes: vec![0, 1],
            lambda: Some(0.4),
            scales: s3,
        },
    ]
}

/// Exact thermal Green's function of a bath level.
fn k_tau(tau: f64, w: f64) -> f64 {
    -(-w * tau).exp() / (1.0 + (-BETA * w).exp())
}

fn ed_reference(f: &Fixture, hs: &HilbertSpace, scale: f64) -> EdReference {
    let bath: Vec<BathCoupling> = f
        .bath
        .iter()
        .map(|b| BathCoupling {
            imp_mode: b.imp_mode,
            eps: b.eps,
            v: b.v * scale,
        })
        .collect();
    let phonon = f.lambda.map(|lambda| PhononSpec {
        nph: NPH,
        omega: OMEGA,
        couplings: vec![PhononCoupling {
            ops: vec![(0, true), (0, false), (1, true), (1, false)],
            lambda: vec![lambda; NT + 2],
        }],
    });
    EdReference::build_with_phonon(&f.ham, hs, &bath, &[], BETA, phonon.as_ref())
}

/// Boson bath of the doublon-phonon coupling series `gvec`.
fn boson_bath(hs: &HilbertSpace, nt: usize, ntau: usize, h: f64, gvec: &[f64]) -> BosonBath {
    let mut d0 = nqist::boson::green_from_eps_phonon(nt, ntau, BETA, h, OMEGA);
    nqist::boson::scale_phonon(&mut d0, gvec);
    let bcontour = Contour::new(nt as i64, ntau as i64, 1, 1, nt as f64 * h, BETA);
    let mut d0_cc = ContourGreen::new(&bcontour, Sign::Bose, Complex::ZERO);
    nqist::boson::set_bwd_from_fwd_boson_mat(&mut d0_cc, &d0);
    for tstp in 0..=(nt as i64) {
        nqist::boson::set_bwd_from_fwd_boson_tstp(tstp, &mut d0_cc, &d0);
    }
    BosonBath {
        d0,
        d0_cc,
        blocks: nqist::boson::docc_blocks(hs),
    }
}

struct Outcome {
    gerr: Vec<f64>,
    nerr: Vec<f64>,
    derr: Vec<f64>,
    iters: usize,
    secs: f64,
}

fn run(f: &Fixture, solver: Solver, order: usize, scale: f64) -> (Outcome, Vec<f64>, Vec<f64>) {
    let hs = build(f.norb, SectorClass::SpinResolved);
    let h = 0.02;
    let mut imp = ImpuritySolver::with_solver(
        NT,
        f.ntau,
        BETA,
        h,
        5,
        solver,
        order,
        hs.clone(),
        f.ham.clone(),
    );
    imp.solve_atomic();
    if let Some(lambda) = f.lambda {
        imp.set_boson_bath(Some(boson_bath(&hs, NT, f.ntau, h, &vec![lambda; NT + 2])));
    }

    let contour = Contour::new(NT as i64, f.ntau as i64, 1, 1, 0.1, BETA);
    let mut fwd = Vec::new();
    let mut cc = Vec::new();
    for b in &f.bath {
        let mut d = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        for m in 0..=f.ntau {
            let tau = BETA * m as f64 / f.ntau as f64;
            d.mat.block_mut(m)[0] = Complex::new((b.v * scale).powi(2) * k_tau(tau, b.eps), 0.0);
        }
        let mut c = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        set_bwd_from_fwd_mat(&mut c, &d);
        fwd.push(d);
        cc.push(c);
    }
    let mut ints = Vec::new();
    for ((b, d), c) in f.bath.iter().zip(&fwd).zip(&cc) {
        let (orb, spin) = (b.imp_mode % f.norb, b.imp_mode / f.norb);
        ints.push(Hybridization {
            delta: d,
            op_cre_fl: hs.flavor(orb, spin, true),
            op_ann_fl: hs.flavor(orb, spin, false),
            fermion: true,
            backward: false,
        });
        ints.push(Hybridization {
            delta: c,
            op_cre_fl: hs.flavor(orb, spin, false),
            op_ann_fl: hs.flavor(orb, spin, true),
            fermion: true,
            backward: true,
        });
    }
    let verts: Vec<GfVertex> = f
        .modes
        .iter()
        .map(|&m| GfVertex {
            op_ann_fl: m,
            op_cre_fl: hs.flavor(m % f.norb, m / f.norb, true),
        })
        .collect();

    let t0 = std::time::Instant::now();
    let mut last: Option<Vec<Vec<f64>>> = None;
    let mut iters = 0usize;
    for _ in 0..200 {
        imp.pp_step(-1, &ints);
        let g = imp.get_spgf_mat(&verts, &ints);
        let cur: Vec<Vec<f64>> = g.iter().map(|r| r.iter().map(|z| z.re).collect()).collect();
        iters += 1;
        let done = last.as_ref().is_some_and(|l| {
            l.iter()
                .flatten()
                .zip(cur.iter().flatten())
                .all(|(a, b)| (a - b).abs() < 1e-10)
        });
        last = Some(cur);
        if done {
            break;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let g = last.expect("at least one iteration");
    imp.update_density_matrix(-1);

    let ed = ed_reference(f, &hs, scale);
    let dtau = BETA / f.ntau as f64;
    let gerr: Vec<f64> = f
        .modes
        .iter()
        .enumerate()
        .map(|(k, &m)| {
            (1..f.ntau)
                .map(|i| (g[k][i] - ed.greens_tau(m, dtau * i as f64)).abs())
                .fold(0.0f64, f64::max)
        })
        .collect();
    if std::env::var_os("NQIST_ACC_PROFILE").is_some() {
        let prof: Vec<String> = f
            .modes
            .iter()
            .enumerate()
            .map(|(k, &m)| {
                (1..=10usize)
                    .map(|i| format!("{:+.3e}", g[k][i] - ed.greens_tau(m, dtau * i as f64)))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        println!("PROFILE {:?} order {} scale {}: {}", solver, order, scale, prof.join(" | "));
    }
    let mut nerr = Vec::new();
    let mut derr = Vec::new();
    let mut n_ed = Vec::new();
    let mut d_ed = Vec::new();
    for o in 0..f.norb {
        let (fu, fd) = (o, f.norb + o);
        let n =
            ed.expectation(&[(fu, true), (fu, false)]) + ed.expectation(&[(fd, true), (fd, false)]);
        let d = ed.expectation(&[(fu, true), (fu, false), (fd, true), (fd, false)]);
        let (mut ns, mut ds) = (0.0f64, 0.0f64);
        for s in 0..hs.ns() {
            for (i, &psi) in hs.sectors[s].states.iter().enumerate() {
                let w = imp.rho[0][s].get(i, i).re;
                let (nu, nd) = (((psi >> fu) & 1) as f64, ((psi >> fd) & 1) as f64);
                ns += w * (nu + nd);
                ds += w * nu * nd;
            }
        }
        n_ed.push(n.re);
        d_ed.push(d.re);
        nerr.push((ns - n.re).abs());
        derr.push((ds - d.re).abs());
    }
    (
        Outcome {
            gerr,
            nerr,
            derr,
            iters,
            secs,
        },
        n_ed,
        d_ed,
    )
}

fn fmt(v: &[f64]) -> String {
    v.iter()
        .map(|x| format!("{x:.2e}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let quick = args.iter().any(|a| a == "--quick");
    let only: Vec<String> = args
        .iter()
        .position(|a| a == "--only")
        .map(|i| args[i + 1].split(',').map(String::from).collect())
        .unwrap_or_default();
    let methods = [
        ("NCA", Solver::Ppsc, 1usize),
        ("OCA", Solver::Ppsc, 2),
        ("XCA1", Solver::Xca, 1),
        ("XCA2", Solver::Xca, 2),
    ];
    println!(
        "{:<22} {:>5} {:<5} {:<34} {:>9} {:>9} {:>3} {:>7}",
        "fixture",
        "scale",
        "solver",
        "max |G - G_ED| per mode",
        "max |dn|",
        "max |dd|",
        "it",
        "secs"
    );
    for f in fixtures() {
        if !only.is_empty() && !only.iter().any(|k| f.name.contains(k)) {
            continue;
        }
        for &scale in f.scales {
            if quick && scale != 1.0 {
                continue;
            }
            let hs = build(f.norb, SectorClass::SpinResolved);
            let ed = ed_reference(&f, &hs, scale);
            let mut n = Vec::new();
            let mut d = Vec::new();
            for o in 0..f.norb {
                let (fu, fd) = (o, f.norb + o);
                n.push(
                    (ed.expectation(&[(fu, true), (fu, false)])
                        + ed.expectation(&[(fd, true), (fd, false)]))
                    .re,
                );
                d.push(
                    ed.expectation(&[(fu, true), (fu, false), (fd, true), (fd, false)])
                        .re,
                );
            }
            println!(
                "{:<22} {:>5.2} {:<5} n_ED {}   d_ED {}",
                f.name,
                scale,
                "ED",
                fmt(&n),
                fmt(&d)
            );
            for (name, solver, order) in methods {
                let (o, _, _) = run(&f, solver, order, scale);
                println!(
                    "{:<22} {:>5.2} {:<5} {:<34} {:>9} {:>9} {:>3} {:>7.1}",
                    f.name,
                    scale,
                    name,
                    fmt(&o.gerr),
                    format!("{:.2e}", o.nerr.iter().cloned().fold(0.0, f64::max)),
                    format!("{:.2e}", o.derr.iter().cloned().fold(0.0, f64::max)),
                    o.iters,
                    o.secs
                );
            }
        }
    }
}
