//! ED cross-check dump.
//!
//! Prints the built-in exact-diagonalization reference (equilibrium
//! Lehmann correlators and the quenched unitary-evolution correlators
//! of `nqist::ed`) for the dimer fixtures of the solver gates, together
//! with the model parameters, in a plain-text record format. The
//! external-ED cross-check (`scripts/ed_crosscheck`) feeds this dump to
//! open-source exact-diagonalization packages, which rebuild the same
//! models from the printed parameters and the documented Hamiltonian
//! formulas and compare the records one by one.
//!
//! Usage: `cargo run --release --example ed_crosscheck -- > nqist.txt`

use nqist::ed::{
    BathCoupling, EdEvolution, EdReference, PhononCoupling, PhononSpec,
};
use nqist::hamilton::LocalHamiltonian;
use nqist::hilbert::{SectorClass, build};

const BETA: f64 = 5.0;
const DT: f64 = 0.05;
const NT: usize = 12;
/// Power of two, so that the external references can use the same
/// truncation (Pomerol only builds `2^k` phonon levels).
const NPH: usize = 16;
const KINDS: [&str; 3] = ["ret", "les", "gtr"];
const DOCC: [(usize, bool); 4] = [(0, true), (0, false), (1, true), (1, false)];

#[derive(Clone, Copy)]
enum Variant {
    SingleBand,
    TwoBand,
    TwoBandKanamori,
    ThreeBand,
    ThreeBandKanamori,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Variant::SingleBand => "SingleBand",
            Variant::TwoBand => "TwoBand",
            Variant::TwoBandKanamori => "TwoBandKanamori",
            Variant::ThreeBand => "ThreeBand",
            Variant::ThreeBandKanamori => "ThreeBandKanamori",
        }
    }

    fn norb(self) -> usize {
        match self {
            Variant::SingleBand => 1,
            Variant::TwoBand | Variant::TwoBandKanamori => 2,
            _ => 3,
        }
    }
}

struct Spec {
    name: &'static str,
    ntau: usize,
    variant: Variant,
    /// `[mu, U, J, delta, Bz, eps_up, eps_do]`
    p: [f64; 7],
    /// Discrete bath levels.
    bath: Vec<BathCoupling>,
    /// `(omega, lambda)` of the Holstein phonon
    phonon: Option<(f64, f64)>,
    /// Quenched parameter name and value after `t = 0`.
    quench: (&'static str, f64),
}

fn series(x: f64) -> Vec<f64> {
    vec![x; 3]
}

fn ham(variant: Variant, p: &[f64; 7]) -> LocalHamiltonian {
    let [mu, u, j, delta, bz, eps_up, eps_do] = *p;
    match variant {
        Variant::SingleBand => LocalHamiltonian::SingleBand {
            mu,
            u: series(u),
            eps_up: series(eps_up),
            eps_do: series(eps_do),
        },
        Variant::TwoBand => LocalHamiltonian::TwoBand {
            mu,
            u: series(u),
            j: series(j),
            delta: series(delta),
            bz: series(bz),
        },
        Variant::TwoBandKanamori => LocalHamiltonian::TwoBandKanamori {
            mu,
            u: series(u),
            j: series(j),
            delta: series(delta),
            bz: series(bz),
        },
        Variant::ThreeBand => LocalHamiltonian::ThreeBand {
            mu,
            u: series(u),
            j: series(j),
            delta: series(delta),
            bz: series(bz),
        },
        Variant::ThreeBandKanamori => LocalHamiltonian::ThreeBandKanamori {
            mu,
            u: series(u),
            j: series(j),
            delta: series(delta),
            bz: series(bz),
        },
    }
}

fn bath(entries: &[(usize, f64, f64)]) -> Vec<BathCoupling> {
    entries
        .iter()
        .map(|&(imp_mode, eps, v)| BathCoupling { imp_mode, eps, v })
        .collect()
}

fn phonon(omega: f64, lambda: f64) -> PhononSpec {
    PhononSpec {
        nph: NPH,
        omega,
        couplings: vec![PhononCoupling {
            ops: DOCC.to_vec(),
            lambda: series(lambda),
        }],
    }
}

/// Position of the quenchable parameter in `Spec::p` (`None`: the
/// phonon coupling).
fn quench_index(name: &str) -> Option<usize> {
    match name {
        "u" => Some(1),
        "j" => Some(2),
        "delta" => Some(3),
        "bz" => Some(4),
        "lambda" => None,
        _ => unreachable!("unknown quench parameter {name}"),
    }
}

fn specs() -> Vec<Spec> {
    // [mu, U, J, delta, Bz, eps_up, eps_do]
    let single = |eps_up, eps_do| [0.0, 2.0, 0.0, 0.0, 0.0, eps_up, eps_do];
    let two = |bz| [0.0, 2.0, 0.4, 0.8, bz, 0.0, 0.0];
    let three = |bz| [0.0, 2.0, 0.4, 0.8, bz, 0.0, 0.0];
    let two_bath = [(0, 0.5, 0.3), (1, -0.8, 0.35)];
    let three_bath = [(0, 0.5, 0.3), (1, -0.8, 0.35), (2, 0.9, 0.3)];
    vec![
        // tests/singleband_dimer
        Spec {
            name: "singleband",
            ntau: 96,
            variant: Variant::SingleBand,
            p: single(0.15, -0.10),
            bath: bath(&[(0, 0.5, 0.40), (1, -0.3, 0.35)]),
            phonon: None,
            quench: ("u", 2.4),
        },
        // tests/xca_twoband_dimer: density-density and full Kanamori
        Spec {
            name: "twoband_density",
            ntau: 96,
            variant: Variant::TwoBand,
            p: two(0.0),
            bath: bath(&two_bath),
            phonon: None,
            quench: ("u", 2.4),
        },
        Spec {
            name: "twoband_kanamori",
            ntau: 96,
            variant: Variant::TwoBandKanamori,
            p: two(0.0),
            bath: bath(&two_bath),
            phonon: None,
            quench: ("j", 0.6),
        },
        Spec {
            name: "twoband_kanamori_bz",
            ntau: 96,
            variant: Variant::TwoBandKanamori,
            p: two(0.3),
            bath: bath(&two_bath),
            phonon: None,
            quench: ("bz", 0.5),
        },
        // tests/threeband_dimer
        Spec {
            name: "threeband_density",
            ntau: 64,
            variant: Variant::ThreeBand,
            p: three(0.0),
            bath: bath(&three_bath),
            phonon: None,
            quench: ("delta", 1.0),
        },
        Spec {
            name: "threeband_kanamori",
            ntau: 64,
            variant: Variant::ThreeBandKanamori,
            p: three(0.0),
            bath: bath(&three_bath),
            phonon: None,
            quench: ("j", 0.6),
        },
        Spec {
            name: "threeband_kanamori_bz",
            ntau: 64,
            variant: Variant::ThreeBandKanamori,
            p: three(0.2),
            bath: bath(&three_bath),
            phonon: None,
            quench: ("bz", 0.5),
        },
        // tests/holstein_dimer
        Spec {
            name: "holstein",
            ntau: 64,
            variant: Variant::SingleBand,
            p: single(0.0, 0.0),
            bath: bath(&[(0, 0.5, 0.4), (1, 0.5, 0.4)]),
            phonon: Some((0.7, 0.4)),
            quench: ("lambda", 0.5),
        },
    ]
}

/// Quenched parameter array and phonon coupling.
fn quenched(spec: &Spec) -> ([f64; 7], Option<(f64, f64)>) {
    let mut q = spec.p;
    match quench_index(spec.quench.0) {
        Some(i) => q[i] = spec.quench.1,
        None => return (q, spec.phonon.map(|(omega, _)| (omega, spec.quench.1))),
    }
    (q, spec.phonon)
}

fn dump(spec: &Spec) {
    let name = spec.name;
    let norb = spec.variant.norb();
    let nmodes = 2 * norb;
    let hs = build(norb, SectorClass::SpinResolved);
    let ham_eq = ham(spec.variant, &spec.p);
    let (pq, phonon_q) = quenched(spec);
    let ham_q = ham(spec.variant, &pq);
    let ph_eq = spec.phonon.map(|(omega, lambda)| phonon(omega, lambda));
    let ph_q = phonon_q.map(|(omega, lambda)| phonon(omega, lambda));
    let nfbits = nmodes + spec.bath.len();
    let dim = (1usize << nfbits) * ph_eq.as_ref().map_or(1, |p| p.nph);
    let [mu, u, j, delta, bz, eps_up, eps_do] = spec.p;

    println!(
        "MODEL {name} variant={} norb={norb} beta={beta:.16e} ntau={} nbath={} dim={dim} dt={dt:.16e} nt={nt}",
        spec.variant.name(),
        spec.ntau,
        spec.bath.len(),
        beta = BETA,
        dt = DT,
        nt = NT,
    );
    println!(
        "PARAM {name} mu={mu:.16e} u={u:.16e} j={j:.16e} delta={delta:.16e} bz={bz:.16e} \
         eps_up={eps_up:.16e} eps_do={eps_do:.16e}",
    );
    println!(
        "QUENCH {name} param={} from={:.16e} to={:.16e}",
        spec.quench.0,
        match quench_index(spec.quench.0) {
            Some(i) => spec.p[i],
            None => spec.phonon.map_or(f64::NAN, |(_, lambda)| lambda),
        },
        spec.quench.1,
    );
    for b in &spec.bath {
        println!(
            "BATH {name} imp_mode={} eps={:.16e} v={:.16e}",
            b.imp_mode, b.eps, b.v
        );
    }
    if let Some((omega, lambda)) = spec.phonon {
        let ops: Vec<String> = DOCC
            .iter()
            .map(|&(m, cre)| format!("{m}:{}", u8::from(cre)))
            .collect();
        println!(
            "PHONON {name} nph={NPH} omega={omega:.16e} lambda={lambda:.16e} ops={}",
            ops.join(",")
        );
    }

    // ---- equilibrium reference (Lehmann) ----
    let ed = EdReference::build_with_phonon(
        &ham_eq,
        &hs,
        &spec.bath,
        &[],
        BETA,
        ph_eq.as_ref(),
    );
    let emin = ed
        .eigenvalues()
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    println!("Z {name} {:.16e} {emin:.16e}", ed.partition());
    let mut ev = ed.eigenvalues().to_vec();
    ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let evs: Vec<String> = ev.iter().map(|e| format!("{e:.16e}")).collect();
    println!("EV {name} {}", evs.join(" "));
    let weights: Vec<f64> = ed
        .eigenvalues()
        .iter()
        .map(|&e| (-BETA * (e - emin)).exp())
        .collect();
    let energy: f64 = ed
        .eigenvalues()
        .iter()
        .zip(&weights)
        .map(|(&e, &w)| e * w)
        .sum();
    println!("OBS {name} energy {:.16e}", energy / ed.partition());
    for mode in 0..nmodes {
        let n = ed.expectation(&[(mode, true), (mode, false)]);
        println!("OBS {name} n{mode} {:.16e}", n.re);
    }
    for a in 0..norb {
        for b in 0..norb {
            let d = ed.expectation(&[(a, true), (a, false), (norb + b, true), (norb + b, false)]);
            println!("OBS {name} docc_{a}_{b} {:.16e}", d.re);
        }
    }
    if spec.phonon.is_some() {
        println!("OBS {name} nphonon {:.16e}", ed.phonon_number());
    }
    let dtau = BETA / spec.ntau as f64;
    for mode in 0..nmodes {
        for i in 0..=spec.ntau {
            let tau = dtau * i as f64;
            println!(
                "M {name} {mode} {i} {tau:.16e} {:.16e}",
                ed.greens_tau(mode, tau)
            );
        }
    }
    for mode in 0..nmodes {
        for k in 0..=NT {
            let t = DT * k as f64;
            for kind in KINDS {
                let g = match kind {
                    "ret" => ed.greens_ret(mode, t),
                    "les" => ed.greens_les(mode, t),
                    _ => ed.greens_gtr(mode, t),
                };
                println!("RTEQ {name} {kind} {mode} {t:.16e} {:.16e} {:.16e}", g.re, g.im);
            }
        }
    }

    // ---- single quench at t = 0 (unitary evolution) ----
    let (h0, h1) = match (ph_eq.as_ref(), ph_q.as_ref()) {
        (Some(p0), Some(p1)) => (
            nqist::ed::hamiltonian_at_phonon(&ham_eq, &hs, &spec.bath, &[], 0, p0),
            nqist::ed::hamiltonian_at_phonon(&ham_q, &hs, &spec.bath, &[], 0, p1),
        ),
        _ => (
            nqist::ed::hamiltonian_at(&ham_eq, &hs, &spec.bath, &[], 0),
            nqist::ed::hamiltonian_at(&ham_q, &hs, &spec.bath, &[], 0),
        ),
    };
    let modes: Vec<usize> = (0..nmodes).collect();
    let evo = EdEvolution::build_quench_with_layout(&h0, &h1, NT, DT, BETA, &modes, nfbits);
    for mode in 0..nmodes {
        for n in 0..=NT {
            for np in 0..=n {
                for kind in KINDS {
                    let g = match kind {
                        "ret" => evo.greens_ret(mode, n, np),
                        "les" => evo.greens_les(mode, n, np),
                        _ => evo.greens_gtr(mode, n, np),
                    };
                    println!(
                        "RT {name} {kind} {mode} {n} {np} {:.16e} {:.16e}",
                        g.re, g.im
                    );
                }
            }
        }
    }
}

fn main() {
    for spec in specs() {
        dump(&spec);
    }
}
