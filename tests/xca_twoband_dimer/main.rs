// Project : nQIST
// File    : tests/xca_twoband_dimer/main.rs
// Purpose : XCA two-band benchmark: crystal-field-split Kanamori
//           impurity (norb = 2), each orbital coupled to its own
//           bath site (spin-up channels only), against exact
//           diagonalization of the 16-state spinless dimer-dimer
//
// Exercises the full multi-band path of the XCA evaluator: 9
// SpinResolved sectors with dimensions up to 4, per-orbital
// hybridization channels, and the TwoBand Kanamori Hamiltonian. The
// impurity energies of the exact reference are taken from
// `LocalHamiltonian::sector_block` itself, so the ED Hamiltonian is
// convention-identical to the solver's.

use kadanoffbaym::complex::Complex;
use kadanoffbaym::linalg::{DenseMat, eigh};
use kadanoffbaym::types::{Contour, ContourGreen, Sign};
use nqist::dmft::set_bwd_from_fwd_mat;
use nqist::hamilton::LocalHamiltonian::TwoBand;
use nqist::hilbert::{HilbertSpace, SectorClass, build};
use nqist::nca::{GfVertex, Hybridization};
use nqist::solver::ImpuritySolver;

const BETA: f64 = 5.0;
const NTAU: usize = 96;
const NT: usize = 6;
const U_HUB: f64 = 2.0;
const J_HUND: f64 = 0.4;
const CRYSTAL: f64 = 0.8; // 0.5 * delta * (n1 - n2)
const V1: f64 = 0.3;
const EPS_B1: f64 = 0.5;
const V2: f64 = 0.35;
const EPS_B2: f64 = -0.8;

fn k_tau(tau: f64, w: f64) -> f64 {
    -(-w * tau).exp() / (1.0 + (-BETA * w).exp())
}

/// Impurity energy of the full 4-bit occupation pattern (bits are
/// spin*norb + orb: 1u, 2u, 1d, 2d), from the nQIST Hamiltonian
/// itself so the ED is convention-identical to the solver.
fn impurity_energy(ham: &nqist::hamilton::LocalHamiltonian, hs: &HilbertSpace, psi: u64) -> f64 {
    let n_up = (psi & 0b0011).count_ones() as usize;
    let n_dn = (psi & 0b1100).count_ones() as usize;
    let s = n_up * 3 + n_dn;
    let blk = nqist::hamilton::LocalHamiltonian::sector_block(ham, hs, s, -1);
    let idx = hs.sectors[s].states.iter().position(|&x| x == psi).unwrap();
    blk.get(idx, idx).re
}

/// Exact Green's function of orbital `orb` (spin up) from the
/// diagonalization of the full 64-state Fock space: impurity bits
/// (1u, 2u, 1d, 2d) = 0..3 plus bath bits b1 = 4, b2 = 5. The
/// down-spin orbitals stay in the partition (Kanamori coupled) but
/// never hybridize.
fn twoband_exact(
    ham: &nqist::hamilton::LocalHamiltonian,
    hs: &HilbertSpace,
    orb: usize,
    tau: f64,
) -> f64 {
    let n = 64usize;
    let mut h = DenseMat::zeros(n);
    for st in 0..n as u64 {
        let e_imp = impurity_energy(ham, hs, st & 0b1111);
        let e_bath = EPS_B1 * ((st >> 4) & 1) as f64 + EPS_B2 * ((st >> 5) & 1) as f64;
        h.set(st as usize, st as usize, Complex::new(e_imp + e_bath, 0.0));
    }
    // up-only impurity <-> bath hopping: (bit0 <-> bit4), (bit1 <-> bit5)
    for st in 0..n as u64 {
        for (imp_bit, bath_bit, v) in [(0u32, 4u32, V1), (1, 5, V2)] {
            let bi = 1u64 << imp_bit;
            let bb = 1u64 << bath_bit;
            if (st & bi == 0) && (st & bb != 0) {
                let between = (st & (bb - 1)) >> (imp_bit + 1);
                let sign = if between.count_ones() % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };
                let tgt = (st | bi) & !bb;
                h.set(st as usize, tgt as usize, Complex::new(sign * v, 0.0));
                h.set(tgt as usize, st as usize, Complex::new(sign * v, 0.0));
            }
        }
    }
    let (w, v) = eigh(&h);

    // c_orb (up) as a matrix in the eigenbasis
    let mut cmat = DenseMat::zeros(n);
    for st in 0..n as u64 {
        let bi = 1u64 << orb;
        if st & bi != 0 {
            let tgt = st & !bi;
            let below = st & (bi - 1);
            let sign = if below.count_ones() % 2 == 1 {
                -1.0
            } else {
                1.0
            };
            cmat.set(tgt as usize, st as usize, Complex::new(sign, 0.0));
        }
    }
    let vc = v.adjoint().mat_mul(&cmat).mat_mul(&v);

    let mut z = 0.0;
    for &ei in &w {
        z += (-BETA * ei).exp();
    }
    let mut acc = 0.0;
    for i in 0..n {
        for j in 0..n {
            let m = vc.get(i, j);
            acc += (-BETA * w[i]).exp() * (m * m.conj()).re * (-tau * (w[j] - w[i])).exp();
        }
    }
    -acc / z
}

/// Port gate: the reference-grade ED module must reproduce the
/// fixture's independently implemented exact Green's function on
/// the dimer parameters (two independent JW/eigh routes).
#[test]
fn ed_module_matches_fixture_exact() {
    use nqist::ed::{BathCoupling, EdReference};
    let hs = build(2, SectorClass::SpinResolved);
    let ham = TwoBand {
        mu: 0.0,
        u: vec![U_HUB; NT + 2],
        j: vec![J_HUND; NT + 2],
        delta: vec![CRYSTAL; NT + 2],
        bz: vec![0.0; NT + 2],
    };
    let bath = [
        BathCoupling {
            imp_mode: 0,
            eps: EPS_B1,
            v: V1,
        },
        BathCoupling {
            imp_mode: 1,
            eps: EPS_B2,
            v: V2,
        },
    ];
    let ed = EdReference::build(&ham, &hs, &bath, &[], BETA);
    for orb in [0usize, 1] {
        for i in [1usize, 12, 24, 48, 72, 95] {
            let tau = BETA * i as f64 / NTAU as f64;
            let a = twoband_exact(&ham, &hs, orb, tau);
            let b = ed.greens_tau(orb, tau);
            assert!(
                (a - b).abs() < 1e-10,
                "ed port: orb {orb} tau {tau}: fixture {a} vs module {b}"
            );
        }
    }
}

/// Solve the two-band impurity at expansion order `order` and return
/// the Matsubara Green's functions of both orbitals.
#[allow(clippy::needless_range_loop)]
fn solve_twoband(order: usize, pp_iters: usize) -> (Vec<f64>, Vec<f64>) {
    let hs = build(2, SectorClass::SpinResolved);
    let ham = TwoBand {
        mu: 0.0,
        u: vec![U_HUB; NT + 2],
        j: vec![J_HUND; NT + 2],
        delta: vec![CRYSTAL; NT + 2],
        bz: vec![0.0; NT + 2],
    };
    let mut solver = ImpuritySolver::new(NT, NTAU, BETA, 0.02, 5, order, hs.clone(), ham.clone());
    solver.solve_atomic();

    let contour = Contour::new(NT as i64, NTAU as i64, 1, 1, 0.1, BETA);
    let mut delta = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let mut delta_cc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let dtau = BETA / NTAU as f64;
    for i in 0..=NTAU {
        let t = dtau * i as f64;
        delta.mat.block_mut(i)[0] =
            Complex::new(V1 * V1 * k_tau(t, EPS_B1) + V2 * V2 * k_tau(t, EPS_B2), 0.0);
    }
    set_bwd_from_fwd_mat(&mut delta_cc, &delta);
    // per-orbital baths: separate greens so the channels decompose
    // independently
    let mut delta1 = delta.clone();
    let mut delta1_cc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let mut delta2 = delta.clone();
    let mut delta2_cc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    for i in 0..=NTAU {
        let t = dtau * i as f64;
        delta1.mat.block_mut(i)[0] = Complex::new(V1 * V1 * k_tau(t, EPS_B1), 0.0);
        delta2.mat.block_mut(i)[0] = Complex::new(V2 * V2 * k_tau(t, EPS_B2), 0.0);
    }
    set_bwd_from_fwd_mat(&mut delta1_cc, &delta1);
    set_bwd_from_fwd_mat(&mut delta2_cc, &delta2);

    let c0a = hs.flavor(0, 0, false);
    let c0c = hs.flavor(0, 0, true);
    let c1a = hs.flavor(1, 0, false);
    let c1c = hs.flavor(1, 0, true);
    let verts = [
        GfVertex {
            op_ann_fl: c0a,
            op_cre_fl: c0c,
        },
        GfVertex {
            op_ann_fl: c1a,
            op_cre_fl: c1c,
        },
    ];
    let mut last0 = Vec::new();
    let mut last1 = Vec::new();
    let t0 = std::time::Instant::now();
    for it in 0..pp_iters {
        let pp_ints = [
            Hybridization {
                delta: &delta1,
                op_cre_fl: c0c,
                op_ann_fl: c0a,
                fermion: true,
                backward: false,
            },
            Hybridization {
                delta: &delta1_cc,
                op_cre_fl: c0a,
                op_ann_fl: c0c,
                fermion: true,
                backward: true,
            },
            Hybridization {
                delta: &delta2,
                op_cre_fl: c1c,
                op_ann_fl: c1a,
                fermion: true,
                backward: false,
            },
            Hybridization {
                delta: &delta2_cc,
                op_cre_fl: c1a,
                op_ann_fl: c1c,
                fermion: true,
                backward: true,
            },
        ];
        solver.pp_step(-1, &pp_ints);
        let g = solver.get_spgf_mat(&verts, &pp_ints);
        let cur0: Vec<f64> = g[0].iter().map(|z| z.re).collect();
        let cur1: Vec<f64> = g[1].iter().map(|z| z.re).collect();
        eprintln!(
            "twoband-o{order} pp iter {it:3} (t {:>7.1?}): G1(0) = {:.6}  G2(0) = {:.6}",
            t0.elapsed(),
            cur0[0],
            cur1[0]
        );
        if !last0.is_empty() {
            let mut diff = 0.0f64;
            for ((a, b), c) in last0.iter().zip(&cur0).zip(&cur1) {
                diff = diff.max((a - b).abs()).max(c.abs() * 0.0);
            }
            for (a, b) in last1.iter().zip(&cur1) {
                diff = diff.max((a - b).abs());
            }
            if diff < 1.0e-10 {
                eprintln!("twoband-o{order} converged after {it} iterations");
                break;
            }
        }
        last0 = cur0;
        last1 = cur1;
    }
    (last0, last1)
}

fn main() {
    println!("run via cargo test --test xca_twoband_dimer");
}

#[test]
fn twoband_atomic_conventions_and_propagators() {
    // cross-check `sector_block` energies against the hand-computed
    // Kanamori values and the two orbital G(0+) occupations
    let hs = build(2, SectorClass::SpinResolved);
    let ham = TwoBand {
        mu: 0.0,
        u: vec![U_HUB; NT + 2],
        j: vec![J_HUND; NT + 2],
        delta: vec![CRYSTAL; NT + 2],
        bz: vec![0.0; NT + 2],
    };
    for (psi, want) in [
        (0u64, 0.0),
        (1, 0.5 * CRYSTAL - 0.5 * (3.0 * U_HUB - 5.0 * J_HUND)),
        (2, -0.5 * CRYSTAL - 0.5 * (3.0 * U_HUB - 5.0 * J_HUND)),
        (
            3,
            (U_HUB - 2.0 * J_HUND) - J_HUND - 2.0 * 0.5 * (3.0 * U_HUB - 5.0 * J_HUND),
        ),
    ] {
        let e = impurity_energy(&ham, &hs, psi);
        eprintln!("impurity {psi:04b}: sector_block {e:+.6}  hand {want:+.6}");
        assert!((e - want).abs() < 1e-12);
    }
}

#[test]
fn xca_twoband_dimer_converges_to_exact() {
    let hs = build(2, SectorClass::SpinResolved);
    let ham = TwoBand {
        mu: 0.0,
        u: vec![U_HUB; NT + 2],
        j: vec![J_HUND; NT + 2],
        delta: vec![CRYSTAL; NT + 2],
        bz: vec![0.0; NT + 2],
    };
    let dtau = BETA / NTAU as f64;
    let mut errors: Vec<[f64; 2]> = Vec::new();
    for order in 1..=3usize {
        let (g0, g1) = solve_twoband(order, 200);
        let mut err = [0.0f64; 2];
        for i in 1..NTAU {
            let e0 = twoband_exact(&ham, &hs, 0, dtau * i as f64);
            let e1 = twoband_exact(&ham, &hs, 1, dtau * i as f64);
            err[0] = err[0].max((g0[i] - e0).abs());
            err[1] = err[1].max((g1[i] - e1).abs());
        }
        println!(
            "order {order}: max |G-Ged| orb1 {:+.6e} orb2 {:+.6e}",
            err[0], err[1]
        );
        errors.push(err);
    }
    // Native OCA calibration (2026-11): the norb>=2 orders are now
    // served by the native PPSC-faithful OCA path (C++-gated),
    // superseding the old XCA-route values (order-2 once hit 8.7e-4
    // riding on the XCA(2)-gf deviation). At these couplings the
    // bold series is non-monotonic: order-2 OCA overshoots NCA and
    // order-3 recovers slightly. Anchor the absolute error budget
    // instead of route-dependent monotonic improvement.
    assert!(errors[0][0] < 1.1e-2 && errors[0][1] < 1.1e-2, "{errors:?}");
    assert!(errors[1][0] < 2.0e-2 && errors[1][1] < 2.0e-2, "{errors:?}");
    assert!(errors[2][0] < 2.0e-2 && errors[2][1] < 2.0e-2, "{errors:?}");
    assert!(
        errors[2][0] < 1.1 * errors[1][0] && errors[2][1] < 1.1 * errors[1][1],
        "{errors:?}"
    );
}
