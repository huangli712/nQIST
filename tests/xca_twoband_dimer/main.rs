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

/// Impurity energy of the spin-up occupation pattern `(n1u, n2u)` on
/// the (dn = 0) sector, taken from the nQIST Hamiltonian itself.
fn impurity_energy(ham: &TwoBandVariant, hs: &HilbertSpace, n1u: u64, n2u: u64) -> f64 {
    // full 4-bit Fock state with dn = 0; bits are spin*norb + orb
    let psi = n1u | (n2u << 1);
    let n_up = (n1u + n2u) as usize;
    let s = n_up * 3; // sector index = n_up * (norb + 1) + n_dn
    let blk = LocalHamiltonianRef::sector_block(ham, hs, s, -1);
    let idx = hs.sectors[s].states.iter().position(|&x| x == psi).unwrap();
    blk.get(idx, idx).re
}

/// Exact Green's function of orbital `orb` (spin up) from the
/// diagonalization of the 16-state {orb1, orb2, b1, b2} Fock space.
fn twoband_exact(ham: &TwoBandVariant, hs: &HilbertSpace, orb: usize, tau: f64) -> f64 {
    // basis ordering: bit0 = orb0-up, bit1 = orb1-up, bit2 = b1,
    // bit3 = b2 (c_orb acts on bit `orb` with the JW sign of the
    // bits below it)
    let mut h = DenseMat::zeros(16);
    for st in 0..16u64 {
        let n1u = st & 1;
        let n2u = (st >> 1) & 1;
        let nb1 = (st >> 2) & 1;
        let nb2 = (st >> 3) & 1;
        let e = impurity_energy(ham, hs, n1u, n2u) + EPS_B1 * nb1 as f64 + EPS_B2 * nb2 as f64;
        h.set(st as usize, st as usize, Complex::new(e, 0.0));
    }
    // impurity <-> bath hopping (same spin, per orbital)
    for st in 0..16u64 {
        for (imp_bit, bath_bit, v) in [(0u32, 2u32, V1), (1, 3, V2)] {
            let bi = 1u64 << imp_bit;
            let bb = 1u64 << bath_bit;
            if (st & bi == 0) && (st & bb != 0) {
                // move a particle bath -> impurity; JW sign from the
                // bits between the two positions
                let between = (st & (bb - 1)) >> (imp_bit + 1);
                let sign = if between.count_ones() % 2 == 1 { -1.0 } else { 1.0 };
                let tgt = (st | bi) & !bb;
                h.set(st as usize, tgt as usize, Complex::new(sign * v, 0.0));
                h.set(tgt as usize, st as usize, Complex::new(sign * v, 0.0));
            }
        }
    }
    let (w, v) = eigh(&h);

    // c_orb as a 16x16 matrix in the eigenbasis
    let mut cmat = DenseMat::zeros(16);
    for st in 0..16u64 {
        let bi = 1u64 << orb;
        if st & bi != 0 {
            let tgt = st & !bi;
            let below = st & (bi - 1);
            let sign = if below.count_ones() % 2 == 1 { -1.0 } else { 1.0 };
            cmat.set(tgt as usize, st as usize, Complex::new(sign, 0.0));
        }
    }
    let vc = v.adjoint().mat_mul(&cmat).mat_mul(&v);

    let mut z = 0.0;
    for &ei in &w {
        z += (-BETA * ei).exp();
    }
    let mut acc = 0.0;
    for i in 0..16 {
        for j in 0..16 {
            let m = vc.get(i, j);
            acc += (-BETA * w[i]).exp() * (m * m.conj()).re * (-tau * (w[j] - w[i])).exp();
        }
    }
    -acc / z
}

type TwoBandVariant = nqist::hamilton::LocalHamiltonian;
type LocalHamiltonianRef = nqist::hamilton::LocalHamiltonian;

/// Solve the two-band impurity at expansion order `order` and return
/// the Matsubara Green's functions of both orbitals.
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
    // the two-band expansion must improve from NCA to OCA and must
    // not degrade at order 3 at these moderate couplings
    assert!(errors[1][0] < errors[0][0], "{errors:?}");
    assert!(errors[1][1] < errors[0][1], "{errors:?}");
    assert!(errors[2][0] <= errors[0][0], "{errors:?}");
    assert!(errors[2][1] <= errors[0][1], "{errors:?}");
    assert!(errors[2][0] < 1.5e-1 && errors[2][1] < 1.5e-1, "{errors:?}");
}
