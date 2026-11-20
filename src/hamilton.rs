// Project : nQIST
// File    : src/hamilton.rs
// Purpose : sector-blocked local Hamiltonians with time-dependent
//           parameters (blueprints ppsc/hamiltonians/{single_band,
//           two_band}_hubbard.hpp)

use crate::hilbert::HilbertSpace;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::linalg::DenseMat;

/// One off-diagonal Hamiltonian term: `coeff` times the ordered
/// operator product `ops`, applied right-to-left; `(mode, true)` is
/// a creation operator, mode convention `spin * norb + orb`. This is
/// the single declaration layer shared by the solver-side compiler
/// (`sector_block` operator chains) and the ED-side compiler (full
/// Fock-space string application) - the two meet at the Tier 0a
/// per-element equality gate. Hermitian conjugates must be supplied
/// explicitly.
#[derive(Clone, Debug)]
pub struct OpTerm {
    pub coeff: Complex,
    pub ops: Vec<(usize, bool)>,
}

/// Local impurity Hamiltonian, diagonal in the Fock basis except
/// the full-Kanamori variant (off-diagonal spin-flip / pair-hopping
/// terms). The time-dependent parameter arrays follow the PPSC
/// convention: index `tstp + 1` (so index 0 is the
/// equilibrium/Matsubara value, `U[0]` is `U0`), each of length
/// `nt + 2`.
#[derive(Clone, Debug)]
pub enum LocalHamiltonian {
    /// Single-band Hubbard in the particle-hole symmetric form
    /// `H = U(t) n_up n_dn + (eps_up(t) - mu - U(t)/2) n_up
    ///      + (eps_do(t) - mu - U(t)/2) n_dn`
    /// (requires `norb = 1`).
    SingleBand {
        /// Chemical potential.
        mu: f64,
        /// `U(tstp + 1)`.
        u: Vec<f64>,
        /// `eps_up(tstp + 1)`.
        eps_up: Vec<f64>,
        /// `eps_do(tstp + 1)`.
        eps_do: Vec<f64>,
    },
    /// Two-band Kanamori density-density Hamiltonian
    /// `H = Bz(t) m + 0.5 delta(t) (n1 - n2) - mu n
    ///      - 0.5 (3U - 5J) n + U (n1u n1d + n2u n2d)
    ///      + (U - 2J) n1 n2 - J (n1u n2u + n1d n2d)`
    /// with the magnetization `m = n1u - n1d + n2u - n2d` and the
    /// half-filling shift `0.5 (3U - 5J) n` (requires `norb = 2`).
    /// Effective pair interactions: same orbital `U`, inter-orbital
    /// opposite spin `U - 2J`, inter-orbital parallel spin `U - 3J`.
    TwoBand {
        /// Chemical potential.
        mu: f64,
        /// `U(tstp + 1)`.
        u: Vec<f64>,
        /// `J(tstp + 1)` (Hund's coupling).
        j: Vec<f64>,
        /// Level splitting `delta(tstp + 1)`.
        delta: Vec<f64>,
        /// Magnetic field `Bz(tstp + 1)`.
        bz: Vec<f64>,
    },
    /// Full rotationally-invariant two-band Kanamori Hamiltonian:
    /// the density-density part of [`LocalHamiltonian::TwoBand`]
    /// plus the spin-flip and pair-hopping strings of
    /// [`kanamori_offdiag_terms`] (same parameter meaning and
    /// `tstp + 1` time convention; requires `norb = 2` and
    /// `SpinResolved` sectors, which the off-diagonal strings
    /// preserve). Validated for NCA/OCA and the ED reference; the
    /// XCA backbone paths are untested with this variant. Pass to
    /// the ED module without extra `terms` (the off-diagonal part
    /// is already inside `sector_block`).
    TwoBandKanamori {
        /// Chemical potential.
        mu: f64,
        /// `U(tstp + 1)`.
        u: Vec<f64>,
        /// `J(tstp + 1)` (Hund's coupling).
        j: Vec<f64>,
        /// Level splitting `delta(tstp + 1)`.
        delta: Vec<f64>,
        /// Magnetic field `Bz(tstp + 1)`.
        bz: Vec<f64>,
    },
}

impl LocalHamiltonian {
    /// Dense block `H_s(tstp)` of the Hamiltonian in the Fock basis of
    /// sector `s` (diagonal for these density-density Hamiltonians).
    pub fn sector_block(&self, hs: &HilbertSpace, s: usize, tstp: i64) -> DenseMat {
        assert!(s < hs.ns(), "sector_block: sector out of range");
        match self {
            LocalHamiltonian::SingleBand {
                mu,
                u,
                eps_up,
                eps_do,
            } => {
                assert_eq!(hs.norb, 1, "SingleBand Hamiltonian needs norb = 1");
                let i = time_index(tstp, u.len(), "U");
                let iu = time_index(tstp, eps_up.len(), "eps_up");
                let id = time_index(tstp, eps_do.len(), "eps_do");
                let dim = hs.sectors[s].dim;
                let mut h = DenseMat::zeros(dim);
                for (col, &psi) in hs.sectors[s].states.iter().enumerate() {
                    let n_up = occ(psi, 0); // bit = spin*norb + orb
                    let n_dn = occ(psi, 1);
                    let e = u[i] * n_up * n_dn
                        + (eps_up[iu] - mu - u[i] * 0.5) * n_up
                        + (eps_do[id] - mu - u[i] * 0.5) * n_dn;
                    h.set(col, col, Complex::new(e, 0.0));
                }
                h
            }
            LocalHamiltonian::TwoBand {
                mu,
                u,
                j,
                delta,
                bz,
            } => {
                assert_eq!(hs.norb, 2, "TwoBand Hamiltonian needs norb = 2");
                let iu = time_index(tstp, u.len(), "U");
                let ij = time_index(tstp, j.len(), "J");
                let idl = time_index(tstp, delta.len(), "delta");
                let ib = time_index(tstp, bz.len(), "Bz");
                twoband_diagonal_block(hs, s, *mu, u[iu], j[ij], delta[idl], bz[ib])
            }
            LocalHamiltonian::TwoBandKanamori {
                mu,
                u,
                j,
                delta,
                bz,
            } => {
                assert_eq!(hs.norb, 2, "TwoBandKanamori Hamiltonian needs norb = 2");
                let iu = time_index(tstp, u.len(), "U");
                let ij = time_index(tstp, j.len(), "J");
                let idl = time_index(tstp, delta.len(), "delta");
                let ib = time_index(tstp, bz.len(), "Bz");
                let mut h = twoband_diagonal_block(hs, s, *mu, u[iu], j[ij], delta[idl], bz[ib]);
                for term in kanamori_offdiag_terms(j[ij]) {
                    add_term_via_cop(&mut h, hs, s, &term);
                }
                h
            }
        }
    }
}

/// Map `tstp` (with -1 = Matsubara) to the time-array index and
/// check the length.
fn time_index(tstp: i64, len: usize, name: &str) -> usize {
    let i = (tstp + 1) as usize;
    assert!(tstp >= -1, "{name}: tstp must be >= -1");
    assert!(i < len, "{name}: index {i} out of range (len {len})");
    i
}

/// Occupation bit of `psi` at position `b`.
fn occ(psi: u64, b: usize) -> f64 {
    ((psi >> b) & 1) as f64
}

/// Density-diagonal block shared by the two-band Hamiltonian
/// variants (verbatim fill of the former `TwoBand` branch).
fn twoband_diagonal_block(
    hs: &HilbertSpace,
    s: usize,
    mu: f64,
    ut: f64,
    jt: f64,
    del: f64,
    bzt: f64,
) -> DenseMat {
    let half_filling_shift = 0.5 * (3.0 * ut - 5.0 * jt);
    let dim = hs.sectors[s].dim;
    let mut h = DenseMat::zeros(dim);
    for (col, &psi) in hs.sectors[s].states.iter().enumerate() {
        // bits: A=0, B=1 per spin; up block low, dn block high
        let n1u = occ(psi, 0);
        let n2u = occ(psi, 1);
        let n1d = occ(psi, 2);
        let n2d = occ(psi, 3);
        let n1 = n1u + n1d;
        let n2 = n2u + n2d;
        let n = n1 + n2;
        let m = n1u - n1d + n2u - n2d;
        let e = bzt * m + 0.5 * del * (n1 - n2) - mu * n - half_filling_shift * n
            + ut * (n1u * n1d + n2u * n2d)
            + (ut - 2.0 * jt) * (n1 * n2)
            - jt * (n1u * n2u + n1d * n2d);
        h.set(col, col, Complex::new(e, 0.0));
    }
    h
}

/// Compile one operator-string term into the sector-`s` block as a
/// chain product of the sector c/cdag operator blocks
/// (`HilbertSpace::ops`) - the solver-side compiler, mechanically
/// independent of the ED-side per-state string application in
/// `ed.rs` (Tier 0a gates the two against each other). Terms whose
/// chain leaves the sector contribute nothing.
fn add_term_via_cop(h: &mut DenseMat, hs: &HilbertSpace, s: usize, term: &OpTerm) {
    let dim = hs.sectors[s].dim;
    let mut m = DenseMat::zeros(dim);
    for i in 0..dim {
        m.set(i, i, Complex::new(1.0, 0.0));
    }
    let mut cur = s;
    for &(mode, cre) in term.ops.iter().rev() {
        let orb = mode % hs.norb;
        let spin = mode / hs.norb;
        let Some((tsec, blk)) = hs.ops[hs.flavor(orb, spin, cre)].block(cur) else {
            return;
        };
        m = blk.mat_mul(&m);
        cur = *tsec;
    }
    if cur == s {
        for i in 0..dim {
            for j in 0..dim {
                let v = h.get(i, j) + term.coeff * m.get(i, j);
                h.set(i, j, v);
            }
        }
    }
}

/// Off-diagonal terms of the rotationally-invariant two-band
/// Kanamori interaction: the spin-flip and pair-hopping strings with
/// their Hermitian conjugates (coefficients `-J` / `+J`; the
/// fermionic signs are fixed entirely by the operator construction
/// and gated by the N = 2 singlet/triplet closed forms). Mode
/// convention `spin * norb + orb`: 0 = 1u, 1 = 2u, 2 = 1d, 3 = 2d.
/// Together with the density-diagonal fill this completes the full
/// rotationally-invariant Kanamori interaction; no diagonal
/// re-derivation is needed (the diagonal already realizes
/// U / U - 2J / U - 3J).
pub fn kanamori_offdiag_terms(j: f64) -> Vec<OpTerm> {
    let mj = Complex::new(-j, 0.0);
    let pj = Complex::new(j, 0.0);
    vec![
        // spin-flip: -J (c^dag_{1u} c_{1d}) (c^dag_{2d} c_{2u}) + h.c.
        OpTerm {
            coeff: mj,
            ops: vec![(0, true), (2, false), (3, true), (1, false)],
        },
        OpTerm {
            coeff: mj,
            ops: vec![(1, true), (3, false), (2, true), (0, false)],
        },
        // pair-hopping: +J c^dag_{1u} c^dag_{1d} c_{2d} c_{2u} + h.c.
        OpTerm {
            coeff: pj,
            ops: vec![(0, true), (2, true), (3, false), (1, false)],
        },
        OpTerm {
            coeff: pj,
            ops: vec![(1, true), (3, true), (2, false), (0, false)],
        },
    ]
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hilbert::{SectorClass, build};
    use kadanoffbaym::linalg::eigh;

    /// Constant time series of length `nt + 2` (index `tstp + 1`).
    fn series(v: f64, nt: usize) -> Vec<f64> {
        vec![v; nt + 2]
    }

    fn diag(h: &DenseMat) -> Vec<f64> {
        (0..h.size()).map(|i| h.get(i, i).re).collect()
    }

    /// Element-wise comparison within a tolerance (floating-point
    /// accumulation order differs between the check and the builder).
    fn assert_close(left: &[f64], right: &[f64], msg: &str) {
        assert_eq!(left.len(), right.len(), "{msg}: length");
        for (a, b) in left.iter().zip(right) {
            assert!((a - b).abs() < 1e-12, "{msg}: {a} vs {b}");
        }
    }

    #[test]
    fn single_band_benchmark_diagonal() {
        // benchmark equilibrium parameters: U0 = 3, mu = 0, eps = 0,
        // beta = 5. Sector energies (particle-hole symmetric form):
        // E(0) = 0, E(1) = -U/2, E(2) = 0 — the values printed by the
        // C++ reference run.log.
        let hs = build(1, SectorClass::SpinResolved);
        let ham = LocalHamiltonian::SingleBand {
            mu: 0.0,
            u: series(3.0, 3),
            eps_up: series(0.0, 3),
            eps_do: series(0.0, 3),
        };
        let e0 = diag(&ham.sector_block(&hs, 0, -1));
        let e1 = diag(&ham.sector_block(&hs, 1, -1));
        let e2 = diag(&ham.sector_block(&hs, 2, -1));
        let e3 = diag(&ham.sector_block(&hs, 3, -1));
        assert_eq!(e0, vec![0.0]);
        assert_eq!(e1, vec![-1.5]);
        assert_eq!(e2, vec![-1.5]);
        assert_eq!(e3, vec![0.0]);
    }

    #[test]
    fn single_band_time_dependent_hand_check() {
        // U quench 3.0 -> 3.5 at tstp = 0, asymmetric level and mu:
        // E(1, up) = eps_up - mu - U/2, E(1, dn) = eps_do - mu - U/2,
        // E(2) = U + (eps_up - mu - U/2) + (eps_do - mu - U/2)
        let hs = build(1, SectorClass::SpinResolved);
        let mut u = series(3.0, 3);
        u[1] = 3.5; // tstp = 0
        let ham = LocalHamiltonian::SingleBand {
            mu: 0.1,
            u,
            eps_up: series(0.25, 3),
            eps_do: series(-0.15, 3),
        };
        // equilibrium (tstp = -1, U = 3.0)
        assert_eq!(diag(&ham.sector_block(&hs, 2, -1)), vec![0.25 - 0.1 - 1.5]);
        assert_eq!(diag(&ham.sector_block(&hs, 1, -1)), vec![-0.15 - 0.1 - 1.5]);
        assert_eq!(
            diag(&ham.sector_block(&hs, 3, -1)),
            vec![3.0 + (0.25 - 0.1 - 1.5) + (-0.15 - 0.1 - 1.5)]
        );
        // after the quench (tstp = 0, U = 3.5)
        assert_eq!(diag(&ham.sector_block(&hs, 2, 0)), vec![0.25 - 0.1 - 1.75]);
        assert_eq!(
            diag(&ham.sector_block(&hs, 3, 0)),
            vec![3.5 + (0.25 - 0.1 - 1.75) + (-0.15 - 0.1 - 1.75)]
        );
    }

    #[test]
    fn single_band_atomic_pp_mu_matches_cpp() {
        // atomic-limit pseudo-particle chemical potential
        // pp_mu = emin - ln(Z)/beta, Z = sum over ALL atomic
        // eigenvalues of exp(-beta (E - emin))
        // (blueprint zeroth_order, ppsc.cpp). Benchmark: U0 = 3,
        // mu = 0, eps = 0, beta = 5 -> eigenvalues {0, -1.5, -1.5, 0}
        // -> pp_mu = -1.5 - ln(2 + 2 e^{-7.5})/5 = -1.63874...
        // (first printed line of the C++ run.log).
        let beta = 5.0;
        let hs = build(1, SectorClass::SpinResolved);
        let ham = LocalHamiltonian::SingleBand {
            mu: 0.0,
            u: series(3.0, 3),
            eps_up: series(0.0, 3),
            eps_do: series(0.0, 3),
        };
        let mut evals: Vec<f64> = Vec::new();
        for s in 0..hs.ns() {
            let (w, _v) = eigh(&ham.sector_block(&hs, s, -1));
            evals.extend(w);
        }
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(evals, vec![-1.5, -1.5, 0.0, 0.0]);

        let emin = evals[0];
        let z: f64 = evals.iter().map(|&e| (-beta * (e - emin)).exp()).sum();
        let pp_mu = emin - z.ln() / beta;

        // analytic value of the same sum (two states at 0, two at -1.5)
        let analytic = -1.5 - (2.0 + 2.0 * (-7.5f64).exp()).ln() / 5.0;
        assert!((pp_mu - analytic).abs() < 1e-12);
        // C++ first print "pp_dens, pp_mu = 2.00111, -1.63874"
        // (6 significant digits -> tolerance 5e-6)
        assert!((pp_mu - (-1.63874)).abs() < 5.0e-6);
        assert!((z - 2.00111).abs() < 5.0e-6);
    }

    #[test]
    fn two_band_kanamori_spectrum() {
        // U = 3, J = 0.4, mu = delta = Bz = 0:
        // half-filling shift s = 0.5 (3U - 5J) = 3.5
        // E(0) = 0, E(N=1) = E(N=3) = -s, E(N=4) = 0 (p-h symmetry),
        // pair energies E(2P config) - 2 E(1P):
        //   same-orbital U = 3, inter-orbital opp. spin U - 2J = 2.2,
        //   inter-orbital parallel U - 3J = 1.8  (prb96 Fig.1 pattern)
        let hs = build(2, SectorClass::SpinResolved);
        let ham = LocalHamiltonian::TwoBand {
            mu: 0.0,
            u: series(3.0, 3),
            j: series(0.4, 3),
            delta: series(0.0, 3),
            bz: series(0.0, 3),
        };
        let s = 0.5 * (3.0 * 3.0 - 5.0 * 0.4);

        // empty and full sectors (particle-hole symmetry puts the
        // fully-filled state at the empty-state energy, up to rounding)
        assert_close(&diag(&ham.sector_block(&hs, 0, -1)), &[0.0], "sector 0");
        assert_close(&diag(&ham.sector_block(&hs, 8, -1)), &[0.0], "sector 8");

        // one-particle sectors 1 (0,1): [A-, B-], 3 (1,0): [A+, B+]
        assert_close(&diag(&ham.sector_block(&hs, 1, -1)), &[-s, -s], "sector 1");
        assert_close(&diag(&ham.sector_block(&hs, 3, -1)), &[-s, -s], "sector 3");

        // three-particle sectors 5 (1,2): [1101, 1110], 7 (2,1): [0111, 1011]
        assert_close(&diag(&ham.sector_block(&hs, 5, -1)), &[-s, -s], "sector 5");
        assert_close(&diag(&ham.sector_block(&hs, 7, -1)), &[-s, -s], "sector 7");

        // two-particle parallel-spin sectors 2 (0,2): [A-B-],
        // 6 (2,0): [A+B+]: pair energy U - 3J
        let e_par = -2.0 * s + (3.0 - 3.0 * 0.4);
        assert_close(&diag(&ham.sector_block(&hs, 2, -1)), &[e_par], "sector 2");
        assert_close(&diag(&ham.sector_block(&hs, 6, -1)), &[e_par], "sector 6");

        // (1,1) sector 4, states [0101, 0110, 1001, 1010] =
        // [A+A-, B+A-, A+B-, B+B-]: same-orbital U, inter-orbital U-2J
        let e_same = -2.0 * s + 3.0;
        let e_opp = -2.0 * s + (3.0 - 2.0 * 0.4);
        assert_close(
            &diag(&ham.sector_block(&hs, 4, -1)),
            &[e_same, e_opp, e_opp, e_same],
            "sector 4",
        );

        // extracted pair energies match the prb96 Fig.1 multiplet
        assert_eq!(e_same - 2.0 * (-s), 3.0); // E_[up dn, 0] = U
        assert_eq!(e_opp - 2.0 * (-s), 3.0 - 2.0 * 0.4); // E_[up, dn] = U - 2J
        assert_eq!(e_par - 2.0 * (-s), 3.0 - 3.0 * 0.4); // E_high = U - 3J
    }

    #[test]
    fn two_band_delta_and_field() {
        // delta = 0.6 splits the orbitals by 0.5*delta*(n1 - n2),
        // Bz = 0.7 couples to the magnetization m
        let hs = build(2, SectorClass::SpinResolved);
        let ham = LocalHamiltonian::TwoBand {
            mu: 0.0,
            u: series(3.0, 3),
            j: series(0.4, 3),
            delta: series(0.6, 3),
            bz: series(0.7, 3),
        };
        let s = 0.5 * (3.0 * 3.0 - 5.0 * 0.4);
        // sector 3 (1,0): states [A+ (0001), B+ (0010)]
        assert_close(
            &diag(&ham.sector_block(&hs, 3, -1)),
            &[-s + 0.3 + 0.7, -s - 0.3 + 0.7],
            "sector 3",
        );
        // sector 1 (0,1): states [A- (0100), B- (1000)]
        assert_close(
            &diag(&ham.sector_block(&hs, 1, -1)),
            &[-s + 0.3 - 0.7, -s - 0.3 - 0.7],
            "sector 1",
        );
    }

    #[test]
    fn kanamori_tier0a_blocks_match_ed_compiler() {
        // Tier 0a: the solver-side compiler (sector c/cdag chain
        // products inside sector_block) must equal the ED-side
        // full-space string application (ed::hamiltonian_at with the
        // density-density variant plus the explicit terms) element
        // by element, on every sector and both time slices.
        let hs = build(2, SectorClass::SpinResolved);
        let mut u = series(3.0, 3);
        u[1] = 3.4; // tstp = 0 quench exercises the per-slice terms
        let mut j = series(0.4, 3);
        j[1] = 0.5;
        let hamk = LocalHamiltonian::TwoBandKanamori {
            mu: 0.0,
            u: u.clone(),
            j: j.clone(),
            delta: series(0.8, 3),
            bz: series(0.3, 3),
        };
        let hamd = LocalHamiltonian::TwoBand {
            mu: 0.0,
            u,
            j,
            delta: series(0.8, 3),
            bz: series(0.3, 3),
        };
        for tstp in [-1i64, 0] {
            let jt = if tstp < 0 { 0.4 } else { 0.5 };
            let terms = kanamori_offdiag_terms(jt);
            let full = crate::ed::hamiltonian_at(&hamd, &hs, &[], &terms, tstp);
            for s in 0..hs.ns() {
                let blk = hamk.sector_block(&hs, s, tstp);
                let d = hs.sectors[s].dim;
                for i in 0..d {
                    for jj in 0..d {
                        let a = blk.get(i, jj);
                        let b = full.get(
                            hs.sectors[s].states[i] as usize,
                            hs.sectors[s].states[jj] as usize,
                        );
                        let dz = a - b;
                        assert!(
                            (dz * dz.conj()).re < 1e-24,
                            "tstp {tstp} sector {s} ({i},{jj}): {a:?} vs {b:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn kanamori_tier0b_multiplet_closed_forms() {
        // Tier 0b (no hand-derived signs involved): with U = 3,
        // J = 0.4 (shift s = 3.5, delta = Bz = 0) the N = 2 multiplet
        // must be the textbook rotationally-invariant pattern:
        // inter-orbital singlet U - J, triplet U - 3J degenerate with
        // the M_S = +-1 parallel-spin sectors (that degeneracy is
        // the rotational-invariance signature; density-density only
        // would give the diagonal {-4.8, -4.8, -4.4, -4.4} with no
        // triplet), intra-orbital triplet U - J and singlet U + J.
        let hs = build(2, SectorClass::SpinResolved);
        let ham = LocalHamiltonian::TwoBandKanamori {
            mu: 0.0,
            u: series(3.0, 3),
            j: series(0.4, 3),
            delta: series(0.0, 3),
            bz: series(0.0, 3),
        };
        let s = 0.5 * (3.0 * 3.0 - 5.0 * 0.4);
        let e_t = -2.0 * s + (3.0 - 3.0 * 0.4); // S = 1, U - 3J
        let e_uj = -2.0 * s + (3.0 - 0.4); // U - J (S = 0 inter, S = 1 intra)
        let e_s = -2.0 * s + (3.0 + 0.4); // S = 0 intra, U + J
        let mut w4: Vec<f64> = {
            let (w, _) = eigh(&ham.sector_block(&hs, 4, -1));
            w
        };
        w4.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let want4 = [e_t, e_uj, e_uj, e_s];
        for (a, b) in w4.iter().zip(want4) {
            assert!(
                (a - b).abs() < 1e-10,
                "sector 4 spectrum {w4:?} vs {want4:?}"
            );
        }
        // M_S = +-1 sectors degenerate with the S = 1 triplet
        assert_close(&diag(&ham.sector_block(&hs, 2, -1)), &[e_t], "sector 2");
        assert_close(&diag(&ham.sector_block(&hs, 6, -1)), &[e_t], "sector 6");
        // PH consistency of the half-filling shift at delta = Bz = 0
        // (the off-diagonal strings live in the N = 2 sector only)
        assert_close(&diag(&ham.sector_block(&hs, 0, -1)), &[0.0], "sector 0");
        assert_close(&diag(&ham.sector_block(&hs, 8, -1)), &[0.0], "sector 8");
        let mut w1 = diag(&ham.sector_block(&hs, 1, -1));
        let mut w3 = diag(&ham.sector_block(&hs, 5, -1));
        w1.extend(diag(&ham.sector_block(&hs, 3, -1)));
        w3.extend(diag(&ham.sector_block(&hs, 7, -1)));
        w1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        w3.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_close(&w1, &[-s, -s, -s, -s], "N = 1");
        assert_close(&w3, &[-s, -s, -s, -s], "N = 3");
    }
}
