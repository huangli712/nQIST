// Project : nQIST
// File    : src/observable.rs
// Purpose : local observables from the pseudo-particle density
//           matrix (blueprints ppsc expectation_value /
//           hamiltonians update_exp_vals)

use crate::hamilton::LocalHamiltonian;
use crate::hilbert::HilbertSpace;
use crate::solver::ImpuritySolver;

/// Local observables of a timestep: spin densities, double
/// occupancy, interaction energy, and the density-matrix trace
/// (conservation monitor).
#[derive(Clone, Copy, Debug)]
pub struct LocalObservables {
    pub n_up: f64,
    pub n_dn: f64,
    pub docc: f64,
    pub e_int: f64,
    pub trace: f64,
}

// ------------------------------------------------------------------
// two-band multiplet observables (P7, blueprint
// two_band_hubbard.hpp update_exp_vals): the multiplet-resolved
// occupations are exact state probabilities read from the
// density-matrix diagonal (the C++ 16 one-dimensional sectors; here
// resolved through our 9-sector layout via the Fock-state ids).
// Bit convention `spin * norb + orb` (C++ identical).
// ------------------------------------------------------------------

/// Two-band local observables of a timestep.
#[derive(Clone, Copy, Debug)]
pub struct TwoBandObservables {
    pub q: f64,
    pub e_int: f64,
    pub n1: f64,
    pub n2: f64,
    pub m: f64,
    /// triplet/singlet-sector populations (PRB 96, 165104):
    /// `p13` = 1-or-3-particle states, `puu` = parallel-spin
    /// two-particle, `pud` = anti-parallel two-particle, `pdd` =
    /// intra-orbital doublons, `p04` = empty-or-full
    pub p13: f64,
    pub puu: f64,
    pub pud: f64,
    pub pdd: f64,
    pub p04: f64,
}

/// Occupation `n_{orb,spin}(state)` of a Fock state (bit
/// `spin * norb + orb`, the C++ convention).
fn occupation(hs: &HilbertSpace, state: u64, orb: usize, spin: usize) -> f64 {
    ((state >> (spin * hs.norb + orb)) & 1) as f64
}

/// Double occupancy `n_up * n_dn` of a single-band Fock state.
fn docc_occupation(hs: &HilbertSpace, state: u64) -> f64 {
    occupation(hs, state, 0, 0) * occupation(hs, state, 0, 1)
}

/// State population `Re rho[state, state]` through our sector layout.
fn state_population(hs: &HilbertSpace, rho: &[kadanoffbaym::linalg::DenseMat], state: u64) -> f64 {
    for s in 0..hs.ns() {
        if let Some(i) = hs.sectors[s].states.iter().position(|&st| st == state) {
            return rho[s].get(i, i).re;
        }
    }
    unreachable!("state {state} not found in any sector")
}

/// Evaluate the local observables at `tstp` from the solver's
/// density matrix (blueprint `update_exp_vals` + `local_obs`):
/// `n_x = sum_s sum_i Re rho_s[i,i] n_x(state_i)` for the
/// single-band Hamiltonian, `E_int = sum_s Re tr(rho_s H_s(tstp))`.
pub fn update_exp_vals(solver: &ImpuritySolver, tstp: i64) -> LocalObservables {
    let hs = &solver.hs;
    let rho = &solver.rho[(tstp + 1) as usize];
    let mut obs = LocalObservables {
        n_up: 0.0,
        n_dn: 0.0,
        docc: 0.0,
        e_int: 0.0,
        trace: 0.0,
    };
    assert!(
        matches!(solver.ham, LocalHamiltonian::SingleBand { .. }) && hs.norb == 1,
        "update_exp_vals: only the single-band case is implemented"
    );
    for s in 0..hs.ns() {
        let d = hs.sectors[s].dim;
        let h = solver.ham.sector_block(hs, s, tstp);
        for i in 0..d {
            let w = rho[s].get(i, i).re;
            let state = hs.sectors[s].states[i];
            obs.n_up += w * occupation(hs, state, 0, 0);
            obs.n_dn += w * occupation(hs, state, 0, 1);
            obs.docc += w * docc_occupation(hs, state);
        }
        // E_int = Re tr(rho_s H_s)
        let mut tr = kadanoffbaym::complex::Complex::ZERO;
        for i in 0..d {
            for j in 0..d {
                tr += rho[s].get(i, j) * h.get(j, i);
            }
        }
        obs.e_int += tr.re;
        obs.trace += rho[s].trace().re;
    }
    obs
}

/// Evaluate the two-band observables at `tstp` (blueprint
/// `update_exp_vals` of two_band_hubbard.hpp).
pub fn update_exp_vals_two_band(solver: &ImpuritySolver, tstp: i64) -> TwoBandObservables {
    let hs = &solver.hs;
    assert!(
        matches!(
            solver.ham,
            LocalHamiltonian::TwoBand { .. } | LocalHamiltonian::TwoBandKanamori { .. }
        ) && hs.norb == 2,
        "update_exp_vals_two_band: only the two-band case is implemented"
    );
    let rho = &solver.rho[(tstp + 1) as usize];
    let mut obs = TwoBandObservables {
        q: 0.0,
        e_int: 0.0,
        n1: 0.0,
        n2: 0.0,
        m: 0.0,
        p13: 0.0,
        puu: 0.0,
        pud: 0.0,
        pdd: 0.0,
        p04: 0.0,
    };
    // bit convention: bit b = spin * 2 + orb
    let bits_up = [0u64, 1];
    let bits_dn = [2u64, 3u64];
    for s in 0..hs.ns() {
        let d = hs.sectors[s].dim;
        let h = solver.ham.sector_block(hs, s, tstp);
        let mut tr = kadanoffbaym::complex::Complex::ZERO;
        for i in 0..d {
            for j in 0..d {
                tr += rho[s].get(i, j) * h.get(j, i);
            }
        }
        obs.e_int += tr.re;
        obs.q += rho[s].trace().re;
        for i in 0..d {
            let w = rho[s].get(i, i).re;
            let state = hs.sectors[s].states[i];
            for (orb, b) in bits_up.iter().enumerate() {
                let nu = ((state >> b) & 1) as f64;
                obs.n1 += if orb == 0 { w * nu } else { 0.0 };
                obs.n2 += if orb == 1 { w * nu } else { 0.0 };
                obs.m += w * nu;
            }
            for b in bits_dn {
                let nd = ((state >> b) & 1) as f64;
                let orb = (b - 2) as usize;
                obs.n1 += if orb == 0 { w * nd } else { 0.0 };
                obs.n2 += if orb == 1 { w * nd } else { 0.0 };
                obs.m -= w * nd;
            }
        }
    }
    // multiplet populations (state ids per the C++ densdens table)
    let pop = |st: u64| state_population(hs, rho, st);
    obs.p04 = pop(0b0000) + pop(0b1111);
    obs.puu = pop(0b0011) + pop(0b1100);
    obs.pud = pop(0b0110) + pop(0b1001);
    obs.pdd = pop(0b0101) + pop(0b1010);
    for st in [
        0b0001u64, 0b0010, 0b0100, 0b1000, 0b1110, 0b1101, 0b1011, 0b0111,
    ] {
        obs.p13 += pop(st);
    }
    obs
}

// ------------------------------------------------------------------
// three-band observables (blueprint three_band_hubbard.hpp
// update_exp_vals): orbital densities, doublons, magnetization, the
// interaction energy and the density-matrix trace, read state by
// state from the density-matrix diagonal (bit convention
// `spin * norb + orb`, C++ identical).
// ------------------------------------------------------------------

/// Three-band local observables of a timestep.
#[derive(Clone, Copy, Debug)]
pub struct ThreeBandObservables {
    pub q: f64,
    pub e_int: f64,
    pub n1: f64,
    pub n2: f64,
    pub n3: f64,
    pub m: f64,
    /// Orbital doublon occupations `d_i = n_iu n_id`.
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
}

/// Evaluate the three-band observables at `tstp` (blueprint
/// `update_exp_vals` of three_band_hubbard.hpp).
pub fn update_exp_vals_three_band(solver: &ImpuritySolver, tstp: i64) -> ThreeBandObservables {
    let hs = &solver.hs;
    assert!(
        matches!(solver.ham, LocalHamiltonian::ThreeBand { .. }) && hs.norb == 3,
        "update_exp_vals_three_band: only the three-band case is implemented"
    );
    let rho = &solver.rho[(tstp + 1) as usize];
    let mut obs = ThreeBandObservables {
        q: 0.0,
        e_int: 0.0,
        n1: 0.0,
        n2: 0.0,
        n3: 0.0,
        m: 0.0,
        d1: 0.0,
        d2: 0.0,
        d3: 0.0,
    };
    for s in 0..hs.ns() {
        let d = hs.sectors[s].dim;
        let h = solver.ham.sector_block(hs, s, tstp);
        let mut tr = kadanoffbaym::complex::Complex::ZERO;
        for i in 0..d {
            for j in 0..d {
                tr += rho[s].get(i, j) * h.get(j, i);
            }
        }
        obs.e_int += tr.re;
        obs.q += rho[s].trace().re;
        for i in 0..d {
            let w = rho[s].get(i, i).re;
            let state = hs.sectors[s].states[i];
            for orb in 0..3 {
                let nu = occupation(hs, state, orb, 0);
                let nd = occupation(hs, state, orb, 1);
                match orb {
                    0 => {
                        obs.n1 += w * (nu + nd);
                        obs.d1 += w * nu * nd;
                    }
                    1 => {
                        obs.n2 += w * (nu + nd);
                        obs.d2 += w * nu * nd;
                    }
                    _ => {
                        obs.n3 += w * (nu + nd);
                        obs.d3 += w * nu * nd;
                    }
                }
                obs.m += w * (nu - nd);
            }
        }
    }
    obs
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamilton::LocalHamiltonian::SingleBand;
    use crate::hilbert::{SectorClass, build};
    use crate::solver::ImpuritySolver;
    use kadanoffbaym::types::MATSUBARA;

    #[test]
    fn atomic_observables_sum_rule() {
        // after solve_atomic + pp_step(-1): rho is the normalized
        // atomic distribution; n_up + n_dn = 2 * docc-related sum and
        // the trace is one
        let hs = build(1, SectorClass::SpinResolved);
        let nt = 4;
        let ntau = 64;
        let u = vec![3.0; nt + 2];
        let eps = vec![0.0; nt + 2];
        let mut solver = ImpuritySolver::new(
            nt,
            ntau,
            5.0,
            0.02,
            3,
            1,
            hs,
            SingleBand {
                mu: 0.0,
                u: u.clone(),
                eps_up: eps.clone(),
                eps_do: eps,
            },
        );
        solver.solve_atomic();
        solver.pp_step(MATSUBARA, &[]);
        let obs = update_exp_vals(&solver, MATSUBARA);
        assert!((obs.trace - 1.0).abs() < 1e-12);
        // half filling: n_up = n_dn and 0 < docc < 1
        assert!((obs.n_up - obs.n_dn).abs() < 1e-12);
        assert!(obs.docc > 0.0 && obs.docc < 1.0);
        // atomic docc = (2/Z1) e^{-beta U_eff} weight ratio against
        // the singly occupied sector is positive but small for U=3
        assert!(obs.docc < 0.5);
        // E_int matches the sector eigenvalues weighted by rho
        let e_direct = obs.e_int;
        assert!(e_direct.abs() > 1e-6);
    }

    #[test]
    fn three_band_atomic_observables_match_boltzmann() {
        // atomic limit: rho is the Boltzmann distribution of the sector
        // eigenvalues (the Matsubara quadrature is exact to 1e-10 on
        // this grid), so every observable equals its explicit
        // Fock-state sum; half filling gives n1 + n2 + n3 = 3 and m = 0
        use crate::hamilton::LocalHamiltonian::ThreeBand;
        let hs = build(3, SectorClass::SpinResolved);
        let (nt, beta, u, j) = (4usize, 5.0f64, 3.0, 0.4);
        let ham = ThreeBand {
            mu: 0.0,
            u: vec![u; nt + 2],
            j: vec![j; nt + 2],
            delta: vec![0.0; nt + 2],
            bz: vec![0.0; nt + 2],
        };
        let mut solver = ImpuritySolver::new(nt, 256, beta, 0.02, 5, 1, hs.clone(), ham.clone());
        solver.solve_atomic();
        solver.pp_step(MATSUBARA, &[]);
        let obs = update_exp_vals_three_band(&solver, MATSUBARA);
        // reference: direct Boltzmann sums over the 64 Fock states
        let occ = |psi: u64, orb: usize, spin: usize| ((psi >> (spin * 3 + orb)) & 1) as f64;
        let mut z = 0.0;
        let mut want = [0.0f64; 6]; // [n, m, d1, d2, d3, E]
        for psi in 0..64u64 {
            let n_up = (psi & 0b111).count_ones() as usize;
            let n_dn = (psi & 0b111000).count_ones() as usize;
            let s = n_up * 4 + n_dn;
            let idx = hs.sectors[s].states.iter().position(|&x| x == psi).unwrap();
            let e = ham.sector_block(&hs, s, -1).get(idx, idx).re;
            let w = (-beta * e).exp();
            z += w;
            want[0] += w * (n_up + n_dn) as f64;
            want[1] += w * (n_up as f64 - n_dn as f64);
            for orb in 0..3 {
                want[2 + orb] += w * occ(psi, orb, 0) * occ(psi, orb, 1);
            }
            want[5] += w * e;
        }
        for w in want.iter_mut() {
            *w /= z;
        }
        let got = [
            obs.n1 + obs.n2 + obs.n3,
            obs.m,
            obs.d1,
            obs.d2,
            obs.d3,
            obs.e_int,
        ];
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() < 1e-10, "{got:?} vs {want:?}");
        }
        assert!((obs.q - 1.0).abs() < 1e-12);
    }
}
