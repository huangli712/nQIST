// Project : nQIST
// File    : src/ed.rs
// Purpose : Reference-grade exact-diagonalization impurity reference.
//
//           Joint impurity + discrete-bath Fock space, dense
//           diagonalization, Lehmann correlators (Matsubara and
//           equilibrium real-time) and explicit unitary-evolution
//           correlators for quenched (piecewise-constant)
//           Hamiltonians. NOT a DMFT solver: no bath fitting, no
//           self-consistency loop.
//
//           Role: independent verification anchor for interaction
//           forms and regimes the C++ PPSC port does not cover
//           (the port is density-density only). Off-diagonal terms -
//           e.g. the planned rotationally-invariant Kanamori
//           spin-flip / pair-hopping strings - enter as `OpTerm`s
//           (coefficient times an ordered operator product) whose
//           Jordan-Wigner signs share the machinery of the bath
//           hybridization couplings.
//
//           Conventions: mode indices follow the occupation-bit
//           layout of `hamilton`/`hilbert`, bit `spin * norb + orb`
//           (two-band impurity: 0 = 1u, 1 = 2u, 2 = 1d, 3 = 2d).
//           Bath modes occupy the bits above the impurity modes, in
//           `bath` order. Operator strings act right-to-left;
//           Hermitian conjugates must be supplied explicitly (the
//           builder asserts Hermiticity).

use crate::hamilton::LocalHamiltonian;
use crate::hilbert::HilbertSpace;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::linalg::{DenseMat, eigh};
use std::cell::{Ref, RefCell};

/// One bath level at energy `eps` (measured relative to any chemical
/// potential already folded into the impurity Hamiltonian) coupled
/// to impurity mode `imp_mode` with amplitude `v`.
pub struct BathCoupling {
    pub imp_mode: usize,
    pub eps: f64,
    pub v: f64,
}

// `OpTerm` (the shared off-diagonal-term declaration) lives in
// `hamilton.rs`, the declaration owner; re-exported here for the ED
// call sites.
pub use crate::hamilton::OpTerm;

// ------------------------------------------------------------------
// static equilibrium reference (Lehmann)
// ------------------------------------------------------------------

/// Exact diagonalization of the joint impurity + bath space with
/// Lehmann correlators for the equilibrium reference.
pub struct EdReference {
    dim: usize,
    nmodes: usize,
    evals: Vec<f64>,
    evecs: DenseMat,
    /// Boltzmann weights `e^{-beta (E - Emin)}` (unnormalized).
    weights: Vec<f64>,
    zpart: f64,
    /// c matrices in the eigenbasis, compiled lazily per mode.
    c_eig: RefCell<Vec<Option<DenseMat>>>,
}

impl EdReference {
    /// Diagonalize impurity + bath at the equilibrium time slice
    /// (`tstp = -1`).
    pub fn build(
        ham: &LocalHamiltonian,
        hs_imp: &HilbertSpace,
        bath: &[BathCoupling],
        terms: &[OpTerm],
        beta: f64,
    ) -> EdReference {
        let h = joint_hamiltonian(ham, hs_imp, bath, terms, -1);
        let nmodes = 2 * hs_imp.norb + bath.len();
        let dim = h.size();
        let (evals, evecs) = eigh(&h);
        let emin = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        let weights: Vec<f64> = evals.iter().map(|&e| (-beta * (e - emin)).exp()).collect();
        let zpart = weights.iter().sum();
        EdReference {
            dim,
            nmodes,
            evals,
            evecs,
            weights,
            zpart,
            c_eig: RefCell::new(vec![None; nmodes]),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn nmodes(&self) -> usize {
        self.nmodes
    }

    pub fn eigenvalues(&self) -> &[f64] {
        &self.evals
    }

    /// Partition function (numerically shifted by `Emin`).
    pub fn partition(&self) -> f64 {
        self.zpart
    }

    fn c_eig(&self, mode: usize) -> Ref<'_, DenseMat> {
        if self.c_eig.borrow()[mode].is_none() {
            let cm = c_matrix(self.dim, mode);
            let ce = self.evecs.adjoint().mat_mul(&cm).mat_mul(&self.evecs);
            self.c_eig.borrow_mut()[mode] = Some(ce);
        }
        Ref::map(self.c_eig.borrow(), |v| v[mode].as_ref().unwrap())
    }

    /// Matsubara Green's function `-<c(tau) c^dag>` for `0 < tau < beta`
    /// (Lehmann; the dimer-fixture formula).
    pub fn greens_tau(&self, mode: usize, tau: f64) -> f64 {
        let c = self.c_eig(mode);
        let mut acc = 0.0f64;
        for i in 0..self.dim {
            let wi = self.weights[i];
            for j in 0..self.dim {
                let m = c.get(i, j);
                acc += wi * (m * m.conj()).re * (-tau * (self.evals[j] - self.evals[i])).exp();
            }
        }
        -acc / self.zpart
    }

    /// Equilibrium greater component `G^>(t) = -i <c(t) c^dag(0)>`
    /// with `t = t - t'`.
    pub fn greens_gtr(&self, mode: usize, t: f64) -> Complex {
        let c = self.c_eig(mode);
        let mut acc = Complex::ZERO;
        for i in 0..self.dim {
            for j in 0..self.dim {
                let m = c.get(i, j);
                let ph = (self.evals[i] - self.evals[j]) * t;
                acc += (m * m.conj())
                    * Complex::new(self.weights[i], 0.0)
                    * Complex::new(ph.cos(), ph.sin());
            }
        }
        acc * Complex::new(0.0, -1.0 / self.zpart)
    }

    /// Equilibrium lesser component `G^<(t) = +i <c^dag(0) c(t)>`.
    pub fn greens_les(&self, mode: usize, t: f64) -> Complex {
        let c = self.c_eig(mode);
        let mut acc = Complex::ZERO;
        for i in 0..self.dim {
            for j in 0..self.dim {
                let m = c.get(i, j);
                let ph = (self.evals[i] - self.evals[j]) * t;
                acc += (m * m.conj())
                    * Complex::new(self.weights[j], 0.0)
                    * Complex::new(ph.cos(), ph.sin());
            }
        }
        acc * Complex::new(0.0, 1.0 / self.zpart)
    }

    /// Equilibrium retarded component for `t > 0` (no theta):
    /// `-i <{c(t), c^dag(0)}>`.
    pub fn greens_ret(&self, mode: usize, t: f64) -> Complex {
        let c = self.c_eig(mode);
        let mut acc = Complex::ZERO;
        for i in 0..self.dim {
            for j in 0..self.dim {
                let m = c.get(i, j);
                let ph = (self.evals[i] - self.evals[j]) * t;
                acc += (m * m.conj())
                    * Complex::new(self.weights[i] + self.weights[j], 0.0)
                    * Complex::new(ph.cos(), ph.sin());
            }
        }
        acc * Complex::new(0.0, -1.0 / self.zpart)
    }

    /// Thermal expectation `<O>` of an operator string.
    pub fn expectation(&self, ops: &[(usize, bool)]) -> Complex {
        let mut om = DenseMat::zeros(self.dim);
        for st in 0..self.dim as u64 {
            if let Some((tgt, s)) = apply_ops(ops, st) {
                om.set(tgt as usize, st as usize, s);
            }
        }
        let oe = self.evecs.adjoint().mat_mul(&om).mat_mul(&self.evecs);
        let mut acc = Complex::ZERO;
        for i in 0..self.dim {
            acc += oe.get(i, i) * Complex::new(self.weights[i], 0.0);
        }
        acc / Complex::new(self.zpart, 0.0)
    }
}

// ------------------------------------------------------------------
// quenched real-time reference (unitary evolution)
// ------------------------------------------------------------------

/// Real-time correlators of a quenched impurity + bath system: the
/// thermal initial state comes from `h0`, and the propagation is
/// piecewise constant with the per-step Hamiltonians. Only the
/// requested `modes` are tracked. Memory grows as
/// `modes.len() * (nsteps + 1) * dim^2`.
pub struct EdEvolution {
    dt: f64,
    modes: Vec<usize>,
    rho0: DenseMat,
    /// Heisenberg-evolved annihilation operators `A(t_n) = U^dag c U`
    /// at `t_n = n * dt`, per tracked mode.
    a_t: Vec<Vec<DenseMat>>,
}

impl EdEvolution {
    /// General piecewise-constant protocol: `h_steps[k]` acts on
    /// `[t_k, t_{k+1})`. Consecutive identical Hamiltonians share
    /// their propagator (the quench case diagonalizes once).
    pub fn build(
        h0: &DenseMat,
        h_steps: &[DenseMat],
        dt: f64,
        beta: f64,
        modes: &[usize],
    ) -> EdEvolution {
        let mut a_t: Vec<Vec<DenseMat>> = modes
            .iter()
            .map(|&m| vec![c_matrix(h0.size(), m)])
            .collect();
        let mut u = identity(h0.size());
        let mut prop: Option<DenseMat> = None;
        for (k, hk) in h_steps.iter().enumerate() {
            let reuse = prop.is_some() && k > 0 && same_matrix(hk, &h_steps[k - 1]);
            if !reuse {
                prop = Some(spectral_propagator(hk, dt));
            }
            let p = prop.as_ref().unwrap();
            u = p.mat_mul(&u);
            let ua = u.adjoint();
            for (mi, &m) in modes.iter().enumerate() {
                let cm = c_matrix(h0.size(), m);
                a_t[mi].push(ua.mat_mul(&cm).mat_mul(&u));
            }
        }
        EdEvolution {
            dt,
            modes: modes.to_vec(),
            rho0: thermal_density(h0, beta),
            a_t,
        }
    }

    /// Single instantaneous quench `h0 -> h1` at `t = 0` with `nt`
    /// steps of width `dt` (the dimer-driver protocol).
    pub fn build_quench(
        h0: &DenseMat,
        h1: &DenseMat,
        nt: usize,
        dt: f64,
        beta: f64,
        modes: &[usize],
    ) -> EdEvolution {
        let mut a_t: Vec<Vec<DenseMat>> = modes
            .iter()
            .map(|&m| vec![c_matrix(h0.size(), m)])
            .collect();
        let mut u = identity(h0.size());
        let prop = spectral_propagator(h1, dt);
        for _ in 0..nt {
            u = prop.mat_mul(&u);
            let ua = u.adjoint();
            for (mi, &m) in modes.iter().enumerate() {
                let cm = c_matrix(h0.size(), m);
                a_t[mi].push(ua.mat_mul(&cm).mat_mul(&u));
            }
        }
        EdEvolution {
            dt,
            modes: modes.to_vec(),
            rho0: thermal_density(h0, beta),
            a_t,
        }
    }

    /// Number of stored time points (`nsteps + 1`, `t_n = n * dt`).
    pub fn ntimes(&self) -> usize {
        self.a_t.first().map(|v| v.len()).unwrap_or(0)
    }

    /// `G^>(t_n, t_np) = -i Tr[rho0 A(t_n) A^dag(t_np)]`.
    pub fn greens_gtr(&self, mode: usize, n: usize, np: usize) -> Complex {
        let k = self.mode_index(mode);
        let b = self.a_t[k][n].mat_mul(&self.a_t[k][np].adjoint());
        Complex::new(0.0, -1.0) * trace_prod(&self.rho0, &b)
    }

    /// `G^<(t_n, t_np) = +i Tr[rho0 A^dag(t_np) A(t_n)]`.
    pub fn greens_les(&self, mode: usize, n: usize, np: usize) -> Complex {
        let k = self.mode_index(mode);
        let b = self.a_t[k][np].adjoint().mat_mul(&self.a_t[k][n]);
        Complex::new(0.0, 1.0) * trace_prod(&self.rho0, &b)
    }

    /// Retarded correlator with the `t^+` convention at equal times:
    /// `-i <{A(t_n), A^dag(t_np)}>` for `n >= np`, `0` otherwise.
    pub fn greens_ret(&self, mode: usize, n: usize, np: usize) -> Complex {
        if np > n {
            return Complex::ZERO;
        }
        let k = self.mode_index(mode);
        let aa = self.a_t[k][n].mat_mul(&self.a_t[k][np].adjoint());
        let ad = self.a_t[k][np].adjoint().mat_mul(&self.a_t[k][n]);
        Complex::new(0.0, -1.0) * (trace_prod(&self.rho0, &aa) + trace_prod(&self.rho0, &ad))
    }

    fn mode_index(&self, mode: usize) -> usize {
        self.modes
            .iter()
            .position(|&m| m == mode)
            .unwrap_or_else(|| panic!("ed: mode {mode} not tracked"))
    }
}

/// Apply an operator string right-to-left on the occupation-basis
/// state `st` with the Jordan-Wigner sign `(-1)^{n_below}`.
/// `None` when any operator hits an incompatible occupation.
fn apply_ops(ops: &[(usize, bool)], mut st: u64) -> Option<(u64, Complex)> {
    let mut sign = 1.0f64;
    for &(mode, cre) in ops.iter().rev() {
        let bit = 1u64 << mode;
        let odd = (st & (bit - 1)).count_ones() % 2 == 1;
        if cre {
            if st & bit != 0 {
                return None;
            }
            if odd {
                sign = -sign;
            }
            st |= bit;
        } else {
            if st & bit == 0 {
                return None;
            }
            if odd {
                sign = -sign;
            }
            st &= !bit;
        }
    }
    Some((st, Complex::new(sign, 0.0)))
}

/// Impurity diagonal energy of the occupation pattern `psi`, taken
/// from the solver Hamiltonian itself (same route as the dimer
/// fixture, so the ED is convention-identical to the solver).
fn impurity_energy(ham: &LocalHamiltonian, hs: &HilbertSpace, psi: u64, tstp: i64) -> f64 {
    for (s, sec) in hs.sectors.iter().enumerate() {
        if let Some(idx) = sec.states.iter().position(|&x| x == psi) {
            return ham.sector_block(hs, s, tstp).get(idx, idx).re;
        }
    }
    panic!("ed: impurity pattern {psi:b} not found in any sector");
}

fn apply_term(h: &mut DenseMat, dim: usize, coeff: Complex, ops: &[(usize, bool)]) {
    for st in 0..dim as u64 {
        if let Some((tgt, s)) = apply_ops(ops, st) {
            let cur = h.get(tgt as usize, st as usize);
            h.set(tgt as usize, st as usize, cur + coeff * s);
        }
    }
}

/// Full joint-space Hamiltonian: the complete impurity sector
/// blocks via `sector_block` at time `tstp` (diagonal AND, for the
/// full-Kanamori variant, the off-diagonal spin-flip / pair-hopping
/// entries) + bath occupations + bath couplings and caller terms
/// through the operator-string machinery.
fn joint_hamiltonian(
    ham: &LocalHamiltonian,
    hs_imp: &HilbertSpace,
    bath: &[BathCoupling],
    terms: &[OpTerm],
    tstp: i64,
) -> DenseMat {
    let imp_modes = 2 * hs_imp.norb;
    let dim = 1usize << (imp_modes + bath.len());
    let mut h = DenseMat::zeros(dim);
    // scatter every impurity sector block (full matrix per sector)
    // into EVERY bath configuration subspace (H_imp tensor I_bath)
    let ncfg = 1usize << bath.len();
    for s in 0..hs_imp.ns() {
        let blk = ham.sector_block(hs_imp, s, tstp);
        for cfg in 0..ncfg {
            let off = ((cfg as u64) << imp_modes) as usize;
            for (i, &si) in hs_imp.sectors[s].states.iter().enumerate() {
                for (j, &sj) in hs_imp.sectors[s].states.iter().enumerate() {
                    h.set(si as usize + off, sj as usize + off, blk.get(i, j));
                }
            }
        }
    }
    for st in 0..dim as u64 {
        let mut e = 0.0;
        for (b, bc) in bath.iter().enumerate() {
            e += bc.eps * ((st >> (imp_modes + b)) & 1) as f64;
        }
        if e != 0.0 {
            let cur = h.get(st as usize, st as usize);
            h.set(st as usize, st as usize, cur + Complex::new(e, 0.0));
        }
    }
    for (b, bc) in bath.iter().enumerate() {
        let bb = imp_modes + b;
        apply_term(
            &mut h,
            dim,
            Complex::new(bc.v, 0.0),
            &[(bc.imp_mode, true), (bb, false)],
        );
        apply_term(
            &mut h,
            dim,
            Complex::new(bc.v, 0.0),
            &[(bb, true), (bc.imp_mode, false)],
        );
    }
    for t in terms {
        apply_term(&mut h, dim, t.coeff, &t.ops);
    }
    // Hermiticity guard: a missing h.c. term fails here.
    for i in 0..dim {
        for j in (i + 1)..dim {
            let d = h.get(i, j) - h.get(j, i).conj();
            assert!(
                (d * d.conj()).re < 1e-24,
                "ed: non-Hermitian term assembly at ({i},{j}): {d:?}"
            );
        }
    }
    h
}

/// Assemble the full joint-space Hamiltonian at time slice `tstp`
/// (solver `time_index` convention; `-1` = equilibrium slice).
pub fn hamiltonian_at(
    ham: &LocalHamiltonian,
    hs_imp: &HilbertSpace,
    bath: &[BathCoupling],
    terms: &[OpTerm],
    tstp: i64,
) -> DenseMat {
    joint_hamiltonian(ham, hs_imp, bath, terms, tstp)
}

/// Annihilation matrix of `mode` on the full space (Jordan-Wigner).
fn c_matrix(dim: usize, mode: usize) -> DenseMat {
    let mut cm = DenseMat::zeros(dim);
    let bit = 1u64 << mode;
    for st in 0..dim as u64 {
        if st & bit != 0 {
            let sign = if (st & (bit - 1)).count_ones() % 2 == 1 {
                -1.0
            } else {
                1.0
            };
            cm.set((st & !bit) as usize, st as usize, Complex::new(sign, 0.0));
        }
    }
    cm
}

fn identity(dim: usize) -> DenseMat {
    let mut m = DenseMat::zeros(dim);
    for i in 0..dim {
        m.set(i, i, Complex::new(1.0, 0.0));
    }
    m
}

/// `Tr[a b]` as a dense double sum.
fn trace_prod(a: &DenseMat, b: &DenseMat) -> Complex {
    let n = a.size();
    let mut t = Complex::ZERO;
    for i in 0..n {
        for j in 0..n {
            t += a.get(i, j) * b.get(j, i);
        }
    }
    t
}

fn same_matrix(a: &DenseMat, b: &DenseMat) -> bool {
    let n = a.size();
    let mut d2 = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let d = a.get(i, j) - b.get(i, j);
            d2 += (d * d.conj()).re;
        }
    }
    d2 < 1e-24
}

/// `exp(-i h dt)` for Hermitian `h` via its spectral decomposition.
fn spectral_propagator(h: &DenseMat, dt: f64) -> DenseMat {
    let n = h.size();
    let (e, v) = eigh(h);
    let mut ph = DenseMat::zeros(n);
    for i in 0..n {
        let x = -e[i] * dt;
        ph.set(i, i, Complex::new(x.cos(), x.sin()));
    }
    v.mat_mul(&ph).mat_mul(&v.adjoint())
}

/// Thermal density matrix `e^{-beta h0} / Z` (shifted by `Emin`).
fn thermal_density(h0: &DenseMat, beta: f64) -> DenseMat {
    let n = h0.size();
    let (e, v) = eigh(h0);
    let emin = e.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut w = DenseMat::zeros(n);
    let mut z = 0.0f64;
    for i in 0..n {
        let x = (-beta * (e[i] - emin)).exp();
        w.set(i, i, Complex::new(x, 0.0));
        z += x;
    }
    for i in 0..n {
        let cur = w.get(i, i);
        w.set(i, i, cur / Complex::new(z, 0.0));
    }
    v.mat_mul(&w).mat_mul(&v.adjoint())
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hilbert::{SectorClass, build};

    const BETA: f64 = 5.0;

    fn twoband_ham(u: f64, j: f64, delta: f64, len: usize) -> LocalHamiltonian {
        LocalHamiltonian::TwoBand {
            mu: 0.0,
            u: vec![u; len],
            j: vec![j; len],
            delta: vec![delta; len],
            bz: vec![0.0; len],
        }
    }

    fn singleband_ham(eps: f64, len: usize) -> LocalHamiltonian {
        LocalHamiltonian::SingleBand {
            mu: 0.0,
            u: vec![0.0; len],
            eps_up: vec![eps; len],
            eps_do: vec![eps; len],
        }
    }

    fn dimer_bath() -> [BathCoupling; 2] {
        [
            BathCoupling {
                imp_mode: 0,
                eps: 0.5,
                v: 0.3,
            },
            BathCoupling {
                imp_mode: 1,
                eps: -0.8,
                v: 0.35,
            },
        ]
    }

    #[test]
    fn jw_signs_match_hand_rules() {
        // c_2 |100> = |000> (nothing below mode 2)
        let (t, s) = apply_ops(&[(2, false)], 0b100).unwrap();
        assert_eq!(t, 0);
        assert!((s.re - 1.0).abs() < 1e-15 && s.im.abs() < 1e-15);
        // c_2 |101> = -|001> (one fermion below mode 2)
        let (t, s) = apply_ops(&[(2, false)], 0b101).unwrap();
        assert_eq!(t, 0b001);
        assert!((s.re + 1.0).abs() < 1e-15);
        // hop c^dag_0 c_2 |110> = -|011> (crossing the mode-1 fermion)
        let (t, s) = apply_ops(&[(0, true), (2, false)], 0b110).unwrap();
        assert_eq!(t, 0b011);
        assert!((s.re + 1.0).abs() < 1e-15);
        // Pauli blocking
        assert!(apply_ops(&[(0, true)], 0b001).is_none());
        assert!(apply_ops(&[(0, false)], 0b000).is_none());
    }

    #[test]
    fn op_term_two_level_block() {
        // inter-mode hopping t (c^dag_0 c_1 + h.c.) on a U = 0
        // single-band impurity: the {|01>, |10>} block is [[0, t],
        // [t, 0]], so the spectrum is {-t, 0, 0, +t}
        let hs = build(1, SectorClass::SpinResolved);
        let ham = singleband_ham(0.0, 2);
        let t = Complex::new(0.7, 0.0);
        let terms = [
            OpTerm {
                coeff: t,
                ops: vec![(0, true), (1, false)],
            },
            OpTerm {
                coeff: t,
                ops: vec![(1, true), (0, false)],
            },
        ];
        let ed = EdReference::build(&ham, &hs, &[], &terms, BETA);
        let mut w = ed.eigenvalues().to_vec();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (got, want) in w.iter().zip([-0.7, 0.0, 0.0, 0.7]) {
            assert!((got - want).abs() < 1e-12, "spectrum {w:?}");
        }
    }

    #[test]
    fn atomic_partition_matches_sector_eigh() {
        let hs = build(2, SectorClass::SpinResolved);
        let ham = twoband_ham(2.0, 0.4, 0.8, 2);
        let ed = EdReference::build(&ham, &hs, &[], &[], BETA);
        // sector-wise reference: eigenvalues and Boltzmann sum
        let mut evals = Vec::new();
        for s in 0..hs.ns() {
            let (w, _) = eigh(&ham.sector_block(&hs, s, -1));
            evals.extend(w);
        }
        let emin = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        let z: f64 = evals.iter().map(|&e| (-BETA * (e - emin)).exp()).sum();
        assert!((ed.partition() / z - 1.0).abs() < 1e-12);
        let mut got = ed.eigenvalues().to_vec();
        let mut want = evals;
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        want.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-10, "evals {got:?} vs {want:?}");
        }
    }

    #[test]
    fn equal_time_anticommutator_identity() {
        // canonical identity G^>(0) - G^<(0) = -i holds for every mode
        let hs = build(2, SectorClass::SpinResolved);
        let ham = twoband_ham(2.0, 0.4, 0.8, 2);
        let bath = dimer_bath();
        let ed = EdReference::build(&ham, &hs, &bath, &[], BETA);
        for m in 0..4 {
            let d = ed.greens_gtr(m, 0.0) - ed.greens_les(m, 0.0) + Complex::new(0.0, 1.0);
            assert!((d * d.conj()).re < 1e-20, "mode {m}: {d:?}");
        }
    }

    #[test]
    fn no_quench_evolution_matches_lehmann() {
        // with H(t) = h0 the explicit unitary evolution must
        // reproduce the equilibrium Lehmann correlators exactly
        // (two independent computational routes)
        let hs = build(2, SectorClass::SpinResolved);
        let ham = twoband_ham(2.0, 0.4, 0.8, 8);
        let bath = dimer_bath();
        let h = hamiltonian_at(&ham, &hs, &bath, &[], -1);
        let (nt, dt) = (40usize, 0.05);
        let ev = EdEvolution::build_quench(&h, &h, nt, dt, BETA, &[0, 1]);
        let ed = EdReference::build(&ham, &hs, &bath, &[], BETA);
        for m in [0usize, 1] {
            for (n, np) in [(5usize, 0usize), (20, 3), (40, 0), (17, 16)] {
                let t = (n - np) as f64 * dt;
                let pairs = [
                    (ev.greens_gtr(m, n, np), ed.greens_gtr(m, t)),
                    (ev.greens_les(m, n, np), ed.greens_les(m, t)),
                    (ev.greens_ret(m, n, np), ed.greens_ret(m, t)),
                ];
                for (a, b) in pairs {
                    let d = a - b;
                    assert!(
                        (d * d.conj()).re < 1e-18,
                        "mode {m} ({n},{np}): {a:?} vs {b:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn single_level_quench_closed_form() {
        // decoupled spins at eps = 0 (rho0 = 1/4 identity) quenched
        // to eps = 0.3: mode-0 correlators are the single-level
        // closed forms with <c c^dag>_0 = <c^dag c>_0 = 1/2
        let hs = build(1, SectorClass::SpinResolved);
        let h0 = hamiltonian_at(&singleband_ham(0.0, 2), &hs, &[], &[], -1);
        let h1 = hamiltonian_at(&singleband_ham(0.3, 2), &hs, &[], &[], -1);
        let (nt, dt) = (30usize, 0.07);
        let ev = EdEvolution::build_quench(&h0, &h1, nt, dt, BETA, &[0]);
        for (n, np) in [(7usize, 0usize), (30, 12), (5, 4)] {
            let d = (n - np) as f64 * dt;
            let ph = Complex::new((-0.3f64 * d).cos(), (-0.3f64 * d).sin());
            let half = Complex::new(0.5, 0.0);
            let want_gtr = Complex::new(0.0, -1.0) * ph * half;
            let want_les = Complex::new(0.0, 1.0) * ph * half;
            let want_ret = Complex::new(0.0, -1.0) * ph;
            for (got, want, tag) in [
                (ev.greens_gtr(0, n, np), want_gtr, "gtr"),
                (ev.greens_les(0, n, np), want_les, "les"),
                (ev.greens_ret(0, n, np), want_ret, "ret"),
            ] {
                let d2 = got - want;
                assert!(
                    (d2 * d2.conj()).re < 1e-24,
                    "{tag} ({n},{np}): {got:?} vs {want:?}"
                );
            }
        }
    }

    #[test]
    fn evolution_anticommutator_across_grid() {
        // unitary evolution preserves G^>(t,t) - G^<(t,t) = -i at
        // every grid point
        let hs = build(2, SectorClass::SpinResolved);
        let h0 = hamiltonian_at(&twoband_ham(2.0, 0.4, 0.8, 8), &hs, &dimer_bath(), &[], -1);
        let h1 = hamiltonian_at(&twoband_ham(3.0, 0.4, 0.8, 8), &hs, &dimer_bath(), &[], -1);
        let ev = EdEvolution::build_quench(&h0, &h1, 25, 0.06, BETA, &[0, 1, 2, 3]);
        for m in [0usize, 1, 2, 3] {
            for n in [0usize, 9, 25] {
                let d = ev.greens_gtr(m, n, n) - ev.greens_les(m, n, n) + Complex::new(0.0, 1.0);
                assert!((d * d.conj()).re < 1e-20, "mode {m} n {n}: {d:?}");
            }
        }
    }
}
