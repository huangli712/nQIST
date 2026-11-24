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
//
//           Phonon mode (Holstein): an optional truncated Einstein
//           oscillator can be added to the joint space
//           (`PhononSpec`): `H_ph = omega b^dag b + sum_k
//           lambda_k(t) F_k (b + b^dag)`. The composite basis index
//           packs the fermion occupation bits below the phonon
//           number, so the fermion-only layout is the `nph = 1`
//           special case.

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

/// One electron-phonon coupling term of the joint space:
/// `lambda(tstp + 1) * (fermionic string) (b + b^dag)`, with the
/// string in the `OpTerm` mode convention (Holstein: the doublon
/// `n_up n_dn`).
#[derive(Clone, Debug)]
pub struct PhononCoupling {
    pub ops: Vec<(usize, bool)>,
    /// `lambda(tstp + 1)` (length `nt + 2`, like the Hamiltonian arrays).
    pub lambda: Vec<f64>,
}

/// A truncated Einstein phonon mode added to the joint impurity +
/// bath space: `H_ph = omega b^dag b + sum_k lambda_k(t) F_k (b +
/// b^dag)` with `nph` retained number states `0..nph-1`.
#[derive(Clone, Debug)]
pub struct PhononSpec {
    /// Number of retained phonon levels.
    pub nph: usize,
    /// Phonon frequency.
    pub omega: f64,
    /// Electron-phonon coupling terms.
    pub couplings: Vec<PhononCoupling>,
}

/// Composite joint-space layout: the fermion occupation bits in the
/// low `nfbits` bits, the phonon number above them (`nph = 1` is the
/// fermion-only layout).
#[derive(Clone, Copy)]
struct Layout {
    nfbits: usize,
    nph: usize,
}

impl Layout {
    fn new(nfbits: usize, nph: usize) -> Layout {
        assert!(nph >= 1, "ed: at least one phonon level is required");
        Layout { nfbits, nph }
    }

    fn dim(&self) -> usize {
        (1usize << self.nfbits) * self.nph
    }

    fn split(&self, idx: usize) -> (usize, usize) {
        (idx & ((1usize << self.nfbits) - 1), idx >> self.nfbits)
    }

    fn join(&self, f: usize, p: usize) -> usize {
        f | (p << self.nfbits)
    }
}

/// Fermionic bit count of a fermion-only joint space of dimension `dim`.
fn nfbits_of_pow2(dim: usize) -> usize {
    assert!(dim.is_power_of_two(), "ed: {dim} is not a fermion-only dimension");
    dim.trailing_zeros() as usize
}

// ------------------------------------------------------------------
// static equilibrium reference (Lehmann)
// ------------------------------------------------------------------

/// Exact diagonalization of the joint impurity + bath space with
/// Lehmann correlators for the equilibrium reference.
pub struct EdReference {
    dim: usize,
    nmodes: usize,
    /// Joint-space layout (fermion bits + optional phonon levels).
    lay: Layout,
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
        Self::build_with_phonon(ham, hs_imp, bath, terms, beta, None)
    }

    /// Diagonalize impurity + bath + phonon mode at the equilibrium
    /// time slice (`tstp = -1`); `phonon = None` is [`EdReference::build`].
    pub fn build_with_phonon(
        ham: &LocalHamiltonian,
        hs_imp: &HilbertSpace,
        bath: &[BathCoupling],
        terms: &[OpTerm],
        beta: f64,
        phonon: Option<&PhononSpec>,
    ) -> EdReference {
        let h = joint_hamiltonian(ham, hs_imp, bath, terms, -1, phonon);
        let nmodes = 2 * hs_imp.norb + bath.len();
        let dim = h.size();
        let (evals, evecs) = eigh(&h);
        let emin = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        let weights: Vec<f64> = evals.iter().map(|&e| (-beta * (e - emin)).exp()).collect();
        let zpart = weights.iter().sum();
        EdReference {
            dim,
            nmodes,
            lay: Layout::new(nmodes, phonon.map_or(1, |p| p.nph)),
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
            let cm = c_matrix(&self.lay, mode);
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

    /// Equilibrium spectral function `A(ω) = -(1/π) Im G^R(ω)` with a
    /// Lorentzian broadening `eta`, per flavor `mode`.
    pub fn spectral_function(&self, mode: usize, omega: &[f64], eta: f64) -> Vec<f64> {
        assert!(eta > 0.0, "spectral_function: eta must be positive");
        let c = self.c_eig(mode);
        let mut a = vec![0.0f64; omega.len()];
        let pref = 1.0 / (std::f64::consts::PI * self.zpart);
        for i in 0..self.dim {
            for j in 0..self.dim {
                let m = c.get(i, j);
                let m2 = (m * m.conj()).re;
                if m2 == 0.0 {
                    continue;
                }
                let w = self.weights[i] + self.weights[j];
                let e = self.evals[j] - self.evals[i];
                for (k, &om) in omega.iter().enumerate() {
                    let d = om - e;
                    a[k] += pref * m2 * w * eta / (d * d + eta * eta);
                }
            }
        }
        a
    }

    /// Thermal expectation `<O>` of an operator string.
    pub fn expectation(&self, ops: &[(usize, bool)]) -> Complex {
        let mut om = DenseMat::zeros(self.dim);
        for idx in 0..self.dim {
            let (f, p) = self.lay.split(idx);
            if let Some((tgt, s)) = apply_ops(ops, f as u64) {
                om.set(self.lay.join(tgt as usize, p), idx, s);
            }
        }
        let oe = self.evecs.adjoint().mat_mul(&om).mat_mul(&self.evecs);
        let mut acc = Complex::ZERO;
        for i in 0..self.dim {
            acc += oe.get(i, i) * Complex::new(self.weights[i], 0.0);
        }
        acc / Complex::new(self.zpart, 0.0)
    }

    /// Thermal phonon number `<b^dag b>` (zero without a phonon mode).
    pub fn phonon_number(&self) -> f64 {
        let mut acc = 0.0f64;
        for i in 0..self.dim {
            let mut n = 0.0f64;
            for idx in 0..self.dim {
                let a = self.evecs.get(idx, i);
                n += self.lay.split(idx).1 as f64 * (a * a.conj()).re;
            }
            acc += self.weights[i] * n;
        }
        acc / self.zpart
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
        let nfbits = nfbits_of_pow2(h0.size());
        Self::build_with_layout(h0, h_steps, dt, beta, modes, nfbits)
    }

    /// [`EdEvolution::build`] on an explicitly laid out joint space:
    /// `nfbits` is the number of fermionic bits (the phonon levels
    /// are `h0.size() / 2^nfbits`).
    pub fn build_with_layout(
        h0: &DenseMat,
        h_steps: &[DenseMat],
        dt: f64,
        beta: f64,
        modes: &[usize],
        nfbits: usize,
    ) -> EdEvolution {
        let lay = Layout::new(nfbits, h0.size() >> nfbits);
        assert_eq!(lay.dim(), h0.size(), "ed: layout does not match the Hamiltonian size");
        let mut a_t: Vec<Vec<DenseMat>> = modes
            .iter()
            .map(|&m| vec![c_matrix(&lay, m)])
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
                let cm = c_matrix(&lay, m);
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
        let nfbits = nfbits_of_pow2(h0.size());
        Self::build_quench_with_layout(h0, h1, nt, dt, beta, modes, nfbits)
    }

    /// [`EdEvolution::build_quench`] on an explicitly laid out joint
    /// space (see [`EdEvolution::build_with_layout`]).
    pub fn build_quench_with_layout(
        h0: &DenseMat,
        h1: &DenseMat,
        nt: usize,
        dt: f64,
        beta: f64,
        modes: &[usize],
        nfbits: usize,
    ) -> EdEvolution {
        let lay = Layout::new(nfbits, h0.size() >> nfbits);
        assert_eq!(lay.dim(), h0.size(), "ed: layout does not match the Hamiltonian size");
        let mut a_t: Vec<Vec<DenseMat>> = modes
            .iter()
            .map(|&m| vec![c_matrix(&lay, m)])
            .collect();
        let mut u = identity(h0.size());
        let prop = spectral_propagator(h1, dt);
        for _ in 0..nt {
            u = prop.mat_mul(&u);
            let ua = u.adjoint();
            for (mi, &m) in modes.iter().enumerate() {
                let cm = c_matrix(&lay, m);
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

fn apply_term(h: &mut DenseMat, lay: &Layout, coeff: Complex, ops: &[(usize, bool)]) {
    for idx in 0..lay.dim() {
        let (f, p) = lay.split(idx);
        if let Some((tgt, s)) = apply_ops(ops, f as u64) {
            let row = lay.join(tgt as usize, p);
            let cur = h.get(row, idx);
            h.set(row, idx, cur + coeff * s);
        }
    }
}

/// Full joint-space Hamiltonian: the complete impurity sector
/// blocks via `sector_block` at time `tstp` (diagonal AND, for the
/// full-Kanamori variant, the off-diagonal spin-flip / pair-hopping
/// entries) + bath occupations + bath couplings and caller terms
/// through the operator-string machinery, plus the optional phonon
/// mode (`omega b^dag b` and the e-ph terms).
fn joint_hamiltonian(
    ham: &LocalHamiltonian,
    hs_imp: &HilbertSpace,
    bath: &[BathCoupling],
    terms: &[OpTerm],
    tstp: i64,
    phonon: Option<&PhononSpec>,
) -> DenseMat {
    let imp_modes = 2 * hs_imp.norb;
    let lay = Layout::new(imp_modes + bath.len(), phonon.map_or(1, |p| p.nph));
    let dim = lay.dim();
    let mut h = DenseMat::zeros(dim);
    // scatter every impurity sector block (full matrix per sector)
    // into EVERY bath configuration and phonon number subspace
    // (H_imp tensor I_bath tensor I_ph)
    let ncfg = 1usize << bath.len();
    for s in 0..hs_imp.ns() {
        let blk = ham.sector_block(hs_imp, s, tstp);
        for cfg in 0..ncfg {
            let off = ((cfg as u64) << imp_modes) as usize;
            for p in 0..lay.nph {
                for (i, &si) in hs_imp.sectors[s].states.iter().enumerate() {
                    for (j, &sj) in hs_imp.sectors[s].states.iter().enumerate() {
                        h.set(
                            lay.join(si as usize + off, p),
                            lay.join(sj as usize + off, p),
                            blk.get(i, j),
                        );
                    }
                }
            }
        }
    }
    for idx in 0..dim {
        let (f, p) = lay.split(idx);
        let mut e = match phonon {
            Some(ph) => ph.omega * p as f64,
            None => 0.0,
        };
        for (b, bc) in bath.iter().enumerate() {
            e += bc.eps * ((f >> (imp_modes + b)) & 1) as f64;
        }
        if e != 0.0 {
            let cur = h.get(idx, idx);
            h.set(idx, idx, cur + Complex::new(e, 0.0));
        }
    }
    for (b, bc) in bath.iter().enumerate() {
        let bb = imp_modes + b;
        apply_term(
            &mut h,
            &lay,
            Complex::new(bc.v, 0.0),
            &[(bc.imp_mode, true), (bb, false)],
        );
        apply_term(
            &mut h,
            &lay,
            Complex::new(bc.v, 0.0),
            &[(bb, true), (bc.imp_mode, false)],
        );
    }
    for t in terms {
        apply_term(&mut h, &lay, t.coeff, &t.ops);
    }
    // e-ph terms: lambda(tstp) * F (b + b^dag), F applied to the
    // fermion part (Jordan-Wigner signs), b / b^dag to the phonon number
    if let Some(ph) = phonon {
        let i = (tstp + 1) as usize;
        for cpl in &ph.couplings {
            assert!(
                tstp >= -1 && i < cpl.lambda.len(),
                "ed: phonon coupling index {i} out of range (len {})",
                cpl.lambda.len()
            );
            let lam = cpl.lambda[i];
            for idx in 0..dim {
                let (f, p) = lay.split(idx);
                let Some((tf, s)) = apply_ops(&cpl.ops, f as u64) else {
                    continue;
                };
                let amp = lam * s.re;
                if p >= 1 {
                    let row = lay.join(tf as usize, p - 1);
                    let cur = h.get(row, idx);
                    h.set(row, idx, cur + Complex::new(amp * (p as f64).sqrt(), 0.0));
                }
                if p + 1 < lay.nph {
                    let row = lay.join(tf as usize, p + 1);
                    let cur = h.get(row, idx);
                    h.set(
                        row,
                        idx,
                        cur + Complex::new(amp * ((p + 1) as f64).sqrt(), 0.0),
                    );
                }
            }
        }
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
    joint_hamiltonian(ham, hs_imp, bath, terms, tstp, None)
}

/// [`hamiltonian_at`] with the phonon mode of `phonon` added to the
/// joint space (dimension `2^(2 norb + nbath) * nph`).
pub fn hamiltonian_at_phonon(
    ham: &LocalHamiltonian,
    hs_imp: &HilbertSpace,
    bath: &[BathCoupling],
    terms: &[OpTerm],
    tstp: i64,
    phonon: &PhononSpec,
) -> DenseMat {
    joint_hamiltonian(ham, hs_imp, bath, terms, tstp, Some(phonon))
}

/// Annihilation matrix of `mode` on the joint space (Jordan-Wigner
/// on the fermion part, identity on the phonon levels).
fn c_matrix(lay: &Layout, mode: usize) -> DenseMat {
    let dim = lay.dim();
    let mut cm = DenseMat::zeros(dim);
    let bit = 1u64 << mode;
    for idx in 0..dim {
        let (f, p) = lay.split(idx);
        let st = f as u64;
        if st & bit != 0 {
            let sign = if (st & (bit - 1)).count_ones() % 2 == 1 {
                -1.0
            } else {
                1.0
            };
            cm.set(lay.join((st & !bit) as usize, p), idx, Complex::new(sign, 0.0));
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

    // ------------------------------------------------------------------
    // phonon mode
    // ------------------------------------------------------------------

    /// Doublon string `n_up n_dn` (the Holstein vertex).
    const DOCC: [(usize, bool); 4] = [(0, true), (0, false), (1, true), (1, false)];

    #[test]
    fn phonon_unit_layout_reproduces_fermion_only() {
        // a single phonon level without couplings is the fermion-only
        // space: both construction routes must agree bit for bit
        let hs = build(2, SectorClass::SpinResolved);
        let ham = twoband_ham(2.0, 0.4, 0.8, 2);
        let ham_q = twoband_ham(3.0, 0.4, 0.8, 2);
        let bath = dimer_bath();
        let phonon = PhononSpec {
            nph: 1,
            omega: 0.9,
            couplings: Vec::new(),
        };
        let a = EdReference::build(&ham, &hs, &bath, &[], BETA);
        let b = EdReference::build_with_phonon(&ham, &hs, &bath, &[], BETA, Some(&phonon));
        assert_eq!(a.dim(), b.dim());
        assert!((a.partition() / b.partition() - 1.0).abs() < 1e-12);
        for m in 0..4 {
            for i in [1usize, 12, 24, 48, 72, 95] {
                let tau = BETA * i as f64 / 96.0;
                assert!((a.greens_tau(m, tau) - b.greens_tau(m, tau)).abs() < 1e-12);
            }
        }
        let n_up = [(0usize, true), (0, false)];
        let d = a.expectation(&n_up) - b.expectation(&n_up);
        assert!((d * d.conj()).re < 1e-24);
        assert_eq!(b.phonon_number(), 0.0);
        let ha = hamiltonian_at(&ham, &hs, &bath, &[], -1);
        let hb = hamiltonian_at_phonon(&ham, &hs, &bath, &[], -1, &phonon);
        for i in 0..64usize {
            for j in 0..64usize {
                let d = ha.get(i, j) - hb.get(i, j);
                assert!((d * d.conj()).re < 1e-28);
            }
        }
        let h1a = hamiltonian_at(&ham_q, &hs, &bath, &[], 0);
        let h1b = hamiltonian_at_phonon(&ham_q, &hs, &bath, &[], 0, &phonon);
        let ev_a = EdEvolution::build_quench(&ha, &h1a, 10, 0.05, BETA, &[0, 1]);
        let ev_b =
            EdEvolution::build_quench_with_layout(&hb, &h1b, 10, 0.05, BETA, &[0, 1], 6);
        for (n, np) in [(5usize, 0usize), (10, 3), (10, 0)] {
            for (x, y) in [
                (ev_a.greens_gtr(0, n, np), ev_b.greens_gtr(0, n, np)),
                (ev_a.greens_les(1, np, n), ev_b.greens_les(1, np, n)),
                (ev_a.greens_ret(1, n, np), ev_b.greens_ret(1, n, np)),
            ] {
                let d = x - y;
                assert!((d * d.conj()).re < 1e-28);
            }
        }
    }

    #[test]
    fn displaced_oscillator_spectrum() {
        // independent-boson closed form: H = eps n_up + omega b^dag b
        // + lambda n_up (b + b^dag) has the empty ladder m omega and
        // the displaced ladder eps - lambda^2/omega + m omega
        let hs = build(1, SectorClass::SpinResolved);
        let (eps, omega, lam, nph) = (0.3f64, 0.8, 0.5, 24usize);
        let ham = LocalHamiltonian::SingleBand {
            mu: 0.0,
            u: vec![0.0; 2],
            eps_up: vec![eps; 2],
            eps_do: vec![50.0; 2], // the down level is pushed away
        };
        let phonon = PhononSpec {
            nph,
            omega,
            couplings: vec![PhononCoupling {
                ops: vec![(0, true), (0, false)],
                lambda: vec![lam; 2],
            }],
        };
        let ed = EdReference::build_with_phonon(&ham, &hs, &[], &[], BETA, Some(&phonon));
        assert_eq!(ed.dim(), 4 * nph);
        let mut got = ed.eigenvalues().to_vec();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut want: Vec<f64> = (0..nph)
            .flat_map(|m| [m as f64 * omega, eps - lam * lam / omega + m as f64 * omega])
            .collect();
        want.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // the low half of the spectrum is truncation-converged
        for (a, b) in got.iter().zip(&want).take(nph / 2) {
            assert!((a - b).abs() < 1e-9, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn displaced_oscillator_thermal_phonon_number() {
        // doublon-projector coupling: the displaced thermal oscillator
        // on the doublon has <n> = n_B + (lambda/omega)^2, every other
        // state the free thermal value n_B, hence
        // <n_ph> = n_B(omega) + P_d (lambda/omega)^2 with P_d the
        // thermal doublon probability of the atomic impurity with the
        // polaron-shifted doublon energy E_d = -lambda^2/omega
        // (particle-hole form: E_0 = E_d^0 = 0, E_s = -U/2)
        let hs = build(1, SectorClass::SpinResolved);
        let (u, omega, lam, nph) = (3.0f64, 0.8, 0.5, 24usize);
        let ham = LocalHamiltonian::SingleBand {
            mu: 0.0,
            u: vec![u; 2],
            eps_up: vec![0.0; 2],
            eps_do: vec![0.0; 2],
        };
        let phonon = PhononSpec {
            nph,
            omega,
            couplings: vec![PhononCoupling {
                ops: DOCC.to_vec(),
                lambda: vec![lam; 2],
            }],
        };
        let ed = EdReference::build_with_phonon(&ham, &hs, &[], &[], BETA, Some(&phonon));
        let e_d = -lam * lam / omega;
        let z = 1.0 + 2.0 * (BETA * u * 0.5).exp() + (-BETA * e_d).exp();
        let p_d = (-BETA * e_d).exp() / z;
        let n_b = 1.0 / ((BETA * omega).exp() - 1.0);
        let want = n_b + p_d * (lam / omega).powi(2);
        let got = ed.phonon_number();
        println!("thermal phonon number: {got:.12} vs closed form {want:.12}");
        assert!((got - want).abs() < 1e-8, "{got} vs {want}");
    }

    #[test]
    fn phonon_truncation_convergence() {
        // doublon-coupled phonon: G_up(tau) must converge with the
        // number of retained phonon levels
        let hs = build(1, SectorClass::SpinResolved);
        let (omega, lam) = (0.8f64, 0.5);
        let ham = LocalHamiltonian::SingleBand {
            mu: 0.0,
            u: vec![3.0; 2],
            eps_up: vec![0.0; 2],
            eps_do: vec![0.0; 2],
        };
        let g = |nph: usize| -> Vec<f64> {
            let phonon = PhononSpec {
                nph,
                omega,
                couplings: vec![PhononCoupling {
                    ops: DOCC.to_vec(),
                    lambda: vec![lam; 2],
                }],
            };
            let ed = EdReference::build_with_phonon(&ham, &hs, &[], &[], BETA, Some(&phonon));
            (1..=20)
                .map(|i| ed.greens_tau(0, BETA * i as f64 / 20.0))
                .collect()
        };
        let (g6, g9, g12) = (g(6), g(9), g(12));
        let dist = |a: &[f64], b: &[f64]| {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
        };
        let (d69, d912) = (dist(&g6, &g9), dist(&g9, &g12));
        println!("phonon truncation: |G6 - G9| {d69:.3e}, |G9 - G12| {d912:.3e}");
        assert!(d912 < 1e-2 * d69 && d912 < 1e-9, "{d69:.3e} -> {d912:.3e}");
    }

    #[test]
    fn no_quench_evolution_with_phonon_matches_lehmann() {
        // with a phonon-coupled impurity and H(t) = h0 the explicit
        // unitary evolution must reproduce the equilibrium Lehmann
        // correlators exactly
        let hs = build(1, SectorClass::SpinResolved);
        let (omega, lam, nph) = (0.8f64, 0.5, 12usize);
        let ham = LocalHamiltonian::SingleBand {
            mu: 0.0,
            u: vec![3.0; 22],
            eps_up: vec![0.0; 22],
            eps_do: vec![0.0; 22],
        };
        let phonon = PhononSpec {
            nph,
            omega,
            couplings: vec![PhononCoupling {
                ops: DOCC.to_vec(),
                lambda: vec![lam; 22],
            }],
        };
        let h = hamiltonian_at_phonon(&ham, &hs, &[], &[], -1, &phonon);
        let ev = EdEvolution::build_quench_with_layout(&h, &h, 20, 0.05, BETA, &[0, 1], 2);
        let ed = EdReference::build_with_phonon(&ham, &hs, &[], &[], BETA, Some(&phonon));
        for m in [0usize, 1] {
            for (n, np) in [(5usize, 0usize), (20, 3), (20, 0), (17, 16)] {
                let t = (n - np) as f64 * 0.05;
                for (a, b) in [
                    (ev.greens_gtr(m, n, np), ed.greens_gtr(m, t)),
                    (ev.greens_les(m, n, np), ed.greens_les(m, t)),
                    (ev.greens_ret(m, n, np), ed.greens_ret(m, t)),
                ] {
                    let d = a - b;
                    assert!((d * d.conj()).re < 1e-18, "mode {m} ({n},{np}): {a:?} vs {b:?}");
                }
            }
        }
    }

    #[test]
    fn evolution_anticommutator_with_phonon_quench() {
        // a lambda quench 0.5 -> 0.8 with the phonon: the canonical
        // identity G^>(t,t) - G^<(t,t) = -i survives the evolution
        let hs = build(1, SectorClass::SpinResolved);
        let mut lambda = vec![0.8f64; 22];
        lambda[0] = 0.5;
        let ham = LocalHamiltonian::SingleBand {
            mu: 0.0,
            u: vec![3.0; 22],
            eps_up: vec![0.0; 22],
            eps_do: vec![0.0; 22],
        };
        let phonon = PhononSpec {
            nph: 12,
            omega: 0.8,
            couplings: vec![PhononCoupling {
                ops: DOCC.to_vec(),
                lambda,
            }],
        };
        let h0 = hamiltonian_at_phonon(&ham, &hs, &[], &[], -1, &phonon);
        let h1 = hamiltonian_at_phonon(&ham, &hs, &[], &[], 0, &phonon);
        // doublon state (bits 0, 1) <-> one phonon: the lambda sqrt(1)
        // coupling element
        assert_ne!(h0.get(3, 7), h1.get(3, 7));
        let ev = EdEvolution::build_quench_with_layout(&h0, &h1, 25, 0.06, BETA, &[0, 1], 2);
        for m in [0usize, 1] {
            for n in [0usize, 9, 25] {
                let d = ev.greens_gtr(m, n, n) - ev.greens_les(m, n, n) + Complex::new(0.0, 1.0);
                assert!((d * d.conj()).re < 1e-20, "mode {m} n {n}: {d:?}");
            }
        }
    }

    #[test]
    fn spectral_function_atomic_two_peaks() {
        // Half-filled atomic limit (eps = mu = 0): the exact A(omega)
        // is 1/2 delta(omega + U/2) + 1/2 delta(omega - U/2), broadened
        // by eta. Validates the Lehmann weights and the peak signs.
        let hs = build(1, SectorClass::SpinResolved);
        let ham = LocalHamiltonian::SingleBand {
            mu: 0.0,
            u: vec![2.0; 2],
            eps_up: vec![0.0; 2],
            eps_do: vec![0.0; 2],
        };
        let ed = EdReference::build(&ham, &hs, &[], &[], 5.0);
        let eta = 0.05;
        let omega = crate::spectral::omega_grid(3.0, 301);
        let a = ed.spectral_function(0, &omega, eta);
        let lor = |x: f64, e: f64| {
            eta / (std::f64::consts::PI * ((x - e).powi(2) + eta * eta))
        };
        let mut worst = 0.0f64;
        for (&om, &v) in omega.iter().zip(&a) {
            worst = worst.max((v - 0.5 * (lor(om, -1.0) + lor(om, 1.0))).abs());
        }
        println!("atomic ED spectral function: worst {worst:.3e}");
        assert!(worst < 1e-12, "atomic ED A(omega) mismatch: {worst:.3e}");
    }
}
