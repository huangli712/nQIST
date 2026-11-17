// Project : nQIST
// File    : src/hilbert.rs
// Purpose : Fock-space Hilbert space engine: sectors, c/cdag operators
//           (blueprint ppsc/hilbert_spaces/hilbert_space_base.hpp)

use kadanoffbaym::complex::Complex;
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::types::Sign;

/// Spin degeneracy of the fermionic impurity problems (always 2).
const SPIN_DEG: usize = 2;

/// Sector classification strategy (which quantum numbers are
/// conserved). Blueprint `hilbert_space_base::init(norb, 2,
/// spin_conservation)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectorClass {
    /// Sectors labeled by the total particle number N only
    /// (`spin_conservation = false`; one sector per N).
    ParticleNumber,
    /// Sectors labeled by (N_spin0, N_spin1) — needed for
    /// density-density interactions; the single-band probe
    /// (`single_band_fermi_diag`) and the two-band Kanamori case use
    /// this class. Sector index = N_spin0 * (norb + 1) + N_spin1.
    SpinResolved,
}

/// One symmetry sector of the Fock space.
#[derive(Clone, Debug, PartialEq)]
pub struct Sector {
    /// Sector dimension (`ssdim`).
    pub dim: usize,
    /// Statistics sign: even fermion number -> Bose (+1), odd -> Fermi
    /// (-1), from the sector particle count.
    pub sig: Sign,
    /// Conserved quantum numbers: `[N]` for `ParticleNumber`,
    /// `[N_spin0, N_spin1]` for `SpinResolved`.
    pub npart: Vec<usize>,
    /// Ordered Fock bitstates of this sector (`nstock`), ascending.
    pub states: Vec<u64>,
}

/// Sparse annihilation (or creation) operator for one flavor:
/// per source sector a dense block mapping into exactly one target
/// sector (blueprint `ppsc::operators::Operator`: `to_sector_[s]`,
/// `M_[s]`).
#[derive(Clone, Debug)]
pub struct COp {
    /// For each source sector s: (target sector, block matrix of shape
    /// dim(target) x dim(s)), or `None` where the operator vanishes.
    pub blocks: Vec<Option<(usize, DenseMat)>>,
}

impl COp {
    /// Block acting on source sector s, if any.
    pub fn block(&self, s: usize) -> Option<&(usize, DenseMat)> {
        self.blocks[s].as_ref()
    }
}

/// A classified Fock space with c/cdag operators.
#[derive(Clone, Debug)]
pub struct HilbertSpace {
    /// Number of orbitals (spin+orbital quantum numbers per spin).
    pub norb: usize,
    /// Symmetry sectors.
    pub sectors: Vec<Sector>,
    /// `2 * norb * SPIN_DEG` operators: annihilation flavors first
    /// (spin-major, orbital-minor), then creation flavors
    /// (blueprint flavor(orb, spin, dag) = dag*norb*2 + spin*norb + orb).
    pub ops: Vec<COp>,
    /// Fock state -> index within its sector (`ninv`).
    pub ninv: Vec<usize>,
    /// Fock state -> sector number (`sector_`).
    pub sector_of: Vec<usize>,
}

impl HilbertSpace {
    /// Flavor index of the annihilation (`cre = false`) or creation
    /// (`cre = true`) operator for (orb, spin).
    pub fn flavor(&self, orb: usize, spin: usize, cre: bool) -> usize {
        assert!(orb < self.norb, "flavor: orb out of range");
        assert!(spin < SPIN_DEG, "flavor: spin out of range");
        (cre as usize) * self.norb * SPIN_DEG + spin * self.norb + orb
    }

    /// Number of sectors.
    pub fn ns(&self) -> usize {
        self.sectors.len()
    }

    /// Total Fock-space dimension `2^(2*norb)`.
    pub fn nh(&self) -> usize {
        1usize << (self.norb * SPIN_DEG)
    }
}

/// Bit position of (orb, spin): blueprint convention
/// `bit = spin * norb + orb` (equals `2*orb + spin` only for
/// `norb = 1`).
fn bit_pos(norb: usize, orb: usize, spin: usize) -> usize {
    spin * norb + orb
}

/// Apply c or cdag for (orb, spin) to the Fock state `psi`:
/// returns `(target_state, sign)` with `sign = 0` when the operator
/// annihilates the state (blueprint `hilbert_space_base::c`/`cdag`,
/// Jordan-Wigner sign from the occupied bits below the flavor bit).
fn apply_op(norb: usize, orb: usize, spin: usize, cre: bool, psi: u64) -> (u64, i32) {
    let b = bit_pos(norb, orb, spin);
    let bit = (psi >> b) & 1;
    if (cre && bit == 0) || (!cre && bit == 1) {
        let target = if cre { psi | (1 << b) } else { psi & !(1 << b) };
        // parity of occupied flavors ordered before b
        let below = (psi & ((1 << b) - 1)).count_ones();
        let sign = if below % 2 == 1 { -1 } else { 1 };
        (target, sign)
    } else {
        (psi, 0)
    }
}

/// Build the Fock space `2^(2*norb)` classified into sectors and
/// construct all c/cdag operator blocks (blueprint `init`).
pub fn build(norb: usize, class: SectorClass) -> HilbertSpace {
    assert!(norb >= 1, "hilbert::build: norb must be >= 1");
    assert!(norb <= 31, "hilbert::build: norb too large");
    let nflavor = norb * SPIN_DEG;
    let nh = 1usize << nflavor;
    let ns = match class {
        SectorClass::ParticleNumber => nflavor + 1,
        SectorClass::SpinResolved => (norb + 1) * (norb + 1),
    };

    let mut sectors: Vec<Sector> = Vec::new();
    let mut ninv = vec![0usize; nh];
    let mut sector_of = vec![0usize; nh];

    // classify all states (blueprint init_states: ascending psi, so the
    // states of each sector end up ascending)
    for _ in 0..ns {
        sectors.push(Sector {
            dim: 0,
            sig: Sign::Bose,
            npart: Vec::new(),
            states: Vec::new(),
        });
    }
    for psi in 0..nh {
        // per-spin occupation counts
        let mut nspin = [0usize; SPIN_DEG];
        for spin in 0..SPIN_DEG {
            let bits = (psi >> (spin * norb)) & ((1usize << norb) - 1);
            nspin[spin] = bits.count_ones() as usize;
        }
        let np: usize = nspin.iter().sum();
        let sector = match class {
            SectorClass::ParticleNumber => np,
            SectorClass::SpinResolved => nspin[0] * (norb + 1) + nspin[1],
        };
        ninv[psi] = sectors[sector].dim;
        sector_of[psi] = sector;
        sectors[sector].dim += 1;
        sectors[sector].states.push(psi as u64);
        sectors[sector].sig = if np % 2 == 0 { Sign::Bose } else { Sign::Fermi };
        sectors[sector].npart = match class {
            SectorClass::ParticleNumber => vec![np],
            SectorClass::SpinResolved => vec![nspin[0], nspin[1]],
        };
    }

    // c and cdag operators (blueprint init_c_cdag / set_operator_matrix_{c,cdag})
    let nf = norb * SPIN_DEG * 2;
    let mut ops = Vec::with_capacity(nf);
    for cre in [false, true] {
        for spin in 0..SPIN_DEG {
            for orb in 0..norb {
                ops.push(build_cop(norb, &sectors, &ninv, &sector_of, orb, spin, cre));
            }
        }
    }

    HilbertSpace {
        norb,
        sectors,
        ops,
        ninv,
        sector_of,
    }
}

/// Assemble the blocks of one operator (blueprint
/// `set_operator_matrix_c`/`set_operator_matrix_cdag`).
fn build_cop(
    norb: usize,
    sectors: &[Sector],
    ninv: &[usize],
    sector_of: &[usize],
    orb: usize,
    spin: usize,
    cre: bool,
) -> COp {
    let ns = sectors.len();
    let mut blocks: Vec<Option<(usize, DenseMat)>> = Vec::with_capacity(ns);
    for s in 0..ns {
        // target sector: the first state on which the operator acts
        let mut target: Option<usize> = None;
        for &psi in &sectors[s].states {
            let (tstate, sign) = apply_op(norb, orb, spin, cre, psi as u64);
            if sign != 0 {
                target = Some(sector_of[tstate as usize]);
                break;
            }
        }
        let Some(ts) = target else {
            blocks.push(None);
            continue;
        };
        let mut m = DenseMat::zeros_shape(sectors[ts].dim, sectors[s].dim);
        for (j, &psi) in sectors[s].states.iter().enumerate() {
            let (tstate, sign) = apply_op(norb, orb, spin, cre, psi as u64);
            if sign != 0 {
                let i = ninv[tstate as usize];
                let old = m.get(i, j);
                m.set(i, j, old + Complex::new(sign as f64, 0.0));
            }
        }
        blocks.push(Some((ts, m)));
    }
    COp { blocks }
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn re(z: Complex) -> f64 {
        z.re
    }

    #[test]
    fn single_band_spin_resolved_sectors() {
        // blueprint single_band_fermi_diag: init(1, 2, spin_conservation = true)
        let hs = build(1, SectorClass::SpinResolved);
        assert_eq!(hs.ns(), 4);
        assert_eq!(hs.nh(), 4);
        let dims: Vec<usize> = hs.sectors.iter().map(|s| s.dim).collect();
        let sigs: Vec<Sign> = hs.sectors.iter().map(|s| s.sig).collect();
        let nparts: Vec<Vec<usize>> = hs.sectors.iter().map(|s| s.npart.clone()).collect();
        let states: Vec<Vec<u64>> = hs.sectors.iter().map(|s| s.states.clone()).collect();
        // sector = N_up*2 + N_dn; up = bit 0, dn = bit 1
        assert_eq!(dims, vec![1, 1, 1, 1]);
        assert_eq!(sigs, vec![Sign::Bose, Sign::Fermi, Sign::Fermi, Sign::Bose]);
        assert_eq!(nparts, vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);
        assert_eq!(states, vec![vec![0], vec![2], vec![1], vec![3]]);
        assert_eq!(hs.sector_of, vec![0, 2, 1, 3]);
        assert_eq!(hs.ninv, vec![0, 0, 0, 0]);
    }

    #[test]
    fn single_band_particle_number_sectors() {
        let hs = build(1, SectorClass::ParticleNumber);
        assert_eq!(hs.ns(), 3);
        let dims: Vec<usize> = hs.sectors.iter().map(|s| s.dim).collect();
        let sigs: Vec<Sign> = hs.sectors.iter().map(|s| s.sig).collect();
        assert_eq!(dims, vec![1, 2, 1]);
        assert_eq!(sigs, vec![Sign::Bose, Sign::Fermi, Sign::Bose]);
        assert_eq!(hs.sectors[1].states, vec![1, 2]);
        assert_eq!(hs.sector_of, vec![0, 1, 1, 2]);
        assert_eq!(hs.ninv, vec![0, 0, 1, 0]);
    }

    #[test]
    fn single_band_c_op_matrix_elements() {
        let hs = build(1, SectorClass::SpinResolved);
        let cua = &hs.ops[hs.flavor(0, 0, false)]; // c_up
        let cuc = &hs.ops[hs.flavor(0, 0, true)]; // cdag_up
        let cda = &hs.ops[hs.flavor(0, 1, false)]; // c_dn
        let cdc = &hs.ops[hs.flavor(0, 1, true)]; // cdag_dn

        // c_up: (1,0)->(0,0) and (1,1)->(0,1), both signs +1
        let (t, m) = cua.block(2).unwrap();
        assert_eq!(*t, 0);
        assert_eq!(re(m.get(0, 0)), 1.0);
        let (t, m) = cua.block(3).unwrap();
        assert_eq!(*t, 1);
        assert_eq!(re(m.get(0, 0)), 1.0);
        // c_up annihilates the empty and dn-only sectors
        assert!(cua.block(0).is_none());
        assert!(cua.block(1).is_none());

        // c_dn: (0,1)->(0,0) sign +1; (1,1)->(1,0) sign -1 (bit 0 set)
        let (t, m) = cda.block(1).unwrap();
        assert_eq!(*t, 0);
        assert_eq!(re(m.get(0, 0)), 1.0);
        let (t, m) = cda.block(3).unwrap();
        assert_eq!(*t, 2);
        assert_eq!(re(m.get(0, 0)), -1.0);
        assert!(cda.block(0).is_none());
        assert!(cda.block(2).is_none());

        // cdag_up from dn-only sector: (0,1)->(1,1), sign +1
        // (no occupied bits below bit 0)
        let (t, m) = cuc.block(1).unwrap();
        assert_eq!(*t, 3);
        assert_eq!(re(m.get(0, 0)), 1.0);
        // cdag_dn from up-only sector: (1,0)->(1,1), sign -1 (bit 0 set)
        let (t, m) = cdc.block(2).unwrap();
        assert_eq!(*t, 3);
        assert_eq!(re(m.get(0, 0)), -1.0);
    }

    #[test]
    fn cdag_is_adjoint_of_c() {
        // global check over both classes and several norb
        for (norb, class) in [
            (1, SectorClass::SpinResolved),
            (1, SectorClass::ParticleNumber),
            (2, SectorClass::SpinResolved),
            (2, SectorClass::ParticleNumber),
            (3, SectorClass::SpinResolved),
        ] {
            let hs = build(norb, class);
            let nflavor = norb * SPIN_DEG;
            for f in 0..nflavor {
                let c = &hs.ops[f];
                let cdag = &hs.ops[f + nflavor];
                for s in 0..hs.ns() {
                    if let Some((t, m)) = c.block(s) {
                        // cdag must map t back to s with the adjoint block
                        let (s2, madj) = cdag.block(*t).expect("cdag block exists");
                        assert_eq!(*s2, s, "norb={norb} class={class:?} f={f} s={s}");
                        // element-wise comparison with the adjoint block
                        for i in 0..m.nrows() {
                            for j in 0..m.ncols() {
                                let diff = (m.get(i, j) - madj.get(j, i).conj()).abs();
                                assert!(diff < 1e-15, "block mismatch f={f} s={s}: {diff}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn two_band_spin_resolved_sectors() {
        // 16 Fock states, 9 sectors labeled by (N_up, N_dn);
        // sector = N_up*3 + N_dn
        let hs = build(2, SectorClass::SpinResolved);
        assert_eq!(hs.ns(), 9);
        assert_eq!(hs.nh(), 16);
        let dims: Vec<usize> = hs.sectors.iter().map(|s| s.dim).collect();
        assert_eq!(dims, vec![1, 2, 1, 2, 4, 2, 1, 2, 1]);
        let sigs: Vec<Sign> = hs.sectors.iter().map(|s| s.sig).collect();
        assert_eq!(
            sigs,
            vec![
                Sign::Bose,
                Sign::Fermi,
                Sign::Bose,
                Sign::Fermi,
                Sign::Bose,
                Sign::Fermi,
                Sign::Bose,
                Sign::Fermi,
                Sign::Bose
            ]
        );
        // the (1,1) sector holds the four inter-orbital/intra-orbital
        // singly-doubly occupied states: A+A-, A+B-, B+A-, B+B-
        // (bits: A=0, B=1 per spin; up block low, dn block high)
        assert_eq!(hs.sectors[4].states, vec![5, 6, 9, 10]);
        // quantum numbers of sector 5 = (N_up, N_dn) = (1, 2)
        assert_eq!(hs.sectors[5].npart, vec![1, 2]);
        // all 16 states classified exactly once
        let total: usize = hs.sectors.iter().map(|s| s.dim).sum();
        assert_eq!(total, 16);
    }

    #[test]
    fn two_band_c_op_block() {
        let hs = build(2, SectorClass::SpinResolved);
        // c_{B,up} (orb = 1, spin = 0, bit 1) on the (1,1) sector 4:
        // states [5, 6, 9, 10]; acts on 6 (=B+A-) -> 4 (=A-) and
        // 10 (=B+A-) -> 8 (=A-), both signs +1 (bit 0 empty);
        // target sector 1 (N_up=0, N_dn=1) with states [4, 8]
        let cbua = &hs.ops[hs.flavor(1, 0, false)];
        let (t, m) = cbua.block(4).unwrap();
        assert_eq!(*t, 1);
        assert_eq!((m.nrows(), m.ncols()), (2, 4));
        assert_eq!(re(m.get(0, 1)), 1.0); // 6 -> 4 (row ninv[4]=0)
        assert_eq!(re(m.get(1, 3)), 1.0); // 10 -> 8 (row ninv[8]=1)
        // c annihilates A+A- (5) and A+B- (9): no bit 1
        assert_eq!(re(m.get(0, 0)), 0.0);
        assert_eq!(re(m.get(0, 2)), 0.0);
        assert_eq!(re(m.get(1, 0)), 0.0);
        assert_eq!(re(m.get(1, 2)), 0.0);
        // c_{A,dn} (bit 2) on state 5 (A+A-, bits 0,2): sign = (-1)^1 = -1
        let cada = &hs.ops[hs.flavor(0, 1, false)];
        let (t, m) = cada.block(4).unwrap();
        assert_eq!(*t, 3); // (N_up, N_dn) = (1, 0)
        assert_eq!(re(m.get(0, 0)), -1.0);
    }
}
