// Project : nQIST
// File    : src/boson.rs
// Purpose : Holstein/Hirsch boson machinery: the free phonon
//           propagator `D0` (blueprint `ppsc::boson_utils::
//           green_from_eps_phonon`, boson_propagator.hpp), the bosonic
//           backward channel (`set_bwd_from_fwd` with statistics
//           `xi = +1`), the `g(t) g(t')` coupling scaling, and the
//           diagonal doublon vertex blocks for the boson
//           pseudo-particle diagrams.

use crate::hilbert::HilbertSpace;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{Contour, ContourGreen, Sign};

/// Bose function `n_B(x) = 1/(e^{beta x} - 1)`.
pub fn bose(beta: f64, x: f64) -> f64 {
    1.0 / ((beta * x).exp() - 1.0)
}

/// Auxiliary phonon exponent (blueprint `bose_exp_1`,
/// boson_propagator.hpp):
/// `bose_exp_1 = Exp(-omega tau) (1 - Exp(-omega beta)) =
/// -Exp(-omega tau) f_B(-omega)`, evaluated stably in two branches.
pub fn bose_exp_1(beta: f64, tau: f64, omega: f64) -> f64 {
    if omega > 0.0 {
        -(-omega * tau).exp() * bose(beta, -omega)
    } else {
        (-(tau - beta) * omega).exp() * bose(beta, omega)
    }
}

/// Free phonon propagator `D0` at frequency `omega` as a scalar Bose
/// `ContourGreen` (blueprint `green_from_eps_phonon_dispatch`):
/// Matsubara, tv, retarded and lesser components of the noninteracting
/// oscillator, `expp[m] = exp(-i omega m h)`.
pub fn green_from_eps_phonon(
    nt: usize,
    ntau: usize,
    beta: f64,
    h: f64,
    omega: f64,
) -> ContourGreen {
    let contour = Contour::new(nt as i64, ntau as i64, 1, 1, nt as f64 * h, beta);
    let mut g = ContourGreen::new(&contour, Sign::Bose, Complex::ZERO);
    let dtau = beta / ntau as f64;
    let expp: Vec<Complex> = (0..=nt)
        .map(|m| {
            let a = omega * m as f64 * h;
            Complex::new(a.cos(), -a.sin())
        })
        .collect();
    let fb = bose(beta, omega);
    let fbm = -bose(beta, -omega);
    for m in 0..=ntau {
        let tau = m as f64 * dtau;
        let be1_p = bose_exp_1(beta, tau, omega);
        let be1_m = bose_exp_1(beta, tau, -omega);
        g.mat.block_mut(m)[0] = Complex::new(-(be1_p - be1_m), 0.0);
        for n in 0..=nt {
            let v = Complex::new(0.0, -1.0) * (be1_p * expp[n].conj() - expp[n] * be1_m);
            g.lmix.raw_block_at_mut(n as i64, m as i64)[0] = v;
        }
    }
    for m in 0..=nt {
        for n in 0..=m {
            let d = expp[m - n];
            g.ret.raw_block_at_mut(m as i64, n as i64)[0] =
                Complex::new(0.0, 1.0) * (-d + d.conj());
            g.less.raw_block_at_mut(n as i64, m as i64)[0] =
                Complex::new(0.0, -1.0) * (fb * d.conj() + fbm * d);
        }
    }
    g
}

/// Bosonic backward Matsubara component: `bwd(m) = +fwd(N - m)`
/// (periodic continuation; no fermionic minus).
pub fn set_bwd_from_fwd_boson_mat(bwd: &mut ContourGreen, fwd: &ContourGreen) {
    let ntau = fwd.ntau;
    for m in 0..=ntau {
        let v = fwd.mat.block(ntau - m)[0];
        bwd.mat.block_mut(m)[0] = v;
    }
}

/// Bosonic backward real-time row `tstp` (blueprint `ppsc::
/// set_bwd_from_fwd` with `xi = +1`; the retarded/lesser structure is
/// statistics-independent, the tv component carries `-xi`):
/// `bwd^R(t,j) = fwd^R(t,j)^dag`, `bwd^<(j,t) = fwd^R(t,j) -
/// fwd^<(j,t)^dag` (equal time from the analytic continuation), and
/// `bwd^tv(t, N-m) = -fwd^tv(t,m)^dag`.
pub fn set_bwd_from_fwd_boson_tstp(tstp: i64, bwd: &mut ContourGreen, fwd: &ContourGreen) {
    for j in 0..tstp {
        let r = fwd.ret.raw_block_at(tstp, j)[0];
        bwd.ret.raw_block_at_mut(tstp, j)[0] = r.conj();
        let l = fwd.less.raw_block_at(j, tstp)[0];
        bwd.less.raw_block_at_mut(j, tstp)[0] = r - l.conj();
    }
    let r = fwd.ret.raw_block_at(tstp, tstp)[0];
    bwd.ret.raw_block_at_mut(tstp, tstp)[0] = -r;
    let l = fwd.less.raw_block_at(tstp, tstp)[0];
    bwd.less.raw_block_at_mut(tstp, tstp)[0] = -r.conj() + l;
    let ntau = fwd.ntau as i64;
    for m in 0..=ntau {
        let v = fwd.lmix.raw_block_at(tstp, m)[0];
        bwd.lmix.raw_block_at_mut(tstp, ntau - m)[0] = -v.conj();
    }
}

/// Scale the free phonon propagator in place by the coupling,
/// `D0(t,t') = g(t) D0_free(t,t') g(t')`, with `gvec` in the C++
/// time-array convention (`gvec[0]` the Matsubara value, `gvec[tstp +
/// 1]` the value at `tstp`).
pub fn scale_phonon(d0: &mut ContourGreen, gvec: &[f64]) {
    let nt = d0.nt;
    let ntau = d0.ntau;
    let g0 = gvec[0];
    for m in 0..=ntau {
        let v = d0.mat.block(m)[0];
        d0.mat.block_mut(m)[0] = Complex::new(g0 * g0, 0.0) * v;
    }
    for n in 0..=(nt as i64) {
        let gn = gvec[n as usize + 1];
        for j in 0..=n {
            let gj = gvec[j as usize + 1];
            let f = Complex::new(gn * gj, 0.0);
            let v = d0.ret.raw_block_at(n, j)[0];
            d0.ret.raw_block_at_mut(n, j)[0] = f * v;
            let v = d0.less.raw_block_at(j, n)[0];
            d0.less.raw_block_at_mut(j, n)[0] = f * v;
        }
        let fm = Complex::new(gn * g0, 0.0);
        for m in 0..=(ntau as i64) {
            let v = d0.lmix.raw_block_at(n, m)[0];
            d0.lmix.raw_block_at_mut(n, m)[0] = fm * v;
        }
    }
}

/// Scale every stored component of a boson propagator by `s`
/// (`D -> s D` on the Matsubara, retarded, lesser and tv parts).
/// [`crate::solver::ImpuritySolver::set_boson_bath`] uses `s = 0.5`
/// to count the phonon vertex once: the diagram builders feed the
/// forward `d0` and the backward `d0_cc` channel, which for bosons
/// are the same physical line.
pub fn scale_boson_propagator(d0: &mut ContourGreen, s: f64) {
    let nt = d0.nt;
    let ntau = d0.ntau;
    let f = Complex::new(s, 0.0);
    for m in 0..=ntau {
        let v = d0.mat.block(m)[0];
        d0.mat.block_mut(m)[0] = f * v;
    }
    for n in 0..=(nt as i64) {
        for j in 0..=n {
            let v = d0.ret.raw_block_at(n, j)[0];
            d0.ret.raw_block_at_mut(n, j)[0] = f * v;
            let v = d0.less.raw_block_at(j, n)[0];
            d0.less.raw_block_at_mut(j, n)[0] = f * v;
        }
        for m in 0..=(ntau as i64) {
            let v = d0.lmix.raw_block_at(n, m)[0];
            d0.lmix.raw_block_at_mut(n, m)[0] = f * v;
        }
    }
}

/// Diagonal blocks of the number operator `n_sigma` in the local
/// Hilbert space: for each sector, `n_sigma(ss) = (cre . ann)(ss)`
/// composed from the single-flavor c-operators (zero blocks omitted).
fn number_blocks(hs: &HilbertSpace, spin: usize) -> Vec<(usize, kadanoffbaym::linalg::DenseMat)> {
    let ann = hs.flavor(0, spin, false);
    let cre = hs.flavor(0, spin, true);
    let mut out = Vec::new();
    for ss in 0..hs.ns() {
        if let Some((sa, m_ann)) = hs.ops[ann].block(ss) {
            if let Some((back, m_cre)) = hs.ops[cre].block(*sa) {
                if *back == ss {
                    let blk = m_cre.mat_mul(m_ann);
                    let mut nonzero = false;
                    for i in 0..blk.nrows() {
                        for j in 0..blk.ncols() {
                            if blk.get(i, j) != kadanoffbaym::complex::Complex::ZERO {
                                nonzero = true;
                            }
                        }
                    }
                    if nonzero {
                        out.push((ss, blk));
                    }
                }
            }
        }
    }
    out
}

/// Diagonal blocks of the doublon (Hirsch) vertex `docc = n_up n_dn`
/// for the single-band Hilbert space (the Holstein driver's boson
/// `pp_int` operator).
pub fn docc_blocks(hs: &HilbertSpace) -> Vec<(usize, kadanoffbaym::linalg::DenseMat)> {
    let nu = number_blocks(hs, 0);
    let nd = number_blocks(hs, 1);
    let mut out = Vec::new();
    for (ss, bu) in &nu {
        if let Some((_, bd)) = nd.iter().find(|(s, _)| s == ss) {
            out.push((*ss, bu.mat_mul(bd)));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The free phonon propagator at equilibrium satisfies the
    /// bosonic periodicity `D0^M(N - m) = D0^M(m)` and the oscillator
    /// trace identity `D0^R(t, t) = 0`.
    #[test]
    fn free_phonon_structure() {
        let (nt, ntau, beta, h, omega) = (8, 40, 5.0, 0.02, 1.0);
        let g = green_from_eps_phonon(nt, ntau, beta, h, omega);
        assert_eq!(g.sign, Sign::Bose);
        for m in 0..=ntau {
            let a = g.mat.block(m)[0];
            let b = g.mat.block(ntau - m)[0];
            assert!((a - b).abs() < 1e-12, "periodicity at {m}: {a:?} vs {b:?}");
        }
        for t in 0..=(nt as i64) {
            assert!(g.ret.raw_block_at(t, t)[0].abs() < 1e-12);
        }
        // equal-time lesser = -i (1 + 2 n_B(omega))
        let want = Complex::new(0.0, -(1.0 + 2.0 * bose(beta, omega)));
        for t in 0..=(nt as i64) {
            let got = g.less.raw_block_at(t, t)[0];
            assert!((got - want).abs() < 1e-12, "{t}: {got:?} vs {want:?}");
        }
    }

    /// The bosonic backward channel of the free propagator is the
    /// Matsubara flip with a PLUS sign and the tv flip with a MINUS.
    #[test]
    fn boson_bwd_channel() {
        let (nt, ntau, beta, h, omega) = (8, 40, 5.0, 0.02, 1.0);
        let fwd = green_from_eps_phonon(nt, ntau, beta, h, omega);
        let contour = Contour::new(nt as i64, ntau as i64, 1, 1, nt as f64 * h, beta);
        let mut bwd = ContourGreen::new(&contour, Sign::Bose, Complex::ZERO);
        set_bwd_from_fwd_boson_mat(&mut bwd, &fwd);
        for m in 0..=ntau {
            assert_eq!(bwd.mat.block(m)[0], fwd.mat.block(ntau - m)[0]);
        }
        for t in 0..=(nt as i64) {
            set_bwd_from_fwd_boson_tstp(t, &mut bwd, &fwd);
        }
        // spot checks: ret row zero (A of a hermitian oscillator),
        // tv sign flip
        for t in 1..=(nt as i64) {
            let r = bwd.ret.raw_block_at(t, 0)[0];
            assert!((r - fwd.ret.raw_block_at(t, 0)[0].conj()).abs() < 1e-14);
            for m in 0..=(ntau as i64) {
                let a = bwd.lmix.raw_block_at(t, ntau as i64 - m)[0];
                let b = -fwd.lmix.raw_block_at(t, m)[0].conj();
                assert!((a - b).abs() < 1e-14, "tv at t={t} m={m}");
            }
        }
    }

    /// `scale_boson_propagator` multiplies every stored component.
    #[test]
    fn constant_scale_hits_all_components() {
        let (nt, ntau, beta, h, omega) = (6, 20, 5.0, 0.02, 1.0);
        let a = green_from_eps_phonon(nt, ntau, beta, h, omega);
        let mut b = green_from_eps_phonon(nt, ntau, beta, h, omega);
        scale_boson_propagator(&mut b, 0.5);
        let half = Complex::new(0.5, 0.0);
        for m in 0..=ntau {
            assert!((b.mat.block(m)[0] - a.mat.block(m)[0] * half).abs() < 1e-15);
        }
        for n in 0..=(nt as i64) {
            for j in 0..=n {
                let dr = b.ret.raw_block_at(n, j)[0] - a.ret.raw_block_at(n, j)[0] * half;
                assert!(dr.abs() < 1e-15);
                let dl = b.less.raw_block_at(j, n)[0] - a.less.raw_block_at(j, n)[0] * half;
                assert!(dl.abs() < 1e-15);
            }
            for m in 0..=(ntau as i64) {
                let d = b.lmix.raw_block_at(n, m)[0] - a.lmix.raw_block_at(n, m)[0] * half;
                assert!(d.abs() < 1e-15);
            }
        }
    }
}
