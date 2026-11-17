// Project : nQIST
// File    : tests/xca_dmft/main.rs
// Purpose : equilibrium DMFT smoke test with the XCA solver at
//           order 3 (spinless Bethe bath, half-filled)
//
// Verifies that the third-order crossing topologies compose with an
// equilibrium DMFT loop {pp_step -> spgf -> Delta = v*^2 Gloc
// mixing}: the loop converges to a finite, physical bath. Only the
// spin-up channels hybridize (a factor 8 fewer channel assignments
// than the full two-spin driver at order 3) and the loop is driven
// manually, mirroring `run_equilibrium` with the Bethe
// self-consistency.

use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{Contour, ContourGreen, Sign};
use nqist::dmft::set_bwd_from_fwd_mat;
use nqist::hamilton::LocalHamiltonian::SingleBand;
use nqist::hilbert::{SectorClass, build};
use nqist::nca::{GfVertex, Hybridization};
use nqist::solver::ImpuritySolver;

const BETA: f64 = 5.0;
const NTAU: usize = 48;
const NT: usize = 8;
const MIXING: f64 = 0.7;
const ERRMAX: f64 = 5.0e-3;
const ITERMAX: usize = 40;

#[test]
#[ignore = "release recommended: cargo test --release --test xca_dmft -- --ignored"]
#[allow(clippy::needless_range_loop)]
fn xca_order3_equilibrium_dmft_converges() {
    let hs = build(1, SectorClass::SpinResolved);
    let mut u = vec![2.0; NT + 2];
    u[0] = 2.0;
    let ham = SingleBand {
        mu: 0.0,
        u,
        eps_up: vec![0.0; NT + 2],
        eps_do: vec![0.0; NT + 2],
    };
    let mut solver = ImpuritySolver::new(NT, NTAU, BETA, 0.02, 5, 3, hs.clone(), ham);
    solver.solve_atomic();

    let contour = Contour::new(NT as i64, NTAU as i64, 1, 1, 0.12, BETA);
    let mut gloc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let mut delta = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let mut delta_cc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let mut gprev = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);

    let cua = hs.flavor(0, 0, false);
    let cuc = hs.flavor(0, 0, true);
    let verts = [GfVertex {
        op_ann_fl: cua,
        op_cre_fl: cuc,
    }];

    let mut err = f64::INFINITY;
    for iter in 1..=ITERMAX {
        let t0 = std::time::Instant::now();
        let pp_ints = [
            Hybridization {
                delta: &delta,
                op_cre_fl: cuc,
                op_ann_fl: cua,
                fermion: true,
                backward: false,
            },
            Hybridization {
                delta: &delta_cc,
                op_cre_fl: cua,
                op_ann_fl: cuc,
                fermion: true,
                backward: true,
            },
        ];
        solver.pp_step(-1, &pp_ints);
        let g_new = solver.get_spgf_mat(&verts, &pp_ints);
        for m in 0..=NTAU {
            let old = gloc.mat.block(m)[0];
            let mixed =
                Complex::new(MIXING, 0.0) * old + Complex::new(1.0 - MIXING, 0.0) * g_new[0][m];
            gloc.mat.block_mut(m)[0] = mixed;
        }
        err = gprev.distance_at(&gloc, -1);
        for m in 0..=NTAU {
            let v = gloc.mat.block(m)[0];
            gprev.mat.block_mut(m)[0] = v;
            // Bethe self-consistency, v* = 1
            delta.mat.block_mut(m)[0] = v;
        }
        set_bwd_from_fwd_mat(&mut delta_cc, &delta);
        eprintln!(
            "dmft-o3 iter {iter:2}: err {err:.3e}  gloc(0) = {:?}  [{:>6.1?}]",
            gloc.mat.block(0)[0],
            t0.elapsed()
        );
        if err < ERRMAX {
            break;
        }
    }
    assert!(err < ERRMAX, "dmft did not converge, last err {err}");

    // physical bath: finite local G and hybridization everywhere,
    // Beta-bath antisymmetry Delta_cc = -Delta(beta - .)
    for m in 0..=NTAU {
        let g = gloc.mat.block(m)[0];
        assert!(
            g.abs() < 10.0 && g.re.is_finite() && g.im.is_finite(),
            "gloc {g} at m {m}"
        );
        let d = delta.mat.block(m)[0];
        let dcc = delta_cc.mat.block(m)[0];
        let want = -delta.mat.block(NTAU - m)[0];
        assert!(
            (dcc - want).abs() < 1.0e-12,
            "Delta_cc antisymmetry broken at m {m}"
        );
        assert!(
            d.re.is_finite() && d.im.is_finite() && d.abs() < 10.0,
            "delta {d} at m {m}"
        );
    }
    // half filling of the spinless channel (eps = 0): with the
    // G(0+) = -(1 - n) convention, n = 1 + Re G(0+)
    let n = 1.0 + gloc.mat.block(0)[0].re;
    assert!((n - 0.5).abs() < 0.05, "density {n}");
}
