// Project : nQIST
// File    : tests/xca_dmft/main.rs
// Purpose : equilibrium DMFT smoke test with the XCA solver at
//           order 3 (Bethe bath, half-filled single band)
//
// Verifies that the third-order crossing topologies compose with the
// existing equilibrium DMFT driver: the loop converges to a finite
// local Green's function and the bath stays physical.

use kadanoffbaym::complex::Complex;
use nqist::dmft::run_equilibrium;
use nqist::hamilton::LocalHamiltonian::SingleBand;
use nqist::hilbert::{SectorClass, build};
use nqist::solver::ImpuritySolver;

#[test]
#[ignore = "expensive: the continuous Bethe hybridization needs the
             full pole budget per iteration; run with
             cargo test --release --test xca_dmft -- --ignored"]
fn xca_order3_equilibrium_dmft_converges() {
    let (nt, ntau, kt) = (8, 64, 5);
    let beta = 5.0;
    let hs = build(1, SectorClass::SpinResolved);
    let mut u = vec![2.0; nt + 2];
    u[0] = 2.0;
    let ham = SingleBand {
        mu: 0.0,
        u,
        eps_up: vec![0.0; nt + 2],
        eps_do: vec![0.0; nt + 2],
    };
    let mut solver = ImpuritySolver::new(nt, ntau, beta, 0.02, kt, 3, hs, ham);
    let (log, gloc, delta, delta_cc) = run_equilibrium(&mut solver, 0.5, 5.0e-3, 12);

    assert!(log.converged, "dmft did not converge: {:?}", log.iter_lines);
    // physical bath: finite local G and hybridization everywhere,
    // Beta-bath antisymmetry Delta_cc = -Delta(beta - .)
    for m in 0..=ntau {
        let g = gloc.mat.block(m)[0];
        assert!(
            g.abs() < 10.0 && g.re.is_finite() && g.im.is_finite(),
            "gloc {g} at m {m}"
        );
        let d = delta.mat.block(m)[0];
        let dcc = delta_cc.mat.block(m)[0];
        let want = -delta.mat.block(ntau - m)[0];
        assert!(
            (dcc - want).abs() < 1.0e-12,
            "Delta_cc antisymmetry broken at m {m}"
        );
        assert!(
            d.re.is_finite() && d.im.is_finite() && d.abs() < 10.0,
            "delta {d} at m {m}"
        );
    }
    // half filling within numerical tolerance of the loop
    let n = -(gloc.mat.block(ntau)[0].re - gloc.mat.block(0)[0].re) / 2.0;
    assert!((n - 0.5).abs() < 0.05, "density {n}");
    let _ = Complex::ZERO;
}
