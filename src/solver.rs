// Project : nQIST
// File    : src/solver.rs
// Purpose : PPSC pseudo-particle impurity solver state machine
//           (blueprint ppsc/solver.hpp + ppsc.cpp)

use crate::hamilton::LocalHamiltonian;
use crate::hilbert::HilbertSpace;
use crate::io::{dens_pp_mu_line, pp_dens_pp_mu_line};
use crate::nca::{self, GfVertex, Hybridization, adjoint_raw};
use crate::oca;
use crate::xca;
use crate::xca_rt;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::dyson::{pseudo_dyson_mat, pseudo_dyson_start, pseudo_dyson_timestep};
use kadanoffbaym::linalg::{DenseMat, eigh};
use kadanoffbaym::types::{Contour, GreenTimeStep, MATSUBARA, PseudoGreen, Sign, TimeFunction};
use kadanoffbaym::weights::Integrator;

/// Initialize the `t = 0` row (ret/les/lmix) of a pseudo-particle
/// green from its Matsubara component: `lmix(0,m) = i*sig*mat(ntau-m)`,
/// `less(0,0) = lmix(0,0)`, and `ret(0,0) = i*mat(0)`. Local copy of
/// the KB `variants::pseudo::set_t0_from_mat_pseudo` kernel, which the
/// KB module-surface narrowing made crate-private (identical body).
fn set_t0_from_mat_pseudo(g: &mut PseudoGreen) {
    let ntau = g.ntau;
    let s = Complex::new(0.0, g.sign.as_i64() as f64);
    for m in 0..=ntau {
        let src = g.mat.block(ntau - m).to_vec();
        for (d, x) in g.lmix.raw_block_at_mut(0, m as i64).iter_mut().zip(src) {
            *d = s * x;
        }
    }
    let l00 = g.lmix.raw_block_at(0, 0).to_vec();
    g.less.raw_block_at_mut(0, 0).copy_from_slice(&l00);
    let m0 = g.mat.block(0).to_vec();
    for (d, x) in g.ret.raw_block_at_mut(0, 0).iter_mut().zip(m0) {
        *d = Complex::I * x;
    }
}

/// Extrapolation weights `p_l` (`l = 0..k`) of the k-th order
/// polynomial row extrapolation. Local copy of the KB
/// `math::extrapolation_weights` kernel, which the KB module-surface
/// narrowing made crate-private (identical algorithm).
fn extrapolation_weights(integ: &Integrator) -> Vec<f64> {
    let k = integ.k;
    assert!(
        (0..=5).contains(&k),
        "extrapolation order 0 <= k <= 5, got {k}"
    );
    let piw = kadanoffbaym::weights::PolynomialInterpolationWeights::new(k);
    let n = (k + 1) as usize;
    let mut p = vec![0.0f64; n];
    for l in 0..n {
        let sign = 1.0 - 2.0 * (l % 2) as f64;
        for (m, pm) in p.iter_mut().enumerate() {
            *pm += sign * piw.at(l, m);
        }
    }
    p
}

/// PPSC impurity solver: per-sector pseudo-particle Green's functions
/// and self-energies, the density matrix, and the atomic-limit
/// initialization (blueprint `ppsc::solver`).
///
/// All Matsubara-axis operations follow the clean NESSi/KB grid
/// (`dtau = beta / ntau`, points `m = 0..=ntau`, density matrix at
/// `-mat[ntau]`), exactly like the C++ reference: the ppsc sources
/// appear to use `beta / (ntau() - 1)` and half-open `range(0, ntau())`
/// loops, but their `herm_matrix_matrix_ref` view reports `ntau() =
/// ntau + 1` (the number of POINTS), which resolves those expressions
/// to the clean grid.
/// Solver family for the strong-coupling impurity engine. The user
/// selects the family first and then the expansion `order` of that
/// family ([`ImpuritySolver::with_solver`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solver {
    /// PPSC-like native kernels (the C++ PPSC port): `order = 1` is
    /// NCA, `order = 2` is NCA + native OCA. No crossing topologies.
    Ppsc,
    /// XCA backbone: the sum of crossing orders `m = 1..=order`
    /// evaluated by the SOE-separated skeleton diagrams. Order 1
    /// reproduces NCA and order 2 reproduces OCA up to the fit and
    /// Gregory-warmup residual; orders >= 3 add the crossing
    /// topologies.
    Xca,
    /// IPT (weak-coupling second-order sunset on the Bethe lattice).
    /// Select it through [`crate::ipt::IptSolver`], not through
    /// [`ImpuritySolver`]; only `order = 2` is implemented today.
    Ipt,
}

pub struct ImpuritySolver {
    /// Solver-family selection; the meaning of `order` depends on it.
    pub solver: Solver,
    /// Route the real-time order-2 self-energy through the XCA
    /// backbones (`xca_rt::eval2`, the calibrated equivalent of the
    /// OCA kernel). Plumbing/verification switch; default false.
    pub rt_xca_m2: bool,
    /// m=1 real-time Sigma through the real-time XCA route (`eval1`)
    /// for both the fermion NCA diagrams and the Holstein boson
    /// diagrams (anchored at machine precision against the native
    /// kernels). Opt-in: the route reconstructs the hybridization
    /// from its column `t' = 0` through an exponential sum in the time
    /// difference, i.e. it assumes a time-translation invariant bath
    /// (a static bath; gated by the dimer real-time tests).
    /// Self-consistent non-equilibrium DMFT breaks the premise (the
    /// U-quench Bethe bath deviates from time translation invariance
    /// by 8e-3), so orders 1 and 2 default to the native kernels.
    pub rt_xca_m1: bool,
    /// Enable real-time XCA orders >= 3 through the validated
    /// block-cut component families (xca_rt::evalfam2). Fermion and
    /// mixed boson channels are routed through evalfam2/evalfam2m;
    /// the mixed boson R-cut calibration is still open.
    ///
    /// KNOWN BOUNDARIES (see xca_rt::evalfam2 for details):
    /// - Fit-resolvable grids: the fit window nt*h must resolve the
    ///   bath structure. A too-short window returns garbage poles
    ///   whose kernel factors e^{-w tau} are then extrapolated
    ///   exponentially over the full Matsubara axis.
    /// - The backward-channel greater component is genuinely
    ///   TWO-pole structured; use rt_xca_poles >= 2 when the
    ///   backward sectors dominate.
    /// - The all-imaginary les cut's reversed vertex pairing is
    ///   anchored against the reference kernel for TWO internal
    ///   vertices (order 2). The order-3 all-I cut uses the FORWARD
    ///   walk - ED-ADJUDICATED. XCA_M3_REV=1 selects the reversed
    ///   variant for comparison studies.
    pub rt_xca_m3: bool,
    /// Pole budget for the real-time XCA fits of orders >= 3 (cost
    /// knob: every pole costs poles^m in the diagram sums).
    pub rt_xca_poles: usize,
    /// Real-time XCA hybridization fit (orders 1 and 2): root-mean-
    /// square error tolerance of each channel's complex-exponential
    /// fit of the greater component on the real-time grid. The pole
    /// count grows until the tolerance is met, up to `xca_max_poles`
    /// inside the pole window `xca_w_max` (`xca_rt::fit::csoe_fit_tol`).
    pub rt_xca_fit_tol: f64,
    /// Equilibrium XCA hybridization fit (`Solver::Xca`, orders >= 2):
    /// root-mean-square error tolerance of each channel's SOE on the
    /// Matsubara grid, the pole-count upper bound, and the spectral
    /// cutoff. The pole count grows until the tolerance is met (up to
    /// the bound). The DMFT drivers tie `xca_fit_tol` to their
    /// convergence criterion (0.1 x errmax; blueprint TRIQS/xca
    /// `hyb_tol = 0.1 * tol`).
    pub xca_fit_tol: f64,
    pub xca_max_poles: usize,
    pub xca_w_max: f64,
    pub nt: usize,
    pub ntau: usize,
    pub beta: f64,
    pub h: f64,
    /// Gregory quadrature order / start-up window length (blueprint
    /// `kt`).
    pub kt: usize,
    /// Expansion order, interpreted by the selected [`Solver`]:
    /// [`Solver::Ppsc`] accepts 1 (NCA) and 2 (OCA); [`Solver::Xca`]
    /// accepts any order >= 1 (1 = XCA(1), the NCA equivalent, 2 =
    /// XCA(1)+XCA(2), the OCA equivalent, >= 3 = with the crossing
    /// topologies).
    pub order: usize,
    pub hs: HilbertSpace,
    pub ham: LocalHamiltonian,
    /// Pseudo-particle propagator per sector (matrix size = sector
    /// dimension, statistics sign = sector sign).
    pub pp_g: Vec<PseudoGreen>,
    /// Pseudo-particle self-energy per sector (zero until the NCA
    /// diagrams are attached in the next phase).
    pub pp_sigma: Vec<PseudoGreen>,
    /// Hamiltonian blocks `H_s(tstp)` per sector (blueprint
    /// `H` = list of `cntr::function`).
    pub h_func: Vec<TimeFunction>,
    /// Optional static boson bath (Holstein/Hirsch vertex); it feeds
    /// the NCA, OCA, and XCA channels at every order.
    pub boson: Option<crate::nca::BosonBath>,
    /// Density matrix per time step (`rho[tstp + 1][sector]`).
    pub rho: Vec<Vec<DenseMat>>,
    /// Pseudo-particle chemical potential.
    pub pp_mu: f64,
}

impl ImpuritySolver {
    /// Allocate the solver and fill the Hamiltonian time functions
    /// from `ham` for all `tstp = -1..=nt` (blueprint constructor +
    /// `update_hamiltonian`).
    ///
    /// Convenience entry point: `order <= 2` maps to
    /// [`Solver::Ppsc`] (1 = NCA, 2 = OCA), `order >= 3` maps to
    /// [`Solver::Xca`] (the pure XCA series). Use
    /// [`Self::with_solver`] to choose the family and the order
    /// explicitly.
    pub fn new(
        nt: usize,
        ntau: usize,
        beta: f64,
        h: f64,
        kt: usize,
        order: usize,
        hs: HilbertSpace,
        ham: LocalHamiltonian,
    ) -> Self {
        assert!(ntau >= 2, "ntau must be >= 2");
        assert!(beta > 0.0, "beta must be positive");
        let solver = if order <= 2 {
            Solver::Ppsc
        } else {
            Solver::Xca
        };
        let tmax = nt as f64 * h;
        let ns = hs.ns();
        let make_ppgf = |dim: usize, sig: Sign| {
            let c = Contour::new(nt as i64, ntau as i64, dim as i64, dim as i64, tmax, beta);
            PseudoGreen::new(&c, sig, Complex::ZERO)
        };
        let pp_g = hs.sectors.iter().map(|s| make_ppgf(s.dim, s.sig)).collect();
        let pp_sigma = hs.sectors.iter().map(|s| make_ppgf(s.dim, s.sig)).collect();
        let h_func = (0..ns)
            .map(|s| TimeFunction::zeros(nt, hs.sectors[s].dim, hs.sectors[s].dim))
            .collect();
        let rho = (0..nt + 2)
            .map(|_| {
                (0..ns)
                    .map(|s| DenseMat::zeros(hs.sectors[s].dim))
                    .collect()
            })
            .collect();
        let mut solver = ImpuritySolver {
            rt_xca_m2: false,
            rt_xca_m1: false,
            rt_xca_m3: false,
            rt_xca_poles: crate::xca::MAX_POLES_DEFAULT,
            rt_xca_fit_tol: crate::xca::FIT_TOL_DEFAULT,
            xca_fit_tol: crate::xca::FIT_TOL_DEFAULT,
            xca_max_poles: crate::xca::EQ_MAX_POLES_DEFAULT,
            xca_w_max: crate::xca::W_MAX_DEFAULT,
            solver,
            nt,
            ntau,
            beta,
            h,
            kt,
            order,
            hs,
            ham,
            pp_g,
            pp_sigma,
            h_func,
            rho,
            pp_mu: 0.0,
            boson: None,
        };
        solver.update_hamiltonian();
        solver
    }

    /// Build the solver with an explicit solver-family/order selection.
    ///
    /// - `Solver::Ppsc` (PPSC-like native kernels): `order` 1 = NCA,
    ///   2 = NCA + OCA.
    /// - `Solver::Xca`: `order` 1 = XCA(1) (NCA equivalent), 2 =
    ///   XCA(1)+XCA(2) (OCA equivalent), >= 3 = the full XCA series
    ///   truncated at `order`. Orders 1 and 2 take the native NCA/OCA
    ///   real-time kernels: the real-time XCA routes
    ///   ([`Self::rt_xca_m1`], [`Self::rt_xca_m2`]) assume a
    ///   time-translation invariant bath and are opt-in. Orders >= 3
    ///   need the real-time XCA route
    ///   ([`Self::rt_xca_m3`], enabled here) with the same premise.
    pub fn with_solver(
        nt: usize,
        ntau: usize,
        beta: f64,
        h: f64,
        kt: usize,
        solver: Solver,
        order: usize,
        hs: HilbertSpace,
        ham: LocalHamiltonian,
    ) -> Self {
        match solver {
            Solver::Ppsc => assert!(
                order == 1 || order == 2,
                "PPSC solver supports order 1 (NCA) and 2 (OCA)"
            ),
            Solver::Xca => assert!(order >= 1, "XCA solver needs order >= 1"),
            Solver::Ipt => panic!(
                "the IPT solver is a separate type: use ipt::IptSolver::with_solver"
            ),
        }
        let mut s = Self::new(nt, ntau, beta, h, kt, order, hs, ham);
        s.solver = solver;
        if solver == Solver::Xca {
            s.rt_xca_m3 = order >= 3;
        }
        s
    }

    /// Zero the Matsubara pseudo-particle self-energies before the XCA
    /// backbone accumulates its order sum (the native `set_sigma_mat`
    /// writes, the XCA evaluators add).
    fn zero_sigma(&mut self) {
        for g in &mut self.pp_sigma {
            for i in 0..=self.ntau {
                for z in g.mat.block_mut(i) {
                    *z = Complex::ZERO;
                }
            }
        }
    }

    /// Attach (or detach) a static boson bath for Holstein/Hirsch
    /// calculations. The bath carries the coupling-scaled free phonon
    /// propagators and the diagonal vertex blocks; it contributes
    /// additional NCA/OCA/XCA diagrams to the pseudo-particle
    /// self-energies in [`Self::update_sigma`] and to the
    /// equilibrium single-particle Green's function.
    ///
    /// The diagram builders feed BOTH the forward `d0` and the
    /// backward `d0_cc` boson channel of the vertex. For bosons those
    /// two channels are the same physical line
    /// (`D(beta - tau) = D(tau)`, and the real-time mirror is the
    /// contour flip of the same propagator), so counting both would
    /// double the phonon self-energy and replace the Hamiltonian
    /// coupling `lambda` with `sqrt(2) lambda`. Each channel is
    /// therefore scaled by 1/2 on entry: the library convention is
    /// the physical `H_ph = omega0 b^dag b + lambda docc (b + b^dag)`
    /// with `lambda = g`. The PPSC Holstein driver counts the line
    /// twice, so reproducing it requires `g = sqrt(2) g_ppsc`.
    pub fn set_boson_bath(&mut self, bath: Option<crate::nca::BosonBath>) {
        // The real-time order >= 3 gate lives in `update_sigma`
        // (`assert!(order <= 2 || rt_xca_m3)`); the equilibrium
        // Matsubara path carries boson channels at every order.
        self.boson = bath.map(|mut b| {
            crate::boson::scale_boson_propagator(&mut b.d0, 0.5);
            crate::boson::scale_boson_propagator(&mut b.d0_cc, 0.5);
            b
        });
    }

    /// Fill `h_func[s](tstp)` from the local Hamiltonian for all
    /// sectors and `tstp = -1..=nt` (blueprint `update_hamiltonian`).
    pub fn update_hamiltonian(&mut self) {
        let ns = self.hs.ns();
        for s in 0..ns {
            for tstp in -1..=(self.nt as i64) {
                let blk = self.ham.sector_block(&self.hs, s, tstp);
                self.h_func[s].block_mut(tstp).copy_from_slice(blk.raw());
            }
        }
    }

    /// Atomic-limit initialization (blueprint `zeroth_order`):
    /// diagonalize `H_s(-1)` per sector, set the pseudo-particle
    /// chemical potential from the atomic partition function, and
    /// fill the Matsubara component with
    /// `mat[m] = V diag(-exp(-dtau m (E - pp_mu))) V^dagger` on the
    /// clean grid `dtau = beta / ntau`, `m = 0..=ntau`. Returns the
    /// `pp_dens, pp_mu = ...` log line.
    pub fn solve_atomic(&mut self) -> String {
        let _timed = crate::progress::Timed::new("solver", "solve_atomic");
        let ns = self.hs.ns();
        let mut evals: Vec<Vec<f64>> = Vec::with_capacity(ns);
        let mut evecs: Vec<DenseMat> = Vec::with_capacity(ns);
        let mut emin = f64::INFINITY;
        for s in 0..ns {
            let h0 = self.ham.sector_block(&self.hs, s, -1);
            let (w, v) = eigh(&h0);
            emin = emin.min(w[0]);
            evals.push(w);
            evecs.push(v);
        }

        // pp_dens = sum over ALL atomic eigenvalues of exp(-beta (E - emin))
        let mut pp_dens = 0.0;
        for w in &evals {
            for &e in w {
                pp_dens += (-self.beta * (e - emin)).exp();
            }
        }
        self.pp_mu = emin - pp_dens.ln() / self.beta;

        // Matsubara fill on the clean grid (see the struct docs on the
        // herm_matrix_matrix_ref convention)
        let dtau = self.beta / self.ntau as f64;
        for s in 0..ns {
            let d = self.pp_g[s].size1;
            for m in 0..=self.ntau {
                // D = diag(-exp(-dtau m (E - pp_mu)))
                let mut diag = DenseMat::zeros(d);
                for (i, &e) in evals[s].iter().enumerate() {
                    let x = -((-dtau) * m as f64 * (e - self.pp_mu)).exp();
                    diag.set(i, i, Complex::new(x, 0.0));
                }
                let g = evecs[s].mat_mul(&diag).mat_mul(&evecs[s].adjoint());
                self.pp_g[s].mat.block_mut(m).copy_from_slice(g.raw());
            }
        }
        pp_dens_pp_mu_line(pp_dens, self.pp_mu)
    }

    /// Initialize the `t = 0` row of every sector propagator from its
    /// Matsubara component (blueprint `init_real_time`).
    pub fn set_t0(&mut self) {
        for s in 0..self.hs.ns() {
            set_t0_from_mat_pseudo(&mut self.pp_g[s]);
        }
    }

    /// Sector density matrix at `tstp` (blueprint `density_matrix`):
    /// `-mat[ntau]` on the Matsubara axis, `i * sig * les(t, t)`
    /// otherwise.
    fn density_matrix(&self, s: usize, tstp: i64) -> DenseMat {
        let g = &self.pp_g[s];
        let d = g.size1;
        if tstp == MATSUBARA {
            let mut rho = DenseMat::from_slice(d, d, g.mat.block(self.ntau));
            for z in rho.raw_mut() {
                *z = -*z;
            }
            rho
        } else {
            let fac = Complex::I * Complex::new(g.sign.as_i64() as f64, 0.0);
            let mut rho = DenseMat::from_slice(d, d, g.less.raw_block_at(tstp, tstp));
            for z in rho.raw_mut() {
                *z = fac * *z;
            }
            rho
        }
    }

    /// Refresh `rho[tstp + 1]` for all sectors (blueprint
    /// `update_density_matrix`).
    pub fn update_density_matrix(&mut self, tstp: i64) {
        assert!(
            (-1..=(self.nt as i64)).contains(&tstp),
            "update_density_matrix: tstp out of range"
        );
        for s in 0..self.hs.ns() {
            self.rho[(tstp + 1) as usize][s] = self.density_matrix(s, tstp);
        }
    }

    /// Normalize the Matsubara components so that the pseudo-particle
    /// density becomes one and shift `pp_mu` accordingly (blueprint
    /// `normalize_ppgfs_mat`: rescale `m = 0..=ntau` with the clean
    /// grid `dtau = beta / ntau`). Returns the `dens, pp_mu = ...`
    /// log line (with the pre-normalization density, like the C++
    /// print).
    pub fn normalize_ppgf(&mut self) -> String {
        let mut dens = 0.0;
        for s in 0..self.hs.ns() {
            dens += self.density_matrix(s, MATSUBARA).trace().re;
        }
        assert!(dens > 0.0, "normalize_ppgf: cannot normalize dens <= 0");

        let dlam = -dens.ln() / self.beta;
        let dtau = self.beta / self.ntau as f64;
        for s in 0..self.hs.ns() {
            for m in 0..=self.ntau {
                let f = Complex::new((dlam * dtau * m as f64).exp(), 0.0);
                for z in self.pp_g[s].mat.block_mut(m) {
                    *z = *z * f;
                }
            }
        }
        self.pp_mu += dlam;
        dens_pp_mu_line(dens, self.pp_mu)
    }

    /// Symmetrize the Matsubara component
    /// `mat[m] = 0.5 * (mat[m] + mat[m]^dag)` for `m = 0..=ntau`
    /// (blueprint `force_matsubara_hermitian`).
    fn force_matsubara_hermitian(g: &mut PseudoGreen, ntau: usize) {
        for m in 0..=ntau {
            let d = g.size1;
            let a = DenseMat::from_slice(d, d, g.mat.block(m));
            let mut out = a.clone();
            out.axpy(1.0, &a.adjoint());
            out.scale(0.5);
            g.mat.block_mut(m).copy_from_slice(out.raw());
        }
    }

    /// Solve the pseudo-particle Dyson equation for all sectors at
    /// `tstp` (blueprint `solve_pp_dyson` window dispatch:
    /// `-1` Matsubara, `0` t0-from-mat, `1..=kt` start window,
    /// `> kt` single time step).
    pub fn solve_dyson(&mut self, tstp: i64) {
        let integ = Integrator::new(self.kt as i64);
        for s in 0..self.hs.ns() {
            if tstp == MATSUBARA {
                Self::force_matsubara_hermitian(&mut self.pp_sigma[s], self.ntau);
                let d = self.pp_g[s].size1;
                let h0 = DenseMat::from_slice(d, d, self.h_func[s].block(MATSUBARA));
                pseudo_dyson_mat(
                    &mut self.pp_g[s],
                    self.pp_mu,
                    &h0,
                    &self.pp_sigma[s],
                    &integ,
                    self.beta,
                );
                Self::force_matsubara_hermitian(&mut self.pp_g[s], self.ntau);
            } else if tstp == 0 {
                set_t0_from_mat_pseudo(&mut self.pp_g[s]);
            } else if tstp <= self.kt as i64 {
                pseudo_dyson_start(
                    &mut self.pp_g[s],
                    self.pp_mu,
                    &self.h_func[s],
                    &self.pp_sigma[s],
                    &integ,
                    self.beta,
                    self.h,
                );
            } else {
                pseudo_dyson_timestep(
                    tstp,
                    &mut self.pp_g[s],
                    self.pp_mu,
                    &self.h_func[s],
                    &self.pp_sigma[s],
                    &integ,
                    self.beta,
                    self.h,
                    None,
                );
            }
        }
    }

    /// Refresh the pseudo-particle self-energies at `tstp` from the
    /// NCA diagrams (blueprint `update_sigma`): Matsubara axis or
    /// real-time kernels, dispatched on `tstp`.
    pub fn update_sigma(&mut self, tstp: i64, pp_ints: &[Hybridization]) {
        if tstp == MATSUBARA {
            match self.solver {
                Solver::Ppsc => {
                    assert!(
                        self.order == 1 || self.order == 2,
                        "PPSC solver supports order 1 (NCA) and 2 (OCA)"
                    );
                    nca::set_sigma_mat(&self.hs, &self.pp_g, pp_ints, &mut self.pp_sigma);
                    if let Some(bath) = &self.boson {
                        nca::add_boson_sigma_mat(bath, &self.pp_g, &mut self.pp_sigma);
                    }
                    if self.order == 2 {
                        self.add_sigma_mat_oca(pp_ints);
                    }
                }
                Solver::Xca => {
                    assert!(self.order >= 1, "XCA solver needs order >= 1");
                    self.zero_sigma();
                    let xca_channels = oca::channels_with_bath(pp_ints, self.boson.as_ref());
                    let fits = if self.order == 1 {
                        // order 1 reads the raw hybridization: no fit
                        xca::unused_fits(xca_channels.len(), self.beta)
                    } else {
                        xca::fit_channels_tol(
                            &xca_channels,
                            self.beta,
                            self.ntau,
                            self.xca_max_poles,
                            self.xca_w_max,
                            self.xca_fit_tol,
                        )
                    };
                    let integ = Integrator::new(self.kt as i64);
                    let grids = xca::sigma::Grids {
                        ntau: self.ntau,
                        dtau: self.beta / self.ntau as f64,
                        pp_g: &self.pp_g,
                    };
                    for m in 1..=self.order {
                        xca::add_sigma_mat_xca(
                            m,
                            &self.hs,
                            &xca_channels,
                            &fits,
                            &grids,
                            &integ,
                            &mut self.pp_sigma,
                        );
                    }
                }
                Solver::Ipt => panic!("ImpuritySolver cannot run the IPT solver"),
            }
        } else {
            assert!(
                self.order <= 2 || self.rt_xca_m3,
                "update_sigma: real-time steps support NCA/OCA only; \
                 XCA orders >= 3 need rt_xca_m3 (block-cut families)"
            );
            if self.rt_xca_m1 {
                // m=1 through the rt-XCA machinery: the fermion NCA
                // diagrams and the Holstein boson diagrams BOTH
                // through eval1 (fits per delta)
                //
                // fresh timestep: zero the touched rows (the
                // stock set_sigma_tstp semantics; the xca
                // evaluators accumulate only)
                for si in 0..self.hs.ns() {
                    for j in 0..=tstp {
                        for z in self.pp_sigma[si].ret.raw_block_at_mut(tstp, j) {
                            *z = Complex::ZERO;
                        }
                        for z in self.pp_sigma[si].less.raw_block_at_mut(j, tstp) {
                            *z = Complex::ZERO;
                        }
                    }
                    for m in 0..=(self.pp_sigma[si].ntau as i64) {
                        for z in self.pp_sigma[si].lmix.raw_block_at_mut(tstp, m) {
                            *z = Complex::ZERO;
                        }
                    }
                }

                let diagrams = nca::build_sigma_diagrams(&self.hs, pp_ints);
                let fits = xca_rt::fit_channels_rt_tol(
                    pp_ints,
                    self.h,
                    tstp.max(self.kt as i64) as usize,
                    self.xca_max_poles,
                    self.xca_w_max,
                    self.rt_xca_fit_tol,
                );
                let pp_g = &self.pp_g;
                let h = self.h;
                for d in &diagrams {
                    let i1 = pp_ints
                        .iter()
                        .position(|c| std::ptr::eq(c.delta, d.delta))
                        .unwrap_or(0);
                    xca_rt::eval1::add_sigma_tstp_xca1(
                        tstp,
                        d,
                        pp_g,
                        h,
                        &fits[i1],
                        &mut self.pp_sigma,
                    );
                }
                if let Some(bath) = &self.boson {
                    let mut d0_cc = bath.d0_cc.clone();
                    for m in 0..=(self.ntau as i64) {
                        let v = bath.d0.lmix.raw_block_at(0, m)[0];
                        d0_cc.lmix.raw_block_at_mut(0, self.ntau as i64 - m)[0] = -v.conj();
                    }
                    d0_cc.less.block_mut(d0_cc.less.idx(0, 0))[0] = bath.d0.ret.raw_block_at(0, 0)
                        [0]
                        - bath.d0.less.raw_block_at(0, 0)[0].conj();
                    let d0fit = xca_rt::fit_one_delta_tol(
                        &bath.d0,
                        self.h,
                        tstp.max(self.kt as i64) as usize,
                        self.xca_max_poles,
                        self.xca_w_max,
                        self.rt_xca_fit_tol,
                    );
                    // the boson diagrams reference the ORIGINAL
                    // bath deltas; eval1 reads through the fit only,
                    // so rebuild the diagram set against the FILLED
                    // d0_cc by hand (the structure is trivial)
                    for delta in [&bath.d0, &d0_cc] {
                        for (sb, blk) in &bath.blocks {
                            let d = nca::SigmaDiagram {
                                out: *sb,
                                sa: *sb,
                                delta,
                                op1: blk.clone(),
                                op2: blk.clone(),
                                prefactor: Complex::I,
                            };
                            xca_rt::eval1::add_sigma_tstp_xca1(
                                tstp,
                                &d,
                                pp_g,
                                h,
                                &d0fit,
                                &mut self.pp_sigma,
                            );
                        }
                    }
                }
            } else {
                nca::set_sigma_tstp(tstp, &self.hs, &self.pp_g, pp_ints, &mut self.pp_sigma);
                if let Some(bath) = &self.boson {
                    nca::add_boson_sigma_tstp(tstp, bath, &self.pp_g, &mut self.pp_sigma);
                }
            }
            if self.order >= 2 {
                if self.rt_xca_m2 && self.order == 2 {
                    // NOTE: add-on-top semantics - the NCA base
                    // (set_sigma_tstp with its own zeroing) runs
                    // before this branch in the order>=2 flow; do
                    // NOT zero here (that wipes the NCA base)
                    //
                    // order-2 through the real-time XCA backbones
                    // (calibrated equivalent of the OCA kernel; see
                    // xca_rt::eval2). With a boson bath the MIXED
                    // diagram set (channels_with_bath) runs through
                    // eval2 as well: the boson-walk diagrams are
                    // machine-anchored against the stock
                    // (rt_m2_holstein_anchor_vs_stock). The d0_cc's
                    // t=0 row - left unfilled by the production
                    // constructor - is filled locally per the same
                    // convention (bwd^tv(0, N-m) = -fwd(0,m)^dag, the
                    // (0,0) contact).
                    let mut fits = xca_rt::fit_channels_rt_tol(
                        pp_ints,
                        self.h,
                        tstp.max(self.kt as i64) as usize,
                        self.xca_max_poles,
                        self.xca_w_max,
                        self.rt_xca_fit_tol,
                    );
                    if let Some(bath) = &self.boson {
                        let mut d0_cc = bath.d0_cc.clone();
                        for m in 0..=(self.ntau as i64) {
                            let v = bath.d0.lmix.raw_block_at(0, m)[0];
                            d0_cc.lmix.raw_block_at_mut(0, self.ntau as i64 - m)[0] = -v.conj();
                        }
                        d0_cc.less.block_mut(d0_cc.less.idx(0, 0))[0] =
                            bath.d0.ret.raw_block_at(0, 0)[0]
                                - bath.d0.less.raw_block_at(0, 0)[0].conj();
                        let bfit = xca_rt::fit_one_delta_tol(
                            &bath.d0,
                            self.h,
                            tstp.max(self.kt as i64) as usize,
                            self.xca_max_poles,
                            self.xca_w_max,
                            self.rt_xca_fit_tol,
                        );
                        fits.push(bfit.clone());
                        fits.push(bfit);
                        let mut channels: Vec<oca::OcaChannel> = pp_ints
                            .iter()
                            .map(|h| oca::OcaChannel {
                                delta: h.delta,
                                fermion: true,
                                backward: h.backward,
                                ann: oca::VertSrc::Flavor(h.op_ann_fl),
                                cre: oca::VertSrc::Flavor(h.op_cre_fl),
                            })
                            .collect();
                        channels.push(oca::OcaChannel {
                            delta: &bath.d0,
                            fermion: false,
                            backward: false,
                            ann: oca::VertSrc::Diag(&bath.blocks),
                            cre: oca::VertSrc::Diag(&bath.blocks),
                        });
                        channels.push(oca::OcaChannel {
                            delta: &d0_cc,
                            fermion: false,
                            backward: true,
                            ann: oca::VertSrc::Diag(&bath.blocks),
                            cre: oca::VertSrc::Diag(&bath.blocks),
                        });
                        let diagrams = oca::build_sigma_diagrams_oca_channels(&self.hs, &channels);
                        let integ = Integrator::new(self.kt as i64);
                        let pp_g = &self.pp_g;
                        let beta = self.beta;
                        let h = self.h;
                        let kt = self.kt as i64;
                        for d in &diagrams {
                            let i1 = channels
                                .iter()
                                .position(|c| std::ptr::eq(c.delta, d.lam1))
                                .unwrap_or(0);
                            let i2 = channels
                                .iter()
                                .position(|c| std::ptr::eq(c.delta, d.lam2))
                                .unwrap_or(0);
                            xca_rt::add_sigma_tstp_xca2(
                                tstp,
                                d,
                                pp_g,
                                beta,
                                h,
                                kt,
                                &integ,
                                &fits[i1.min(fits.len() - 1)],
                                &fits[i2.min(fits.len() - 1)],
                                &mut self.pp_sigma,
                            );
                        }
                    } else {
                        let diagrams = oca::build_sigma_diagrams_oca(&self.hs, pp_ints);
                        let integ = Integrator::new(self.kt as i64);
                        let pp_g = &self.pp_g;
                        let beta = self.beta;
                        let h = self.h;
                        let kt = self.kt as i64;
                        for d in &diagrams {
                            let i1 = pp_ints
                                .iter()
                                .position(|c| std::ptr::eq(c.delta, d.lam1))
                                .unwrap_or(0);
                            let i2 = pp_ints
                                .iter()
                                .position(|c| std::ptr::eq(c.delta, d.lam2))
                                .unwrap_or(0);
                            xca_rt::add_sigma_tstp_xca2(
                                tstp,
                                d,
                                pp_g,
                                beta,
                                h,
                                kt,
                                &integ,
                                &fits[i1],
                                &fits[i2],
                                &mut self.pp_sigma,
                            );
                        }
                    }
                } else {
                    let channels = oca::channels_with_bath(pp_ints, self.boson.as_ref());
                    let diagrams = oca::build_sigma_diagrams_oca_channels(&self.hs, &channels);
                    let integ = Integrator::new(self.kt as i64);
                    let pp_g = &self.pp_g;
                    let beta = self.beta;
                    let h = self.h;
                    let kt = self.kt as i64;
                    if std::env::var("NQIST_SKIP_OCA_SIGMA").is_ok() {
                        static WARNED: std::sync::Once = std::sync::Once::new();
                        WARNED.call_once(|| {
                            eprintln!(
                                "warning: NQIST_SKIP_OCA_SIGMA is set: the real-time OCA self-energy kernel is disabled (diagnostic switch)"
                            );
                        });
                    }
                    for d in &diagrams {
                        if std::env::var("NQIST_SKIP_OCA_SIGMA").is_err() {
                            oca::add_sigma_tstp_oca(
                                tstp,
                                d,
                                pp_g,
                                beta,
                                h,
                                kt,
                                &integ,
                                &mut self.pp_sigma,
                            );
                        }
                    }
                }
            }
            if self.order >= 3 {
                assert!(self.rt_xca_m3);
                // real-time XCA orders 3..=order via the block-cut
                // component families (greater/les/mix), all three
                // output components per anchor point. With a boson
                // bath the MIXED channels ride the same evaluator
                // (anchored vs the stock at m=2; the m=3 anchors:
                // separated==direct 5.4e-6, mixed greater 0.26%, les
                // coupling classes 8/16/64 machine - memory #2575)
                let integ = Integrator::new(self.kt as i64);
                let mut fits = xca_rt::fit_channels_rt_ext(
                    pp_ints,
                    self.h,
                    self.nt,
                    self.rt_xca_poles,
                    xca::W_MAX_DEFAULT,
                );
                // the boson channels (d0 fwd + the row-0-filled
                // d0_cc) share the FORWARD d0 fit (the copy
                // structure)
                if let Some(bath) = &self.boson {
                    let bfit = xca_rt::fit_one_delta_rt(
                        &bath.d0,
                        self.h,
                        self.nt,
                        self.rt_xca_poles,
                        xca::W_MAX_DEFAULT,
                    );
                    fits.push(bfit.clone());
                    fits.push(bfit);
                }
                let beta = self.beta;
                let h = self.h;
                let kt = self.kt as i64;
                let dtau = beta / self.ntau as f64;
                let ntau = self.ntau as i64;
                let ns = self.hs.ns();
                let chan_deltas: Vec<&kadanoffbaym::types::ContourGreen> =
                    if let Some(bath) = &self.boson {
                        let nf = pp_ints.len();
                        let mut v: Vec<&kadanoffbaym::types::ContourGreen> =
                            (0..nf).map(|c| pp_ints[c].delta).collect();
                        v.push(&bath.d0);
                        v.push(&bath.d0_cc);
                        v
                    } else {
                        (0..pp_ints.len()).map(|c| pp_ints[c].delta).collect()
                    };
                let proj: Vec<_> = (0..chan_deltas.len())
                    .map(|c| {
                        xca_rt::fit::csoe_project_all(
                            chan_deltas[c],
                            &fits[c],
                            beta,
                            dtau,
                            h,
                            self.nt,
                            self.ntau,
                        )
                    })
                    .collect();
                use xca_rt::evalfam2::{Anchor, sigma_cuts_m};
                if tstp == 1 && std::env::var_os("NQIST_TTI_FIRST_ROW").is_some() {
                    for (c, (delta, p)) in chan_deltas.iter().zip(&proj).enumerate() {
                        let mirrored = pp_ints.get(c).is_some_and(|line| line.backward)
                            && xca_rt::evalfam2::mirrored_line(0, &proj, &[c]);
                        let mut errors = [0.0_f64; 4];
                        for left in 0..=tstp {
                            for right in 0..=tstp {
                                let expected = xca_rt::les_c(delta, left, right);
                                for (i, coefficients) in [&p.c_les, &p.c_gtr].into_iter().enumerate() {
                                    let actual: Complex = p.poles.iter().zip(coefficients)
                                        .map(|(&w, &coefficient)| coefficient * (-Complex::I * w * h * (right - left) as f64).exp())
                                        .fold(Complex::ZERO, |sum, value| sum + value);
                                    errors[i] = errors[i].max((actual - expected).abs());
                                }
                            }
                            for mu in 0..=ntau {
                                let (mut tv, mut vt) = (Complex::ZERO, Complex::ZERO);
                                for (&w, &coefficient) in p.poles.iter().zip(&p.c_tv) {
                                    let thermal = if mirrored { -w } else { w };
                                    // the projection basis convention (fit.rs doc):
                                    // Delta^tv = c_tv e^{-i w t} e^{-w tau}; the legacy
                                    // e^{+iwt} here was the CONJUGATED phase (the
                                    // reconstruction-error red herring)
                                    let phase = (-Complex::I * w * h * left as f64).exp();
                                    tv += coefficient * phase * (-thermal * dtau * mu as f64).exp();
                                    vt += coefficient.conj() * phase.conj() * (-thermal * dtau * (ntau - mu) as f64).exp();
                                }
                                errors[2] = errors[2].max((tv - xca_rt::tv_c(delta, left, mu)).abs());
                                errors[3] = errors[3].max((vt - xca_rt::vt_c(delta, mu, left)).abs());
                            }
                        }
                        eprintln!("bath reconstruction channel={c} poles={:?} mirrored={mirrored} errors [less, greater-as-less, tv, vt]={errors:?} c_tv={:?} c_les={:?}", p.poles, p.c_tv, p.c_les);
                    }
                }
                let matrix = self.pp_g.iter().any(|g| g.size1 > 1);
                let evaluate = |m, anchor: &Anchor| {
                    if matrix {
                        xca_rt::evalfam2m::sigma_cuts_m_mat(
                            m, &self.hs, pp_ints, self.boson.as_ref(), &proj, &self.pp_g,
                            anchor, tstp, h, beta, dtau, kt, &integ,
                        )
                        .into_iter().map(|block| block.raw().to_vec()).collect::<Vec<_>>()
                    } else {
                        sigma_cuts_m(
                            m, &self.hs, pp_ints, self.boson.as_ref(), &proj, &self.pp_g,
                            anchor, tstp, h, beta, dtau, kt, &integ,
                        )
                        .into_iter().map(|value| vec![value]).collect()
                    }
                };
                // XCA_M3_COMPONENTS (diagnostic): restrict the m>=3
                // increment write-back to a comma-separated subset of
                // ret,less,lmix (default: all three) - drift decomposition
                let comps = std::env::var("XCA_M3_COMPONENTS").unwrap_or_default();
                let want = |c: &str| comps.is_empty() || comps.split(',').any(|x| x == c);
                // XCA_M3_SCALE (diagnostic): multiply the whole m>=3
                // increment by a real factor - linear vs superlinear drift
                // scaling separates a fixed forcing from feedback
                let xscale = Complex::new(
                    std::env::var("XCA_M3_SCALE")
                        .ok()
                        .and_then(|v| v.parse::<f64>().ok())
                        .unwrap_or(1.0),
                    0.0,
                );
                // XCA_M3_SCALE_LMIX (diagnostic): extra real factor on the
                // m>=3 MIXED increment only. If the les increment is right
                // and the mixed one merely too weak in the equation of
                // motion, the drift crosses zero at some factor; a wrong
                // mixed structure never does.
                let xlmix = xscale
                    * Complex::new(
                        std::env::var("XCA_M3_SCALE_LMIX")
                            .ok()
                            .and_then(|v| v.parse::<f64>().ok())
                            .unwrap_or(1.0),
                        0.0,
                    );
                for m in 3..=self.order {
                    let use_tau_boundary = matrix && m == 3 && self.boson.is_none()
                        && std::env::var_os("XCA_M3_LESS_BOUNDARY").is_some();
                    for j in 0..=tstp {
                        let g = evaluate(m, &Anchor::JUpper(j));
                        let l = if j == 0 && use_tau_boundary {
                            let mut blocks = evaluate(m, &Anchor::Tau(ntau));
                            for (si, block) in blocks.iter_mut().enumerate() {
                                let dim = self.pp_g[si].size1;
                                let source = block.clone();
                                for row in 0..dim {
                                    for col in 0..dim {
                                        block[row * dim + col] = -source[col * dim + row].conj();
                                    }
                                }
                            }
                            blocks
                        } else {
                            evaluate(m, &Anchor::JLower(j))
                        };
                        for si in 0..ns {
                            if want("ret") {
                                for (dest, src) in self.pp_sigma[si].ret.raw_block_at_mut(tstp, j).iter_mut().zip(&g[si]) {
                                    *dest += xscale * *src;
                                }
                            }
                            if want("less") {
                                for (dest, src) in self.pp_sigma[si].less.raw_block_at_mut(j, tstp).iter_mut().zip(&l[si]) {
                                    *dest += xscale * *src;
                                }
                            }
                        }
                    }
                    for mu in 0..=ntau {
                        let mx = evaluate(m, &Anchor::Tau(mu));
                        for si in 0..ns {
                            if want("lmix") {
                                for (dest, src) in self.pp_sigma[si].lmix.raw_block_at_mut(tstp, ntau - mu).iter_mut().zip(&mx[si]) {
                                    *dest += xlmix * *src;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// OCA Matsubara self-energy contribution (order = 2), added on
    /// top of the NCA values (blueprint `add_to_pp_self_energy`).
    fn add_sigma_mat_oca(&mut self, pp_ints: &[Hybridization]) {
        let channels = oca::channels_with_bath(pp_ints, self.boson.as_ref());
        let diagrams = oca::build_sigma_diagrams_oca_channels(&self.hs, &channels);
        let integ = Integrator::new(self.kt as i64);
        let hs_ptr = &self.hs;
        let pp_g = &self.pp_g;
        let beta = self.beta;
        let kt = self.kt as i64;
        for d in &diagrams {
            oca::add_sigma_mat_oca(d, pp_g, beta, kt, &integ, &mut self.pp_sigma);
        }
        let _ = hs_ptr;
    }

    /// Matsubara component of the NCA single-particle Green's
    /// functions per GF vertex (blueprint `get_spgf` at `tstp = -1`).
    pub fn get_spgf_mat(&self, verts: &[GfVertex], pp_ints: &[Hybridization]) -> Vec<Vec<Complex>> {
        let _timed = crate::progress::Timed::new("solver", "get_spgf_mat");
        match self.solver {
            Solver::Ppsc => {
                assert!(
                    self.order == 1 || self.order == 2,
                    "PPSC solver supports order 1 (NCA) and 2 (OCA)"
                );
                let mut out = nca::gf_mat(&self.hs, &self.pp_g, verts);
                if self.order == 2 {
                    let integ = Integrator::new(self.kt as i64);
                    let mut oca_part = vec![vec![Complex::ZERO; self.ntau + 1]; verts.len()];
                    oca::add_gf_mat_oca(
                        &self.hs,
                        &self.pp_g,
                        verts,
                        pp_ints,
                        self.boson.as_ref(),
                        self.beta,
                        self.kt as i64,
                        &integ,
                        &mut oca_part,
                    );
                    for (o, add) in out.iter_mut().zip(&oca_part) {
                        for q in 0..o.len() {
                            o[q] += add[q];
                        }
                    }
                }
                out
            }
            Solver::Xca => {
                assert!(self.order >= 1, "XCA solver needs order >= 1");
                let mut out = vec![vec![Complex::ZERO; self.ntau + 1]; verts.len()];
                let xca_channels = oca::channels_with_bath(pp_ints, self.boson.as_ref());
                let fits = if self.order == 1 {
                    // order 1 reads the raw hybridization: no fit
                    xca::unused_fits(xca_channels.len(), self.beta)
                } else {
                    xca::fit_channels_tol(
                        &xca_channels,
                        self.beta,
                        self.ntau,
                        self.xca_max_poles,
                        self.xca_w_max,
                        self.xca_fit_tol,
                    )
                };
                let integ = Integrator::new(self.kt as i64);
                let grids = xca::gf::Grids {
                    ntau: self.ntau,
                    dtau: self.beta / self.ntau as f64,
                    pp_g: &self.pp_g,
                };
                for m in 1..=self.order {
                    xca::gf::add_gf_mat_xca(
                        m,
                        &self.hs,
                        &xca_channels,
                        &fits,
                        verts,
                        &grids,
                        &integ,
                        &mut out,
                    );
                }
                out
            }
            Solver::Ipt => panic!("ImpuritySolver cannot run the IPT solver"),
        }
    }

    /// NCA single-particle Green's functions per GF vertex at
    /// `tstp >= 0` (blueprint `get_spgf`, real-time kernels).
    pub fn get_spgf(
        &self,
        verts: &[GfVertex],
        tstp: i64,
        pp_ints: &[Hybridization],
    ) -> Vec<GreenTimeStep> {
        let _timed = (tstp > self.kt as i64)
            .then(|| crate::progress::Timed::new("solver", format!("get_spgf(tstp={tstp})")));
        let contour = Contour::new(
            self.nt as i64,
            self.ntau as i64,
            1,
            1,
            self.nt as f64 * self.h,
            self.beta,
        );
        let mut out = nca::gf_tstp(&contour, tstp, &self.hs, &self.pp_g, verts);
        // OCA vertex diagrams for every order >= 2. The m >= 3
        // single-particle diagrams are not implemented in real time,
        // so order >= 3 uses the same NCA + OCA closure (the XCA
        // self-energy still enters through `pp_g`).
        if self.order >= 2 {
            let channels = oca::channels_with_bath(pp_ints, self.boson.as_ref());
            let diagrams = oca::build_gf_diagrams_oca_channels(&self.hs, verts, &channels);
            let integ = Integrator::new(self.kt as i64);
            let pp_g = &self.pp_g;
            let beta = self.beta;
            let h = self.h;
            let kt = self.kt as i64;
            if std::env::var("NQIST_SKIP_OCA_GF").is_ok() {
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    eprintln!(
                        "warning: NQIST_SKIP_OCA_GF is set: the real-time OCA vertex-correction kernel is disabled (diagnostic switch)"
                    );
                });
            }
            for d in &diagrams {
                if std::env::var("NQIST_SKIP_OCA_GF").is_err() {
                    oca::add_gf_tstp_oca(tstp, d, pp_g, beta, h, kt, &integ, &mut out[d.vertex]);
                }
            }
        }
        out
    }

    /// Full two-time single-particle Green's function on the real-time
    /// grid: evaluates the per-time row kernels and fills every retarded
    /// row and lesser column. The closure is the same as
    /// [`Self::get_spgf`] (NCA, plus OCA when `order >= 2`). Cost is
    /// O(nt) row evaluations; use [`Self::get_spgf`] for a single time.
    pub fn get_spgf_full(
        &self,
        verts: &[GfVertex],
        pp_ints: &[Hybridization],
        t_end: i64,
    ) -> Vec<GreenTimeStep> {
        assert!(
            t_end >= 0 && t_end <= self.nt as i64,
            "get_spgf_full: t_end out of range"
        );
        let mut full = self.get_spgf(verts, t_end, pp_ints);
        for n in 0..=t_end {
            let row = self.get_spgf(verts, n, pp_ints);
            for (f, r) in full.iter_mut().zip(&row) {
                for j in 0..=n {
                    f.ret.raw_block_at_mut(n, j)[0] = r.ret.raw_block_at(n, j)[0];
                    f.less.raw_block_at_mut(j, n)[0] = r.less.raw_block_at(j, n)[0];
                }
            }
        }
        full
    }

    /// k-th order polynomial extrapolation of the pseudo-particle
    /// rows to `n + 1` (blueprint `extrapolate_ppgf_timestep` with
    /// `kt_extrap = min(n, kt)`; `n = MATSUBARA` maps to `set_t0`).
    pub fn extrapolate_timestep(&mut self, n: i64) {
        if n == MATSUBARA {
            self.set_t0();
            return;
        }
        let k = n.min(self.kt as i64).min(5);
        assert!(
            n >= k && n + 1 <= self.nt as i64,
            "extrapolate_timestep: range"
        );
        let p = extrapolation_weights(&Integrator::new(k));
        for s in 0..self.hs.ns() {
            extrapolate_ppgf_row(n, k, &p, &mut self.pp_g[s]);
        }
    }

    /// One pseudo-particle sweep over the window (blueprint
    /// `pp_step`): for a start-window step `0 <= tstp <= kt` the
    /// window `0..=kt` is solved in one shot, otherwise the single
    /// step `tstp`. Returns the `dens, pp_mu = ...` line on the
    /// Matsubara axis, `None` otherwise.

    #[allow(dead_code)]
    fn debug_sigma_finite_pub(&self, tstp: i64, tag: &str) {
        let msub = tstp == -1;
        for sec in 0..self.hs.ns() {
            let sig = &self.pp_sigma[sec];
            if msub {
                for m in 0..=(sig.ntau) {
                    for z in sig.mat.block(m) {
                        assert!(
                            z.re.is_finite() && z.im.is_finite(),
                            "non-finite sigma mat sec={sec} m={m} [{tag}]"
                        );
                    }
                }
            } else {
                for j in 0..=tstp {
                    for z in sig.ret.raw_block_at(tstp, j) {
                        assert!(
                            z.re.is_finite() && z.im.is_finite(),
                            "non-finite sigma ret sec={sec} n={tstp} j={j} [{tag}]"
                        );
                    }
                }
                for m in 0..=(sig.ntau as i64) {
                    for z in sig.lmix.raw_block_at(tstp, m) {
                        assert!(
                            z.re.is_finite() && z.im.is_finite(),
                            "non-finite sigma lmix sec={sec} n={tstp} m={m} [{tag}]"
                        );
                    }
                }
            }
        }
    }

    pub fn pp_step(&mut self, tstp: i64, pp_ints: &[Hybridization]) -> Option<String> {
        let _timed = crate::progress::Timed::new("solver", format!("pp_step(tstp={tstp})"));
        let (n1, n2) = if tstp <= self.kt as i64 && tstp >= 0 {
            (0, self.kt as i64)
        } else {
            (tstp, tstp)
        };
        for n in n1..=n2 {
            self.update_sigma(n, pp_ints);
            #[cfg(debug_assertions)]
            {
                let tag = format!("pp_step n={n}");
                self.debug_sigma_finite_pub(n, &tag);
            }
        }
        self.solve_dyson(tstp);
        let mut line = None;
        if tstp == MATSUBARA {
            line = Some(self.normalize_ppgf());
        }
        for n in n1..=n2 {
            self.update_density_matrix(n);
        }
        line
    }
}

// ------------------------------------------------------------------
// pseudo-particle extrapolation (blueprint extrapolate_timestep_dispatch)
// ------------------------------------------------------------------

/// Polynomial extrapolation of the ret/less/lmix rows of a
/// pseudo-particle Green's function to the time step `n + 1`, with
/// the extrapolation weights `p` of order `k` and the hermitian
/// fallbacks of the blueprint (`-conj^dag` inside the triangles).
fn extrapolate_ppgf_row(n: i64, k: i64, p: &[f64], g: &mut PseudoGreen) {
    let n1 = n + 1;
    let (s1, s2) = (g.size1, g.size2);
    let b = s1 * s2;
    let mut gtmp = vec![Complex::ZERO; b];

    // lmix (= tv) row
    for m in 0..=(g.ntau as i64) {
        gtmp.fill(Complex::ZERO);
        for l in 0..=k {
            let src = g.lmix.raw_block_at(n - l, m);
            for (t, s) in gtmp.iter_mut().zip(src) {
                *t += *s * p[l as usize];
            }
        }
        g.lmix.raw_block_at_mut(n1, m).copy_from_slice(&gtmp);
    }

    // ret row, fixed relative time: j = 0..=(n - k)
    for j in 0..=(n - k) {
        gtmp.fill(Complex::ZERO);
        for l in 0..=k {
            let src = g.ret.block(g.ret.idx(n - l, n - l - j));
            for (t, s) in gtmp.iter_mut().zip(src) {
                *t += *s * p[l as usize];
            }
        }
        let dst = g.ret.idx(n1, n1 - j);
        g.ret.block_mut(dst).copy_from_slice(&gtmp);
    }

    // ret row, the first k + 1 columns (hermitian fallback)
    for c in 0..=k {
        gtmp.fill(Complex::ZERO);
        for l in 0..=k {
            if n + 1 - l <= c + 1 {
                let src = adjoint_raw(g.ret.block(g.ret.idx(c, n - l)), s1, s2);
                for (t, a) in gtmp.iter_mut().zip(&src) {
                    *t -= *a * p[l as usize];
                }
            } else {
                let src = g.ret.block(g.ret.idx(n - l, c));
                for (t, s) in gtmp.iter_mut().zip(src) {
                    *t += *s * p[l as usize];
                }
            }
        }
        let dst = g.ret.idx(n1, c);
        g.ret.block_mut(dst).copy_from_slice(&gtmp);
    }

    // less, the first k + 1 rows (hermitian fallback)
    for r in 0..=k {
        gtmp.fill(Complex::ZERO);
        for l in 0..=k {
            if n + 1 - l < r + 1 {
                let src = adjoint_raw(g.less.block(g.less.idx(n - l, r)), s1, s2);
                for (t, a) in gtmp.iter_mut().zip(&src) {
                    *t -= *a * p[l as usize];
                }
            } else {
                let src = g.less.block(g.less.idx(r, n - l));
                for (t, s) in gtmp.iter_mut().zip(src) {
                    *t += *s * p[l as usize];
                }
            }
        }
        let dst = g.less.idx(r, n1);
        g.less.block_mut(dst).copy_from_slice(&gtmp);
    }

    // less, fixed relative time rows r = k+1..=n+1
    for r in (k + 1)..=n1 {
        gtmp.fill(Complex::ZERO);
        for l in 0..=k {
            let src = g.less.block(g.less.idx(r - 1 - l, n - l));
            for (t, s) in gtmp.iter_mut().zip(src) {
                *t += *s * p[l as usize];
            }
        }
        let dst = g.less.idx(r, n1);
        g.less.block_mut(dst).copy_from_slice(&gtmp);
    }
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamilton::LocalHamiltonian::SingleBand;
    use crate::hilbert::{SectorClass, build};

    /// Small benchmark-like solver: same beta, U0, mu, eps as the
    /// reference case; the atomic limit is independent of nt/ntau.
    fn bench_solver(nt: usize, ntau: usize, kt: usize) -> ImpuritySolver {
        let hs = build(1, SectorClass::SpinResolved);
        let ham = SingleBand {
            mu: 0.0,
            u: vec![3.0; nt + 2],
            eps_up: vec![0.0; nt + 2],
            eps_do: vec![0.0; nt + 2],
        };
        ImpuritySolver::new(nt, ntau, 5.0, 0.02, kt, 1, hs, ham)
    }

    #[test]
    fn solve_atomic_matches_reference_first_line() {
        // gate: digit-for-digit match with the first line of the
        // archived reference run.log (independent of nt/ntau)
        let mut solver = bench_solver(6, 32, 3);
        let line = solver.solve_atomic();
        let reference = include_str!("../data/ppsc/quench_u35/run.log");
        let first = reference.lines().next().expect("run.log not empty");
        assert_eq!(&line, first);
        // analytic anchor (eigenvalues {0, -1.5, -1.5, 0}, beta = 5)
        let analytic = -1.5 - (2.0 + 2.0 * (-7.5f64).exp()).ln() / 5.0;
        assert!((solver.pp_mu - analytic).abs() < 1e-12);
    }

    #[test]
    fn solve_atomic_mat_fill_clean_grid() {
        // sector 2 (single up, E = -1.5): mat[m] = -exp(-dtau m (E - pp_mu))
        // on the clean grid dtau = beta/ntau, m = 0..=ntau (the C++
        // reference fills the full Matsubara axis, including mat[ntau])
        let ntau = 32;
        let mut solver = bench_solver(6, ntau, 3);
        solver.solve_atomic();
        let dtau = 5.0 / ntau as f64;
        let e = -1.5;
        for m in 0..=ntau {
            let want = -((-dtau) * m as f64 * (e - solver.pp_mu)).exp();
            let got = solver.pp_g[2].mat.block(m)[0].re;
            assert!((got - want).abs() < 1e-14, "m = {m}: {got} vs {want}");
        }
        // mat[ntau] carries the full beta Boltzmann weight
        let want = -((-5.0) * (e - solver.pp_mu)).exp();
        assert!((solver.pp_g[2].mat.block(ntau)[0].re - want).abs() < 1e-14);
        // mat[0] = -1 in every sector (V diag(-1) V^dag = -1)
        for s in 0..4 {
            let d = solver.pp_g[s].size1;
            for i in 0..d {
                assert_eq!(
                    solver.pp_g[s].mat.block(0)[i * (d + 1)],
                    Complex::new(-1.0, 0.0)
                );
            }
        }
    }

    #[test]
    fn normalize_rescales_mat_and_shifts_pp_mu() {
        // hand-set Matsubara components: single sector value c = -2
        let ntau = 16;
        let mut solver = bench_solver(4, ntau, 3);
        for s in 0..solver.hs.ns() {
            let d = solver.pp_g[s].size1;
            for m in 0..=ntau {
                for i in 0..d {
                    solver.pp_g[s].mat.block_mut(m)[i * (d + 1)] = Complex::new(-2.0, 0.0);
                }
            }
        }
        // dens = sum_s tr(-mat[ntau]) = 4 sectors * 1 * 2 = 8
        let pp_mu_before = solver.pp_mu;
        let line = solver.normalize_ppgf();
        let dens: f64 = 8.0;
        let dlam = -dens.ln() / 5.0;
        let dtau = 5.0 / ntau as f64;
        assert_eq!(
            line,
            format!(
                "dens, pp_mu = 8, {}",
                crate::io::fmt_g6(pp_mu_before + dlam)
            )
        );
        assert!((solver.pp_mu - (pp_mu_before + dlam)).abs() < 1e-15);
        // mat[m] = -2 * exp(dlam * dtau * m) on the full axis m = 0..=ntau
        for m in 0..=ntau {
            let want = -2.0 * (dlam * dtau * m as f64).exp();
            assert!((solver.pp_g[2].mat.block(m)[0].re - want).abs() < 1e-14);
        }
        // post-normalization invariant: sum of traces of rho(-1) is 1
        solver.update_density_matrix(-1);
        let tr: f64 = solver.rho[0].iter().map(|m| m.trace().re).sum();
        assert!((tr - 1.0).abs() < 1e-12);
    }

    #[test]
    fn density_matrix_branches() {
        // Matsubara branch: rho = -mat[ntau]
        let ntau = 8;
        let mut solver = bench_solver(4, ntau, 3);
        solver.pp_g[2]
            .mat
            .block_mut(ntau)
            .copy_from_slice(&[Complex::new(0.25, -0.5)]);
        solver.update_density_matrix(-1);
        assert_eq!(solver.rho[0][2].get(0, 0), Complex::new(-0.25, 0.5));
        // real-time branch: rho = i * sig * les(t, t); sector 2 has
        // Fermi sign (-1) -> rho = -i * les
        solver.pp_g[2]
            .less
            .raw_block_at_mut(2, 2)
            .copy_from_slice(&[Complex::new(1.0, 1.0)]);
        solver.update_density_matrix(2);
        assert_eq!(solver.rho[3][2].get(0, 0), Complex::new(1.0, -1.0));
        // Bose sector 0 (+1): rho = +i * les
        solver.pp_g[0]
            .less
            .raw_block_at_mut(2, 2)
            .copy_from_slice(&[Complex::new(1.0, 1.0)]);
        solver.update_density_matrix(2);
        assert_eq!(solver.rho[3][0].get(0, 0), Complex::new(-1.0, 1.0));
    }

    #[test]
    fn set_t0_wiring_from_atomic_mat() {
        let mut solver = bench_solver(4, 16, 3);
        solver.solve_atomic();
        solver.set_t0();
        // pseudo-particle t0 map per sector: lmix(0,m) = i*sig*mat[ntau-m],
        // less(0,0) = lmix(0,0), ret(0,0) = i*mat[0] (no -less correction)
        for s in 0..solver.hs.ns() {
            let g = &solver.pp_g[s];
            let sig = Complex::new(0.0, g.sign.as_i64() as f64);
            for m in [0usize, 5, 16] {
                let want = sig * g.mat.block(g.ntau - m)[0];
                assert!(
                    (g.lmix.raw_block_at(0, m as i64)[0] - want).abs() < 1e-14,
                    "s={s} m={m}"
                );
            }
            assert_eq!(g.less.raw_block_at(0, 0)[0], g.lmix.raw_block_at(0, 0)[0]);
            assert_eq!(g.ret.raw_block_at(0, 0)[0], Complex::I * g.mat.block(0)[0]);
        }
    }

    #[test]
    fn pp_step_matsubara_atomic_fixed_point() {
        // Zero self-energy: the Matsubara solve reproduces the atomic
        // Boltzmann form on the clean grid, and because the pp_mu of
        // solve_atomic satisfies sum_E exp(-beta (E - pp_mu)) = 1 as an
        // identity, the very first normalization is already at the
        // fixed point: dens = 1 exactly and pp_mu is unchanged (the
        // C++ reference prints "dens, pp_mu = 1, -1.63874" for
        // iteration 1 of the archived run.log).
        let nt = 6;
        // benchmark ntau: the dyson discretization error (~1e-13 here)
        // must stay below the identity tolerance
        let ntau = 250;
        let kt = 3;
        let mut solver = bench_solver(nt, ntau, kt);
        solver.solve_atomic();
        let pp_mu0 = solver.pp_mu;
        let line1 = solver.pp_step(-1, &[]).expect("matsubara line");
        assert_eq!(line1, "dens, pp_mu = 1, -1.63874");
        // dlam vanished: pp_mu is unchanged
        assert!((solver.pp_mu - pp_mu0).abs() < 1e-9);
        // total density matrix trace is exactly one after normalization
        let tr: f64 = solver.rho[0].iter().map(|m| m.trace().re).sum();
        assert!((tr - 1.0).abs() < 1e-12);

        // clean-grid Boltzmann form on the full Matsubara axis,
        // including the beta point m = ntau
        let dtau = 5.0 / ntau as f64;
        let e = -1.5;
        for m in [0usize, 1, 7, ntau - 1, ntau] {
            // mat[m] = -exp(-tau_m (E - pp_mu)), tau_m = dtau * m
            let exponent = -dtau * m as f64 * (e - solver.pp_mu);
            let want = -exponent.exp();
            let got = solver.pp_g[2].mat.block(m)[0].re;
            assert!((got - want).abs() < 1e-9, "m = {m}: {got} vs {want}");
        }

        // a second sweep stays at the fixed point
        let line2 = solver.pp_step(-1, &[]).expect("matsubara line");
        assert_eq!(line2, "dens, pp_mu = 1, -1.63874");
        assert!((solver.pp_mu - pp_mu0).abs() < 1e-9);
    }

        /// Equilibrium XCA(3) + Holstein: (1) the Matsubara G path must
    /// accept the boson channels - the old plumbing fitted 4 fermion
    /// + 2 boson channels but passed the fermion-only slice into
    /// `add_gf_mat_xca`, whose `one SOE fit per channel` assertion
    /// panicked; (2) the boson lines must actually enter the m >= 3
    /// XCA diagrams, not only the OCA part. The isolated m = 3 boson
    /// content is (G3_b - G2_b) - (G3_n - G2_n) at one fixed pp_g.
    /// Structural only: the XCA boson-pair calibration is still open
    /// (see `xca_holstein_m2_anchor_matches_oca_boson`).
    #[test]
    fn equilibrium_xca_m3_holstein_gf_smoke() {
        use crate::nca::BosonBath;
        use kadanoffbaym::types::ContourGreen;

        let (nt, ntau, beta, h) = (2usize, 8usize, 5.0f64, 0.05f64);
        let make = |order: usize| {
            ImpuritySolver::new(
                nt,
                ntau,
                beta,
                h,
                3,
                order,
                build(1, SectorClass::SpinResolved),
                SingleBand {
                    mu: 0.0,
                    u: vec![3.0; nt + 2],
                    eps_up: vec![0.0; nt + 2],
                    eps_do: vec![0.0; nt + 2],
                },
            )
        };
        let contour = Contour::new(nt as i64, ntau as i64, 1, 1, nt as f64 * h, beta);
        let dtau = beta / ntau as f64;
        let mut delta = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        for m in 0..=ntau {
            delta.mat.block_mut(m)[0] =
                Complex::new(-0.4 * (-0.5 * dtau * m as f64).exp(), 0.0);
        }
        let mut delta_cc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        crate::dmft::set_bwd_from_fwd_mat(&mut delta_cc, &delta);

        let mut s3b = make(3);
        let mut d0 = crate::boson::green_from_eps_phonon(nt, ntau, beta, h, 1.0);
        crate::boson::scale_phonon(&mut d0, &vec![0.3; nt + 2]);
        let mut d0_cc = ContourGreen::new(&contour, Sign::Bose, Complex::ZERO);
        crate::boson::set_bwd_from_fwd_boson_mat(&mut d0_cc, &d0);
        let blocks = crate::boson::docc_blocks(&s3b.hs);
        s3b.set_boson_bath(Some(BosonBath { d0, d0_cc, blocks }));
        s3b.solve_atomic();
        let pp_ints = crate::dmft::get_pp_ints(&delta, &delta_cc, &s3b);
        let verts = crate::dmft::get_gf_verts(&s3b);
        let _ = s3b.pp_step(MATSUBARA, &pp_ints);
        let g3b = s3b.get_spgf_mat(&verts, &pp_ints);
        assert!(
            g3b.iter()
                .flatten()
                .all(|z| z.re.is_finite() && z.im.is_finite()),
            "non-finite G"
        );
        let max = g3b
            .iter()
            .flatten()
            .map(|z| z.abs())
            .fold(0.0f64, f64::max);
        assert!(max > 1e-6, "empty G");

        // same pp_g, bath removed: the XCA m >= 3 path without boson
        let bath = s3b.boson.take();
        let g3n = s3b.get_spgf_mat(&verts, &pp_ints);
        let mut s2b = make(2);
        s2b.boson = bath;
        s2b.pp_g = s3b.pp_g.clone();
        let g2b = s2b.get_spgf_mat(&verts, &pp_ints);
        let mut s2n = make(2);
        s2n.pp_g = s3b.pp_g.clone();
        let g2n = s2n.get_spgf_mat(&verts, &pp_ints);

        // isolate the m = 3 XCA boson content (the OCA boson part is
        // common to g3b/g2b and cancels)
        let mut xca3_scale = 0.0f64;
        let mut boson_m3 = 0.0f64;
        for i in 0..g3b[0].len() {
            xca3_scale = xca3_scale.max((g3b[0][i] - g2b[0][i]).abs());
            let value = (g3b[0][i] - g2b[0][i]) - (g3n[0][i] - g2n[0][i]);
            boson_m3 = boson_m3.max(value.abs());
        }
        eprintln!(
            "[xca m3 holstein] XCA3 increment {xca3_scale:.3e}, boson m=3 content {boson_m3:.3e}"
        );
        assert!(xca3_scale > 1e-6, "XCA(3) increment is inert");
        assert!(
            boson_m3 > 1e-6,
            "boson channels do not enter the XCA m>=3 G: {boson_m3:.3e}"
        );
    }

    /// Real-time single-particle G closure: at order >= 3 the OCA
    /// vertex diagrams must still be included. Previously only
    /// `order == 2` took the branch, so order 3 silently dropped the
    /// OCA GF that order 2 had.
    #[test]
    fn realtime_gf_closure_same_for_order2_and_order3() {
        use kadanoffbaym::types::ContourGreen;

        let (nt, ntau, beta, h, kt) = (6usize, 12usize, 5.0f64, 0.05f64, 3usize);
        let make = |order: usize| {
            ImpuritySolver::new(
                nt,
                ntau,
                beta,
                h,
                kt,
                order,
                build(1, SectorClass::SpinResolved),
                SingleBand {
                    mu: 0.0,
                    u: vec![0.0; nt + 2],
                    eps_up: vec![-1.0; nt + 2],
                    eps_do: vec![0.0; nt + 2],
                },
            )
        };

        let contour = Contour::new(nt as i64, ntau as i64, 1, 1, nt as f64 * h, beta);
        let dtau = beta / ntau as f64;
        let fb = |eps: f64| 1.0 / (1.0 + (beta * eps).exp());
        let bath: [(f64, f64); 2] = [(0.25, 0.5), (0.16, -0.8)];
        let mut d1 = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        let mut d1c = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        for n in 0..=nt as i64 {
            for j in 0..=n {
                let dt = h * (n - j) as f64;
                let (mut g, mut l) = (Complex::ZERO, Complex::ZERO);
                for &(v2, eps) in &bath {
                    let e = (-Complex::I * eps * dt).exp();
                    g += Complex::new(0.0, -1.0) * v2 * (1.0 - fb(eps)) * e;
                    l += Complex::new(0.0, 1.0) * v2 * fb(eps) * e;
                }
                d1.ret.block_mut(d1.ret.idx(n, j))[0] = g;
                d1.less.block_mut(d1.less.idx(j, n))[0] = l;
                d1c.ret.block_mut(d1c.ret.idx(n, j))[0] = l;
                d1c.less.block_mut(d1c.less.idx(j, n))[0] = g;
            }
            for m in 0..=(ntau as i64) {
                let tau = dtau * m as f64;
                let mut tv = Complex::ZERO;
                for &(v2, eps) in &bath {
                    let e = (-Complex::I * eps * h * n as f64).exp();
                    tv += Complex::new(0.0, 1.0)
                        * v2
                        * (1.0 - fb(eps))
                        * e
                        * Complex::new((-eps * tau).exp(), 0.0);
                }
                d1.lmix.raw_block_at_mut(n, m)[0] = tv;
                d1c.lmix.raw_block_at_mut(n, m)[0] = tv.conj();
            }
        }
        for i in 0..=ntau {
            let t = dtau * i as f64;
            let mut val = Complex::ZERO;
            for &(v2, eps) in &bath {
                val += Complex::new(
                    -v2 * ((-(beta - t) * eps).exp() / (1.0 + (-beta * eps).exp())),
                    0.0,
                );
            }
            d1.mat.block_mut(i)[0] = val;
        }
        for i in 0..=ntau {
            let fwd = d1.mat.block(ntau - i)[0];
            d1c.mat.block_mut(i)[0] = -fwd;
        }

        let hs = build(1, SectorClass::SpinResolved);
        let cua = hs.flavor(0, 0, false);
        let cuc = hs.flavor(0, 0, true);
        let cda = hs.flavor(0, 1, false);
        let cdc = hs.flavor(0, 1, true);
        let pp_ints = [
            Hybridization {
                delta: &d1,
                op_cre_fl: cuc,
                op_ann_fl: cua,
                fermion: true,
                backward: false,
            },
            Hybridization {
                delta: &d1c,
                op_cre_fl: cua,
                op_ann_fl: cuc,
                fermion: true,
                backward: true,
            },
            Hybridization {
                delta: &d1,
                op_cre_fl: cdc,
                op_ann_fl: cda,
                fermion: true,
                backward: false,
            },
            Hybridization {
                delta: &d1c,
                op_cre_fl: cda,
                op_ann_fl: cdc,
                fermion: true,
                backward: true,
            },
        ];

        let mut s2 = make(2);
        s2.solve_atomic();
        for _ in 0..4 {
            s2.pp_step(MATSUBARA, &pp_ints);
        }
        s2.set_t0();
        for tstp in 0..=2i64 {
            s2.pp_step(tstp, &pp_ints);
        }

        let mut s1 = make(1);
        s1.pp_g = s2.pp_g.clone();
        let mut s3 = make(3);
        s3.pp_g = s2.pp_g.clone();

        let verts = crate::dmft::get_gf_verts(&s2);
        let g1 = s1.get_spgf(&verts, 2, &pp_ints);
        let g2 = s2.get_spgf(&verts, 2, &pp_ints);
        let g3 = s3.get_spgf(&verts, 2, &pp_ints);

        let max_ret_les = |a: &GreenTimeStep, b: &GreenTimeStep| {
            let n = a.tstp as i64;
            let mut d = 0.0f64;
            for j in 0..=n {
                for (x, y) in a.ret.raw_block_at(n, j).iter().zip(b.ret.raw_block_at(n, j)) {
                    d = d.max((*x - *y).abs());
                }
                for (x, y) in a.less.raw_block_at(j, n).iter().zip(b.less.raw_block_at(j, n)) {
                    d = d.max((*x - *y).abs());
                }
            }
            d
        };
        let oca = max_ret_les(&g1[0], &g2[0]);
        assert!(oca > 1e-8, "order-2 OCA GF closure is inert: {oca:.3e}");
        assert_eq!(
            g2, g3,
            "order >= 3 must take the same closure as order = 2 (all components)"
        );
    }
}
