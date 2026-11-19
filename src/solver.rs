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
use kadanoffbaym::basics::set_t0_from_mat_pseudo;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::dyson::{pseudo_dyson_mat, pseudo_dyson_start, pseudo_dyson_timestep};
use kadanoffbaym::linalg::{DenseMat, eigh};
use kadanoffbaym::math;
use kadanoffbaym::types::{Contour, GreenTimeStep, MATSUBARA, PseudoGreen, Sign, TimeFunction};
use kadanoffbaym::weights::Integrator;

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
pub struct ImpuritySolver {
    pub nt: usize,
    pub ntau: usize,
    pub beta: f64,
    pub h: f64,
    /// Order of the strong-coupling expansion: 1 = NCA, 2 = OCA.
    pub kt: usize,
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
    /// Optional static boson bath (Holstein/Hirsch vertex); NCA only
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

    /// Attach (or detach) a static boson bath for Holstein/Hirsch
    /// calculations. The bath carries the coupling-scaled free phonon
    /// propagators and the diagonal vertex blocks; it contributes
    /// additional NCA diagrams to the pseudo-particle self-energies
    /// in [`Self::update_sigma`].
    pub fn set_boson_bath(&mut self, bath: Option<crate::nca::BosonBath>) {
        if bath.is_some() {
            assert!(
                self.order <= 2,
                "set_boson_bath: the Holstein boson bath supports NCA/OCA (orders 1-2) only"
            );
        }
        self.boson = bath;
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
                );
            }
        }
    }

    /// Refresh the pseudo-particle self-energies at `tstp` from the
    /// NCA diagrams (blueprint `update_sigma`): Matsubara axis or
    /// real-time kernels, dispatched on `tstp`.
    pub fn update_sigma(&mut self, tstp: i64, pp_ints: &[Hybridization]) {
        if tstp == MATSUBARA {
            nca::set_sigma_mat(&self.hs, &self.pp_g, pp_ints, &mut self.pp_sigma);
            if let Some(bath) = &self.boson {
                nca::add_boson_sigma_mat(bath, &self.pp_g, &mut self.pp_sigma);
            }
            // native OCA at order 2 for every band count (the kernels
            // are dimension-generic; the scalar fast path covers the
            // single-band sectors); XCA crossing topologies add orders
            // 3..=order on top
            if self.order >= 2 {
                self.add_sigma_mat_oca(pp_ints);
            }
            let xca_lo = 3;
            if self.order >= xca_lo {
                // XCA crossing topologies of orders 3..=order
                // (SOE-separated backbones on the Matsubara grid)
                let xca_channels = oca::channels_with_bath(pp_ints, self.boson.as_ref());
                let fits = xca::fit_channels(
                    &xca_channels,
                    self.beta,
                    self.ntau,
                    xca::MAX_POLES_DEFAULT,
                    xca::W_MAX_DEFAULT,
                );
                let integ = Integrator::new(self.kt as i64);
                let grids = xca::sigma::Grids {
                    ntau: self.ntau,
                    dtau: self.beta / self.ntau as f64,
                    pp_g: &self.pp_g,
                };
                for m in xca_lo..=self.order {
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
        } else {
            assert!(
                self.order <= 2,
                "update_sigma: real-time steps support NCA/OCA only; \
                 XCA orders >= 3 are Matsubara-axis only"
            );
            nca::set_sigma_tstp(tstp, &self.hs, &self.pp_g, pp_ints, &mut self.pp_sigma);
            if let Some(bath) = &self.boson {
                nca::add_boson_sigma_tstp(tstp, bath, &self.pp_g, &mut self.pp_sigma);
            }
            if self.order == 2 {
                let channels = oca::channels_with_bath(pp_ints, self.boson.as_ref());
                let diagrams = oca::build_sigma_diagrams_oca_channels(&self.hs, &channels);
                let integ = Integrator::new(self.kt as i64);
                let pp_g = &self.pp_g;
                let beta = self.beta;
                let h = self.h;
                let kt = self.kt as i64;
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
        let mut out = nca::gf_mat(&self.hs, &self.pp_g, verts);
        let xca_lo = if self.order >= 2 {
            {
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
                3
            }
        } else if self.order >= 1 {
            3
        } else {
            1
        };
        if self.order >= xca_lo {
            let xca_channels = oca::channels_with_bath(pp_ints, self.boson.as_ref());
            let fits = xca::fit_channels(
                &xca_channels,
                self.beta,
                self.ntau,
                xca::MAX_POLES_DEFAULT,
                xca::W_MAX_DEFAULT,
            );
            let integ = Integrator::new(self.kt as i64);
            let grids = xca::gf::Grids {
                ntau: self.ntau,
                dtau: self.beta / self.ntau as f64,
                pp_g: &self.pp_g,
            };
            let mut xca_part = vec![vec![Complex::ZERO; self.ntau + 1]; verts.len()];
            for m in xca_lo..=self.order {
                xca::gf::add_gf_mat_xca(
                    m,
                    &self.hs,
                    pp_ints,
                    &fits,
                    verts,
                    &grids,
                    &integ,
                    &mut xca_part,
                );
            }
            for (o, add) in out.iter_mut().zip(&xca_part) {
                for q in 0..o.len() {
                    o[q] += add[q];
                }
            }
        }
        out
    }

    /// NCA single-particle Green's functions per GF vertex at
    /// `tstp >= 0` (blueprint `get_spgf`, real-time kernels).
    pub fn get_spgf(
        &self,
        verts: &[GfVertex],
        tstp: i64,
        pp_ints: &[Hybridization],
    ) -> Vec<GreenTimeStep> {
        let contour = Contour::new(
            self.nt as i64,
            self.ntau as i64,
            1,
            1,
            self.nt as f64 * self.h,
            self.beta,
        );
        let mut out = nca::gf_tstp(&contour, tstp, &self.hs, &self.pp_g, verts);
        if self.order == 2 {
            let channels = oca::channels_with_bath(pp_ints, self.boson.as_ref());
            let diagrams = oca::build_gf_diagrams_oca_channels(&self.hs, verts, &channels);
            let integ = Integrator::new(self.kt as i64);
            let pp_g = &self.pp_g;
            let beta = self.beta;
            let h = self.h;
            let kt = self.kt as i64;
            for d in &diagrams {
                if std::env::var("NQIST_SKIP_OCA_GF").is_err() {
                    oca::add_gf_tstp_oca(tstp, d, pp_g, beta, h, kt, &integ, &mut out[d.vertex]);
                }
            }
        }
        out
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
        let p = math::extrapolation_weights(&Integrator::new(k));
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
}
