// Project : nQIST
// File    : src/ipt.rs
// Purpose : IPT (bare second-order weak-coupling) impurity solver,
//           blueprint Tsuji & Werner PRB 88, 165115 (2013)
//           (arXiv:1306.0307v2) Sec. III and V.

use kadanoffbaym::complex::Complex;
use kadanoffbaym::matsubara::MatsubaraMethod;
use kadanoffbaym::dyson::{dyson_mat, dyson_start, dyson_timestep};
use kadanoffbaym::types::{Contour, ContourGreen, MATSUBARA, Sign, TimeFunction};
use kadanoffbaym::weights::Integrator;

/// IPT impurity solver (single-band Hubbard impurity on the Bethe
/// lattice, `Delta_sigma = v*^2 G_sigma_bar`, scalar 1x1 blocks).
///
/// Bare second-order expansion in the Weiss Green's function `G0`
/// (paper Eq. 31),
/// `Sigma_sigma(t,t') = U(t) U(t') G0_sigma(t,t')
/// G0_sbar(t',t) G0_sbar(t,t')`,
/// plus the bare Hartree tadpole `h_sigma(t) = U(t)(n0_sbar(t) -
/// alpha_sbar)` (Eq. 39), which enters the Dyson solvers as an
/// instantaneous single-particle term (never as a `ret[n,n]` sample
/// of Sigma).
///
/// The Dyson solves fold the hybridization into the self-energy
/// argument (`Sigma_KB = Delta [+ Sigma_sunset]`) and carry the
/// one-particle term in `H` with `mu_kb = 0`:
/// `G0^{-1} = idt + mu - U alpha_sbar - Delta`,
/// `G^{-1}  = idt + mu - U n0_sbar - Delta - Sigma_sunset` (the
/// alpha shift cancels against the bare tadpole).
#[derive(Clone)]
pub struct IptSolver {
    pub nt: usize,
    pub ntau: usize,
    pub kt: usize,
    pub beta: f64,
    /// real-time step `dt`
    pub h: f64,
    pub mu: f64,
    /// interaction-symmetrization constants `(alpha_up, alpha_dn)`
    /// (paper Eq. 21-22; `1/2` at half filling)
    pub alpha: (f64, f64),
    /// interaction array of length `nt + 2`: `u[0]` is the initial
    /// (Matsubara-branch) value, `u[tstp + 1]` the value at `tstp`
    pub u: Vec<f64>,
    /// squared Bethe hybridization `v*^2` (`t* = 1`)
    pub vstar2: f64,
    /// Weiss Green's functions (spin up/down)
    pub g0_up: ContourGreen,
    pub g0_dn: ContourGreen,
    /// interacting impurity Green's functions
    pub g_up: ContourGreen,
    pub g_dn: ContourGreen,
    /// bare second-order (sunset) self-energies
    pub sigma_up: ContourGreen,
    pub sigma_dn: ContourGreen,
    /// hybridizations `Delta_sigma = v*^2 G_sigma_bar`
    pub delta_up: ContourGreen,
    pub delta_dn: ContourGreen,
    /// `Delta + Sigma_sunset` per spin, maintained incrementally (the
    /// Dyson self-energy argument for `G`)
    sigma_tot_up: ContourGreen,
    sigma_tot_dn: ContourGreen,
    /// Bose chi-bubble scratch objects for the sunset kernels
    w_up: ContourGreen,
    w_dn: ContourGreen,
    /// Hartree-density source: `false` = bold (interacting `n`),
    /// `true` = bare (Weiss `n0`); both are examined in the paper's
    /// Table 1 (default `false`)
    pub bare_tadpole: bool,
}

impl IptSolver {
    /// New solver with zero initial state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nt: usize,
        ntau: usize,
        beta: f64,
        h: f64,
        kt: usize,
        mu: f64,
        u: Vec<f64>,
        alpha: (f64, f64),
    ) -> IptSolver {
        assert_eq!(
            u.len(),
            nt + 2,
            "interaction array must have nt + 2 entries"
        );
        let contour = Contour::new(nt as i64, ntau as i64, 1, 1, nt as f64 * h, beta);
        let mk = || ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        IptSolver {
            nt,
            ntau,
            kt,
            beta,
            h,
            mu,
            alpha,
            u,
            vstar2: 1.0,
            g0_up: mk(),
            g0_dn: mk(),
            g_up: mk(),
            g_dn: mk(),
            sigma_up: mk(),
            sigma_dn: mk(),
            delta_up: mk(),
            delta_dn: mk(),
            sigma_tot_up: mk(),
            sigma_tot_dn: mk(),
            w_up: ContourGreen::new(&contour, Sign::Bose, Complex::ZERO),
            w_dn: ContourGreen::new(&contour, Sign::Bose, Complex::ZERO),
            bare_tadpole: false,
        }
    }

    /// Interaction at real-time step `tstp >= 0` (`u[tstp + 1]`).
    #[inline]
    pub fn u_at(&self, tstp: i64) -> f64 {
        u_at(&self.u, tstp)
    }

    /// Interaction on the Matsubara branch (`u[0]`).
    #[inline]
    pub fn u_init(&self) -> f64 {
        self.u[0]
    }

    /// Bare tadpole density `n0_sbar(t) = -i G0_sbar^<(t,t)` (real
    /// branch; `G^<(t,t) = i n`).
    #[inline]
    fn n0_of(g0_bar: &ContourGreen, tstp: i64) -> f64 {
        g0_bar.less.raw_block_at(tstp, tstp)[0].im
    }

    /// `TimeFunction` for the Weiss solve, `H_g0(t) = U alpha_sbar -
    /// mu` (alpha-shifted level, no Hartree).
    fn h_g0(&self, _up: bool) -> TimeFunction {
        // PPSC-consistent Weiss: no alpha shift (G0^{-1} = idt + mu -
        // Delta); at half filling the Weiss density is then 1/2
        // exactly, which is the self-consistent particle-hole-
        // symmetric point of the DMFT loop
        let mut f = TimeFunction::zeros(self.nt, 1, 1);
        let c = Complex::new(self.mu, 0.0);
        f.block_mut(MATSUBARA)[0] = c;
        for tstp in 0..=(self.nt as i64) {
            f.block_mut(tstp)[0] = c;
        }
        f
    }

    /// `TimeFunction` for the impurity solve: `H_g(t) = mu -
    /// U(t) n_bar(t)` with the interacting (bold) density
    /// `n_bar = -i G_bar^<(t,t)`; at half filling this gives the
    /// `-U/2` level of the PPSC Hamiltonian bookkeeping.
    fn h_g(&self, up: bool) -> TimeFunction {
        // density feeding the Hartree: bold (interacting) or bare
        // (Weiss), see Table 1 of the paper
        let g_bar = if up { &self.g_dn } else { &self.g_up };
        let g0_bar = if up { &self.g0_dn } else { &self.g0_up };
        let src = if self.bare_tadpole { g0_bar } else { g_bar };
        let n_init = -src.mat.block(self.ntau)[0].re;
        let mut f = TimeFunction::zeros(self.nt, 1, 1);
        f.block_mut(MATSUBARA)[0] = Complex::new(self.mu - self.u_init() * n_init, 0.0);
        for tstp in 0..=(self.nt as i64) {
            let n = Self::n0_of(src, tstp);
            f.block_mut(tstp)[0] = Complex::new(self.mu - self.u_at(tstp) * n, 0.0);
        }
        f
    }

    fn refresh_sigma_tot_mat(&mut self) {
        for (tot, d, s) in [
            (&mut self.sigma_tot_up, &self.delta_up, &self.sigma_up),
            (&mut self.sigma_tot_dn, &self.delta_dn, &self.sigma_dn),
        ] {
            for m in 0..=self.ntau {
                tot.mat.block_mut(m)[0] = d.mat.block(m)[0] + s.mat.block(m)[0];
            }
        }
    }

    /// Refresh `sigma_tot` row `tstp` (ret/les/tv).
    fn refresh_sigma_tot_row(&mut self, tstp: i64) {
        for (tot, d, s) in [
            (&mut self.sigma_tot_up, &self.delta_up, &self.sigma_up),
            (&mut self.sigma_tot_dn, &self.delta_dn, &self.sigma_dn),
        ] {
            for j in 0..=tstp {
                tot.ret.raw_block_at_mut(tstp, j)[0] =
                    d.ret.raw_block_at(tstp, j)[0] + s.ret.raw_block_at(tstp, j)[0];
                tot.less.raw_block_at_mut(j, tstp)[0] =
                    d.less.raw_block_at(j, tstp)[0] + s.less.raw_block_at(j, tstp)[0];
            }
            for m in 0..=(self.ntau as i64) {
                tot.lmix.raw_block_at_mut(tstp, m)[0] =
                    d.lmix.raw_block_at(tstp, m)[0] + s.lmix.raw_block_at(tstp, m)[0];
            }
        }
    }

    /// Fill the sunset Matsubara components for both spins.
    pub fn sigma_bare_second_mat(&mut self) {
        self.sigma_bare_second(MATSUBARA);
    }

    /// Fill the sunset rows `tstp` (`MATSUBARA` for the mat axis) for
    /// both spins.
    pub fn sigma_bare_second(&mut self, tstp: i64) {
        let (ntau, u) = (self.ntau, &self.u);
        let Self {
            g0_up,
            g0_dn,
            sigma_up,
            sigma_dn,
            w_up,
            w_dn,
            ..
        } = self;
        sigma_second_spin(tstp, ntau, u, g0_up, g0_dn, w_up, sigma_up);
        sigma_second_spin(tstp, ntau, u, g0_dn, g0_up, w_dn, sigma_dn);
    }

    /// Solve the Weiss Matsubara components from the current baths.
    pub fn solve_g0_mat(&mut self) {
        let h_up = self.h_g0(true);
        let h_dn = self.h_g0(false);
        let Self {
            beta,
            kt,
            delta_up,
            delta_dn,
            g0_up,
            g0_dn,
            ..
        } = self;
        dyson_mat(
            g0_up,
            delta_up,
            0.0,
            &h_up,
            *beta,
            *kt as i64,
            MatsubaraMethod::Fixpoint,
            true,
            None,
            1e-12,
        );
        dyson_mat(
            g0_dn,
            delta_dn,
            0.0,
            &h_dn,
            *beta,
            *kt as i64,
            MatsubaraMethod::Fixpoint,
            true,
            None,
            1e-12,
        );
    }

    /// Solve the impurity Matsubara components
    /// (`G = dyson(Delta + Sigma, H_g)`).
    pub fn solve_g_mat(&mut self) {
        self.sigma_bare_second_mat();
        self.refresh_sigma_tot_mat();
        let h_up = self.h_g(true);
        let h_dn = self.h_g(false);
        let Self {
            beta,
            kt,
            sigma_tot_up,
            sigma_tot_dn,
            g_up,
            g_dn,
            ..
        } = self;
        dyson_mat(
            g_up,
            sigma_tot_up,
            0.0,
            &h_up,
            *beta,
            *kt as i64,
            MatsubaraMethod::Fixpoint,
            true,
            None,
            1e-12,
        );
        dyson_mat(
            g_dn,
            sigma_tot_dn,
            0.0,
            &h_dn,
            *beta,
            *kt as i64,
            MatsubaraMethod::Fixpoint,
            true,
            None,
            1e-12,
        );
    }

    /// One pass of the start-window solves (`tstp = 0..=kt`): Weiss
    /// rows, sunset rows, impurity rows.
    pub fn solve_start_rows(&mut self) {
        let h_g0_up = self.h_g0(true);
        let h_g0_dn = self.h_g0(false);
        {
            let Self {
                beta,
                h,
                kt,
                delta_up,
                delta_dn,
                g0_up,
                g0_dn,
                ..
            } = self;
            dyson_start(g0_up, 0.0, &h_g0_up, delta_up, *beta, *h, *kt as i64);
            dyson_start(g0_dn, 0.0, &h_g0_dn, delta_dn, *beta, *h, *kt as i64);
        }
        self.sigma_bare_second_mat();
        for n in 0..=(self.kt as i64) {
            self.sigma_bare_second(n);
        }
        self.refresh_sigma_tot_mat();
        for tstp in 0..=(self.kt as i64) {
            self.refresh_sigma_tot_row(tstp);
        }
        let h_g_up = self.h_g(true);
        let h_g_dn = self.h_g(false);
        let Self {
            beta,
            h,
            kt,
            sigma_tot_up,
            sigma_tot_dn,
            g_up,
            g_dn,
            ..
        } = self;
        dyson_start(g_up, 0.0, &h_g_up, sigma_tot_up, *beta, *h, *kt as i64);
        dyson_start(g_dn, 0.0, &h_g_dn, sigma_tot_dn, *beta, *h, *kt as i64);
    }

    /// One pass of the real-time row `tstp > kt` solves: sunset row
    /// (from the current Weiss rows), Weiss row, sunset row (fresh),
    /// impurity row.
    pub fn solve_step_rows(&mut self, tstp: i64) {
        self.sigma_bare_second(tstp);
        let h_g0_up = self.h_g0(true);
        let h_g0_dn = self.h_g0(false);
        {
            let Self {
                beta,
                h,
                kt,
                delta_up,
                delta_dn,
                g0_up,
                g0_dn,
                ..
            } = self;
            dyson_timestep(tstp, g0_up, 0.0, &h_g0_up, delta_up, *beta, *h, *kt as i64);
            dyson_timestep(tstp, g0_dn, 0.0, &h_g0_dn, delta_dn, *beta, *h, *kt as i64);
        }
        self.sigma_bare_second(tstp);
        self.refresh_sigma_tot_row(tstp);
        let h_g_up = self.h_g(true);
        let h_g_dn = self.h_g(false);
        let Self {
            beta,
            h,
            kt,
            sigma_tot_up,
            sigma_tot_dn,
            g_up,
            g_dn,
            ..
        } = self;
        dyson_timestep(
            tstp,
            g_up,
            0.0,
            &h_g_up,
            sigma_tot_up,
            *beta,
            *h,
            *kt as i64,
        );
        dyson_timestep(
            tstp,
            g_dn,
            0.0,
            &h_g_dn,
            sigma_tot_dn,
            *beta,
            *h,
            *kt as i64,
        );
    }

    /// Update the bath rows `tstp` from the Bethe self-consistency
    /// `Delta_sigma(tstp) = v*^2 G_sbar(tstp)` (ret/les/tv row).
    pub fn update_bath_row(&mut self, tstp: i64) {
        let v2 = Complex::new(self.vstar2, 0.0);
        for (delta, g_bar) in [
            (&mut self.delta_up, &self.g_dn),
            (&mut self.delta_dn, &self.g_up),
        ] {
            for j in 0..=tstp {
                delta.ret.raw_block_at_mut(tstp, j)[0] = v2 * g_bar.ret.raw_block_at(tstp, j)[0];
                delta.less.raw_block_at_mut(j, tstp)[0] = v2 * g_bar.less.raw_block_at(j, tstp)[0];
            }
            for m in 0..=(self.ntau as i64) {
                delta.lmix.raw_block_at_mut(tstp, m)[0] = v2 * g_bar.lmix.raw_block_at(tstp, m)[0];
            }
        }
    }

    /// Refresh the bath rows `tstp` inside `sigma_tot` (after
    /// [`Self::update_bath_row`], before the next `G` solve).
    pub fn refresh_tot_bath_row(&mut self, tstp: i64) {
        self.refresh_sigma_tot_row(tstp);
    }

    /// Matsubara-axis bath mixing
    /// `Delta <- mixing Delta + (1 - mixing) v*^2 G_bar`; returns the
    /// maximal absolute change of the mixed bath.
    pub fn mix_bath_mat(&mut self, mixing: f64) -> f64 {
        let v2 = Complex::new(self.vstar2, 0.0);
        let mut err = 0.0f64;
        for (delta, g_bar) in [
            (&mut self.delta_up, &self.g_dn),
            (&mut self.delta_dn, &self.g_up),
        ] {
            for m in 0..=self.ntau {
                let old = delta.mat.block(m)[0];
                let new = v2 * g_bar.mat.block(m)[0];
                err = err.max((new - old).abs());
                delta.mat.block_mut(m)[0] =
                    Complex::new(mixing, 0.0) * old + Complex::new(1.0 - mixing, 0.0) * new;
            }
        }
        err
    }

    /// Matsubara distance between the current bath and the Bethe
    /// self-consistency (equilibrium convergence measure).
    pub fn bath_selfconsistency_error(&self) -> f64 {
        let v2 = Complex::new(self.vstar2, 0.0);
        let mut err = 0.0f64;
        for (delta, g_bar) in [(&self.delta_up, &self.g_dn), (&self.delta_dn, &self.g_up)] {
            for m in 0..=self.ntau {
                let want = v2 * g_bar.mat.block(m)[0];
                err = err.max((delta.mat.block(m)[0] - want).abs());
            }
        }
        err
    }

    /// Real-time row distance `||v*^2 G_bar(tstp) - Delta(tstp)||`
    /// (ret/les/tv row max norm).
    pub fn row_selfconsistency_error(&self, tstp: i64) -> f64 {
        let v2 = Complex::new(self.vstar2, 0.0);
        let mut err = 0.0f64;
        for (delta, g_bar) in [(&self.delta_up, &self.g_dn), (&self.delta_dn, &self.g_up)] {
            for j in 0..=tstp {
                err = err.max(
                    (delta.ret.raw_block_at(tstp, j)[0] - v2 * g_bar.ret.raw_block_at(tstp, j)[0])
                        .abs(),
                );
                err = err.max(
                    (delta.less.raw_block_at(j, tstp)[0]
                        - v2 * g_bar.less.raw_block_at(j, tstp)[0])
                        .abs(),
                );
            }
            for m in 0..=(self.ntau as i64) {
                err = err.max(
                    (delta.lmix.raw_block_at(tstp, m)[0]
                        - v2 * g_bar.lmix.raw_block_at(tstp, m)[0])
                        .abs(),
                );
            }
        }
        err
    }

    /// Spin density `n_sigma(tstp) = -i G_sigma^<(t,t)`.
    pub fn density(&self, up: bool, tstp: i64) -> f64 {
        let g = if up { &self.g_up } else { &self.g_dn };
        g.less.raw_block_at(tstp, tstp)[0].im
    }

    /// Equal-time lesser convolution `(A * B)^<(t,t)` of two scalar
    /// contour functions via the KB-tested
    /// [`kadanoffbaym::convolution::convolution_les_timediag`].
    pub fn conv_les_eq(&self, tstp: i64, a: &ContourGreen, b: &ContourGreen) -> Complex {
        let integ = Integrator::new(self.kt as i64);
        kadanoffbaym::convolution::convolution_les_timediag(
            tstp, a, a, None, b, b, &integ, self.beta, self.h,
        )
        .get(0, 0)
    }

    /// Hartree tadpole `h_sigma(tstp) = U(t)(n0_sbar - alpha_sbar)`.
    /// Full Hartree `U(t) n_sbar(t)` entering the equation-of-motion
    /// double-occupancy formula (paper Eq. 55); NOT alpha-subtracted:
    /// at `t = 0+` it alone must reproduce `d(0+) = 1/4`.
    fn hartree(&self, up: bool, tstp: i64) -> f64 {
        let g_bar = if up { &self.g_dn } else { &self.g_up };
        self.u_at(tstp) * Self::n0_of(g_bar, tstp)
    }

    /// Double occupancy `d(t) = -i/(2U) sum_sigma (Sigma_sigma^int *
    /// G_sigma)^<(t,t)` (paper Eq. 55), including the instantaneous
    /// Hartree contribution `h_sigma(t) G_sigma^<(t,t)`.
    pub fn double_occupancy(&self, tstp: i64) -> f64 {
        let u = self.u_at(tstp);
        let mut acc = Complex::ZERO;
        for up in [true, false] {
            let (sig, g) = if up {
                (&self.sigma_up, &self.g_up)
            } else {
                (&self.sigma_dn, &self.g_dn)
            };
            acc += self.conv_les_eq(tstp, sig, g);
            let gl = g.less.raw_block_at(tstp, tstp)[0];
            acc += Complex::new(self.hartree(up, tstp), 0.0) * gl;
        }
        (-Complex::I * acc / Complex::new(2.0 * u, 0.0)).re
    }

    /// Total energy `E(t) = E_kin(t) + U(t)(d(t) - 1/4)` with
    /// `E_kin = -i v*^2 sum_sigma (G_sbar * G_sigma)^<(t,t)`
    /// (paper Eqs. 57-58).
    pub fn total_energy(&self, tstp: i64) -> f64 {
        let mut acc = self.conv_les_eq(tstp, &self.g_dn, &self.g_up);
        acc += self.conv_les_eq(tstp, &self.g_up, &self.g_dn);
        let e_kin = (-Complex::I * acc * Complex::new(self.vstar2, 0.0)).re;
        let u = self.u_at(tstp);
        e_kin + u * (self.double_occupancy(tstp) - 0.25)
    }
}

#[inline]
fn u_at(u: &[f64], tstp: i64) -> f64 {
    u[tstp as usize + 1]
}

/// Fill the real-time rows `tstp` of one spin's sunset self-energy
/// (free function so the caller can split-borrow disjoint fields).
///
/// Component decomposition of
/// `Sigma = U(t)U(t') A(t,t') B(t',t) C(t,t')` with `A = G0_sigma`,
/// `B = C = G0_sbar`:
/// * les `(j <= n)`: `u_j u_n A^<(j,n) (B^R(n,j) + B^<(j,n)) C^<(j,n)`
///   (the reversed factor at (later, earlier) is simply the greater
///   component `B^> = B^R + B^<`; greater and lesser are related only
///   through their own anti-hermiticity, never by conjugation)
/// * ret `(j <= n)` from `Sigma^> - Sigma^<`:
///   `u_n u_j [(A^R + A^<) B^< (C^R + C^<)
///   - A^< (B^R + B^<) C^<]`
///
/// Both components then satisfy the exact fermionic anti-hermiticity
/// `Sigma^<(n,j) = -[Sigma^<(j,n)]^*` (and likewise for `>`) by
/// construction, which preserves the particle-hole symmetry of the
/// half-filled state.
/// * tv `(n, m)`: `u_n u_i A^tv(n,m) conj(B^tv(n, N-m)) C^tv(n,m)`
///   (fermionic `vt` view of the reversed factor)
/// Fill the rows `tstp` (or the Matsubara component when `tstp ==
/// MATSUBARA`) of one spin's sunset self-energy using the KB bubble
/// primitives, composing the scheme of
/// `ness2_bethe_prop.cpp::get_Sigma_timestep`:
/// `chi(t,t') = i^2 G0_sbar(t,t') G0_sbar(t',t)` [`bubble1_herm`,
/// a Bose bubble], `W = U(t)U(t') chi` (scalar row/column scaling),
/// `Sigma = -i^2 G0_sigma(t,t') W(t',t)` [`bubble2_herm`].
/// On the Matsubara axis this yields exactly
/// `Sigma^M[m] = U_i^2 G0^M[m]^2 G0_sbar^M[N-m]` (paper Eq. 31).
fn sigma_second_spin(
    tstp: i64,
    ntau: usize,
    u: &[f64],
    ga: &ContourGreen,
    gb: &ContourGreen,
    w: &mut ContourGreen,
    sig: &mut ContourGreen,
) {
    kadanoffbaym::convolution::bubble1_herm(tstp, w, 0, 0, gb, 0, 0, gb, 0, 0);
    // W = U(t)U(t') chi: scale row tstp (and the diagonal-adjacent
    // lesser column / lmix row); MATSUBARA scales the mat axis with
    // the initial interaction
    if tstp < 0 {
        let uu = Complex::new(u[0] * u[0], 0.0);
        for m in 0..=ntau {
            w.mat.block_mut(m)[0] *= uu;
        }
    } else {
        let u_n = u_at(u, tstp);
        for j in 0..=tstp {
            let uu = Complex::new(u_at(u, j) * u_n, 0.0);
            w.ret.raw_block_at_mut(tstp, j)[0] *= uu;
            w.less.raw_block_at_mut(j, tstp)[0] *= uu;
        }
        let uu = Complex::new(u[0] * u_n, 0.0);
        for m in 0..=(ntau as i64) {
            w.lmix.raw_block_at_mut(tstp, m)[0] *= uu;
        }
    }
    kadanoffbaym::convolution::bubble2_herm(tstp, sig, 0, 0, ga, 0, 0, w, 0, 0);
    // final -1 sign
    if tstp < 0 {
        for m in 0..=ntau {
            sig.mat.block_mut(m)[0] = -sig.mat.block_mut(m)[0];
        }
    } else {
        for j in 0..=tstp {
            let r = sig.ret.raw_block_at_mut(tstp, j)[0];
            sig.ret.raw_block_at_mut(tstp, j)[0] = -r;
            let l = sig.less.raw_block_at_mut(j, tstp)[0];
            sig.less.raw_block_at_mut(j, tstp)[0] = -l;
        }
        for m in 0..=(ntau as i64) {
            let t = sig.lmix.raw_block_at_mut(tstp, m)[0];
            sig.lmix.raw_block_at_mut(tstp, m)[0] = -t;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solver(nt: usize, ntau: usize, beta: f64, u0: f64, uf: f64) -> IptSolver {
        let mut u = vec![uf; nt + 2];
        u[0] = u0;
        IptSolver::new(nt, ntau, beta, 0.02, 5, 0.0, u, (0.5, 0.5))
    }

    #[test]
    fn sunset_atomic_mat() {
        // free level at the particle-hole point: G0^M(tau) = -1/2 for
        // all tau  =>  Sigma^M[m] = U_i^2 (-1/2)^3 = -U_i^2/8
        let mut s = solver(8, 40, 5.0, 2.0, 2.0);
        for m in 0..=s.ntau {
            s.g0_up.mat.block_mut(m)[0] = Complex::new(-0.5, 0.0);
            s.g0_dn.mat.block_mut(m)[0] = Complex::new(-0.5, 0.0);
        }
        s.sigma_bare_second_mat();
        let want = -4.0 * 0.125; // -U_i^2/8 with U_i = 2
        for m in 0..=s.ntau {
            let z = s.sigma_up.mat.block(m)[0];
            assert!(
                (z.re - want).abs() < 1e-15 && z.im.abs() < 1e-15,
                "{m}: {z:?}"
            );
        }
    }

    #[test]
    fn sunset_equal_time_atomic_anchor() {
        // particle-hole symmetric Weiss rows (G0^R(t,t) = -i,
        // G0^<(t,t) = i/2): the composed sunset must give the atomic
        // equal-time value Sigma^R(t,t) = -i U^2/4 (the time-domain
        // image of the atomic self-energy U^2/(4(omega + i0)))
        let mut s = solver(8, 40, 5.0, 3.0, 4.0);
        for g in [&mut s.g0_up, &mut s.g0_dn] {
            for j in 0..=2i64 {
                g.ret.raw_block_at_mut(2, j)[0] = -Complex::I;
                g.less.raw_block_at_mut(j, 2)[0] = Complex::new(0.0, 0.5);
            }
            for m in 0..=(s.ntau as i64) {
                g.lmix.raw_block_at_mut(2, m)[0] = Complex::ZERO;
            }
        }
        s.sigma_bare_second(2);
        let got = s.sigma_up.ret.raw_block_at(2, 2)[0];
        let want = Complex::new(0.0, -(4.0 * 4.0) / 4.0); // -i U_f^2/4
        assert!(
            (got - want).abs() < 1e-12,
            "Sigma^R(t,t) = {got:?} vs atomic anchor {want:?}"
        );
    }

    #[test]
    fn sunset_tv_handcheck() {
        // Sigma^tv(n, m) = u_n u_i A(n,m) conj(B(n, N-m)) C(n,m)
        let mut s = solver(8, 40, 5.0, 2.0, 4.0);
        let a = Complex::new(0.1, 0.4);
        let b = Complex::new(-0.3, 0.2);
        for m in 0..=(s.ntau as i64) {
            s.g0_up.lmix.raw_block_at_mut(2, m)[0] = a;
            s.g0_dn.lmix.raw_block_at_mut(2, m)[0] = b;
        }
        s.sigma_bare_second(2);
        // Sigma_up has A = G0_up = a, B = C = G0_dn = b
        let want = Complex::new(4.0 * 2.0, 0.0) * a * b.conj() * b;
        for m in 0..=(s.ntau as i64) {
            let got = s.sigma_up.lmix.raw_block_at(2, m)[0];
            assert!((got - want).abs() < 1e-15, "{m}: {got:?} vs {want:?}");
        }
    }
}
