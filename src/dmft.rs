// Project : nQIST
// File    : src/dmft.rs
// Purpose : Bethe-lattice DMFT driver, equilibrium (Matsubara) stage
//           (blueprint ppsc_programs/single_band_hubbard_bethe.cpp)

use crate::io::{iter_line, obs_line, start_iter_line, tstp_line};
use crate::ipt::IptSolver;
use crate::nca::{GfVertex, Hybridization, adjoint_raw};
use crate::observable::update_exp_vals;
use crate::solver::ImpuritySolver;
use kadanoffbaym::storage::{set_timestep_at, timestep_at};
use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{Contour, ContourGreen, GreenTimeStep, MATSUBARA, Sign};
use kadanoffbaym::weights::Integrator;

/// Bath hybridization channels of the single-band Bethe driver
/// (blueprint `get_pp_ints`: forward and backward channels for both
/// spins, assuming spin symmetry, `v* = 1` so `Delta = Gloc`).
pub fn get_pp_ints<'a>(
    delta: &'a ContourGreen,
    delta_cc: &'a ContourGreen,
    solver: &ImpuritySolver,
) -> Vec<Hybridization<'a>> {
    let hs = &solver.hs;
    let cua = hs.flavor(0, 0, false);
    let cuc = hs.flavor(0, 0, true);
    let cda = hs.flavor(0, 1, false);
    let cdc = hs.flavor(0, 1, true);
    vec![
        Hybridization {
            delta,
            op_cre_fl: cuc,
            op_ann_fl: cua,
            fermion: true,
            backward: false,
        }, // spin up fwd
        Hybridization {
            delta: delta_cc,
            op_cre_fl: cua,
            op_ann_fl: cuc,
            fermion: true,
            backward: true,
        }, // spin up bwd
        Hybridization {
            delta,
            op_cre_fl: cdc,
            op_ann_fl: cda,
            fermion: true,
            backward: false,
        }, // spin dn fwd
        Hybridization {
            delta: delta_cc,
            op_cre_fl: cda,
            op_ann_fl: cdc,
            fermion: true,
            backward: true,
        }, // spin dn bwd
    ]
}

/// Single-band spin-up measurement vertex (blueprint `get_gf_verts`).
pub fn get_gf_verts(solver: &ImpuritySolver) -> Vec<GfVertex> {
    vec![GfVertex {
        op_ann_fl: solver.hs.flavor(0, 0, false),
        op_cre_fl: solver.hs.flavor(0, 0, true),
    }]
}

/// Backward channel Matsubara component from the forward one
/// (blueprint `set_bwd_from_fwd` at `t = -1`). The reference binary
/// applies the extra fermionic minus sign (the commented-out
/// "additional minus sign assuming fermionic" variant in ppsc.cpp,
/// verified against the compiled library):
/// `Delta_cc^M(tau) = - Delta^M(ntau - tau)` over `tau = 0..=ntau`.
pub fn set_bwd_from_fwd_mat(bwd: &mut ContourGreen, fwd: &ContourGreen) {
    assert_eq!(bwd.sign, fwd.sign, "set_bwd_from_fwd_mat: sign mismatch");
    let ntau = fwd.ntau;
    for m in 0..=ntau {
        let v = fwd.mat.block(ntau - m)[0];
        bwd.mat.block_mut(m)[0] = -v;
    }
}

/// Output of the equilibrium DMFT stage: the log lines in reference
/// print order, ready to diff against the C++ run.log.
#[derive(Clone, Debug, Default)]
pub struct EquilibriumLog {
    /// `pp_dens, pp_mu = ...` line of `solve_atomic`.
    pub atomic_line: String,
    /// `dens, pp_mu = ...` line per DMFT iteration.
    pub dens_lines: Vec<String>,
    /// `iter:  N err: X` per DMFT iteration.
    pub iter_lines: Vec<String>,
    pub converged: bool,
}

/// Equilibrium (Matsubara) DMFT loop of the Bethe driver: iterate
/// {update_diagrams -> pp_step(-1) -> spgf -> linear mixing into Gloc
/// -> Delta = Gloc -> Delta_cc}, convergence when the Gloc distance
/// drops below `errmax` (blueprint MATSUBARA block of the driver).
/// Returns the log plus the converged bath Green's functions
/// (Gloc, Delta, Delta_cc), ready for the start window.
pub fn run_equilibrium(
    solver: &mut ImpuritySolver,
    linear_mixing: f64,
    errmax: f64,
    itermax: usize,
) -> (EquilibriumLog, ContourGreen, ContourGreen, ContourGreen) {
    run_equilibrium_probe(
        solver,
        linear_mixing,
        errmax,
        itermax,
        |_, _, _, _, _, _| {},
    )
}

/// `run_equilibrium` with a per-iteration probe hook, called after
/// the bath update of every iteration with `(iter, solver, Gloc,
/// Delta, Delta_cc, spgf)` — the spgf is the pre-mixing NCA local
/// Green's function of that iteration (component verification
/// against the golden dump).
#[allow(clippy::type_complexity)]
pub fn run_equilibrium_probe<F>(
    solver: &mut ImpuritySolver,
    linear_mixing: f64,
    errmax: f64,
    itermax: usize,
    mut probe: F,
) -> (EquilibriumLog, ContourGreen, ContourGreen, ContourGreen)
where
    F: FnMut(usize, &ImpuritySolver, &ContourGreen, &ContourGreen, &ContourGreen, &[Vec<Complex>]),
{
    let nt = solver.nt;
    let ntau = solver.ntau;
    let beta = solver.beta;
    let h = solver.h;
    let tmax = nt as f64 * h;

    let contour = Contour::new(nt as i64, ntau as i64, 1, 1, tmax, beta);
    let mut gloc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let mut delta = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let mut delta_cc = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let mut gtmp = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);

    let mut log = EquilibriumLog::default();
    log.atomic_line = solver.solve_atomic();

    let verts = get_gf_verts(solver);

    let mut converged = false;
    for iter in 1..=itermax {
        let pp_ints = get_pp_ints(&delta, &delta_cc, solver);
        if let Some(line) = solver.pp_step(MATSUBARA, &pp_ints) {
            log.dens_lines.push(line);
        }

        // -- linear mixing of the local Green's function (mat only)
        let g_new = solver.get_spgf_mat(&verts, &pp_ints);
        for m in 0..=ntau {
            let old = gloc.mat.block(m)[0];
            let mixed = Complex::new(linear_mixing, 0.0) * old
                + Complex::new(1.0 - linear_mixing, 0.0) * g_new[0][m];
            gloc.mat.block_mut(m)[0] = mixed;
        }

        // -- convergence check against the previous Gloc
        let err = gtmp.distance_at(&gloc, MATSUBARA);
        for m in 0..=ntau {
            let v = gloc.mat.block(m)[0];
            gtmp.mat.block_mut(m)[0] = v;
        }

        // -- Bethe self-consistency (v* = 1): Delta = Gloc
        for m in 0..=ntau {
            let v = gloc.mat.block(m)[0];
            delta.mat.block_mut(m)[0] = v;
        }
        set_bwd_from_fwd_mat(&mut delta_cc, &delta);

        probe(iter, solver, &gloc, &delta, &delta_cc, &g_new);

        log.iter_lines.push(iter_line(iter, err));
        if err < errmax {
            converged = true;
            break;
        }
    }

    solver.update_density_matrix(MATSUBARA);
    log.converged = converged;
    (log, gloc, delta, delta_cc)
}

// ------------------------------------------------------------------
// start window and real-time stage
// ------------------------------------------------------------------

/// Backward channel real-time components of timestep `t >= 0` from
/// the forward ones (blueprint `set_bwd_from_fwd`; the compiled
/// reference applies the OLD commented variant times an overall
/// minus, probes 5b/5c):
/// `ret_bwd(t,j) = +fwd_ret(t,j)^dag` (`j < t`),
/// `ret_bwd(t,t) = -fwd_ret(t,t)`,
/// `les_bwd(j,t) = fwd_ret(t,j) - fwd_les(j,t)^dag` (`j < t`),
/// `les_bwd(t,t) = fwd_les(t,t) - fwd_ret(t,t)^dag`,
/// `tv_bwd(t,m) = +fwd_tv(t, ntau - m)^dag`.
pub fn set_bwd_from_fwd_tstp(t: i64, bwd: &mut ContourGreen, fwd: &ContourGreen) {
    assert_eq!(bwd.sign, fwd.sign, "set_bwd_from_fwd_tstp: sign mismatch");
    let (s1, s2) = (fwd.size1, fwd.size2);
    for j in 0..t {
        // ret: +conj-transpose of the stored forward block
        let src = adjoint_raw(fwd.ret.raw_block_at(t, j), s1, s2);
        bwd.ret.raw_block_at_mut(t, j).copy_from_slice(&src);
        // les: fwd_ret(t,j) - fwd_les(j,t)^dag
        let r = fwd.ret.raw_block_at(t, j).to_vec();
        let ladj = adjoint_raw(fwd.less.raw_block_at(j, t), s1, s2);
        let blk = bwd.less.raw_block_at_mut(j, t);
        for q in 0..blk.len() {
            blk[q] = r[q] - ladj[q];
        }
    }
    // equal-time ret: -fwd_ret(t,t)
    let r = fwd.ret.raw_block_at(t, t).to_vec();
    let blk = bwd.ret.raw_block_at_mut(t, t);
    for q in 0..blk.len() {
        blk[q] = -r[q];
    }
    // equal-time les: fwd_les(t,t) - fwd_ret(t,t)^dag
    let l = fwd.less.raw_block_at(t, t).to_vec();
    let radj = adjoint_raw(&r, s1, s2);
    let blk = bwd.less.raw_block_at_mut(t, t);
    for q in 0..blk.len() {
        blk[q] = l[q] - radj[q];
    }
    // tv: +fwd_tv(t, ntau - m)^dag
    let ntau = fwd.ntau as i64;
    for m in 0..=ntau {
        let src = adjoint_raw(fwd.lmix.raw_block_at(t, ntau - m), s1, s2);
        bwd.lmix.raw_block_at_mut(t, m).copy_from_slice(&src);
    }
}

/// Write ONLY the time step `f.tstp` row of `f` into `g` (ret row,
/// less column, tv row). NESSi's `set_timestep` semantics for
/// sparsely filled timesteps: unlike KB's `set_timestep_at` this
/// does not touch the earlier rows (which carry no data in a
/// diagram-evaluated timestep and would be zeroed by a prefix
/// copy).
pub fn set_gloc_row(g: &mut ContourGreen, f: &GreenTimeStep) {
    let n = f.tstp as i64;
    for j in 0..=n {
        let src = f.ret.block(f.ret.idx(n, j)).to_vec();
        g.ret.block_mut(g.ret.idx(n, j)).copy_from_slice(&src);
        let src = f.less.block(f.less.idx(j, n)).to_vec();
        g.less.block_mut(g.less.idx(j, n)).copy_from_slice(&src);
    }
    for m in 0..=(f.ntau as i64) {
        let src = f.lmix.raw_block_at(0, m).to_vec();
        g.lmix.raw_block_at_mut(n, m).copy_from_slice(&src);
    }
}

/// Bath row from the local Green's function with a time-dependent
/// hopping: `Delta(t_n, t_j) = t(t_n) t(t_j) Gloc(t_n, t_j)` on the
/// real branch, `Delta(t_n, tau) = t(t_n) t_init Gloc(t_n, tau)` on
/// the mixed component. `tvec` follows the C++ time-array convention
/// (`tvec[tstp + 1]`, `tvec[0]` = equilibrium).
fn set_delta_row(delta: &mut ContourGreen, gloc: &ContourGreen, n: i64, tvec: &[f64]) {
    let tn = tvec[(n + 1) as usize];
    for j in 0..=n {
        let tj = tvec[(j + 1) as usize];
        let f = Complex::new(tn * tj, 0.0);
        let src = gloc.ret.block(gloc.ret.idx(n, j)).to_vec();
        for (d, sv) in delta.ret.block_mut(delta.ret.idx(n, j)).iter_mut().zip(src) {
            *d = f * sv;
        }
        let src = gloc.less.block(gloc.less.idx(j, n)).to_vec();
        for (d, sv) in delta
            .less
            .block_mut(delta.less.idx(j, n))
            .iter_mut()
            .zip(src)
        {
            *d = f * sv;
        }
    }
    let ti = Complex::new(tn * tvec[0], 0.0);
    for m in 0..=(gloc.ntau as i64) {
        let src = gloc.lmix.raw_block_at(n, m).to_vec();
        for (d, sv) in delta.lmix.raw_block_at_mut(n, m).iter_mut().zip(src) {
            *d = ti * sv;
        }
    }
}

/// START-window log: `START: iter:  N err: X` lines.
#[derive(Clone, Debug, Default)]
pub struct StartLog {
    pub start_lines: Vec<String>,
    pub converged: bool,
}

/// START window of the driver (blueprint `init_real_time` + the START
/// block): `set_t0`, then iterate {pp_step(kt) -> spgf(0..=kt) ->
/// Gloc/Delta rows -> Delta_cc} until the Gloc distance at `kt`
/// drops below `errmax`. Mutates the bath Green's functions in place.
pub fn run_start_window(
    solver: &mut ImpuritySolver,
    gloc: &mut ContourGreen,
    delta: &mut ContourGreen,
    delta_cc: &mut ContourGreen,
    errmax: f64,
    itermax: usize,
) -> StartLog {
    run_start_window_probe(
        solver,
        gloc,
        delta,
        delta_cc,
        errmax,
        itermax,
        |_, _, _, _| {},
    )
}

/// `run_start_window` with a per-iteration probe hook, called after
/// the bath update of every iteration (component verification
/// against the golden dump).
#[allow(clippy::type_complexity)]
pub fn run_start_window_probe<F>(
    solver: &mut ImpuritySolver,
    gloc: &mut ContourGreen,
    delta: &mut ContourGreen,
    delta_cc: &mut ContourGreen,
    errmax: f64,
    itermax: usize,
    mut probe: F,
) -> StartLog
where
    F: FnMut(i64, &ImpuritySolver, &ContourGreen, &ContourGreen),
{
    solver.set_t0();
    let kt = solver.kt as i64;
    let verts = get_gf_verts(solver);
    let mut gtmp = ContourGreen::new(
        &Contour::new(
            solver.nt as i64,
            solver.ntau as i64,
            1,
            1,
            solver.nt as f64 * solver.h,
            solver.beta,
        ),
        Sign::Fermi,
        Complex::ZERO,
    );
    let mut log = StartLog::default();
    for iter in 1..=itermax {
        let pp_ints = get_pp_ints(delta, delta_cc, solver);
        solver.pp_step(kt, &pp_ints);
        // blueprint semantics: the OCA gf reads the hybridization through
        // REFERENCES, so spgf(n) sees the Delta rows 0..n-1 already updated
        // within this loop (invisible for order 1, load-bearing for OCA)
        for n in 0..=kt {
            let pp_ints = get_pp_ints(delta, delta_cc, solver);
            let gf = solver.get_spgf(&verts, n, &pp_ints).remove(0);
            set_gloc_row(gloc, &gf);
            let ts = timestep_at(gloc, n);
            set_timestep_at(delta, &ts);
            set_bwd_from_fwd_tstp(n, delta_cc, delta);
        }
        let err = gtmp.distance_at(gloc, kt);
        let ts = timestep_at(gloc, kt);
        set_timestep_at(&mut gtmp, &ts);
        log.start_lines.push(start_iter_line(iter, err));
        // first argument is the DMFT iteration index (1-based), so
        // probes can capture per-iteration states (blueprint timing:
        // end of the iteration body, after the bath update)
        probe(iter as i64, solver, delta, delta_cc);
        if err < errmax {
            log.converged = true;
            break;
        }
    }
    log
}

/// Real-time stage of the driver (blueprint REALTIME block): per
/// timestep `extrapolate_timestep(tstp - 1)`, `iter_rtime` iterations
/// {pp_step -> spgf -> bath update}, then the local observables.
/// Returns the `tstp:  N` lines and the `obs: ...` lines. The probe
/// hook fires after the last iteration of each timestep (component
/// verification against the golden dump).
pub fn run_realtime<F>(
    solver: &mut ImpuritySolver,
    gloc: &mut ContourGreen,
    delta: &mut ContourGreen,
    delta_cc: &mut ContourGreen,
    iter_rtime: usize,
    from: i64,
    to: i64,
    mut probe: F,
) -> (Vec<String>, Vec<String>)
where
    F: FnMut(i64, &ImpuritySolver, &ContourGreen, &ContourGreen),
{
    let verts = get_gf_verts(solver);
    let mut tstp_lines = Vec::new();
    let mut obs_lines = Vec::new();
    for tstp in from..=to {
        solver.extrapolate_timestep(tstp - 1);
        for _ in 1..=iter_rtime {
            let pp_ints = get_pp_ints(delta, delta_cc, solver);
            solver.pp_step(tstp, &pp_ints);
            let gf = solver.get_spgf(&verts, tstp, &pp_ints);
            set_gloc_row(gloc, &gf[0]);
            let ts = timestep_at(gloc, tstp);
            set_timestep_at(delta, &ts);
            set_bwd_from_fwd_tstp(tstp, delta_cc, delta);
        }
        solver.update_density_matrix(tstp);
        let obs = update_exp_vals(solver, tstp);
        probe(tstp, solver, delta, delta_cc);
        tstp_lines.push(tstp_line(tstp as usize));
        obs_lines.push(obs_line(
            tstp as usize,
            tstp as f64 * solver.h,
            obs.n_up,
            obs.n_dn,
            obs.docc,
        ));
    }
    (tstp_lines, obs_lines)
}

// ------------------------------------------------------------------
// IPT driver (P6, blueprint arXiv:1306.0307v2 Sec. III/V): the bare
// second-order impurity solve is explicit in G0, so each DMFT stage
// iterates {bath update from Bethe self-consistency; Weiss solve;
// sunset self-energy; impurity solve} without an inner impurity
// fixed point.
// ------------------------------------------------------------------

/// IPT equilibrium (Matsubara) DMFT loop. Returns the per-iteration
/// error log lines.
pub fn run_equilibrium_ipt(
    solver: &mut IptSolver,
    mixing: f64,
    errmax: f64,
    itermax: usize,
) -> Vec<String> {
    let mut log = Vec::new();
    for iter in 1..=itermax {
        solver.solve_g0_mat();
        solver.solve_g_mat();
        let err = solver.mix_bath_mat(mixing);
        log.push(format!("iter: {iter} err: {}", crate::io::fmt_g6(err)));
        if err < errmax {
            break;
        }
    }
    log
}

/// IPT start window (`tstp = 0..=kt`): seeded with the Matsubara
/// `t0` rows, iterate the window to self-consistency.
pub fn run_start_ipt(solver: &mut IptSolver, errmax: f64, itermax: usize) -> Vec<String> {
    kadanoffbaym::equilibrium::from_mat(&mut solver.g_up);
    kadanoffbaym::equilibrium::from_mat(&mut solver.g_dn);
    kadanoffbaym::equilibrium::from_mat(&mut solver.g0_up);
    kadanoffbaym::equilibrium::from_mat(&mut solver.g0_dn);
    let kt = solver.kt as i64;
    let mut log = Vec::new();
    for iter in 1..=itermax {
        for n in 0..=kt {
            solver.update_bath_row(n);
        }
        solver.solve_start_rows();
        let mut err = 0.0f64;
        for n in 0..=kt {
            err = err.max(solver.row_selfconsistency_error(n));
        }
        log.push(format!("iter: {iter} err: {}", crate::io::fmt_g6(err)));
        if err < errmax {
            break;
        }
    }
    log
}

/// IPT real-time sweep `from..=to`: polynomial extrapolation of the
/// impurity rows as the predictor, then `iter_rtime` DMFT
/// iterations per step, then the observation hook.
pub fn run_realtime_ipt<F>(
    solver: &mut IptSolver,
    iter_rtime: usize,
    from: i64,
    to: i64,
    mut obs: F,
) where
    F: FnMut(i64, &IptSolver),
{
    let kt = solver.kt as i64;
    let integ = Integrator::new(kt);
    for tstp in from..=to {
        // predictor: extrapolate row `tstp` of G and G0 (KB fills row
        // n + 1, hence the tstp - 1 argument); the G0 rows keep the
        // Hartree density `n0(tst)` and thus H(tst) continuous
        kadanoffbaym::math::extrapolate_timestep(tstp - 1, &mut solver.g_up, &integ);
        kadanoffbaym::math::extrapolate_timestep(tstp - 1, &mut solver.g_dn, &integ);
        kadanoffbaym::math::extrapolate_timestep(tstp - 1, &mut solver.g0_up, &integ);
        kadanoffbaym::math::extrapolate_timestep(tstp - 1, &mut solver.g0_dn, &integ);
        for _ in 0..iter_rtime {
            solver.update_bath_row(tstp);
            solver.solve_step_rows(tstp);
            if solver.row_selfconsistency_error(tstp) < 1e-7 {
                break;
            }
        }
        obs(tstp, solver);
    }
}

// ------------------------------------------------------------------
// multi-channel (two-band) driver, P7: mirrors the single-band loops
// with per spin-orbital Gloc/Delta/Delta_cc channels (blueprint
// two_band_hubbard_bethe.cpp). Channel order is the C++ Gloc order
// (o1up, o1do, o2up, o2do), i.e. channel = 2 * orb + spin. The
// hopping is a constant `t_hop` (Delta = t_hop^2 Gloc).
// ------------------------------------------------------------------

/// Eight hybridization entries of the two-band problem (4 forward +
/// 4 backward), one per spin-orbital channel.
pub fn get_pp_ints_multi<'a>(
    deltas: &'a [ContourGreen],
    deltas_cc: &'a [ContourGreen],
    solver: &ImpuritySolver,
) -> Vec<Hybridization<'a>> {
    let hs = &solver.hs;
    let norb = hs.norb;
    let mut out = Vec::with_capacity(4 * norb);
    for orb in 0..norb {
        for spin in 0..2 {
            let c = 2 * orb + spin;
            let cre = hs.flavor(orb, spin, true);
            let ann = hs.flavor(orb, spin, false);
            out.push(Hybridization {
                delta: &deltas[c],
                op_cre_fl: cre,
                op_ann_fl: ann,
                fermion: true,
                backward: false,
            });
            out.push(Hybridization {
                delta: &deltas_cc[c],
                op_cre_fl: ann,
                op_ann_fl: cre,
                fermion: true,
                backward: true,
            });
        }
    }
    out
}

/// The spin-orbital Green's-function vertices in channel order
/// (`2 * orb + spin`): two for the single-band AFM case, four for
/// the two-band case.
pub fn get_gf_verts_multi(solver: &ImpuritySolver) -> Vec<GfVertex> {
    let hs = &solver.hs;
    let mut out = Vec::with_capacity(2 * hs.norb);
    for orb in 0..hs.norb {
        for spin in 0..2 {
            out.push(GfVertex {
                op_ann_fl: hs.flavor(orb, spin, false),
                op_cre_fl: hs.flavor(orb, spin, true),
            });
        }
    }
    out
}

/// Two-band equilibrium (Matsubara) DMFT loop; per-channel linear
/// mixing of Gloc, convergence measured on channel 0 (blueprint
/// `distance_norm2(-1, gtmp, Gloc.o1up)`).
pub fn run_equilibrium_multi(
    solver: &mut ImpuritySolver,
    tvec: &[f64],
    // bath channel map: `delta[c] = t_hop^2 gloc[bath_map[c]]`
    // (identity for the paramagnetic case, `[1, 0]` for the AFM
    // cross Bethe self-consistency)
    bath_map: &[usize],
    linear_mixing: f64,
    errmax: f64,
    itermax: usize,
) -> (
    EquilibriumLog,
    Vec<ContourGreen>,
    Vec<ContourGreen>,
    Vec<ContourGreen>,
) {
    let nt = solver.nt;
    let ntau = solver.ntau;
    let contour = Contour::new(
        nt as i64,
        ntau as i64,
        1,
        1,
        nt as f64 * solver.h,
        solver.beta,
    );
    let mk = || ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let nchan = 2 * solver.hs.norb;
    let mut gloc = (0..nchan).map(|_| mk()).collect::<Vec<_>>();
    let mut delta = (0..nchan).map(|_| mk()).collect::<Vec<_>>();
    let mut delta_cc = (0..nchan).map(|_| mk()).collect::<Vec<_>>();
    // per-channel convergence references (a shared reference would
    // leak the cross-channel distance into the AFM error)
    let mut gtmps: Vec<ContourGreen> = (0..nchan).map(|_| mk()).collect();

    let mut log = EquilibriumLog::default();
    log.atomic_line = solver.solve_atomic();
    let verts = get_gf_verts_multi(solver);
    let tsq = Complex::new(tvec[0] * tvec[0], 0.0);

    let mut converged = false;
    for iter in 1..=itermax {
        let pp_ints = get_pp_ints_multi(&delta, &delta_cc, solver);
        if let Some(line) = solver.pp_step(MATSUBARA, &pp_ints) {
            log.dens_lines.push(line);
        }
        let g_new = solver.get_spgf_mat(&verts, &pp_ints);
        for c in 0..nchan {
            for m in 0..=ntau {
                let old = gloc[c].mat.block(m)[0];
                let mixed = Complex::new(linear_mixing, 0.0) * old
                    + Complex::new(1.0 - linear_mixing, 0.0) * g_new[c][m];
                gloc[c].mat.block_mut(m)[0] = mixed;
                delta[c].mat.block_mut(m)[0] = tsq * gloc[bath_map[c]].mat.block(m)[0];
            }
            set_bwd_from_fwd_mat(&mut delta_cc[c], &delta[c]);
        }
        let err = (0..nchan)
            .map(|c| gtmps[c].distance_at(&gloc[c], MATSUBARA))
            .fold(0.0f64, f64::max);
        for c in 0..nchan {
            for m in 0..=ntau {
                let v = gloc[c].mat.block(m)[0];
                gtmps[c].mat.block_mut(m)[0] = v;
            }
        }
        log.iter_lines.push(iter_line(iter, err));
        if err < errmax {
            converged = true;
            break;
        }
    }
    solver.update_density_matrix(MATSUBARA);
    log.converged = converged;
    (log, gloc, delta, delta_cc)
}

/// Two-band START window: `set_t0`, then iterate {pp_step(kt) ->
/// per-channel spgf(0..=kt) -> Gloc/Delta rows -> Delta_cc} until the
/// channel-0 Gloc distance at `kt` drops below `errmax`.
pub fn run_start_window_multi(
    solver: &mut ImpuritySolver,
    tvec: &[f64],
    // bath channel map: `delta[c] = t_hop^2 gloc[bath_map[c]]`
    // (identity for the paramagnetic case, `[1, 0]` for the AFM
    // cross Bethe self-consistency)
    bath_map: &[usize],
    gloc: &mut [ContourGreen],
    delta: &mut [ContourGreen],
    delta_cc: &mut [ContourGreen],
    errmax: f64,
    itermax: usize,
) -> StartLog {
    solver.set_t0();
    let kt = solver.kt as i64;
    let verts = get_gf_verts_multi(solver);
    let mut gtmp = ContourGreen::new(
        &Contour::new(
            solver.nt as i64,
            solver.ntau as i64,
            1,
            1,
            solver.nt as f64 * solver.h,
            solver.beta,
        ),
        Sign::Fermi,
        Complex::ZERO,
    );
    let nchan = gloc.len();
    let mut log = StartLog::default();
    for iter in 1..=itermax {
        let pp_ints = get_pp_ints_multi(delta, delta_cc, solver);
        solver.pp_step(kt, &pp_ints);
        for n in 0..=kt {
            let pp_ints = get_pp_ints_multi(delta, delta_cc, solver);
            let gfs = solver.get_spgf(&verts, n, &pp_ints);
            for c in 0..nchan {
                set_gloc_row(&mut gloc[c], &gfs[c]);
            }
            for c in 0..nchan {
                let src = bath_map[c];
                set_delta_row(&mut delta[c], &gloc[src], n, tvec);
                set_bwd_from_fwd_tstp(n, &mut delta_cc[c], &delta[c]);
            }
        }
        let err = gtmp.distance_at(&gloc[0], kt);
        let ts = timestep_at(&gloc[0], kt);
        set_timestep_at(&mut gtmp, &ts);
        log.start_lines.push(start_iter_line(iter, err));
        if err < errmax {
            log.converged = true;
            break;
        }
    }
    log
}

/// Two-band real-time sweep with the multiplet-observable table
/// (`obs: tstp t p13 puu pud pdd p04 n Q`, blueprint
/// two_band_hubbard_bethe.cpp).
pub fn run_realtime_multi<F>(
    solver: &mut ImpuritySolver,
    tvec: &[f64],
    // bath channel map: `delta[c] = t_hop^2 gloc[bath_map[c]]`
    // (identity for the paramagnetic case, `[1, 0]` for the AFM
    // cross Bethe self-consistency)
    bath_map: &[usize],
    gloc: &mut [ContourGreen],
    delta: &mut [ContourGreen],
    delta_cc: &mut [ContourGreen],
    iter_rtime: usize,
    from: i64,
    to: i64,
    mut probe: F,
) -> (Vec<String>, Vec<String>)
where
    F: FnMut(i64, &ImpuritySolver),
{
    let verts = get_gf_verts_multi(solver);
    let nchan = gloc.len();
    let mut tstp_lines = Vec::new();
    let mut obs_lines = Vec::new();
    for tstp in from..=to {
        solver.extrapolate_timestep(tstp - 1);
        for _ in 1..=iter_rtime {
            let pp_ints = get_pp_ints_multi(delta, delta_cc, solver);
            solver.pp_step(tstp, &pp_ints);
            let gfs = solver.get_spgf(&verts, tstp, &pp_ints);
            for c in 0..nchan {
                set_gloc_row(&mut gloc[c], &gfs[c]);
            }
            for c in 0..nchan {
                let src = bath_map[c];
                set_delta_row(&mut delta[c], &gloc[src], tstp, tvec);
                set_bwd_from_fwd_tstp(tstp, &mut delta_cc[c], &delta[c]);
            }
        }
        solver.update_density_matrix(tstp);
        let line = if solver.hs.norb == 1 {
            // single-band AFM format (blueprint
            // single_band_hubbard_afm_bethe.cpp); m is printed as
            // (n_up - n_dn)/2
            let o = crate::observable::update_exp_vals(solver, tstp);
            format!(
                "obs: {} {} {} {} {} {} {}",
                tstp,
                crate::io::fmt_g6(tstp as f64 * solver.h),
                crate::io::fmt_g6(o.n_up),
                crate::io::fmt_g6(o.n_dn),
                crate::io::fmt_g6(o.docc),
                crate::io::fmt_g6(0.5 * (o.n_up - o.n_dn)),
                crate::io::fmt_g6(o.trace)
            )
        } else if solver.hs.norb == 3 {
            // three-band format: the observable set of the C++
            // update_exp_vals print (three_band_hubbard.hpp); `n` is
            // the total density n1 + n2 + n3
            let obs = crate::observable::update_exp_vals_three_band(solver, tstp);
            format!(
                "obs: {} {} {} {} {} {} {} {}",
                tstp,
                crate::io::fmt_g6(tstp as f64 * solver.h),
                crate::io::fmt_g6(obs.n1 + obs.n2 + obs.n3),
                crate::io::fmt_g6(obs.d1),
                crate::io::fmt_g6(obs.d2),
                crate::io::fmt_g6(obs.d3),
                crate::io::fmt_g6(obs.m),
                crate::io::fmt_g6(obs.q)
            )
        } else {
            let obs = crate::observable::update_exp_vals_two_band(solver, tstp);
            format!(
                "obs: {} {} {} {} {} {} {} {} {}",
                tstp,
                crate::io::fmt_g6(tstp as f64 * solver.h),
                crate::io::fmt_g6(obs.p13),
                crate::io::fmt_g6(obs.puu),
                crate::io::fmt_g6(obs.pud),
                crate::io::fmt_g6(obs.pdd),
                crate::io::fmt_g6(obs.p04),
                crate::io::fmt_g6(obs.n1 + obs.n2),
                crate::io::fmt_g6(obs.q)
            )
        };
        probe(tstp, solver);
        tstp_lines.push(tstp_line(tstp as usize));
        obs_lines.push(line);
    }
    (tstp_lines, obs_lines)
}
