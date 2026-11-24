// Golden dump: full equilibrium DMFT loop of the single-band Bethe
// benchmark, replicating single_band_hubbard_bethe.cpp step by step,
// with component-level dumps (Sigma^M, ppG^M, spgf, Gloc/Delta).
#define NCA_SOLVER_ASSERT_0 0
#define NCA_SOLVER_ASSERT_1 0
#include <complex>
#include <cstdio>
#include <vector>

#include "cntr.hpp"
#include "./ppsc/ppsc.hpp"
#include "./ppsc/solver.hpp"
#include "./ppsc/hilbert_spaces/single_band_fermi_diag.hpp"
#include "./ppsc/hamiltonians/single_band_hubbard.hpp"

typedef ppsc::hilbert_spaces::single_band_fermi_diag hilbert_space_type;
typedef ppsc::hamiltonians::single_band_hubbard<hilbert_space_type> hamiltonian_type;
typedef ppsc::solver<hamiltonian_type> solver_type;

static int MS[11] = {0, 1, 2, 5, 25, 50, 100, 125, 200, 249, 250};

static void dump_pp(const char *tag, int iter, int s, cntr::herm_pseudo<double> &G) {
    for (int i = 0; i < 11; i++) {
        int m = MS[i];
        std::complex<double> x;
        G.get_mat(m, x);
        printf("%s %d %d %d %.17g %.17g\n", tag, iter, s, m, x.real(), x.imag());
    }
}

static void dump_gf(const char *tag, int iter, ppsc::gf_tstp_type &g) {
    for (int i = 0; i < 11; i++) {
        int m = MS[i];
        std::complex<double> x = *g.matptr(m);
        printf("%s %d 0 %d %.17g %.17g\n", tag, iter, m, x.real(), x.imag());
    }
}

static void dump_hm(const char *tag, int iter, cntr::herm_matrix<double> &g) {
    for (int i = 0; i < 11; i++) {
        int m = MS[i];
        std::complex<double> x;
        g.get_mat(m, x);
        printf("%s %d 0 %d %.17g %.17g\n", tag, iter, m, x.real(), x.imag());
    }
}

template <class HILB>
ppsc::pp_ints_type get_pp_ints(ppsc::gf_type &Delta_up, ppsc::gf_type &Delta_up_cc, HILB &hil_) {
    ppsc::operator_type cua = hil_.c_op_[hil_.flavor(0, 0, 0)];
    ppsc::operator_type cuc = hil_.c_op_[hil_.flavor(0, 0, 1)];
    ppsc::operator_type cda = hil_.c_op_[hil_.flavor(0, 1, 0)];
    ppsc::operator_type cdc = hil_.c_op_[hil_.flavor(0, 1, 1)];
    int fermion = -1, fwd = +1, bwd = -1;
    ppsc::pp_ints_type pp_ints;
    pp_ints.push_back(ppsc::pp_int_type(Delta_up, cuc, cua, fermion, fwd));
    pp_ints.push_back(ppsc::pp_int_type(Delta_up_cc, cua, cuc, fermion, bwd));
    pp_ints.push_back(ppsc::pp_int_type(Delta_up, cdc, cda, fermion, fwd));
    pp_ints.push_back(ppsc::pp_int_type(Delta_up_cc, cda, cdc, fermion, bwd));
    return pp_ints;
}

template <class HILB>
ppsc::gf_verts_type get_gf_verts(HILB &hil_) {
    ppsc::operator_type cua = hil_.c_op_[hil_.flavor(0, 0, 0)];
    ppsc::operator_type cuc = hil_.c_op_[hil_.flavor(0, 0, 1)];
    ppsc::gf_verts_type gf_verts;
    gf_verts.push_back(ppsc::gf_vert_type(0, 0, cua, cuc));
    return gf_verts;
}

static void dump_row_tags(const char *tag, int n, cntr::herm_matrix<double> &g) {
    std::complex<double> x;
    for (int j = 0; j <= n; j++) {
        g.get_ret(n, j, x);
        printf("%s RET %d %d %.17g %.17g\n", tag, n, j, x.real(), x.imag());
    }
    for (int j = 0; j <= n; j++) {
        g.get_les(j, n, x);
        printf("%s LES %d %d %.17g %.17g\n", tag, n, j, x.real(), x.imag());
    }
    int ms[8] = {0, 1, 2, 5, 125, 248, 249, 250};
    for (int i = 0; i < 8; i++) {
        g.get_tv(n, ms[i], x);
        printf("%s TV %d %d %.17g %.17g\n", tag, n, ms[i], x.real(), x.imag());
    }
}

static void dump_pp_row(const char *tag, int s, int n, cntr::herm_pseudo<double> &g) {
    std::complex<double> x;
    for (int j = 0; j <= n; j++) {
        g.get_ret(n, j, x);
        printf("%s RET %d %d %.17g %.17g\n", tag, s, j, x.real(), x.imag());
    }
    for (int j = 0; j <= n; j++) {
        g.get_les(j, n, x);
        printf("%s LES %d %d %.17g %.17g\n", tag, s, j, x.real(), x.imag());
    }
    int ms[8] = {0, 1, 2, 5, 125, 248, 249, 250};
    for (int i = 0; i < 8; i++) {
        g.get_tv(n, ms[i], x);
        printf("%s TV %d %d %.17g %.17g\n", tag, s, ms[i], x.real(), x.imag());
    }
}

int main() {
    int nt = 300, ntau = 250, kt = 5;
    double beta = 5.0, h = 0.02, errmax = 1e-8, linear_mixing = 0.5;
    int itermax = 300, iter_rtime = 4, nomp = 1, order = 1;

    hilbert_space_type hil; hil.init();
    solver_type imp(nt, ntau, beta, h, kt, nomp, hil, order);
    imp.hamiltonian.mu = 0.0;
    imp.hamiltonian.U.assign(nt + 2, 3.5); imp.hamiltonian.U[0] = 3.0;
    imp.hamiltonian.eps_up.assign(nt + 2, 0.0);
    imp.hamiltonian.eps_do.assign(nt + 2, 0.0);
    imp.update_hamiltonian();

    cntr::herm_matrix<double> Gloc(nt, ntau, 1, -1), Delta(nt, ntau, 1, -1), Delta_cc(nt, ntau, 1, -1);
    cntr::herm_matrix<double> gtmp(nt, ntau, 1, -1), gtmp2(kt, ntau, 1, -1);
    imp.solve_atomic();

    double err;
    int eq_iters = 0;
    for (int iter = 1; iter <= itermax; iter++) {
        ppsc::pp_ints_type pp_ints = get_pp_ints(Delta, Delta_cc, hil);
        ppsc::gf_verts_type gf_verts = get_gf_verts(hil);
        imp.update_diagrams(pp_ints, gf_verts);
        imp.pp_step(-1);
        ppsc::gf_tstps_type gf = imp.get_spgf(-1);
        ppsc::gf_tstp_type mix(-1, ntau, 1), old(-1, ntau, 1);
        mix.clear(); Gloc.get_timestep(-1, old);
        mix.incr(old, linear_mixing); mix.incr(gf[0], 1.0 - linear_mixing);
        Gloc.set_timestep(-1, mix);
        err = cntr::distance_norm2(-1, gtmp, Gloc);
        gtmp.set_timestep(-1, Gloc);
        cntr::herm_matrix_timestep<double> tmp;
        Gloc.get_timestep(-1, tmp);
        Delta.set_timestep(-1, tmp);
        ppsc::set_bwd_from_fwd(-1, Delta_cc, Delta);
        eq_iters = iter;
        if (err < errmax) break;
    }
    printf("EQCONV %d pp_mu %.17g\n", eq_iters, imp.pp_mu);

    // START window
    imp.init_real_time();
    int start_iters = 0;
    int dump_it1 = 1;
    for (int iter = 1; iter <= itermax; iter++) {
        ppsc::pp_ints_type pp_ints = get_pp_ints(Delta, Delta_cc, hil);
        ppsc::gf_verts_type gf_verts = get_gf_verts(hil);
        imp.update_diagrams(pp_ints, gf_verts);
        imp.pp_step(kt);
        for (int n = 0; n <= kt; n++) {
            ppsc::gf_tstps_type gf = imp.get_spgf(n);
            Gloc.set_timestep(n, gf[0]);
            cntr::herm_matrix_timestep<double> tmp;
            Gloc.get_timestep(n, tmp);
            Delta.set_timestep(n, tmp);
            ppsc::set_bwd_from_fwd(n, Delta_cc, Delta);
        }
        err = cntr::distance_norm2(kt, gtmp2, Gloc);
        gtmp2.set_timestep(kt, Gloc);
        start_iters = iter;
        if (dump_it1 == 1 || dump_it1 == 2) {
            const char *pfx = dump_it1 == 1 ? "IT1" : "IT2";
            char tag[64];
            snprintf(tag, 64, "%sDRT", pfx); dump_row_tags(tag, kt, Delta);
            for (int nn = 0; nn <= kt; nn++) {
                snprintf(tag, 64, "%sLOWDRT", pfx); dump_row_tags(tag, nn, Delta);
                snprintf(tag, 64, "%sLOWSIG", pfx); dump_pp_row(tag, 0, nn, imp.ppSigmas[0]);
            }
            snprintf(tag, 64, "%sDCC", pfx); dump_row_tags(tag, kt, Delta_cc);
            for (int s = 0; s < 4; s++) {
                snprintf(tag, 64, "%sSIG", pfx); dump_pp_row(tag, s, kt, imp.ppSigmas[s]);
                snprintf(tag, 64, "%sPPG", pfx); dump_pp_row(tag, s, kt, imp.ppGfs[s]);
            }
            ppsc::gf_tstps_type gf1 = imp.get_spgf(kt);
            cntr::herm_matrix<double> tmpg(nt, ntau, 1, -1);
            tmpg.set_timestep(kt, gf1[0]);
            snprintf(tag, 64, "%sSPGF", pfx); dump_row_tags(tag, kt, tmpg);
            dump_it1++;
        }
        if (0) {
        }
        if (err < errmax) break;
    }
    printf("STARTCONV %d\n", start_iters);
    dump_row_tags("DRT", kt, Delta);
    dump_row_tags("DCC", kt, Delta_cc);
    for (int s = 0; s < 4; s++) dump_pp_row("SIGRT", s, kt, imp.ppSigmas[s]);
    for (int s = 0; s < 4; s++) dump_pp_row("PPGRT", s, kt, imp.ppGfs[s]);
    {
        ppsc::gf_tstps_type gf = imp.get_spgf(kt);
        cntr::herm_matrix<double> tmpg(nt, ntau, 1, -1);
        tmpg.set_timestep(kt, gf[0]);
        dump_row_tags("SPGFRT", kt, tmpg);
    }

    // two realtime steps
    for (int tstp = kt + 1; tstp <= kt + 2; tstp++) {
        imp.extrapolate_timestep(tstp - 1);
        for (int iter = 1; iter <= iter_rtime; iter++) {
            ppsc::pp_ints_type pp_ints = get_pp_ints(Delta, Delta_cc, hil);
            ppsc::gf_verts_type gf_verts = get_gf_verts(hil);
            imp.update_diagrams(pp_ints, gf_verts);
            imp.pp_step(tstp);
            ppsc::gf_tstps_type gf = imp.get_spgf(tstp);
            Gloc.set_timestep(tstp, gf[0]);
            cntr::herm_matrix_timestep<double> tmp;
            Gloc.get_timestep(tstp, tmp);
            Delta.set_timestep(tstp, tmp);
            ppsc::set_bwd_from_fwd(tstp, Delta_cc, Delta);
        }
        imp.update_density_matrix(tstp);
        imp.hamiltonian.update_exp_vals(tstp, imp.rho);
        double nup_, ndo_, docc_;
        imp.hamiltonian.local_obs(tstp, nup_, ndo_, docc_);
        printf("OBSRT %d %.17g %.17g %.17g %.17g\n", tstp, tstp * h, nup_, ndo_, docc_);
        dump_row_tags("DRT", tstp, Delta);
        char stag[32];
        snprintf(stag, 32, "SIGRT%d", tstp);
        for (int s = 0; s < 4; s++) dump_pp_row(stag, s, tstp, imp.ppSigmas[s]);
        {
            ppsc::gf_tstps_type gf = imp.get_spgf(tstp);
            cntr::herm_matrix<double> tmpg(nt, ntau, 1, -1);
            tmpg.set_timestep(tstp, gf[0]);
            dump_row_tags("SPGFRT", tstp, tmpg);
        }
    }
    return 0;
}
