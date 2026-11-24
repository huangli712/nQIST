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

int main() {
    int nt = 300, ntau = 250, kt = 5, itermax = 300;
    double beta = 5.0, h = 0.02, errmax = 1e-8, linear_mixing = 0.5;

    hilbert_space_type hil;
    hil.init();
    solver_type imp(nt, ntau, beta, h, kt, 1, hil, 1);

    imp.hamiltonian.mu = 0.0;
    imp.hamiltonian.U.assign(nt + 2, 3.5);
    imp.hamiltonian.U[0] = 3.0;
    imp.hamiltonian.eps_up.assign(nt + 2, 0.0);
    imp.hamiltonian.eps_do.assign(nt + 2, 0.0);
    imp.update_hamiltonian();

    cntr::herm_matrix<double> Gloc_up(nt, ntau, 1, -1);
    cntr::herm_matrix<double> Delta_up(nt, ntau, 1, -1);
    cntr::herm_matrix<double> Delta_up_cc(nt, ntau, 1, -1);
    cntr::herm_matrix<double> gtmp(-1, ntau, 1, -1);

    imp.solve_atomic();

    int iter = 0;
    double dmfterr_equil = 0.0;
    for (iter = 1; iter <= itermax; iter++) {
        ppsc::pp_ints_type pp_ints = get_pp_ints(Delta_up, Delta_up_cc, imp.hamiltonian.hil_);
        ppsc::gf_verts_type gf_verts = get_gf_verts(imp.hamiltonian.hil_);

        imp.update_diagrams(pp_ints, gf_verts);
        imp.pp_step(-1);

        if (iter <= 2) {
            for (int s = 0; s < 4; s++) dump_pp("SIG", iter, s, imp.ppSigmas[s]);
            for (int s = 0; s < 4; s++) dump_pp("PPG", iter, s, imp.ppGfs[s]);
        }

        ppsc::gf_tstps_type gf_tstps = imp.get_spgf(-1);
        if (iter <= 2) dump_gf("SPGF", iter, gf_tstps[0]);

        ppsc::gf_tstp_type gloc_old(-1, ntau, 1);
        ppsc::gf_tstp_type gloc_mix(-1, ntau, 1);
        gloc_mix.clear();
        Gloc_up.get_timestep(-1, gloc_old);
        gloc_mix.incr(gloc_old, linear_mixing);
        gloc_mix.incr(gf_tstps[0], 1.0 - linear_mixing);
        Gloc_up.set_timestep(-1, gloc_mix);

        dmfterr_equil = cntr::distance_norm2(-1, gtmp, Gloc_up);
        gtmp.set_timestep(-1, Gloc_up);

        {
            cntr::herm_matrix_timestep<double> tmp;
            Gloc_up.get_timestep(-1, tmp);
            Delta_up.set_timestep(-1, tmp);
            ppsc::set_bwd_from_fwd(-1, Delta_up_cc, Delta_up);
        }

        printf("ERR %d %.17g\n", iter, dmfterr_equil);
        if (iter <= 2) {
            dump_hm("GLOC", iter, Gloc_up);
            dump_hm("DELTA", iter, Delta_up);
            dump_hm("DELTACC", iter, Delta_up_cc);
        }

        if (dmfterr_equil < errmax) {
            for (int s = 0; s < 4; s++) dump_pp("SIGC", iter, s, imp.ppSigmas[s]);
            for (int s = 0; s < 4; s++) dump_pp("PPGC", iter, s, imp.ppGfs[s]);
            dump_gf("SPGFC", iter, gf_tstps[0]);
            dump_hm("GLOCC", iter, Gloc_up);
            dump_hm("DELTAC", iter, Delta_up);
            dump_hm("DELTACCC", iter, Delta_up_cc);
            printf("CONVERGED %d\n", iter);
            break;
        }
    }

    imp.update_density_matrix(-1);
    imp.hamiltonian.update_exp_vals(-1, imp.rho);
    double nup, ndn, docc;
    imp.hamiltonian.local_obs(-1, nup, ndn, docc);
    printf("OBS_EQ %.17g %.17g %.17g %.17g\n", imp.pp_mu, nup, ndn, docc);
    return 0;
}
