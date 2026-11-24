// Synthetic-input OCA gf kernel comparison: get_spgf(-1) at order 2 minus
// order 1 isolates the OCA correction for hand-set Matsubara profiles.
#define NCA_SOLVER_ASSERT_0 0
#define NCA_SOLVER_ASSERT_1 0

#include <complex>
#include <cstdio>

#include "cntr.hpp"
#include "./ppsc/ppsc.hpp"
#include "./ppsc/solver.hpp"
#include "./ppsc/hilbert_spaces/single_band_fermi_diag.hpp"
#include "./ppsc/hamiltonians/single_band_hubbard.hpp"

typedef ppsc::hilbert_spaces::single_band_fermi_diag hilbert_space_type;
typedef ppsc::hamiltonians::single_band_hubbard<hilbert_space_type> hamiltonian_type;

static ppsc::pp_ints_type get_pp_ints_probe9(ppsc::gf_type &D, ppsc::gf_type &Dcc, hilbert_space_type &hil) {
    ppsc::operator_type cua = hil.c_op_[hil.flavor(0, 0, 0)];
    ppsc::operator_type cuc = hil.c_op_[hil.flavor(0, 0, 1)];
    ppsc::operator_type cda = hil.c_op_[hil.flavor(0, 1, 0)];
    ppsc::operator_type cdc = hil.c_op_[hil.flavor(0, 1, 1)];
    ppsc::pp_ints_type pp;
    int fermion = -1, fwd = 1, bwd = -1;
    pp.push_back(ppsc::pp_int_type(D, cuc, cua, fermion, fwd));
    pp.push_back(ppsc::pp_int_type(Dcc, cua, cuc, fermion, bwd));
    pp.push_back(ppsc::pp_int_type(D, cdc, cda, fermion, fwd));
    pp.push_back(ppsc::pp_int_type(Dcc, cda, cdc, fermion, bwd));
    return pp;
}
static ppsc::gf_verts_type get_gf_verts_probe9(hilbert_space_type &hil) {
    ppsc::operator_type cua = hil.c_op_[hil.flavor(0, 0, 0)];
    ppsc::operator_type cuc = hil.c_op_[hil.flavor(0, 0, 1)];
    ppsc::gf_verts_type v;
    v.push_back(ppsc::gf_vert_type(0, 0, cua, cuc));
    return v;
}

static void standalone_middle_UNUSED(int ntau, int kt, double beta,
                              std::vector<std::vector<std::complex<double>>> &ppg,
                              std::vector<std::complex<double>> &delta) {
  int mmax = ntau;
  int m = 20;
  std::vector<std::complex<double>> f_dc(mmax - m + 1), f_da(m + 1);
  for (int i1 = 0; i1 <= m; i1++) {
    for (int i2 = m; i2 <= mmax; i2++) {
      std::complex<double> gd_flip = -1.0 * ppg[0][mmax - i2]; // sd=0: sig=+1
      f_dc[i2 - m] = delta[mmax - i2 + i1] * (gd_flip * ppg[1][i2 - m]);
    }
    // gregory via the integrator
    std::complex<double> idc(0.0, 0.0);
    for (int l = 0; l <= mmax - m; l++)
      idc += integration::I<double>(kt).gregory_weights(mmax - m, l) * f_dc[l];
    f_da[i1] = idc * ppg[3][m - i1] * ppg[2][i1];
  }
  std::complex<double> ida(0.0, 0.0);
  for (int l = 0; l <= m; l++)
    ida += integration::I<double>(kt).gregory_weights(m, l) * f_da[l];
  std::complex<double> g = -(ida); // (ida * op4).trace() with op4 = +1
  double pref = (beta * beta) / (mmax * mmax);
  printf("STANDALONE %.17g %.17g (raw g %.17g %.17g)\n",
         (g * pref).real(), (g * pref).imag(), g.real(), g.imag());
  // intermediates for cross-check
  {
    int i1 = 0;
    std::complex<double> gd_flip = -1.0 * ppg[0][mmax - m];
    std::complex<double> v0 = delta[mmax - m + i1] * (gd_flip * ppg[1][0]);
    printf("STAND fdc0 %.17g %.17g  gd_flip %.17g %.17g  lam(%d) %.17g %.17g  gc0 %.17g %.17g\n",
           v0.real(), v0.imag(), gd_flip.real(), gd_flip.imag(), mmax - m + i1,
           delta[mmax - m + i1].real(), delta[mmax - m + i1].imag(),
           ppg[1][0].real(), ppg[1][0].imag());
  }
}

static void fill_and_eval(int order, int dump, int zero_mode) { // 0 both, 1 dcc=0, 2 d=0
    int nt = 8, ntau = 250, kt = 5;
    double beta = 5.0, h = 0.02;
    hilbert_space_type hil; hil.init();
    ppsc::solver<hamiltonian_type> imp(nt, ntau, beta, h, kt, 1, hil, order);
    imp.hamiltonian.mu = 0.0;
    imp.hamiltonian.U.assign(nt + 2, 3.0);
    imp.hamiltonian.eps_up.assign(nt + 2, 0.0);
    imp.hamiltonian.eps_do.assign(nt + 2, 0.0);
    imp.update_hamiltonian();
    imp.solve_atomic();

    cntr::herm_matrix<double> Delta(nt, ntau, 1, -1), Delta_cc(nt, ntau, 1, -1);
    // synthetic smooth profiles
    for (int m = 0; m <= ntau; m++) {
        double t = m / (double)ntau;
        std::complex<double> d0(-std::exp(-1.5 * t) * (1.0 + 0.3 * t), 0.2 * t * (1.0 - t));
        std::complex<double> d = (zero_mode == 2 ? std::complex<double>(0.0, 0.0) : d0);
        std::complex<double> dcc = (zero_mode == 1 ? std::complex<double>(0.0, 0.0) : std::conj(d0));
        Delta.set_mat(m, d);
        Delta_cc.set_mat(m, dcc);
        for (int s = 0; s < 4; s++) {
            double re = 0.1 * (s + 1) * std::exp(-0.8 * t) + 0.05 * s * t;
            double im = 0.02 * s * t * (1.0 - t) - 0.5 * t;
            imp.ppGfs[s].set_mat(m, std::complex<double>(re, im));
        }
    }
    ppsc::pp_ints_type pp_ints = get_pp_ints_probe9(Delta, Delta_cc, hil);
    ppsc::gf_verts_type gf_verts = get_gf_verts_probe9(hil);
    imp.update_diagrams(pp_ints, gf_verts);
    ppsc::gf_tstps_type gf = imp.get_spgf(-1);
    if (dump) {
        for (int m = 0; m <= ntau; m++) {
            std::complex<double> x;
            gf[0].get_mat(m, x);
            printf("SPGF%d_%d %d %.17g %.17g\n", order, zero_mode, m, x.real(), x.imag());
        }
        if (zero_mode == 1) {
            std::vector<std::vector<std::complex<double>>> ppg(4, std::vector<std::complex<double>>(ntau + 1));
            std::vector<std::complex<double>> d(ntau + 1);
            for (int mm = 0; mm <= ntau; mm++) {
                double t = mm / (double)ntau;
                d[mm] = std::complex<double>(-std::exp(-1.5 * t) * (1.0 + 0.3 * t), 0.2 * t * (1.0 - t));
                for (int s2 = 0; s2 < 4; s2++) {
                    double re = 0.1 * (s2 + 1) * std::exp(-0.8 * t) + 0.05 * s2 * t;
                    double im = 0.02 * s2 * t * (1.0 - t) - 0.5 * t;
                    ppg[s2][mm] = std::complex<double>(re, im);
                }
            }
            standalone_middle_UNUSED(ntau, kt, beta, ppg, d);
        }
    }
}

// standalone middle-branch replication of gdiagram_mat_integral
// (ga=ppGfs[2], gb=ppGfs[3], gc=ppGfs[1], gd=ppGfs[0], lam=Delta,
//  ops product for the fwd-dn diagram = (+1)*(+1)*(-1)*(+1) = -1, prefactor -1)
int main() {
    fill_and_eval(1, 1, 0);
    fill_and_eval(2, 1, 0);
    fill_and_eval(1, 1, 1);
    fill_and_eval(2, 1, 1);
    fill_and_eval(1, 1, 2);
    fill_and_eval(2, 1, 2);
    return 0;
}
