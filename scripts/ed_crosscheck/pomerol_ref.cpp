// External-ED reference: Pomerol (https://github.com/pomerol-ed/pomerol).
//
// Rebuilds every model of an nQIST ED cross-check dump from the printed
// parameters and the Hamiltonian formulas documented in nQIST
// `src/hamilton.rs` - written here as libcommute operator expressions,
// independent of the nQIST code - and recomputes the equilibrium
// records (Z, spectrum, observables, Matsubara G(tau)) with Pomerol's
// Lehmann representation. Pomerol has no real-time correlators, so the
// RTEQ / RT records are left to the PyED reference. The Holstein phonon
// is a Pomerol bosonic mode truncated to `nph` levels (`nph` a power of
// two).
//
// Usage: pomerol_ref <nqist_dump> <reference_dump>

#include <pomerol.hpp>

#include <mpi_dispatcher/mpi_dispatcher.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace Pomerol;
using LatticePresets::down;
using LatticePresets::spin;
using LatticePresets::undef;
using LatticePresets::up;
using LatticePresets::RealExpr;

namespace {

struct Bath {
    unsigned imp_mode;
    double eps, v;
};

struct Phonon {
    unsigned nph;
    double omega, lambda;
    std::vector<std::pair<unsigned, bool>> ops;  // (impurity mode, creation)
};

struct Model {
    std::string name, variant;
    unsigned norb = 0, nbath = 0;
    double beta = 0;
    std::map<std::string, double> param;
    std::vector<Bath> bath;
    bool has_phonon = false;
    Phonon phonon;
    std::vector<std::array<double, 4>> m_tau;  // mode, i, tau, g
};

std::string orbital_site(unsigned orb) {
    return std::to_string(orb);
}

std::string bath_site(unsigned k) {
    return "b" + std::to_string(k);
}

spin mode_spin(unsigned mode, unsigned norb) {
    return (mode / norb == 0) ? up : down;
}

unsigned mode_orb(unsigned mode, unsigned norb) {
    return mode % norb;
}

/// `key=value` tokens of the rest of the line.
std::map<std::string, std::string> key_values(std::istringstream& ls) {
    std::map<std::string, std::string> kv;
    std::string item;
    while(ls >> item) {
        const size_t eq = item.find('=');
        kv[item.substr(0, eq)] = item.substr(eq + 1);
    }
    return kv;
}

RealExpr number(unsigned mode, unsigned norb) {
    return Operators::n(orbital_site(mode_orb(mode, norb)), static_cast<unsigned short>(0),
                        mode_spin(mode, norb));
}

RealExpr creation(unsigned mode, unsigned norb) {
    return Operators::c_dag(orbital_site(mode_orb(mode, norb)), static_cast<unsigned short>(0),
                            mode_spin(mode, norb));
}

RealExpr annihilation(unsigned mode, unsigned norb) {
    return Operators::c(orbital_site(mode_orb(mode, norb)), static_cast<unsigned short>(0),
                        mode_spin(mode, norb));
}

/// Impurity + bath (+ phonon) Hamiltonian, documented formulas.
RealExpr build_hamiltonian(const Model& m) {
    const double mu = m.param.at("mu");
    const double u = m.param.at("u");
    const double j = m.param.at("j");
    const double delta = m.param.at("delta");
    const double bz = m.param.at("bz");
    const unsigned norb = m.norb;
    RealExpr h;
    if(m.variant == "SingleBand") {
        // H = U n_up n_dn + (eps_up - mu - U/2) n_up + (eps_do - mu - U/2) n_dn
        const double eps_up = m.param.at("eps_up");
        const double eps_do = m.param.at("eps_do");
        h += u * number(0, norb) * number(1, norb);
        h += (eps_up - mu - 0.5 * u) * number(0, norb);
        h += (eps_do - mu - 0.5 * u) * number(1, norb);
    } else {
        // H = Bz m + crystal field - mu n - half-filling shift
        //     + U sum_m n_mu n_md + (U - 2J) sum_{m<m'} n_m n_m'
        //     - J sum_{m<m'} (n_mu n_m'u + n_md n_m'd)  [+ spin-flip, pair-hopping]
        RealExpr mag, ntot, crystal;
        std::vector<RealExpr> big_n(norb);
        for(unsigned o = 0; o < norb; ++o) {
            big_n[o] = number(o, norb) + number(norb + o, norb);
            mag += number(o, norb) - number(norb + o, norb);
            ntot += big_n[o];
        }
        double shift = 0.5 * (3 * u - 5 * j);
        crystal = big_n[0] - big_n[1];
        if(norb == 3) {
            crystal -= big_n[2];
            shift = 0.5 * (5 * u - 10 * j);
        }
        h += bz * mag + 0.5 * delta * crystal - mu * ntot - shift * ntot;
        for(unsigned o = 0; o < norb; ++o) {
            h += u * number(o, norb) * number(norb + o, norb);
        }
        const bool kanamori = m.variant.find("Kanamori") != std::string::npos;
        for(unsigned a = 0; a < norb; ++a) {
            for(unsigned b = a + 1; b < norb; ++b) {
                h += (u - 2 * j) * big_n[a] * big_n[b];
                h -= j * (number(a, norb) * number(b, norb) +
                          number(norb + a, norb) * number(norb + b, norb));
                if(kanamori) {
                    // spin-flip  -J c^dag_{a u} c_{a d} c^dag_{b d} c_{b u} + h.c.
                    h -= j * (creation(a, norb) * annihilation(norb + a, norb) *
                              creation(norb + b, norb) * annihilation(b, norb));
                    h -= j * (creation(b, norb) * annihilation(norb + b, norb) *
                              creation(norb + a, norb) * annihilation(a, norb));
                    // pair-hopping  +J c^dag_{a u} c^dag_{a d} c_{b d} c_{b u} + h.c.
                    h += j * (creation(a, norb) * creation(norb + a, norb) *
                              annihilation(norb + b, norb) * annihilation(b, norb));
                    h += j * (creation(b, norb) * creation(norb + b, norb) *
                              annihilation(norb + a, norb) * annihilation(a, norb));
                }
            }
        }
    }
    for(unsigned k = 0; k < m.nbath; ++k) {
        const Bath& b = m.bath[k];
        const std::string sb = bath_site(k);
        const unsigned short orb = 0;
        h += b.eps * Operators::n(sb, orb, up);
        h += b.v * (creation(b.imp_mode, norb) * Operators::c(sb, orb, up) +
                    Operators::c_dag(sb, orb, up) * annihilation(b.imp_mode, norb));
    }
    if(m.has_phonon) {
        RealExpr coupling;  // the ordered fermion string (a number operator product)
        bool first = true;
        for(const auto& op : m.phonon.ops) {
            const RealExpr factor =
                op.second ? creation(op.first, norb) : annihilation(op.first, norb);
            coupling = first ? factor : coupling * factor;
            first = false;
        }
        const unsigned short extra = 0;
        const RealExpr a_dag = Operators::a_dag("ph", extra, undef);
        const RealExpr a = Operators::a("ph", extra, undef);
        h += m.phonon.omega * a_dag * a;
        h += m.phonon.lambda * coupling * (a_dag + a);
    }
    return h;
}

std::vector<Model> read_dump(const std::string& path) {
    std::vector<Model> models;
    std::ifstream in(path);
    if(!in) throw std::runtime_error("cannot open " + path);
    std::string line;
    Model cur;
    bool open = false;
    while(std::getline(in, line)) {
        std::istringstream ls(line);
        std::string tag;
        if(!(ls >> tag)) continue;
        if(tag == "MODEL") {
            if(open) models.push_back(cur);
            cur = Model();
            open = true;
            ls >> cur.name;
            for(const auto& kv : key_values(ls)) {
                if(kv.first == "variant") cur.variant = kv.second;
                else if(kv.first == "norb") cur.norb = std::stoul(kv.second);
                else if(kv.first == "beta") cur.beta = std::stod(kv.second);
                else if(kv.first == "nbath") cur.nbath = std::stoul(kv.second);
            }
        } else if(tag == "PARAM") {
            std::string name;
            ls >> name;
            for(const auto& kv : key_values(ls)) cur.param[kv.first] = std::stod(kv.second);
        } else if(tag == "BATH") {
            std::string name;
            ls >> name;
            const auto kv = key_values(ls);
            cur.bath.push_back(Bath{static_cast<unsigned>(std::stoul(kv.at("imp_mode"))),
                                    std::stod(kv.at("eps")), std::stod(kv.at("v"))});
        } else if(tag == "PHONON") {
            std::string name;
            ls >> name;
            const auto kv = key_values(ls);
            cur.has_phonon = true;
            cur.phonon.nph = std::stoul(kv.at("nph"));
            cur.phonon.omega = std::stod(kv.at("omega"));
            cur.phonon.lambda = std::stod(kv.at("lambda"));
            std::istringstream os(kv.at("ops"));
            std::string item;
            while(std::getline(os, item, ',')) {
                const size_t colon = item.find(':');
                cur.phonon.ops.emplace_back(std::stoul(item.substr(0, colon)),
                                            std::stoul(item.substr(colon + 1)) != 0);
            }
        } else if(tag == "M") {
            std::string name;
            std::array<double, 4> row{};
            ls >> name >> row[0] >> row[1] >> row[2] >> row[3];
            cur.m_tau.push_back(row);
        }
        // Z / EV / OBS / QUENCH / RTEQ / RT are nQIST results, not inputs
    }
    if(open) models.push_back(cur);
    return models;
}

double ensemble_average(EnsembleAverage& avg) {
    avg.compute();
    return real(avg());
}

void process(const Model& m, std::ostream& out) {
    const RealExpr h_expr = build_hamiltonian(m);
    auto index_info = MakeIndexClassification(h_expr);
    const unsigned bits =
        m.has_phonon ? static_cast<unsigned>(std::lround(std::log2(m.phonon.nph))) : 1;
    auto hs = MakeHilbertSpace(index_info, h_expr, bits);
    hs.compute();
    StatesClassification s;
    s.compute(hs);
    Hamiltonian ham(s);
    ham.prepare(h_expr, hs, MPI_COMM_WORLD);
    ham.compute(MPI_COMM_WORLD);

    // spectrum and partition function (shifted by the ground energy)
    const auto& spectrum = ham.getEigenValues();
    std::vector<double> ev(spectrum.data(), spectrum.data() + spectrum.size());
    std::sort(ev.begin(), ev.end());
    const double emin = ev.front();
    double z = 0, energy = 0;
    for(const double e : ev) {
        const double w = std::exp(-m.beta * (e - emin));
        z += w;
        energy += w * e;
    }
    out << "Z " << m.name << " " << z << " " << emin << "\n";
    out << "EV " << m.name;
    for(const double e : ev) out << " " << e;
    out << "\n";
    out << "OBS " << m.name << " energy " << energy / z << "\n";

    DensityMatrix rho(s, ham, m.beta);
    rho.prepare();
    rho.compute();
    FieldOperatorContainer ops(index_info, hs, s, ham);
    ops.prepareAll(hs);
    ops.computeAll();

    const unsigned nmodes = 2 * m.norb;
    std::vector<ParticleIndex> index(nmodes);
    std::vector<double> density(nmodes);
    for(unsigned mode = 0; mode < nmodes; ++mode) {
        index[mode] = index_info.getIndex(orbital_site(mode_orb(mode, m.norb)),
                                          static_cast<unsigned short>(0), mode_spin(mode, m.norb));
        QuadraticOperator n_op(index_info, hs, s, ham, index[mode], index[mode]);
        n_op.prepare(hs);
        n_op.compute();
        EnsembleAverage n_avg(n_op, rho);
        density[mode] = ensemble_average(n_avg);
        out << "OBS " << m.name << " n" << mode << " " << density[mode] << "\n";
    }
    for(unsigned a = 0; a < m.norb; ++a) {
        for(unsigned b = 0; b < m.norb; ++b) {
            // <n_{a up} n_{b down}> = <c^dag_{a up} c^dag_{b down} c_{b down} c_{a up}>
            const ParticleIndex ia = index[a], ib = index[m.norb + b];
            QuarticOperator d_op(index_info, hs, s, ham, ia, ib, ib, ia);
            d_op.prepare(hs);
            d_op.compute();
            EnsembleAverage d_avg(d_op, rho);
            out << "OBS " << m.name << " docc_" << a << "_" << b << " " << ensemble_average(d_avg)
                << "\n";
        }
    }

    for(unsigned mode = 0; mode < nmodes; ++mode) {
        GreensFunction gf(s, ham, ops.getAnnihilationOperator(index[mode]),
                          ops.getCreationOperator(index[mode]), rho);
        gf.CoefficientTolerance = 1e-16;
        gf.prepare();
        gf.compute();
        std::vector<std::array<double, 4>> rows;
        for(const auto& r : m.m_tau) {
            if(static_cast<unsigned>(r[0]) == mode) rows.push_back(r);
        }
        std::sort(rows.begin(), rows.end(),
                  [](const std::array<double, 4>& x, const std::array<double, 4>& y) {
                      return x[1] < y[1];
                  });
        for(const auto& r : rows) {
            const ComplexType g = gf.of_tau(r[2]);
            if(std::abs(g.imag()) > 1e-12 * std::max(1.0, std::abs(g.real()))) {
                throw std::runtime_error(m.name + ": G(tau) is not real at tau = " +
                                         std::to_string(r[2]));
            }
            out << "M " << m.name << " " << mode << " " << static_cast<unsigned>(r[1]) << " " << r[2]
                << " " << g.real() << "\n";
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int status = 0;
    if(pMPI::rank(MPI_COMM_WORLD) == 0) {
        try {
            if(argc != 3) throw std::runtime_error("usage: pomerol_ref <nqist_dump> <reference_dump>");
            std::ofstream out(argv[2]);
            if(!out) throw std::runtime_error(std::string("cannot write ") + argv[2]);
            out << std::setprecision(17) << std::scientific;
            for(const Model& m : read_dump(argv[1])) process(m, out);
            if(!out) throw std::runtime_error("write error on the reference dump");
        } catch(const std::exception& e) {
            std::cerr << "pomerol_ref: " << e.what() << "\n";
            status = 1;
        }
    }
    MPI_Finalize();
    return status;
}
