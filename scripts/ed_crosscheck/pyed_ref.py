#!/usr/bin/env python3
"""External-ED reference: PyED (https://github.com/HugoStrand/pyed) on TRIQS
operators.

Rebuilds every model of an nQIST ED cross-check dump from the printed
parameters and the Hamiltonian formulas documented in nQIST
`src/hamilton.rs` - written here as TRIQS second-quantized operator
expressions, independent of the nQIST code - and recomputes the
records:

* equilibrium (Z, spectrum, observables, Matsubara G(tau)) with PyED's
  Lehmann representation,
* equilibrium real-time components from the eigen-decomposition,
* the quench records from the spectral overlap sums of a single sudden
  quench (the thermal state is the eigenbasis of H0, the propagation
  that of H1), an algorithm independent of nQIST's stepwise unitary
  evolution.

The Holstein model is built with PyED's fermion-boson Fock states.

Usage: pyed_ref.py <nqist_dump> <reference_dump>
"""

import sys

import numpy as np
from triqs.operators import Operator, c, c_dag

from pyed.SparseExactDiagonalization import SparseExactDiagonalization
from pyed.SparseMatrixFockStates import SparseMatrixFermiBoseCreationOperators
from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

from dumpformat import Writer, read_dump

UP, DOWN = 0, 1
KINDS = ("ret", "les", "gtr")


def mode_of(model, imp_mode):
    """`(spin, orbital)` of impurity mode `imp_mode` (`spin * norb + orb`)."""
    return divmod(imp_mode, model.norb)


def fermion_hamiltonian(model, p):
    """TRIQS operator expression of impurity + discrete bath."""
    norb = model.norb
    n = {(s, o): c_dag(s, o) * c(s, o) for o in range(norb) for s in (UP, DOWN)}
    big_n = {o: n[UP, o] + n[DOWN, o] for o in range(norb)}
    h = Operator()
    if model.variant == "SingleBand":
        # H = U n_up n_dn + (eps_up - mu - U/2) n_up + (eps_do - mu - U/2) n_dn
        h += p["u"] * n[UP, 0] * n[DOWN, 0]
        h += (p["eps_up"] - p["mu"] - p["u"] / 2) * n[UP, 0]
        h += (p["eps_do"] - p["mu"] - p["u"] / 2) * n[DOWN, 0]
    else:
        u, j = p["u"], p["j"]
        # H = Bz m + crystal field - mu n - half-filling shift
        #     + U sum_m n_mu n_md + (U - 2J) sum_{m<m'} n_m n_m'
        #     - J sum_{m<m'} (n_mu n_m'u + n_md n_m'd)  [+ spin-flip, pair-hopping]
        mag = sum((1 - 2 * s) * n[s, o] for o in range(norb) for s in (UP, DOWN))
        ntot = sum(big_n.values())
        crystal = big_n[0] - big_n[1] - (big_n[2] if norb == 3 else 0.0)
        shift = (5 * u - 10 * j) if norb == 3 else (3 * u - 5 * j)
        h += p["bz"] * mag + 0.5 * p["delta"] * crystal - p["mu"] * ntot
        h += -0.5 * shift * ntot
        h += u * sum(n[UP, o] * n[DOWN, o] for o in range(norb))
        for a in range(norb):
            for b in range(a + 1, norb):
                h += (u - 2 * j) * big_n[a] * big_n[b]
                h += -j * (n[UP, a] * n[UP, b] + n[DOWN, a] * n[DOWN, b])
                if "Kanamori" in model.variant:
                    # spin-flip  -J c^dag_{a u} c_{a d} c^dag_{b d} c_{b u} + h.c.
                    h += -j * c_dag(UP, a) * c(DOWN, a) * c_dag(DOWN, b) * c(UP, b)
                    h += -j * c_dag(UP, b) * c(DOWN, b) * c_dag(DOWN, a) * c(UP, a)
                    # pair-hopping  +J c^dag_{a u} c^dag_{a d} c_{b d} c_{b u} + h.c.
                    h += j * c_dag(UP, a) * c_dag(DOWN, a) * c(DOWN, b) * c(UP, b)
                    h += j * c_dag(UP, b) * c_dag(DOWN, b) * c(DOWN, a) * c(UP, a)
    for k, (imp_mode, eps, v) in enumerate(model.bath):
        s, o = mode_of(model, imp_mode)
        nb = c_dag("b", k) * c("b", k)
        h += eps * nb + v * (c_dag(s, o) * c("b", k) + c_dag("b", k) * c(s, o))
    return h


def holstein_matrices(model, p, lam):
    """`(H, c, c^dag, n, b^dag b)` as sparse matrices on the fermion tensor
    truncated-phonon space (PyED's fermion-boson Fock states)."""
    norb = model.norb
    nfermion = 2 * norb + len(model.bath)
    ops = SparseMatrixFermiBoseCreationOperators(nfermion, 1, model.phonon["nph"])
    cd = ops.c_dag
    c_ = [x.getH() for x in cd]
    n = [cd[i] * c_[i] for i in range(nfermion)]
    # H_imp = U n_up n_dn + (eps_up - mu - U/2) n_up + (eps_do - mu - U/2) n_dn
    h = p["u"] * n[0] * n[1]
    h = h + (p["eps_up"] - p["mu"] - p["u"] / 2) * n[0]
    h = h + (p["eps_do"] - p["mu"] - p["u"] / 2) * n[1]
    for k, (imp_mode, eps, v) in enumerate(model.bath):
        b = 2 * norb + k
        h = h + eps * n[b]
        h = h + v * (cd[imp_mode] * c_[b] + cd[b] * c_[imp_mode])
    bd, b = ops.b_dag[0], ops.b_dag[0].getH()
    h = h + model.phonon["omega"] * (bd * b) + lam * n[0] * n[1] * (b + bd)
    return h.tocsr(), c_, cd, n, (bd * b).tocsr()


def equilibrium_realtime(ed, c_mat, t):
    """`(ret, les, gtr)` at the times `t` of the state diagonalized in
    `ed` (PyED stores `E - E0` and the weights `e^{-beta E} / Z`).

        G^>(t) = -i sum_nm w_n |c_nm|^2 e^{i (E_n - E_m) t}
        G^<(t) =  i sum_nm w_m |c_nm|^2 e^{i (E_n - E_m) t},   G^R = G^> - G^<
    """
    vecs, energies = np.asarray(ed.U), ed.get_eigen_values()
    weights = np.exp(-ed.beta * energies) / ed.Z
    ce = vecs.conj().T @ c_mat @ vecs
    amp = np.abs(ce) ** 2  # |<n|c|m>|^2
    dim = energies.size
    d_e = (energies[:, None] - energies[None, :]).ravel()
    phase = np.exp(1j * np.outer(t, d_e)).reshape(t.size, dim, dim)
    gtr = -1j * np.einsum("n,nm,tnm->t", weights, amp, phase)
    les = 1j * np.einsum("m,nm,tnm->t", weights, amp, phase)
    return gtr - les, les, gtr


def quench_realtime(ed0, c_mat, h1, beta, t):
    """`(ret, les, gtr)[n, np]` on the grid `t` after `H0 -> H1` at `t = 0`;
    the thermal state is that of `H0` (`ed0`), the propagation that of `h1`.
    With `|a>`, `p_a` the eigenstates and weights of `H0` and `E1_b`,
    `|b>` those of `H1`,

        <b|c(t_n)|a>     = e^{i E1_b t_n} (C1 e^{-i E1 t_n} S)_ba
        <b|c^dag(t_n)|a> = e^{i E1_b t_n} (C1^dag e^{-i E1 t_n} S)_ba

    with the same Heisenberg evolution for both operators (`S_ba = <b|a>`,
    `C1 = <b|c|b'>` in the `H1` basis), and

        G^>(t_n, t_np) = -i sum_{ab} p_a conj(<b|c^dag(t_n)|a>) <b|c^dag(t_np)|a>
        G^<(t_n, t_np) =  i sum_{ab} p_a conj(<b|c(t_n)|a>)   <b|c(t_np)|a>
    """
    e1, v1 = np.linalg.eigh(np.asarray(h1))
    e0, v0 = ed0.get_eigen_values(), np.asarray(ed0.U)
    p_a = np.exp(-beta * e0) / ed0.Z
    c1 = v1.conj().T @ c_mat @ v1
    s = v1.conj().T @ v0
    dim = e1.size
    amp_c = np.empty((t.size, dim, dim), complex)
    amp_cd = np.empty((t.size, dim, dim), complex)
    for k, tk in enumerate(t):
        fwd = np.exp(1j * e1 * tk)
        bwd = np.exp(-1j * e1 * tk)
        amp_c[k] = fwd[:, None] * (c1 @ (bwd[:, None] * s))
        amp_cd[k] = fwd[:, None] * (c1.conj().T @ (bwd[:, None] * s))
    gtr = -1j * np.einsum("nba,a,mba->nm", amp_cd.conj(), p_a, amp_cd)
    les = 1j * np.conj(np.einsum("nba,a,mba->nm", amp_c.conj(), p_a, amp_c))
    return gtr - les, les, gtr


def emit_greens(model, out, mode, ed, c_mat, cd_mat, h1, beta):
    """Matsubara and real-time records of impurity mode `mode`;
    `c_mat` / `cd_mat` are the sparse annihilation / creation matrices."""
    name = model.name
    rows = model.tau_list(mode)
    g = ed.get_tau_greens_function_component(
        np.array([tau for _, tau in rows]), c_mat, cd_mat
    )
    for (i, tau), x in zip(rows, g):
        out.m_tau(name, mode, i, tau, x.real)
    c_dense = np.asarray(c_mat.todense())
    t = np.array(model.rteq_list("ret", mode))
    if [model.rteq_list(kind, mode) for kind in KINDS] != [list(t)] * 3:
        raise RuntimeError("equilibrium real-time grids differ per component")
    for kind, vals in zip(KINDS, equilibrium_realtime(ed, c_dense, t)):
        for x, y in zip(t, vals):
            out.rteq(name, kind, mode, x, y)
    grid = np.arange(model.nt + 1) * model.dt
    if [model.rt_list(kind, mode) for kind in KINDS] != [model.rt_list("ret", mode)] * 3:
        raise RuntimeError("quench real-time grids differ per component")
    for kind, vals in zip(KINDS, quench_realtime(ed, c_dense, h1, beta, grid)):
        for n, np_ in model.rt_list(kind, mode):
            out.rt(name, kind, mode, n, np_, vals[n, np_])


def process_fermions(model, p, q, out):
    """PyED/TRIQS path of the purely fermionic models (`q`: quenched parameters)."""
    name, beta = model.name, model.beta
    h_eq, h_q = fermion_hamiltonian(model, p), fermion_hamiltonian(model, q)
    fundamental = [c(s, o) for o in range(model.norb) for s in (UP, DOWN)]
    fundamental += [c("b", k) for k in range(len(model.bath))]
    ed = TriqsExactDiagonalization(h_eq, fundamental, beta)
    out.z(name, ed.get_partition_function(), ed.get_ground_state_energy())
    out.ev(name, np.sort(ed.get_ground_state_energy() + ed.ed.get_eigen_values()))
    out.obs(name, "energy", ed.get_expectation_value(h_eq).real)
    norb = model.norb
    n = {(s, o): c_dag(s, o) * c(s, o) for o in range(norb) for s in (UP, DOWN)}
    for mode in range(2 * norb):
        out.obs(name, "n{}".format(mode), ed.get_expectation_value(n[mode_of(model, mode)]).real)
    for a in range(norb):
        for b in range(norb):
            out.obs(
                name,
                "docc_{}_{}".format(a, b),
                ed.get_expectation_value(n[UP, a] * n[DOWN, b]).real,
            )
    h_q_dense = np.asarray(ed.rep.sparse_matrix(h_q).todense())
    for mode in range(2 * norb):
        s, o = mode_of(model, mode)
        emit_greens(
            model,
            out,
            mode,
            ed.ed,
            ed.rep.sparse_matrix(c(s, o)),
            ed.rep.sparse_matrix(c_dag(s, o)),
            h_q_dense,
            beta,
        )


def process_holstein(model, p, q, out):
    """Fermion-boson path of the Holstein model."""
    name, beta, norb = model.name, model.beta, model.norb
    lam_q = q["lambda"]
    h_eq, c_, cd, n, n_ph = holstein_matrices(model, p, model.phonon["lambda"])
    c_ = [x.tocsr() for x in c_]
    cd = [x.tocsr() for x in cd]
    h_q = holstein_matrices(model, p, lam_q)[0].todense()
    ed = SparseExactDiagonalization(h_eq, beta)
    out.z(name, ed.Z, ed.E0)
    out.ev(name, np.sort(ed.E0 + ed.E))
    out.obs(name, "energy", ed.get_expectation_value(h_eq).real)
    for mode in range(2 * norb):
        out.obs(name, "n{}".format(mode), ed.get_expectation_value(n[mode]).real)
    for a in range(norb):
        for b in range(norb):
            out.obs(
                name,
                "docc_{}_{}".format(a, b),
                ed.get_expectation_value(n[a] * n[norb + b]).real,
            )
    out.obs(name, "nphonon", ed.get_expectation_value(n_ph).real)
    for mode in range(2 * norb):
        emit_greens(
            model, out, mode, ed, c_[mode], cd[mode], h_q, beta
        )


def process(model, out):
    p = model.params
    q = dict(p, **{model.quench["param"]: model.quench["to"]})
    if model.phonon is None:
        process_fermions(model, p, q, out)
    else:
        process_holstein(model, p, q, out)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    out = Writer()
    for model in read_dump(sys.argv[1]).values():
        process(model, out)
    out.write(sys.argv[2])


if __name__ == "__main__":
    main()
