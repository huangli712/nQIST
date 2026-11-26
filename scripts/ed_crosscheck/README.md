# ED cross-check: nQIST's built-in ED against open-source ED packages

`nqist::ed` (`src/ed.rs`: `EdReference`, the equilibrium Lehmann
reference, and `EdEvolution`, the quenched unitary-evolution reference)
is the arbiter of every solver-vs-ED gate. This directory checks it
against two independent open-source exact-diagonalization packages:

- **PyED** (<https://github.com/HugoStrand/pyed>, pure Python on TRIQS
  operators): equilibrium records through PyED's Lehmann representation;
  real-time records through spectral sums built from the same
  eigen-decomposition (the equilibrium components, and the quench
  components from the overlap sums of a single sudden quench - an
  algorithm independent of the stepwise unitary evolution of
  `EdEvolution`).
- **Pomerol** (<https://github.com/pomerol-ed/pomerol>, C++ with
  libcommute): equilibrium records through its own Hilbert-space,
  symmetry-sector and Lehmann machinery, including the Holstein phonon
  as a bosonic mode. Pomerol has no real-time correlators.

Both rebuild the models from the parameters printed by nQIST and from
the Hamiltonian formulas documented in `src/hamilton.rs`, written as
operator expressions in their own algebra - nothing is shared with the
nQIST code except the numbers to compare.

## Files

| file | role |
|---|---|
| `../../examples/ed_crosscheck.rs` | dumps the nQIST reference (`cargo run --release --example ed_crosscheck --`) |
| `dumpformat.py` | record format of the dump (reader and writer) |
| `pyed_ref.py` | PyED reference |
| `pomerol_ref.cpp`, `CMakeLists.txt` | Pomerol reference |
| `compare.py` | joins two dumps record by record and prints the maximum deviations |
| `run.sh` | the whole cross-check |

## Models

The dimer fixtures of the solver gates, all at `beta = 5`: single-band
Hubbard (`tests/singleband_dimer`), two-band density-density and full
Kanamori (`tests/xca_twoband_dimer`, plus a `Bz = 0.3` variant),
three-band density-density and full Kanamori (`tests/threeband_dimer`,
plus a `Bz = 0.2` variant), and the Hubbard-Holstein dimer
(`tests/holstein_dimer`, phonon truncated to 16 levels so that all
three codes use the same truncation). Each model is dumped with its
spectrum, partition function, observables, Matsubara `G(tau)` on the
gate grid, the equilibrium real-time components, and the greater /
lesser / retarded components on a `13 x 13` time grid after one sudden
quench (`U`, `J`, `delta`, `Bz` or the e-ph coupling `lambda`).

## Environment

One conda environment serves both references (the `pomerol` package
lives on the `krivenko` channel and does not pull in its own header
dependencies):

```
conda create -p <env> -c conda-forge -c krivenko \
    python=3.12 triqs 'pomerol=2.3=openmpi*' eigen=3.4.0 libcommute=1.0.0 \
    libboost-devel openmpi gxx_linux-64 gcc_linux-64 cmake make
conda activate <env>
pip install --no-deps git+https://github.com/HugoStrand/pyed.git
```

## Run

```
scripts/ed_crosscheck/run.sh <work-dir>
```

`compare.py` prints the maximum deviation per record class and `PASS`
when every class is within tolerance (`1e-10` absolute for energies and
correlators, `1e-12` relative for the partition function).

## Result of 2026-11-25

| model | PyED: Z (rel) / spectrum / observ. / G(tau) / G_eq(t) / quench | Pomerol: Z (rel) / spectrum / observ. / G(tau) |
|---|---|---|
| singleband | 3.0e-15 / 7.4e-16 / 7.2e-16 / 1.1e-15 / 1.3e-15 / 1.0e-14 | 4.4e-15 / 2.0e-15 / 1.8e-15 / 2.2e-15 |
| twoband_density | 1.3e-14 / 3.6e-15 / 5.2e-15 / 6.2e-15 / 6.3e-15 / 1.1e-14 | 1.1e-15 / 2.7e-15 / 1.1e-15 / 1.3e-15 |
| twoband_kanamori | 1.1e-14 / 3.1e-15 / 3.1e-15 / 1.9e-15 / 2.4e-15 / 7.3e-15 | 5.7e-15 / 4.0e-15 / 8.3e-16 / 7.2e-16 |
| twoband_kanamori_bz | 2.4e-15 / 4.0e-15 / 4.4e-15 / 1.2e-15 / 2.8e-15 / 8.4e-15 | 9.2e-16 / 3.1e-15 / 1.1e-15 / 1.6e-15 |
| threeband_density | 2.4e-15 / 1.5e-14 / 3.0e-14 / 5.4e-15 / 6.9e-15 / 2.5e-14 | 3.0e-14 / 1.6e-14 / 1.3e-14 / 1.4e-14 |
| threeband_kanamori | 3.2e-14 / 2.8e-14 / 2.1e-14 / 7.7e-15 / 9.9e-15 / 6.2e-14 | 3.9e-16 / 3.6e-14 / 1.4e-14 / 1.2e-14 |
| threeband_kanamori_bz | 1.3e-15 / 3.6e-14 / 5.3e-15 / 6.3e-15 / 8.7e-15 / 3.0e-14 | 3.7e-15 / 3.6e-14 / 3.8e-15 / 6.4e-15 |
| holstein | 2.6e-14 / 2.3e-13 / 6.9e-15 / 5.7e-15 / 1.6e-14 / 3.2e-13 | 1.2e-14 / 2.0e-13 / 3.4e-15 / 9.8e-15 |

Both references: PASS. The hand-built two-band reference of
`tests/xca_twoband_dimer` (`twoband_exact`) is tied to this calibration
by `twoband_test_reference_matches_ed_module` (same file, `1e-12`).

Note on the PyED quench formula: it was debugged against its own
equilibrium formula in the no-quench limit (`H1 = H0`) and the
equal-time sum rules before the comparison above was accepted.
