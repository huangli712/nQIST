# PPSC reference data

Reference outputs of the C++ PPSC programs (`../KadanoffBaym/.nessi/ppsc-port`)
used by `tests/ppsc_ref` and by unit tests in `src/`. Several files are
`include_str!`-ed at compile time, so a missing file breaks the build: the
tracked set below is the minimum needed to build and run the test suite.

Everything here is tracked except `quench_u35_oca/data_ppsc.h5` (~8 MB HDF5
program output that no code reads).

## Cases

| dir | model / method | files | consumed by |
|---|---|---|---|
| quench_u35 | single-band Hubbard/Bethe, NCA, U quench 3.0 -> 3.5 | case.inp, case.toml, run.log, golden_eq.txt, golden_rt.txt, probe3_golden.cpp, probe3b_golden.cpp | `ppsc_ref::equilibrium_matches_reference`, `ppsc_ref::realtime_matches_reference`; `src/solver.rs` unit test |
| quench_u35_oca | same, OCA, full-precision dump (452k lines, 27 MB) | case.inp, run.log, golden_oca.txt, sym_eq.txt, sym_rt.txt, probe3c_golden.cpp | `ppsc_ref::oca_equilibrium_matches_reference`, `ppsc_ref::oca_realtime_matches_reference` (ignored) |
| oca_tiny | single-band OCA tiny probe (component-level 17-digit dump) | golden_tiny.txt, probe10.cpp | `ppsc_ref::oca_realtime_tiny_matches_reference` |
| oca_tiny_fine | same probe on the fine grid (ntau = 250) | golden_tiny_fine.txt, probe10_fine.cpp | `ppsc_ref::oca_realtime_tiny_fine_matches_reference`, `ppsc_ref::xca_oca_realtime_tiny_fine_matches_reference` |
| two_band_u35 | two-band Kanamori, NCA | case.inp, case.toml, run.log | `ppsc_ref::two_band_matches_reference` |
| two_band_oca | two-band Kanamori, OCA | case.inp, case.toml, golden.txt | `ppsc_ref::two_band_oca_matches_reference` (heavy tier; reference from the sign-patched C++ build) |
| three_band_u35 | three-band Kanamori, NCA | case.inp, case.toml, golden.txt | `ppsc_ref::three_band_matches_reference` |
| three_band_u35_oca | three-band Kanamori, OCA | case.inp, case.toml, golden.txt | `ppsc_ref::three_band_oca_matches_reference` (heavy tier; reference from the sign-patched C++ build) |
| two_band_bz_delta | two-band Kanamori, static Bz = 0.3 and delta = 0.5, NCA | case.inp, case.toml, golden.txt | `ppsc_ref::two_band_bz_delta_matches_reference` |
| two_band_bz_delta_oca | same, OCA | case.inp, case.toml, golden.txt | `ppsc_ref::two_band_bz_delta_oca_matches_reference` (heavy tier; reference from the sign-patched C++ build) |
| two_band_hop_quench | two-band Kanamori, hopping quench t 1.0 -> 0.8, NCA | case.inp, golden.txt | `ppsc_ref::two_band_hop_quench_matches_reference` |
| two_band_hop_quench_oca | same, OCA | case.inp, golden.txt | `ppsc_ref::two_band_hop_quench_oca_matches_reference` (heavy tier; reference from the sign-patched C++ build) |
| three_band_bz_delta | three-band Kanamori, static Bz = 0.3 and delta = 0.5, NCA | case.inp, case.toml, golden.txt | `ppsc_ref::three_band_bz_delta_matches_reference` |
| three_band_bz_delta_oca | same, OCA | case.inp, case.toml, golden.txt | `ppsc_ref::three_band_bz_delta_oca_matches_reference` (heavy tier; reference from the sign-patched C++ build) |
| three_band_hop_quench | three-band Kanamori, hopping quench t 1.0 -> 0.8, NCA | case.inp, golden.txt | `ppsc_ref::three_band_hop_quench_matches_reference` |
| three_band_hop_quench_oca | same, OCA | case.inp, golden.txt | `ppsc_ref::three_band_hop_quench_oca_matches_reference` (heavy tier; reference from the sign-patched C++ build) |
| holstein_u46 | single-band Hubbard-Holstein, NCA (g_ppsc = 0.3) | case.inp, case.toml, useries.txt, golden.txt | `ppsc_ref::holstein_matches_reference` |
| holstein_u46_oca | same, OCA | case.inp, case.toml, useries.txt, golden.txt | `ppsc_ref::holstein_oca_matches_reference` |
| holstein_u46_oca_fine | same, OCA, fine grid (ntau = 250) | case.inp, case.toml, useries.txt, golden.txt | `ppsc_ref::holstein_oca_fine_matches_reference`, `ppsc_ref::holstein_xca_oca_fine_matches_reference` |
| prb96_photodoping | photodoping reference trajectory | case.inp, case.toml, tseries.txt, run.log | `ppsc_ref::prb96_photodoping_matches_reference` (ignored) |
| probe9_spgf.cpp/.txt | single-particle GF component probe | source + dump | `src/oca.rs` unit test |
| ppsc_oca_gf_statistics_sign.patch | C++ library patch: statistics sign of the OCA one-line vertex diagram (multi-orbital models) | patch + report | applied to the C++ port; see "Build the C++ programs" |

Per-case details (C++ source file, sha256, exact run command, compared line
kinds) are in each `<case>/README.txt`.

The four `*_bz_delta` cases lift the spin and orbital degeneracies (static
Bz = 0.3, delta = 0.5) and the four `*_hop_quench` cases quench the hopping
t 1.0 -> 0.8, so the Bethe Weiss field Delta(t, t') = t(t) t(t') G_loc(t, t')
is not translation invariant; they are the parameter points outside the
symmetric base cases. Three conventions matter there:

- The C++ drivers measure the DMFT convergence error on channel 0 only
  (`distance_norm2(-1, gtmp, Gloc.o1up)`, and the same at `kt` for the
  START window). `nqist::dmft::run_equilibrium_multi` and
  `run_start_window_multi` follow that criterion; the channels only differ
  once Bz or delta lifts the degeneracy.
- The C++ three-band driver prints `n1 + 2 n2` in the `n` column (it prints
  `n2` twice instead of `n3`); nQIST compares the same combination.
- The `*_hop_quench` cases have no `case.toml`: the examples take a static
  hopping, so the hopping quench is covered by the tests only.

Known gap: `examples/single_band_hubbard_afm_bethe.rs` defaults to
`data/ppsc/afm_bethe/case.toml`, which does not exist; pass an explicit path.

## Build the C++ programs

The programs are the stock PPSC sources in
`../KadanoffBaym/.nessi/ppsc-port/ppsc_programs/` (KadanoffBaym git repo; the
references were produced at commit 03fa63f and the ppsc-port subtree has not
changed since). The port's `CMakeLists.txt` hardcodes:

- NESSi/libcntr install: `<port>/nessi-install`
- Eigen: `<port>/../eigen3-install`
- HDF5: `/opt/anaconda3`

Build (the build dir already exists; an out-of-tree rebuild is fine):

```
port=../KadanoffBaym/.nessi/ppsc-port
cmake -S "$port" -B "$port/build"
cmake --build "$port/build" -j
```

Binaries land in `$port/build/ppsc_programs/<program>.ex` (one per source in
`ppsc_programs/CMakeLists.txt`).

The port's library carries one local change, applied on 2026-11-26 and
followed by a full rebuild of `build/`:
`ppsc_oca_gf_statistics_sign.patch` (this directory) fixes the statistics
sign of the OCA one-line vertex diagram of the Green's function
(`ppsc/oca/gf_diag.hpp`, `gf.cpp`, `gf_mat.hpp`, `gf_disp.cpp`), which the
pristine code takes from the particle-number parity of the last sector
instead of the hybridization-line direction. The pristine sign is right on
one orbital only; on two and three orbitals it makes the OCA Green's
function worse than NCA (1.9e-2 against the exact diagonalization on the
two-band dimer, 6.7e-4 with the sign fixed). The port is gitignored in the
KadanoffBaym repository, so the patch file is the versioned record; the
pristine library is kept in `$port/ppsc-port-pristine.tar.gz`. Single-band
outputs are bit-identical with and without the patch; the `two_band_oca`
and `three_band_u35_oca` references were regenerated with the patched
programs (they no longer match the pristine ones).

The binaries embed an absolute RPATH; if the checkout moves, set

```
export LD_LIBRARY_PATH="$port/build/ppsc:$port/build/ppsc/nca:$port/build/ppsc/oca:$port/nessi-install/lib:/opt/anaconda3/lib"
```

(needed: libppsc, libnca, liboca, libcntr, libcntr_hdf5, libhdf5).

## Regenerate one reference

```
work=$(mktemp -d)
cp <case>/case.inp "$work"/
cp <case>/useries.txt <case>/tseries.txt "$work"/ 2>/dev/null || true
cd "$work" && <binary> case.inp > run.log
```

The deck's `__*` keys are the program inputs; `useries.txt` / `tseries.txt` are
referenced from inside the deck. Keep the compared line kinds listed in
`<case>/README.txt` as `golden*.txt`. Before refreshing a golden, diff the new
output against the tracked one at the budget the tests use (1e-6 absolute or
1e-5 relative).

## Run the gates

```
cargo test --release --test ppsc_ref                  # fast gates (24: 15 PPSC + 9 XCA twins)
cargo test --release --test ppsc_ref -- --ignored     # heavy gates (10; run them through the script instead)
scripts/heavy_gates.sh                                # selected heavy OCA tier + XCA(2) twins
scripts/heavy_gates.sh --list                         # list the tier and ETAs
scripts/heavy_gates.sh --background                   # detached run (does not block)
scripts/heavy_gates.sh --status                       # run state + progress tail
cargo run --release --example <example> -- data/ppsc/<case>/case.toml
```

The selected heavy OCA tier is `scripts/heavy_gates.sh`
(`xca_twoband_dimer::kanamori_dimer_realtime_oca_vs_ed`,
`threeband_dimer::threeband_dimer_realtime_oca_vs_ed`,
`threeband_dimer::threeband_kanamori_realtime_oca_vs_ed`,
`ppsc_ref::oca_realtime_matches_reference`, each followed by its
`..._xca_...` twin, plus the multi-band OCA gates against the C++
references: `ppsc_ref::two_band_oca_matches_reference`,
`ppsc_ref::three_band_oca_matches_reference` and their static-Bz/delta
and hopping-quench variants `ppsc_ref::two_band_bz_delta_oca_matches_reference`,
`ppsc_ref::two_band_hop_quench_oca_matches_reference`,
`ppsc_ref::three_band_bz_delta_oca_matches_reference`,
`ppsc_ref::three_band_hop_quench_oca_matches_reference`); run it as a
separate CI job so the default suite stays fast. `--background` detaches the run
(log `target/heavy_gates/latest.log`); `--status` tails it. The script
exports `NQIST_PROGRESS=1` and passes `--nocapture`, so the DMFT
iteration and solver solve timings stream while it runs.

## Solver-selection parity gates

Every PPSC reference gate here runs twice: with the PPSC-like solver
(`Solver::Ppsc`, order 1 = NCA, order 2 = OCA) and with the XCA series
(`Solver::Xca`, order 1 = XCA(1), order 2 = XCA(1)+XCA(2)), through the
same check function and the same (native) tolerances. The twins are
named after the native gate with an `xca` infix
(`xca_equilibrium_matches_reference`,
`xca_oca_equilibrium_matches_reference`, `xca_realtime_matches_reference`,
`xca_oca_realtime_tiny_fine_matches_reference`,
`two_band_xca_matches_reference`, `holstein_xca_matches_reference`,
`holstein_xca_oca_fine_matches_reference`,
`three_band_xca_matches_reference`,
`prb96_xca_photodoping_matches_reference`, and the heavy
`xca_oca_realtime_matches_reference`; the multi-band OCA twins
`two_band_xca_oca_matches_reference` and
`three_band_xca_oca_matches_reference` run in reference-only mode until
XCA parity is settled - see "Multi-band OCA references" below).

Coarse-grid references (`oca_tiny`, `holstein_u46_oca`, ntau = 40 or 64)
have PPSC-like gates only. The XCA
series evaluates its skeletons on a sum-of-poles representation of the
hybridization, and the coarse tau grid carries O(1e-5) discretization
roughness that no sum of poles follows below rms 5e-6 (independent
multistart least squares with up to 18 poles finds the same floor), so
the XCA fixed point cannot reach the native tolerances there. The XCA
twins of the coarse references therefore run on fine-grid references
regenerated with the C++ programs at the native tolerances
(`oca_tiny_fine`, `holstein_u46_oca_fine`; both ntau = 250).

Multi-band OCA references: `ppsc_ref::two_band_oca_matches_reference`
and `ppsc_ref::three_band_oca_matches_reference` compare the native
multi-band OCA solver with the C++ PPSC programs at the native
tolerances, DMFT trajectories included. They pass since both sides carry
the statistics-sign fix (nQIST: `oca::OcaGfDiagram::gd_sign`; C++:
`ppsc_oca_gf_statistics_sign.patch`). With the pristine C++ library the
two cases differ from nQIST from DMFT iteration 2 (28 against 34
equilibrium iterations on the two-band case). The dimer gates in
`tests/xca_twoband_dimer` and `tests/threeband_dimer` compare the same
solver with the exact diagonalization (calibrated against PyED and
Pomerol to 1e-14, `scripts/ed_crosscheck`): 6.7e-4 on the two-band
Kanamori dimer and 8.6e-4 on the three-band dimer (1.9e-2 and 2.1e-2
before the fix). Single-band references are unaffected by the fix.

Real time: `Solver::Xca` at orders 1 and 2 takes the native real-time
NCA/OCA kernels. The real-time XCA route (`rt_xca_m1`/`rt_xca_m2`)
reconstructs the hybridization from its column `t' = 0` and therefore
assumes a time-translation invariant bath, which self-consistent
non-equilibrium DMFT does not provide (the U-quench Bethe bath deviates
by 8e-3); it is gated separately on static dimer baths
(`tests/singleband_dimer`, `tests/holstein_dimer`).

## Holstein coupling convention

PPSC feeds the phonon propagator through the forward AND the backward boson
channel; for bosons both are the same physical line, so the PPSC coupling is
`lambda = sqrt(2) g_ppsc`. nQIST counts the line once (`set_boson_bath` halves
each channel), so its `g` is `lambda`: the Holstein decks here use
`g = sqrt(2) * 0.3 = 0.4242640687119285` to reproduce the PPSC runs with
`g_ppsc = 0.3`.

## Size note

`quench_u35_oca/golden_oca.txt` is 27 MB (452k lines) because it is the
full-precision dump the ignored OCA gates compare against; it is an
`include_str!` target and therefore must be tracked. The 8 MB
`quench_u35_oca/data_ppsc.h5` is a program output that no code reads and is
gitignored.
