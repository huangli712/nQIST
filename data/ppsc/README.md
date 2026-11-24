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
| two_band_u35 | two-band Kanamori, NCA | case.inp, case.toml, run.log | `ppsc_ref::two_band_matches_reference` |
| two_band_oca | two-band Kanamori, OCA | case.inp, case.toml, golden.txt | `ppsc_ref::two_band_oca_matches_reference` (ignored, ~15 min) |
| three_band_u35 | three-band Kanamori, NCA | case.inp, case.toml, golden.txt | `ppsc_ref::three_band_matches_reference` |
| three_band_u35_oca | three-band Kanamori, OCA | case.inp, case.toml, golden.txt | `ppsc_ref::three_band_oca_matches_reference` (ignored, ~2.2 h) |
| holstein_u46 | single-band Hubbard-Holstein, NCA (g_ppsc = 0.3) | case.inp, case.toml, useries.txt, golden.txt | `ppsc_ref::holstein_matches_reference` |
| holstein_u46_oca | same, OCA | case.inp, case.toml, useries.txt, golden.txt | `ppsc_ref::holstein_oca_matches_reference` |
| prb96_photodoping | photodoping reference trajectory | case.inp, case.toml, tseries.txt, run.log | `ppsc_ref::prb96_photodoping_matches_reference` (ignored) |
| probe9_spgf.cpp/.txt | single-particle GF component probe | source + dump | `src/oca.rs` unit test |

Per-case details (C++ source file, sha256, exact run command, compared line
kinds) are in each `<case>/README.txt`.

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
cargo test --release --test ppsc_ref                  # fast gates (7)
cargo test --release --test ppsc_ref -- --ignored     # heavy gates (5)
cargo run --release --example <example> -- data/ppsc/<case>/case.toml
```

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
