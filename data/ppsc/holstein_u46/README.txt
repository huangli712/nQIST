Golden reference for tests/ppsc_ref::holstein_matches_reference
(Hubbard-Holstein U quench 4.0 -> 6.0, NCA).

Shared layout and the C++ build recipe: see ../README.md.

C++ side (produces golden.txt):
  source : ../KadanoffBaym/.nessi/ppsc-port/ppsc_programs/single_band_hubbard_holstein_bethe.cpp
           (KadanoffBaym commit 03fa63f; sha256 b6dedfcc465641ec7971e0bd405ecebb184b1bde9d6178e74dab8bf7cb2180da)
  binary : ../KadanoffBaym/.nessi/ppsc-port/build/ppsc_programs/single_band_hubbard_holstein_bethe.ex
  input  : case.inp (this directory; passed as argv[1])
  run    : work=$(mktemp -d) && cp case.inp "$work" && cd "$work" && \
           /abs/path/single_band_hubbard_holstein_bethe.ex case.inp > run.log
  output : run.log (stdout) and data_ppsc.h5 (binary, not kept)
  golden : golden.txt keeps only the compared line kinds (pp_dens, INIT,
           iter, sector densities, START time, time, AT TIME) of run.log.

nQIST side:
  test   : cargo test --release --test ppsc_ref holstein_matches_reference
  input  : case.toml (this directory)
  run    : cargo run --release --example single_band_hubbard_holstein_bethe -- \
           data/ppsc/holstein_u46/case.toml

Boson-line convention: the PPSC driver feeds D0 through the forward AND the
backward boson channel, so its effective coupling is lambda = sqrt(2) g_ppsc.
nQIST counts the phonon line once (src/solver.rs, set_boson_bath), so case.toml
uses g = sqrt(2) * 0.3 = 0.4242640687119285 to reproduce this reference.
