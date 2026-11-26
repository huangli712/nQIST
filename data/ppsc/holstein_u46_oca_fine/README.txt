Fine-grid reference for tests/ppsc_ref::holstein_oca_fine_matches_reference
and tests/ppsc_ref::holstein_xca_oca_fine_matches_reference
(Hubbard-Holstein U quench 4.0 -> 6.0, OCA; same case as ../holstein_u46_oca
with ntau = 250 instead of 40, see ../oca_tiny_fine/README.txt for why).

Shared layout and the C++ build recipe: see ../README.md.

C++ side (produces golden.txt):
  source : ../KadanoffBaym/.nessi/ppsc-port/ppsc_programs/single_band_hubbard_holstein_bethe.cpp
           (KadanoffBaym commit 03fa63f; sha256 b6dedfcc465641ec7971e0bd405ecebb184b1bde9d6178e74dab8bf7cb2180da)
  binary : ../KadanoffBaym/.nessi/ppsc-port/build/ppsc_programs/single_band_hubbard_holstein_bethe.ex
  input  : case.inp (this directory; passed as argv[1]), useries.txt
  run    : work=$(mktemp -d) && cp case.inp useries.txt "$work" && cd "$work" && \
           /abs/path/single_band_hubbard_holstein_bethe.ex case.inp > run.log
  output : run.log (stdout) and data_ppsc.h5 (binary, not kept)
  golden : golden.txt keeps only the compared line kinds (pp_dens, INIT,
           iter, sector densities, START time, time, AT TIME) of run.log.

nQIST side:
  test   : cargo test --release --test ppsc_ref holstein_oca_fine
  input  : case.toml (this directory)
  run    : cargo run --release --example single_band_hubbard_holstein_bethe -- \
           data/ppsc/holstein_u46_oca_fine/case.toml

Boson-line convention: as in ../holstein_u46_oca (the PPSC driver counts
the phonon line twice; case.toml uses g = sqrt(2) * 0.3).
