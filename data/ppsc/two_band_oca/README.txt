Golden reference for tests/ppsc_ref::two_band_oca_matches_reference
(two-band Kanamori U quench 3.0 -> 3.5 at J = 0.4, OCA; heavy gate,
~15 min in release).

Shared layout and the C++ build recipe: see ../README.md.

C++ side (produces golden.txt):
  source : ../KadanoffBaym/.nessi/ppsc-port/ppsc_programs/two_band_hubbard_bethe.cpp
           (KadanoffBaym commit 03fa63f; sha256 281f8bf64c75bf74e9a1282f073a24cea0128be951b8f757327f95cff3654363)
  binary : ../KadanoffBaym/.nessi/ppsc-port/build/ppsc_programs/two_band_hubbard_bethe.ex
  input  : case.inp (this directory; passed as argv[1])
  run    : work=$(mktemp -d) && cp case.inp "$work" && cd "$work" && \
           /abs/path/two_band_hubbard_bethe.ex case.inp > run.log
  output : run.log (stdout) and data_ppsc.h5 (binary, not kept)
  golden : golden.txt keeps only the compared line kinds (pp_dens, iter,
           eqobs, WARMUP, obs) of run.log.

nQIST side:
  test   : cargo test --release --test ppsc_ref two_band_oca_matches_reference -- --ignored
  input  : case.toml (this directory)
  run    : cargo run --release --example two_band_hubbard_bethe -- \
           data/ppsc/two_band_oca/case.toml
