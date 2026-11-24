Golden reference for tests/ppsc_ref::three_band_matches_reference
(three-band Kanamori U quench 3.0 -> 3.5 at J = 0.4, NCA).

Shared layout and the C++ build recipe: see ../README.md.

C++ side (produces golden.txt):
  source : ../KadanoffBaym/.nessi/ppsc-port/ppsc_programs/three_band_hubbard_bethe.cpp
           (KadanoffBaym commit 03fa63f; sha256 8aa640c63a9342116175c4e89daf8994723ed7db03f234fd4451110a5ecfcff6)
  binary : ../KadanoffBaym/.nessi/ppsc-port/build/ppsc_programs/three_band_hubbard_bethe.ex
  input  : case.inp (this directory; passed as argv[1])
  run    : work=$(mktemp -d) && cp case.inp "$work" && cd "$work" && \
           /abs/path/three_band_hubbard_bethe.ex case.inp > run.log
  output : run.log (stdout) and data_ppsc.h5 (binary, not kept)
  golden : golden.txt keeps only the compared line kinds (pp_dens, iter,
           WARMUP, "tstp, n, d1, d2, d3, m, Q") of run.log.

nQIST side:
  test   : cargo test --release --test ppsc_ref three_band_matches_reference
  input  : case.toml (this directory)
  run    : cargo run --release --example three_band_hubbard_bethe -- \
           data/ppsc/three_band_u35/case.toml
