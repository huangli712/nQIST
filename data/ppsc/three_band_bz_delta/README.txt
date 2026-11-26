Golden reference for tests/ppsc_ref::three_band_bz_delta_matches_reference
(three-band density-density Kanamori, STATIC Bz = 0.3 and crystal field delta = 0.5, U quench 3.0 -> 3.5; NCA. Default gate).

Shared layout and the C++ build recipe: see ../README.md.

C++ side (produces golden.txt):
  source : ../KadanoffBaym/.nessi/ppsc-port/ppsc_programs/three_band_hubbard_bethe.cpp
           (KadanoffBaym commit 03fa63f; sha256 8aa640c63a9342116175c4e89daf8994723ed7db03f234fd4451110a5ecfcff6)
  library : ../KadanoffBaym/.nessi/ppsc-port/ppsc/oca WITH the one-line vertex statistics-sign patch
           ../ppsc_oca_gf_statistics_sign.patch (see ../README.md)
  deck    : case.inp (this directory) = three_band_u35_oca/case.inp with
           __order=1, __Bz0=__Bz=0.3, __delta0=__delta=0.5
  binary : ../KadanoffBaym/.nessi/ppsc-port/build/ppsc_programs/three_band_hubbard_bethe.ex
  run    : work=$(mktemp -d) && cp case.inp "$work" && cd "$work" && \
           /abs/path/three_band_hubbard_bethe.ex case.inp > run.log
  output : run.log (stdout) and data_ppsc.h5 (binary, not kept)
  golden : golden.txt keeps only the compared line kinds (pp_dens, iter, WARMUP, "tstp, n, d1, d2, d3, m, Q") of run.log.

Parameters: beta = 5, nt = 16, ntau = 40, h = 0.02, kt = 5, mu = 0, J = 0.4, U 3.0 -> 3.5, Bz = 0.3, delta = 0.5, t = 1.0 (static field and crystal field: the spin and orbital degeneracies are lifted, the six channels differ). NOTE the C++ "n" column is n1 + 2 n2 (the program prints n2 twice), so nQIST compares n1 + 2 n2 as well.

nQIST side:
  input  : case.toml (this directory)
  run    : cargo run --release --example three_band_hubbard_bethe -- \
           data/ppsc/three_band_bz_delta/case.toml
  test   : cargo test --release --test ppsc_ref three_band_bz_delta_matches_reference
