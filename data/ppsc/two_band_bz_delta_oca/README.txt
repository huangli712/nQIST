Golden reference for tests/ppsc_ref::two_band_bz_delta_oca_matches_reference
(two-band density-density Kanamori, STATIC Bz = 0.3 and crystal field delta = 0.5, U quench 3.0 -> 3.5; OCA. Heavy gate, ~13 min in release).

Shared layout and the C++ build recipe: see ../README.md.

C++ side (produces golden.txt):
  source : ../KadanoffBaym/.nessi/ppsc-port/ppsc_programs/two_band_hubbard_bethe.cpp
           (KadanoffBaym commit 03fa63f; sha256 281f8bf64c75bf74e9a1282f073a24cea0128be951b8f757327f95cff3654363)
  library : ../KadanoffBaym/.nessi/ppsc-port/ppsc/oca WITH the one-line vertex statistics-sign patch
           ../ppsc_oca_gf_statistics_sign.patch (see ../README.md)
  deck    : case.inp (this directory) = two_band_oca/case.inp with
           __order=2, __Bz0=__Bz=0.3, __delta0=__delta=0.5
  binary : ../KadanoffBaym/.nessi/ppsc-port/build/ppsc_programs/two_band_hubbard_bethe.ex
  run    : work=$(mktemp -d) && cp case.inp "$work" && cd "$work" && \
           /abs/path/two_band_hubbard_bethe.ex case.inp > run.log
  output : run.log (stdout) and data_ppsc.h5 (binary, not kept)
  golden : golden.txt keeps only the compared line kinds (pp_dens, iter, eqobs, WARMUP, obs) of run.log.

Parameters: beta = 5, nt = 30, ntau = 64, h = 0.02, kt = 5, mu = 0, J = 0.4, U 3.0 -> 3.5, Bz = 0.3, delta = 0.5, t = 1.0 (static field and crystal field)

nQIST side:
  input  : case.toml (this directory)
  run    : cargo run --release --example two_band_hubbard_bethe -- \
           data/ppsc/two_band_bz_delta_oca/case.toml
  test   : scripts/heavy_gates.sh two_band_bz_delta_oca_matches_reference
           or cargo test --release --test ppsc_ref two_band_bz_delta_oca_matches_reference -- --ignored
