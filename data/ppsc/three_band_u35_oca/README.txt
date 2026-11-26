Golden reference for tests/ppsc_ref::three_band_oca_matches_reference
(three-band Kanamori U quench 3.0 -> 3.5 at J = 0.4, OCA; heavy gate,
~85 min alone in release, 146 min when four gates ran in parallel). Its
XCA(2) twin tests/ppsc_ref::three_band_xca_oca_matches_reference runs in
reference-only mode (XCA parity pending).

Shared layout and the C++ build recipe: see ../README.md.

C++ side (produces golden.txt):
  source : ../KadanoffBaym/.nessi/ppsc-port/ppsc_programs/three_band_hubbard_bethe.cpp
           (KadanoffBaym commit 03fa63f; sha256 8aa640c63a9342116175c4e89daf8994723ed7db03f234fd4451110a5ecfcff6)
  library: ../KadanoffBaym/.nessi/ppsc-port/ppsc/oca WITH the one-line
           vertex statistics-sign patch ../ppsc_oca_gf_statistics_sign.patch
           (applied to the port and rebuilt 2026-11-26; sha256 of the
           patch 7c743998cfc6f78e90d39787cbdc17d33ea7a03605e0a8daa948f4e711ed485d).
           The pristine library (ppsc-port-pristine.tar.gz) gives the
           wrong multi-orbital OCA Green's function: the iteration-2
           DMFT error is 1.45032 instead of 1.60303 on this case.
  binary : ../KadanoffBaym/.nessi/ppsc-port/build/ppsc_programs/three_band_hubbard_bethe.ex
  input  : case.inp (this directory; passed as argv[1])
  run    : work=$(mktemp -d) && cp case.inp "$work" && cd "$work" && \
           /abs/path/three_band_hubbard_bethe.ex case.inp > run.log
  output : run.log (stdout) and data_ppsc.h5 (binary, not kept)
  golden : golden.txt keeps only the compared line kinds (pp_dens, iter,
           WARMUP, "tstp, n, d1, d2, d3, m, Q") of run.log.

nQIST side:
  test   : scripts/heavy_gates.sh three_band_oca_matches_reference
           or cargo test --release --test ppsc_ref three_band_oca_matches_reference -- --ignored
  input  : case.toml (this directory)
  run    : cargo run --release --example three_band_hubbard_bethe -- \
           data/ppsc/three_band_u35_oca/case.toml
