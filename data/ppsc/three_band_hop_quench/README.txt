Golden reference for tests/ppsc_ref::three_band_hop_quench_matches_reference
(three-band density-density Kanamori, hopping quench t 1.0 -> 0.8; NCA. Default gate).

Shared layout and the C++ build recipe: see ../README.md.

C++ side (produces golden.txt):
  source : ../KadanoffBaym/.nessi/ppsc-port/ppsc_programs/three_band_hubbard_bethe.cpp
           (KadanoffBaym commit 03fa63f; sha256 8aa640c63a9342116175c4e89daf8994723ed7db03f234fd4451110a5ecfcff6)
  library : ../KadanoffBaym/.nessi/ppsc-port/ppsc/oca WITH the one-line vertex statistics-sign patch
           ../ppsc_oca_gf_statistics_sign.patch (see ../README.md)
  deck    : case.inp (this directory) = three_band_u35_oca/case.inp with
           __order=1, __U0=__U=3.0, __t0=1.0, __t=0.8
  binary : ../KadanoffBaym/.nessi/ppsc-port/build/ppsc_programs/three_band_hubbard_bethe.ex
  run    : work=$(mktemp -d) && cp case.inp "$work" && cd "$work" && \
           /abs/path/three_band_hubbard_bethe.ex case.inp > run.log
  output : run.log (stdout) and data_ppsc.h5 (binary, not kept)
  golden : golden.txt keeps only the compared line kinds (pp_dens, iter, WARMUP, "tstp, n, d1, d2, d3, m, Q") of run.log.

Parameters: beta = 5, nt = 16, ntau = 40, h = 0.02, kt = 5, mu = 0, J = 0.4, U = 3.0 (no U quench), Bz = 0, delta = 0, hopping t(t): 1.0 in equilibrium and 0.8 after the quench (Delta(t, t') = t(t) t(t') G_loc(t, t')). NOTE the C++ "n" column is n1 + 2 n2 (the program prints n2 twice), so nQIST compares n1 + 2 n2 as well.

nQIST side:
  note   : no case.toml - the examples take a static hopping (`t`), so the
           hopping quench is covered by the test below only
  test   : cargo test --release --test ppsc_ref three_band_hop_quench_matches_reference
