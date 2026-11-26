Golden reference for tests/ppsc_ref::two_band_hop_quench_matches_reference
(two-band density-density Kanamori, hopping quench t 1.0 -> 0.8; NCA. Default gate).

Shared layout and the C++ build recipe: see ../README.md.

C++ side (produces golden.txt):
  source : ../KadanoffBaym/.nessi/ppsc-port/ppsc_programs/two_band_hubbard_bethe.cpp
           (KadanoffBaym commit 03fa63f; sha256 281f8bf64c75bf74e9a1282f073a24cea0128be951b8f757327f95cff3654363)
  library : ../KadanoffBaym/.nessi/ppsc-port/ppsc/oca WITH the one-line vertex statistics-sign patch
           ../ppsc_oca_gf_statistics_sign.patch (see ../README.md)
  deck    : case.inp (this directory) = two_band_oca/case.inp with
           __order=1, __U0=__U=3.0, __t0=1.0, __t=0.8
  binary : ../KadanoffBaym/.nessi/ppsc-port/build/ppsc_programs/two_band_hubbard_bethe.ex
  run    : work=$(mktemp -d) && cp case.inp "$work" && cd "$work" && \
           /abs/path/two_band_hubbard_bethe.ex case.inp > run.log
  output : run.log (stdout) and data_ppsc.h5 (binary, not kept)
  golden : golden.txt keeps only the compared line kinds (pp_dens, iter, eqobs, WARMUP, obs) of run.log.

Parameters: beta = 5, nt = 30, ntau = 64, h = 0.02, kt = 5, mu = 0, J = 0.4, U = 3.0 (no U quench), Bz = 0, delta = 0, hopping t(t): 1.0 in equilibrium and 0.8 after the quench (the Bethe Weiss field is Delta(t, t') = t(t) t(t') G_loc(t, t'), so the bath becomes time-translation non-invariant)

nQIST side:
  note   : no case.toml - the examples take a static hopping (`t`), so the
           hopping quench is covered by the test below only
  test   : cargo test --release --test ppsc_ref two_band_hop_quench_matches_reference
