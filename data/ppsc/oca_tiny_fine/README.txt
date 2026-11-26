Fine-grid reference for
tests/ppsc_ref::oca_realtime_tiny_fine_matches_reference (PPSC-like OCA)
tests/ppsc_ref::xca_oca_realtime_tiny_fine_matches_reference (XCA order 2)
(single-band Hubbard/Bethe, U quench 3.0 -> 3.5, OCA, tiny real-time
window nt = 8, kt = 5, h = 0.02, beta = 5; Matsubara grid ntau = 250).

Why a fine grid: the XCA order-2 series evaluates its skeletons on a
sum-of-poles representation of the hybridization, and on the coarse grid
of ../oca_tiny (ntau = 40) the grid values carry O(1e-5) discretization
roughness that no sum of poles follows below rms 5e-6. The XCA twin of
the coarse reference can therefore not reach the native tolerances; both
solvers are gated on this grid instead, at the native tolerances.

C++ side (produces golden_tiny_fine.txt):
  source : probe10_fine.cpp (this directory; ../oca_tiny/probe10.cpp with
           ntau = 250, the Matsubara sample lists of the original fit
           this grid), linked against the stock PPSC libraries of
           ../KadanoffBaym/.nessi/ppsc-port (KadanoffBaym commit 03fa63f)
  build  (out of tree; <port> = ../KadanoffBaym/.nessi/ppsc-port):
    g++ -std=c++17 -O2 -I<port> -I<port>/cereal-1.2.1/include \
      -I<port>/nessi-install/include/cntr -I<port>/nessi-install/include/cntr/hdf5 \
      -I<port>/nessi-install/include/cntr/utils -I<port>/../eigen3-install/include \
      -I<port>/../eigen3-install/include/eigen3 -I/opt/anaconda3/include \
      -DCNTR_USE_HDF5 probe10_fine.cpp \
      -L<port>/build/ppsc -L<port>/build/ppsc/nca -L<port>/build/ppsc/oca \
      -L<port>/nessi-install/lib -lppsc -lnca -loca -lcntr -lcntr_hdf5 \
      /opt/anaconda3/lib/libhdf5.so \
      -Wl,-rpath,<port>/build/ppsc:<port>/build/ppsc/nca:<port>/build/ppsc/oca:<port>/nessi-install/lib:/usr/lib/x86_64-linux-gnu:/opt/anaconda3/lib \
      -o probe10_fine
  run    : work=$(mktemp -d) && cd "$work" && /abs/path/probe10_fine > golden_tiny_fine.txt
  output : the complete stdout of the probe (2237 lines, %.17g)

nQIST side:
  test   : cargo test --release --test ppsc_ref oca_realtime_tiny
