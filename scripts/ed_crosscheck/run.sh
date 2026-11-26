#!/usr/bin/env bash
# ED cross-check: the built-in ED of nQIST against external
# exact-diagonalization packages.
#
#   scripts/ed_crosscheck/run.sh <work-dir>
#
# Run it from an activated environment that provides TRIQS + PyED,
# Pomerol (with libcommute, Eigen and Boost headers), a C++ compiler,
# MPI, CMake and Make (see README.md). The script
#   1. dumps the nQIST reference (examples/ed_crosscheck.rs),
#   2. rebuilds every model with PyED (TRIQS operators) and compares,
#   3. rebuilds every model with Pomerol (libcommute) and compares.
# Both comparisons must print PASS; the exit status is non-zero
# otherwise.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
work=${1:?usage: run.sh <work-dir>}
mkdir -p "$work"

echo "== nQIST ED dump"
cargo run --release --manifest-path "$root/Cargo.toml" --example ed_crosscheck -- > "$work/nqist.txt"

echo "== PyED reference"
(cd "$here" && python pyed_ref.py "$work/nqist.txt" "$work/pyed.txt")
(cd "$here" && python compare.py "$work/nqist.txt" "$work/pyed.txt")

echo "== Pomerol reference"
cmake -S "$here" -B "$work/pomerol-build" -DCMAKE_BUILD_TYPE=Release > "$work/pomerol-cmake.log"
cmake --build "$work/pomerol-build" -j > "$work/pomerol-build.log"
mpirun -np 1 "$work/pomerol-build/pomerol_ref" "$work/nqist.txt" "$work/pomerol.txt" \
    > "$work/pomerol-run.log" 2>&1
(cd "$here" && python compare.py "$work/nqist.txt" "$work/pomerol.txt")
