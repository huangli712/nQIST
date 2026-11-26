#!/usr/bin/env python3
"""Compare an external-ED reference dump with the nQIST ED cross-check dump.

Records are joined per model on their keys `(mode, i)`, `(kind, mode, t)`
and `(kind, mode, n, np)`; a record present in only one dump is an error.

Usage: compare.py <nqist_dump> <reference_dump> [--abs TOL] [--z-rel TOL]

Exit status 0 when every maximum deviation is within tolerance.
"""

import argparse
import sys

from dumpformat import read_dump

DEFAULT_ABS = 1e-10
DEFAULT_Z_REL = 1e-12

# (column, record column name, width)
COLUMNS = [
    ("model", "model", 24),
    ("Z (rel)", "Z (rel)", 10),
    ("E_min", "E_min", 10),
    ("spectrum", "spectrum", 10),
    ("observ.", "observ.", 10),
    ("G(tau)", "G(tau)", 10),
    ("G_eq(t)", "G_eq(t)", 10),
    ("quench", "quench", 10),
]


def joined_max(rows_a, rows_b, key, value):
    """`(maximum |value(a) - value(b)| over the common keys, key of the
    maximum)`; exits on a record present on one side only."""
    b = {key(r): r for r in rows_b}
    if len(b) != len(rows_b):
        sys.exit("duplicate keys in the reference dump")
    if set(b) != {key(r) for r in rows_a}:
        sys.exit("record keys differ between the dumps")
    worst, where = 0.0, None
    for r in rows_a:
        d = abs(value(r) - value(b[key(r)]))
        if d > worst:
            worst, where = d, key(r)
    return worst, where


def cval(*idx):
    return lambda x: abs(complex(x[idx[0]], x[idx[1]]))


def model_columns(m, r, args):
    """`OrderedDict(record column -> (worst deviation, tolerance))` of the
    records the reference computed for this model."""
    cols = {}
    if r.z is not None:
        cols["Z (rel)"] = (abs(r.z - m.z) / abs(m.z), args.z_rel)
    if r.emin is not None:
        cols["E_min"] = (abs(r.emin - m.emin), args.abs)
    if r.ev:
        if len(m.ev) != len(r.ev):
            sys.exit("{}: spectra of different length ({} vs {})".format(m.name, len(m.ev), len(r.ev)))
        cols["spectrum"] = (max(abs(a - b) for a, b in zip(m.ev, r.ev)), args.abs)
    if r.obs:
        if not set(r.obs) <= set(m.obs):
            sys.exit("{}: observables unknown to the nQIST dump".format(m.name))
        cols["observ."] = (max(abs(m.obs[k] - r.obs[k]) for k in r.obs), args.abs)
    if r.m_tau:
        cols["G(tau)"] = (
            joined_max(m.m_tau, r.m_tau, lambda x: x[:2], lambda x: x[3])[0],
            args.abs,
        )
    if r.rteq:
        cols["G_eq(t)"] = joined_max(m.rteq, r.rteq, lambda x: x[:3], cval(3, 4))[0], args.abs
    if r.rt:
        cols["quench"] = joined_max(m.rt, r.rt, lambda x: x[:4], cval(4, 5))[0], args.abs
    return cols


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("nqist")
    ap.add_argument("reference")
    ap.add_argument("--abs", type=float, default=DEFAULT_ABS,
                    help="absolute tolerance of energies and correlators")
    ap.add_argument("--z-rel", type=float, default=DEFAULT_Z_REL,
                    help="relative tolerance of the partition function")
    args = ap.parse_args()

    nq = read_dump(args.nqist)
    ref = read_dump(args.reference)
    for name in ref:
        if name not in nq:
            sys.exit("model {} is not in the nQIST dump".format(name))

    header = " ".join(
        "{:<{}}".format(name, width) if name == "model" else "{:>{}}".format(name, width)
        for name, _, width in COLUMNS
    )
    print(header)
    print("-" * len(header))
    worst_all, ok, per_column = 0.0, True, {}
    for name, r in ref.items():
        cols = model_columns(nq[name], r, args)
        cells = []
        for cname, _, width in COLUMNS:
            if cname == "model":
                cells.append("{:<{}}".format(name, width))
            elif cname in cols:
                worst, tol = cols[cname]
                cells.append("{:>{}.1e}{}".format(worst, width - 1, "!" if worst > tol else " "))
                worst_all = max(worst_all, worst / tol)
                ok = ok and worst <= tol
                per_column[cname] = max(per_column.get(cname, 0.0), worst / tol)
            else:
                cells.append("{:>{}}".format("-", width))
        print(" ".join(cells))
    print("-" * len(header))
    print("max deviation / tolerance per column: " + ", ".join(
        "{} {:.3g}".format(c, v) for c, v in per_column.items()))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
