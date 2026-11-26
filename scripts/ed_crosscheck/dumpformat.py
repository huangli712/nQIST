"""Reader and writer for the ED cross-check dump format.

The format is produced by `cargo run --release --example ed_crosscheck`
(`examples/ed_crosscheck.rs`). One whitespace-separated record per line;
`<name>` is the model name:

    MODEL  <name> variant=<v> norb=<n> beta=<b> ntau=<n> nbath=<n> dim=<d> dt=<h> nt=<n>
    PARAM  <name> mu=<x> u=<x> j=<x> delta=<x> bz=<x> eps_up=<x> eps_do=<x>
    QUENCH <name> param=<p> from=<x> to=<x>
    BATH   <name> imp_mode=<m> eps=<x> v=<x>
    PHONON <name> nph=<n> omega=<x> lambda=<x> ops=<mode>:<0|1>,...
    Z      <name> <zpart> <emin>
    EV     <name> <E0> <E1> ...
    OBS    <name> <key> <value>
    M      <name> <mode> <i> <tau> <G>                 (Matsubara -<T c(tau) c^dag>)
    RTEQ   <name> <kind> <mode> <t> <Re> <Im>          (equilibrium real time)
    RT     <name> <kind> <mode> <n> <np> <Re> <Im>     (after the quench, n >= np)

A reference dump carries the same tags with the records it computes;
`compare.py` joins the two on the record keys.
"""

import collections

EPS = ".16e"


class Model:
    """All records of one model."""

    def __init__(self, name):
        self.name = name
        self.kv = {}
        self.params = {}
        self.quench = {}
        self.bath = []  # (imp_mode, eps, v)
        self.phonon = None  # dict(nph, omega, lambda, ops=[(mode, cre)])
        self.z = None
        self.emin = None
        self.ev = []
        self.obs = collections.OrderedDict()
        self.m_tau = []  # (mode, i, tau, g)
        self.rteq = []  # (kind, mode, t, re, im)
        self.rt = []  # (kind, mode, n, np, re, im)

    @property
    def variant(self):
        return self.kv["variant"]

    @property
    def norb(self):
        return int(self.kv["norb"])

    @property
    def beta(self):
        return float(self.kv["beta"])

    @property
    def dim(self):
        return int(self.kv["dim"])

    @property
    def dt(self):
        return float(self.kv["dt"])

    @property
    def nt(self):
        return int(self.kv["nt"])

    def tau_list(self, mode):
        """`[(i, tau)]` of the Matsubara records of `mode`, ordered by `i`."""
        rows = sorted((i, tau) for (m, i, tau, _) in self.m_tau if m == mode)
        return rows

    def rteq_list(self, kind, mode):
        """`[t]` of the equilibrium real-time records, ordered as printed."""
        return [t for (k, m, t, _, _) in self.rteq if k == kind and m == mode]

    def rt_list(self, kind, mode):
        """`[(n, np)]` of the quench records, ordered as printed."""
        return [(n, np) for (k, m, n, np, _, _) in self.rt if k == kind and m == mode]


RESULT_TAGS = ("Z", "EV", "OBS", "M", "RTEQ", "RT")


def read_dump(path):
    """`OrderedDict(name -> Model)` of the dump at `path`. A reference dump
    carries result records only; its models are created on demand."""
    models = collections.OrderedDict()
    cur = None
    with open(path) as f:
        for line in f:
            w = line.split()
            if not w:
                continue
            tag = w[0]
            if tag == "MODEL":
                cur = Model(w[1])
                cur.kv = dict(x.split("=") for x in w[2:])
                models[cur.name] = cur
            elif tag in RESULT_TAGS:
                cur = models.setdefault(w[1], Model(w[1]))
            elif tag not in ("PARAM", "QUENCH", "BATH", "PHONON"):
                raise ValueError("unknown record tag '{}' in {}".format(tag, path))
            if tag == "PARAM":
                cur.params = {k: float(v) for k, v in (x.split("=") for x in w[2:])}
            elif tag == "QUENCH":
                cur.quench = dict(x.split("=") for x in w[2:])
                for k in ("from", "to"):
                    cur.quench[k] = float(cur.quench[k])
            elif tag == "BATH":
                kv = dict(x.split("=") for x in w[2:])
                cur.bath.append(
                    (int(kv["imp_mode"]), float(kv["eps"]), float(kv["v"]))
                )
            elif tag == "PHONON":
                kv = dict(x.split("=") for x in w[2:])
                ops = [tuple(int(y) for y in x.split(":")) for x in kv["ops"].split(",")]
                cur.phonon = dict(
                    nph=int(kv["nph"]),
                    omega=float(kv["omega"]),
                    **{"lambda": float(kv["lambda"])},
                    ops=ops,
                )
            elif tag == "Z":
                cur.z, cur.emin = float(w[2]), float(w[3])
            elif tag == "EV":
                cur.ev = [float(x) for x in w[2:]]
            elif tag == "OBS":
                cur.obs[w[2]] = float(w[3])
            elif tag == "M":
                cur.m_tau.append((int(w[2]), int(w[3]), float(w[4]), float(w[5])))
            elif tag == "RTEQ":
                cur.rteq.append((w[2], int(w[3]), float(w[4]), float(w[5]), float(w[6])))
            elif tag == "RT":
                cur.rt.append(
                    (w[2], int(w[3]), int(w[4]), int(w[5]), float(w[6]), float(w[7]))
                )
    return models


class Writer:
    """Accumulates reference records in the dump format."""

    def __init__(self):
        self.lines = []

    def z(self, name, z, emin):
        self.lines.append("Z {} {} {}".format(name, fmt(z), fmt(emin)))

    def ev(self, name, ev):
        self.lines.append("EV {} {}".format(name, " ".join(fmt(e) for e in ev)))

    def obs(self, name, key, value):
        self.lines.append("OBS {} {} {}".format(name, key, fmt(value)))

    def m_tau(self, name, mode, i, tau, g):
        self.lines.append("M {} {} {} {} {}".format(name, mode, i, fmt(tau), fmt(g)))

    def rteq(self, name, kind, mode, t, g):
        self.lines.append(
            "RTEQ {} {} {} {} {} {}".format(name, kind, mode, fmt(t), fmt(g.real), fmt(g.imag))
        )

    def rt(self, name, kind, mode, n, np_, g):
        self.lines.append(
            "RT {} {} {} {} {} {} {}".format(
                name, kind, mode, n, np_, fmt(g.real), fmt(g.imag)
            )
        )

    def write(self, path):
        with open(path, "w") as f:
            for line in self.lines:
                f.write(line + "\n")


def fmt(x):
    return "{:.16e}".format(x)
