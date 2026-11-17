// Project : nQIST
// File    : src/io.rs
// Purpose : TOML-subset configuration parser and C++-aligned text
//           output helpers

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

// ------------------------------------------------------------------
// parse error
// ------------------------------------------------------------------

/// Parse error carrying the 1-based source line number.
#[derive(Clone, Debug, PartialEq)]
pub struct IoError {
    /// 1-based line number of the offending line.
    pub line: usize,
    pub msg: String,
}

impl Display for IoError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "io: line {}: {}", self.line, self.msg)
    }
}

impl Error for IoError {}

fn err(line: usize, msg: &str) -> IoError {
    IoError {
        line,
        msg: msg.to_string(),
    }
}

// ------------------------------------------------------------------
// document model
// ------------------------------------------------------------------

/// A parsed value of the TOML subset: integers, floats, booleans,
/// strings, and flat numeric arrays.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    /// Flat numeric array (all entries stored as `f64`).
    Array(Vec<f64>),
}

impl Value {
    /// Human-readable type name, for error messages.
    fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "an integer",
            Value::Float(_) => "a float",
            Value::Bool(_) => "a bool",
            Value::Str(_) => "a string",
            Value::Array(_) => "an array",
        }
    }
}

/// One level of `key -> value` bindings.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TomlTable {
    entries: BTreeMap<String, Value>,
}

impl TomlTable {
    /// Raw lookup of a key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.get(key)
    }

    /// Type-mismatch message for `key`.
    fn want(&self, key: &str, want: &str) -> String {
        let got = self.get(key).map(|v| v.type_name()).unwrap_or("nothing");
        format!("key '{key}': expected {want}, got {got}")
    }

    /// Read an integer.
    pub fn get_i64(&self, key: &str) -> Result<i64, String> {
        match self.get(key) {
            Some(Value::Int(v)) => Ok(*v),
            _ => Err(self.want(key, "an integer")),
        }
    }

    /// Read a non-negative integer.
    pub fn get_usize(&self, key: &str) -> Result<usize, String> {
        match self.get_i64(key) {
            Ok(v) if v >= 0 => Ok(v as usize),
            Ok(v) => Err(format!(
                "key '{key}': expected a non-negative integer, got {v}"
            )),
            Err(e) => Err(e),
        }
    }

    /// Read a float (integers coerce).
    pub fn get_f64(&self, key: &str) -> Result<f64, String> {
        match self.get(key) {
            Some(Value::Int(v)) => Ok(*v as f64),
            Some(Value::Float(v)) => Ok(*v),
            _ => Err(self.want(key, "a float")),
        }
    }

    /// Read a boolean.
    pub fn get_bool(&self, key: &str) -> Result<bool, String> {
        match self.get(key) {
            Some(Value::Bool(v)) => Ok(*v),
            _ => Err(self.want(key, "a bool")),
        }
    }

    /// Read a string.
    pub fn get_str(&self, key: &str) -> Result<&str, String> {
        match self.get(key) {
            Some(Value::Str(v)) => Ok(v),
            _ => Err(self.want(key, "a string")),
        }
    }

    /// Read a numeric scalar or array: a scalar broadcasts to a
    /// one-element vector (time-dependent parameters written as
    /// constants in the PPSC `case.inp` style).
    pub fn get_f64_vec(&self, key: &str) -> Result<Vec<f64>, String> {
        match self.get(key) {
            Some(Value::Int(v)) => Ok(vec![*v as f64]),
            Some(Value::Float(v)) => Ok(vec![*v]),
            Some(Value::Array(v)) => Ok(v.clone()),
            _ => Err(self.want(key, "a float or a numeric array")),
        }
    }
}

/// A parsed TOML-subset document: a root table plus one-level
/// sections (`[name]`); nested tables are rejected.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TomlDoc {
    root: TomlTable,
    sections: BTreeMap<String, TomlTable>,
}

impl TomlDoc {
    /// The root table (keys before any `[section]` header).
    pub fn root(&self) -> &TomlTable {
        &self.root
    }

    /// Look up a section by name.
    pub fn section(&self, name: &str) -> Result<&TomlTable, String> {
        self.sections
            .get(name)
            .ok_or_else(|| format!("no section '[{name}]'"))
    }

    /// Section names in declaration-independent (sorted) order.
    pub fn section_names(&self) -> Vec<&str> {
        self.sections.keys().map(|k| k.as_str()).collect()
    }
}

// ------------------------------------------------------------------
// parser
// ------------------------------------------------------------------

/// Parse TOML-subset source into a document. The subset covers
/// comments, blank lines, `key = value` pairs with integers, floats,
/// booleans, quoted strings, and flat numeric arrays, plus one-level
/// `[section]` tables.
pub fn parse_toml(src: &str) -> Result<TomlDoc, IoError> {
    let mut doc = TomlDoc::default();
    let mut current: Option<String> = None;

    for (idx, raw) in src.lines().enumerate() {
        let lineno = idx + 1;
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix('[') {
            let name = rest
                .strip_suffix(']')
                .ok_or_else(|| err(lineno, "unterminated section header"))?;
            if name.contains('.') {
                return Err(err(lineno, "nested sections are not supported"));
            }
            if name.is_empty() || !is_bare_key(name) {
                return Err(err(lineno, &format!("invalid section name '{name}'")));
            }
            if doc.sections.contains_key(name) {
                return Err(err(lineno, &format!("duplicate section '[{name}]'")));
            }
            doc.sections.insert(name.to_string(), TomlTable::default());
            current = Some(name.to_string());
            continue;
        }

        let (key, val) = line
            .split_once('=')
            .ok_or_else(|| err(lineno, "expected 'key = value'"))?;
        let key = key.trim();
        if !is_bare_key(key) {
            return Err(err(lineno, &format!("invalid key '{key}'")));
        }
        let value = parse_value(val.trim(), lineno)?;

        let table = match &current {
            None => &mut doc.root,
            Some(s) => doc.sections.get_mut(s).expect("section exists"),
        };
        if table.entries.insert(key.to_string(), value).is_some() {
            return Err(err(lineno, &format!("duplicate key '{key}'")));
        }
    }
    Ok(doc)
}

/// True for a non-empty bare key (`A-Z`, `a-z`, `0-9`, `_`, `-`).
fn is_bare_key(text: &str) -> bool {
    !text.is_empty()
        && text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Cut a `#` comment, ignoring `#` inside quoted strings.
fn strip_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut in_str = false;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            // inside a string: skip the escaped character
            b'\\' if in_str => i += 1,
            b'"' => in_str = !in_str,
            b'#' if !in_str => return &line[..i],
            _ => {}
        }
        i += 1;
    }
    line
}

/// Parse a scalar or array value.
fn parse_value(text: &str, lineno: usize) -> Result<Value, IoError> {
    if text.is_empty() {
        return Err(err(lineno, "missing value"));
    }

    if text.starts_with('"') {
        let inner = text
            .strip_prefix('"')
            .and_then(|t| t.strip_suffix('"'))
            .ok_or_else(|| err(lineno, "unterminated string"))?;
        let mut out = String::new();
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some(_) => return Err(err(lineno, "unsupported escape sequence")),
                    None => return Err(err(lineno, "dangling escape sequence")),
                }
            } else {
                out.push(c);
            }
        }
        return Ok(Value::Str(out));
    }

    if text.starts_with('[') {
        let inner = text
            .strip_prefix('[')
            .and_then(|t| t.strip_suffix(']'))
            .ok_or_else(|| err(lineno, "unterminated array"))?;
        let inner = inner.trim();
        if inner.is_empty() {
            return Ok(Value::Array(Vec::new()));
        }
        let mut out = Vec::new();
        for part in inner.split(',') {
            let part = part.trim();
            if part.is_empty() {
                return Err(err(lineno, "empty array entry"));
            }
            out.push(parse_number(part, lineno)?);
        }
        return Ok(Value::Array(out));
    }

    if text == "true" {
        return Ok(Value::Bool(true));
    }
    if text == "false" {
        return Ok(Value::Bool(false));
    }

    if is_int_literal(text) {
        return text
            .parse::<i64>()
            .map(Value::Int)
            .map_err(|_| err(lineno, "integer literal out of range"));
    }
    Ok(Value::Float(parse_number(text, lineno)?))
}

/// True for an optionally signed digit string.
fn is_int_literal(text: &str) -> bool {
    let t = text.strip_prefix(['+', '-']).unwrap_or(text);
    !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit())
}

/// Parse a float literal (also used for array entries).
fn parse_number(text: &str, lineno: usize) -> Result<f64, IoError> {
    text.parse::<f64>()
        .map_err(|_| err(lineno, &format!("cannot parse number '{text}'")))
}

// ------------------------------------------------------------------
// benchmark case configuration
// ------------------------------------------------------------------

/// Input parameters of a single-band PPSC case, mirroring all fields
/// of the PPSC `case.inp` probe input.
#[derive(Clone, Debug, PartialEq)]
pub struct CaseConfig {
    pub nt: usize,
    pub ntau: usize,
    pub beta: f64,
    pub h: f64,
    pub itermax: usize,
    pub errmax: f64,
    pub iter_rtime: usize,
    pub kt: usize,
    /// 1 = NCA, 2 = OCA.
    pub order: usize,
    /// OpenMP thread count of the C++ code; kept for input fidelity,
    /// the Rust solver is sequential.
    pub nomp: usize,
    pub store_pp: bool,
    pub linear_mixing: f64,
    pub read_eq_sym: bool,
    pub read_rt_sym: bool,
    pub read_state_from_file: bool,
    pub mu: f64,
    pub u0: f64,
    /// Onsite interaction `U(t)`: a constant (one element, broadcast
    /// downstream) or an explicit series.
    pub u: Vec<f64>,
    /// Onsite level `eps(t)`; same constant/series convention as `u`.
    pub eps: Vec<f64>,
}

impl CaseConfig {
    /// Parse a case configuration from TOML-subset source (the case
    /// file is flat: root-level keys).
    pub fn from_str(src: &str) -> Result<Self, String> {
        let doc = parse_toml(src).map_err(|e| e.to_string())?;
        Self::from_doc(doc.root())
    }

    /// Build the configuration from a parsed table.
    pub fn from_doc(t: &TomlTable) -> Result<Self, String> {
        Ok(Self {
            nt: t.get_usize("nt")?,
            ntau: t.get_usize("ntau")?,
            beta: t.get_f64("beta")?,
            h: t.get_f64("h")?,
            itermax: t.get_usize("itermax")?,
            errmax: t.get_f64("errmax")?,
            iter_rtime: t.get_usize("iter_rtime")?,
            kt: t.get_usize("kt")?,
            order: t.get_usize("order")?,
            nomp: t.get_usize("nomp").unwrap_or(1),
            store_pp: t.get_bool("store_pp").unwrap_or(false),
            linear_mixing: t.get_f64("linear_mixing")?,
            read_eq_sym: t.get_bool("read_eq_sym").unwrap_or(false),
            read_rt_sym: t.get_bool("read_rt_sym").unwrap_or(false),
            read_state_from_file: t.get_bool("read_state_from_file").unwrap_or(false),
            mu: t.get_f64("mu")?,
            u0: t.get_f64("U0")?,
            u: t.get_f64_vec("U")?,
            eps: t.get_f64_vec("eps")?,
        })
    }
}

// ------------------------------------------------------------------
// C++-aligned output formatting
// ------------------------------------------------------------------

/// C `%g` with 6 significant digits: the default formatting of a
/// `double` by C++ `std::ostream`, matched digit for digit (including
/// the two-digit exponent of the scientific branch), so that log
/// output can be diffed against the PPSC `run.log` references.
pub fn fmt_g6(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if x.is_nan() {
        return "nan".to_string();
    }
    if !x.is_finite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    // round to 6 significant digits first: the rounded exponent
    // decides the branch (e.g. 999999.6 -> "1e+06", like printf %g)
    let raw = format!("{x:.5e}");
    let (mant, e) = raw.split_once('e').expect("scientific format");
    let ev: i32 = e.parse().expect("exponent");
    if (-4..6).contains(&ev) {
        let prec = (5 - ev) as usize;
        let s = format!("{x:.prec$}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        let mant = mant.trim_end_matches('0').trim_end_matches('.');
        let (sign, ea) = if ev < 0 { ('-', -ev) } else { ('+', ev) };
        format!("{mant}e{sign}{ea:02}")
    }
}

/// `iter:  N err: X` convergence line (Matsubara iterations).
pub fn iter_line(iter: usize, err: f64) -> String {
    format!("iter:  {iter} err: {}", fmt_g6(err))
}

/// `START: iter:  N err: X` convergence line (start-window iterations).
pub fn start_iter_line(iter: usize, err: f64) -> String {
    format!("START: iter:  {iter} err: {}", fmt_g6(err))
}

/// `tstp:  N` progress line (real-time steps).
pub fn tstp_line(tstp: usize) -> String {
    format!("tstp:  {tstp}")
}

/// `dens, pp_mu = D, M` line printed by `normalize_ppgf`.
pub fn dens_pp_mu_line(dens: f64, pp_mu: f64) -> String {
    format!("dens, pp_mu = {}, {}", fmt_g6(dens), fmt_g6(pp_mu))
}

/// `pp_dens, pp_mu = D, M` line printed by `solve_atomic`.
pub fn pp_dens_pp_mu_line(dens: f64, pp_mu: f64) -> String {
    format!("pp_dens, pp_mu = {}, {}", fmt_g6(dens), fmt_g6(pp_mu))
}

/// `obs: tstp t n_up n_dn docc` local-observable line.
pub fn obs_line(tstp: usize, t: f64, n_up: f64, n_dn: f64, docc: f64) -> String {
    format!(
        "obs: {tstp} {} {} {} {}",
        fmt_g6(t),
        fmt_g6(n_up),
        fmt_g6(n_dn),
        fmt_g6(docc)
    )
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- parser: values ------------------------------------------------

    #[test]
    fn values_scalars() {
        let doc = parse_toml(
            "# leading comment\n\
             i = 42\n\
             n = -7\n\
             p = +5\n\
             f = 5.0\n\
             e = 1e-8\n\
             E = 1.5E+2\n\
             s = -0.25\n\
             b1 = true\n\
             b2 = false\n\
             str = \"a string\"\n",
        )
        .expect("parse");
        let t = doc.root();
        assert_eq!(t.get_i64("i"), Ok(42));
        assert_eq!(t.get_i64("n"), Ok(-7));
        assert_eq!(t.get_i64("p"), Ok(5));
        assert_eq!(t.get_f64("f"), Ok(5.0));
        assert_eq!(t.get_f64("e"), Ok(1e-8));
        assert_eq!(t.get_f64("E"), Ok(150.0));
        assert_eq!(t.get_f64("s"), Ok(-0.25));
        assert_eq!(t.get_bool("b1"), Ok(true));
        assert_eq!(t.get_bool("b2"), Ok(false));
        assert_eq!(t.get_str("str"), Ok("a string"));
    }

    #[test]
    fn values_arrays_and_int_coercion() {
        let doc = parse_toml("arr = [1, -2.5, 3e-2]\nempty = []\nsingle = [7]\nint_float = 5\n")
            .expect("parse");
        let t = doc.root();
        assert_eq!(t.get_f64_vec("arr"), Ok(vec![1.0, -2.5, 0.03]));
        assert_eq!(t.get_f64_vec("empty"), Ok(vec![]));
        assert_eq!(t.get_f64_vec("single"), Ok(vec![7.0]));
        // integers coerce to floats
        assert_eq!(t.get_f64("int_float"), Ok(5.0));
        // scalars broadcast to one-element vectors
        assert_eq!(t.get_f64_vec("int_float"), Ok(vec![5.0]));
    }

    #[test]
    fn comments_and_strings_with_hash() {
        let doc = parse_toml(
            "# full-line comment\n\
             x = 1 # trailing comment\n\
             \n\
             s = \"keep # inside\"\n\
             esc = \"a\\\"b # c\"\n",
        )
        .expect("parse");
        let t = doc.root();
        assert_eq!(t.get_i64("x"), Ok(1));
        assert_eq!(t.get_str("s"), Ok("keep # inside"));
        assert_eq!(t.get_str("esc"), Ok("a\"b # c"));
    }

    #[test]
    fn sections() {
        let doc = parse_toml("k = 1\n[case]\nk = 2.5\nname = \"quench\"\n[other]\nflag = true\n")
            .expect("parse");
        assert_eq!(doc.root().get_i64("k"), Ok(1));
        let case = doc.section("case").expect("section");
        assert_eq!(case.get_f64("k"), Ok(2.5));
        assert_eq!(case.get_str("name"), Ok("quench"));
        assert_eq!(doc.section("other").unwrap().get_bool("flag"), Ok(true));
        assert_eq!(doc.section_names(), vec!["case", "other"]);
        assert!(doc.section("nope").is_err());
    }

    // -- parser: errors ------------------------------------------------

    #[test]
    fn parse_errors() {
        let cases: &[(&str, usize, &str)] = &[
            ("nt = 300\nnt = 5\n", 2, "duplicate key 'nt'"),
            ("beta = x\n", 1, "cannot parse number 'x'"),
            ("[a.b]\n", 1, "nested sections"),
            ("justkey\n", 1, "expected 'key = value'"),
            ("= 5\n", 1, "invalid key"),
            ("str = \"abc\n", 1, "unterminated string"),
            ("a = [1, 2\n", 1, "unterminated array"),
            ("a = [1, 2,]\n", 1, "empty array entry"),
            ("[s]\nk = 1\n[s]\n", 3, "duplicate section '[s]'"),
            ("x = 99999999999999999999\n", 1, "out of range"),
            ("s = \"a\\nb\"\n", 1, "unsupported escape"),
        ];
        for (src, line, msg) in cases {
            let e = parse_toml(src).expect_err(src);
            assert_eq!(e.line, *line, "line of {src:?}");
            assert!(e.msg.contains(msg), "msg of {src:?}: {}", e.msg);
        }
    }

    #[test]
    fn getter_type_errors() {
        let doc = parse_toml("a = 1\nb = 2.0\nc = true\nd = \"s\"\ne = [1.0]\n").expect("parse");
        let t = doc.root();
        assert!(t.get_f64("a").is_ok());
        assert!(t.get_f64("c").unwrap_err().contains("expected a float"));
        assert!(t.get_i64("b").unwrap_err().contains("expected an integer"));
        assert!(t.get_bool("d").unwrap_err().contains("expected a bool"));
        assert!(t.get_str("e").unwrap_err().contains("expected a string"));
        assert!(
            t.get_f64_vec("d")
                .unwrap_err()
                .contains("expected a float or a numeric array")
        );
        assert!(t.get_i64("missing").unwrap_err().contains("got nothing"));
        let doc2 = parse_toml("neg = -3\n").expect("parse");
        assert!(
            doc2.root()
                .get_usize("neg")
                .unwrap_err()
                .contains("non-negative")
        );
    }

    // -- case configuration ---------------------------------------------

    /// The P0-N benchmark case: single-band Hubbard, Bethe lattice,
    /// interaction quench U0 = 3.0 -> U = 3.5 (PLAN.md section 3.1),
    /// mirroring all fields of the PPSC probe `case.inp`.
    const BENCH_CASE: &str = "\
# nQIST benchmark case: single-band Hubbard on the Bethe lattice,
# interaction quench 3.0 -> 3.5
nt = 300
ntau = 250
beta = 5.0
h = 0.02
itermax = 300
errmax = 1e-8
iter_rtime = 4
kt = 5
order = 1
nomp = 1
store_pp = false
linear_mixing = 0.5
read_eq_sym = false
read_rt_sym = false
read_state_from_file = false
mu = 0.0
U0 = 3.0
U = 3.5
eps = 0.0
";

    #[test]
    fn benchmark_case_all_fields() {
        let cfg = CaseConfig::from_str(BENCH_CASE).expect("config");
        assert_eq!(cfg.nt, 300);
        assert_eq!(cfg.ntau, 250);
        assert_eq!(cfg.beta, 5.0);
        assert_eq!(cfg.h, 0.02);
        assert_eq!(cfg.itermax, 300);
        assert_eq!(cfg.errmax, 1e-8);
        assert_eq!(cfg.iter_rtime, 4);
        assert_eq!(cfg.kt, 5);
        assert_eq!(cfg.order, 1);
        assert_eq!(cfg.nomp, 1);
        assert!(!cfg.store_pp);
        assert_eq!(cfg.linear_mixing, 0.5);
        assert!(!cfg.read_eq_sym);
        assert!(!cfg.read_rt_sym);
        assert!(!cfg.read_state_from_file);
        assert_eq!(cfg.mu, 0.0);
        assert_eq!(cfg.u0, 3.0);
        assert_eq!(cfg.u, vec![3.5]);
        assert_eq!(cfg.eps, vec![0.0]);
    }

    #[test]
    fn case_config_series_and_defaults() {
        let cfg = CaseConfig::from_str(
            "nt = 10\nntau = 10\nbeta = 5\nh = 0.02\nitermax = 3\n\
             errmax = 1e-8\niter_rtime = 4\nkt = 5\norder = 2\n\
             linear_mixing = 0.5\nmu = 0\nU0 = 3.0\n\
             U = [3.0, 3.5]\neps = 0\n",
        )
        .expect("config");
        assert_eq!(cfg.u, vec![3.0, 3.5]);
        assert_eq!(cfg.eps, vec![0.0]);
        // defaults: nomp = 1, flags = false
        assert_eq!(cfg.nomp, 1);
        assert!(!cfg.store_pp);
        assert!(!cfg.read_state_from_file);
        // integer literals coerce into float fields
        assert_eq!(cfg.beta, 5.0);
        assert_eq!(cfg.mu, 0.0);
    }

    #[test]
    fn case_config_missing_field() {
        let e = CaseConfig::from_str("nt = 300\n").unwrap_err();
        assert!(e.contains("ntau"), "unexpected error: {e}");
        let e = CaseConfig::from_str(
            "nt = 300\nntau = 250\nbeta = 5.0\nh = 0.02\nitermax = 3\n\
             errmax = 1e-8\niter_rtime = 4\nkt = 5\norder = 1\n\
             linear_mixing = 0.5\nmu = 0.0\nU0 = 3.0\neps = 0.0\n",
        )
        .unwrap_err();
        assert!(e.contains("U"), "unexpected error: {e}");
    }

    // -- C++-aligned formatting ------------------------------------------

    #[test]
    fn fmt_g6_matches_cpp_cout() {
        // samples taken from the PPSC run.log references
        assert_eq!(fmt_g6(16.8995), "16.8995");
        assert_eq!(fmt_g6(10.0125), "10.0125");
        assert_eq!(fmt_g6(2.00111), "2.00111");
        assert_eq!(fmt_g6(-1.63874), "-1.63874");
        assert_eq!(fmt_g6(1e-8), "1e-08");
        assert_eq!(fmt_g6(1e-5), "1e-05");
        // branch boundaries and rounding
        assert_eq!(fmt_g6(0.0), "0");
        assert_eq!(fmt_g6(0.0001), "0.0001");
        assert_eq!(fmt_g6(0.02), "0.02");
        assert_eq!(fmt_g6(6.04), "6.04");
        assert_eq!(fmt_g6(123456.7), "123457");
        assert_eq!(fmt_g6(999999.6), "1e+06");
        assert_eq!(fmt_g6(1000000.0), "1e+06");
        assert_eq!(fmt_g6(0.9999996), "1");
        assert_eq!(fmt_g6(-1234567.0), "-1.23457e+06");
        assert_eq!(fmt_g6(f64::NAN), "nan");
        assert_eq!(fmt_g6(f64::INFINITY), "inf");
        assert_eq!(fmt_g6(f64::NEG_INFINITY), "-inf");
    }

    #[test]
    fn log_lines_match_cpp_source() {
        // spacing taken verbatim from ppsc.cpp / single_band_hubbard_bethe.cpp
        assert_eq!(iter_line(1, 16.8995), "iter:  1 err: 16.8995");
        assert_eq!(start_iter_line(3, 5.98087), "START: iter:  3 err: 5.98087");
        assert_eq!(tstp_line(6), "tstp:  6");
        assert_eq!(
            dens_pp_mu_line(2.00111, -1.63874),
            "dens, pp_mu = 2.00111, -1.63874"
        );
        assert_eq!(
            pp_dens_pp_mu_line(2.00111, -1.63874),
            "pp_dens, pp_mu = 2.00111, -1.63874"
        );
        assert_eq!(
            obs_line(0, 0.0, 0.5, 0.5, 0.123456789),
            "obs: 0 0 0.5 0.5 0.123457"
        );
        assert_eq!(
            obs_line(6, 0.12, 0.499999, 0.499999, 0.1),
            "obs: 6 0.12 0.499999 0.499999 0.1"
        );
    }
}
