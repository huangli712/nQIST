//! Opt-in progress reporting for long-running DMFT and solver runs.
//!
//! Nothing is printed unless `NQIST_PROGRESS` is set, so the library's
//! default output and the golden-file comparisons are unchanged. The heavy
//! gate script (`scripts/heavy_gates.sh`) exports the variable and passes
//! `--nocapture`, so the lines stream while a gate runs.

use std::time::{Duration, Instant};

/// True when progress reporting is enabled (`NQIST_PROGRESS` is set).
pub fn enabled() -> bool {
    std::env::var_os("NQIST_PROGRESS").is_some()
}

/// Print one progress line (no-op unless enabled).
pub fn line(scope: &str, msg: &str) {
    if enabled() {
        eprintln!("[{scope}] {msg}");
    }
}

fn fmt_dur(d: Duration) -> String {
    let s = d.as_secs_f64();
    if s < 60.0 {
        format!("{s:.1}s")
    } else if s < 3600.0 {
        format!("{}m{:04.1}s", (s / 60.0) as u64, s % 60.0)
    } else {
        format!("{}h{:04.1}m", (s / 3600.0) as u64, (s % 3600.0) / 60.0)
    }
}

/// Iteration/step reporter with elapsed time and a linear ETA.
pub struct Progress {
    scope: &'static str,
    total: usize,
    start: Instant,
}

impl Progress {
    /// `total` is the iteration/timestep budget of the loop.
    pub fn new(scope: &'static str, total: usize) -> Self {
        Self {
            scope,
            total,
            start: Instant::now(),
        }
    }

    /// Report step `done` (1-based) with an optional trailing detail
    /// (e.g. `err=1.2e-04`).
    pub fn step(&self, done: usize, detail: &str) {
        if !enabled() {
            return;
        }
        let elapsed = self.start.elapsed();
        let total = self.total.max(1);
        let pct = 100.0 * done as f64 / total as f64;
        let eta = if done == 0 {
            Duration::ZERO
        } else {
            elapsed.mul_f64((total.saturating_sub(done)) as f64 / done as f64)
        };
        let detail = if detail.is_empty() {
            String::new()
        } else {
            format!(" {detail}")
        };
        eprintln!(
            "[{}] {done}/{total} ({pct:.0}%){detail} elapsed {} eta {}",
            self.scope,
            fmt_dur(elapsed),
            fmt_dur(eta)
        );
    }
}

/// Scoped timing line, printed on drop when progress is enabled.
pub struct Timed {
    scope: &'static str,
    what: String,
    start: Instant,
    enabled: bool,
}

impl Timed {
    pub fn new(scope: &'static str, what: impl Into<String>) -> Self {
        Self {
            scope,
            what: what.into(),
            start: Instant::now(),
            enabled: enabled(),
        }
    }
}

impl Drop for Timed {
    fn drop(&mut self) {
        if self.enabled {
            eprintln!(
                "[{}] {} done in {}",
                self.scope,
                self.what,
                fmt_dur(self.start.elapsed())
            );
        }
    }
}
