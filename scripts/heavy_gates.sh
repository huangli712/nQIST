#!/usr/bin/env bash
# Heavy NCA/OCA gates: the OCA tier that is too slow for the default
# `cargo test` suite. Run it as a separate CI job, or detached so it
# does not block the session.
#
#   scripts/heavy_gates.sh                  # foreground, cargo test
#   scripts/heavy_gates.sh --frozen         # build once, run the test binaries
#   scripts/heavy_gates.sh --background     # detached --frozen run
#   scripts/heavy_gates.sh --status         # run state + last 20 progress lines
#   scripts/heavy_gates.sh --list           # gate list and ETAs
#   scripts/heavy_gates.sh <filter>         # only gates matching target::name
#
# Gates (one thread, reference machine):
#   - xca_twoband_dimer::kanamori_dimer_realtime_oca_vs_ed   (~603 s)
#   - threeband_dimer::threeband_kanamori_bz_oca_vs_ed       (~285 s)
#   - threeband_dimer::threeband_kanamori_realtime_oca_vs_ed (~80 min)
#   - ppsc_ref::oca_realtime_matches_reference               (~30-60 min)
#   - ppsc_ref::three_band_oca_matches_reference             (~2.2 h)
#   - ppsc_ref::prb96_photodoping_matches_reference          (~3 min)
#
# Progress: the script exports NQIST_PROGRESS=1 and passes --nocapture;
# the DMFT drivers print `[dmft eq|start|rt] i/N (pct%) ... elapsed/eta`
# and the solver prints per-solve timings `[solver] pp_step(...) done in`.
set -euo pipefail
self="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.."
export NQIST_PROGRESS=1

gates=(
    "xca_twoband_dimer|kanamori_dimer_realtime_oca_vs_ed|603 s"
    "threeband_dimer|threeband_kanamori_bz_oca_vs_ed|285 s"
    "threeband_dimer|threeband_kanamori_realtime_oca_vs_ed|80 min"
    "ppsc_ref|oca_realtime_matches_reference|30-60 min"
    "ppsc_ref|three_band_oca_matches_reference|2.2 h"
    "ppsc_ref|prb96_photodoping_matches_reference|3 min"
)

log_dir="target/heavy_gates"
latest_log="$log_dir/latest.log"
latest_pid="$log_dir/latest.pid"

if [[ "${1:-}" == "--list" ]]; then
    for gate in "${gates[@]}"; do
        IFS='|' read -r target test eta <<< "$gate"
        printf '%-22s %-46s ~%s\n' "$target" "$test" "$eta"
    done
    exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
    if [[ -f "$latest_pid" ]]; then
        pid=$(cat "$latest_pid")
        if kill -0 "$pid" 2>/dev/null; then
            echo "heavy tier: running (pid $pid)"
        else
            echo "heavy tier: not running (last pid $pid)"
        fi
    else
        echo "heavy tier: no detached run recorded"
    fi
    if [[ -f "$latest_log" ]]; then
        echo "--- $latest_log (last 20 lines) ---"
        tail -n 20 "$latest_log"
    fi
    exit 0
fi

if [[ "${1:-}" == "--background" ]]; then
    shift
    filter="${1:-}"
    mkdir -p "$log_dir"
    args=("$self" --frozen)
    if [[ -n "$filter" ]]; then
        args+=("$filter")
    fi
    nohup setsid "${args[@]}" > "$latest_log" 2>&1 < /dev/null &
    echo $! > "$latest_pid"
    echo "heavy tier: started detached (pid $!)"
    echo "log: $latest_log"
    echo "watch: scripts/heavy_gates.sh --status"
    exit 0
fi

mode="cargo"
if [[ "${1:-}" == "--frozen" ]]; then
    mode="frozen"
    shift
fi
filter="${1:-}"

if [[ "$mode" == "frozen" ]]; then
    echo "=== building test binaries (release) started $(date -Is) ==="
    cargo test --release --no-run --tests
    declare -A bins
    for gate in "${gates[@]}"; do
        IFS='|' read -r target _ _ <<< "$gate"
        if [[ -z "${bins[$target]:-}" ]]; then
            bins[$target]=$(ls -t target/release/deps/"$target"-* | grep -v '\.d$' | head -1)
        fi
    done
fi

overall=$(date +%s)
for gate in "${gates[@]}"; do
    IFS='|' read -r target test eta <<< "$gate"
    if [[ -n "$filter" && "$target::$test" != *"$filter"* ]]; then
        continue
    fi
    echo "=== heavy gate: $target::$test (expected ~$eta) started $(date -Is) ==="
    start=$(date +%s)
    if [[ "$mode" == "frozen" ]]; then
        "${bins[$target]}" --ignored --exact "$test" --nocapture
    else
        cargo test --release --test "$target" -- --ignored --exact "$test" --nocapture
    fi
    echo "=== $test passed in $(( $(date +%s) - start )) s ==="
done
echo "=== heavy tier finished in $(( $(date +%s) - overall )) s ==="
