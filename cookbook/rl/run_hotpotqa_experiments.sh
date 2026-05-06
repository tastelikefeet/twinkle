#!/usr/bin/env bash
# Sequentially run the tool-augmented HotpotQA GRPO script and then the
# full-context baseline.  Designed to be launched asynchronously (e.g.
# ``nohup bash run_hotpotqa_experiments.sh >runner.log 2>&1 &``) and read
# the next morning.
#
# Behaviour:
#   * Each experiment writes its own timestamped log under ``logs/``.
#   * The baseline ALWAYS runs, even if the tool-augmented run crashes,
#     so one failure doesn't waste the whole night.
#   * Exit status of the overall script = 0 iff both experiments returned 0;
#     otherwise the script exits 1 after both have completed.
#   * A short summary (start/end timestamps, exit codes) is appended to
#     ``logs/summary.log`` so you can grep that one file in the morning.

set -u  # catch unset vars; do NOT ``set -e`` -- we want second run even if first fails

# Resolve the script's directory WITHOUT relying on ``${BASH_SOURCE[0]}``.
# ``BASH_SOURCE`` is a bash-only array; Ubuntu's ``sh`` is ``dash`` and
# reports "Bad substitution" on that token.  ``$0`` works under both.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$LOG_DIR/summary.log"
TOOL_LOG="$LOG_DIR/with_tools_${STAMP}.log"
BASE_LOG="$LOG_DIR/baseline_${STAMP}.log"

PYTHON_BIN="${PYTHON_BIN:-python}"

log_summary() {
    # Tee to summary file and stdout so ``nohup`` tail stays informative.
    echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"
}

run_experiment() {
    local name="$1"
    local script="$2"
    local log_path="$3"

    log_summary "=== START $name -> $log_path ==="
    local t0; t0=$(date +%s)
    # ``-u`` on python keeps stdout unbuffered so the log grows in real time
    # while ``tail -f`` can be used during the night.
    "$PYTHON_BIN" -u "$script" >"$log_path" 2>&1
    local rc=$?
    local t1; t1=$(date +%s)
    local mins=$(( (t1 - t0) / 60 ))
    log_summary "=== END   $name exit=$rc duration=${mins}min ==="
    return $rc
}

log_summary "Runner launched (PID=$$). Log dir: $LOG_DIR"

run_experiment "with_tools"  "$SCRIPT_DIR/short_math_grpo_with_tools.py"         "$TOOL_LOG"
rc_tool=$?

run_experiment "baseline"    "$SCRIPT_DIR/short_math_grpo_hotpotqa_baseline.py"  "$BASE_LOG"
rc_base=$?

log_summary "Both experiments finished. with_tools=$rc_tool baseline=$rc_base"

if [[ $rc_tool -ne 0 || $rc_base -ne 0 ]]; then
    exit 1
fi
exit 0
